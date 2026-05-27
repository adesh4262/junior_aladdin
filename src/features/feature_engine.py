# FILE: src/features/feature_engine.py
"""
Junior Aladdin - Feature Engine Orchestrator (Institutional Hardened v2.0.0)
===========================================================================

RESPONSIBILITY:
Safely orchestrate computation of all feature modules in dependency order,
validate candle structure, distinguish failure vs unavailable, and expose
transparent feature availability status.

MANDATORY HARDENING (Audit Fixes I1-I8):
I1  Candle structure validation with explicit invalid status
I2  Thread-safe internal histories via self._state_lock (RLock)
I3  Refined degraded_mode based ONLY on critical stages
I4  _safe_compute returns sentinel on exception; stage_status uses failed vs unavailable
I5  Validate fundamental data on injection
I6  Startup dependency check for required callables
I7  Safer swing point extraction (no fragile comprehensions)
I8  Feature bundle version meta tag: feature_engine_version = "2.0.0"

PATCH (Replay/Staleness Audit):
- compute_all(...) supports keyword-only flag skip_freshness_check
- skip_freshness_check is forwarded into compute_momentum_features(...)
  so offline replay does not fail "freshness/stale" gating.

PATCH (Momentum Aggregation Audit):
- compute_momentum_features is called once with all timeframes, then its
    prefixed outputs (e.g. 1min_rsi) are redistributed back into each tf bucket.

PUBLIC API SIGNATURES:
- compute_all(...)
- set_fundamental_data(...)
- get_status()
- reset()

RETURN BUNDLE (backward compatible keys):
- per_tf, volume_profile, key_levels, options, smart_money_5m, smart_money_15m,
  microstructure, fundamental, mtf, stage_status, meta
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger

# Feature modules (dependency order)
from src.features.price_features import compute_price_features
from src.features.momentum_features import compute_momentum_features
from src.features.volatility_features import compute_volatility_features
from src.features.volume_profile import compute_volume_profile_features
from src.features.key_levels import compute_key_level_features
from src.features.options_features import compute_options_features
from src.features.smart_money import compute_smart_money_features, find_swing_points
from src.features.microstructure import compute_microstructure_features
from src.features.fundamental import compute_fundamental_features
from src.features.mtf_alignment import compute_mtf_alignment

_logger = setup_logger("feature_engine")

_FEATURE_ERROR_SENTINEL_KEY = "__feature_error__"


class FeatureEngine:
    """
    Institutional-grade master orchestrator for feature computation.
    """

    # I3: degraded only if these are failed/unavailable
    CRITICAL_FEATURES = {"1min_status", "mtf_status", "key_levels_status"}

    def __init__(self):
        self._logger = _logger

        # I2: thread-safe histories/state
        self._state_lock = threading.RLock()

        # Compute metrics (observability)
        self.total_computations: int = 0
        self.last_compute_time_ms: float = 0.0
        self.avg_compute_time_ms: float = 0.0
        self._compute_times: deque = deque(maxlen=100)

        self.errors: Dict[str, int] = {}

        # Fundamental caches (validated in setter)
        self._fii_data: Optional[Dict] = None
        self._global_data: Optional[Dict] = None

        # Stateful histories (thread-safe modifications required)
        self._prev_pcr_oi: Optional[float] = None
        self._iv_history: List[float] = []
        self._spread_history: List[float] = []
        self._prev_depth: Optional[Dict] = None

        # I6: startup dependency check
        self._verify_dependencies()

    # ------------------------------------------------------------------
    # I6: Startup dependency verification
    # ------------------------------------------------------------------
    def _verify_dependencies(self) -> None:
        required: List[Tuple[str, Any]] = [
            ("compute_price_features", compute_price_features),
            ("compute_momentum_features", compute_momentum_features),
            ("compute_volatility_features", compute_volatility_features),
            ("compute_volume_profile_features", compute_volume_profile_features),
            ("compute_key_level_features", compute_key_level_features),
            ("compute_options_features", compute_options_features),
            ("compute_smart_money_features", compute_smart_money_features),
            ("find_swing_points", find_swing_points),
            ("compute_microstructure_features", compute_microstructure_features),
            ("compute_fundamental_features", compute_fundamental_features),
            ("compute_mtf_alignment", compute_mtf_alignment),
        ]
        missing = [name for name, fn in required if not callable(fn)]
        if missing:
            raise RuntimeError(f"FeatureEngine dependency check failed; missing/not-callable: {missing}")

    # ------------------------------------------------------------------
    # I1: Candle validation
    # ------------------------------------------------------------------
    def _validate_candles(self, candles: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate candle structure to prevent silent corruption.

        Required keys: open, high, low, close, timestamp
        Basic type checks: prices must be float-coercible; timestamp non-null.
        """
        if not isinstance(candles, list):
            return False, "candles_not_list"
        if len(candles) == 0:
            return False, "empty"

        required = ("open", "high", "low", "close", "timestamp")

        # Sample first 50 to validate structure without O(N) overhead each call
        for i, c in enumerate(candles[:50]):
            if not isinstance(c, dict):
                return False, f"candle_not_dict_at_{i}"
            for k in required:
                if k not in c:
                    return False, f"missing_key_{k}_at_{i}"
                if c[k] is None:
                    return False, f"null_key_{k}_at_{i}"

            for pk in ("open", "high", "low", "close"):
                v = c.get(pk)
                try:
                    if isinstance(v, bool):
                        return False, f"bool_price_{pk}_at_{i}"
                    float(v)
                except Exception:
                    return False, f"non_numeric_{pk}_at_{i}"

            ts = c.get("timestamp")
            if isinstance(ts, str) and not ts.strip():
                return False, f"empty_timestamp_at_{i}"

        return True, ""

    def _candles_to_ohlcv_dict(self, candles: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Adapter: Convert list-of-candle-dicts into dict-of-columns.
        This matches feature modules in your environment that expect a dict input.
        Keys provided: timestamp, open, high, low, close, volume (volume may be None).
        """
        out: Dict[str, List[Any]] = {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        for c in candles:
            out["timestamp"].append(c.get("timestamp"))
            out["open"].append(c.get("open"))
            out["high"].append(c.get("high"))
            out["low"].append(c.get("low"))
            out["close"].append(c.get("close"))
            out["volume"].append(c.get("volume"))
        return out

    # ------------------------------------------------------------------
    # I4: Safe wrapper with explicit sentinel
    # ------------------------------------------------------------------
    def _safe_compute(self, module_name: str, tf: str, func: Callable, *args, **kwargs) -> Dict:
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                return result
            return {}
        except Exception as e:
            self._register_error(f"{module_name}_{tf}")
            self._logger.error(
                f"Feature computation failed: {module_name} ({tf})",
                extra={"error": str(e)[:500], "count": self.errors.get(f"{module_name}_{tf}", 0)},
            )
            return {_FEATURE_ERROR_SENTINEL_KEY: True, "error": str(e)[:1000]}

    def _is_failed(self, d: Dict[str, Any]) -> bool:
        return isinstance(d, dict) and bool(d.get(_FEATURE_ERROR_SENTINEL_KEY))

    def _strip_failure_sentinel(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(d, dict):
            return {}
        if self._is_failed(d):
            return {}
        return d

    def _register_error(self, key: str):
        self.errors[key] = self.errors.get(key, 0) + 1

    # ------------------------------------------------------------------
    # I5: Fundamental injection validation
    # ------------------------------------------------------------------
    def set_fundamental_data(self, fii_data: Optional[Dict] = None, global_data: Optional[Dict] = None):
        """
        Cache fundamental session data (validated).
        fii_data must contain numeric (int/float) fii_net and dii_net.
        """
        validated_fii: Optional[Dict] = None
        if fii_data is not None:
            if not isinstance(fii_data, dict):
                self._logger.warning("Invalid fii_data type; ignoring", extra={"type": str(type(fii_data))})
            else:
                fii_net = fii_data.get("fii_net")
                dii_net = fii_data.get("dii_net")
                ok = True
                if fii_net is None or dii_net is None:
                    ok = False
                if isinstance(fii_net, bool) or isinstance(dii_net, bool):
                    ok = False
                # strict numeric types only
                if not isinstance(fii_net, (int, float)) or not isinstance(dii_net, (int, float)):
                    ok = False

                if ok:
                    validated_fii = dict(fii_data)
                else:
                    self._logger.warning(
                        "Invalid fii_data payload; ignoring",
                        extra={
                            "keys": list(fii_data.keys())[:20],
                            "fii_net_type": str(type(fii_net)),
                            "dii_net_type": str(type(dii_net)),
                        },
                    )

        validated_global: Optional[Dict] = None
        if global_data is not None:
            if isinstance(global_data, dict):
                validated_global = dict(global_data)
            else:
                self._logger.warning("Invalid global_data type; ignoring", extra={"type": str(type(global_data))})

        with self._state_lock:
            self._fii_data = validated_fii
            self._global_data = validated_global

        self._logger.info(
            "Fundamental data set for session",
            extra={"has_fii": validated_fii is not None, "has_global": validated_global is not None},
        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def compute_all(
        self,
        candles_by_tf: Dict[str, Deque],
        option_chain: Optional[Dict] = None,
        market_depth: Optional[Dict] = None,
        spot_price: float = 0.0,
        previous_day_candles: Optional[List[Dict]] = None,
        context_meta: Optional[Dict[str, Any]] = None,
        *,
        skip_freshness_check: bool = False,
    ) -> Dict[str, Dict]:
        start_time = time.time()
        context_meta = context_meta or {}

        features_by_tf: Dict[str, Dict] = {}
        stage_status: Dict[str, Dict[str, Any]] = {}

        # --------------------------------------------------------------
        # Momentum: called ONCE with all timeframes
        # --------------------------------------------------------------
        mf_global_raw = self._safe_compute(
            "momentum_features",
            "all_tf",
            compute_momentum_features,
            candles_by_tf,
            skip_freshness_check=skip_freshness_check,
        )
        mf_global = self._strip_failure_sentinel(mf_global_raw)
        if self._is_failed(mf_global_raw):
            mf_global = {}

        # --------------------------------------------------------------
        # Per-timeframe: Price/Momentum/Volatility
        # --------------------------------------------------------------
        for tf in ["1min", "3min", "5min", "15min"]:
            candles = list(candles_by_tf.get(tf, deque()))
            tf_features: Dict[str, Any] = {}
            tf_prefix = f"{tf}_"

            ok_struct, reason = (self._validate_candles(candles) if candles else (False, "empty"))
            if not ok_struct:
                features_by_tf[tf] = {}
                stage_status[f"{tf}_status"] = {
                    "available": False,
                    "status": "invalid_candle_structure",
                    "reason": reason,
                    "count": len(candles),
                }
                continue

            if len(candles) < 2:
                features_by_tf[tf] = {}
                stage_status[f"{tf}_status"] = {
                    "available": False,
                    "status": "unavailable",
                    "reason": "insufficient_candles",
                    "count": len(candles),
                }
                continue

            pf_raw = self._safe_compute("price_features", tf, compute_price_features, candles_by_tf)

            # Price features are emitted with timeframe prefixes; strip them for downstream consumers.
            pf = self._strip_failure_sentinel(pf_raw)
            for k, v in pf.items():
                if k.startswith(tf_prefix):
                    tf_features[k[len(tf_prefix):]] = v
                else:
                    tf_features[k] = v

            for k, v in mf_global.items():
                if k.startswith(tf_prefix):
                    normalized_key = k[len(tf_prefix):]
                    tf_features[normalized_key] = v

            accel_norm = tf_features.get("price_acceleration_norm")
            if accel_norm is not None and tf_features.get("price_acceleration") is None:
                tf_features["price_acceleration"] = accel_norm
            elif tf_features.get("price_acceleration_raw") is not None and tf_features.get("price_acceleration") is None:
                tf_features["price_acceleration"] = tf_features["price_acceleration_raw"]

            for roc_alias_key in ("roc_5_smooth", "roc_5_raw"):
                roc_alias_val = tf_features.get(roc_alias_key)
                if roc_alias_val is not None:
                    tf_features.setdefault("roc_5", roc_alias_val)
                    break

            # Volatility features consume raw candle lists (ATR alignment depends on candle rows).
            vf_raw = self._safe_compute("volatility_features", tf, compute_volatility_features, candles)

            failed_modules = []
            for name, raw in (("price_features", pf_raw), ("momentum_features", mf_global_raw), ("volatility_features", vf_raw)):
                if self._is_failed(raw):
                    failed_modules.append(name)

            vf = self._strip_failure_sentinel(vf_raw)

            tf_features.update(vf)

            if failed_modules:
                stage_status[f"{tf}_status"] = {
                    "available": False,
                    "status": "failed",
                    "reason": "module_exception",
                    "failed_modules": failed_modules,
                    "count": len(candles),
                    "feature_count": len(tf_features),
                }
            elif not tf_features:
                stage_status[f"{tf}_status"] = {
                    "available": False,
                    "status": "unavailable",
                    "reason": "empty_result",
                    "count": len(candles),
                    "feature_count": 0,
                }
            else:
                stage_status[f"{tf}_status"] = {
                    "available": True,
                    "status": "ok",
                    "reason": "",
                    "count": len(candles),
                    "feature_count": len(tf_features),
                }

            features_by_tf[tf] = tf_features

        # Session-level candles (list-of-dicts used by other modules)
        candles_1m = list(candles_by_tf.get("1min", deque()))
        candles_5m = list(candles_by_tf.get("5min", deque()))
        candles_15m = list(candles_by_tf.get("15min", deque()))

        ok_1m_struct, _ = (self._validate_candles(candles_1m) if candles_1m else (False, "empty"))

        # --------------------------------------------------------------
        # Volume profile
        # --------------------------------------------------------------
        volume_profile: Dict[str, Any] = {}
        if not ok_1m_struct or len(candles_1m) < 2:
            stage_status["volume_profile_status"] = {
                "available": False,
                "status": "unavailable",
                "reason": "invalid_or_insufficient_1m_candles",
            }
        else:
            vp_raw = self._safe_compute("volume_profile", "session", compute_volume_profile_features, candles_1m)
            if self._is_failed(vp_raw):
                stage_status["volume_profile_status"] = {
                    "available": False,
                    "status": "failed",
                    "reason": "module_exception",
                    "error": vp_raw.get("error", ""),
                }
            else:
                volume_profile = vp_raw
                stage_status["volume_profile_status"] = {
                    "available": bool(volume_profile),
                    "status": "ok" if volume_profile else "unavailable",
                    "reason": "" if volume_profile else "empty_result",
                }

                if isinstance(features_by_tf.get("1min"), dict):
                    features_by_tf["1min"].update(volume_profile)
                    if features_by_tf["1min"].get("volume_ratio") is None:
                        recent_vols: List[float] = []
                        for candle in candles_1m[-21:]:
                            if not isinstance(candle, dict):
                                continue
                            volume = candle.get("volume")
                            if isinstance(volume, (int, float)) and volume >= 0:
                                recent_vols.append(float(volume))
                        if len(recent_vols) >= 2:
                            prior_vols = recent_vols[:-1]
                            current_vol = recent_vols[-1]
                            avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 0.0
                            if avg_vol > 0 and current_vol > 0:
                                features_by_tf["1min"]["volume_ratio"] = round(float(current_vol / avg_vol), 4)
                            else:
                                body_ratio = features_by_tf["1min"].get("candle_body_ratio")
                                price_accel = features_by_tf["1min"].get("price_acceleration")
                                if isinstance(body_ratio, (int, float)):
                                    synthetic_volume_ratio = 1.0 + abs(float(body_ratio)) * 3.0
                                    if isinstance(price_accel, (int, float)):
                                        synthetic_volume_ratio += min(abs(float(price_accel)) * 0.1, 0.5)
                                    features_by_tf["1min"]["volume_ratio"] = round(min(max(synthetic_volume_ratio, 1.0), 4.0), 4)

        # --------------------------------------------------------------
        # Swing points (I7)
        # --------------------------------------------------------------
        swing_highs_list: List[float] = []
        swing_lows_list: List[float] = []
        swing_status: Dict[str, Any] = {"available": False, "status": "unavailable", "reason": "insufficient_1m_candles"}

        if ok_1m_struct and len(candles_1m) >= 11:
            try:
                sh, sl = find_swing_points(candles_1m, lookback=5)
                try:
                    if isinstance(sh, list):
                        swing_highs_list = [float(s.get("price")) for s in sh if isinstance(s, dict) and s.get("price") is not None]
                    if isinstance(sl, list):
                        swing_lows_list = [float(s.get("price")) for s in sl if isinstance(s, dict) and s.get("price") is not None]
                    swing_status = {"available": True, "status": "ok", "reason": "", "highs": len(swing_highs_list), "lows": len(swing_lows_list)}
                except Exception as e2:
                    swing_highs_list, swing_lows_list = [], []
                    swing_status = {"available": False, "status": "failed", "reason": "extraction_error", "error": str(e2)[:200]}
                    self._logger.warning("Swing point extraction list conversion failed", extra={"error": str(e2)[:200]})
            except Exception as e:
                self._register_error("swing_points_session")
                swing_highs_list, swing_lows_list = [], []
                swing_status = {"available": False, "status": "failed", "reason": "find_swing_points_exception", "error": str(e)[:200]}
                self._logger.warning("Swing point detection failed", extra={"error": str(e)[:200]})

        stage_status["swing_points_status"] = swing_status

        # --------------------------------------------------------------
        # Key levels (critical)
        # --------------------------------------------------------------
        key_levels: Dict[str, Any] = {}
        if not ok_1m_struct or len(candles_1m) < 2:
            stage_status["key_levels_status"] = {"available": False, "status": "unavailable", "reason": "invalid_or_insufficient_1m_candles"}
        else:
            kl_raw = self._safe_compute(
                "key_levels",
                "session",
                compute_key_level_features,
                candles_1m,
                previous_day_candles or [],
                swing_highs_list,
                swing_lows_list,
            )
            if self._is_failed(kl_raw):
                stage_status["key_levels_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": kl_raw.get("error", "")}
            else:
                key_levels = kl_raw
                stage_status["key_levels_status"] = {
                    "available": bool(key_levels),
                    "status": "ok" if key_levels else "unavailable",
                    "reason": "" if key_levels else "empty_result",
                }

        # --------------------------------------------------------------
        # Options (thread-safe histories)
        # --------------------------------------------------------------
        options_feat: Dict[str, Any] = {}
        if not option_chain or spot_price <= 0:
            stage_status["options_status"] = {"available": False, "status": "unavailable", "reason": "missing_chain_or_spot"}
        else:
            with self._state_lock:
                prev_pcr_oi = self._prev_pcr_oi
                iv_history_snapshot = list(self._iv_history)

            opt_raw = self._safe_compute("options_features", "session", compute_options_features, option_chain, spot_price, prev_pcr_oi, iv_history_snapshot)
            if self._is_failed(opt_raw):
                stage_status["options_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": opt_raw.get("error", "")}
            else:
                options_feat = opt_raw
                stage_status["options_status"] = {"available": bool(options_feat), "status": "ok" if options_feat else "unavailable", "reason": "" if options_feat else "empty_result"}

                with self._state_lock:
                    pcr_oi = options_feat.get("pcr_oi", 0)
                    if isinstance(pcr_oi, (int, float)) and pcr_oi > 0:
                        self._prev_pcr_oi = float(pcr_oi)

                    atm_iv = options_feat.get("atm_iv", 0)
                    if isinstance(atm_iv, (int, float)) and atm_iv > 0:
                        self._iv_history.append(float(atm_iv))
                        if len(self._iv_history) > 100:
                            self._iv_history = self._iv_history[-100:]

        # --------------------------------------------------------------
        # Smart money (5m/15m)
        # --------------------------------------------------------------
        smart_money_5m: Dict[str, Any] = {}
        smart_money_15m: Dict[str, Any] = {}

        ok_5m_struct, _ = (self._validate_candles(candles_5m) if candles_5m else (False, "empty"))
        ok_15m_struct, _ = (self._validate_candles(candles_15m) if candles_15m else (False, "empty"))

        if ok_5m_struct and len(candles_5m) >= 13:
            sm5_raw = self._safe_compute("smart_money", "5min", compute_smart_money_features, candles_5m)
            if self._is_failed(sm5_raw):
                stage_status["smart_money_5m_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": sm5_raw.get("error", "")}
            else:
                smart_money_5m = sm5_raw
                stage_status["smart_money_5m_status"] = {"available": bool(smart_money_5m), "status": "ok" if smart_money_5m else "unavailable", "reason": "" if smart_money_5m else "empty_result"}
        else:
            stage_status["smart_money_5m_status"] = {"available": False, "status": "unavailable", "reason": "insufficient_or_invalid_candles"}

        if ok_15m_struct and len(candles_15m) >= 13:
            sm15_raw = self._safe_compute("smart_money", "15min", compute_smart_money_features, candles_15m)
            if self._is_failed(sm15_raw):
                stage_status["smart_money_15m_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": sm15_raw.get("error", "")}
            else:
                smart_money_15m = sm15_raw
                stage_status["smart_money_15m_status"] = {"available": bool(smart_money_15m), "status": "ok" if smart_money_15m else "unavailable", "reason": "" if smart_money_15m else "empty_result"}
        else:
            stage_status["smart_money_15m_status"] = {"available": False, "status": "unavailable", "reason": "insufficient_or_invalid_candles"}

        # --------------------------------------------------------------
        # Microstructure (thread-safe histories)
        # --------------------------------------------------------------
        microstructure: Dict[str, Any] = {}
        if not ok_1m_struct or len(candles_1m) < 2:
            stage_status["microstructure_status"] = {"available": False, "status": "unavailable", "reason": "invalid_or_insufficient_1m_candles"}
        else:
            rsi_value = features_by_tf.get("1min", {}).get("rsi")
            with self._state_lock:
                spread_hist_snapshot = list(self._spread_history)
                prev_depth_snapshot = self._prev_depth

            micro_raw = self._safe_compute(
                "microstructure",
                "session",
                compute_microstructure_features,
                candles_1m,
                market_depth,
                spread_hist_snapshot,
                prev_depth_snapshot,
                rsi_value,
                spot_price,
            )

            if self._is_failed(micro_raw):
                stage_status["microstructure_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": micro_raw.get("error", "")}
            else:
                microstructure = micro_raw
                stage_status["microstructure_status"] = {"available": bool(microstructure), "status": "ok" if microstructure else "unavailable", "reason": "" if microstructure else "empty_result"}

                with self._state_lock:
                    spread = microstructure.get("spread", 0)
                    if isinstance(spread, (int, float)) and spread > 0:
                        self._spread_history.append(float(spread))
                        if len(self._spread_history) > 300:
                            self._spread_history = self._spread_history[-300:]
                    self._prev_depth = market_depth

        # --------------------------------------------------------------
        # Fundamental
        # --------------------------------------------------------------
        with self._state_lock:
            fii_snapshot = self._fii_data
            global_snapshot = self._global_data

        fundamental_raw = self._safe_compute("fundamental", "session", compute_fundamental_features, fii_snapshot, global_snapshot)
        fundamental: Dict[str, Any] = {}
        if self._is_failed(fundamental_raw):
            stage_status["fundamental_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": fundamental_raw.get("error", "")}
        else:
            fundamental = fundamental_raw
            stage_status["fundamental_status"] = {"available": bool(fundamental), "status": "ok" if fundamental else "unavailable", "reason": "" if fundamental else "missing_cached_data_or_empty_result"}

        # --------------------------------------------------------------
        # MTF alignment (critical)
        # --------------------------------------------------------------
        mtf_input: Dict[str, Dict[str, Any]] = {}
        for tf in ("1min", "3min", "5min", "15min"):
            if features_by_tf.get(tf):
                mtf_input[tf] = features_by_tf[tf]

        mtf: Dict[str, Any] = {}
        if not mtf_input:
            stage_status["mtf_status"] = {"available": False, "status": "unavailable", "reason": "no_tf_inputs", "tf_count_used": 0}
        else:
            mtf_raw = self._safe_compute("mtf_alignment", "cross_tf", compute_mtf_alignment, mtf_input)
            if self._is_failed(mtf_raw):
                stage_status["mtf_status"] = {"available": False, "status": "failed", "reason": "module_exception", "error": mtf_raw.get("error", ""), "tf_count_used": len(mtf_input)}
            else:
                mtf = mtf_raw
                stage_status["mtf_status"] = {"available": bool(mtf), "status": "ok" if mtf else "unavailable", "reason": "" if mtf else "empty_result", "tf_count_used": len(mtf_input)}

        # --------------------------------------------------------------
        # Timing / meta
        # --------------------------------------------------------------
        elapsed_ms = (time.time() - start_time) * 1000.0
        self.last_compute_time_ms = round(elapsed_ms, 2)

        with self._state_lock:
            self._compute_times.append(elapsed_ms)
            self.avg_compute_time_ms = round(sum(self._compute_times) / len(self._compute_times), 2)
            self.total_computations += 1

        total_errors = sum(self.errors.values())

        # I3: degraded_mode based ONLY on critical stages failed/unavailable
        degraded = False
        for k in self.CRITICAL_FEATURES:
            st = stage_status.get(k, {})
            if st.get("status") in ("failed", "unavailable"):
                degraded = True
                break

        result = {
            "per_tf": features_by_tf,
            "volume_profile": volume_profile,
            "key_levels": key_levels,
            "options": options_feat,
            "smart_money_5m": smart_money_5m,
            "smart_money_15m": smart_money_15m,
            "microstructure": microstructure,
            "fundamental": fundamental,
            "mtf": mtf,
            "stage_status": stage_status,
            "meta": {
                "feature_engine_version": "2.0.0",  # I8
                "compute_time_ms": self.last_compute_time_ms,
                "avg_compute_time_ms": self.avg_compute_time_ms,
                "total_computations": self.total_computations,
                "errors": dict(self.errors),
                "total_errors": total_errors,
                "degraded_mode": degraded,
            },
        }

        if self.total_computations % 5 == 0:
            self._logger.info(
                "Features computed",
                extra={
                    "computation": self.total_computations,
                    "time_ms": self.last_compute_time_ms,
                    "avg_ms": self.avg_compute_time_ms,
                    "candles_1m": len(candles_1m),
                    "total_errors": total_errors,
                    "degraded_mode": degraded,
                    "critical_status": {k: stage_status.get(k, {}).get("status") for k in self.CRITICAL_FEATURES},
                },
            )

        return result

    # ------------------------------------------------------------------
    # Status / reset
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        with self._state_lock:
            has_fundamental = self._fii_data is not None or self._global_data is not None
            iv_len = len(self._iv_history)
            spread_len = len(self._spread_history)

        return {
            "feature_engine_version": "2.0.0",
            "total_computations": self.total_computations,
            "last_compute_time_ms": self.last_compute_time_ms,
            "avg_compute_time_ms": self.avg_compute_time_ms,
            "errors": dict(self.errors),
            "total_errors": sum(self.errors.values()),
            "has_fundamental": has_fundamental,
            "iv_history_length": iv_len,
            "spread_history_length": spread_len,
        }

    def reset(self):
        with self._state_lock:
            self.total_computations = 0
            self.last_compute_time_ms = 0.0
            self.avg_compute_time_ms = 0.0
            self._compute_times.clear()
            self.errors.clear()
            self._prev_pcr_oi = None
            self._iv_history.clear()
            self._spread_history.clear()
            self._prev_depth = None
            self._fii_data = None
            self._global_data = None


def _run_tests():
    from collections import deque as _dq

    print("=" * 66)
    print(" JUNIOR ALADDIN — Feature Engine Test (Institutional v2.0.0)")
    print("=" * 66)
    print()

    passed = 0
    failed = 0

    print(" [Test 1] Create engine + dependency check...")
    try:
        fe = FeatureEngine()
        print(" ✅ Feature Engine created")
        passed += 1
    except Exception as e:
        print(f" ❌ Failed: {e}")
        failed += 1
        print("\n" + "=" * 66)
        print(f" Results: {passed} passed, {failed} failed")
        return

    print("\n [Test 2] Invalid candle structure validation...")
    r2 = fe.compute_all(
        candles_by_tf={
            "1min": _dq([{"open": 1, "high": 2}]),  # malformed
            "3min": _dq(),
            "5min": _dq(),
            "15min": _dq(),
        },
        spot_price=0,
    )
    st_1m = r2.get("stage_status", {}).get("1min_status", {})
    if st_1m.get("status") == "invalid_candle_structure":
        print(" ✅ Invalid candle structure detected")
        passed += 1
    else:
        print(f" ❌ Expected invalid_candle_structure, got: {st_1m}")
        failed += 1

    print("\n [Test 3] Empty candles handled (unavailable/invalid)...")
    r3 = fe.compute_all(
        candles_by_tf={"1min": _dq(), "3min": _dq(), "5min": _dq(), "15min": _dq()},
        spot_price=0,
    )
    st3 = r3.get("stage_status", {}).get("1min_status", {})
    if st3.get("status") in ("invalid_candle_structure", "unavailable"):
        print(" ✅ Empty handled safely")
        passed += 1
    else:
        print(f" ❌ Unexpected status: {st3}")
        failed += 1

    print("\n [Test 4] Fundamental cache setter validation...")
    fe.set_fundamental_data(
        fii_data={"fii_net": -1000.0, "dii_net": 500.0},
        global_data={"sp500": {"change_pct": 0.01}},
    )
    st4 = fe.get_status()
    if st4["has_fundamental"]:
        print(" ✅ Fundamental cache validated and set")
        passed += 1
    else:
        print(" ❌ Fundamental cache not set")
        failed += 1

    print("\n [Test 5] Thread-safety smoke test (concurrent compute_all)...")
    # IMPORTANT: When running via `python -m`, this file is executed as __main__.
    # Patch the CURRENT running module instance, not a second import.
    mod = sys.modules[__name__]

    def _dummy_options_features(option_chain, spot, prev_pcr, iv_hist):
        return {"pcr_oi": 1.2, "atm_iv": 0.15}

    def _dummy_microstructure_features(candles_1m, depth, spread_hist, prev_depth, rsi, spot):
        return {"spread": 0.05}

    original_opt = getattr(mod, "compute_options_features")
    original_micro = getattr(mod, "compute_microstructure_features")
    setattr(mod, "compute_options_features", _dummy_options_features)
    setattr(mod, "compute_microstructure_features", _dummy_microstructure_features)

    try:
        now = datetime.utcnow()
        candles = [
            {"timestamp": now, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 10},
            {"timestamp": now, "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 12},
        ]
        candles_by_tf = {
            "1min": _dq(candles),
            "3min": _dq(candles),
            "5min": _dq(candles * 7),   # 14 candles
            "15min": _dq(candles * 7),  # 14 candles
        }

        exceptions: List[str] = []

        def worker():
            try:
                fe.compute_all(
                    candles_by_tf=candles_by_tf,
                    option_chain={"dummy": 1},
                    market_depth={"dummy": 2},
                    spot_price=20000.0,
                )
            except Exception as e:
                exceptions.append(str(e))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if exceptions:
            print(f" ❌ Exceptions in concurrent compute_all: {exceptions}")
            failed += 1
        else:
            st = fe.get_status()
            if st["iv_history_length"] >= 1 and st["spread_history_length"] >= 1:
                print(" ✅ Thread-safe histories updated under concurrency")
                passed += 1
            else:
                print(f" ❌ Expected history updates, got: {st}")
                failed += 1
    finally:
        setattr(mod, "compute_options_features", original_opt)
        setattr(mod, "compute_microstructure_features", original_micro)

    print("\n [Test 6] Reset...")
    fe.reset()
    st6 = fe.get_status()
    if st6["total_computations"] == 0 and st6["total_errors"] == 0 and st6["iv_history_length"] == 0:
        print(" ✅ Reset works")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st6}")
        failed += 1

    print("\n" + "=" * 66)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print(" ✅ Feature Engine (Institutional v2.0.0) ready")
    else:
        print(" ⚠️ Fix failing tests before proceeding")


if __name__ == "__main__":
    _run_tests()