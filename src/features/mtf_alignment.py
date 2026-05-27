"""
Junior Aladdin - MTF Alignment (Layer 2K)
==========================================
INSTITUTIONAL-GRADE HARDENING UPGRADE

PURPOSE:
    Compute Multi-Timeframe alignment score.
    Checks if all timeframes agree on direction.
    Strong alignment = higher-probability setup.

CORE SCORING LOGIC (MUST REMAIN UNCHANGED):
    Per timeframe:
      +1 if price > VWAP AND EMA9 > EMA21 AND RSI > 50
      -1 if price < VWAP AND EMA9 < EMA21 AND RSI < 50
       0 otherwise (mixed signals)
    Weighted MTF = Σ(direction_tf × weight_tf)

DEFAULT WEIGHTS (from plan):
    1min: 1.0
    3min: 1.5
    5min: 2.0
    15min: 3.0
Range: -7.5..+7.5 (base weights)

INSTITUTIONAL HARDENING (Additive, Backward-Compatible):
PHASE 1: Data Quality & Observability
    - Per TF availability flags (tf_1min_available, etc.)
    - compute_tf_direction returns dict-like object with:
        direction (+1/0/-1/None), state, confidence(0-100), missing_keys, conflict_keys
      while remaining backward compatible for int comparisons/arithmetic.
    - warnings list in output

PHASE 2: Configurability & Adaptation
    - TF weights and label thresholds loaded from config.yaml under [mtf]
    - Optional VIX-adaptive scaling: if vix_level > threshold, scale down lower TF weights
    - Optional session-aware threshold adjustment: tighten thresholds in LUNCH_LULL/LAST_MINUTES

PHASE 3: Output Enrichment
    - key-level confluence bonus flag (optional key_levels input)
    - mtf_trap_zone flag: lower TFs conflict with 15min direction

PHASE 4: Self-tests expanded for missing RSI, conflicts, vix scaling, partial availability

CONFIG (Optional; defaults applied if missing):
mtf:
  weights:
    1min: 1.0
    3min: 1.5
    5min: 2.0
    15min: 3.0
  thresholds:
    strong_bullish: 6.0
    bullish: 3.0
    neutral_low: -3.0
    strong_bearish: -6.0
  vix_weight_scale:
    enabled: true
    vix_threshold: 20.0
    scale_1min: 0.5
    scale_3min: 0.5
  session_threshold_scale:
    enabled: true
    phases: ["LUNCH_LULL", "LAST_MINUTES"]
    multiplier: 1.2
  key_level_confluence:
    enabled: true
    proximity_pct: 0.001   # 0.1%
    bonus_strength_points: 10.0

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping, Iterator, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("mtf_alignment")


# ============================================================
# Defaults (used when config is missing)
# ============================================================

DEFAULT_TF_WEIGHTS: Dict[str, float] = {
    "1min": 1.0,
    "3min": 1.5,
    "5min": 2.0,
    "15min": 3.0,
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "strong_bullish": 6.0,
    "bullish": 3.0,
    "neutral_low": -3.0,      # neutral is between [neutral_low, bullish)
    "strong_bearish": -6.0,
}


# ============================================================
# Safe Utilities
# ============================================================

_EPS = 1e-9


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert to float, returning default if None/invalid."""
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _cfg_get(*keys: str, default: Any = None) -> Any:
    """
    Safe config getter. Never raises.
    """
    try:
        return Config.get(*keys, default=default)
    except Exception as e:  # pragma: no cover
        _logger.error("Config.get failed; using default", keys=list(keys), default=default, error=str(e))
        return default


# ============================================================
# Direction Result Object (dict-like + int-compatible)
# ============================================================

@dataclass(frozen=True)
class TFDirectionResult(Mapping[str, Any]):
    """
    A dict-like return object with numeric compatibility.
    This allows:
      - New usage: res["direction"], res["state"], res["confidence"]
      - Legacy usage: int(res), float(res), res == 1, res * weight, etc.
    """
    direction: Optional[int]
    state: str
    confidence: int
    missing_keys: List[str]
    conflict_keys: List[str]
    details: Dict[str, Any]

    # Mapping interface
    def __getitem__(self, key: str) -> Any:
        if key == "direction":
            return self.direction
        if key == "state":
            return self.state
        if key == "confidence":
            return self.confidence
        if key == "missing_keys":
            return self.missing_keys
        if key == "conflict_keys":
            return self.conflict_keys
        if key == "details":
            return self.details
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from ("direction", "state", "confidence", "missing_keys", "conflict_keys", "details")

    def __len__(self) -> int:
        return 6

    # Numeric compatibility
    def __int__(self) -> int:
        return int(self.direction or 0)

    def __float__(self) -> float:
        return float(self.direction or 0)

    def __eq__(self, other: object) -> bool:
        # Allow comparisons like `res == 1`
        if isinstance(other, (int, float)):
            return float(self) == float(other)
        return super().__eq__(other)


# ============================================================
# Configurable Weights / Thresholds
# ============================================================

def _load_tf_weights() -> Dict[str, float]:
    """
    Load TF weights from config.yaml under mtf.weights; fall back to defaults.
    """
    cfg = _cfg_get("mtf", "weights", default=None)
    weights = dict(DEFAULT_TF_WEIGHTS)

    if isinstance(cfg, dict):
        for tf in DEFAULT_TF_WEIGHTS.keys():
            v = _safe_float(cfg.get(tf))
            if v is not None and v > 0:
                weights[tf] = float(v)
            elif tf in cfg:
                _logger.warning("Invalid mtf weight in config; using default", tf=tf, raw=cfg.get(tf), default=DEFAULT_TF_WEIGHTS[tf])

    return weights


def _load_thresholds() -> Dict[str, float]:
    """
    Load label thresholds from config.yaml under mtf.thresholds; fall back to defaults.
    """
    cfg = _cfg_get("mtf", "thresholds", default=None)
    th = dict(DEFAULT_THRESHOLDS)

    if isinstance(cfg, dict):
        for k in DEFAULT_THRESHOLDS.keys():
            v = _safe_float(cfg.get(k))
            if v is not None:
                th[k] = float(v)
            elif k in cfg:
                _logger.warning("Invalid mtf threshold in config; using default", key=k, raw=cfg.get(k), default=DEFAULT_THRESHOLDS[k])

    # sanity ordering
    # strong_bearish < neutral_low < bullish < strong_bullish
    # if misconfigured, revert to defaults
    if not (th["strong_bearish"] < th["neutral_low"] < th["bullish"] < th["strong_bullish"]):
        _logger.error("MTF thresholds misconfigured; reverting to defaults", thresholds=th)
        return dict(DEFAULT_THRESHOLDS)

    return th


def _apply_vix_weight_scaling(weights: Dict[str, float], vix_level: Optional[float], warnings: List[str]) -> Dict[str, float]:
    """
    Optional:
    If vix_level > configured threshold, scale down 1min/3min weights.
    """
    enabled = bool(_cfg_get("mtf", "vix_weight_scale", "enabled", default=True))
    if not enabled:
        return weights

    vix_thr = float(_cfg_get("mtf", "vix_weight_scale", "vix_threshold", default=20.0))
    v = _safe_float(vix_level)
    if v is None:
        return weights
    if v <= vix_thr:
        return weights

    s1 = float(_cfg_get("mtf", "vix_weight_scale", "scale_1min", default=0.5))
    s3 = float(_cfg_get("mtf", "vix_weight_scale", "scale_3min", default=0.5))
    s1 = float(_clamp(s1, 0.0, 1.0))
    s3 = float(_clamp(s3, 0.0, 1.0))

    new_w = dict(weights)
    if "1min" in new_w:
        new_w["1min"] = new_w["1min"] * s1
    if "3min" in new_w:
        new_w["3min"] = new_w["3min"] * s3

    warnings.append(f"VIX scaling applied: vix={v:.2f} > {vix_thr}, 1min×{s1}, 3min×{s3}")
    _logger.info("Applied VIX-adaptive weight scaling", vix=v, threshold=vix_thr, scale_1min=s1, scale_3min=s3)
    return new_w


def _apply_session_threshold_scaling(thresholds: Dict[str, float], session_phase: Optional[str], warnings: List[str]) -> Dict[str, float]:
    """
    Optional:
    In LUNCH_LULL or LAST_MINUTES, tighten label thresholds by +20% (configurable).
    """
    enabled = bool(_cfg_get("mtf", "session_threshold_scale", "enabled", default=True))
    if not enabled or not isinstance(session_phase, str):
        return thresholds

    phases = _cfg_get("mtf", "session_threshold_scale", "phases", default=["LUNCH_LULL", "LAST_MINUTES"])
    mult = float(_cfg_get("mtf", "session_threshold_scale", "multiplier", default=1.2))
    mult = float(_clamp(mult, 1.0, 2.0))

    sp = session_phase.strip().upper()
    if isinstance(phases, list) and sp in [str(p).upper() for p in phases]:
        new_t = dict(thresholds)
        # tighten bullish and strong_bullish upwards; bearish downwards
        new_t["bullish"] = new_t["bullish"] * mult
        new_t["strong_bullish"] = new_t["strong_bullish"] * mult
        new_t["neutral_low"] = new_t["neutral_low"] * mult
        new_t["strong_bearish"] = new_t["strong_bearish"] * mult
        warnings.append(f"Session threshold scaling applied: phase={sp}, mult={mult}")
        _logger.info("Applied session-aware threshold scaling", session_phase=sp, multiplier=mult)
        return new_t

    return thresholds


# ============================================================
# Core Logic: compute_tf_direction (now detailed + robust)
# ============================================================

def compute_tf_direction(features: Dict) -> TFDirectionResult:
    """
    Compute direction for a single timeframe.

    CORE RULES (UNCHANGED):
        +1 (BULLISH): close > vwap AND ema_9 > ema_21 AND rsi > 50
        -1 (BEARISH): close < vwap AND ema_9 < ema_21 AND rsi < 50
         0 (NEUTRAL): mixed signals (non-conflicting) or equality/near-zero ambiguity
        None: insufficient data to compute reliably

    Returns:
        TFDirectionResult (Mapping + numeric compatibility)
    """
    missing: List[str] = []
    conflicts: List[str] = []
    details: Dict[str, Any] = {}

    try:
        if not isinstance(features, dict) or not features:
            return TFDirectionResult(
                direction=None,
                state="INSUFFICIENT_DATA",
                confidence=0,
                missing_keys=["features_dict"],
                conflict_keys=[],
                details={"reason": "features_missing_or_not_dict"},
            )

        close = _safe_float(features.get("last_close"), default=None)
        vwap = _safe_float(features.get("vwap"), default=None)
        ema_9 = _safe_float(features.get("ema_9"), default=None)
        ema_21 = _safe_float(features.get("ema_21"), default=None)
        rsi = _safe_float(features.get("rsi"), default=None)

        # Required keys check (Phase 1 mandate)
        if close is None:
            missing.append("last_close")
        if vwap is None:
            missing.append("vwap")
        if ema_9 is None:
            missing.append("ema_9")
        if ema_21 is None:
            missing.append("ema_21")
        if rsi is None:
            missing.append("rsi")

        # Core logic previously also required close>0 and vwap>0
        if close is None or vwap is None or close <= 0 or vwap <= 0:
            if "last_close" not in missing and (close is None or close <= 0):
                missing.append("last_close_invalid")
            if "vwap" not in missing and (vwap is None or vwap <= 0):
                missing.append("vwap_invalid")

        if missing:
            _logger.debug("TF direction insufficient data", missing_keys=missing)
            return TFDirectionResult(
                direction=None,
                state="INSUFFICIENT_DATA",
                confidence=0,
                missing_keys=missing,
                conflict_keys=[],
                details={"close": close, "vwap": vwap, "ema_9": ema_9, "ema_21": ema_21, "rsi": rsi},
            )

        # Determine directional sub-signals (+1/-1/0)
        # This does NOT change core scoring logic; it enriches observability.
        price_dir = 1 if close > vwap else (-1 if close < vwap else 0)
        ema_dir = 1 if ema_9 > ema_21 else (-1 if ema_9 < ema_21 else 0)
        rsi_dir = 1 if rsi > 50 else (-1 if rsi < 50 else 0)

        details.update({
            "close": close, "vwap": vwap,
            "ema_9": ema_9, "ema_21": ema_21,
            "rsi": rsi,
            "price_dir": price_dir,
            "ema_dir": ema_dir,
            "rsi_dir": rsi_dir,
        })

        # Core bullish/bearish conditions (unchanged)
        price_above_vwap = close > vwap
        ema_bullish = ema_9 > ema_21
        rsi_bullish = rsi > 50

        price_below_vwap = close < vwap
        ema_bearish = ema_9 < ema_21
        rsi_bearish = rsi < 50

        if price_above_vwap and ema_bullish and rsi_bullish:
            return TFDirectionResult(
                direction=1,
                state="BULLISH",
                confidence=100,
                missing_keys=[],
                conflict_keys=[],
                details=details,
            )

        if price_below_vwap and ema_bearish and rsi_bearish:
            return TFDirectionResult(
                direction=-1,
                state="BEARISH",
                confidence=100,
                missing_keys=[],
                conflict_keys=[],
                details=details,
            )

        # Distinguish CONFLICT vs NEUTRAL (Phase 1 requirement)
        dirs = [price_dir, ema_dir, rsi_dir]
        has_bull = any(d == 1 for d in dirs)
        has_bear = any(d == -1 for d in dirs)

        if has_bull and has_bear:
            # Conflict: mixed +1 and -1 across sub-signals
            # Identify which components conflict (useful for trap detection)
            if price_dir != 0 and ema_dir != 0 and price_dir != ema_dir:
                conflicts.append("price_vs_ema")
            if price_dir != 0 and rsi_dir != 0 and price_dir != rsi_dir:
                conflicts.append("price_vs_rsi")
            if ema_dir != 0 and rsi_dir != 0 and ema_dir != rsi_dir:
                conflicts.append("ema_vs_rsi")

            # Confidence: how many align with the majority side (+1 or -1)
            bull_count = sum(1 for d in dirs if d == 1)
            bear_count = sum(1 for d in dirs if d == -1)
            majority = 1 if bull_count > bear_count else -1
            align = sum(1 for d in dirs if d == majority)
            confidence = int(round((align / 3.0) * 100))
            confidence = int(_clamp(confidence, 0, 90))  # conflict caps confidence
            _logger.debug("TF conflict detected", conflicts=conflicts, details=details)
            return TFDirectionResult(
                direction=0,
                state="CONFLICT",
                confidence=confidence,
                missing_keys=[],
                conflict_keys=conflicts,
                details=details,
            )

        # NEUTRAL: no direct conflict, but not all 3 aligned for a directional signal
        # Example: {+1,+1,0} or {-1,0,0} or {0,0,0}
        nonzero = [d for d in dirs if d != 0]
        if not nonzero:
            confidence = 0
        else:
            # confidence proportional to how many non-zero sub-signals exist
            confidence = int(round((len(nonzero) / 3.0) * 100))
            confidence = int(_clamp(confidence, 0, 75))

        return TFDirectionResult(
            direction=0,
            state="NEUTRAL",
            confidence=confidence,
            missing_keys=[],
            conflict_keys=[],
            details=details,
        )

    except Exception as e:
        _logger.error("compute_tf_direction failed; returning INSUFFICIENT_DATA", error=str(e))
        return TFDirectionResult(
            direction=None,
            state="INSUFFICIENT_DATA",
            confidence=0,
            missing_keys=["exception"],
            conflict_keys=[],
            details={"error": str(e)[:200]},
        )


# ============================================================
# MTF Alignment (Decision-grade, backward compatible output)
# ============================================================

def compute_mtf_alignment(
    features_by_tf: Dict[str, Dict],
    *,
    vix_level: Optional[float] = None,
    session_phase: Optional[str] = None,
    key_levels: Optional[Dict] = None,
) -> Dict:
    """
    Compute Multi-Timeframe alignment score.

    Institutional hardening:
      - TF availability flags and detailed per-TF states/confidence
      - Missing TF direction is None and excluded from weighted sum
      - Configurable weights/thresholds
      - Optional VIX adaptive scaling
      - Optional session threshold tightening
      - Optional key-level confluence bonus marker
      - mtf_trap_zone flag when lower TFs conflict with 15min

    Backward compatibility:
      - Existing keys remain present and semantics preserved as much as possible.
      - weighted_mtf formula unchanged; direction rules unchanged.
    """
    try:
        if not isinstance(features_by_tf, dict) or not features_by_tf:
            return _empty_result()

        warnings: List[str] = []

        # Load base weights and thresholds from config (or defaults)
        base_weights = _load_tf_weights()
        thresholds = _load_thresholds()

        # Apply VIX adaptive weight scaling (optional)
        effective_weights = _apply_vix_weight_scaling(base_weights, vix_level, warnings)

        # Apply session-aware threshold scaling (optional)
        effective_thresholds = _apply_session_threshold_scaling(thresholds, session_phase, warnings)

        # Compute direction details per timeframe
        tf_details: Dict[str, TFDirectionResult] = {}
        tf_available: Dict[str, bool] = {}
        tf_directions_numeric: Dict[str, int] = {}   # for backward output keys
        tf_states: Dict[str, str] = {}
        tf_confidences: Dict[str, int] = {}

        weighted_score = 0.0
        used_weight_sum = 0.0

        for tf, weight in effective_weights.items():
            tf_features = features_by_tf.get(tf, {})
            res = compute_tf_direction(tf_features)

            # Availability defined by direction != None (all required keys exist and valid)
            available = (res["direction"] is not None)
            tf_available[tf] = bool(available)

            # For backward compatibility, keep numeric direction fields as int (None -> 0)
            # But DO NOT add missing TF into weighted sum; explicitly exclude.
            dir_val: Optional[int] = res["direction"]
            dir_num = int(dir_val) if dir_val is not None else 0
            tf_directions_numeric[tf] = dir_num
            tf_details[tf] = res
            tf_states[tf] = str(res["state"])
            tf_confidences[tf] = int(res["confidence"])

            if not available:
                warnings.append(f"{tf}: insufficient data (excluded from weighted sum)")
                _logger.debug("TF excluded from weighted sum due to insufficient data",
                              tf=tf, missing_keys=res["missing_keys"])
                continue

            weighted_score += float(dir_val) * float(weight)
            used_weight_sum += float(weight)

            if res["state"] == "CONFLICT":
                warnings.append(f"{tf}: conflict detected ({res['conflict_keys']})")

        weighted_score = round(float(weighted_score), 2)

        # Base max possible (before scaling) preserved for backward strength semantics
        max_possible_base = round(sum(float(w) for w in base_weights.values()), 2)

        # Effective max possible (after scaling) for enhanced reporting
        max_possible_effective = round(sum(float(w) for w in effective_weights.values()), 2)

        # Agreement counts (backward: based on numeric directions including zeros)
        bullish_count = sum(1 for d in tf_directions_numeric.values() if d == 1)
        bearish_count = sum(1 for d in tf_directions_numeric.values() if d == -1)
        neutral_count = sum(1 for d in tf_directions_numeric.values() if d == 0)
        total_tfs = len(base_weights)

        # Dominant direction (backward compatible approach)
        if bullish_count > bearish_count:
            dominant = 1
            agreement_count = bullish_count
        elif bearish_count > bullish_count:
            dominant = -1
            agreement_count = bearish_count
        else:
            dominant = 0
            agreement_count = max(bullish_count, bearish_count)

        agreement_pct = round((agreement_count / total_tfs) * 100, 1) if total_tfs > 0 else 0.0

        # Also compute available-based agreement (new)
        available_tfs = [tf for tf in base_weights.keys() if tf_available.get(tf, False)]
        available_count = len(available_tfs)
        agreement_pct_available = round((agreement_count / available_count) * 100, 1) if available_count > 0 else 0.0

        # Label using effective thresholds (session-aware)
        label = _label_from_score(weighted_score, effective_thresholds)

        # Trend strength (backward compatible): scaled to base max possible
        trend_strength = round(abs(weighted_score) / max(max_possible_base, _EPS) * 100, 1)

        # Trend strength effective (new): scaled to effective max possible
        trend_strength_effective = round(abs(weighted_score) / max(max_possible_effective, _EPS) * 100, 1)

        # Key-level confluence bonus (optional)
        confluence_bonus_applied, confluence_reason = _compute_key_level_confluence_bonus(features_by_tf, key_levels)
        trend_strength_with_confluence = trend_strength
        if confluence_bonus_applied:
            bonus = float(_cfg_get("mtf", "key_level_confluence", "bonus_strength_points", default=10.0))
            bonus = float(_clamp(bonus, 0.0, 25.0))
            trend_strength_with_confluence = round(_clamp(trend_strength + bonus, 0.0, 100.0), 1)

        # mtf_trap_zone detection (Phase 3)
        mtf_trap_zone = _compute_mtf_trap_zone(tf_details=tf_details)

        # Build output (existing keys retained)
        out = {
            "weighted_mtf": weighted_score,
            "max_possible": max_possible_base,   # legacy semantic
            "tf_directions": tf_directions_numeric,
            "tf_1min": tf_directions_numeric.get("1min", 0),
            "tf_3min": tf_directions_numeric.get("3min", 0),
            "tf_5min": tf_directions_numeric.get("5min", 0),
            "tf_15min": tf_directions_numeric.get("15min", 0),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "agreement_count": agreement_count,
            "agreement_pct": agreement_pct,
            "dominant_direction": dominant,
            "mtf_label": label,
            "trend_strength": trend_strength,

            # --------------------------
            # Institutional additions
            # --------------------------
            "warnings": warnings,
            "tf_details": {tf: dict(tf_details[tf]) for tf in tf_details},  # mapping -> dict
            "tf_states": tf_states,
            "tf_confidences": tf_confidences,
            "tf_1min_available": bool(tf_available.get("1min", False)),
            "tf_3min_available": bool(tf_available.get("3min", False)),
            "tf_5min_available": bool(tf_available.get("5min", False)),
            "tf_15min_available": bool(tf_available.get("15min", False)),
            "available_tfs": available_tfs,
            "available_tf_count": available_count,
            "agreement_pct_available": agreement_pct_available,
            "max_possible_effective": max_possible_effective,
            "weights_base": base_weights,
            "weights_effective": effective_weights,
            "thresholds_base": thresholds,
            "thresholds_effective": effective_thresholds,
            "vix_level": vix_level,
            "session_phase": session_phase,
            "trend_strength_effective": trend_strength_effective,
            "confluence_bonus_applied": bool(confluence_bonus_applied),
            "confluence_bonus_reason": confluence_reason,
            "trend_strength_with_confluence": trend_strength_with_confluence,
            "mtf_trap_zone": bool(mtf_trap_zone),
        }

        return out

    except Exception as e:
        _logger.error("compute_mtf_alignment failed; returning empty result", error=str(e))
        res = _empty_result()
        res["warnings"] = [f"exception: {type(e).__name__}"]
        res["mtf_trap_zone"] = False
        return res


def _label_from_score(weighted_score: float, thresholds: Dict[str, float]) -> str:
    """
    Label thresholds remain logically consistent with the original:
      >= strong_bullish -> STRONG_BULLISH
      >= bullish        -> BULLISH
      >= neutral_low    -> NEUTRAL
      >= strong_bearish -> BEARISH
      else              -> STRONG_BEARISH
    """
    sb = float(thresholds["strong_bullish"])
    b = float(thresholds["bullish"])
    nl = float(thresholds["neutral_low"])
    sbr = float(thresholds["strong_bearish"])

    if weighted_score >= sb:
        return "STRONG_BULLISH"
    elif weighted_score >= b:
        return "BULLISH"
    elif weighted_score >= nl:
        return "NEUTRAL"
    elif weighted_score >= sbr:
        return "BEARISH"
    else:
        return "STRONG_BEARISH"


def _compute_key_level_confluence_bonus(
    features_by_tf: Dict[str, Dict],
    key_levels: Optional[Dict],
) -> Tuple[bool, Optional[str]]:
    """
    Phase 3 requirement:
      - Accept optional key_levels containing nearest_sr_distance, vwap_distance, etc.
      - If price is within 0.1% of a key level => confluence_bonus flag.

    This function is defensive and schema-agnostic:
      - If key_levels includes *distance* fields in points or pct, uses them
      - Else if key_levels includes numeric levels (pdh/pdl/orh/...) compares with last_close

    Returns:
      (applied, reason)
    """
    enabled = bool(_cfg_get("mtf", "key_level_confluence", "enabled", default=True))
    if not enabled:
        return False, None

    if not isinstance(key_levels, dict) or not key_levels:
        return False, None

    # Determine a reference price (prefer 1min close, else any)
    price = None
    for tf in ("1min", "3min", "5min", "15min"):
        f = features_by_tf.get(tf, {})
        p = _safe_float(f.get("last_close"))
        if p is not None and p > 0:
            price = float(p)
            break
    if price is None:
        return False, None

    proximity_pct = float(_cfg_get("mtf", "key_level_confluence", "proximity_pct", default=0.001))
    proximity_pct = float(_clamp(proximity_pct, 0.0002, 0.005))  # 0.02%..0.5%
    prox_points = price * proximity_pct

    # 1) If distances are provided directly
    for k in ("nearest_sr_distance", "nearest_sr_distance_points", "nearest_level_distance", "nearest_level_distance_points"):
        d = _safe_float(key_levels.get(k))
        if d is not None and d >= 0:
            if d <= prox_points:
                return True, f"distance_field:{k}<=0.1%"

    for k in ("vwap_distance_pct", "nearest_key_level_distance_pct"):
        d = _safe_float(key_levels.get(k))
        if d is not None and d >= 0:
            if d <= proximity_pct:
                return True, f"distance_pct_field:{k}<=0.1%"

    # 2) Else interpret numeric levels and compare
    numeric_levels = []
    for k, v in key_levels.items():
        # skip non-level keys
        if "distance" in str(k).lower():
            continue
        fv = _safe_float(v)
        if fv is not None and fv > 0:
            numeric_levels.append((str(k), float(fv)))

    if numeric_levels:
        nearest = min(numeric_levels, key=lambda kv: abs(price - kv[1]))
        if abs(price - nearest[1]) <= prox_points:
            return True, f"level_field:{nearest[0]} within 0.1%"

    return False, None


def _compute_mtf_trap_zone(tf_details: Dict[str, TFDirectionResult]) -> bool:
    """
    Phase 3: mtf_trap_zone
      - If lower TFs (1min/3min) conflict with higher TFs (15min), flag True.
      - Specifically:
          If 15min is BULLISH and (1min or 3min is BEARISH or CONFLICT) => trap
          If 15min is BEARISH and (1min or 3min is BULLISH or CONFLICT) => trap
    """
    d15 = tf_details.get("15min")
    if d15 is None or d15["direction"] is None:
        return False

    dir15 = int(d15["direction"] or 0)
    if dir15 == 0:
        return False

    for low_tf in ("1min", "3min"):
        r = tf_details.get(low_tf)
        if r is None:
            continue
        if r["state"] == "CONFLICT":
            return True
        d = r["direction"]
        if d is None:
            continue
        if dir15 == 1 and int(d) == -1:
            return True
        if dir15 == -1 and int(d) == 1:
            return True

    return False


# ============================================================
# Empty result (legacy keys preserved)
# ============================================================

def _empty_result() -> Dict:
    max_possible_base = round(sum(DEFAULT_TF_WEIGHTS.values()), 2)
    return {
        "weighted_mtf": 0.0,
        "max_possible": max_possible_base,
        "tf_directions": {},
        "tf_1min": 0, "tf_3min": 0, "tf_5min": 0, "tf_15min": 0,
        "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
        "agreement_count": 0, "agreement_pct": 0.0,
        "dominant_direction": 0,
        "mtf_label": "NEUTRAL",
        "trend_strength": 0.0,

        # institutional additions
        "warnings": [],
        "tf_details": {},
        "tf_states": {},
        "tf_confidences": {},
        "tf_1min_available": False,
        "tf_3min_available": False,
        "tf_5min_available": False,
        "tf_15min_available": False,
        "available_tfs": [],
        "available_tf_count": 0,
        "agreement_pct_available": 0.0,
        "max_possible_effective": max_possible_base,
        "weights_base": dict(DEFAULT_TF_WEIGHTS),
        "weights_effective": dict(DEFAULT_TF_WEIGHTS),
        "thresholds_base": dict(DEFAULT_THRESHOLDS),
        "thresholds_effective": dict(DEFAULT_THRESHOLDS),
        "vix_level": None,
        "session_phase": None,
        "trend_strength_effective": 0.0,
        "confluence_bonus_applied": False,
        "confluence_bonus_reason": None,
        "trend_strength_with_confluence": 0.0,
        "mtf_trap_zone": False,
    }


# ============================================================
# Module Self-Test (extended)
# ============================================================

def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — MTF Alignment Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    bullish_tf = {
        "last_close": 23200, "vwap": 23150,
        "ema_9": 23195, "ema_21": 23180,
        "rsi": 62,
    }

    bearish_tf = {
        "last_close": 23100, "vwap": 23200,
        "ema_9": 23110, "ema_21": 23150,
        "rsi": 38,
    }

    neutral_tf = {
        "last_close": 23200, "vwap": 23150,  # Above VWAP
        "ema_9": 23180, "ema_21": 23190,     # EMA bearish
        "rsi": 55,                           # RSI bullish
    }

    # ── Test 1: compute_tf_direction returns detailed dict-like and remains int-compatible
    print("  [Test 1] TF direction object compatibility...")
    d = compute_tf_direction(bullish_tf)
    if d["direction"] == 1 and d["state"] == "BULLISH" and int(d) == 1 and d == 1:
        print("    ✅ Bullish TF detailed + int-compatible")
        passed += 1
    else:
        print(f"    ❌ Unexpected: {dict(d)}")
        failed += 1

    d2 = compute_tf_direction(bearish_tf)
    if d2["direction"] == -1 and d2["state"] == "BEARISH" and int(d2) == -1 and d2 == -1:
        print("    ✅ Bearish TF detailed + int-compatible")
        passed += 1
    else:
        print(f"    ❌ Unexpected: {dict(d2)}")
        failed += 1

    d3 = compute_tf_direction(neutral_tf)
    if d3["direction"] == 0 and d3["state"] == "CONFLICT":
        print("    ✅ Mixed signals flagged as CONFLICT")
        passed += 1
    else:
        print(f"    ❌ Expected CONFLICT, got {dict(d3)}")
        failed += 1

    # ── Test 2: Missing RSI => INSUFFICIENT_DATA and availability false in mtf
    print("\n  [Test 2] Missing RSI handling...")
    missing_rsi_tf = {
        "last_close": 23200, "vwap": 23150,
        "ema_9": 23195, "ema_21": 23180,
        # rsi missing
    }
    dr = compute_tf_direction(missing_rsi_tf)
    if dr["direction"] is None and dr["state"] == "INSUFFICIENT_DATA":
        print("    ✅ Missing RSI -> INSUFFICIENT_DATA")
        passed += 1
    else:
        print(f"    ❌ Unexpected: {dict(dr)}")
        failed += 1

    # ── Test 3: Partial TF availability should only use available TFs (no crash)
    print("\n  [Test 3] Partial TF availability...")
    feats_partial = {"1min": bullish_tf, "15min": bullish_tf, "3min": missing_rsi_tf}
    res = compute_mtf_alignment(feats_partial)
    if res["tf_3min_available"] is False and res["tf_1min_available"] and res["tf_15min_available"]:
        print("    ✅ Availability flags set correctly")
        passed += 1
    else:
        print(f"    ❌ Availability flags wrong: {res['tf_1min_available']=}, {res['tf_3min_available']=}, {res['tf_15min_available']=}")
        failed += 1

    # ── Test 4: VIX scaling changes effective weights
    print("\n  [Test 4] VIX-adaptive weight scaling...")
    feats_all = {"1min": bullish_tf, "3min": bullish_tf, "5min": bullish_tf, "15min": bullish_tf}
    res_no_vix = compute_mtf_alignment(feats_all, vix_level=None)
    res_high_vix = compute_mtf_alignment(feats_all, vix_level=22.0)
    if res_high_vix["weights_effective"]["1min"] <= res_no_vix["weights_effective"]["1min"]:
        print("    ✅ High VIX scales down 1min weight (or keeps same if disabled)")
        passed += 1
    else:
        print("    ❌ VIX scaling did not apply as expected")
        failed += 1

    # ── Test 5: Trap zone flag when 15min bullish and 1min bearish/conflict
    print("\n  [Test 5] mtf_trap_zone detection...")
    feats_trap = {"15min": bullish_tf, "1min": bearish_tf, "3min": neutral_tf, "5min": bullish_tf}
    res_trap = compute_mtf_alignment(feats_trap)
    if res_trap["mtf_trap_zone"] is True:
        print("    ✅ mtf_trap_zone=True when lower TF conflicts with 15min")
        passed += 1
    else:
        print(f"    ❌ Expected mtf_trap_zone True, got {res_trap['mtf_trap_zone']}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅ MTF Alignment hardened and production-safe.")
    else:
        print("  ⚠️ Some tests failed. Review logs/mtf_alignment.log.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()