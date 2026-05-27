"""
Junior Aladdin - Momentum & Oscillators (Layer 2B) — Institutional Grade (Controlled Upgrade)

This module computes momentum indicators from multi-timeframe candles with:
- Timeframe synchronization (alignment across 1m/3m/5m/15m)
- Data freshness validation (latency + stale gating) with offline bypass option
- Gap-aware handling (consecutive missing candles at tail -> invalidate TF for this cycle)
- Volume/liquidity filter (illiquid candle skip)
- RSI (Wilder), MACD, Stochastic, Acceleration (raw+normalized), ROC (raw+EMA3 smooth)
- MACD histogram slope smoothing (SMA(3) then slope over last 5)
- Optional MTF trend integration filter using mtf_alignment_signal from trend_context
- Optional event awareness (hard override under major event within 15 minutes; soft risk under 30)

CONTROLLED FINAL UPGRADES INCLUDED
---------------------------------
1) Instrument Intelligence (controlled overrides):
   - Reads instrument_class from instrument_spec (INDEX/STOCK/FUTURE/OPTION)
   - Applies class-specific config overrides ONLY for:
       * rsi_period and rsi_period_volatile
       * illiquid_volume_ratio
       * rsi_extreme_level
   - Uses helper _get_instrument_config(key, default, instrument_class):
       * checks features.{instrument_class}_{key}
       * falls back to features.{key}

2) MTF Trend Integration (FILTER, not replacement):
   - Extracts mtf_alignment and mtf_alignment_signal robustly from trend_context
   - Computes mom_dir using existing RSI+MACD heuristic
   - Sets momentum_trend_conflict if mom_dir != mtf_alignment_signal (when both non-zero)
   - trend_aligned_rsi = RSI only if conflict is False else None
   - Does NOT alter original RSI/MACD/Stoch computations.

3) Event Awareness (smart, not over-defensive):
   - Reads event_severity and event_minutes_away from session_context
   - If severity==2 and minutes_away<15: event_override=True, data_freshness="EVENT_RISK"
     -> nullify all per-timeframe indicator features (metadata preserved), skip calculations
   - Else if severity==2 and minutes_away<30: event_risk=True (soft mode)
     -> keep features but add event_risk=True so downstream can be cautious

OFFLINE / SELF-TEST NOTE
------------------------
Use skip_freshness_check=True for offline self-tests/historical replay so past timestamps
are not treated as stale. In this mode data_freshness="OFFLINE".

CRASH-PROOF GUARANTEE
---------------------
compute_momentum_features() MUST NEVER raise. Returns {} on catastrophic failure.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.utils.helpers import IST, ist_now

_LOG = setup_logger("momentum_features")


# ------------------------- Logger shim -------------------------

def _emit_log(level: str, msg: str, **fields: Any) -> None:
    """Safe logger shim supporting structlog-style and stdlib logging.Logger. Never raises."""
    try:
        fn = getattr(_LOG, level, None)
        if fn is None:
            return
        if not fields:
            fn(msg)
            return
        try:
            fn(msg, **fields)
            return
        except TypeError:
            parts = []
            for k in sorted(fields.keys()):
                try:
                    parts.append(f"{k}={fields[k]!r}")
                except Exception:
                    parts.append(f"{k}=<unrepr>")
            fn(f"{msg} | " + ", ".join(parts))
    except Exception:
        return


def _rate_limited(state: Dict[str, Any], key: str, now: datetime, window_sec: float = 60.0) -> bool:
    """Return True if allowed to log now (rate-limited), else False."""
    try:
        last = state.get(key)
        if last is None:
            state[key] = now
            return True
        if isinstance(last, datetime):
            if (now - last).total_seconds() >= window_sec:
                state[key] = now
                return True
        return False
    except Exception:
        return False


# ------------------------- Safe parsing -------------------------

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            f = float(x)
            return f if math.isfinite(f) else None
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            s = s.replace(",", "")
            f = float(s)
            return f if math.isfinite(f) else None
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        if isinstance(x, bool):
            return default
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            if not math.isfinite(x):
                return default
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return default
            s = s.replace(",", "")
            f = float(s)
            if not math.isfinite(f):
                return default
            return int(f)
        return int(x)
    except Exception:
        return default


def _normalize_tf_key(tf: str) -> Optional[str]:
    if not isinstance(tf, str):
        return None
    t = tf.strip().lower()
    mapping = {
        "1m": "1min", "1min": "1min", "1minute": "1min",
        "3m": "3min", "3min": "3min", "3minute": "3min",
        "5m": "5min", "5min": "5min", "5minute": "5min",
        "15m": "15min", "15min": "15min", "15minute": "15min",
    }
    return mapping.get(t)


def _cfg_int(section: str, key: str, default: int) -> int:
    try:
        v = Config.get(section, key, default=default)
        try:
            return int(float(v))
        except Exception:
            return int(default)
    except Exception:
        return int(default)


def _cfg_float(section: str, key: str, default: float) -> float:
    try:
        v = Config.get(section, key, default=default)
        f = _safe_float(v)
        return float(default if f is None else f)
    except Exception:
        return float(default)


# ------------------------- Controlled upgrade helpers -------------------------

def _get_instrument_config(key: str, default: Any, instrument_class: str) -> Any:
    """
    Controlled instrument override helper:
    1) Try features.{INSTRUMENT_CLASS}_{key}
    2) Fallback features.{key}
    """
    cls = str(instrument_class or "").strip().upper()
    if cls:
        v = Config.get("features", f"{cls}_{key}", default=None)
        if v is not None:
            return v
    return Config.get("features", key, default=default)


def _get_mtf_alignment(trend_context: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[int]]:
    """
    Robustly extract (mtf_alignment, mtf_alignment_signal).
    Handles missing keys and nested dicts without crashing.
    """
    try:
        if not isinstance(trend_context, dict):
            return None, None

        def coerce_float(x: Any) -> Optional[float]:
            f = _safe_float(x)
            return float(f) if f is not None else None

        def coerce_signal(x: Any) -> Optional[int]:
            if x is None or isinstance(x, bool):
                return None
            try:
                iv = int(float(x))
            except Exception:
                return None
            if iv > 0:
                return 1
            if iv < 0:
                return -1
            return 0

        # direct
        if "mtf_alignment" in trend_context or "mtf_alignment_signal" in trend_context:
            a = coerce_float(trend_context.get("mtf_alignment"))
            s = coerce_signal(trend_context.get("mtf_alignment_signal"))
            if s is None and a is not None:
                s = 0 if a == 0 else (1 if a > 0 else -1)
            return a, s

        # common nest keys
        for k in ("trend", "price_features", "context", "features", "global"):
            v = trend_context.get(k)
            if isinstance(v, dict):
                a, s = _get_mtf_alignment(v)
                if a is not None or s is not None:
                    return a, s

        # scan shallow nested dicts
        for v in trend_context.values():
            if isinstance(v, dict):
                if "mtf_alignment" in v or "mtf_alignment_signal" in v:
                    a = coerce_float(v.get("mtf_alignment"))
                    s = coerce_signal(v.get("mtf_alignment_signal"))
                    if s is None and a is not None:
                        s = 0 if a == 0 else (1 if a > 0 else -1)
                    return a, s

        return None, None
    except Exception:
        return None, None


# ------------------------- Timestamp parsing -------------------------

def _parse_ts_to_ist(ts: Any) -> Optional[datetime]:
    """Parse candle timestamp into tz-aware IST datetime."""
    try:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=IST)
            return ts.astimezone(IST)

        if isinstance(ts, (int, float)):
            iv = int(ts)
            if iv <= 0:
                return None
            digits = len(str(abs(iv)))
            if digits == 13:
                sec = iv / 1000.0
            elif digits == 10:
                sec = float(iv)
            else:
                return None
            return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(IST)

        if isinstance(ts, str):
            s = ts.strip()
            if not s:
                return None
            if s.isdigit():
                if len(s) == 13:
                    sec = int(s) / 1000.0
                elif len(s) == 10:
                    sec = float(int(s))
                else:
                    return None
                return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(IST)

            s_iso = s.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s_iso)
            except Exception:
                dt = None
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                    try:
                        dt = datetime.strptime(s, fmt)
                        break
                    except Exception:
                        continue
                if dt is None:
                    return None

            if dt.tzinfo is None:
                return dt.replace(tzinfo=IST)
            return dt.astimezone(IST)

        return None
    except Exception:
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    try:
        return dt.isoformat() if isinstance(dt, datetime) else None
    except Exception:
        return None


# ------------------------- Indicator math -------------------------

def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    n = len(values)
    if period <= 1 or n == 0:
        return [None] * n
    out: List[Optional[float]] = [None] * n
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    ema = sma
    for i in range(period, n):
        ema = alpha * values[i] + (1.0 - alpha) * ema
        out[i] = ema
    return out


def _sma_optional(values: List[Optional[float]], period: int) -> List[Optional[float]]:
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if period <= 0 or n < period:
        return out
    for i in range(period - 1, n):
        window = values[i - period + 1:i + 1]
        if any(v is None for v in window):
            out[i] = None
            continue
        out[i] = sum(float(v) for v in window) / period  # type: ignore[arg-type]
    return out


def _linreg_slope(y: List[float]) -> Optional[float]:
    n = len(y)
    if n < 2:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    num = 0.0
    den = 0.0
    for i, yi in enumerate(y):
        dx = i - x_mean
        dy = yi - y_mean
        num += dx * dy
        den += dx * dx
    if den <= 0.0:
        return None
    return num / den


def _rsi_wilder_series(closes: List[float], period: int) -> List[Optional[float]]:
    n = len(closes)
    out: List[Optional[float]] = [None] * n
    if period <= 0 or n < period + 1:
        return out

    gains = []
    losses = []
    for i in range(1, period + 1):
        chg = closes[i] - closes[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    out[period] = 100.0 if avg_loss == 0.0 else (100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

    for i in range(period + 1, n):
        chg = closes[i] - closes[i - 1]
        gain = max(chg, 0.0)
        loss = max(-chg, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = 100.0 if avg_loss == 0.0 else (100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

    return out


def _macd_series(
    closes: List[float],
    fast: int,
    slow: int,
    signal: int,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    n = len(closes)
    macd_line: List[Optional[float]] = [None] * n
    signal_line: List[Optional[float]] = [None] * n
    hist: List[Optional[float]] = [None] * n

    if fast <= 0 or slow <= 0 or signal <= 0 or n == 0:
        return macd_line, signal_line, hist
    if n < slow + signal:
        return macd_line, signal_line, hist

    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)
    for i in range(n):
        if ema_fast[i] is None or ema_slow[i] is None:
            macd_line[i] = None
        else:
            macd_line[i] = float(ema_fast[i] - ema_slow[i])

    first_valid = None
    for i in range(n):
        if macd_line[i] is not None:
            first_valid = i
            break
    if first_valid is None:
        return macd_line, signal_line, hist

    seed_end = None
    consec = 0
    for i in range(first_valid, n):
        if macd_line[i] is None:
            consec = 0
            continue
        consec += 1
        if consec >= signal:
            seed_end = i
            break
    if seed_end is None:
        return macd_line, signal_line, hist

    alpha = 2.0 / (signal + 1.0)
    seed_vals = [macd_line[j] for j in range(seed_end - signal + 1, seed_end + 1)]
    if any(v is None for v in seed_vals):
        return macd_line, signal_line, hist
    ema_sig = sum(float(v) for v in seed_vals) / signal  # type: ignore[arg-type]
    signal_line[seed_end] = ema_sig

    for i in range(seed_end + 1, n):
        ml = macd_line[i]
        if ml is None:
            signal_line[i] = None
            continue
        ema_sig = alpha * float(ml) + (1.0 - alpha) * ema_sig
        signal_line[i] = ema_sig

    for i in range(n):
        if macd_line[i] is None or signal_line[i] is None:
            hist[i] = None
        else:
            hist[i] = float(macd_line[i] - signal_line[i])

    return macd_line, signal_line, hist


def _stochastic_last(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    k_period: int,
    d_period: int,
) -> Tuple[Optional[float], Optional[float]]:
    n = len(closes)
    if k_period <= 0 or d_period <= 0:
        return None, None
    if n < k_period:
        return None, None

    k_vals: List[Optional[float]] = [None] * n
    for i in range(k_period - 1, n):
        ll = min(lows[i - k_period + 1:i + 1])
        hh = max(highs[i - k_period + 1:i + 1])
        denom = hh - ll
        if denom <= 0.0:
            k_vals[i] = 0.0
        else:
            k_vals[i] = 100.0 * (closes[i] - ll) / denom

    k_last = k_vals[-1]
    if k_last is None:
        return None, None
    window = [v for v in k_vals[-d_period:] if v is not None]
    if len(window) != d_period:
        return float(k_last), None
    d_last = sum(float(v) for v in window) / d_period
    return float(k_last), float(d_last)


def _stddev(values: List[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, var))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _nullify_tf_features(out: Dict[str, Any], roc_key: str) -> None:
    """Nullify all per-timeframe indicator features; keep metadata already present."""
    for tf in ("1min", "3min", "5min", "15min"):
        prefix = f"{tf}_"
        out[prefix + "is_valid"] = False
        out[prefix + "insufficient_data"] = True
        out[prefix + "illiquid_skip"] = False

        out[prefix + "momentum_trend_conflict"] = None
        out[prefix + "trend_aligned_rsi"] = None

        out[prefix + "rsi_raw"] = None
        out[prefix + "rsi"] = None
        out[prefix + "rsi_extreme"] = None

        out[prefix + "macd_line"] = None
        out[prefix + "macd_signal"] = None
        out[prefix + "macd_histogram"] = None
        out[prefix + "macd_hist_slope"] = None

        out[prefix + "stoch_k"] = None
        out[prefix + "stoch_d"] = None

        out[prefix + "price_acceleration_raw"] = None
        out[prefix + "price_acceleration_norm"] = None

        out[prefix + f"{roc_key}_raw"] = None
        out[prefix + f"{roc_key}_smooth"] = None


# ------------------------- Core computation -------------------------

def compute_momentum_features(
    candles: Dict[str, Any],
    instrument_spec: Optional[Dict[str, Any]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    trend_context: Optional[Dict[str, Any]] = None,
    *,
    skip_freshness_check: bool = False,
) -> Dict[str, Any]:
    """
    Compute momentum features across timeframes with institutional safeguards.
    MUST NEVER raise; returns {} on catastrophic failure.

    skip_freshness_check:
        When True, bypasses stale gating and sets data_freshness="OFFLINE".
        Intended for offline self-tests / historical replay.
    """
    warn_state: Dict[str, Any] = {}
    try:
        if not isinstance(candles, dict):
            _emit_log("error", "compute_momentum_features: candles input is not a dict", candles_type=str(type(candles)))
            return {}

        instrument_spec = instrument_spec if isinstance(instrument_spec, dict) else {}
        instrument_class = str(instrument_spec.get("instrument_class") or instrument_spec.get("type") or "INDEX").strip().upper()
        if instrument_class not in {"INDEX", "STOCK", "FUTURE", "OPTION"}:
            instrument_class = "INDEX"

        session_context = session_context or {}
        session_phase = str(session_context.get("session_phase") or "").strip().upper()
        regime = str(session_context.get("regime") or session_context.get("market_regime") or "").strip().upper()

        # Trend context can be passed or embedded
        if trend_context is None:
            tc = session_context.get("trend_context")
            if isinstance(tc, dict):
                trend_context = tc

        mtf_alignment, mtf_alignment_signal = _get_mtf_alignment(trend_context)

        # Controlled event awareness (TOP of function, before indicator calculations)
        event_severity = _safe_int(session_context.get("event_severity"), default=0)
        event_minutes_away = _safe_float(session_context.get("event_minutes_away"))
        event_override = False
        event_risk = False
        data_freshness_override: Optional[str] = None

        if event_severity == 2 and event_minutes_away is not None:
            if event_minutes_away < 15.0:
                event_override = True
                data_freshness_override = "EVENT_RISK"
            elif event_minutes_away < 30.0:
                event_risk = True

        # Base output metadata
        out: Dict[str, Any] = {
            "session_phase": session_phase or None,
            "regime": regime or None,
            "instrument_class": instrument_class,
            "mtf_alignment": float(mtf_alignment) if mtf_alignment is not None else None,
            "mtf_alignment_signal": int(mtf_alignment_signal) if mtf_alignment_signal is not None else None,
            "event_override": bool(event_override),
            "event_risk": bool(event_risk),
            "event_severity": int(event_severity),
            "event_minutes_away": float(event_minutes_away) if event_minutes_away is not None else None,
        }

        suppress = session_phase in {"PRE_MARKET", "OPENING_AUCTION"}

        # Config (existing safeguards)
        max_latency_sec = _cfg_float("features", "max_latency_sec", default=5.0)
        align_tol_sec = _cfg_float("features", "alignment_tolerance_sec", default=1.0)
        gap_reset_threshold = _cfg_int("features", "gap_reset_threshold", default=2)
        avg_vol_period = _cfg_int("features", "avg_volume_period", default=20)

        # Controlled instrument overrides (ONLY these params)
        rsi_p_raw = _get_instrument_config("rsi_period", default=_cfg_int("features", "rsi_period", 14), instrument_class=instrument_class)
        rsi_p_volatile_raw = _get_instrument_config("rsi_period_volatile", default=_cfg_int("features", "rsi_period_volatile", 21), instrument_class=instrument_class)
        illiquid_ratio_raw = _get_instrument_config("illiquid_volume_ratio", default=_cfg_float("features", "illiquid_volume_ratio", 0.2), instrument_class=instrument_class)
        rsi_extreme_level_raw = _get_instrument_config("rsi_extreme_level", default=_cfg_float("features", "rsi_extreme_level", 95.0), instrument_class=instrument_class)

        rsi_p = _safe_int(rsi_p_raw, default=14)
        rsi_p_volatile = _safe_int(rsi_p_volatile_raw, default=21)
        illiquid_ratio = float(_safe_float(illiquid_ratio_raw) or 0.2)
        rsi_extreme_level = float(_safe_float(rsi_extreme_level_raw) or 95.0)

        # Remaining periods (unchanged)
        macd_fast = _cfg_int("features", "macd_fast", default=12)
        macd_slow = _cfg_int("features", "macd_slow", default=26)
        macd_signal = _cfg_int("features", "macd_signal", default=9)
        stoch_k = _cfg_int("features", "stoch_k_period", default=14)
        stoch_d = _cfg_int("features", "stoch_d_period", default=3)
        roc_p = _cfg_int("features", "roc_period", default=5)
        roc_key = f"roc_{roc_p}"

        # Volatile overrides (existing)
        if regime == "VOLATILE":
            rsi_p = rsi_p_volatile
            macd_fast = _cfg_int("features", "macd_fast_volatile", default=16)
            macd_slow = _cfg_int("features", "macd_slow_volatile", default=32)
            macd_signal = _cfg_int("features", "macd_signal_volatile", default=12)
            stoch_k = _cfg_int("features", "stoch_k_period_volatile", default=21)
            stoch_d = _cfg_int("features", "stoch_d_period_volatile", default=3)

        # Normalize tf candles
        tf_map: Dict[str, List[Dict[str, Any]]] = {}
        for k, dq in candles.items():
            nk = _normalize_tf_key(str(k))
            if nk is None:
                continue
            if dq is None:
                tf_map[nk] = []
                continue
            try:
                tf_map[nk] = list(dq)
            except Exception:
                tf_map[nk] = []
        for tf in ("1min", "3min", "5min", "15min"):
            tf_map.setdefault(tf, [])

        now = ist_now()

        # If event_override: return early, but still populate minimal per-tf metadata + nullified features
        if event_override:
            out["data_freshness"] = data_freshness_override or "EVENT_RISK"
            out["aligned_timestamp"] = None
            out["is_aligned"] = False
            out["alignment_tolerance_sec"] = float(align_tol_sec)

            # Minimal metadata for each tf
            for tf in ("1min", "3min", "5min", "15min"):
                prefix = f"{tf}_"
                raw_list = tf_map.get(tf, [])
                out[prefix + "raw_candle_count"] = int(len(raw_list))
                lr = _parse_ts_to_ist(raw_list[-1].get("timestamp")) if raw_list else None
                out[prefix + "last_candle_timestamp"] = _iso(lr)
                if lr is not None:
                    lat = max(0.0, (now - lr).total_seconds())
                    out[prefix + "latency_sec"] = float(lat)
                    out[prefix + "is_stale"] = bool(lat > max_latency_sec)
                else:
                    out[prefix + "latency_sec"] = None
                    out[prefix + "is_stale"] = None
                out[prefix + "effective_candle_count"] = 0
                out[prefix + "consecutive_missing"] = 0
                out[prefix + "last_valid_timestamp"] = None

            _nullify_tf_features(out, roc_key)
            return out

        # Normal path: build per-tf valid series + alignment + freshness + session suppression + indicators
        volatility_context = session_context.get("volatility_context")
        if not isinstance(volatility_context, dict):
            volatility_context = {}

        valid_ts_set: Dict[str, List[int]] = {}
        last_valid_ts: Dict[str, Optional[datetime]] = {}
        per_tf_data: Dict[str, Dict[str, Any]] = {}

        for tf in ("1min", "3min", "5min", "15min"):
            raw_list = tf_map.get(tf, [])
            prefix = f"{tf}_"
            per_tf_data[tf] = {}

            out[prefix + "raw_candle_count"] = int(len(raw_list))

            lr = _parse_ts_to_ist(raw_list[-1].get("timestamp")) if raw_list else None
            out[prefix + "last_candle_timestamp"] = _iso(lr)
            if lr is not None:
                lat = max(0.0, (now - lr).total_seconds())
                out[prefix + "latency_sec"] = float(lat)
                out[prefix + "is_stale"] = bool(lat > max_latency_sec)
            else:
                out[prefix + "latency_sec"] = None
                out[prefix + "is_stale"] = None

            # consecutive missing OHLC at end
            consec_missing = 0
            for c in reversed(raw_list):
                ts = _parse_ts_to_ist(c.get("timestamp"))
                cl = _safe_float(c.get("close"))
                h = _safe_float(c.get("high"))
                l = _safe_float(c.get("low"))
                if ts is None or cl is None or h is None or l is None:
                    consec_missing += 1
                else:
                    break
            out[prefix + "consecutive_missing"] = int(consec_missing)

            gap_reset = bool(consec_missing > gap_reset_threshold)
            per_tf_data[tf]["gap_reset"] = gap_reset
            if gap_reset and _rate_limited(warn_state, f"gap_reset_{tf}", now, 60.0):
                _emit_log(
                    "warning",
                    "Consecutive missing candles exceeded threshold; invalidating timeframe momentum features",
                    timeframe=tf,
                    consecutive_missing=consec_missing,
                    threshold=gap_reset_threshold,
                )

            closes: List[float] = []
            highs: List[float] = []
            lows: List[float] = []
            volumes: List[float] = []
            ts_valid: List[datetime] = []
            missing_count = 0

            for c in raw_list:
                ts = _parse_ts_to_ist(c.get("timestamp"))
                cl = _safe_float(c.get("close"))
                h = _safe_float(c.get("high"))
                l = _safe_float(c.get("low"))
                v = _safe_float(c.get("volume"))
                if ts is None or cl is None or h is None or l is None:
                    missing_count += 1
                    continue
                closes.append(cl)
                highs.append(h)
                lows.append(l)
                volumes.append(0.0 if v is None or v < 0 else float(v))
                ts_valid.append(ts)

            out[prefix + "effective_candle_count"] = int(len(closes))

            if missing_count > 0 and _rate_limited(warn_state, f"missing_ohlc_{tf}", now, 60.0):
                _emit_log(
                    "warning",
                    "Missing/invalid OHLC in candles; momentum features may be degraded",
                    timeframe=tf,
                    missing_count=missing_count,
                    raw_candles=len(raw_list),
                    effective_candles=len(closes),
                )

            lv = ts_valid[-1] if ts_valid else None
            last_valid_ts[tf] = lv
            out[prefix + "last_valid_timestamp"] = _iso(lv)

            secs = []
            for tsv in ts_valid[-200:]:
                try:
                    secs.append(int(tsv.timestamp()))
                except Exception:
                    continue
            valid_ts_set[tf] = secs

            per_tf_data[tf]["closes"] = closes
            per_tf_data[tf]["highs"] = highs
            per_tf_data[tf]["lows"] = lows
            per_tf_data[tf]["volumes"] = volumes

        # Alignment
        common: Optional[set] = None
        for tf in ("1min", "3min", "5min", "15min"):
            s = set(valid_ts_set.get(tf, []))
            if common is None:
                common = s
            else:
                common.intersection_update(s)

        aligned_dt: Optional[datetime] = None
        is_aligned = False
        if common and len(common) > 0:
            aligned_sec = max(common)
            aligned_dt = datetime.fromtimestamp(aligned_sec, tz=timezone.utc).astimezone(IST)
            ok = True
            for tf in ("1min", "3min", "5min", "15min"):
                found = False
                for sec in valid_ts_set.get(tf, [])[-80:]:
                    if abs(sec - aligned_sec) <= int(math.ceil(align_tol_sec)):
                        found = True
                        break
                if not found:
                    ok = False
                    break
            is_aligned = ok
        else:
            aligned_dt = last_valid_ts.get("1min")
            is_aligned = False

        out["aligned_timestamp"] = aligned_dt.isoformat() if aligned_dt else None
        out["is_aligned"] = bool(is_aligned)
        out["alignment_tolerance_sec"] = float(align_tol_sec)

        # Data freshness gate (existing safeguard)
        one_min_stale = bool(out.get("1min_is_stale") is True)
        if skip_freshness_check:
            out["data_freshness"] = "OFFLINE"
        else:
            out["data_freshness"] = "STALE" if one_min_stale else "FRESH"

        if one_min_stale and not skip_freshness_check:
            _nullify_tf_features(out, roc_key)
            return out

        # Session suppression (existing safeguard)
        if suppress:
            out["note"] = "suppressed_due_to_session_phase"
            _nullify_tf_features(out, roc_key)
            return out

        # Soft event risk mode (keep features; just flag)
        # (event_risk already set in out)

        # Indicators per timeframe (existing stable logic)
        for tf in ("1min", "3min", "5min", "15min"):
            prefix = f"{tf}_"
            closes = per_tf_data[tf]["closes"]
            highs = per_tf_data[tf]["highs"]
            lows = per_tf_data[tf]["lows"]
            volumes = per_tf_data[tf]["volumes"]
            n = len(closes)

            gap_reset = bool(per_tf_data[tf].get("gap_reset", False))

            # Illiquid skip (existing safeguard, but ratio is instrument-aware now)
            illiquid_skip = False
            if len(volumes) >= avg_vol_period:
                avg_vol = sum(volumes[-avg_vol_period:]) / float(avg_vol_period)
                curr_vol = volumes[-1] if volumes else 0.0
                if avg_vol > 0 and curr_vol < avg_vol * illiquid_ratio:
                    illiquid_skip = True
            out[prefix + "illiquid_skip"] = bool(illiquid_skip)

            if illiquid_skip or gap_reset:
                out[prefix + "is_valid"] = False
                out[prefix + "insufficient_data"] = True

                out[prefix + "momentum_trend_conflict"] = None
                out[prefix + "trend_aligned_rsi"] = None

                out[prefix + "rsi_raw"] = None
                out[prefix + "rsi"] = None
                out[prefix + "rsi_extreme"] = None

                out[prefix + "macd_line"] = None
                out[prefix + "macd_signal"] = None
                out[prefix + "macd_histogram"] = None
                out[prefix + "macd_hist_slope"] = None

                out[prefix + "stoch_k"] = None
                out[prefix + "stoch_d"] = None

                out[prefix + "price_acceleration_raw"] = None
                out[prefix + "price_acceleration_norm"] = None

                out[prefix + f"{roc_key}_raw"] = None
                out[prefix + f"{roc_key}_smooth"] = None
                continue

            need_rsi = rsi_p + 1
            need_macd = macd_slow + macd_signal
            need_stoch = stoch_k + stoch_d
            need_accel = 4
            need_roc = roc_p + 1

            is_valid = n >= max(need_rsi, need_macd, need_stoch, need_accel, need_roc)
            out[prefix + "is_valid"] = bool(is_valid)
            out[prefix + "insufficient_data"] = bool(not is_valid)

            # RSI raw/clamped/extreme
            rsi_raw = None
            rsi_clamped = None
            rsi_extreme = None
            if n >= need_rsi:
                rsi_series = _rsi_wilder_series(closes, rsi_p)
                rsi_raw = rsi_series[-1]
                if rsi_raw is not None and math.isfinite(float(rsi_raw)):
                    rsi_clamped = float(_clamp(float(rsi_raw), 0.1, 99.9))
                raw_vals = [v for v in rsi_series if v is not None and math.isfinite(float(v))]
                # extreme persistence: use last 10 RSI values (or configured)
                extreme_bars = _cfg_int("features", "rsi_extreme_bars", default=10)
                if len(raw_vals) >= extreme_bars:
                    last_vals = raw_vals[-extreme_bars:]
                    rsi_extreme = all(float(v) > float(rsi_extreme_level) for v in last_vals)
                else:
                    rsi_extreme = False

            out[prefix + "rsi_raw"] = float(rsi_raw) if rsi_raw is not None else None
            out[prefix + "rsi"] = float(rsi_clamped) if rsi_clamped is not None else None
            out[prefix + "rsi_extreme"] = bool(rsi_extreme) if rsi_extreme is not None else None

            # MACD + hist slope smoothing
            macd_line_last = macd_sig_last = macd_hist_last = macd_hist_slope = None
            if n >= need_macd:
                macd_line, macd_sig, macd_hist = _macd_series(closes, macd_fast, macd_slow, macd_signal)
                macd_line_last = macd_line[-1]
                macd_sig_last = macd_sig[-1]
                macd_hist_last = macd_hist[-1]
                hist_sma3 = _sma_optional(macd_hist, 3)
                hist_vals = [float(v) for v in hist_sma3 if v is not None and math.isfinite(float(v))]
                if len(hist_vals) >= 5:
                    macd_hist_slope = _linreg_slope(hist_vals[-5:])

            out[prefix + "macd_line"] = float(macd_line_last) if macd_line_last is not None else None
            out[prefix + "macd_signal"] = float(macd_sig_last) if macd_sig_last is not None else None
            out[prefix + "macd_histogram"] = float(macd_hist_last) if macd_hist_last is not None else None
            out[prefix + "macd_hist_slope"] = float(macd_hist_slope) if macd_hist_slope is not None else None

            # Stochastic
            stoch_k_last = stoch_d_last = None
            if n >= need_stoch and len(highs) == n and len(lows) == n:
                stoch_k_last, stoch_d_last = _stochastic_last(highs, lows, closes, stoch_k, stoch_d)
            out[prefix + "stoch_k"] = float(stoch_k_last) if stoch_k_last is not None else None
            out[prefix + "stoch_d"] = float(stoch_d_last) if stoch_d_last is not None else None

            # Acceleration raw/norm (ATR from context if available, else stddev)
            accel_raw = accel_norm = None
            if n >= need_accel:
                c0, c1, c2, c3 = closes[-1], closes[-2], closes[-3], closes[-4]
                accel_raw = (c0 - c2) - (c1 - c3)
                if accel_raw is not None and math.isfinite(float(accel_raw)):
                    atr_proxy = None
                    for k in (f"{tf}_atr", f"{tf}_atr_14", "atr", "atr_14"):
                        vv = _safe_float(volatility_context.get(k))
                        if vv is not None and vv > 0:
                            atr_proxy = vv
                            break
                    if atr_proxy is None and n >= 20:
                        sd = _stddev(closes[-20:])
                        if sd is not None and sd > 0:
                            atr_proxy = sd
                    if atr_proxy is not None and atr_proxy > 0:
                        accel_norm = float(accel_raw) / float(atr_proxy)

            out[prefix + "price_acceleration_raw"] = float(accel_raw) if accel_raw is not None else None
            out[prefix + "price_acceleration_norm"] = float(accel_norm) if accel_norm is not None else None

            # ROC raw + EMA(3)
            roc_raw = roc_smooth = None
            roc_series: List[Optional[float]] = [None] * n
            if roc_p > 0 and n >= need_roc:
                for i in range(roc_p, n):
                    base = closes[i - roc_p]
                    if base == 0.0:
                        roc_series[i] = None
                    else:
                        r = (closes[i] - base) / base * 100.0
                        roc_series[i] = float(r) if math.isfinite(float(r)) else None

                roc_raw = roc_series[-1]
                tail: List[float] = []
                for v in reversed(roc_series):
                    if v is None:
                        break
                    tail.append(float(v))
                tail = list(reversed(tail))
                if len(tail) >= 3:
                    ema3 = _ema_series(tail, 3)
                    roc_smooth = ema3[-1]

            out[prefix + f"{roc_key}_raw"] = float(roc_raw) if roc_raw is not None else None
            out[prefix + f"{roc_key}_smooth"] = float(roc_smooth) if roc_smooth is not None else None

            # ---------------- MTF Trend Integration (FILTER) ----------------
            mom_dir = 0
            if rsi_clamped is not None and macd_hist_last is not None and math.isfinite(float(macd_hist_last)):
                if rsi_clamped > 55.0 and float(macd_hist_last) > 0.0:
                    mom_dir = +1
                elif rsi_clamped < 45.0 and float(macd_hist_last) < 0.0:
                    mom_dir = -1

            conflict = False
            if mom_dir != 0 and mtf_alignment_signal is not None and int(mtf_alignment_signal) != 0:
                if mom_dir != int(mtf_alignment_signal):
                    conflict = True

            out[prefix + "momentum_trend_conflict"] = bool(conflict)
            out[prefix + "trend_aligned_rsi"] = float(rsi_clamped) if (rsi_clamped is not None and not conflict) else None

        return out

    except Exception as e:
        _emit_log("exception", "compute_momentum_features failed", error=str(e))
        return {}


# ------------------------- Self-test -------------------------

def _make_mock_candles(
    start_ts: datetime,
    n: int,
    start_price: float,
    drift_per_bar: float,
    vol: int,
) -> List[Dict[str, Any]]:
    candles: List[Dict[str, Any]] = []
    price = float(start_price)
    for i in range(n):
        ts = start_ts + timedelta(minutes=i)
        o = price
        h = o + abs(drift_per_bar) * 2.0 + 0.5
        l = o - abs(drift_per_bar) * 1.5 - 0.5
        c = o + drift_per_bar
        price = c
        candles.append(
            {
                "timestamp": ts,
                "open": float(o),
                "high": float(max(h, o, c)),
                "low": float(min(l, o, c)),
                "close": float(c),
                "volume": int(vol),
            }
        )
    return candles


def _aggregate_candles(candles_1m: List[Dict[str, Any]], block: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(0, len(candles_1m), block):
        chunk = candles_1m[i:i + block]
        if len(chunk) < block:
            break
        out.append(
            {
                "timestamp": chunk[0]["timestamp"],
                "open": chunk[0]["open"],
                "high": max(c["high"] for c in chunk),
                "low": min(c["low"] for c in chunk),
                "close": chunk[-1]["close"],
                "volume": sum(int(c["volume"]) for c in chunk),
            }
        )
    return out


if __name__ == "__main__":
    from collections import deque

    fixed_day = date(2026, 4, 9)
    start = datetime(fixed_day.year, fixed_day.month, fixed_day.day, 9, 15, 0, tzinfo=IST)

    candles_1m = _make_mock_candles(start, 200, 23000.0, 0.8, 1000)
    candles_3m = _aggregate_candles(candles_1m, 3)
    candles_5m = _aggregate_candles(candles_1m, 5)
    candles_15m = _aggregate_candles(candles_1m, 15)

    candles_input = {
        "1min": deque(candles_1m, maxlen=400),
        "3min": deque(candles_3m, maxlen=140),
        "5min": deque(candles_5m, maxlen=80),
        "15min": deque(candles_15m, maxlen=30),
    }

    # trend_context includes MTF alignment signal
    trend_context = {"mtf_alignment": 6.0, "mtf_alignment_signal": 1}

    session_context = {
        "session_phase": "GOLDEN_AM",
        "regime": "TRENDING",
        "trend_context": trend_context,
        "event_severity": 0,
        "event_minutes_away": None,
    }

    feats = compute_momentum_features(
        candles_input,
        instrument_spec={"symbol": "NIFTY", "instrument_class": "INDEX"},
        session_context=session_context,
        skip_freshness_check=True,
    )

    print("Self-test momentum summary (controlled final upgrade):")
    print("  instrument_class:", feats.get("instrument_class"))
    print("  event_override:", feats.get("event_override"), "event_risk:", feats.get("event_risk"))
    print("  data_freshness:", feats.get("data_freshness"))
    print("  mtf_alignment_signal:", feats.get("mtf_alignment_signal"))
    print("  1min_rsi:", feats.get("1min_rsi"), "1min_macd_histogram:", feats.get("1min_macd_histogram"))
    print("  1min_momentum_trend_conflict:", feats.get("1min_momentum_trend_conflict"))
    print("  1min_trend_aligned_rsi:", feats.get("1min_trend_aligned_rsi"))
    print("  15min_rsi:", feats.get("15min_rsi"), "15min_macd_line:", feats.get("15min_macd_line"))

    assert feats.get("data_freshness") == "OFFLINE", "Self-test should run in OFFLINE freshness mode"
    assert feats.get("event_override") is False, "Self-test should not trigger event override"
    assert feats.get("1min_rsi") is not None, "1min RSI should be computed"
    assert feats.get("1min_macd_line") is not None, "1min MACD line should be computed"
    assert feats.get("15min_rsi") is None, "15min RSI should be None due to insufficient data"
    assert feats.get("15min_macd_line") is None, "15min MACD should be None due to insufficient data"

    print("\nSelf-test PASSED.")