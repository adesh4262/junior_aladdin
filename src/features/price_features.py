"""
Junior Aladdin - Price & Trend Features (Layer 2A)

Computes institutional-grade Price & Trend features from multi-timeframe candles.

INPUTS
------
candles: Dict[str, deque|list]
    Keys: "1min","3min","5min","15min" (also accepts "1m","3m","5m","15m").
    Each candle dict should contain:
        - timestamp (datetime|epoch ms|epoch sec|ISO string)
        - open, high, low, close (float-like)
        - volume (int/float-like; optional)

instrument_spec: Dict (optional)
    token, symbol, instrument_class, tick_size, lot_size (optional)

session_context: Dict (optional)
    session_phase, day_type, is_expiry, regime, etc.

OUTPUTS (flat dict; JSON-serializable)
-------------------------------------
Global fields:
- symbol, instrument_class, tick_size
- session_phase (upper), ib_phase (bool), regime (upper)
- mtf_alignment (float|None)
- mtf_alignment_signal (int: +1/-1/0)

Per-timeframe fields for each tf in {1min,3min,5min,15min}:
- {tf}_raw_candle_count (int)                # unfiltered deque/list length
- {tf}_candle_count (int)                    # alias for raw (backward compatibility)
- {tf}_effective_candle_count (int)          # candles used for EMA/Supertrend (session-filtered & OHLC-valid)
- {tf}_last_close_raw (float|None)           # last close from raw candles (unfiltered)
- {tf}_last_close (float|None)               # last close from effective valid candles
- {tf}_ema_9, {tf}_ema_21, {tf}_ema_50 (float|None)
- {tf}_vwap (float|None)
- {tf}_vwap_upper_1, {tf}_vwap_lower_1 (float|None)
- {tf}_vwap_upper_2, {tf}_vwap_lower_2 (float|None)
- {tf}_vwap_slope (float|None)
- {tf}_supertrend (float|None)
- {tf}_supertrend_direction (int|None)       # +1/-1
- {tf}_trend_direction (int|None)            # +1/-1/0, or None when insufficient data
- {tf}_trend_direction_method (str|None)     # "standard", "volatile_longer_emas", "volatile_fallback_standard", "insufficient_data"
- {tf}_price_vs_vwap_pct (float|None)

Session-aware behavior:
- PRE_MARKET / OPENING_AUCTION: candle-dependent features set to None; mtf_alignment None.
- VWAP resets daily at market open (configured, default 09:15 IST) via session-filtered candles.

CONFIG KEYS USED (with defaults)
--------------------------------
market.market_open: "09:15"

features.ema_periods: [9, 21, 50]
features.supertrend_period: 10
features.supertrend_multiplier: 3.0
features.vwap_slope_window: 10
features.min_candles_trend_direction: 10

features.mtf_weight_1min: 1.0
features.mtf_weight_3min: 1.5
features.mtf_weight_5min: 2.0
features.mtf_weight_15min: 3.0

Robustness:
- compute_price_features() MUST NEVER raise; returns {} on catastrophic failure.
- _safe_float rejects NaN/inf/-inf.
- Rate-limited warnings (once per 60s per timeframe) for:
  - missing OHLC in effective candles
  - VWAP not computable due to zero cumulative volume (debug)
"""

from __future__ import annotations

from datetime import datetime, date, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.utils.helpers import IST

_LOG = setup_logger("price_features")


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


def _rate_limited(warn_state: Dict[str, Any], key: str, now: datetime, window_sec: float = 60.0) -> bool:
    """
    Return True if allowed to log now (rate-limited), else False.
    Stores last log time in warn_state[key].
    """
    try:
        last = warn_state.get(key)
        if last is None:
            warn_state[key] = now
            return True
        if isinstance(last, datetime):
            if (now - last).total_seconds() >= window_sec:
                warn_state[key] = now
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


def _parse_ts_to_ist(ts: Any) -> Optional[datetime]:
    """
    Candle timestamps may be:
    - datetime (naive/aware)
    - epoch ms/sec (int/float/str digits)
    - ISO string
    """
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
        return int(_safe_int(v, default=default))
    except Exception:
        return int(default)


def _cfg_float(section: str, key: str, default: float) -> float:
    try:
        v = Config.get(section, key, default=default)
        f = _safe_float(v)
        return float(default if f is None else f)
    except Exception:
        return float(default)


# ------------------------- Math primitives -------------------------

def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    """
    EMA series; returns list aligned with values.
    Indices < period-1 -> None.
    Seeded with SMA(period) at index period-1.
    """
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


def _true_ranges(highs: List[float], lows: List[float], closes: List[float]) -> List[Optional[float]]:
    n = len(closes)
    out: List[Optional[float]] = [None] * n
    if n == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        out[i] = tr
    return out


def _wilder_atr(trs: List[Optional[float]], period: int) -> List[Optional[float]]:
    n = len(trs)
    out: List[Optional[float]] = [None] * n
    if period <= 1 or n < period:
        return out
    seed_vals = [trs[i] for i in range(period) if trs[i] is not None]
    if len(seed_vals) != period:
        return out
    atr = sum(seed_vals) / period
    out[period - 1] = atr
    for i in range(period, n):
        tr = trs[i]
        if tr is None:
            out[i] = None
            continue
        atr = (atr * (period - 1) + tr) / period
        out[i] = atr
    return out


def _supertrend(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int,
    multiplier: float,
) -> Tuple[List[Optional[float]], List[Optional[int]]]:
    n = len(closes)
    st: List[Optional[float]] = [None] * n
    direction: List[Optional[int]] = [None] * n
    if n == 0:
        return st, direction
    if period <= 1 or n < period + 1:
        return st, direction

    trs = _true_ranges(highs, lows, closes)
    atr = _wilder_atr(trs, period)

    basic_ub: List[Optional[float]] = [None] * n
    basic_lb: List[Optional[float]] = [None] * n
    for i in range(n):
        if atr[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2.0
        basic_ub[i] = hl2 + multiplier * atr[i]
        basic_lb[i] = hl2 - multiplier * atr[i]

    final_ub: List[Optional[float]] = [None] * n
    final_lb: List[Optional[float]] = [None] * n

    start = period - 1
    if basic_ub[start] is None or basic_lb[start] is None:
        return st, direction

    final_ub[start] = basic_ub[start]
    final_lb[start] = basic_lb[start]

    direction[start] = +1 if closes[start] >= final_lb[start] else -1
    st[start] = final_lb[start] if direction[start] == +1 else final_ub[start]

    for i in range(start + 1, n):
        if basic_ub[i] is None or basic_lb[i] is None or final_ub[i - 1] is None or final_lb[i - 1] is None:
            continue

        if basic_ub[i] < final_ub[i - 1] or closes[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or closes[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        prev_dir = direction[i - 1] if direction[i - 1] is not None else +1
        if prev_dir == +1:
            direction[i] = -1 if closes[i] < final_lb[i] else +1
        else:
            direction[i] = +1 if closes[i] > final_ub[i] else -1

        st[i] = final_lb[i] if direction[i] == +1 else final_ub[i]

    return st, direction


# ------------------------- VWAP and session handling -------------------------

def _session_filter_candles(candles: List[Dict[str, Any]], market_open: time) -> List[Dict[str, Any]]:
    """
    For intraday VWAP reset:
    - Use last candle date as session date
    - Keep only candles on that date with time >= market_open
    """
    if not candles:
        return candles
    last_ts = _parse_ts_to_ist(candles[-1].get("timestamp"))
    if last_ts is None:
        return candles
    session_day = last_ts.date()
    out: List[Dict[str, Any]] = []
    for c in candles:
        ts = _parse_ts_to_ist(c.get("timestamp"))
        if ts is None:
            continue
        if ts.date() != session_day:
            continue
        if ts.timetz().replace(tzinfo=None) < market_open:
            continue
        out.append(c)
    return out


def _last_finite(values: List[Optional[float]]) -> Optional[float]:
    for v in reversed(values):
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                return fv
        except Exception:
            continue
    return None


def _compute_vwap_series(
    candles: List[Dict[str, Any]],
) -> Tuple[List[Optional[float]], List[Optional[float]], int, bool]:
    """
    VWAP and volume-weighted stddev series using typical price (H+L+C)/3.

    IMPORTANT (Institutional Fix):
    - If a candle has missing OHLC, this function sets vwap[i]=None and stdev[i]=None.
      It does NOT propagate previous values (no silent stalling).
    - Cumulative sums are not updated for missing OHLC candles.

    Returns:
        (vwap_series, stdev_series, missing_ohlc_count, ever_had_volume)
    """
    n = len(candles)
    vwap: List[Optional[float]] = [None] * n
    stdev: List[Optional[float]] = [None] * n
    if n == 0:
        return vwap, stdev, 0, False

    cum_v = 0.0
    cum_pv = 0.0
    cum_p2v = 0.0
    missing_ohlc = 0
    ever_had_volume = False

    for i, c in enumerate(candles):
        h = _safe_float(c.get("high"))
        l = _safe_float(c.get("low"))
        cl = _safe_float(c.get("close"))
        vol = _safe_float(c.get("volume"))

        if h is None or l is None or cl is None:
            missing_ohlc += 1
            vwap[i] = None
            stdev[i] = None
            continue

        if vol is None or vol < 0:
            vol = 0.0

        tp = (h + l + cl) / 3.0
        cum_v += vol
        cum_pv += tp * vol
        cum_p2v += (tp * tp) * vol

        if cum_v <= 0.0:
            vwap[i] = None
            stdev[i] = None
            continue

        ever_had_volume = True
        vw = cum_pv / cum_v
        vwap[i] = vw
        mean_sq = cum_p2v / cum_v
        var = max(0.0, mean_sq - (vw * vw))
        stdev[i] = math.sqrt(var)

    return vwap, stdev, missing_ohlc, ever_had_volume


# ------------------------- Main feature computation -------------------------

def compute_price_features(
    candles: Dict[str, Any],
    instrument_spec: Optional[Dict[str, Any]] = None,
    session_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute Price & Trend features across timeframes.
    MUST NEVER RAISE.
    """
    warn_state: Dict[str, Any] = {}
    try:
        if not isinstance(candles, dict):
            _emit_log("error", "compute_price_features: candles input is not a dict", candles_type=str(type(candles)))
            return {}

        if instrument_spec is None:
            instrument_spec = {}
        elif not isinstance(instrument_spec, dict):
            # recommended improvement: graceful handling + rate-limited warning
            now = datetime.now(tz=IST)
            if _rate_limited(warn_state, "instrument_spec_type", now, 60.0):
                _emit_log("warning", "instrument_spec is not a dict; treating as empty", instrument_spec_type=str(type(instrument_spec)))
            instrument_spec = {}

        session_context = session_context or {}
        session_phase = str(session_context.get("session_phase") or "").strip().upper()
        regime = str(session_context.get("regime") or session_context.get("market_regime") or "").strip().upper()

        ib_phase = session_phase == "INITIAL_BALANCE"
        suppress = session_phase in {"PRE_MARKET", "OPENING_AUCTION"}

        ema_periods = Config.get("features", "ema_periods", default=[9, 21, 50])
        if not isinstance(ema_periods, list) or not ema_periods:
            ema_periods = [9, 21, 50]
        cleaned: List[int] = []
        for p in ema_periods:
            try:
                pi = int(float(p))
                if pi > 0:
                    cleaned.append(pi)
            except Exception:
                continue
        ema_periods = sorted(list(set(cleaned))) if cleaned else [9, 21, 50]

        st_period = _cfg_int("features", "supertrend_period", default=10)
        st_mult = _cfg_float("features", "supertrend_multiplier", default=3.0)
        vwap_slope_window = _cfg_int("features", "vwap_slope_window", default=10)
        min_candles_trend = _cfg_int("features", "min_candles_trend_direction", default=10)

        mopen_str = str(Config.get("market", "market_open", default="09:15"))
        try:
            hh, mm = mopen_str.split(":")
            market_open = time(int(hh), int(mm))
        except Exception:
            market_open = time(9, 15)

        symbol = str(instrument_spec.get("symbol") or instrument_spec.get("name") or "")
        instrument_class = str(instrument_spec.get("instrument_class") or instrument_spec.get("type") or "UNKNOWN").strip().upper()
        tick_size = _safe_float(instrument_spec.get("tick_size"))

        out: Dict[str, Any] = {
            "symbol": symbol or None,
            "instrument_class": instrument_class or None,
            "tick_size": tick_size,
            "session_phase": session_phase or None,
            "ib_phase": bool(ib_phase),
            "regime": regime or None,
        }

        # Normalize candles dict
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

        for tf, raw_list in tf_map.items():
            out[f"{tf}_raw_candle_count"] = int(len(raw_list))
            out[f"{tf}_candle_count"] = int(len(raw_list))  # backward compatible alias

        if suppress:
            for tf in ("1min", "3min", "5min", "15min"):
                prefix = f"{tf}_"
                out[prefix + "effective_candle_count"] = 0
                out[prefix + "last_close_raw"] = None
                out[prefix + "last_close"] = None
                for key in (
                    "ema_9", "ema_21", "ema_50",
                    "vwap", "vwap_upper_1", "vwap_lower_1", "vwap_upper_2", "vwap_lower_2",
                    "vwap_slope",
                    "supertrend", "supertrend_direction",
                    "trend_direction", "trend_direction_method",
                    "price_vs_vwap_pct",
                ):
                    out[prefix + key] = None
            out["mtf_alignment"] = None
            out["mtf_alignment_signal"] = 0
            out["note"] = "suppressed_due_to_session_phase"
            return out

        tf_trend_dir: Dict[str, Optional[int]] = {}

        for tf in ("1min", "3min", "5min", "15min"):
            raw_list = tf_map.get(tf, [])
            prefix = f"{tf}_"

            last_close_raw = _safe_float(raw_list[-1].get("close")) if raw_list else None
            out[prefix + "last_close_raw"] = float(last_close_raw) if last_close_raw is not None else None

            # session-filtered candles for intraday computations
            effective = _session_filter_candles(raw_list, market_open=market_open)

            # build valid OHLC arrays for EMA/Supertrend; count missing OHLC in effective
            closes: List[float] = []
            highs: List[float] = []
            lows: List[float] = []
            volumes: List[float] = []
            missing_ohlc_count = 0

            for c in effective:
                o = _safe_float(c.get("open"))
                h = _safe_float(c.get("high"))
                l = _safe_float(c.get("low"))
                cl = _safe_float(c.get("close"))
                if o is None or h is None or l is None or cl is None:
                    missing_ohlc_count += 1
                    continue
                v = _safe_float(c.get("volume"))
                closes.append(cl)
                highs.append(h)
                lows.append(l)
                volumes.append(0.0 if v is None or v < 0 else float(v))

            n = len(closes)
            out[prefix + "effective_candle_count"] = int(n)
            out[prefix + "last_close"] = float(closes[-1]) if n > 0 else None

            if missing_ohlc_count > 0:
                now = datetime.now(tz=IST)
                if _rate_limited(warn_state, f"missing_ohlc_{tf}", now, 60.0):
                    _emit_log(
                        "warning",
                        "Missing OHLC fields detected in effective candles; features may be degraded",
                        timeframe=tf,
                        missing_ohlc_count=missing_ohlc_count,
                        raw_candles=len(raw_list),
                        session_filtered_candles=len(effective),
                        effective_valid_candles=n,
                    )

            if n == 0:
                for key in (
                    "ema_9", "ema_21", "ema_50",
                    "vwap", "vwap_upper_1", "vwap_lower_1", "vwap_upper_2", "vwap_lower_2",
                    "vwap_slope",
                    "supertrend", "supertrend_direction",
                    "trend_direction", "trend_direction_method",
                    "price_vs_vwap_pct",
                ):
                    out[prefix + key] = None
                tf_trend_dir[tf] = None
                continue

            # VWAP uses session-filtered candles (NOT only valid list), but will output None where OHLC missing.
            vwap_series, vwap_std, vwap_missing, ever_had_volume = _compute_vwap_series(effective)
            vwap_last = _last_finite(vwap_series)
            std_last = _last_finite(vwap_std)

            # if VWAP never computed due to zero volume, debug (rate-limited)
            if not ever_had_volume:
                now = datetime.now(tz=IST)
                if _rate_limited(warn_state, f"vwap_zero_volume_{tf}", now, 60.0):
                    _emit_log(
                        "debug",
                        "VWAP not computable (zero cumulative volume) for timeframe",
                        timeframe=tf,
                        raw_candles=len(raw_list),
                        session_filtered_candles=len(effective),
                    )

            out[prefix + "vwap"] = float(vwap_last) if vwap_last is not None else None
            if vwap_last is not None and std_last is not None:
                out[prefix + "vwap_upper_1"] = float(vwap_last + std_last)
                out[prefix + "vwap_lower_1"] = float(vwap_last - std_last)
                out[prefix + "vwap_upper_2"] = float(vwap_last + 2.0 * std_last)
                out[prefix + "vwap_lower_2"] = float(vwap_last - 2.0 * std_last)
            else:
                out[prefix + "vwap_upper_1"] = None
                out[prefix + "vwap_lower_1"] = None
                out[prefix + "vwap_upper_2"] = None
                out[prefix + "vwap_lower_2"] = None

            vwap_vals = [float(v) for v in vwap_series if v is not None and math.isfinite(float(v))]
            if len(vwap_vals) >= 2:
                window = min(vwap_slope_window, len(vwap_vals))
                slope = _linreg_slope(vwap_vals[-window:])
                out[prefix + "vwap_slope"] = float(slope) if slope is not None else None
            else:
                out[prefix + "vwap_slope"] = None

            # EMAs
            ema_needed = sorted(set(ema_periods + [9, 21, 50, 100]))
            ema_last: Dict[int, Optional[float]] = {}
            for p in ema_needed:
                series = _ema_series(closes, p)
                ema_last[p] = series[-1] if series else None

            out[prefix + "ema_9"] = float(ema_last.get(9)) if ema_last.get(9) is not None else None
            out[prefix + "ema_21"] = float(ema_last.get(21)) if ema_last.get(21) is not None else None
            out[prefix + "ema_50"] = float(ema_last.get(50)) if ema_last.get(50) is not None else None

            # Supertrend
            st_line, st_dir = _supertrend(highs, lows, closes, period=st_period, multiplier=st_mult)
            st_last = st_line[-1] if st_line else None
            st_dir_last = st_dir[-1] if st_dir else None
            out[prefix + "supertrend"] = float(st_last) if st_last is not None else None
            out[prefix + "supertrend_direction"] = int(st_dir_last) if st_dir_last is not None else None

            # trend_direction: None when insufficient candles (Institutional Fix)
            td: Optional[int] = 0
            td_method: Optional[str] = None

            if n < min_candles_trend:
                td = None
                td_method = "insufficient_data"
            else:
                e9 = ema_last.get(9)
                e21 = ema_last.get(21)
                e50 = ema_last.get(50)

                if regime == "VOLATILE":
                    td_method = "volatile_longer_emas"
                    e100 = ema_last.get(100)
                    if e21 is not None and e50 is not None and e100 is not None:
                        if e21 > e50 > e100:
                            td = +1
                        elif e21 < e50 < e100:
                            td = -1
                        else:
                            td = 0
                    else:
                        td_method = "volatile_fallback_standard"
                        if e9 is not None and e21 is not None and e50 is not None:
                            if e9 > e21 > e50:
                                td = +1
                            elif e9 < e21 < e50:
                                td = -1
                            else:
                                td = 0
                        else:
                            td = 0
                else:
                    td_method = "standard"
                    if e9 is not None and e21 is not None and e50 is not None:
                        if e9 > e21 > e50:
                            td = +1
                        elif e9 < e21 < e50:
                            td = -1
                        else:
                            td = 0
                    else:
                        td = 0

            out[prefix + "trend_direction"] = int(td) if td is not None else None
            out[prefix + "trend_direction_method"] = td_method
            tf_trend_dir[tf] = td  # keep None to skip MTF alignment

            # price_vs_vwap_pct
            if vwap_last is not None and vwap_last != 0.0 and math.isfinite(float(vwap_last)):
                out[prefix + "price_vs_vwap_pct"] = float((closes[-1] - vwap_last) / vwap_last * 100.0)
            else:
                out[prefix + "price_vs_vwap_pct"] = None

        # MTF alignment (weighted); skip tf where td is None (Institutional Fix)
        w1 = _cfg_float("features", "mtf_weight_1min", default=1.0)
        w3 = _cfg_float("features", "mtf_weight_3min", default=1.5)
        w5 = _cfg_float("features", "mtf_weight_5min", default=2.0)
        w15 = _cfg_float("features", "mtf_weight_15min", default=3.0)
        weights = {"1min": w1, "3min": w3, "5min": w5, "15min": w15}

        mtf_alignment = 0.0
        used = 0
        for tf in ("1min", "3min", "5min", "15min"):
            td = tf_trend_dir.get(tf)
            if td is None:
                continue
            mtf_alignment += float(td) * float(weights.get(tf, 0.0))
            used += 1

        out["mtf_alignment"] = float(mtf_alignment) if used > 0 else None
        if used == 0 or mtf_alignment == 0.0:
            out["mtf_alignment_signal"] = 0
        else:
            out["mtf_alignment_signal"] = 1 if mtf_alignment > 0 else -1

        return out

    except Exception as e:
        _emit_log("exception", "compute_price_features failed", error=str(e))
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

    # Fixed past trading day for deterministic testing (MANDATORY FIX)
    fixed_day = date(2026, 4, 9)  # Thursday
    start = datetime(fixed_day.year, fixed_day.month, fixed_day.day, 9, 15, 0, tzinfo=IST)

    candles_1m = _make_mock_candles(
        start_ts=start,
        n=200,
        start_price=23000.0,
        drift_per_bar=0.8,
        vol=1000,
    )
    candles_3m = _aggregate_candles(candles_1m, 3)
    candles_5m = _aggregate_candles(candles_1m, 5)
    candles_15m = _aggregate_candles(candles_1m, 15)

    candles_input = {
        "1min": deque(candles_1m, maxlen=400),
        "3min": deque(candles_3m, maxlen=140),
        "5min": deque(candles_5m, maxlen=80),
        "15min": deque(candles_15m, maxlen=30),
    }

    instrument_spec = {"token": "99926000", "symbol": "NIFTY", "instrument_class": "INDEX", "tick_size": "0.05"}
    session_context = {"session_phase": "GOLDEN_AM", "day_type": "TREND_DAY", "is_expiry": False, "regime": "TRENDING"}

    feats = compute_price_features(candles_input, instrument_spec=instrument_spec, session_context=session_context)

    keys_show = [
        "1min_raw_candle_count", "1min_effective_candle_count", "1min_last_close_raw", "1min_last_close",
        "1min_vwap", "1min_vwap_slope", "1min_ema_9", "1min_ema_21", "1min_ema_50",
        "1min_trend_direction", "1min_trend_direction_method", "1min_price_vs_vwap_pct",
        "15min_ema_21", "15min_ema_50", "15min_trend_direction", "15min_trend_direction_method",
        "mtf_alignment", "mtf_alignment_signal",
    ]
    print("Self-test feature summary:")
    for k in keys_show:
        print(f"  {k}: {feats.get(k)}")

    assert feats.get("1min_vwap") is not None, "VWAP should be computed for 1min"
    assert feats.get("1min_ema_9") is not None, "EMA9 should be computed for 1min"
    assert feats.get("1min_ema_21") is not None, "EMA21 should be computed for 1min"
    assert feats.get("1min_ema_50") is not None, "EMA50 should be computed for 1min"
    assert feats.get("mtf_alignment") is not None, "MTF alignment should be computed"

    print("\nSelf-test PASSED.")