"""
Junior Aladdin - Volatility Features (Layer 2C)
===============================================
PURPOSE:
    Compute volatility-related features used by:
    - Risk Engine (SL sizing, position sizing)
    - Regime Engine (VOLATILE / CHOP detection)
    - Strategy Engine (breakout validity, squeeze conditions)

MANDATORY FIXES IMPLEMENTED:
1) compute_atr_series return type + alignment:
   - Returns list aligned to candles length, entries are float or None.
   - First ATR value placed at index `period` (0-based), using TRs indices 1..period.
   - Skips malformed candles instead of aborting entire series.
2) compute_atr_percentile empty input crash:
   - if not seq: return None before accessing seq[0].
3) compute_volatility_features BB None handling:
   - Uses `if bb is not None:` before out.update(bb).
4) Robust OHLC extraction in loops:
   - compute_atr_series, compute_candle_range_avg, compute_intraday_range_pct skip malformed candles.
5) Intraday date detection replay-safe:
   - Uses last candle's timestamp date (not ist_today()).
   - If no timestamps, uses all candles.

RESTRICTIONS:
- Public function signatures unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("volatility_features")


# -----------------------------
# Internal safety helpers
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _get_ohlc(candle: Dict[str, Any]) -> Optional[tuple]:
    """
    Returns (open, high, low, close) as floats, or None if missing/invalid.
    """
    o = _safe_float(candle.get("open"))
    h = _safe_float(candle.get("high"))
    l = _safe_float(candle.get("low"))
    c = _safe_float(candle.get("close"))
    if o is None or h is None or l is None or c is None:
        return None
    return o, h, l, c


def _as_list(candles: Union[List[Dict[str, Any]], deque, Sequence]) -> List:
    try:
        return list(candles) if candles is not None else []
    except Exception:
        return []


# -----------------------------
# Public helpers (signatures unchanged)
# -----------------------------
def compute_atr(candles, period: int = 14) -> Optional[float]:
    """
    Compute last available ATR (Wilder) from aligned ATR series.
    Returns None if ATR not available.
    """
    atr_aligned = compute_atr_series(candles, period=period)
    if not atr_aligned:
        return None
    last = atr_aligned[-1]
    return float(last) if isinstance(last, (int, float, np.floating)) else None


def compute_atr_series(candles, period: int = 14) -> List:
    """
    Compute ATR series aligned to candle indices.

    Returns:
        List of length == len(candles), entries are float or None.
        First ATR value is at index `period` (0-based), using TR indices 1..period.

    Robustness:
        - Works with list/deque input (converted to list).
        - Skips malformed candles: sets TR/ATR to None for those indices.
        - After a malformed stretch, ATR can resume once a full clean TR window exists.
    """
    candle_list: List[Dict[str, Any]] = _as_list(candles)

    n = len(candle_list)
    if period <= 0 or n == 0:
        return [None] * n

    # Need at least period+1 candles to compute TRs 1..period (requires prev close)
    if n < period + 1:
        return [None] * n

    tr: List[Optional[float]] = [None] * n
    prev_close: Optional[float] = None

    for i, candle in enumerate(candle_list):
        if not isinstance(candle, dict):
            # malformed container
            tr[i] = None
            continue

        ohlc = _get_ohlc(candle)
        if ohlc is None:
            tr[i] = None
            # do not update prev_close on malformed candle
            continue

        _o, high, low, close = ohlc
        if high < low:
            # data error; skip this candle
            tr[i] = None
            prev_close = close  # still update prev_close to allow subsequent TR calculations
            continue

        if prev_close is None:
            # TR for first candle: high-low (not used in first ATR window but kept aligned)
            tr_val = high - low
        else:
            tr_val = max(high - low, abs(high - prev_close), abs(low - prev_close))

        if tr_val < 0:
            tr[i] = None
        else:
            tr[i] = float(tr_val)

        prev_close = close

    atr: List[Optional[float]] = [None] * n

    # First ATR at index = period uses TR indices 1..period inclusive
    first_window = tr[1 : period + 1]
    if all(v is not None for v in first_window):
        atr[period] = float(np.mean(np.array(first_window, dtype=float)))
    else:
        atr[period] = None

    # Wilder smoothing forward; if missing data breaks chain, re-seed with SMA when possible
    for i in range(period + 1, n):
        tr_i = tr[i]
        prev_atr = atr[i - 1]

        if tr_i is not None and prev_atr is not None:
            atr[i] = float((prev_atr * (period - 1) + tr_i) / period)
            continue

        # If we can't Wilder-smooth, attempt reseed from last clean window (strict, no partial windows)
        window = tr[i - period + 1 : i + 1]
        if len(window) == period and all(v is not None for v in window):
            atr[i] = float(np.mean(np.array(window, dtype=float)))
        else:
            atr[i] = None

    return atr


def compute_atr_percentile(candles, period: int = 14, lookback: int = 100) -> Optional[float]:
    """
    Compute ATR percentile (0-100).

    Supports two call modes (signature unchanged):
    - candles is a candle list/deque -> compute aligned ATR series internally
    - candles is an ATR series (list of floats / None) -> compute percentile directly

    Rules:
    - If latest ATR is None -> returns None (no bogus percentile).
    - Uses last `lookback` non-None ATR values for distribution.
    """
    if lookback <= 1:
        return None

    try:
        seq = list(candles) if candles is not None else []
    except Exception:
        return None

    # MANDATE: protect empty input
    if not seq:
        return None

    # Detect "ATR series input" vs "candles input"
    # We find first non-None element and inspect its type.
    first_non_none = None
    for x in seq:
        if x is not None:
            first_non_none = x
            break

    if first_non_none is None:
        return None

    if isinstance(first_non_none, (int, float, np.floating)):
        # Treat as ATR series (may contain None)
        atr_series_aligned: List[Optional[float]] = []
        for x in seq:
            if x is None:
                atr_series_aligned.append(None)
            else:
                v = _safe_float(x)
                atr_series_aligned.append(v)
    else:
        # Treat as candles
        atr_series_aligned = compute_atr_series(seq, period=period)

    if not atr_series_aligned:
        return None

    latest = atr_series_aligned[-1]
    if latest is None:
        return None

    # Distribution from recent non-None values
    recent_vals = [v for v in atr_series_aligned if v is not None]
    if not recent_vals:
        return None

    recent = recent_vals[-lookback:] if lookback > 0 else recent_vals
    if not recent:
        return None

    current = float(latest)
    arr = np.array(recent, dtype=float)
    pct = float(np.sum(arr <= current) / max(len(arr), 1) * 100.0)
    return pct


def compute_candle_range_avg(candles, period: int = 20) -> Optional[float]:
    """
    Average candle range (high-low) over last `period` candles.

    MANDATE:
        - Do NOT silently reduce period. If insufficient candles, return None.
        - Skip malformed candles instead of failing entire computation.
    """
    candle_list: List[Dict[str, Any]] = _as_list(candles)
    if period <= 0:
        return None
    if len(candle_list) < period:
        return None

    ranges: List[float] = []
    for candle in candle_list[-period:]:
        if not isinstance(candle, dict):
            continue
        ohlc = _get_ohlc(candle)
        if ohlc is None:
            continue
        _o, h, l, _c = ohlc
        r = h - l
        if r < 0:
            # skip candle with invalid range
            continue
        ranges.append(float(r))

    return float(np.mean(ranges)) if ranges else None


def compute_bollinger_bands(closes, period: int = 20, num_std: float = 2.0) -> Optional[Dict[str, float]]:
    """
    Compute Bollinger Bands for the latest close.

    Returns dict:
      bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b

    Returns None on failure.
    """
    try:
        close_list = [float(x) for x in list(closes)]
    except Exception:
        return None

    if period <= 1 or len(close_list) < period:
        return None

    window = np.array(close_list[-period:], dtype=float)
    middle = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    upper = middle + float(num_std) * std
    lower = middle - float(num_std) * std

    width = upper - lower
    last_close = float(close_list[-1])
    denom = (upper - lower) if (upper - lower) != 0 else 1e-12
    pct_b = float((last_close - lower) / denom)

    return {
        "bb_upper": float(upper),
        "bb_middle": float(middle),
        "bb_lower": float(lower),
        "bb_width": float(width),
        "bb_pct_b": float(pct_b),
    }


def compute_bb_width_percentile(
    closes,
    period: int = 20,
    num_std: float = 2.0,
    lookback: int = 100,
) -> Optional[float]:
    """
    Compute Bollinger width percentile (0-100) based on historical widths.

    Notes:
      - Computes all possible widths, then percentiles on widths[-lookback:].
    """
    try:
        close_list = [float(x) for x in list(closes)]
    except Exception:
        return None

    if period <= 1 or len(close_list) < period:
        return None

    widths: List[float] = []
    for i in range(period, len(close_list) + 1):
        window = np.array(close_list[i - period : i], dtype=float)
        middle = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        upper = middle + float(num_std) * std
        lower = middle - float(num_std) * std
        widths.append(float(upper - lower))

    if not widths:
        return None

    recent = widths[-lookback:] if lookback and lookback > 0 else widths
    current = recent[-1]
    arr = np.array(recent, dtype=float)
    pct = float(np.sum(arr <= current) / max(len(arr), 1) * 100.0)
    return pct


def compute_intraday_range_pct(candles) -> Optional[float]:
    """
    Intraday range % = (day_high - day_low) / day_open * 100

    MANDATE:
        - Replay-safe: determine trading date from last candle timestamp (not ist_today()).
        - Skip malformed candles; do not fail entire computation.
        - If no timestamps, use all candles.
    """
    candle_list: List[Dict[str, Any]] = _as_list(candles)
    if not candle_list:
        return None

    # Determine target day from last candle timestamp
    target_day: Optional[date] = None
    for c in reversed(candle_list):
        if isinstance(c, dict) and isinstance(c.get("timestamp"), datetime):
            target_day = c["timestamp"].date()
            break

    filtered: List[Dict[str, Any]] = []
    if target_day is not None:
        for c in candle_list:
            if not isinstance(c, dict):
                continue
            ts = c.get("timestamp")
            if isinstance(ts, datetime) and ts.date() == target_day:
                filtered.append(c)
        # If filtering yields nothing (bad data), fall back to all
        if not filtered:
            filtered = [c for c in candle_list if isinstance(c, dict)]
    else:
        filtered = [c for c in candle_list if isinstance(c, dict)]

    if not filtered:
        return None

    # Day open from first valid candle open
    day_open: Optional[float] = None
    for c in filtered:
        ohlc = _get_ohlc(c)
        if ohlc is None:
            continue
        day_open = float(ohlc[0])
        break

    if day_open is None or day_open <= 0:
        return None

    day_high: Optional[float] = None
    day_low: Optional[float] = None

    for c in filtered:
        ohlc = _get_ohlc(c)
        if ohlc is None:
            continue
        _o, h, l, _cl = ohlc
        if h < l:
            continue
        day_high = h if day_high is None else max(day_high, h)
        day_low = l if day_low is None else min(day_low, l)

    if day_high is None or day_low is None:
        return None
    if day_high < day_low:
        return None

    return float((day_high - day_low) / day_open * 100.0)


def _empty_features() -> Dict[str, Any]:
    return {
        "atr": None,
        "atr_percentile": None,
        "bb_upper": None,
        "bb_middle": None,
        "bb_lower": None,
        "bb_width": None,
        "bb_width_percentile": None,
        "bb_pct_b": None,
        "candle_range_avg": None,
        "candle_body_ratio": None,
        "intraday_range_pct": None,
    }


def compute_volatility_features(
    candles,
    atr_period: int = 14,
    atr_percentile_lookback: int = 100,
    bb_period: int = 20,
    bb_std: float = 2.0,
    bb_width_lookback: int = 100,
    candle_range_period: int = 20,
) -> Dict[str, Any]:
    """
    Master function for volatility features.

    PERFORMANCE:
        Computes aligned ATR series once and reuses for ATR + percentile.

    Returns:
        dict of features (values may be None if insufficient data)
    """
    candle_list: List[Dict[str, Any]] = _as_list(candles)
    if not candle_list:
        return _empty_features()

    # Extract closes safely; skip malformed candles for closes collection
    closes: List[float] = []
    for c in candle_list:
        if not isinstance(c, dict):
            continue
        cl = _safe_float(c.get("close"))
        if cl is None:
            continue
        closes.append(float(cl))

    out = _empty_features()

    # ATR series cached (aligned, may contain None)
    atr_aligned = compute_atr_series(candle_list, period=atr_period)
    if atr_aligned:
        last_atr = atr_aligned[-1]
        out["atr"] = float(last_atr) if last_atr is not None else None
        out["atr_percentile"] = compute_atr_percentile(
            atr_aligned, period=atr_period, lookback=atr_percentile_lookback
        )

    # Bollinger Bands
    bb = compute_bollinger_bands(closes, period=bb_period, num_std=bb_std)
    # MANDATE: handle None explicitly
    if bb is not None:
        out.update(bb)
        out["bb_width_percentile"] = compute_bb_width_percentile(
            closes, period=bb_period, num_std=bb_std, lookback=bb_width_lookback
        )

    # Candle range avg
    out["candle_range_avg"] = compute_candle_range_avg(candle_list, period=candle_range_period)

    # Candle body ratio (latest valid candle)
    last_ratio = None
    for c in reversed(candle_list):
        if not isinstance(c, dict):
            continue
        ohlc = _get_ohlc(c)
        if ohlc is None:
            continue
        o, h, l, cl = ohlc
        rng = h - l
        if rng < 0:
            continue
        last_ratio = float(abs(cl - o) / (rng + 0.01))
        break
    out["candle_body_ratio"] = last_ratio

    # Intraday range %
    out["intraday_range_pct"] = compute_intraday_range_pct(candle_list)

    return out


# ---------------------------------------
# Module self-test (basic safety checks)
# ---------------------------------------
if __name__ == "__main__":
    IST_LOCAL = timezone(timedelta(hours=5, minutes=30))

    def _mk_candle(ts, o, h, l, c, v=0):
        return {
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "is_complete": True,
        }

    print("=" * 60)
    print(" JUNIOR ALADDIN — Volatility Features Self-Test")
    print("=" * 60)

    now = datetime.now(IST_LOCAL).replace(second=0, microsecond=0)
    candles_deque = deque(maxlen=400)

    px = 24500.0
    for i in range(30):
        ts = now - timedelta(minutes=(30 - i))
        o = px
        h = px + 5
        l = px - 4
        c = px + (i % 3) - 1
        candles_deque.append(_mk_candle(ts, o, h, l, c, v=1000 + i))
        px = c

    feats = compute_volatility_features(candles_deque, atr_period=14, bb_period=20)
    assert feats["atr"] is not None, "ATR should compute with 30 candles"
    assert feats["bb_upper"] is not None, "BB should compute with 30 closes"
    assert feats["intraday_range_pct"] is not None, "intraday_range_pct should compute"

    # ATR alignment check: first ATR at index period
    atr_series = compute_atr_series(candles_deque, period=14)
    assert len(atr_series) == len(candles_deque), "ATR series must align to candles length"
    assert atr_series[14] is not None, "First ATR value must be at index=period"
    assert all(x is None for x in atr_series[:14]), "ATR before index=period must be None"

    # Empty input should not crash
    assert compute_atr_percentile([], period=14, lookback=100) is None, "Empty input must return None"

    # Malformed candle should be skipped, not kill series
    bad = list(candles_deque)
    bad[10] = {"timestamp": now}  # malformed
    atr_bad = compute_atr_series(bad, period=14)
    assert len(atr_bad) == len(bad), "ATR series must remain aligned even with malformed candle"

    # compute_candle_range_avg insufficient data returns None
    short = list(candles_deque)[:5]
    assert compute_candle_range_avg(short, period=20) is None, "Insufficient candles must return None"

    print(" ✅ All self-tests passed")