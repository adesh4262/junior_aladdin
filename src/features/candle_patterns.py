# src/features/candle_patterns.py
"""
Junior Aladdin - Candle Patterns (Layer 2I) — Institutional Grade
=================================================================
PURPOSE:
    Detect the 8 high-significance candle patterns from OHLCV candle data.

SUPPORTED PATTERNS (Blueprint 8):
1. HAMMER
2. SHOOTING_STAR
3. BULLISH_ENGULFING
4. BEARISH_ENGULFING
5. DOJI
6. MORNING_STAR
7. EVENING_STAR
8. INSIDE_BAR

BASE CONFIDENCE (Blueprint):
- Hammer / Shooting Star: 30
- Bullish / Bearish Engulfing: 35
- Doji / Inside Bar: 25
- Morning Star / Evening Star: 40
Volume confirmation adds +20 (ONLY if volume_sma is provided).

MANDATES IMPLEMENTED:
- Blueprint-aligned base scores and pattern rules (no hidden extra constraints).
- Volume context enhancement:
    - volume_ratio (vol / volume_sma)
    - optional volume_percentile (if volume_history provided)
    - volume_quality score (0-100) if volume_sma provided, else None
    - volume bonus is NOT applied if volume_sma is missing.
- Multi-timeframe confirmation (optional higher_tf_candles):
    - determines simple HTF trend and sets mtf_confirmed, confidence +10 if aligned.
- Context-aware filtering (optional regime/session_phase):
    - can suppress patterns in adverse context, sets context_filtered + reason.
- Reliability tracking (module-level rolling outcomes):
    - tracks last 50 occurrences per pattern
    - computes win_rate_20
    - if win_rate_20 < 40% => confidence reduced by 30%
    - if win_rate_20 < 25% => pattern disabled (not emitted)
  NOTE: outcomes require evaluation after future bars; supported via optional future_candles
        (backtest/replay) or via register_outcome() API called by orchestrator.
- Duplicate detection prevention:
    - a per-candle cache prevents emitting the same pattern repeatedly for the same candle_id.

PUBLIC API:
- detect_candle_patterns(candles, volume_sma=None, volume_history=None, higher_tf_candles=None,
                        regime=None, session_phase=None, at_key_level=False, future_candles=None,
                        outcome_horizon_bars=10, outcome_threshold_pct=0.01) -> List[Dict]
"""

from __future__ import annotations

import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Deque, Tuple

from src.utils.logger import setup_logger

_logger = setup_logger("candle_patterns")


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        return v
    except (TypeError, ValueError):
        return default


def _get_ohlc(candle: Dict[str, Any]) -> Dict[str, float]:
    return {
        "open": float(_safe_float(candle.get("open"), 0.0) or 0.0),
        "high": float(_safe_float(candle.get("high"), 0.0) or 0.0),
        "low": float(_safe_float(candle.get("low"), 0.0) or 0.0),
        "close": float(_safe_float(candle.get("close"), 0.0) or 0.0),
        "volume": float(_safe_float(candle.get("volume"), 0.0) or 0.0),
    }


def _range_size(c: Dict[str, float]) -> float:
    return max(0.0, c["high"] - c["low"])


def _body_size(c: Dict[str, float]) -> float:
    return abs(c["close"] - c["open"])


def _upper_wick(c: Dict[str, float]) -> float:
    return max(0.0, c["high"] - max(c["open"], c["close"]))


def _lower_wick(c: Dict[str, float]) -> float:
    return max(0.0, min(c["open"], c["close"]) - c["low"])


def _is_bullish(c: Dict[str, float]) -> bool:
    return c["close"] > c["open"]


def _is_bearish(c: Dict[str, float]) -> bool:
    return c["close"] < c["open"]


def _extract_candle_id(candle: Dict[str, Any], fallback_index: int) -> str:
    ts = candle.get("timestamp")
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, str) and ts.strip():
        return ts.strip()
    return f"idx:{fallback_index}"


def _volume_ratio(c: Dict[str, float], volume_sma: Optional[float]) -> Optional[float]:
    if volume_sma is None or volume_sma <= 0:
        return None
    return float(c["volume"] / volume_sma) if volume_sma > 0 else None


def _volume_percentile(volume_ratio: Optional[float], volume_history: Optional[List[float]]) -> Optional[float]:
    if volume_ratio is None or not volume_history:
        return None
    vals = []
    for v in volume_history:
        try:
            fv = float(v)
            if fv > 0:
                vals.append(fv)
        except Exception:
            continue
    if len(vals) < 5:
        return None
    below = sum(1 for v in vals if v < volume_ratio)
    return round((below / len(vals)) * 100.0, 1)


def _volume_quality(volume_ratio: Optional[float]) -> Optional[int]:
    if volume_ratio is None:
        return None
    vr = float(volume_ratio)
    if vr < 0.5:
        return 10
    if vr < 0.8:
        return 35
    if vr < 1.2:
        return 60
    if vr < 2.0:
        return 80
    return 95


def _confidence(base_form_score: int, volume_sma: Optional[float], volume_ratio: Optional[float]) -> int:
    score = int(base_form_score)
    # Volume bonus only if volume_sma provided
    if volume_sma is not None and volume_sma > 0 and volume_ratio is not None:
        if float(volume_ratio) >= 0.8:
            score += 20
    return max(0, min(100, score))


def _htf_trend(higher_tf_candles: Optional[Union[List[Dict], deque]], lookback: int = 10) -> Optional[str]:
    if not higher_tf_candles:
        return None
    cl = list(higher_tf_candles)
    if len(cl) < 2:
        return None
    w = cl[-lookback:] if len(cl) >= lookback else cl
    try:
        first = float(_safe_float(w[0].get("close"), None) or 0.0)
        last = float(_safe_float(w[-1].get("close"), None) or 0.0)
    except Exception:
        return None
    if last > first:
        return "UP"
    if last < first:
        return "DOWN"
    return "FLAT"


# ---------------------------------------------------------------------
# Reliability tracking (module-level)
# ---------------------------------------------------------------------
@dataclass
class _Outcome:
    ts: datetime
    success: bool


class PatternReliabilityTracker:
    def __init__(self, maxlen: int = 50):
        self._lock = threading.RLock()
        self._outcomes: Dict[str, Deque[_Outcome]] = defaultdict(lambda: deque(maxlen=maxlen))

    def record_outcome(self, pattern: str, success: bool, ts: Optional[datetime] = None) -> None:
        if not pattern:
            return
        with self._lock:
            self._outcomes[pattern].append(_Outcome(ts=ts or datetime.utcnow(), success=bool(success)))

    def win_rate_20(self, pattern: str) -> Optional[float]:
        with self._lock:
            arr = list(self._outcomes.get(pattern, []))
        if len(arr) < 5:
            return None
        last = arr[-20:] if len(arr) >= 20 else arr
        wins = sum(1 for o in last if o.success)
        return round((wins / len(last)) * 100.0, 1)


_RELIABILITY = PatternReliabilityTracker(maxlen=50)

_RELIABILITY_LOG_LOCK = threading.Lock()
_RELIABILITY_LAST_LOG: Dict[str, float] = {}

_DEDUP_LOCK = threading.RLock()
_DEDUP_STATE = {"candle_id": None, "seen": set()}  # type: ignore[var-annotated]


def _dedup_allow(candle_id: str, pattern: str) -> bool:
    with _DEDUP_LOCK:
        if _DEDUP_STATE["candle_id"] != candle_id:
            _DEDUP_STATE["candle_id"] = candle_id
            _DEDUP_STATE["seen"] = set()
        key = (candle_id, pattern)
        if key in _DEDUP_STATE["seen"]:
            return False
        _DEDUP_STATE["seen"].add(key)
        return True


def _rate_limited_log(key: str, level: str, message: str, extra: Optional[Dict[str, Any]] = None, min_seconds: int = 300) -> None:
    import time as _t
    now = _t.time()
    with _RELIABILITY_LOG_LOCK:
        last = _RELIABILITY_LAST_LOG.get(key, 0.0)
        if now - last < min_seconds:
            return
        _RELIABILITY_LAST_LOG[key] = now

    if level == "warning":
        _logger.warning(message, extra=extra or {})
    elif level == "error":
        _logger.error(message, extra=extra or {})
    else:
        _logger.info(message, extra=extra or {})


def _evaluate_outcome(entry_close: float, direction: str, future_candles: List[Dict[str, Any]], threshold_pct: float) -> Optional[bool]:
    if entry_close <= 0 or not future_candles:
        return None
    try:
        highs = [float(_safe_float(c.get("high"), None) or 0.0) for c in future_candles]
        lows = [float(_safe_float(c.get("low"), None) or 0.0) for c in future_candles]
        if not highs or not lows:
            return None
        if direction.upper() == "BULLISH":
            return max(highs) >= entry_close * (1.0 + threshold_pct)
        if direction.upper() == "BEARISH":
            return min(lows) <= entry_close * (1.0 - threshold_pct)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------
def detect_candle_patterns(
    candles: Union[List[Dict], deque],
    volume_sma: Optional[float] = None,
    volume_history: Optional[List[float]] = None,
    higher_tf_candles: Optional[Union[List[Dict], deque]] = None,
    regime: Optional[str] = None,
    session_phase: Optional[str] = None,
    at_key_level: bool = False,
    future_candles: Optional[Union[List[Dict], deque]] = None,
    outcome_horizon_bars: int = 10,
    outcome_threshold_pct: float = 0.01,
) -> List[Dict]:
    candle_list = list(candles) if candles is not None else []
    patterns: List[Dict] = []
    if len(candle_list) < 1:
        return patterns

    last = candle_list[-1]
    prev = candle_list[-2] if len(candle_list) >= 2 else None
    prev2 = candle_list[-3] if len(candle_list) >= 3 else None

    c_last = _get_ohlc(last)
    rng = _range_size(c_last)
    body = _body_size(c_last)
    upper = _upper_wick(c_last)
    lower = _lower_wick(c_last)

    candle_id = _extract_candle_id(last, len(candle_list) - 1)

    body_ratio = float(body / (rng + 0.01)) if rng >= 0 else 0.0
    upper_wick_ratio = float(upper / (rng + 1e-12)) if rng > 0 else 0.0
    lower_wick_ratio = float(lower / (rng + 1e-12)) if rng > 0 else 0.0

    vol_ratio = _volume_ratio(c_last, volume_sma)
    vol_pct = _volume_percentile(vol_ratio, volume_history)
    vol_quality = _volume_quality(vol_ratio)

    htf = _htf_trend(higher_tf_candles)
    regime_u = regime.upper() if isinstance(regime, str) else None
    session_u = session_phase.upper() if isinstance(session_phase, str) else None

    def base_payload() -> Dict[str, Any]:
        return {
            "body_ratio": round(body_ratio, 4),
            "upper_wick_ratio": round(upper_wick_ratio, 4),
            "lower_wick_ratio": round(lower_wick_ratio, 4),
            "volume_ratio": round(float(vol_ratio), 4) if vol_ratio is not None else None,
            "volume_percentile": vol_pct,
            "volume_quality": vol_quality,
            "candle_range": round(rng, 4),
            "mtf_trend": htf,
            "session_phase": session_u,
            "regime": regime_u,
        }

    def apply_context_filters(pattern_name: str, direction: str) -> Tuple[bool, Optional[str]]:
        if htf is None or regime_u is None:
            return False, None
        if regime_u == "TRENDING":
            if pattern_name == "HAMMER" and direction == "BULLISH" and htf == "DOWN" and not at_key_level:
                return True, "hammer_suppressed_in_downtrend_without_key_level"
            if pattern_name == "SHOOTING_STAR" and direction == "BEARISH" and htf == "UP" and not at_key_level:
                return True, "shooting_star_suppressed_in_uptrend_without_key_level"
        if session_u == "LUNCH_LULL" and not at_key_level:
            return False, "lunch_lull_noise"
        return False, None

    def apply_mtf_bonus(conf: int, direction: str) -> Tuple[int, bool]:
        if htf is None:
            return conf, False
        if direction == "BULLISH" and htf == "UP":
            return min(100, conf + 10), True
        if direction == "BEARISH" and htf == "DOWN":
            return min(100, conf + 10), True
        return conf, False

    def apply_reliability(pattern_name: str, conf: int) -> Tuple[Optional[int], Dict[str, Any]]:
        wr = _RELIABILITY.win_rate_20(pattern_name)
        info = {"win_rate_20": wr, "confidence_adjusted": False, "disabled": False}
        if wr is None:
            return conf, info
        if wr < 25.0:
            info["disabled"] = True
            _rate_limited_log(
                key=f"pattern_disabled:{pattern_name}",
                level="error",
                message="Pattern disabled due to low win_rate_20",
                extra={"pattern": pattern_name, "win_rate_20": wr},
                min_seconds=600,
            )
            return None, info
        if wr < 40.0:
            info["confidence_adjusted"] = True
            _rate_limited_log(
                key=f"pattern_reduced:{pattern_name}",
                level="warning",
                message="Pattern confidence reduced due to low win_rate_20",
                extra={"pattern": pattern_name, "win_rate_20": wr},
                min_seconds=600,
            )
            return int(round(conf * 0.7)), info
        return conf, info

    future_list = list(future_candles) if future_candles is not None else None
    if future_list is not None and outcome_horizon_bars > 0:
        future_list = future_list[:outcome_horizon_bars]

    def emit(pattern_name: str, direction: str, base_score: int, extra_fields: Optional[Dict[str, Any]] = None) -> None:
        if not _dedup_allow(candle_id, pattern_name):
            return

        context_filtered, context_reason = apply_context_filters(pattern_name, direction)

        conf = _confidence(base_score, volume_sma, vol_ratio)
        conf, mtf_confirmed = apply_mtf_bonus(conf, direction)
        conf_or_none, reliability_info = apply_reliability(pattern_name, conf)
        if conf_or_none is None:
            return

        out = {
            "pattern": pattern_name,
            "direction": direction,
            "confidence": int(conf_or_none),
            "candle_index": len(candle_list) - 1,
            "candle_id": candle_id,
            "mtf_confirmed": bool(mtf_confirmed),
            "context_filtered": bool(context_filtered),
            "context_reason": context_reason,
            "reliability": reliability_info,
            **base_payload(),
        }
        if extra_fields:
            out.update(extra_fields)

        if future_list is not None:
            entry_close = float(c_last["close"])
            success = _evaluate_outcome(entry_close, direction, future_list, outcome_threshold_pct)
            if success is not None:
                _RELIABILITY.record_outcome(pattern_name, success=success, ts=datetime.utcnow())

        patterns.append(out)

    # Blueprint pattern rules
    if rng > 0 and body > 0 and lower >= body * 2 and upper <= body * 0.5:
        emit("HAMMER", "BULLISH", 30)

    if rng > 0 and body > 0 and upper >= body * 2 and lower <= body * 0.5:
        emit("SHOOTING_STAR", "BEARISH", 30)

    if rng > 0 and body <= rng * 0.1:
        emit("DOJI", "NEUTRAL", 25)

    if prev is not None:
        c_prev = _get_ohlc(prev)
        prev_body = _body_size(c_prev)
        if _is_bearish(c_prev) and _is_bullish(c_last) and body > prev_body:
            if c_last["open"] <= c_prev["close"] and c_last["close"] >= c_prev["open"]:
                emit("BULLISH_ENGULFING", "BULLISH", 35, {"prev_body": round(prev_body, 4)})

    if prev is not None:
        c_prev = _get_ohlc(prev)
        prev_body = _body_size(c_prev)
        if _is_bullish(c_prev) and _is_bearish(c_last) and body > prev_body:
            if c_last["open"] >= c_prev["close"] and c_last["close"] <= c_prev["open"]:
                emit("BEARISH_ENGULFING", "BEARISH", 35, {"prev_body": round(prev_body, 4)})

    if prev is not None:
        c_prev = _get_ohlc(prev)
        if c_last["high"] <= c_prev["high"] and c_last["low"] >= c_prev["low"]:
            emit("INSIDE_BAR", "NEUTRAL", 25, {"mother_high": c_prev["high"], "mother_low": c_prev["low"]})

    if prev is not None and prev2 is not None:
        c_prev = _get_ohlc(prev)
        c_prev2 = _get_ohlc(prev2)
        prev2_rng = _range_size(c_prev2)
        prev_body = _body_size(c_prev)
        if (
            _is_bearish(c_prev2)
            and prev2_rng > 0
            and prev_body < prev2_rng * 0.3
            and _is_bullish(c_last)
            and c_last["close"] > (c_prev2["open"] + c_prev2["close"]) / 2.0
        ):
            emit("MORNING_STAR", "BULLISH", 40)

    if prev is not None and prev2 is not None:
        c_prev = _get_ohlc(prev)
        c_prev2 = _get_ohlc(prev2)
        prev2_rng = _range_size(c_prev2)
        prev_body = _body_size(c_prev)
        if (
            _is_bullish(c_prev2)
            and prev2_rng > 0
            and prev_body < prev2_rng * 0.3
            and _is_bearish(c_last)
            and c_last["close"] < (c_prev2["open"] + c_prev2["close"]) / 2.0
        ):
            emit("EVENING_STAR", "BEARISH", 40)

    return patterns


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Candle Patterns Test (Institutional)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    print(" [Test 1] Hammer detection...")
    candles1 = [
        {"open": 23100, "high": 23110, "low": 23100, "close": 23105, "volume": 5000},
        {"open": 23100, "high": 23106, "low": 23080, "close": 23104, "volume": 8000},
    ]
    p1 = detect_candle_patterns(candles1, volume_sma=6000)
    if any(x["pattern"] == "HAMMER" for x in p1):
        print(" ✅ Hammer detected")
        passed += 1
    else:
        print(f" ❌ Hammer not detected: {p1}")
        failed += 1

    print("\n [Test 2] Engulfing detection...")
    candles2 = [
        {"open": 23100, "high": 23105, "low": 23080, "close": 23085, "volume": 5000},
        {"open": 23080, "high": 23120, "low": 23075, "close": 23115, "volume": 10000},
    ]
    p2 = detect_candle_patterns(candles2, volume_sma=6000)
    if any(x["pattern"] == "BULLISH_ENGULFING" for x in p2):
        print(" ✅ Bullish engulfing detected")
        passed += 1
    else:
        print(f" ❌ Engulfing not detected: {p2}")
        failed += 1

    print("\n [Test 3] Volume bonus not applied when volume_sma missing (dedup-safe)...")
    # Use a different candle_id by adding a timestamp so dedup doesn't suppress emission
    candles1b = [
        {"open": 23100, "high": 23110, "low": 23100, "close": 23105, "volume": 5000, "timestamp": datetime(2026, 4, 1, 10, 0, 0)},
        {"open": 23100, "high": 23106, "low": 23080, "close": 23104, "volume": 8000, "timestamp": datetime(2026, 4, 1, 10, 1, 0)},
    ]
    p3 = detect_candle_patterns(candles1b, volume_sma=None)
    h = next((x for x in p3 if x["pattern"] == "HAMMER"), None)
    if h and h["confidence"] == 30:
        print(" ✅ No volume_sma => no volume bonus")
        passed += 1
    else:
        print(f" ❌ Volume bonus behavior wrong: {h}")
        failed += 1

    print("\n [Test 4] Empty input safe...")
    if detect_candle_patterns([]) == []:
        print(" ✅ Empty safe")
        passed += 1
    else:
        print(" ❌ Empty not safe")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n ✅ Candle Patterns institutional test passed.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")


if __name__ == "__main__":
    _run_tests()