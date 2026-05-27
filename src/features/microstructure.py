import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Union, Any

from src.utils.logger import setup_logger

_logger = setup_logger("microstructure")


# ============================================
# Market Depth Analysis
# ============================================

def _validate_market_depth(market_depth: Any) -> Tuple[bool, str, List[Dict], List[Dict]]:
    """
    Validate that market_depth contains usable bids/asks.

    Returns:
        (available, quality, bids, asks)
        quality in {"OK","MISSING","INVALID"}
    """
    if market_depth is None:
        return False, "MISSING", [], []
    if not isinstance(market_depth, dict):
        return False, "INVALID", [], []

    bids = market_depth.get("bids", market_depth.get("best_5_buy", []))
    asks = market_depth.get("asks", market_depth.get("best_5_sell", []))

    if not isinstance(bids, list) or not isinstance(asks, list):
        return False, "INVALID", [], []
    if len(bids) == 0 or len(asks) == 0:
        return False, "MISSING", [], []

    # Require at least one dict with a valid price in each side
    def _has_valid_price(levels: List[Dict]) -> bool:
        for x in levels:
            if isinstance(x, dict):
                p = _safe_float(x.get("price", None), default=None)
                if p is not None and p > 0:
                    return True
        return False

    if not _has_valid_price(bids) or not _has_valid_price(asks):
        return False, "INVALID", [], []

    return True, "OK", bids, asks


def compute_spread(
    market_depth: Dict,
    spot_price: float = 0.0,
) -> Dict:
    """
    Compute bid-ask spread from market depth data.

    NOTE:
        This function is a low-level utility and returns numeric defaults for empty inputs.
        The master compute_microstructure_features() is responsible for converting missing/invalid
        depth into None + data_quality flags (institutional safety).
    """
    bids = market_depth.get("bids", market_depth.get("best_5_buy", []))
    asks = market_depth.get("asks", market_depth.get("best_5_sell", []))

    best_bid = 0.0
    best_ask = 0.0
    total_bid_qty = 0
    total_ask_qty = 0

    if bids and isinstance(bids, list):
        for b in bids:
            if isinstance(b, dict):
                price = _safe_float(b.get("price", 0))
                qty = _safe_int(b.get("qty", 0))
                if price > best_bid:
                    best_bid = price
                total_bid_qty += qty

    if asks and isinstance(asks, list):
        for a in asks:
            if isinstance(a, dict):
                price = _safe_float(a.get("price", 0))
                qty = _safe_int(a.get("qty", 0))
                if best_ask == 0 or (price > 0 and price < best_ask):
                    best_ask = price
                total_ask_qty += qty

    spread = round(best_ask - best_bid, 2) if best_ask > best_bid else 0.0

    mid_price = spot_price if spot_price > 0 else ((best_bid + best_ask) / 2 if best_bid > 0 else 0)
    spread_bps = round((spread / mid_price) * 10000, 2) if mid_price > 0 else 0.0

    total = total_bid_qty + total_ask_qty
    imbalance = round((total_bid_qty - total_ask_qty) / total, 4) if total > 0 else 0.0

    return {
        "spread": spread,
        "spread_bps": spread_bps,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "total_bid_qty": total_bid_qty,
        "total_ask_qty": total_ask_qty,
        "bid_ask_imbalance": imbalance,
    }


def compute_spread_zscore(
    current_spread: Optional[float],
    spread_history: List[float],
) -> Optional[float]:
    """
    Compute Z-score of current spread vs recent history.
    Returns None if current_spread is None or insufficient history.
    """
    if current_spread is None:
        return None
    if len(spread_history) < 10:
        return None

    recent = spread_history[-300:]
    avg = sum(recent) / len(recent)

    if avg <= 0:
        return 0.0

    variance = sum((s - avg) ** 2 for s in recent) / len(recent)
    std = math.sqrt(variance)

    if std <= 0:
        if abs(current_spread - avg) > 0.001:
            return 10.0 if current_spread > avg else -10.0
        return 0.0

    return round((current_spread - avg) / std, 4)


def compute_ofi(
    current_depth: Dict,
    previous_depth: Dict,
) -> float:
    """
    Compute Order Flow Imbalance (OFI).

    NOTE:
        No normalization here (kept minimal). Consumers must interpret with caution.
        Master function will output ofi=None if depth data is missing/invalid.
    """
    curr_bid = _safe_int(current_depth.get("total_bid_qty", 0))
    curr_ask = _safe_int(current_depth.get("total_ask_qty", 0))
    prev_bid = _safe_int(previous_depth.get("total_bid_qty", 0))
    prev_ask = _safe_int(previous_depth.get("total_ask_qty", 0))

    delta_bid = curr_bid - prev_bid
    delta_ask = curr_ask - prev_ask

    return float(delta_bid - delta_ask)


# ============================================
# Candle-Based Microstructure
# ============================================

def compute_wick_ratios(candle: Dict) -> Dict:
    """
    Compute wick ratios for a single candle.
    """
    try:
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
    except Exception:
        return {
            "upper_wick_ratio": 0.0,
            "lower_wick_ratio": 0.0,
            "body_ratio": 0.0,
            "is_bullish": False,
        }

    total_range = h - l
    if total_range <= 0:
        return {
            "upper_wick_ratio": 0.0,
            "lower_wick_ratio": 0.0,
            "body_ratio": 0.0,
            "is_bullish": c >= o,
        }

    body_top = max(o, c)
    body_bottom = min(o, c)
    body = body_top - body_bottom

    upper_wick = h - body_top
    lower_wick = body_bottom - l

    return {
        "upper_wick_ratio": round(upper_wick / total_range, 4),
        "lower_wick_ratio": round(lower_wick / total_range, 4),
        "body_ratio": round(body / (total_range + 0.01), 4),
        "is_bullish": c >= o,
    }


def compute_trade_intensity(candle: Dict) -> float:
    """
    Trade intensity = volume per point of range.

    NOTE:
        Kept as float for backward compatibility.
        Missing volume is treated as 0.0 (low-intensity). Consumers should use
        data_quality flags if they need strict validation.
    """
    vol = _safe_int(candle.get("volume", 0))
    try:
        rng = float(candle["high"]) - float(candle["low"])
    except Exception:
        return 0.0

    if rng <= 0 or vol <= 0:
        return 0.0

    return round(vol / rng, 2)


# ============================================
# Pattern Detection
# ============================================

def detect_absorption(
    candles: List[Dict],
    volume_ratio_threshold: float = 2.0,
    body_ratio_threshold: float = 0.3,
    consecutive_required: int = 2,
    price_position: Optional[str] = None,  # NEW: LOW/HIGH/MID
) -> Dict:
    """
    Detect absorption — high volume with tiny body.

    Direction Hardening:
      - If wick ratios are ambiguous (difference < 0.1), use price_position:
            LOW -> BULLISH, HIGH -> BEARISH, MID/None -> NONE
    """
    if len(candles) < max(21, consecutive_required):
        _logger.debug("Absorption not evaluated: insufficient candles", extra={"candles": len(candles)})
        return {
            "absorption_detected": False,
            "absorption_direction": "NONE",
            "absorption_count": 0,
        }

    # Average volume excluding last pattern candles (as before), but ensure float math and zero guard
    lookback = min(20, len(candles) - consecutive_required)
    base_slice = candles[-lookback - consecutive_required : -consecutive_required]
    vols = [float(_safe_int(c.get("volume", 0))) for c in base_slice if isinstance(c, dict)]
    avg_vol = (sum(vols) / float(len(vols))) if vols else 0.0

    if avg_vol <= 0:
        _logger.debug("Absorption avg volume <=0; cannot evaluate reliably")
        return {
            "absorption_detected": False,
            "absorption_direction": "NONE",
            "absorption_count": 0,
        }

    consecutive = 0
    absorption_direction = "NONE"

    for candle in candles[-consecutive_required:]:
        if not isinstance(candle, dict):
            consecutive = 0
            absorption_direction = "NONE"
            continue

        vol = float(_safe_int(candle.get("volume", 0)))
        if vol <= 0:
            consecutive = 0
            absorption_direction = "NONE"
            continue

        vol_ratio = vol / avg_vol if avg_vol > 0 else 0.0

        try:
            rng = float(candle["high"]) - float(candle["low"])
            body = abs(float(candle["close"]) - float(candle["open"]))
        except Exception:
            consecutive = 0
            absorption_direction = "NONE"
            continue

        body_ratio = body / (rng + 0.01) if rng >= 0 else 999.0

        if vol_ratio >= volume_ratio_threshold and body_ratio <= body_ratio_threshold:
            consecutive += 1
            wicks = compute_wick_ratios(candle)

            # Default wick-based direction
            wick_diff = abs(wicks["lower_wick_ratio"] - wicks["upper_wick_ratio"])
            if wick_diff < 0.1 and price_position is not None:
                pp = price_position.upper()
                if pp == "LOW":
                    absorption_direction = "BULLISH"
                elif pp == "HIGH":
                    absorption_direction = "BEARISH"
                else:
                    absorption_direction = "NONE"
            else:
                if wicks["lower_wick_ratio"] > wicks["upper_wick_ratio"]:
                    absorption_direction = "BULLISH"
                else:
                    absorption_direction = "BEARISH"
        else:
            consecutive = 0
            absorption_direction = "NONE"

    detected = consecutive >= consecutive_required
    return {
        "absorption_detected": detected,
        "absorption_direction": absorption_direction if detected else "NONE",
        "absorption_count": consecutive,
    }


def detect_exhaustion(
    candles: List[Dict],
    lookback: int = 10,
    volume_decline_threshold: float = 0.7,
    rsi_extreme_high: float = 70.0,
    rsi_extreme_low: float = 30.0,
    rsi_value: Optional[float] = None,
) -> Dict:
    """
    Detect exhaustion — new extreme with declining volume AND RSI extreme.

    MANDATE FIX:
      - RSI is REQUIRED. If rsi_value is None, return detected=False and log WARNING.
    """
    if rsi_value is None:
        _logger.warning("Exhaustion not evaluated: rsi_value is None (RSI required)")
        return {"exhaustion_detected": False, "exhaustion_type": "NONE"}

    if len(candles) < lookback + 1:
        _logger.debug("Exhaustion not evaluated: insufficient candles", extra={"candles": len(candles), "need": lookback + 1})
        return {"exhaustion_detected": False, "exhaustion_type": "NONE"}

    current = candles[-1]
    recent = candles[-lookback - 1 : -1]

    try:
        recent_high = max(float(c["high"]) for c in recent if isinstance(c, dict) and "high" in c)
        recent_low = min(float(c["low"]) for c in recent if isinstance(c, dict) and "low" in c)
        cur_high = float(current["high"])
        cur_low = float(current["low"])
    except Exception:
        return {"exhaustion_detected": False, "exhaustion_type": "NONE"}

    is_new_high = cur_high > recent_high
    is_new_low = cur_low < recent_low

    if not is_new_high and not is_new_low:
        return {"exhaustion_detected": False, "exhaustion_type": "NONE"}

    # Volume decline
    vols = [float(_safe_int(c.get("volume", 0))) for c in recent if isinstance(c, dict)]
    avg_vol = (sum(vols) / len(vols)) if vols else 0.0
    current_vol = float(_safe_int(current.get("volume", 0)))

    vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 1.0
    volume_declining = vol_ratio < volume_decline_threshold

    # RSI extreme required
    rsi_extreme = False
    if is_new_high and float(rsi_value) > rsi_extreme_high:
        rsi_extreme = True
    elif is_new_low and float(rsi_value) < rsi_extreme_low:
        rsi_extreme = True

    detected = bool(volume_declining and rsi_extreme)

    exhaustion_type = "NONE"
    if detected:
        if is_new_high:
            exhaustion_type = "BULLISH_EXHAUSTION"
        elif is_new_low:
            exhaustion_type = "BEARISH_EXHAUSTION"

    return {
        "exhaustion_detected": detected,
        "exhaustion_type": exhaustion_type,
        "new_high": is_new_high,
        "new_low": is_new_low,
        "volume_ratio": round(vol_ratio, 4),
    }


def detect_stop_hunt(
    candles: List[Dict],
    swing_lookback: int = 5,
    pierce_points: float = 3.0,
    volume_threshold: float = 1.5,
    wick_body_ratio: float = 2.0,
    swing_lows: Optional[List[float]] = None,   # NEW: provided by Key Levels
    swing_highs: Optional[List[float]] = None,  # NEW: provided by Key Levels
) -> Dict:
    """
    Detect stop hunt — price pierces beyond a swing level, then reclaims.

    MANDATE FIX:
      - Swing points must be supplied by caller (Key Levels engine), not recomputed here.
    """
    result = {
        "stop_hunt_detected": False,
        "stop_hunt_type": "NONE",
        "stop_hunt_level": 0.0,
    }

    if len(candles) < 3:
        _logger.debug("Stop hunt not evaluated: insufficient candles", extra={"candles": len(candles)})
        return result

    if not swing_lows and not swing_highs:
        _logger.debug("Stop hunt not evaluated: swing points not provided")
        return result

    current = candles[-1]

    # Volume window: use last up to 20 prior candles for stability
    analysis_window = candles[-21:-1] if len(candles) >= 22 else candles[:-1]
    if not analysis_window:
        _logger.debug("Stop hunt not evaluated: no analysis window")
        return result

    avg_vol = sum(float(_safe_int(c.get("volume", 0))) for c in analysis_window if isinstance(c, dict)) / max(len(analysis_window), 1)
    current_vol = float(_safe_int(current.get("volume", 0)))
    vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 0.0

    try:
        body = abs(float(current["close"]) - float(current["open"]))
        lower_wick = min(float(current["open"]), float(current["close"])) - float(current["low"])
        upper_wick = float(current["high"]) - max(float(current["open"]), float(current["close"]))
        cur_low = float(current["low"])
        cur_high = float(current["high"])
        cur_close = float(current["close"])
    except Exception:
        return result

    # BUY-side stop hunt: pierce below swing low then close above
    if swing_lows:
        for lvl in swing_lows:
            try:
                swing_low = float(lvl)
            except Exception:
                continue
            pierce = swing_low - cur_low
            if (
                pierce >= pierce_points
                and cur_close > swing_low
                and vol_ratio >= volume_threshold
                and body > 0
                and lower_wick >= body * wick_body_ratio
            ):
                result["stop_hunt_detected"] = True
                result["stop_hunt_type"] = "BUY_HUNT"
                result["stop_hunt_level"] = float(swing_low)
                return result

    # SELL-side stop hunt: pierce above swing high then close below
    if swing_highs:
        for lvl in swing_highs:
            try:
                swing_high = float(lvl)
            except Exception:
                continue
            pierce = cur_high - swing_high
            if (
                pierce >= pierce_points
                and cur_close < swing_high
                and vol_ratio >= volume_threshold
                and body > 0
                and upper_wick >= body * wick_body_ratio
            ):
                result["stop_hunt_detected"] = True
                result["stop_hunt_type"] = "SELL_HUNT"
                result["stop_hunt_level"] = float(swing_high)
                return result

    return result


# ============================================
# Master Feature Computation
# ============================================

def compute_microstructure_features(
    candles: Union[List[Dict], deque],
    market_depth: Optional[Dict] = None,
    spread_history: Optional[List[float]] = None,
    previous_depth: Optional[Dict] = None,
    rsi_value: Optional[float] = None,
    spot_price: float = 0.0,
    swing_lows: Optional[List[float]] = None,   # NEW (optional)
    swing_highs: Optional[List[float]] = None,  # NEW (optional)
) -> Dict:
    """
    Compute all microstructure features.

    NEW:
      - depth_data_available + depth_data_quality
      - swing_lows/swing_highs passed through to stop hunt detector
    """
    candle_list = list(candles) if candles is not None else []
    if not candle_list:
        return _empty_features()

    # Determine price_position context for absorption direction
    price_position = None
    try:
        last_close = float(candle_list[-1]["close"])
        lookback = candle_list[-20:] if len(candle_list) >= 20 else candle_list
        hi = max(float(c["high"]) for c in lookback if isinstance(c, dict) and "high" in c)
        lo = min(float(c["low"]) for c in lookback if isinstance(c, dict) and "low" in c)
        rng = hi - lo
        if rng > 0:
            pos = (last_close - lo) / rng
            if pos <= 0.25:
                price_position = "LOW"
            elif pos >= 0.75:
                price_position = "HIGH"
            else:
                price_position = "MID"
    except Exception:
        price_position = None

    # Depth validation (MANDATE)
    depth_available, depth_quality, bids, asks = _validate_market_depth(market_depth)

    if depth_available:
        depth_data = compute_spread({"bids": bids, "asks": asks}, spot_price)
        spread_z = compute_spread_zscore(depth_data["spread"], spread_history or [])
        ofi_val = compute_ofi(depth_data, previous_depth or {}) if previous_depth else 0.0
        spread_val: Optional[float] = depth_data["spread"]
        spread_bps_val: Optional[float] = depth_data["spread_bps"]
        ofi: Optional[float] = float(ofi_val)
        imbalance: Optional[float] = depth_data["bid_ask_imbalance"]
        total_bid_qty: Optional[int] = depth_data["total_bid_qty"]
        total_ask_qty: Optional[int] = depth_data["total_ask_qty"]
    else:
        # MANDATE: None, not 0.0
        if market_depth is not None:
            _logger.warning("Depth data missing/invalid; spread/ofi set to None", extra={"quality": depth_quality})
        depth_data = {
            "spread": None, "spread_bps": None,
            "best_bid": None, "best_ask": None,
            "total_bid_qty": None, "total_ask_qty": None,
            "bid_ask_imbalance": None,
        }
        spread_z = None
        spread_val = None
        spread_bps_val = None
        ofi = None
        imbalance = None
        total_bid_qty = None
        total_ask_qty = None

    # Last candle wick analysis
    last_candle = candle_list[-1]
    wicks = compute_wick_ratios(last_candle)
    intensity = compute_trade_intensity(last_candle)

    # Absorption detection (direction hardened)
    absorption = detect_absorption(candle_list, price_position=price_position)

    # Exhaustion detection (RSI required)
    exhaustion = detect_exhaustion(candle_list, rsi_value=rsi_value)

    # Stop hunt detection (swing points integrated)
    stop_hunt = detect_stop_hunt(candle_list, swing_lows=swing_lows, swing_highs=swing_highs)

    result = {
        # Depth Data Quality Flags (MANDATE)
        "depth_data_available": bool(depth_available),
        "depth_data_quality": depth_quality,

        # Spread & Depth
        "spread": spread_val,
        "spread_bps": spread_bps_val,
        "spread_zscore": spread_z,
        "bid_ask_imbalance": imbalance,
        "total_bid_qty": total_bid_qty,
        "total_ask_qty": total_ask_qty,
        "ofi": ofi,

        # Candle Microstructure
        "upper_wick_ratio": wicks["upper_wick_ratio"],
        "lower_wick_ratio": wicks["lower_wick_ratio"],
        "body_ratio": wicks["body_ratio"],
        "trade_intensity": intensity,

        # Pattern Detection
        "absorption_detected": absorption["absorption_detected"],
        "absorption_direction": absorption["absorption_direction"],
        "exhaustion_detected": exhaustion["exhaustion_detected"],
        "exhaustion_type": exhaustion["exhaustion_type"],
        "stop_hunt_detected": stop_hunt["stop_hunt_detected"],
        "stop_hunt_type": stop_hunt["stop_hunt_type"],
        "stop_hunt_level": stop_hunt["stop_hunt_level"],

        # Metadata (backward compatible)
        "candle_count": len(candle_list),
        "has_depth_data": bool(depth_available),
    }

    return result


# ============================================
# Utility
# ============================================

def _safe_float(value, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: Optional[int] = 0) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _empty_features() -> Dict:
    return {
        "depth_data_available": False,
        "depth_data_quality": "MISSING",

        "spread": None,
        "spread_bps": None,
        "spread_zscore": None,
        "bid_ask_imbalance": None,
        "total_bid_qty": None,
        "total_ask_qty": None,
        "ofi": None,

        "upper_wick_ratio": 0.0,
        "lower_wick_ratio": 0.0,
        "body_ratio": 0.0,
        "trade_intensity": 0.0,

        "absorption_detected": False,
        "absorption_direction": "NONE",
        "exhaustion_detected": False,
        "exhaustion_type": "NONE",
        "stop_hunt_detected": False,
        "stop_hunt_type": "NONE",
        "stop_hunt_level": 0.0,

        "candle_count": 0,
        "has_depth_data": False,
    }


# ============================================
# Module Self-Test
# ============================================

def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Microstructure Features Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: Spread computation (utility) ──
    print("  [Test 1] Spread computation...")
    depth = {
        "bids": [
            {"price": 22399.0, "qty": 150},
            {"price": 22398.5, "qty": 200},
            {"price": 22398.0, "qty": 300},
        ],
        "asks": [
            {"price": 22399.5, "qty": 180},
            {"price": 22400.0, "qty": 250},
            {"price": 22400.5, "qty": 350},
        ],
    }
    spread_data = compute_spread(depth, 22400)
    if spread_data["spread"] == 0.5:
        print(f"    ✅ Spread = {spread_data['spread']} (correct)")
        passed += 1
    else:
        print(f"    ❌ Spread = {spread_data['spread']} (expected 0.5)")
        failed += 1

    # ── Test 2: Exhaustion requires RSI ──
    print("\n  [Test 2] Exhaustion requires RSI...")
    exh_candles = []
    for i in range(15):
        vol = 10000 - i * 500
        exh_candles.append({
            "open": 23000 + i * 5,
            "high": 23000 + i * 5 + 8,
            "low": 23000 + i * 5 - 3,
            "close": 23000 + i * 5 + 5,
            "volume": max(100, vol),
        })
    ex_none = detect_exhaustion(exh_candles, rsi_value=None)
    if ex_none["exhaustion_detected"] is False:
        print("    ✅ RSI None => exhaustion_detected False")
        passed += 1
    else:
        print("    ❌ Exhaustion should be False when RSI missing")
        failed += 1

    # ── Test 3: Missing depth data flags + None outputs ──
    print("\n  [Test 3] Missing depth data flags...")
    feats_no_depth = compute_microstructure_features(exh_candles, market_depth=None)
    if feats_no_depth["depth_data_available"] is False and feats_no_depth["spread"] is None and feats_no_depth["ofi"] is None:
        print("    ✅ Missing depth => depth_data_available False and spread/ofi None")
        passed += 1
    else:
        print(f"    ❌ Depth flags/values wrong: {feats_no_depth['depth_data_available']}, spread={feats_no_depth['spread']}, ofi={feats_no_depth['ofi']}")
        failed += 1

    # ── Test 4: Stop hunt with externally provided swing points ──
    print("\n  [Test 4] Stop hunt with external swing points...")
    sh_candles = []
    # Create a known swing low at 23020, then pierce to 23010 and close above.
    for i in range(12):
        sh_candles.append({
            "open": 23050 - i * 3,
            "high": 23055 - i * 3,
            "low": 23045 - i * 3,
            "close": 23048 - i * 3,
            "volume": 5000,
        })
    # Hunt candle
    sh_candles.append({
        "open": 23030,
        "high": 23035,
        "low": 23010,
        "close": 23028,   # close above swing low
        "volume": 12000,
    })
    sh = detect_stop_hunt(sh_candles, swing_lows=[23020.0], swing_highs=[])
    if sh["stop_hunt_detected"] and sh["stop_hunt_type"] == "BUY_HUNT":
        print(f"    ✅ Stop hunt detected at {sh['stop_hunt_level']}")
        passed += 1
    else:
        print(f"    ❌ Stop hunt not detected: {sh}")
        failed += 1

    # ── Test 5: Full computation with valid depth ──
    print("\n  [Test 5] Full compute_microstructure_features() with depth...")
    features = compute_microstructure_features(
        candles=exh_candles,
        market_depth=depth,
        spot_price=22400,
        rsi_value=75,
        swing_lows=[22950.0],
        swing_highs=[23100.0],
    )

    required = [
        "depth_data_available", "depth_data_quality",
        "spread", "spread_bps", "spread_zscore", "ofi",
        "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "trade_intensity",
        "absorption_detected", "absorption_direction",
        "exhaustion_detected", "exhaustion_type",
        "stop_hunt_detected", "stop_hunt_type", "stop_hunt_level",
        "candle_count", "has_depth_data",
    ]
    missing = [k for k in required if k not in features]
    if not missing:
        print("    ✅ All required keys present")
        passed += 1
    else:
        print(f"    ❌ Missing keys: {missing}")
        failed += 1

    if features["depth_data_available"] is True and features["spread"] is not None:
        print("    ✅ Depth processed with numeric spread")
        passed += 1
    else:
        print("    ❌ Depth should be available with numeric spread")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n  ✅ Microstructure Features hardened successfully!")
    else:
        print(f"\n  ⚠️  {failed} tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()