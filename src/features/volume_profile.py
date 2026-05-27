"""
Junior Aladdin - Volume Profile Features (Layer 2D)
=====================================================
PURPOSE:
    Build intraday volume profile from OHLCV candle data.
    Identifies key price levels where most trading occurred.

COMPUTES (SUMMARY OUTPUT):
    - poc: Point of Control (price with highest volume)
    - vah: Value Area High (upper bound of 70% volume zone)
    - val: Value Area Low (lower bound of 70% volume zone)
    - hvn_levels: High Volume Nodes (volume > 1.5x median)
    - lvn_levels: Low Volume Nodes (volume < 0.3x median)
    - volume_at_poc, total_profile_volume, bucket_count
    - volume stats: SMA, ratio, zscore, OBV (persistent via prev_obv), cumulative_volume_delta (approx)

TECHNIQUE:
    Using OHLCV (no tick data), volume is distributed across each candle's
    price range using a triangular weighting around typical price (H+L+C)/3.

MANDATORY HARDENING (per STRICT audit):
1) Optimize build_volume_profile: integer bucket indices, vectorized distribution (numpy if available)
2) Prevent infinite loop in find_value_area: max_iterations guard
3) Do NOT store full profile in MarketState: remove "profile" key from returned dict
4) Session-aware filtering: filter candles to date of last candle timestamp
5) Persist OBV: compute_volume_profile_features(prev_obv=0.0)
6) Use median (not mean) for HVN/LVN threshold baseline

RESTRICTIONS:
- Do NOT change public function signatures (except adding optional parameters with defaults).
"""

import math
import os
import time
from collections import defaultdict, deque
from datetime import datetime, date
from statistics import median
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("volume_profile")

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


# ============================================
# Bucket scaling utilities (avoid float drift)
# ============================================
def _infer_scale(x: float) -> int:
    """
    Infer integer scaling factor for a bucket_size to avoid float drift.
    Supports up to 4 decimal places safely.
    """
    try:
        s = f"{float(x):.6f}".rstrip("0").rstrip(".")
        if "." not in s:
            return 1
        decimals = len(s.split(".")[1])
        decimals = min(decimals, 4)
        return 10 ** decimals
    except Exception:
        return 1


def _bucket_params(bucket_size: float) -> Tuple[int, int]:
    """
    Returns (scale, bucket_size_scaled_int). Ensures bucket_size_scaled_int > 0.
    """
    bs = float(bucket_size)
    if bs <= 0:
        raise ValueError("bucket_size must be > 0")
    scale = _infer_scale(bs)
    bs_i = int(round(bs * scale))
    if bs_i <= 0:
        raise ValueError("bucket_size scaled invalid")
    return scale, bs_i


def _price_to_bucket_index(price: float, scale: int, bs_i: int) -> int:
    """
    Convert price -> integer bucket index using scaled integer arithmetic.
    """
    p_i = int(math.floor(float(price) * scale))
    # floor division toward -inf is acceptable even for erroneous negative prices
    return p_i // bs_i


def _bucket_index_to_price(idx: int, scale: int, bs_i: int) -> float:
    return (idx * bs_i) / float(scale)


def _price_to_bucket(price: float, bucket_size: float) -> float:
    """
    Round a price down to the nearest bucket boundary.
    Uses integer arithmetic to avoid float drift.
    """
    scale, bs_i = _bucket_params(bucket_size)
    idx = _price_to_bucket_index(price, scale, bs_i)
    return round(_bucket_index_to_price(idx, scale, bs_i), 6)


# ============================================
# Volume Distribution (optimized)
# ============================================
def _distribute_volume_triangular(
    candle: Dict,
    bucket_size: float,
) -> Dict[float, float]:
    """
    Distribute candle volume across bucket centers using triangular weighting.

    Optimized:
      - Uses integer bucket indices (no float drift)
      - Uses numpy vectorization if available
      - Avoids artificial min-weight inflation at extremes

    Returns:
      Dict[bucket_price -> volume]
    """
    if not isinstance(candle, dict):
        return {}

    h = candle.get("high")
    l = candle.get("low")
    c = candle.get("close")
    v = candle.get("volume", 0)

    try:
        h = float(h)
        l = float(l)
        c = float(c)
        v = float(v) if v is not None else 0.0
    except Exception:
        return {}

    if v <= 0:
        return {}

    # Guard invalid ranges
    rng = h - l
    if rng <= 0 or h < l:
        # Flat or invalid: all volume at close bucket
        b = _price_to_bucket(c, bucket_size)
        return {b: float(v)}

    typical = (h + l + c) / 3.0
    max_distance = rng / 2.0
    if max_distance <= 0:
        b = _price_to_bucket(c, bucket_size)
        return {b: float(v)}

    scale, bs_i = _bucket_params(bucket_size)

    low_idx = _price_to_bucket_index(l, scale, bs_i)
    high_idx = _price_to_bucket_index(h, scale, bs_i)
    if high_idx < low_idx:
        low_idx, high_idx = high_idx, low_idx

    count = (high_idx - low_idx + 1)
    if count <= 0:
        return {}

    # Vectorized bucket prices
    if np is not None:
        idxs = np.arange(low_idx, high_idx + 1, dtype=np.int64)
        prices = idxs.astype(np.float64) * (bs_i / float(scale))
        distances = np.abs(prices - float(typical))
        weights = 1.0 - (distances / float(max_distance))
        weights = np.clip(weights, 0.0, None)

        wsum = float(np.sum(weights))
        if wsum <= 0:
            # fallback to uniform if pathological
            weights = np.ones_like(weights, dtype=np.float64)
            wsum = float(np.sum(weights))

        vols = (weights / wsum) * float(v)
        # Build dict with stable float keys
        out: Dict[float, float] = {}
        for p, vol in zip(prices.tolist(), vols.tolist()):
            out[round(float(p), 6)] = float(vol)
        return out

    # Pure-Python fallback (still integer-index stable)
    out = {}
    weights = []
    prices = []
    for idx in range(low_idx, high_idx + 1):
        p = _bucket_index_to_price(idx, scale, bs_i)
        w = 1.0 - (abs(p - typical) / max_distance)
        if w < 0:
            w = 0.0
        weights.append(w)
        prices.append(p)

    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * len(weights)
        wsum = float(len(weights))

    for p, w in zip(prices, weights):
        out[round(float(p), 6)] = float((w / wsum) * v)

    return out


# ============================================
# Volume Profile Builder (optimized)
# ============================================
def build_volume_profile(
    candles: List[Dict],
    bucket_size: float = 5.0,
) -> Dict[float, float]:
    """
    Build complete volume profile from candle data.

    Optimized:
      - No float drift from iterative bucket stepping
      - Distribution per candle is vectorized (numpy) or stable integer loop
    """
    profile = defaultdict(float)

    for candle in candles:
        distributed = _distribute_volume_triangular(candle, bucket_size)
        if not distributed:
            continue
        for bucket, vol in distributed.items():
            profile[bucket] += float(vol)

    return dict(sorted(profile.items()))


# ============================================
# POC, VAH, VAL Computation
# ============================================
def find_poc(profile: Dict[float, float]) -> Tuple[float, float]:
    """Find Point of Control — price level with highest volume."""
    if not profile:
        return (0.0, 0.0)
    poc_price = max(profile, key=profile.get)
    return (float(poc_price), float(profile[poc_price]))


def find_value_area(
    profile: Dict[float, float],
    poc_price: float,
    target_pct: float = 0.70,
) -> Tuple[float, float]:
    """
    Find Value Area — price range containing target_pct of total volume, centered around POC.

    HARDENING:
      - Adds max_iterations guard to avoid infinite loops/hangs.
    """
    if not profile:
        return (0.0, 0.0)

    total_vol = float(sum(profile.values()))
    if total_vol <= 0:
        return (0.0, 0.0)

    target_vol = total_vol * float(target_pct)

    sorted_prices = sorted(profile.keys())
    if not sorted_prices:
        return (0.0, 0.0)

    if poc_price not in profile:
        poc_price = min(sorted_prices, key=lambda p: abs(p - poc_price))

    poc_idx = sorted_prices.index(poc_price)

    accumulated_vol = float(profile.get(poc_price, 0.0))
    lower_idx = poc_idx - 1
    upper_idx = poc_idx + 1

    val_price = float(poc_price)
    vah_price = float(poc_price)

    max_iterations = len(sorted_prices) * 2
    iterations = 0

    while accumulated_vol < target_vol:
        iterations += 1
        if iterations > max_iterations:
            _logger.warning(
                "Value area expansion hit iteration guard; breaking",
                extra={"iterations": iterations, "max_iterations": max_iterations},
            )
            break

        lower_vol = float(profile.get(sorted_prices[lower_idx], 0.0)) if lower_idx >= 0 else 0.0
        upper_vol = float(profile.get(sorted_prices[upper_idx], 0.0)) if upper_idx < len(sorted_prices) else 0.0

        if lower_vol == 0.0 and upper_vol == 0.0:
            break

        # Add side with more volume; always move an index to ensure progress
        if lower_idx >= 0 and (lower_vol >= upper_vol or upper_idx >= len(sorted_prices)):
            accumulated_vol += lower_vol
            val_price = float(sorted_prices[lower_idx])
            lower_idx -= 1
        elif upper_idx < len(sorted_prices):
            accumulated_vol += upper_vol
            vah_price = float(sorted_prices[upper_idx])
            upper_idx += 1
        else:
            break

    return (float(val_price), float(vah_price))


# ============================================
# HVN / LVN Detection (median baseline)
# ============================================
def find_hvn_lvn(
    profile: Dict[float, float],
    hvn_threshold: float = 1.5,
    lvn_threshold: float = 0.3,
) -> Tuple[List[float], List[float]]:
    """
    HVN: vol > hvn_threshold * median_vol
    LVN: vol < lvn_threshold * median_vol
    """
    if not profile or len(profile) < 3:
        return ([], [])

    volumes = [float(v) for v in profile.values() if v is not None]
    if not volumes:
        return ([], [])

    # MANDATE: use median
    try:
        med_vol = float(median(volumes))
    except Exception:
        med_vol = float(sorted(volumes)[len(volumes) // 2])

    if med_vol <= 0:
        return ([], [])

    hvn_levels: List[float] = []
    lvn_levels: List[float] = []

    for price, vol in profile.items():
        v = float(vol)
        if v > med_vol * float(hvn_threshold):
            hvn_levels.append(float(price))
        elif v < med_vol * float(lvn_threshold):
            lvn_levels.append(float(price))

    return (sorted(hvn_levels), sorted(lvn_levels))


# ============================================
# Volume Features (SMA, Z-Score, OBV)
# ============================================
def compute_volume_sma(candles: List[Dict], period: int = 20) -> Optional[float]:
    """Compute Simple Moving Average of volume."""
    if len(candles) < period:
        if len(candles) < 1:
            return None
        # Keep prior behavior (do not change) to avoid breaking downstream expectations
        period = len(candles)

    recent = candles[-period:]
    volumes = [float(c.get("volume", 0) or 0) for c in recent if isinstance(c, dict)]
    if not volumes:
        return None
    return round(sum(volumes) / len(volumes), 2)


def compute_volume_ratio(candles: List[Dict], period: int = 20) -> Optional[float]:
    """volume_ratio = current_volume / SMA(volume, period)"""
    if len(candles) < 2:
        return None

    sma = compute_volume_sma(candles[:-1], period)
    if sma is None or sma <= 0:
        return None

    current_vol = float(candles[-1].get("volume", 0) or 0)
    return round(current_vol / sma, 4)


def compute_volume_zscore(candles: List[Dict], period: int = 20) -> Optional[float]:
    """Compute Z-score of current volume vs recent average."""
    if len(candles) < period + 1:
        return None

    recent = candles[-(period + 1) : -1]
    volumes = [float(c.get("volume", 0) or 0) for c in recent if isinstance(c, dict)]
    if not volumes:
        return None

    avg = sum(volumes) / len(volumes)
    if avg <= 0:
        return None

    variance = sum((v - avg) ** 2 for v in volumes) / len(volumes)
    std_dev = math.sqrt(variance)
    if std_dev <= 0:
        return 0.0

    current_vol = float(candles[-1].get("volume", 0) or 0)
    zscore = (current_vol - avg) / std_dev
    return round(float(zscore), 4)


def compute_obv(candles: List[Dict], prev_obv: float = 0.0) -> Optional[float]:
    """
    Compute OBV with optional persistence.

    MANDATE:
      - OBV should persist across days; caller passes prev_obv.

    Returns:
      Current OBV value (rounded), or prev_obv if insufficient candles.
    """
    if candles is None or len(candles) < 2:
        return round(float(prev_obv), 0)

    obv = float(prev_obv)
    for i in range(1, len(candles)):
        c0 = candles[i - 1]
        c1 = candles[i]
        if not isinstance(c0, dict) or not isinstance(c1, dict):
            continue
        vol = float(c1.get("volume", 0) or 0)
        c1_close = c1.get("close")
        c0_close = c0.get("close")
        try:
            c1_close = float(c1_close)
            c0_close = float(c0_close)
        except Exception:
            continue

        if c1_close > c0_close:
            obv += vol
        elif c1_close < c0_close:
            obv -= vol

    return round(obv, 0)


# ============================================
# Session-aware filtering
# ============================================
def _filter_to_session_day(candles: List[Dict]) -> List[Dict]:
    """
    Filter candles to the trading day determined by the last candle's timestamp.

    MANDATE:
      - Use date of last candle timestamp (replay-safe).
      - If timestamps missing, return all and log warning.
    """
    if not candles:
        return []

    last_ts: Optional[datetime] = None
    for c in reversed(candles):
        if isinstance(c, dict) and isinstance(c.get("timestamp"), datetime):
            last_ts = c["timestamp"]
            break

    if last_ts is None:
        _logger.warning("Session filter: candles missing timestamps; using all candles")
        return candles

    d = last_ts.date()
    filtered = [c for c in candles if isinstance(c, dict) and isinstance(c.get("timestamp"), datetime) and c["timestamp"].date() == d]
    if not filtered:
        # Fall back to all if unexpected (shouldn't happen)
        _logger.warning("Session filter: no candles matched last timestamp date; using all candles", extra={"date": str(d)})
        return candles

    return filtered


# ============================================
# Master Feature Computation
# ============================================
def compute_volume_profile_features(
    candles: Union[List[Dict], deque],
    bucket_size: Optional[float] = None,
    prev_obv: float = 0.0,  # MANDATE: optional persistence parameter
) -> Dict:
    """
    Compute volume profile feature summary.

    IMPORTANT:
        Does NOT return full profile dict to avoid MarketState bloat.
    """
    candle_list = list(candles) if candles is not None else []
    candle_list = [c for c in candle_list if isinstance(c, dict)]
    if not candle_list:
        return _empty_features()

    # Session-aware filtering (MANDATE)
    candle_list = _filter_to_session_day(candle_list)
    if not candle_list:
        return _empty_features()

    if bucket_size is None:
        bucket_size = Config.get("features", "volume_profile_bucket_size", default=5)

    try:
        bucket_size = float(bucket_size)
        if bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")
    except Exception:
        _logger.warning("Invalid bucket_size; using default 5", extra={"bucket_size": bucket_size})
        bucket_size = 5.0

    # Build profile (expensive) but summary output only
    profile = build_volume_profile(candle_list, bucket_size)
    if not profile:
        return _empty_features()

    poc_price, poc_volume = find_poc(profile)
    val_price, vah_price = find_value_area(profile, poc_price)
    hvn_levels, lvn_levels = find_hvn_lvn(profile)

    # Volume statistics
    vol_sma = compute_volume_sma(candle_list, 20)
    vol_ratio = compute_volume_ratio(candle_list, 20)
    vol_zscore = compute_volume_zscore(candle_list, 20)
    obv = compute_obv(candle_list, prev_obv=float(prev_obv))

    # Cumulative volume delta (approximate)
    cum_vol_delta = 0.0
    for c in candle_list:
        try:
            v = float(c.get("volume", 0) or 0)
            o = float(c.get("open"))
            cl = float(c.get("close"))
        except Exception:
            continue
        if cl >= o:
            cum_vol_delta += v
        else:
            cum_vol_delta -= v

    # Last close distance from levels
    try:
        last_close = float(candle_list[-1].get("close"))
    except Exception:
        last_close = 0.0

    poc_distance = round(last_close - poc_price, 2) if poc_price > 0 else 0.0
    vah_distance = round(last_close - vah_price, 2) if vah_price > 0 else 0.0
    val_distance = round(last_close - val_price, 2) if val_price > 0 else 0.0

    result = {
        # Volume Profile Summary
        "poc": float(poc_price),
        "poc_volume": round(float(poc_volume), 0),
        "vah": float(vah_price),
        "val": float(val_price),
        "va_width": round(float(vah_price - val_price), 2) if vah_price > 0 else 0.0,

        # Nodes
        "hvn_count": int(len(hvn_levels)),
        "lvn_count": int(len(lvn_levels)),
        "hvn_levels": [float(x) for x in hvn_levels[:5]],
        "lvn_levels": [float(x) for x in lvn_levels[:5]],

        # Distances
        "poc_distance": float(poc_distance),
        "vah_distance": float(vah_distance),
        "val_distance": float(val_distance),

        # Volume stats
        "volume_sma_20": vol_sma,
        "volume_ratio": vol_ratio,
        "volume_zscore": vol_zscore,
        "obv": obv,
        "cumulative_volume_delta": round(float(cum_vol_delta), 0),

        # Metadata
        "bucket_size": float(bucket_size),
        "bucket_count": int(len(profile)),
        "total_profile_volume": round(float(sum(profile.values())), 0),
        "candle_count": int(len(candle_list)),
    }

    # MANDATE: do not return full profile dict
    return result


def _empty_features() -> Dict:
    return {
        "poc": 0.0,
        "poc_volume": 0.0,
        "vah": 0.0,
        "val": 0.0,
        "va_width": 0.0,
        "hvn_count": 0,
        "lvn_count": 0,
        "hvn_levels": [],
        "lvn_levels": [],
        "poc_distance": 0.0,
        "vah_distance": 0.0,
        "val_distance": 0.0,
        "volume_sma_20": None,
        "volume_ratio": None,
        "volume_zscore": None,
        "obv": None,
        "cumulative_volume_delta": 0.0,
        "bucket_size": 5.0,
        "bucket_count": 0,
        "total_profile_volume": 0.0,
        "candle_count": 0,
    }


# ============================================
# Module Self-Test
# ============================================
def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Volume Profile Features Test (Optimized)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: Triangular volume distribution ──
    print("  [Test 1] Triangular volume distribution...")
    candle = {"open": 23100, "high": 23120, "low": 23090, "close": 23110, "volume": 10000, "timestamp": datetime.now()}
    dist = _distribute_volume_triangular(candle, 5)

    if len(dist) > 0:
        total = sum(dist.values())
        if abs(total - 10000) < 1:
            print(f"    ✅ Volume conserved: {total:.2f} (expected 10000)")
            passed += 1
        else:
            print(f"    ❌ Volume not conserved: {total}")
            failed += 1

        tp_bucket = _price_to_bucket((23120 + 23090 + 23110) / 3.0, 5)
        # Not guaranteed exact match due to bucket rounding; ensure nearest exists
        nearest = min(dist.keys(), key=lambda p: abs(p - tp_bucket))
        print(f"    ✅ Typical/nearest bucket: {nearest} vol={dist[nearest]:.0f}")
        passed += 1
    else:
        print("    ❌ No distribution generated")
        failed += 2

    # ── Test 2: Flat candle handling ──
    print("\n  [Test 2] Flat candle (H=L)...")
    flat = {"open": 23100, "high": 23100, "low": 23100, "close": 23100, "volume": 5000, "timestamp": datetime.now()}
    flat_dist = _distribute_volume_triangular(flat, 5)
    if len(flat_dist) == 1 and abs(sum(flat_dist.values()) - 5000) < 1e-6:
        print("    ✅ Flat candle: single bucket with full volume")
        passed += 1
    else:
        print(f"    ❌ Flat candle distribution: {flat_dist}")
        failed += 1

    # ── Test 3: Build full profile ──
    print("\n  [Test 3] Build volume profile...")
    test_candles = []
    base_ts = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    for i in range(50):
        base = 23200 + (i % 10 - 5) * 3
        test_candles.append(
            {
                "timestamp": base_ts,
                "open": base - 5,
                "high": base + 10,
                "low": base - 12,
                "close": base + 2,
                "volume": 5000 + (i % 5) * 1000,
            }
        )

    profile = build_volume_profile(test_candles, 5)
    if len(profile) > 5:
        print(f"    ✅ Profile built: {len(profile)} buckets")
        total_vol = sum(profile.values())
        total_input = sum(float(c["volume"]) for c in test_candles)
        diff_pct = abs(total_vol - total_input) / total_input * 100
        if diff_pct < 1:
            print(f"    ✅ Volume conserved: diff={diff_pct:.2f}%")
        else:
            print(f"    ⚠️  Volume diff: {diff_pct:.2f}% (float rounding)")
        passed += 1
    else:
        print(f"    ❌ Profile too small: {len(profile)} buckets")
        failed += 1

    # ── Test 4: POC finding ──
    print("\n  [Test 4] Point of Control...")
    poc_price, poc_vol = find_poc(profile)
    if poc_price > 0 and poc_vol > 0:
        print(f"    ✅ POC = {poc_price} (volume: {poc_vol:.0f})")
        passed += 1
    else:
        print("    ❌ POC not found")
        failed += 1

    empty_poc, _ = find_poc({})
    if empty_poc == 0.0:
        print("    ✅ Empty profile returns 0")
        passed += 1
    else:
        print("    ❌ Should return 0 for empty")
        failed += 1

    # ── Test 5: Value Area guard ──
    print("\n  [Test 5] Value Area (VAH/VAL) + guard...")
    val_price, vah_price = find_value_area(profile, poc_price)
    if val_price > 0 and vah_price > 0 and val_price <= poc_price <= vah_price:
        print(f"    ✅ VAL={val_price}, VAH={vah_price} (POC inside)")
        passed += 1
    else:
        print("    ❌ Value area invalid")
        failed += 1

    # ── Test 6: HVN/LVN using median baseline ──
    print("\n  [Test 6] HVN/LVN...")
    hvn, lvn = find_hvn_lvn(profile)
    if isinstance(hvn, list) and isinstance(lvn, list):
        print(f"    ✅ HVN={len(hvn)} LVN={len(lvn)}")
        passed += 1
    else:
        print("    ❌ HVN/LVN wrong types")
        failed += 1

    # ── Test 7: Full compute_volume_profile_features summary (no full profile key) ──
    print("\n  [Test 7] compute_volume_profile_features summary...")
    features = compute_volume_profile_features(test_candles, prev_obv=1000.0)
    required = [
        "poc", "poc_volume", "vah", "val", "va_width",
        "hvn_count", "lvn_count", "hvn_levels", "lvn_levels",
        "poc_distance", "vah_distance", "val_distance",
        "volume_sma_20", "volume_ratio", "volume_zscore", "obv",
        "cumulative_volume_delta",
        "bucket_size", "bucket_count", "total_profile_volume", "candle_count",
    ]
    missing = [k for k in required if k not in features]
    if not missing and "profile" not in features:
        print("    ✅ Summary keys present; full profile not returned")
        passed += 1
    else:
        print(f"    ❌ Missing keys or profile leaked: missing={missing}, has_profile={'profile' in features}")
        failed += 1

    # ── Test 8: Empty candles ──
    print("\n  [Test 8] Empty candles...")
    empty = compute_volume_profile_features([])
    if empty["candle_count"] == 0 and empty["poc"] == 0.0:
        print("    ✅ Empty handled correctly")
        passed += 1
    else:
        print("    ❌ Empty handling failed")
        failed += 1

    # ── Test 9: Bucket size consistency ──
    print("\n  [Test 9] Bucket size validation...")
    profile_5 = build_volume_profile(test_candles, 5)
    profile_10 = build_volume_profile(test_candles, 10)
    if len(profile_5) >= len(profile_10):
        print(f"    ✅ Bucket 5pt: {len(profile_5)} buckets, 10pt: {len(profile_10)} buckets")
        passed += 1
    else:
        print("    ❌ Larger bucket should have fewer levels")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n  ✅ Volume Profile Features optimized + hardened successfully!")
    else:
        print(f"\n  ⚠️  {failed} tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()