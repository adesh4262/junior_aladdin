# src/features/key_levels.py
"""
Junior Aladdin - Key Levels (Layer 2H) — Institutional Grade
============================================================
PURPOSE:
    Compute key levels used by strategies and scoring:
      - PDH/PDL/PDC (previous day)
      - OR (Opening Range)
      - IB (Initial Balance)
      - Swing S/R zones
      - Pre-market highs/lows (if provided)
      - Overnight highs/lows (if provided)
      - Options OI walls as levels (if provided)
      - Gap day detection + level confidence adjustments
      - Session-aware expiry timestamp

ABSOLUTE SAFETY RULES:
- Missing levels MUST be None, never 0.0.
- Validity flags MUST be present and propagated.
- Time windows MUST parse timestamps robustly, or log warnings.
- OR/IB are session-specific; if candles span multiple days, OR/IB are invalidated.

SINGLE SOURCE OF TRUTH:
- Candle patterns are provided by src.features.candle_patterns.detect_candle_patterns
  (no duplicate pattern logic here).

PUBLIC API:
- compute_previous_day_levels(prev_candles)
- compute_opening_range(candles, or_start, or_end)
- compute_initial_balance(candles, ib_start, ib_end)
- detect_sr_zones(...)
- compute_key_level_features(today_candles, previous_day_candles=None, swing_highs=None, swing_lows=None,
                            options_features=None, volume_profile=None, pre_market_candles=None,
                            overnight_levels=None, session_phase=None, gift_levels=None,
                            or_start=None, or_end=None, ib_start=None, ib_end=None)
- compute_key_levels(...) alias for backward compatibility
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.features.candle_patterns import detect_candle_patterns

_logger = setup_logger("key_levels")

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------
# Robust timestamp parsing
# ---------------------------------------------------------------------
def _extract_datetime(ts: Any) -> Optional[datetime]:
    """
    Convert multiple timestamp types to datetime:
      - datetime
      - pandas.Timestamp-like: has to_pydatetime()
      - ISO string: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS"
      - fallback: pandas.to_datetime if pandas available
    """
    if ts is None:
        return None

    try:
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
    except Exception:
        pass

    if isinstance(ts, datetime):
        # Ensure tz-aware in IST for consistent comparisons
        if ts.tzinfo is None:
            return ts.replace(tzinfo=IST)
        return ts.astimezone(IST)

    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        s2 = s.replace("T", " ")
        try:
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=IST)
            else:
                dt = dt.astimezone(IST)
            return dt
        except Exception:
            pass

        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(s2, fmt).replace(tzinfo=IST)
                return dt
            except Exception:
                continue

        # pandas fallback if available
        try:
            import pandas as pd  # type: ignore
            dt = pd.to_datetime(s2, errors="coerce")
            if dt is not None and str(dt) != "NaT":
                py = dt.to_pydatetime()
                if py.tzinfo is None:
                    py = py.replace(tzinfo=IST)
                else:
                    py = py.astimezone(IST)
                return py
        except Exception as e:
            _logger.debug("Timestamp parse failed (pandas fallback unavailable/failed)", extra={"error": str(e)})

    return None


def _extract_time(ts: Any) -> Optional[time]:
    dt = _extract_datetime(ts)
    return dt.time() if dt is not None else None


def _filter_by_time(candles: List[Dict[str, Any]], start: time, end: time) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        t = _extract_time(c.get("timestamp"))
        if t is None:
            continue
        if start <= t <= end:
            out.append(c)
    return out


def _infer_session_date_from_candles(candles: List[Dict[str, Any]]) -> Optional[datetime.date]:
    for c in reversed(candles):
        if not isinstance(c, dict):
            continue
        dt = _extract_datetime(c.get("timestamp"))
        if dt is not None:
            return dt.date()
    return None


def _candles_span_multiple_days(candles: List[Dict[str, Any]]) -> bool:
    days = set()
    for c in candles:
        if not isinstance(c, dict):
            continue
        dt = _extract_datetime(c.get("timestamp"))
        if dt is None:
            continue
        days.add(dt.date())
        if len(days) > 1:
            return True
    return False


# ---------------------------------------------------------------------
# Previous day levels
# ---------------------------------------------------------------------
def compute_previous_day_levels(prev_candles: List[Dict]) -> Dict:
    if not prev_candles:
        _logger.warning("Previous day candles missing; PDH/PDL/PDC unavailable")
        return {"pdh": None, "pdl": None, "pdc": None, "pdh_valid": False}

    try:
        highs = [float(c["high"]) for c in prev_candles if isinstance(c, dict) and c.get("high") is not None]
        lows = [float(c["low"]) for c in prev_candles if isinstance(c, dict) and c.get("low") is not None]
        pdc = float(prev_candles[-1]["close"])
        if not highs or not lows:
            raise ValueError("missing highs/lows")
        return {"pdh": round(max(highs), 2), "pdl": round(min(lows), 2), "pdc": round(pdc, 2), "pdh_valid": True}
    except Exception as e:
        _logger.warning("Previous day level computation failed", extra={"error": str(e)})
        return {"pdh": None, "pdl": None, "pdc": None, "pdh_valid": False}


# ---------------------------------------------------------------------
# Dynamic time windows from Config (with overrides)
# ---------------------------------------------------------------------
def _parse_hhmm(s: str, default: time) -> time:
    try:
        parts = s.strip().split(":")
        hh = int(parts[0])
        mm = int(parts[1])
        return time(hh, mm)
    except Exception:
        return default


def _get_or_window(overrides: Optional[Dict[str, str]] = None) -> Tuple[time, time]:
    # Config preferred: sessions.or_formation.start/end
    s = Config.get("sessions", "or_formation", "start", default="09:16")
    e = Config.get("sessions", "or_formation", "end", default="09:30")
    if overrides:
        s = overrides.get("or_start", s)
        e = overrides.get("or_end", e)
    return _parse_hhmm(str(s), time(9, 16)), _parse_hhmm(str(e), time(9, 30))


def _get_ib_window(overrides: Optional[Dict[str, str]] = None) -> Tuple[time, time]:
    s = Config.get("sessions", "initial_balance", "start", default="09:30")
    e = Config.get("sessions", "initial_balance", "end", default="10:15")
    if overrides:
        s = overrides.get("ib_start", s)
        e = overrides.get("ib_end", e)
    return _parse_hhmm(str(s), time(9, 30)), _parse_hhmm(str(e), time(10, 15))


# ---------------------------------------------------------------------
# OR / IB computation (None on missing)
# ---------------------------------------------------------------------
def compute_opening_range(
    candles: List[Dict],
    or_start: time = time(9, 16),
    or_end: time = time(9, 30),
) -> Dict:
    or_candles = _filter_by_time(candles, or_start, or_end)
    if not or_candles:
        _logger.warning(
            "OR window empty; OR unavailable",
            extra={"or_start": str(or_start), "or_end": str(or_end), "total_candles": len(candles)},
        )
        return {"or_high": None, "or_low": None, "or_width": None, "or_valid": False}

    try:
        highs = [float(c["high"]) for c in or_candles if isinstance(c, dict) and c.get("high") is not None]
        lows = [float(c["low"]) for c in or_candles if isinstance(c, dict) and c.get("low") is not None]
        if not highs or not lows:
            raise ValueError("missing highs/lows")
        or_high = max(highs)
        or_low = min(lows)
        return {"or_high": round(or_high, 2), "or_low": round(or_low, 2), "or_width": round(or_high - or_low, 2), "or_valid": True}
    except Exception as e:
        _logger.warning("OR computation failed", extra={"error": str(e)})
        return {"or_high": None, "or_low": None, "or_width": None, "or_valid": False}


def compute_initial_balance(
    candles: List[Dict],
    ib_start: time = time(9, 30),
    ib_end: time = time(10, 15),
) -> Dict:
    ib_candles = _filter_by_time(candles, ib_start, ib_end)
    if not ib_candles:
        _logger.warning(
            "IB window empty; IB unavailable",
            extra={"ib_start": str(ib_start), "ib_end": str(ib_end), "total_candles": len(candles)},
        )
        return {"ib_high": None, "ib_low": None, "ib_width": None, "ib_direction": None, "ib_valid": False}

    try:
        highs = [float(c["high"]) for c in ib_candles if isinstance(c, dict) and c.get("high") is not None]
        lows = [float(c["low"]) for c in ib_candles if isinstance(c, dict) and c.get("low") is not None]
        if not highs or not lows:
            raise ValueError("missing highs/lows")
        ib_high = max(highs)
        ib_low = min(lows)
        ib_width = round(ib_high - ib_low, 2)

        first_open = float(ib_candles[0]["open"])
        last_close = float(ib_candles[-1]["close"])
        ib_direction = 1 if last_close > first_open else (-1 if last_close < first_open else 0)

        return {"ib_high": round(ib_high, 2), "ib_low": round(ib_low, 2), "ib_width": ib_width, "ib_direction": ib_direction, "ib_valid": True}
    except Exception as e:
        _logger.warning("IB computation failed", extra={"error": str(e)})
        return {"ib_high": None, "ib_low": None, "ib_width": None, "ib_direction": None, "ib_valid": False}


# ---------------------------------------------------------------------
# Gap day detection and confidence
# ---------------------------------------------------------------------
def _gap_day_flags(today_open: Optional[float], prev_close: Optional[float], threshold_pct: float = 0.015) -> Tuple[bool, Optional[float]]:
    if today_open is None or prev_close is None or prev_close <= 0:
        return False, None
    gap_pct = (today_open - prev_close) / prev_close
    return abs(gap_pct) > threshold_pct, round(gap_pct * 100.0, 3)


# ---------------------------------------------------------------------
# SR zones with optional volume profile weighting
# ---------------------------------------------------------------------
def detect_sr_zones(
    swing_highs: List[float],
    swing_lows: List[float],
    cluster_distance: float = 15.0,
    volume_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Detect SR zones from swing clusters.

    Enhancement (best-effort):
      - If volume_profile is provided (e.g., {"poc":..., "vah":..., "val":..., "hvn_levels":[...]})
        boost strength when zone is near these high-importance areas.
    """
    all_levels = []
    for p in swing_highs or []:
        try:
            all_levels.append({"price": float(p), "type": "resistance"})
        except Exception:
            continue
    for p in swing_lows or []:
        try:
            all_levels.append({"price": float(p), "type": "support"})
        except Exception:
            continue

    if len(all_levels) < 2:
        return [{"level": l["price"], "strength": 1, "type": l["type"]} for l in all_levels]

    all_levels.sort(key=lambda x: x["price"])

    zones = []
    used = set()

    # Volume profile weighting anchors
    poc = None
    vah = None
    val = None
    hvn = []
    if isinstance(volume_profile, dict):
        poc = volume_profile.get("poc")
        vah = volume_profile.get("vah")
        val = volume_profile.get("val")
        hvn = volume_profile.get("hvn_levels", []) or []

    def _boost(level: float) -> int:
        if not isinstance(volume_profile, dict):
            _logger.debug("SR zones: volume_profile not provided; equal weighting")
            return 0
        boost = 0
        try:
            if poc is not None and abs(level - float(poc)) <= cluster_distance:
                boost += 2
            if vah is not None and abs(level - float(vah)) <= cluster_distance:
                boost += 1
            if val is not None and abs(level - float(val)) <= cluster_distance:
                boost += 1
            for h in hvn[:5]:
                try:
                    if abs(level - float(h)) <= cluster_distance:
                        boost += 1
                        break
                except Exception:
                    continue
        except Exception:
            return 0
        return boost

    for i, l1 in enumerate(all_levels):
        if i in used:
            continue

        cluster = [l1]
        cluster_indices = {i}

        for j in range(i + 1, len(all_levels)):
            if j in used:
                continue
            if abs(all_levels[j]["price"] - l1["price"]) <= cluster_distance:
                cluster.append(all_levels[j])
                cluster_indices.add(j)

        used.update(cluster_indices)
        avg_price = sum(c["price"] for c in cluster) / len(cluster)

        res_count = sum(1 for c in cluster if c["type"] == "resistance")
        sup_count = len(cluster) - res_count
        zone_type = "resistance" if res_count >= sup_count else "support"

        strength = len(cluster) + _boost(avg_price)
        zones.append({"level": round(avg_price, 2), "strength": int(strength), "type": zone_type})

    zones.sort(key=lambda z: z["strength"], reverse=True)
    return zones


# ---------------------------------------------------------------------
# Options wall integration
# ---------------------------------------------------------------------
def _extract_oi_levels(options_features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(options_features, dict):
        return {
            "oi_resistance": None,
            "oi_support": None,
            "oi_resistance_strength": None,
            "oi_support_strength": None,
        }

    ce_strike = options_features.get("highest_ce_oi_strike")
    pe_strike = options_features.get("highest_pe_oi_strike")
    ce_oi = options_features.get("highest_ce_oi")
    pe_oi = options_features.get("highest_pe_oi")

    def _strength(oi: Any) -> Optional[int]:
        try:
            v = float(oi)
            if v <= 0:
                return None
            # log scaling to 1-5
            import math
            s = int(max(1, min(5, round(math.log10(v) - 3))))
            return s
        except Exception:
            return None

    try:
        oi_res = int(ce_strike) if ce_strike is not None else None
    except Exception:
        oi_res = None
    try:
        oi_sup = int(pe_strike) if pe_strike is not None else None
    except Exception:
        oi_sup = None

    return {
        "oi_resistance": oi_res,
        "oi_support": oi_sup,
        "oi_resistance_strength": _strength(ce_oi),
        "oi_support_strength": _strength(pe_oi),
    }


# ---------------------------------------------------------------------
# Confluence zones (multiple levels within proximity)
# ---------------------------------------------------------------------
def _compute_confluence(level_items: List[Tuple[str, Optional[float]]], proximity_pts: float = 10.0) -> List[Dict[str, Any]]:
    vals = [(name, float(v)) for name, v in level_items if v is not None]
    vals.sort(key=lambda x: x[1])
    zones: List[Dict[str, Any]] = []
    if not vals:
        return zones

    i = 0
    while i < len(vals):
        base = vals[i][1]
        cluster = [vals[i]]
        j = i + 1
        while j < len(vals) and abs(vals[j][1] - base) <= proximity_pts:
            cluster.append(vals[j])
            j += 1

        if len(cluster) >= 2:
            level = sum(v for _n, v in cluster) / len(cluster)
            zones.append({
                "zone_level": round(level, 2),
                "members": [n for n, _v in cluster],
                "count": len(cluster),
            })
        i = j

    zones.sort(key=lambda z: z["count"], reverse=True)
    return zones


# ---------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------
def compute_key_level_features(
    today_candles: Union[List[Dict], deque],
    previous_day_candles: Optional[List[Dict]] = None,
    swing_highs: Optional[List[float]] = None,
    swing_lows: Optional[List[float]] = None,
    options_features: Optional[Dict[str, Any]] = None,
    volume_profile: Optional[Dict[str, Any]] = None,
    pre_market_candles: Optional[Union[List[Dict], deque]] = None,
    overnight_levels: Optional[Dict[str, Any]] = None,
    session_phase: Optional[str] = None,
    session_time_overrides: Optional[Dict[str, str]] = None,
) -> Dict:
    candle_list = list(today_candles) if today_candles is not None else []
    candle_list = [c for c in candle_list if isinstance(c, dict)]
    if not candle_list:
        return _empty_features()

    # Session date + multi-day guard
    session_date = _infer_session_date_from_candles(candle_list)
    multi_day = _candles_span_multiple_days(candle_list)
    if multi_day:
        _logger.warning("today_candles span multiple days; OR/IB invalidated", extra={"session_date": str(session_date)})

    # Dynamic windows
    or_start, or_end = _get_or_window(session_time_overrides)
    ib_start, ib_end = _get_ib_window(session_time_overrides)

    # Current price
    try:
        current_price = float(candle_list[-1].get("close"))
    except Exception:
        current_price = None
        _logger.warning("Current close missing; distances may be None")

    # Previous day levels
    prev_levels = compute_previous_day_levels(previous_day_candles or [])
    pdh_valid = bool(prev_levels.get("pdh_valid", False))

    # Gap day detection
    today_open = None
    try:
        today_open = float(candle_list[0].get("open"))
    except Exception:
        today_open = None
    gap_day, gap_pct = _gap_day_flags(today_open, prev_levels.get("pdc"))
    or_confidence = 1.0
    ib_confidence = 1.0
    if gap_day:
        or_confidence *= 0.5
        ib_confidence *= 0.5

    # OR / IB
    or_data = compute_opening_range(candle_list, or_start=or_start, or_end=or_end)
    ib_data = compute_initial_balance(candle_list, ib_start=ib_start, ib_end=ib_end)
    or_valid = bool(or_data.get("or_valid", False)) and (not multi_day)
    ib_valid = bool(ib_data.get("ib_valid", False)) and (not multi_day)
    if multi_day:
        or_data = {"or_high": None, "or_low": None, "or_width": None, "or_valid": False}
        ib_data = {"ib_high": None, "ib_low": None, "ib_width": None, "ib_direction": None, "ib_valid": False}

    # SR zones (volume weighted best-effort)
    cluster_dist = float(Config.get("features", "sr_cluster_distance", default=15) or 15)
    sr_zones = detect_sr_zones(swing_highs or [], swing_lows or [], cluster_distance=cluster_dist, volume_profile=volume_profile)
    sr_zones_available = len(sr_zones) > 0

    # Options walls as levels
    oi_levels = _extract_oi_levels(options_features)

    # Pre-market levels
    pre_market_high = None
    pre_market_low = None
    pre_market_valid = False
    if pre_market_candles is not None:
        pm = [c for c in list(pre_market_candles) if isinstance(c, dict)]
        if pm:
            try:
                pre_market_high = round(max(float(c["high"]) for c in pm if c.get("high") is not None), 2)
                pre_market_low = round(min(float(c["low"]) for c in pm if c.get("low") is not None), 2)
                pre_market_valid = True
            except Exception:
                pre_market_high = None
                pre_market_low = None
                pre_market_valid = False

    # Overnight levels (external)
    overnight_high = None
    overnight_low = None
    overnight_valid = False
    if isinstance(overnight_levels, dict):
        try:
            overnight_high = overnight_levels.get("overnight_high")
            overnight_low = overnight_levels.get("overnight_low")
            if overnight_high is not None and overnight_low is not None:
                overnight_high = float(overnight_high)
                overnight_low = float(overnight_low)
                overnight_valid = True
        except Exception:
            overnight_valid = False

    # Candle patterns (single source of truth)
    patterns = detect_candle_patterns(candle_list)
    strongest = max(patterns, key=lambda p: p.get("confidence", 0)) if patterns else None

    # Level distances
    levels_for_dist = {
        "pdh": prev_levels.get("pdh"),
        "pdl": prev_levels.get("pdl"),
        "pdc": prev_levels.get("pdc"),
        "or_high": or_data.get("or_high") if or_valid else None,
        "or_low": or_data.get("or_low") if or_valid else None,
        "ib_high": ib_data.get("ib_high") if ib_valid else None,
        "ib_low": ib_data.get("ib_low") if ib_valid else None,
        "sr_zones": sr_zones,
    }
    distances = compute_level_distances(current_price, levels_for_dist)

    # Price vs OR/IB
    def _pos(price: Optional[float], hi: Optional[float], lo: Optional[float]) -> str:
        if price is None or hi is None or lo is None:
            return "UNKNOWN"
        if price > hi:
            return "ABOVE"
        if price < lo:
            return "BELOW"
        return "INSIDE"

    price_vs_or = _pos(current_price, or_data.get("or_high"), or_data.get("or_low")) if or_valid else "UNKNOWN"
    price_vs_ib = _pos(current_price, ib_data.get("ib_high"), ib_data.get("ib_low")) if ib_valid else "UNKNOWN"

    # Levels expiry at end of session (based on session_date + market_close)
    market_close = str(Config.get("market", "market_close", default="15:30"))
    close_t = _parse_hhmm(market_close, time(15, 30))
    levels_expire_at = None
    if session_date is not None:
        levels_expire_at = datetime.combine(session_date, close_t).replace(tzinfo=IST)

    # Confluence zones across level types (raw ingredients)
    confluence_items = [
        ("PDH", prev_levels.get("pdh")),
        ("PDL", prev_levels.get("pdl")),
        ("PDC", prev_levels.get("pdc")),
        ("ORH", or_data.get("or_high") if or_valid else None),
        ("ORL", or_data.get("or_low") if or_valid else None),
        ("IBH", ib_data.get("ib_high") if ib_valid else None),
        ("IBL", ib_data.get("ib_low") if ib_valid else None),
        ("POC", volume_profile.get("poc") if isinstance(volume_profile, dict) else None),
        ("VAH", volume_profile.get("vah") if isinstance(volume_profile, dict) else None),
        ("VAL", volume_profile.get("val") if isinstance(volume_profile, dict) else None),
        ("OI_RES", oi_levels.get("oi_resistance")),
        ("OI_SUP", oi_levels.get("oi_support")),
        ("PMH", pre_market_high if pre_market_valid else None),
        ("PML", pre_market_low if pre_market_valid else None),
        ("ONH", overnight_high if overnight_valid else None),
        ("ONL", overnight_low if overnight_valid else None),
    ]
    confluence_zones = _compute_confluence(confluence_items, proximity_pts=10.0)

    data_quality = {
        "pdh_available": bool(pdh_valid),
        "or_available": bool(or_valid),
        "ib_available": bool(ib_valid),
        "sr_zones_available": bool(sr_zones_available),
        "pre_market_available": bool(pre_market_valid),
        "overnight_available": bool(overnight_valid),
        "options_oi_levels_available": bool(oi_levels.get("oi_resistance") is not None or oi_levels.get("oi_support") is not None),
        "multi_day_invalidated": bool(multi_day),
    }

    return {
        # Previous day
        "pdh": prev_levels.get("pdh"),
        "pdl": prev_levels.get("pdl"),
        "pdc": prev_levels.get("pdc"),
        "pdh_valid": bool(pdh_valid),

        # Gap day flags
        "gap_day": bool(gap_day),
        "gap_pct": gap_pct,

        # OR
        "or_high": or_data.get("or_high") if or_valid else None,
        "or_low": or_data.get("or_low") if or_valid else None,
        "or_width": or_data.get("or_width") if or_valid else None,
        "or_valid": bool(or_valid),
        "or_confidence": float(or_confidence),

        # IB
        "ib_high": ib_data.get("ib_high") if ib_valid else None,
        "ib_low": ib_data.get("ib_low") if ib_valid else None,
        "ib_width": ib_data.get("ib_width") if ib_valid else None,
        "ib_direction": ib_data.get("ib_direction") if ib_valid else None,
        "ib_valid": bool(ib_valid),
        "ib_confidence": float(ib_confidence),

        # Session metadata
        "session_date": str(session_date) if session_date else None,
        "levels_expire_at": levels_expire_at.isoformat() if levels_expire_at else None,
        "session_phase": session_phase,

        # Pre-market / overnight
        "pre_market_high": pre_market_high if pre_market_valid else None,
        "pre_market_low": pre_market_low if pre_market_valid else None,
        "pre_market_valid": bool(pre_market_valid),
        "overnight_high": overnight_high if overnight_valid else None,
        "overnight_low": overnight_low if overnight_valid else None,
        "overnight_valid": bool(overnight_valid),

        # Options OI walls as levels
        **oi_levels,

        # SR zones
        "sr_zone_count": len(sr_zones),
        "sr_zones": sr_zones[:5],

        # Distances
        **distances,

        # Price position
        "price_vs_or": price_vs_or,
        "price_vs_ib": price_vs_ib,

        # Confluence zones
        "confluence_zones": confluence_zones[:5],

        # Candle patterns (from dedicated module)
        "patterns_detected": len(patterns),
        "patterns": patterns,
        "has_bullish_pattern": any(p.get("direction") == "BULLISH" for p in patterns),
        "has_bearish_pattern": any(p.get("direction") == "BEARISH" for p in patterns),
        "strongest_pattern": strongest.get("pattern") if strongest else "NONE",
        "strongest_pattern_confidence": strongest.get("confidence", 0) if strongest else 0,

        # Data quality
        "data_quality": data_quality,

        # Metadata
        "candle_count": len(candle_list),
        "current_price": current_price,
    }


def compute_level_distances(current_price: Optional[float], levels: Dict) -> Dict:
    """
    Distance-to-levels with None-safe output.
    """
    distances: Dict[str, Any] = {}
    level_keys = ["pdh", "pdl", "pdc", "or_high", "or_low", "ib_high", "ib_low"]

    for key in level_keys:
        val = levels.get(key)
        if current_price is None or val is None:
            distances[f"{key}_distance"] = None
        else:
            try:
                distances[f"{key}_distance"] = round(float(current_price) - float(val), 2)
            except Exception:
                distances[f"{key}_distance"] = None

    sr_zones = levels.get("sr_zones", [])
    if sr_zones and current_price is not None:
        try:
            nearest = min(sr_zones, key=lambda z: abs(float(z["level"]) - float(current_price)))
            distances["nearest_sr_level"] = float(nearest["level"])
            distances["nearest_sr_distance"] = round(float(current_price) - float(nearest["level"]), 2)
            distances["nearest_sr_type"] = nearest.get("type", "none")
            distances["nearest_sr_strength"] = int(nearest.get("strength", 0))
        except Exception:
            distances["nearest_sr_level"] = None
            distances["nearest_sr_distance"] = None
            distances["nearest_sr_type"] = "none"
            distances["nearest_sr_strength"] = 0
    else:
        distances["nearest_sr_level"] = None
        distances["nearest_sr_distance"] = None
        distances["nearest_sr_type"] = "none"
        distances["nearest_sr_strength"] = 0

    return distances


# Backward-compatible alias
def compute_key_levels(*args, **kwargs) -> Dict:
    return compute_key_level_features(*args, **kwargs)


def _empty_features() -> Dict:
    return {
        "pdh": None, "pdl": None, "pdc": None, "pdh_valid": False,
        "gap_day": False, "gap_pct": None,
        "or_high": None, "or_low": None, "or_width": None, "or_valid": False, "or_confidence": 0.0,
        "ib_high": None, "ib_low": None, "ib_width": None, "ib_direction": None, "ib_valid": False, "ib_confidence": 0.0,
        "session_date": None, "levels_expire_at": None, "session_phase": None,
        "pre_market_high": None, "pre_market_low": None, "pre_market_valid": False,
        "overnight_high": None, "overnight_low": None, "overnight_valid": False,
        "oi_resistance": None, "oi_support": None, "oi_resistance_strength": None, "oi_support_strength": None,
        "sr_zone_count": 0, "sr_zones": [],
        "pdh_distance": None, "pdl_distance": None, "pdc_distance": None,
        "or_high_distance": None, "or_low_distance": None,
        "ib_high_distance": None, "ib_low_distance": None,
        "nearest_sr_level": None, "nearest_sr_distance": None, "nearest_sr_type": "none", "nearest_sr_strength": 0,
        "price_vs_or": "UNKNOWN", "price_vs_ib": "UNKNOWN",
        "confluence_zones": [],
        "patterns_detected": 0, "patterns": [],
        "has_bullish_pattern": False, "has_bearish_pattern": False,
        "strongest_pattern": "NONE", "strongest_pattern_confidence": 0,
        "data_quality": {
            "pdh_available": False,
            "or_available": False,
            "ib_available": False,
            "sr_zones_available": False,
            "pre_market_available": False,
            "overnight_available": False,
            "options_oi_levels_available": False,
            "multi_day_invalidated": False,
        },
        "candle_count": 0, "current_price": None,
    }


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Key Levels Test (Institutional)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # Test 1: Missing prev day => None + invalid
    print("  [Test 1] Previous day missing => None...")
    p = compute_previous_day_levels([])
    if p["pdh"] is None and p["pdh_valid"] is False:
        print("    ✅ PDH missing handled")
        passed += 1
    else:
        print(f"    ❌ {p}")
        failed += 1

    # Test 2: ISO timestamp parsing in OR filter
    print("\n  [Test 2] Timestamp parsing (ISO)...")
    candles = [
        {"timestamp": "2026-04-01 09:20:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        {"timestamp": "2026-04-01T09:25:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
    ]
    f = _filter_by_time(candles, time(9, 16), time(9, 30))
    if len(f) == 2:
        print("    ✅ ISO parse ok")
        passed += 1
    else:
        print(f"    ❌ filtered={len(f)}")
        failed += 1

    # Test 3: OR/IB empty => None + invalid
    print("\n  [Test 3] OR/IB empty => None...")
    outside = [{"timestamp": "2026-04-01 12:00:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5}]
    or_d = compute_opening_range(outside)
    ib_d = compute_initial_balance(outside)
    if or_d["or_valid"] is False and or_d["or_high"] is None and ib_d["ib_valid"] is False and ib_d["ib_high"] is None:
        print("    ✅ OR/IB empty handled")
        passed += 1
    else:
        print(f"    ❌ OR={or_d} IB={ib_d}")
        failed += 1

    # Test 4: Gap day detection
    print("\n  [Test 4] Gap day flag...")
    prev = [{"high": 100, "low": 90, "close": 100}, {"high": 101, "low": 95, "close": 100}]
    today = [{"timestamp": "2026-04-01 09:16:00", "open": 103, "high": 104, "low": 102, "close": 103.5}]
    feat = compute_key_level_features(today, previous_day_candles=prev)
    if feat["gap_day"] is True:
        print(f"    ✅ gap_day True gap_pct={feat['gap_pct']}")
        passed += 1
    else:
        print(f"    ❌ gap_day expected True: {feat['gap_day']}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n  ✅ Key Levels institutional test passed.")
    else:
        print(f"\n  ⚠️ {failed} tests failed.")


if __name__ == "__main__":
    _run_tests()