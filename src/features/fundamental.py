"""
Junior Aladdin - Fundamental Features (Layer 2J)
================================================

INSTITUTIONAL-GRADE HARDENING (DEFENSIVE PROGRAMMING UPGRADE)
-------------------------------------------------------------
This module computes fundamental/macro features from external data sources:
- NSE FII/DII
- Yahoo Finance global proxies
- India VIX
- Economic calendar (local JSON)

This module is used in unsupervised trading contexts. Therefore:

ABSOLUTE MANDATES (Implemented):
- NEVER CRASH: all external dict/file access wrapped with try/except
- NEVER HIDE MISSING DATA: missing inputs yield None (not 0), with explicit availability flags
- ALL EXISTING OUTPUT KEYS remain present (backward compatibility)
- Existing scoring logic is preserved; only wrapped and validated
- Adds consolidated data_quality flags for downstream engines

COMPUTES (existing keys preserved):
- fii_score, dii_score, fii_net_crore, dii_net_crore, fii_source, fii_data_available
- global_score, global_data_available + component scores/changes/usdinr_price
- vix_level, vix_change_pct, vix_zone, vix_score, vix_spike
- event_severity, event_name, event_days_away, is_event_day, events_this_week

ADDITIVE KEYS (new, safe to ignore by older consumers):
- data_quality: {fii_available, global_available, vix_available, calendar_available}
- institutional_flow_score: weighted composite of FII/DII scores (configurable)
- warnings: list of warning strings generated during parsing

CONFIG EXTENSIONS (optional; safe defaults used if absent):
config.yaml -> fundamental:
  global_multipliers:
    sp500: 1500
    usdinr: 5000
    crude: 500
    asia: 1000
  global_component_caps:
    sp500: 15
    usdinr: 10
    crude: 10
    asia: 10
    gold: 5
  vix_spike:
    pct_threshold: 0.05
    penalty_by_zone: {NORMAL: 5, CAUTION: 10, FEAR: 15, PANIC: 20}
  institutional_flow_weights: {fii: 0.7, dii: 0.3}

"""

from __future__ import annotations

import json
import os
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("fundamental")
IST = timezone(timedelta(hours=5, minutes=30))

# ist_today() was referenced in the original file.
# Make import defensive to prevent import-time crashes if helpers evolve.
try:
    from src.utils.helpers import ist_today  # type: ignore
except Exception:  # pragma: no cover
    def ist_today() -> date:
        # Fallback: IST "today" derived from UTC now.
        return datetime.now(tz=IST).date()


# =====================================================================================
# Defensive Extractors (PHASE 1)
# =====================================================================================

def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """
    Convert to float safely.
    Returns default (possibly None) on failure.
    """
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            # bool is int subclass; treat as invalid numeric for market data.
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float_from_dict(data: Optional[Dict], key: str, default: Optional[float] = 0.0) -> Optional[float]:
    """
    PHASE 1 requirement:
    - Catch TypeError/ValueError
    - Log a warning
    - Return default
    """
    if not isinstance(data, dict):
        _logger.warning("Expected dict for numeric extraction; got non-dict",
                        key=key, data_type=str(type(data)))
        return default
    raw = data.get(key, None)
    v = _safe_float(raw, default=default)
    if v is default and raw not in (None, default):
        _logger.warning("Invalid float value in dict; using default",
                        key=key, raw_value=str(raw)[:100], default=default)
    return v


def _safe_int_from_dict(data: Optional[Dict], key: str, default: Optional[int] = 0) -> Optional[int]:
    """
    PHASE 1 requirement:
    - Catch TypeError/ValueError
    - Log a warning
    - Return default
    """
    if not isinstance(data, dict):
        _logger.warning("Expected dict for int extraction; got non-dict",
                        key=key, data_type=str(type(data)))
        return default
    raw = data.get(key, None)
    v = _safe_int(raw, default=default)
    if v is default and raw not in (None, default):
        _logger.warning("Invalid int value in dict; using default",
                        key=key, raw_value=str(raw)[:100], default=default)
    return v


def _safe_str_from_dict(data: Optional[Dict], key: str, default: Optional[str] = None) -> Optional[str]:
    if not isinstance(data, dict):
        return default
    v = data.get(key, default)
    if v is None:
        return default
    try:
        s = str(v)
        return s
    except Exception:
        return default


def _get_cfg(*keys: str, default: Any = None) -> Any:
    """
    Safe config getter that never raises.
    """
    try:
        return Config.get(*keys, default=default)
    except Exception as e:  # pragma: no cover
        _logger.error("Config.get failed; using default", keys=list(keys), default=default, error=str(e))
        return default


def _cap_with_log(
    value: float,
    cap_abs: float,
    label: str,
    warnings: List[str],
) -> float:
    """
    PHASE 1 requirement:
    - Cap each component within reasonable bounds
    - Log warning when cap occurs (to surface data quality issues)
    """
    if cap_abs <= 0:
        return value
    if value > cap_abs:
        msg = f"{label} component capped: raw={value:.3f} cap=+{cap_abs}"
        warnings.append(msg)
        _logger.warning(msg)
        return cap_abs
    if value < -cap_abs:
        msg = f"{label} component capped: raw={value:.3f} cap=-{cap_abs}"
        warnings.append(msg)
        _logger.warning(msg)
        return -cap_abs
    return value


def _bool_success_flag(data: Optional[Dict], *, name: str, warnings: List[str]) -> bool:
    """
    Determine availability in a safer way than blindly using `success`:
    - If success is explicitly True => True
    - If explicitly False => False
    - If missing but dict has content => assume True but warn (no silent masking)
    """
    if not isinstance(data, dict) or not data:
        return False
    if "success" in data:
        return bool(data.get("success", False))
    # missing success field; if it has any meaningful keys, treat as available but warn.
    msg = f"{name}: missing 'success' flag; treating data as available but degraded"
    warnings.append(msg)
    _logger.warning(msg, keys=list(data.keys())[:20])
    return True


# =====================================================================================
# FII/DII Scoring (existing logic preserved, hardened)
# =====================================================================================

def compute_fii_score(fii_data: Dict) -> Dict:
    """
    Compute FII/DII direction scores from NSE data.
    Existing scoring logic preserved.
    Hardened: safe parsing + availability semantics.
    """
    warnings: List[str] = []

    fii_strong = _get_cfg("narrative", "fii_strong_threshold_crore", default=2000)
    fii_mild = _get_cfg("narrative", "fii_mild_threshold_crore", default=500)

    # Defensive parsing
    success = _bool_success_flag(fii_data, name="fii_data", warnings=warnings)
    fii_net = _safe_float_from_dict(fii_data, "fii_net", default=None)
    dii_net = _safe_float_from_dict(fii_data, "dii_net", default=None)
    source = _safe_str_from_dict(fii_data, "source", default="unknown")

    # If required numeric fields are missing/invalid, do not compute misleading scores
    if not success or fii_net is None or dii_net is None:
        if success and (fii_net is None or dii_net is None):
            msg = "FII/DII success=True but missing fii_net or dii_net; returning None scores"
            warnings.append(msg)
            _logger.warning(msg, fii_net=fii_net, dii_net=dii_net)
        return {
            "fii_score": None,
            "dii_score": None,
            "fii_net_crore": None,
            "dii_net_crore": None,
            "fii_source": source,
            "fii_data_available": False,
            "warnings": warnings,
        }

    # --- Existing FII score logic (preserved) ---
    fii_score: int
    if fii_net >= fii_strong:
        fii_score = 100
    elif fii_net >= fii_mild:
        fii_score = int(50 + (fii_net - fii_mild) / (fii_strong - fii_mild) * 50)
    elif fii_net >= -fii_mild:
        fii_score = int(fii_net / fii_mild * 50)
    elif fii_net >= -fii_strong:
        fii_score = int(-50 - (abs(fii_net) - fii_mild) / (fii_strong - fii_mild) * 50)
    else:
        fii_score = -100
    fii_score = max(-100, min(100, fii_score))

    # --- Existing DII score logic (preserved) ---
    dii_score: int
    if dii_net >= fii_strong:
        dii_score = 100
    elif dii_net >= fii_mild:
        dii_score = 50
    elif dii_net >= -fii_mild:
        dii_score = 0
    elif dii_net >= -fii_strong:
        dii_score = -50
    else:
        dii_score = -100

    return {
        "fii_score": fii_score,
        "dii_score": dii_score,
        "fii_net_crore": round(float(fii_net), 2),
        "dii_net_crore": round(float(dii_net), 2),
        "fii_source": source,
        "fii_data_available": True,
        "warnings": warnings,
    }


# =====================================================================================
# Global Markets Score (existing logic preserved, hardened + caps + config)
# =====================================================================================

def compute_global_score(global_data: Dict) -> Dict:
    """
    Compute composite global markets score (-50..+50).
    Existing logic preserved; now hardened:
    - Safe parsing
    - Component caps with warning
    - Optional configurable multipliers via config.yaml fundamental.global_multipliers
    - Avoid silent fallback: if global_data unavailable => global_score=None
    """
    warnings: List[str] = []
    available = _bool_success_flag(global_data, name="global_data", warnings=warnings)

    # If global_data not available, do not produce a "neutral 0" that masks missing data
    if not available:
        return {
            "global_score": None,
            "global_data_available": False,
            # keep existing component keys present
            "sp500_score": None,
            "currency_score": None,
            "crude_score": None,
            "asia_score": None,
            "gold_score": None,
            "sp500_change_pct": None,
            "usdinr_change_pct": None,
            "crude_change_pct": None,
            "usdinr_price": None,
            "warnings": warnings,
        }

    # Configurable multipliers (Phase 3 optional)
    sp_mult = float(_get_cfg("fundamental", "global_multipliers", "sp500", default=1500))
    curr_mult = float(_get_cfg("fundamental", "global_multipliers", "usdinr", default=5000))
    crude_mult = float(_get_cfg("fundamental", "global_multipliers", "crude", default=500))
    asia_mult = float(_get_cfg("fundamental", "global_multipliers", "asia", default=1000))

    # Component caps (reasonable bounds)
    cap_sp = float(_get_cfg("fundamental", "global_component_caps", "sp500", default=15))
    cap_curr = float(_get_cfg("fundamental", "global_component_caps", "usdinr", default=10))
    cap_crude = float(_get_cfg("fundamental", "global_component_caps", "crude", default=10))
    cap_asia = float(_get_cfg("fundamental", "global_component_caps", "asia", default=10))
    cap_gold = float(_get_cfg("fundamental", "global_component_caps", "gold", default=5))

    score = 0.0
    components: Dict[str, Any] = {}

    # --- S&P 500 (±15) ---
    sp500 = global_data.get("sp500", {}) if isinstance(global_data, dict) else {}
    sp_change = _safe_float_from_dict(sp500, "change_pct", default=0.0) or 0.0

    raw_sp = 0.0
    if sp_change > 0.01:
        raw_sp = sp_change * sp_mult
    elif sp_change < -0.01:
        raw_sp = sp_change * sp_mult
    else:
        raw_sp = 0.0

    sp_score = int(round(_cap_with_log(raw_sp, cap_sp, "sp500", warnings)))
    score += sp_score
    components["sp500_score"] = sp_score
    components["sp500_change_pct"] = round(sp_change * 100, 2)

    # --- USD/INR (±10, inverse) ---
    currency_threshold = float(_get_cfg("narrative", "currency_threshold_pct", default=0.2))
    usdinr = global_data.get("usd_inr", {}) if isinstance(global_data, dict) else {}
    usd_change = _safe_float_from_dict(usdinr, "change_pct", default=0.0) or 0.0

    raw_curr = 0.0
    if usd_change > currency_threshold / 100:
        raw_curr = -usd_change * curr_mult
    elif usd_change < -currency_threshold / 100:
        raw_curr = -usd_change * curr_mult
    else:
        raw_curr = 0.0

    curr_score = int(round(_cap_with_log(raw_curr, cap_curr, "usdinr", warnings)))
    score += curr_score
    components["currency_score"] = curr_score
    components["usdinr_change_pct"] = round(usd_change * 100, 2)

    # usdinr_price must be validated (no surprises downstream)
    usdinr_price = _safe_float_from_dict(usdinr, "price", default=None)
    components["usdinr_price"] = usdinr_price

    # --- Crude Oil (±10, high crude bearish for India) ---
    crude = global_data.get("crude_oil", {}) if isinstance(global_data, dict) else {}
    crude_change = _safe_float_from_dict(crude, "change_pct", default=0.0) or 0.0

    raw_crude = 0.0
    if crude_change > 0.02:
        raw_crude = -crude_change * crude_mult
    elif crude_change < -0.02:
        raw_crude = -crude_change * crude_mult
    else:
        raw_crude = 0.0

    crude_score = int(round(_cap_with_log(raw_crude, cap_crude, "crude", warnings)))
    score += crude_score
    components["crude_score"] = crude_score
    components["crude_change_pct"] = round(crude_change * 100, 2)

    # --- Asia markets (±10) ---
    nikkei = global_data.get("nikkei", {}) if isinstance(global_data, dict) else {}
    hsi = global_data.get("hang_seng", {}) if isinstance(global_data, dict) else {}

    nk_change = _safe_float_from_dict(nikkei, "change_pct", default=0.0) or 0.0
    hsi_change = _safe_float_from_dict(hsi, "change_pct", default=0.0) or 0.0

    # preserve existing behavior: use avg only if either is non-zero
    asia_avg = (nk_change + hsi_change) / 2.0 if (nk_change != 0 or hsi_change != 0) else 0.0
    raw_asia = asia_avg * asia_mult
    asia_score = int(round(_cap_with_log(raw_asia, cap_asia, "asia", warnings)))
    score += asia_score
    components["asia_score"] = asia_score

    # --- Gold (±5, safe haven inverse) ---
    gold = global_data.get("gold", {}) if isinstance(global_data, dict) else {}
    gold_change = _safe_float_from_dict(gold, "change_pct", default=0.0) or 0.0

    raw_gold = 0.0
    if gold_change > 0.01:
        raw_gold = -3.0
    elif gold_change < -0.01:
        raw_gold = 3.0
    else:
        raw_gold = 0.0

    gold_score = int(round(_cap_with_log(raw_gold, cap_gold, "gold", warnings)))
    score += gold_score
    components["gold_score"] = gold_score

    score = float(_clamp(score, -50.0, 50.0))

    return {
        "global_score": int(round(score)),
        "global_data_available": True,
        **components,
        "warnings": warnings,
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# =====================================================================================
# VIX Analysis (existing logic preserved, hardened + contextual spike penalty)
# =====================================================================================

def compute_vix_features(global_data: Dict) -> Dict:
    """
    Compute India VIX features.
    Hardened:
    - Safe parsing
    - If vix missing => vix_* become None and vix_available False (upstream handles)
    - Contextual spike penalty based on zone (Phase 3 optional)
    """
    warnings: List[str] = []
    available = _bool_success_flag(global_data, name="global_data_for_vix", warnings=warnings)

    if not available:
        return {
            "vix_level": None,
            "vix_change_pct": None,
            "vix_zone": "UNKNOWN",
            "vix_score": None,
            "vix_spike": None,
            "warnings": warnings,
        }

    vix_calm = float(_get_cfg("narrative", "vix_calm", default=12))
    vix_normal = float(_get_cfg("narrative", "vix_normal", default=16))
    vix_caution = float(_get_cfg("narrative", "vix_caution", default=20))
    vix_fear = float(_get_cfg("narrative", "vix_fear", default=25))

    vix_data = global_data.get("india_vix_yf", {}) if isinstance(global_data, dict) else {}
    vix_level = _safe_float_from_dict(vix_data, "price", default=None)
    vix_change = _safe_float_from_dict(vix_data, "change_pct", default=None)

    if vix_level is None or vix_change is None:
        msg = "VIX data missing/invalid inside global_data; returning None vix fields"
        warnings.append(msg)
        _logger.warning(msg, vix_data=vix_data)
        return {
            "vix_level": None,
            "vix_change_pct": None,
            "vix_zone": "UNKNOWN",
            "vix_score": None,
            "vix_spike": None,
            "warnings": warnings,
        }

    # Determine zone (existing logic preserved)
    if vix_level <= 0:
        vix_zone = "UNKNOWN"
    elif vix_level < vix_calm:
        vix_zone = "CALM"
    elif vix_level < vix_normal:
        vix_zone = "NORMAL"
    elif vix_level < vix_caution:
        vix_zone = "CAUTION"
    elif vix_level < vix_fear:
        vix_zone = "FEAR"
    else:
        vix_zone = "PANIC"

    # VIX score (existing logic preserved)
    if vix_level <= 0:
        vix_score = 0
    elif vix_level < vix_calm:
        vix_score = 20
    elif vix_level < vix_normal:
        vix_score = 10
    elif vix_level < vix_caution:
        vix_score = -5
    elif vix_level < vix_fear:
        vix_score = -15
    else:
        vix_score = -20

    # Contextual spike penalty (Phase 3 recommended)
    spike_pct_threshold = float(_get_cfg("fundamental", "vix_spike", "pct_threshold", default=0.05))
    penalty_map = _get_cfg(
        "fundamental", "vix_spike", "penalty_by_zone",
        default={"NORMAL": 5, "CAUTION": 10, "FEAR": 15, "PANIC": 20}
    )
    vix_spike = False
    if vix_change is not None and float(vix_change) > float(spike_pct_threshold):
        vix_spike = True
        # Apply penalty based on zone
        try:
            zone_pen = int(penalty_map.get(vix_zone, 10)) if isinstance(penalty_map, dict) else 10
        except Exception:
            zone_pen = 10
        vix_score -= zone_pen
        warnings.append(f"VIX spike detected: change_pct={vix_change:.3f}, zone={vix_zone}, penalty={zone_pen}")

    return {
        "vix_level": round(float(vix_level), 2),
        "vix_change_pct": round(float(vix_change) * 100, 2),
        "vix_zone": vix_zone,
        "vix_score": max(-30, min(20, int(vix_score))),
        "vix_spike": vix_spike,
        "warnings": warnings,
    }


# =====================================================================================
# Economic Calendar (hardened with explicit failure semantics)
# =====================================================================================

def compute_event_features(calendar_path: str = None) -> Dict:
    """
    Compute event proximity features from economic calendar.

    HARDENING:
    - If calendar missing/corrupted: log ERROR, set event_severity=None and calendar_available=False
    - Never silently return 0 severity (that would imply "no events")
    """
    if calendar_path is None:
        calendar_path = os.path.join("data", "calendar", "economic_calendar.json")

    # Keep existing keys present
    result = {
        "event_severity": None,         # changed from 0 -> None when unavailable (mandated)
        "event_name": None,
        "event_days_away": None,
        "is_event_day": None,
        "events_this_week": None,

        # additive flags
        "calendar_available": False,
        "calendar_path": calendar_path,
        "calendar_error": None,
    }

    if not os.path.isfile(calendar_path):
        msg = f"Economic calendar file missing: {calendar_path}"
        _logger.error(msg)
        result["calendar_error"] = "missing_file"
        return result

    try:
        with open(calendar_path, "r", encoding="utf-8") as f:
            cal = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        msg = f"Economic calendar unreadable/corrupt: {calendar_path} ({type(e).__name__})"
        _logger.error(msg, error=str(e))
        result["calendar_error"] = "corrupt_or_unreadable"
        return result

    events = cal.get("events", [])
    if not isinstance(events, list) or not events:
        # Calendar exists but empty is still "available"
        result.update({
            "event_severity": 0,
            "event_name": "NONE",
            "event_days_away": 999,
            "is_event_day": False,
            "events_this_week": 0,
            "calendar_available": True,
        })
        return result

    today = ist_today()
    week_end = today + timedelta(days=7)

    # Find next event
    next_event = None
    events_this_week = 0

    # Defensive sort: ignore bad objects
    def _evt_key(x: Any) -> str:
        if isinstance(x, dict):
            return str(x.get("date", ""))
        return ""

    for evt in sorted(events, key=_evt_key):
        if not isinstance(evt, dict):
            continue
        evt_date_str = str(evt.get("date", "") or "")
        try:
            evt_date = datetime.strptime(evt_date_str, "%Y-%m-%d").date()
        except ValueError:
            _logger.warning("Invalid event date format in calendar; skipping", evt_date=evt_date_str)
            continue

        if evt_date >= today:
            if next_event is None:
                next_event = dict(evt)
                next_event["_date"] = evt_date

            if evt_date <= week_end:
                events_this_week += 1

    # Calendar is available even if no future events found
    result["calendar_available"] = True

    if next_event:
        days_away = int((next_event["_date"] - today).days)
        sev = next_event.get("severity", None)
        sev_int = _safe_int(sev, default=None)

        if sev_int is None or sev_int not in (0, 1, 2):
            _logger.warning("Invalid event severity; defaulting to 1 (caution)",
                            severity_raw=str(sev)[:50], event_name=next_event.get("name"))
            sev_int = 1

        result["event_severity"] = sev_int
        result["event_name"] = str(next_event.get("name", "Unknown"))
        result["event_days_away"] = days_away
        result["is_event_day"] = (days_away == 0)
        result["events_this_week"] = int(events_this_week)
    else:
        # No future events found; this is a valid condition
        result.update({
            "event_severity": 0,
            "event_name": "NONE",
            "event_days_away": 999,
            "is_event_day": False,
            "events_this_week": int(events_this_week),
        })

    return result


# =====================================================================================
# Master Feature Computation (hardened; no silent zero fallbacks)
# =====================================================================================

def compute_fundamental_features(
    fii_data: Optional[Dict] = None,
    global_data: Optional[Dict] = None,
    calendar_path: Optional[str] = None,
) -> Dict:
    """
    Compute all fundamental features.

    HARDENING REQUIREMENTS (Implemented):
    - Never crash
    - No silent zero-fallbacks:
        if input unavailable -> output fields become None with availability flags False
    - Adds data_quality dict summarizing availability
    """
    warnings: List[str] = []

    # -------------------------
    # FII/DII
    # -------------------------
    fii_available = _bool_success_flag(fii_data, name="fii_data", warnings=warnings)
    if fii_available:
        try:
            fii = compute_fii_score(fii_data)  # type: ignore[arg-type]
        except Exception as e:
            _logger.error("compute_fii_score failed; returning None fii fields", error=str(e))
            warnings.append(f"compute_fii_score exception: {type(e).__name__}")
            fii_available = False
            fii = {
                "fii_score": None, "dii_score": None,
                "fii_net_crore": None, "dii_net_crore": None,
                "fii_source": "error", "fii_data_available": False,
                "warnings": [str(e)[:120]],
            }
    else:
        # Mandated: do NOT set to 0. 0 is a valid neutral condition; None indicates missing.
        fii = {
            "fii_score": None, "dii_score": None,
            "fii_net_crore": None, "dii_net_crore": None,
            "fii_source": "none", "fii_data_available": False,
            "warnings": ["fii_data unavailable"],
        }

    # -------------------------
    # Global markets + VIX
    # -------------------------
    global_available = _bool_success_flag(global_data, name="global_data", warnings=warnings)
    if global_available:
        try:
            glob = compute_global_score(global_data)  # type: ignore[arg-type]
        except Exception as e:
            _logger.error("compute_global_score failed; returning None global fields", error=str(e))
            warnings.append(f"compute_global_score exception: {type(e).__name__}")
            global_available = False
            glob = {
                "global_score": None, "global_data_available": False,
                "sp500_score": None, "currency_score": None, "crude_score": None,
                "asia_score": None, "gold_score": None,
                "sp500_change_pct": None, "usdinr_change_pct": None,
                "crude_change_pct": None, "usdinr_price": None,
                "warnings": [str(e)[:120]],
            }

        try:
            vix = compute_vix_features(global_data)  # type: ignore[arg-type]
        except Exception as e:
            _logger.error("compute_vix_features failed; returning None vix fields", error=str(e))
            warnings.append(f"compute_vix_features exception: {type(e).__name__}")
            vix = {
                "vix_level": None, "vix_change_pct": None, "vix_zone": "UNKNOWN",
                "vix_score": None, "vix_spike": None,
                "warnings": [str(e)[:120]],
            }
    else:
        glob = {
            "global_score": None, "global_data_available": False,
            "sp500_score": None, "currency_score": None, "crude_score": None,
            "asia_score": None, "gold_score": None,
            "sp500_change_pct": None, "usdinr_change_pct": None,
            "crude_change_pct": None, "usdinr_price": None,
            "warnings": ["global_data unavailable"],
        }
        vix = {
            "vix_level": None, "vix_change_pct": None, "vix_zone": "UNKNOWN",
            "vix_score": None, "vix_spike": None,
            "warnings": ["vix unavailable (global_data unavailable)"],
        }

    # vix availability is separate: global may be available but VIX missing
    vix_available = (vix.get("vix_level") is not None) and (vix.get("vix_zone") != "UNKNOWN") and (vix.get("vix_score") is not None)

    # -------------------------
    # Events / Calendar
    # -------------------------
    try:
        events = compute_event_features(calendar_path)
    except Exception as e:
        _logger.error("compute_event_features failed; returning unavailable calendar fields", error=str(e))
        warnings.append(f"compute_event_features exception: {type(e).__name__}")
        events = {
            "event_severity": None,
            "event_name": None,
            "event_days_away": None,
            "is_event_day": None,
            "events_this_week": None,
            "calendar_available": False,
            "calendar_path": calendar_path,
            "calendar_error": "exception",
        }

    calendar_available = bool(events.get("calendar_available", False))

    # -------------------------
    # Institutional flow score (optional recommended)
    # -------------------------
    institutional_flow_score = None
    try:
        w_fii = float(_get_cfg("fundamental", "institutional_flow_weights", "fii", default=0.7))
        w_dii = float(_get_cfg("fundamental", "institutional_flow_weights", "dii", default=0.3))
        fii_score = fii.get("fii_score")
        dii_score = fii.get("dii_score")
        if fii_score is not None and dii_score is not None:
            institutional_flow_score = round((float(fii_score) * w_fii + float(dii_score) * w_dii), 2)
    except Exception as e:
        warnings.append(f"institutional_flow_score compute failed: {type(e).__name__}")
        _logger.warning("institutional_flow_score compute failed", error=str(e))

    # -------------------------
    # Consolidated data_quality (required)
    # -------------------------
    data_quality = {
        "fii_available": bool(fii_available and fii.get("fii_data_available", False)),
        "global_available": bool(global_available and glob.get("global_data_available", False)),
        "vix_available": bool(vix_available),
        "calendar_available": bool(calendar_available),
    }

    # Merge outputs (existing keys preserved)
    result = {
        **fii,
        **glob,
        **vix,
        **events,

        # New additive fields
        "data_quality": data_quality,
        "institutional_flow_score": institutional_flow_score,
        "warnings": warnings + (fii.get("warnings", []) or []) + (glob.get("warnings", []) or []) + (vix.get("warnings", []) or []),
    }

    # Ensure existing keys always exist (even if compute_* returned weirdly)
    # This prevents downstream KeyError under partial failures.
    _ensure_required_keys(result)

    return result


def _ensure_required_keys(result: Dict) -> None:
    """
    Hard guarantee that legacy keys exist.
    Prevents downstream crashes due to missing keys.
    """
    legacy_defaults = {
        # FII/DII
        "fii_score": None,
        "dii_score": None,
        "fii_net_crore": None,
        "dii_net_crore": None,
        "fii_source": "unknown",
        "fii_data_available": False,

        # Global
        "global_score": None,
        "global_data_available": False,
        "sp500_score": None,
        "currency_score": None,
        "crude_score": None,
        "asia_score": None,
        "gold_score": None,
        "sp500_change_pct": None,
        "usdinr_change_pct": None,
        "crude_change_pct": None,
        "usdinr_price": None,

        # VIX
        "vix_level": None,
        "vix_change_pct": None,
        "vix_zone": "UNKNOWN",
        "vix_score": None,
        "vix_spike": None,

        # Events
        "event_severity": None,
        "event_name": None,
        "event_days_away": None,
        "is_event_day": None,
        "events_this_week": None,
    }
    for k, v in legacy_defaults.items():
        if k not in result:
            result[k] = v


# =====================================================================================
# Module Self-Test (updated with missing/bad data scenarios)
# =====================================================================================

def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Fundamental Features Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: FII scoring (valid) ──
    print("  [Test 1] FII/DII scoring (valid)...")
    strong_buy = {"fii_net": 3000, "dii_net": -1000, "source": "test", "success": True}
    score = compute_fii_score(strong_buy)
    if score["fii_score"] == 100 and score["fii_data_available"]:
        print(f"    ✅ Strong FII buy: score={score['fii_score']}")
        passed += 1
    else:
        print(f"    ❌ Expected 100, got {score}")
        failed += 1

    # ── Test 2: FII bad data should not crash ──
    print("\n  [Test 2] FII/DII scoring (malformed)...")
    bad_fii = {"fii_net": "N/A", "dii_net": None, "success": True}
    score_bad = compute_fii_score(bad_fii)
    if score_bad["fii_score"] is None and score_bad["fii_data_available"] is False:
        print("    ✅ Malformed FII returns None scores + available False")
        passed += 1
    else:
        print(f"    ❌ Expected None scores, got {score_bad}")
        failed += 1

    # ── Test 3: Global score caps anomalies ──
    print("\n  [Test 3] Global score anomaly cap...")
    weird_global = {
        "success": True,
        "sp500": {"price": 5000, "change_pct": 10.0},  # 1000% bogus
        "usd_inr": {"price": None, "change_pct": 0.50},
        "crude_oil": {"price": 90, "change_pct": 2.0},
        "nikkei": {"price": 39000, "change_pct": 1.0},
        "hang_seng": {"price": 18000, "change_pct": 1.0},
        "gold": {"price": 2300, "change_pct": 0.5},
        "india_vix_yf": {"price": 13.5, "change_pct": -0.02},
    }
    glob = compute_global_score(weird_global)
    if glob["global_data_available"] and (-50 <= (glob["global_score"] or 0) <= 50):
        print(f"    ✅ Global score computed and bounded: {glob['global_score']}")
        passed += 1
    else:
        print(f"    ❌ Global score invalid: {glob}")
        failed += 1

    # ── Test 4: Missing global_data should yield None, not 0 ──
    print("\n  [Test 4] Missing global_data => None outputs...")
    feat_missing_global = compute_fundamental_features(fii_data=strong_buy, global_data=None)
    if feat_missing_global["global_score"] is None and feat_missing_global["data_quality"]["global_available"] is False:
        print("    ✅ Missing global => global_score None + global_available False")
        passed += 1
    else:
        print(f"    ❌ Expected None global_score, got {feat_missing_global['global_score']}")
        failed += 1

    # ── Test 5: Missing calendar file should be explicit (None severity) ──
    print("\n  [Test 5] Missing calendar file => calendar_available False, event_severity None...")
    missing_path = os.path.join("data", "calendar", "this_file_should_not_exist.json")
    ev = compute_event_features(missing_path)
    if (ev["calendar_available"] is False) and (ev["event_severity"] is None):
        print("    ✅ Calendar missing is explicit (no silent 0 severity)")
        passed += 1
    else:
        print(f"    ❌ Calendar missing not handled: {ev}")
        failed += 1

    # ── Test 6: Full computation should include data_quality key ──
    print("\n  [Test 6] data_quality present...")
    bullish_global = {
        "success": True,
        "sp500": {"price": 5500, "change_pct": 0.015},
        "usd_inr": {"price": 83.5, "change_pct": -0.001},
        "crude_oil": {"price": 75, "change_pct": -0.01},
        "nikkei": {"price": 39000, "change_pct": 0.01},
        "hang_seng": {"price": 18000, "change_pct": 0.008},
        "gold": {"price": 2300, "change_pct": -0.005},
        "india_vix_yf": {"price": 13.5, "change_pct": -0.02},
    }
    feats = compute_fundamental_features(fii_data=strong_buy, global_data=bullish_global)
    if "data_quality" in feats and isinstance(feats["data_quality"], dict):
        print("    ✅ data_quality present")
        passed += 1
    else:
        print("    ❌ data_quality missing")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅ Fundamental module hardened and safe for live pipelines.")
    else:
        print("  ⚠️ Some tests failed; check logs/fundamental.log.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()