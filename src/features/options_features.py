"""
Junior Aladdin - Options Features (Layer 2E)
==============================================
PURPOSE:
    Compute options-derived features from the option chain.

WARNING (Blueprint Compliance / Data Contract):
    The input `chain` MUST contain option instruments for a SINGLE expiry only.
    The caller (OptionChainPoller / DataEngine) is responsible for filtering by expiry
    before invoking this module.

    If expiry information is present in the chain payload (e.g., data['expiry'] or
    data['ce']['expiry']), this module will validate that only one expiry is present.
    If multiple expiries are detected, the chain is treated as INVALID and the module
    returns empty features with data_quality.status="INVALID".

CRITICAL HARDENING (Production):
    - Strict input validation (structure + minimum strikes)
    - No silent zero fallbacks for PCR/IV/MaxPain/Walls/GEX
    - No "volume as OI" proxy (removed)
    - Gamma/GEX validity checks
    - Data quality flags embedded in output for downstream gating
    - Logging for all fallbacks/degradations
    - Strike key type safety (normalize to int where needed)
    - PCR change timestamp validation to avoid stale deltas
    - OI Classification (LONG_BUILDUP / SHORT_COVERING / SHORT_BUILDUP / LONG_UNWINDING)

DATA SOURCE:
    Option chain dict:
    {
        strike_int_or_str: {
            "ce": {"ltp", "iv", "oi", "volume", "delta", "gamma", ...},
            "pe": {"ltp", "iv", "oi", "volume", "delta", "gamma", ...},
            # optional: "expiry": "YYYY-MM-DD"
        },
        ...
    }
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.utils.helpers import round_to_strike

_logger = setup_logger("options_features")

IST = timezone(timedelta(hours=5, minutes=30))


# ============================================
# Validation / Normalization
# ============================================

def _validate_chain(chain: Dict) -> Tuple[bool, str]:
    """
    Validate option chain structure.

    Checks:
      - chain is a dict
      - at least 3 strikes
      - each strike maps to dict with "ce" and "pe" sub-dicts
    """
    if not isinstance(chain, dict):
        return False, "chain_not_dict"
    if len(chain) < 3:
        return False, "too_few_strikes"

    checked = 0
    for _k, data in chain.items():
        if not isinstance(data, dict):
            return False, "strike_data_not_dict"
        ce = data.get("ce")
        pe = data.get("pe")
        if not isinstance(ce, dict) or not isinstance(pe, dict):
            return False, "missing_ce_or_pe_dict"
        checked += 1
        if checked >= 3:
            break

    return True, ""


def _iter_strikes_normalized(chain: Dict) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Normalize strikes to int. Skips unparseable strikes with warning.
    Returns list[(strike_int, data_dict)] sorted by strike.
    """
    out: List[Tuple[int, Dict[str, Any]]] = []
    if not isinstance(chain, dict):
        return out

    for strike, data in chain.items():
        if not isinstance(data, dict):
            continue
        try:
            strike_i = int(strike)
        except Exception:
            _logger.warning("Skipping unparseable strike key", extra={"strike": str(strike)[:50]})
            continue
        out.append((strike_i, data))

    out.sort(key=lambda x: x[0])
    return out


def _extract_expiry_if_present(strike_data: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort expiry extraction if present.
    Accepts expiry in:
      - strike_data['expiry']
      - strike_data['ce']['expiry']
      - strike_data['pe']['expiry']
    """
    if not isinstance(strike_data, dict):
        return None
    exp = strike_data.get("expiry")
    if isinstance(exp, str) and exp.strip():
        return exp.strip()

    ce = strike_data.get("ce")
    if isinstance(ce, dict):
        exp = ce.get("expiry")
        if isinstance(exp, str) and exp.strip():
            return exp.strip()

    pe = strike_data.get("pe")
    if isinstance(pe, dict):
        exp = pe.get("expiry")
        if isinstance(exp, str) and exp.strip():
            return exp.strip()

    return None


def _validate_single_expiry_if_present(chain: Dict) -> Tuple[bool, Optional[str], str]:
    """
    If expiry data is present, validates it is consistent (single expiry).
    Returns: (ok, expiry, reason)
      - ok=True means either:
          a) no expiry info present, or
          b) exactly one expiry present
      - ok=False means multiple expiries detected
    """
    expiries = set()
    for _strike, data in _iter_strikes_normalized(chain):
        exp = _extract_expiry_if_present(data)
        if exp:
            expiries.add(exp)

    if len(expiries) <= 1:
        expiry = next(iter(expiries)) if expiries else None
        return True, expiry, ""

    return False, None, "mixed_expiry_detected"


def _detect_oi_available(chain: Dict) -> bool:
    """True if any strike has non-zero OI in CE or PE."""
    for _strike, data in _iter_strikes_normalized(chain):
        ce = data.get("ce", {}) if isinstance(data.get("ce"), dict) else {}
        pe = data.get("pe", {}) if isinstance(data.get("pe"), dict) else {}
        if _safe_int(ce.get("oi"), default=0) > 0 or _safe_int(pe.get("oi"), default=0) > 0:
            return True
    return False


def _detect_iv_available(chain: Dict) -> bool:
    """True if any strike has positive IV in CE or PE."""
    for _strike, data in _iter_strikes_normalized(chain):
        ce = data.get("ce", {}) if isinstance(data.get("ce"), dict) else {}
        pe = data.get("pe", {}) if isinstance(data.get("pe"), dict) else {}
        if _safe_float(ce.get("iv"), default=0.0) > 0 or _safe_float(pe.get("iv"), default=0.0) > 0:
            return True
    return False


# ============================================
# PCR (Put-Call Ratio)
# ============================================

def compute_pcr(chain: Dict) -> Dict:
    """
    Compute Put-Call Ratios.

    HARDENED:
      - If CE and PE OI are both zero => pcr_oi = None (not 0.0)
      - If CE OI is zero but PE OI > 0 => pcr_oi = None (undefined)
      - Same handling for volume PCR (returns None instead of misleading 0)
    """
    total_ce_oi = 0
    total_pe_oi = 0
    total_ce_vol = 0
    total_pe_vol = 0
    total_ce_oi_change = 0
    total_pe_oi_change = 0

    for _strike, data in _iter_strikes_normalized(chain):
        ce = data.get("ce", {}) if isinstance(data.get("ce"), dict) else {}
        pe = data.get("pe", {}) if isinstance(data.get("pe"), dict) else {}

        total_ce_oi += _safe_int(ce.get("oi", 0))
        total_pe_oi += _safe_int(pe.get("oi", 0))
        total_ce_vol += _safe_int(ce.get("volume", 0))
        total_pe_vol += _safe_int(pe.get("volume", 0))
        total_ce_oi_change += _safe_int(ce.get("oi_change", 0))
        total_pe_oi_change += _safe_int(pe.get("oi_change", 0))

    # OI PCR
    if total_ce_oi == 0 and total_pe_oi == 0:
        pcr_oi = None
        _logger.warning("PCR OI unavailable: total CE OI and PE OI are zero")
    elif total_ce_oi == 0:
        pcr_oi = None
        _logger.warning(
            "PCR OI unavailable: total CE OI is zero while PE OI is non-zero",
            extra={"total_pe_oi": total_pe_oi},
        )
    else:
        pcr_oi = round(total_pe_oi / total_ce_oi, 3)

    # Volume PCR
    if total_ce_vol == 0 and total_pe_vol == 0:
        pcr_vol = None
        _logger.debug("PCR Volume unavailable: total CE/PE volume are zero")
    elif total_ce_vol == 0:
        pcr_vol = None
        _logger.debug("PCR Volume unavailable: total CE volume is zero", extra={"total_pe_vol": total_pe_vol})
    else:
        pcr_vol = round(total_pe_vol / total_ce_vol, 3)

    return {
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_vol,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "total_ce_volume": total_ce_vol,
        "total_pe_volume": total_pe_vol,
        "net_oi_change_ce": total_ce_oi_change,
        "net_oi_change_pe": total_pe_oi_change,
        "oi_available": not (total_ce_oi == 0 and total_pe_oi == 0),
    }


# ============================================
# IV (Implied Volatility) Features
# ============================================

def compute_iv_features(
    chain: Dict,
    spot_price: float,
    strike_interval: int = 50,
    iv_history: Optional[List[float]] = None,
) -> Dict:
    """
    Compute IV-based features.

    HARDENED:
      - If ATM IV cannot be computed after fallbacks => atm_iv=None, atm_iv_valid=False
      - Logs WARNING when IV unavailable or when using nearby strikes
    """
    strike_interval = int(strike_interval) if strike_interval else 50
    atm_strike = round_to_strike(spot_price, strike_interval)

    normalized = dict(_iter_strikes_normalized(chain))
    atm_data = normalized.get(int(atm_strike), {})

    ce_iv = _safe_float((atm_data.get("ce") or {}).get("iv", 0), default=0.0)
    pe_iv = _safe_float((atm_data.get("pe") or {}).get("iv", 0), default=0.0)
    strike_used = int(atm_strike)

    if ce_iv == 0.0 and pe_iv == 0.0:
        for offset in (strike_interval, -strike_interval):
            nearby_strike = int(atm_strike + offset)
            nearby = normalized.get(nearby_strike, {})
            if not nearby:
                continue
            nce = nearby.get("ce", {}) if isinstance(nearby.get("ce"), dict) else {}
            npe = nearby.get("pe", {}) if isinstance(nearby.get("pe"), dict) else {}
            nce_iv = _safe_float(nce.get("iv", 0), default=0.0)
            npe_iv = _safe_float(npe.get("iv", 0), default=0.0)
            if nce_iv > 0.0 or npe_iv > 0.0:
                ce_iv = nce_iv
                pe_iv = npe_iv
                strike_used = nearby_strike
                _logger.debug(
                    "Using nearby strike for ATM IV",
                    extra={"atm_strike": int(atm_strike), "strike_used": strike_used, "spot": spot_price},
                )
                break

    atm_iv: Optional[float]
    if ce_iv > 0 and pe_iv > 0:
        atm_iv = round((ce_iv + pe_iv) / 2.0, 4)
    elif ce_iv > 0:
        atm_iv = round(ce_iv, 4)
    elif pe_iv > 0:
        atm_iv = round(pe_iv, 4)
    else:
        atm_iv = None

    atm_iv_valid = atm_iv is not None and atm_iv > 0
    if not atm_iv_valid:
        _logger.warning(
            "ATM IV unavailable after fallbacks; setting IV features to None",
            extra={"spot_price": spot_price, "atm_strike": int(atm_strike), "strike_used": strike_used},
        )

    iv_skew: Optional[float]
    if ce_iv > 0 and pe_iv > 0:
        iv_skew = round(ce_iv - pe_iv, 4)
    else:
        iv_skew = None
        if atm_iv_valid:
            _logger.debug("IV skew unavailable (missing CE or PE IV)", extra={"ce_iv": ce_iv, "pe_iv": pe_iv})

    iv_rank = None
    if iv_history and atm_iv_valid:
        valid_history = [float(v) for v in iv_history if isinstance(v, (int, float)) and float(v) > 0]
        if len(valid_history) >= 2:
            below = sum(1 for v in valid_history if v < float(atm_iv))
            iv_rank = round((below / len(valid_history)) * 100.0, 1)

    return {
        "atm_iv": atm_iv,
        "atm_iv_pct": round(float(atm_iv) * 100.0, 2) if atm_iv_valid else None,
        "atm_iv_valid": bool(atm_iv_valid),
        "ce_iv": round(ce_iv, 4) if ce_iv > 0 else None,
        "pe_iv": round(pe_iv, 4) if pe_iv > 0 else None,
        "iv_skew": iv_skew,
        "iv_skew_pct": round(float(iv_skew) * 100.0, 2) if iv_skew is not None else None,
        "iv_rank_session": iv_rank,
        "atm_strike_used": strike_used,
        "iv_available": bool(atm_iv_valid),
    }


# ============================================
# OI Classification (Blueprint Compliance)
# ============================================

def compute_oi_classification(
    current_chain: Dict,
    previous_chain: Optional[Dict],
    spot_price: float,
) -> Dict:
    """
    Blueprint-required OI classification.

    Logic (per strike per leg):
      Price↑ + OI↑  => LONG_BUILDUP
      Price↑ + OI↓  => SHORT_COVERING
      Price↓ + OI↑  => SHORT_BUILDUP
      Price↓ + OI↓  => LONG_UNWINDING
      else          => NEUTRAL / UNKNOWN

    Price proxy:
      Uses option leg LTP change (CE uses CE LTP, PE uses PE LTP).
      This is a practical proxy given the chain data structure.

    Returns:
      {
        "available": bool,
        "summary": {"LONG_BUILDUP": n, ...},
        "net_score": int,
        "details": {strike: {"ce": {...}, "pe": {...}}}
      }
    """
    if not isinstance(current_chain, dict) or not current_chain:
        return {"available": False, "summary": {}, "net_score": 0, "details": {}}

    if not isinstance(previous_chain, dict) or not previous_chain:
        _logger.debug("OI classification unavailable: previous_chain missing/empty")
        return {"available": False, "summary": {}, "net_score": 0, "details": {}}

    cur = {s: d for s, d in _iter_strikes_normalized(current_chain)}
    prev = {s: d for s, d in _iter_strikes_normalized(previous_chain)}
    if not cur or not prev:
        return {"available": False, "summary": {}, "net_score": 0, "details": {}}

    summary = {
        "LONG_BUILDUP": 0,
        "SHORT_COVERING": 0,
        "SHORT_BUILDUP": 0,
        "LONG_UNWINDING": 0,
        "NEUTRAL": 0,
        "UNKNOWN": 0,
    }
    details: Dict[int, Dict[str, Any]] = {}

    def classify(price_chg: float, oi_chg: int) -> str:
        if price_chg > 0 and oi_chg > 0:
            return "LONG_BUILDUP"
        if price_chg > 0 and oi_chg < 0:
            return "SHORT_COVERING"
        if price_chg < 0 and oi_chg > 0:
            return "SHORT_BUILDUP"
        if price_chg < 0 and oi_chg < 0:
            return "LONG_UNWINDING"
        if price_chg == 0 or oi_chg == 0:
            return "NEUTRAL"
        return "UNKNOWN"

    for strike, cur_data in cur.items():
        prev_data = prev.get(strike)
        if not isinstance(prev_data, dict):
            continue

        strike_detail: Dict[str, Any] = {}
        for leg in ("ce", "pe"):
            c_leg = cur_data.get(leg, {}) if isinstance(cur_data.get(leg), dict) else {}
            p_leg = prev_data.get(leg, {}) if isinstance(prev_data.get(leg), dict) else {}

            c_ltp = _safe_float(c_leg.get("ltp"), default=None)  # type: ignore[arg-type]
            p_ltp = _safe_float(p_leg.get("ltp"), default=None)  # type: ignore[arg-type]
            c_oi = _safe_int(c_leg.get("oi"), default=None)      # type: ignore[arg-type]
            p_oi = _safe_int(p_leg.get("oi"), default=None)      # type: ignore[arg-type]

            if c_ltp is None or p_ltp is None or c_oi is None or p_oi is None:
                strike_detail[leg] = {"classification": "UNKNOWN"}
                summary["UNKNOWN"] += 1
                continue

            price_chg = float(c_ltp) - float(p_ltp)
            oi_chg = int(c_oi) - int(p_oi)
            cls = classify(price_chg, oi_chg)
            summary[cls] = summary.get(cls, 0) + 1

            strike_detail[leg] = {
                "classification": cls,
                "price_change": round(price_chg, 4),
                "oi_change": int(oi_chg),
                "current_ltp": float(c_ltp),
                "prev_ltp": float(p_ltp),
                "current_oi": int(c_oi),
                "prev_oi": int(p_oi),
            }

        if strike_detail:
            details[int(strike)] = strike_detail

    # Net score: positive implies more buildup/covering vs buildup of shorts/unwind
    net_score = (summary["LONG_BUILDUP"] + summary["SHORT_COVERING"]) - (summary["SHORT_BUILDUP"] + summary["LONG_UNWINDING"])

    available = any(v > 0 for k, v in summary.items() if k in ("LONG_BUILDUP", "SHORT_COVERING", "SHORT_BUILDUP", "LONG_UNWINDING"))
    return {"available": bool(available), "summary": summary, "net_score": int(net_score), "details": details}


# ============================================
# Max Pain
# ============================================

def compute_max_pain(chain: Dict, spot_price: float) -> Dict:
    """Compute Max Pain from OI only (no volume proxy)."""
    strikes_data = _iter_strikes_normalized(chain)
    if len(strikes_data) < 3:
        return {"max_pain": None, "max_pain_distance": None, "max_pain_valid": False}

    strikes = [s for s, _d in strikes_data]
    normalized = {s: d for s, d in strikes_data}

    ce_oi_map: Dict[int, int] = {}
    pe_oi_map: Dict[int, int] = {}

    for strike in strikes:
        data = normalized[strike]
        ce = data.get("ce", {}) if isinstance(data.get("ce"), dict) else {}
        pe = data.get("pe", {}) if isinstance(data.get("pe"), dict) else {}
        ce_oi_map[strike] = _safe_int(ce.get("oi", 0))
        pe_oi_map[strike] = _safe_int(pe.get("oi", 0))

    total_oi = sum(ce_oi_map.values()) + sum(pe_oi_map.values())
    if total_oi == 0:
        _logger.warning("Max Pain unavailable: no OI data in chain; returning None")
        return {"max_pain": None, "max_pain_distance": None, "max_pain_valid": False}

    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]

    for test_strike in strikes:
        total_pain = 0.0

        for ce_strike, ce_oi in ce_oi_map.items():
            if test_strike > ce_strike and ce_oi > 0:
                total_pain += (test_strike - ce_strike) * ce_oi

        for pe_strike, pe_oi in pe_oi_map.items():
            if test_strike < pe_strike and pe_oi > 0:
                total_pain += (pe_strike - test_strike) * pe_oi

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike

    distance = round(float(spot_price) - float(max_pain_strike), 2) if spot_price > 0 else None
    return {"max_pain": int(max_pain_strike), "max_pain_distance": distance, "max_pain_valid": True}


# ============================================
# OI Walls
# ============================================

def compute_oi_walls(chain: Dict, spot_price: float) -> Dict:
    """Find highest OI strikes from OI only (no volume proxy)."""
    strikes_data = _iter_strikes_normalized(chain)
    if not strikes_data:
        return {
            "highest_ce_oi_strike": None, "highest_ce_oi": None,
            "highest_pe_oi_strike": None, "highest_pe_oi": None,
            "ce_wall_distance": None, "pe_wall_distance": None,
            "oi_walls_valid": False,
        }

    max_ce_oi = 0
    max_ce_strike: Optional[int] = None
    max_pe_oi = 0
    max_pe_strike: Optional[int] = None

    for strike, data in strikes_data:
        ce = data.get("ce", {}) if isinstance(data.get("ce"), dict) else {}
        pe = data.get("pe", {}) if isinstance(data.get("pe"), dict) else {}

        ce_oi = _safe_int(ce.get("oi", 0))
        pe_oi = _safe_int(pe.get("oi", 0))

        if ce_oi > max_ce_oi:
            max_ce_oi = ce_oi
            max_ce_strike = strike

        if pe_oi > max_pe_oi:
            max_pe_oi = pe_oi
            max_pe_strike = strike

    if max_ce_oi == 0 and max_pe_oi == 0:
        _logger.warning("OI walls unavailable: all OI values are zero; returning None")
        return {
            "highest_ce_oi_strike": None, "highest_ce_oi": None,
            "highest_pe_oi_strike": None, "highest_pe_oi": None,
            "ce_wall_distance": None, "pe_wall_distance": None,
            "oi_walls_valid": False,
        }

    ce_distance = round(float(max_ce_strike) - float(spot_price), 2) if (max_ce_strike is not None and spot_price > 0) else None
    pe_distance = round(float(spot_price) - float(max_pe_strike), 2) if (max_pe_strike is not None and spot_price > 0) else None

    return {
        "highest_ce_oi_strike": max_ce_strike,
        "highest_ce_oi": max_ce_oi if max_ce_strike is not None else None,
        "highest_pe_oi_strike": max_pe_strike,
        "highest_pe_oi": max_pe_oi if max_pe_strike is not None else None,
        "ce_wall_distance": ce_distance,
        "pe_wall_distance": pe_distance,
        "oi_walls_valid": True,
    }


# ============================================
# Synthetic Future & GEX Proxy
# ============================================

def compute_synthetic(
    chain: Dict,
    spot_price: float,
    strike_interval: int = 50,
    lot_size: int = 65,
) -> Dict:
    """Compute Synthetic Future and GEX Proxy with gamma validation."""
    strike_interval = int(strike_interval) if strike_interval else 50
    lot_size = int(lot_size) if lot_size else 65

    strikes_data = _iter_strikes_normalized(chain)
    normalized = {s: d for s, d in strikes_data}

    atm_strike = int(round_to_strike(spot_price, strike_interval))
    atm_data = normalized.get(atm_strike, {})

    ce_ltp = _safe_float((atm_data.get("ce") or {}).get("ltp", 0), default=0.0)
    pe_ltp = _safe_float((atm_data.get("pe") or {}).get("ltp", 0), default=0.0)

    if ce_ltp > 0 and pe_ltp > 0:
        synthetic_future = round(ce_ltp - pe_ltp + atm_strike, 2)
        synthetic_premium = round(synthetic_future - float(spot_price), 2)
    else:
        synthetic_future = None
        synthetic_premium = None
        _logger.warning(
            "Synthetic future unavailable: missing ATM CE/PE LTP",
            extra={"atm_strike": atm_strike, "ce_ltp": ce_ltp, "pe_ltp": pe_ltp},
        )

    gex_raw = 0.0
    used_gamma = 0
    skipped = 0

    for strike, data in strikes_data:
        ce = data.get("ce", {}) if isinstance(data.get("ce"), dict) else {}
        pe = data.get("pe", {}) if isinstance(data.get("pe"), dict) else {}

        ce_oi = _safe_int(ce.get("oi", 0))
        pe_oi = _safe_int(pe.get("oi", 0))
        ce_gamma = _safe_float(ce.get("gamma", None), default=0.0)
        pe_gamma = _safe_float(pe.get("gamma", None), default=0.0)

        ce_term = 0.0
        pe_term = 0.0

        if ce_gamma > 0 and ce_oi > 0:
            ce_term = ce_oi * ce_gamma
            used_gamma += 1
        elif ce_gamma <= 0 and ce_oi > 0:
            skipped += 1

        if pe_gamma > 0 and pe_oi > 0:
            pe_term = pe_oi * pe_gamma
            used_gamma += 1
        elif pe_gamma <= 0 and pe_oi > 0:
            skipped += 1

        gex_raw += (ce_term - pe_term)

    if used_gamma == 0:
        _logger.warning(
            "GEX unavailable: gamma missing/zero across strikes; returning None",
            extra={"skipped_due_to_gamma": skipped, "strikes": len(strikes_data)},
        )
        gex_value = None
        gex_regime = "NEUTRAL"
        gex_available = False
    else:
        gex_value = round(gex_raw * lot_size * float(spot_price), 0) if spot_price > 0 else None
        gex_available = gex_value is not None

        if gex_value is None or gex_value == 0:
            gex_regime = "NEUTRAL"
        elif gex_value > 0:
            gex_regime = "POSITIVE"
        else:
            gex_regime = "NEGATIVE"

    return {
        "synthetic_future": synthetic_future,
        "synthetic_premium": synthetic_premium,
        "gex_proxy": gex_value,
        "gex_regime": gex_regime,
        "gex_available": bool(gex_available),
    }


# ============================================
# Master Feature Computation
# ============================================

def compute_options_features(
    chain: Dict,
    spot_price: float,
    previous_pcr_oi: Optional[float] = None,
    iv_history: Optional[List[float]] = None,
    previous_pcr_timestamp: Optional[datetime] = None,  # NEW: stale-guard
    previous_chain: Optional[Dict] = None,              # NEW: OI classification
) -> Dict:
    """
    Compute all options-derived features.

    NEW:
      - previous_pcr_timestamp: blocks stale pcr_change computation
      - previous_chain: enables blueprint OI classification
    """
    valid, reason = _validate_chain(chain)
    if not valid or spot_price <= 0:
        if not valid:
            _logger.error("Option chain invalid; returning empty features", extra={"reason": reason})
        elif spot_price <= 0:
            _logger.error("Invalid spot_price for options features; returning empty", extra={"spot_price": spot_price})
        out = _empty_features()
        out["data_quality"]["chain_valid"] = False
        out["data_quality"]["status"] = "INVALID"
        out["data_quality"]["reason"] = reason if not valid else "invalid_spot"
        return out

    # Expiry validation (optional if expiry info exists)
    expiry_ok, expiry_val, expiry_reason = _validate_single_expiry_if_present(chain)
    if not expiry_ok:
        _logger.error(
            "Option chain contains multiple expiries; refusing to compute mixed-expiry features",
            extra={"reason": expiry_reason},
        )
        out = _empty_features()
        out["data_quality"]["chain_valid"] = False
        out["data_quality"]["status"] = "INVALID"
        out["data_quality"]["reason"] = expiry_reason
        return out

    strike_interval = int(Config.get("market", "strike_interval", default=50) or 50)
    lot_size = int(Config.get("market", "lot_size", default=65) or 65)

    oi_available = _detect_oi_available(chain)
    iv_available_any = _detect_iv_available(chain)

    pcr_data = compute_pcr(chain)

    # PCR change timestamp validation (MANDATE)
    pcr_change = None
    if previous_pcr_oi is not None and isinstance(previous_pcr_oi, (int, float)) and pcr_data.get("pcr_oi") is not None:
        max_age_min = int(Config.get("options", "pcr_change_max_age_minutes", default=60) or 60)
        if previous_pcr_timestamp is not None:
            ts = previous_pcr_timestamp
            if ts.tzinfo is None:
                _logger.warning("previous_pcr_timestamp is naive; assuming IST for age check")
                ts = ts.replace(tzinfo=IST)

            age_min = (datetime.now(IST) - ts).total_seconds() / 60.0
            if age_min > max_age_min:
                _logger.warning(
                    "Skipping pcr_change due to stale previous_pcr_timestamp",
                    extra={"age_min": round(age_min, 1), "max_age_min": max_age_min},
                )
                pcr_change = None
            else:
                try:
                    pcr_change = round(float(pcr_data["pcr_oi"]) - float(previous_pcr_oi), 3)
                except Exception:
                    pcr_change = None
        else:
            # No timestamp provided: do NOT block, but warn because caller should provide it in production
            _logger.debug("previous_pcr_timestamp not provided; pcr_change computed without staleness check")
            try:
                pcr_change = round(float(pcr_data["pcr_oi"]) - float(previous_pcr_oi), 3)
            except Exception:
                pcr_change = None

    iv_data = compute_iv_features(chain, spot_price, strike_interval, iv_history)
    mp_data = compute_max_pain(chain, spot_price)
    walls = compute_oi_walls(chain, spot_price)
    synth = compute_synthetic(chain, spot_price, strike_interval, lot_size)

    # OI Classification (optional)
    oi_cls = compute_oi_classification(chain, previous_chain, spot_price)

    dq = {
        "chain_valid": True,
        "oi_available": bool(oi_available),
        "iv_available": bool(iv_data.get("iv_available", False)) or bool(iv_available_any),
        "max_pain_available": bool(mp_data.get("max_pain_valid", False)),
        "gex_available": bool(synth.get("gex_available", False)),
        "oi_classification_available": bool(oi_cls.get("available", False)),
        "status": "OK",
        "reason": None,
        "expiry": expiry_val,
    }

    if not dq["iv_available"]:
        dq["status"] = "DEGRADED"
        dq["reason"] = "iv_unavailable"
    elif not dq["oi_available"]:
        dq["status"] = "DEGRADED"
        dq["reason"] = "oi_unavailable"

    result = {
        # PCR
        "pcr_oi": pcr_data.get("pcr_oi"),
        "pcr_volume": pcr_data.get("pcr_volume"),
        "pcr_change": pcr_change,
        "total_ce_oi": pcr_data.get("total_ce_oi", 0),
        "total_pe_oi": pcr_data.get("total_pe_oi", 0),
        "total_ce_volume": pcr_data.get("total_ce_volume", 0),
        "total_pe_volume": pcr_data.get("total_pe_volume", 0),
        "net_oi_change_ce": pcr_data.get("net_oi_change_ce", 0),
        "net_oi_change_pe": pcr_data.get("net_oi_change_pe", 0),

        # IV
        "atm_iv": iv_data.get("atm_iv"),
        "atm_iv_pct": iv_data.get("atm_iv_pct"),
        "atm_iv_valid": iv_data.get("atm_iv_valid", False),
        "ce_iv": iv_data.get("ce_iv"),
        "pe_iv": iv_data.get("pe_iv"),
        "iv_skew": iv_data.get("iv_skew"),
        "iv_skew_pct": iv_data.get("iv_skew_pct"),
        "iv_rank_session": iv_data.get("iv_rank_session"),
        "atm_strike_used": iv_data.get("atm_strike_used"),

        # Max Pain
        "max_pain": mp_data.get("max_pain"),
        "max_pain_distance": mp_data.get("max_pain_distance"),
        "max_pain_valid": mp_data.get("max_pain_valid", False),

        # OI Walls
        "highest_ce_oi_strike": walls.get("highest_ce_oi_strike"),
        "highest_ce_oi": walls.get("highest_ce_oi"),
        "highest_pe_oi_strike": walls.get("highest_pe_oi_strike"),
        "highest_pe_oi": walls.get("highest_pe_oi"),
        "ce_wall_distance": walls.get("ce_wall_distance"),
        "pe_wall_distance": walls.get("pe_wall_distance"),
        "oi_walls_valid": walls.get("oi_walls_valid", False),

        # Synthetic & GEX
        "synthetic_future": synth.get("synthetic_future"),
        "synthetic_premium": synth.get("synthetic_premium"),
        "gex_proxy": synth.get("gex_proxy"),
        "gex_regime": synth.get("gex_regime"),
        "gex_available": synth.get("gex_available", False),

        # OI Classification (Blueprint)
        "oi_classification": oi_cls,

        # Metadata
        "strikes_in_chain": len(chain),
        "spot_price": float(spot_price),

        # Data quality
        "data_quality": dq,
    }

    if dq["status"] != "OK":
        _logger.warning("Options features degraded", extra={"reason": dq["reason"], "dq": dq})

    return result


# ============================================
# Utility Functions
# ============================================

def _safe_float(value, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
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
        "pcr_oi": None,
        "pcr_volume": None,
        "pcr_change": None,
        "total_ce_oi": 0,
        "total_pe_oi": 0,
        "total_ce_volume": 0,
        "total_pe_volume": 0,
        "net_oi_change_ce": 0,
        "net_oi_change_pe": 0,

        "atm_iv": None,
        "atm_iv_pct": None,
        "atm_iv_valid": False,
        "ce_iv": None,
        "pe_iv": None,
        "iv_skew": None,
        "iv_skew_pct": None,
        "iv_rank_session": None,
        "atm_strike_used": None,

        "max_pain": None,
        "max_pain_distance": None,
        "max_pain_valid": False,

        "highest_ce_oi_strike": None,
        "highest_ce_oi": None,
        "highest_pe_oi_strike": None,
        "highest_pe_oi": None,
        "ce_wall_distance": None,
        "pe_wall_distance": None,
        "oi_walls_valid": False,

        "synthetic_future": None,
        "synthetic_premium": None,
        "gex_proxy": None,
        "gex_regime": "NEUTRAL",
        "gex_available": False,

        "oi_classification": {"available": False, "summary": {}, "net_score": 0, "details": {}},

        "strikes_in_chain": 0,
        "spot_price": 0.0,

        "data_quality": {
            "chain_valid": False,
            "oi_available": False,
            "iv_available": False,
            "max_pain_available": False,
            "gex_available": False,
            "oi_classification_available": False,
            "status": "INVALID",
            "reason": "empty",
            "expiry": None,
        },
    }


# ============================================
# Module Self-Test
# ============================================

def _build_test_chain(
    atm: int = 22400,
    strikes_range: int = 5,
    interval: int = 50,
) -> Dict:
    chain = {}
    for i in range(-strikes_range, strikes_range + 1):
        strike = atm + i * interval
        distance = abs(i)
        ce_ltp = max(1, 150 - distance * 25 + (5 - i) * 3)
        pe_ltp = max(1, 150 - distance * 25 + (5 + i) * 3)

        ce_oi = max(0, 5000000 - distance * 800000 + i * 200000)
        pe_oi = max(0, 4500000 - distance * 700000 - i * 200000)

        ce_iv = max(0.05, 0.15 - distance * 0.005 + 0.002 * i)
        pe_iv = max(0.05, 0.14 - distance * 0.005 - 0.002 * i)

        gamma = max(0.0001, 0.002 - distance * 0.0002)

        chain[strike] = {
            "ce": {"ltp": ce_ltp, "iv": ce_iv, "oi": ce_oi, "volume": 0, "oi_change": 0, "gamma": gamma},
            "pe": {"ltp": pe_ltp, "iv": pe_iv, "oi": pe_oi, "volume": 0, "oi_change": 0, "gamma": gamma},
        }
    return chain


def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Options Features Test (Final Polish)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    spot = 22400.0
    chain = _build_test_chain(atm=22400)

    # Test 1: PCR stale timestamp should block pcr_change
    print("  [Test 1] PCR change stale timestamp guard...")
    prev_pcr = 0.9
    stale_ts = datetime.now(IST) - timedelta(minutes=120)
    f = compute_options_features(chain, spot_price=spot, previous_pcr_oi=prev_pcr, previous_pcr_timestamp=stale_ts)
    if f["pcr_change"] is None:
        print("    ✅ Stale pcr_change blocked")
        passed += 1
    else:
        print(f"    ❌ pcr_change should be None when stale, got {f['pcr_change']}")
        failed += 1

    # Test 2: OI classification with previous chain
    print("\n  [Test 2] OI classification...")
    prev_chain = _build_test_chain(atm=22400)
    # Create a clear LONG_BUILDUP on ATM CE: price up and OI up
    atm = 22400
    prev_chain[atm]["ce"]["ltp"] = 100.0
    prev_chain[atm]["ce"]["oi"] = 1000
    chain2 = _build_test_chain(atm=22400)
    chain2[atm]["ce"]["ltp"] = 110.0
    chain2[atm]["ce"]["oi"] = 1200

    f2 = compute_options_features(chain2, spot_price=spot, previous_chain=prev_chain)
    cls = f2["oi_classification"]
    if cls.get("available") and cls.get("summary", {}).get("LONG_BUILDUP", 0) >= 1:
        print("    ✅ OI classification computed with LONG_BUILDUP present")
        passed += 1
    else:
        print(f"    ❌ OI classification missing/incorrect: {cls}")
        failed += 1

    # Test 3: Mixed expiry detection (if expiry keys exist)
    print("\n  [Test 3] Mixed expiry detection (optional)...")
    mixed = _build_test_chain(atm=22400)
    # Inject expiry keys
    for i, k in enumerate(list(mixed.keys())[:2]):
        mixed[k]["expiry"] = "2026-04-01" if i == 0 else "2026-04-08"
    f3 = compute_options_features(mixed, spot_price=spot)
    if f3["data_quality"]["status"] == "INVALID":
        print("    ✅ Mixed expiry detected and blocked")
        passed += 1
    else:
        print(f"    ❌ Mixed expiry should be invalid, got {f3['data_quality']}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n  ✅ Options Features fully production-ready!")
    else:
        print(f"\n  ⚠️  {failed} tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()