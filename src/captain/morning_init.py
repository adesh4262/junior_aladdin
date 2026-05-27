"""
Junior Aladdin - Morning Init
=============================
Pre-market initialization workflow.

Responsibilities:
- Fetch FII/DII and global market data.
- Load economic calendar JSON.
- Compute initial narrative score.
- Load ML model artifacts (LightGBM, GARCH) if present.
- Load Strategy DNA from database.
- Verify static IP and Algo-ID readiness for LIVE.
- Reset daily counters and set system state to BOOT.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.core.market_state import MarketState
from src.core.nse_fetcher import NSEFetcher
from src.features.fundamental import compute_fundamental_features
from src.context.narrative_engine import NarrativeEngine
from src.utils.database import Database
from src.execution.compliance import ComplianceGuard


_log = setup_logger("morning_init")


def run_morning_init(market_state: Optional[MarketState] = None) -> bool:
    _log.info("Morning init started")
    success = True

    if market_state is not None and not isinstance(market_state, MarketState):
        _log.warning(
            "Ignoring invalid market_state override; creating a fresh MarketState",
            override_type=str(type(market_state)),
        )
        market_state = None

    ms = market_state if market_state is not None else MarketState()
    system_mode = _normalize_mode(
        str(os.getenv("JUNIOR_ALADDIN_MODE") or Config.get("system", "mode", default="ALERT"))
    )

    # 1) External data pull
    fetcher = NSEFetcher()

    _log.info("Step 1/8: Fetch FII/DII data")
    fii_data = fetcher.fetch_fii_dii()
    fii_ok = bool(fii_data.get("success", False))
    if not fii_ok:
        _log.warning("FII/DII fetch not successful", source=fii_data.get("source"), error=fii_data.get("error"))

    _log.info("Step 2/8: Fetch global market data")
    global_data = fetcher.fetch_global_data()
    global_ok = bool(global_data.get("success", False))
    if not global_ok:
        _log.warning(
            "Global data fetch not successful",
            source=global_data.get("source"),
            critical_ok=global_data.get("critical_ok"),
            errors=global_data.get("errors"),
        )

    external_quality = fetcher.get_external_data_quality()
    _log.info(
        "External data quality summary",
        overall_quality_score=external_quality.get("overall_quality_score"),
        nse_available=external_quality.get("nse_available"),
        global_critical_ok=external_quality.get("global_critical_ok"),
        stale_age_days_max=external_quality.get("stale_age_days_max"),
        fii_source=external_quality.get("fii_source"),
        global_source=external_quality.get("global_source"),
    )
    if int(external_quality.get("overall_quality_score", 0) or 0) <= 0:
        success = False
        _log.warning("External data quality unavailable", quality=external_quality)

    # 2) Economic calendar
    _log.info("Step 3/8: Load economic calendar JSON")
    calendar_path = str(
        Config.get(
            "fundamental",
            "economic_calendar_path",
            default="data/calendar/economic_calendar.json",
        )
    )
    cal_ok, cal_count = _load_calendar_json(calendar_path)
    if not cal_ok:
        success = False
        _log.warning("Economic calendar not available", calendar_path=calendar_path)
    else:
        _log.info("Economic calendar loaded", calendar_path=calendar_path, entries=cal_count)

    # 3) Narrative seed
    _log.info("Step 4/8: Compute initial narrative score")
    try:
        fundamental = compute_fundamental_features(
            fii_data=fii_data,
            global_data=global_data,
            calendar_path=calendar_path,
        )
        narrative_engine = NarrativeEngine()
        narrative_score = float(narrative_engine.compute(fundamental))
        narrative_label = str(narrative_engine.narrative_label)
        _log.info("Narrative computed", narrative_score=narrative_score, narrative_label=narrative_label)
    except Exception as e:
        success = False
        narrative_score = 0.0
        narrative_label = "NEUTRAL"
        _log.error("Narrative computation failed", error=str(e))

    # 4) ML models
    _log.info("Step 5/8: Load ML models (LightGBM/GARCH) if available")
    model_load = _load_ml_models("models")
    _log.info(
        "ML model load summary",
        lightgbm_found=model_load["lightgbm_found"],
        lightgbm_loaded=model_load["lightgbm_loaded"],
        garch_found=model_load["garch_found"],
        garch_loaded=model_load["garch_loaded"],
    )

    # 5) Strategy DNA
    _log.info("Step 6/8: Load Strategy DNA from database")
    dna_loaded, dna_count = _load_strategy_dna_count()
    if dna_loaded:
        _log.info("Strategy DNA loaded", records=dna_count)
    else:
        success = False
        _log.warning("Strategy DNA load failed")

    # 6) Compliance pre-check for LIVE readiness
    _log.info("Step 7/8: Verify static IP and Algo-ID readiness")
    enforce_live = system_mode == "LIVE"
    ip_ok, ip_reason = _verify_live_identity_with_compliance(enforce=enforce_live)
    if enforce_live:
        if not ip_ok:
            success = False
            _log.warning("LIVE readiness check failed", reason=ip_reason)
        else:
            _log.info("LIVE readiness check passed")
    else:
        _log.info("LIVE readiness check skipped for non-LIVE mode", mode=system_mode, reason=ip_reason)

    # 7) Reset state and publish BOOT
    _log.info("Step 8/8: Reset daily counters and set BOOT state")
    ms.reset_daily()
    mode = _normalize_mode(str(Config.get("system", "mode", default="ALERT")))
    ms.update(
        system_state="BOOT",
        mode=mode,
        narrative_score=narrative_score,
        narrative_label=narrative_label,
    )

    _log.info("Morning init completed", success=success, mode=mode)
    return success


def _load_calendar_json(calendar_path: str) -> Tuple[bool, int]:
    try:
        p = Path(calendar_path)
        if not p.exists() or not p.is_file():
            return False, 0
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return True, len(payload)
        if isinstance(payload, dict):
            return True, len(payload)
        return False, 0
    except Exception:
        return False, 0


def _load_ml_models(models_dir: str) -> Dict[str, Any]:
    root = Path(models_dir)
    result: Dict[str, Any] = {
        "lightgbm_found": False,
        "lightgbm_loaded": False,
        "garch_found": False,
        "garch_loaded": False,
        "loaded_files": [],
    }

    if not root.exists() or not root.is_dir():
        return result

    lightgbm_files = _collect_model_files(root, ["*lightgbm*.pkl", "*lightgbm*.joblib", "*lightgbm*.bin"])
    garch_files = _collect_model_files(root, ["*garch*.pkl", "*garch*.joblib", "*garch*.bin"])

    result["lightgbm_found"] = len(lightgbm_files) > 0
    result["garch_found"] = len(garch_files) > 0

    if lightgbm_files:
        loaded = _try_load_binary_file(lightgbm_files[0])
        result["lightgbm_loaded"] = loaded
        if loaded:
            result["loaded_files"].append(str(lightgbm_files[0]).replace("\\", "/"))

    if garch_files:
        loaded = _try_load_binary_file(garch_files[0])
        result["garch_loaded"] = loaded
        if loaded:
            result["loaded_files"].append(str(garch_files[0]).replace("\\", "/"))

    return result


def _collect_model_files(root: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    out = sorted([p for p in out if p.is_file()])
    return out


def _try_load_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(8)
        return len(head) > 0
    except Exception:
        return False


def _load_strategy_dna_count() -> Tuple[bool, int]:
    db = None
    try:
        db = Database()
        rows = db.fetch_all("SELECT strategy_name, current_threshold, status FROM strategy_dna")
        return True, len(rows)
    except Exception as e:
        _log.warning("Strategy DNA query failed", error=str(e))
        return False, 0
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _verify_live_identity_with_compliance(*, enforce: bool) -> Tuple[bool, str]:
    if not enforce:
        return True, "skipped_non_live_mode"

    algo_id = str(
        os.getenv("JUNIOR_ALADDIN_ALGO_ID")
        or Config.get("compliance", "algo_id", default="")
        or ""
    ).strip()
    registered_ip = str(
        os.getenv("JUNIOR_ALADDIN_REGISTERED_IP")
        or Config.get("compliance", "registered_ip", default="")
        or ""
    ).strip()
    current_ip = str(
        os.getenv("JUNIOR_ALADDIN_CURRENT_IP")
        or Config.get("compliance", "current_ip", default="")
        or ""
    ).strip()

    if not algo_id:
        return False, "missing_algo_id"
    if not registered_ip:
        return False, "missing_registered_ip"
    if not current_ip:
        return False, "missing_current_ip"

    guard = ComplianceGuard()
    decision = guard.validate_order(
        {
            "symbol": "NIFTY",
            "qty": 1,
            "price": 100.0,
            "direction": "BUY",
            "order_type": "LIMIT",
            "algo_id": algo_id,
        },
        mode="LIVE",
        current_ip=current_ip,
        registered_ip=registered_ip,
    )
    if not decision.allow:
        return False, str(decision.reason)
    return True, "ok"


def _normalize_mode(value: str) -> str:
    mode = value.strip().upper()
    if mode in {"ALERT", "PAPER", "LIVE"}:
        return mode
    return "ALERT"


def _run_tests() -> None:
    print("=" * 70)
    print(" JUNIOR ALADDIN - Morning Init Test")
    print("=" * 70)

    try:
        ok = run_morning_init()
        print(f"Morning init returned: {ok}")
        print("PASS: morning init executed without crash")
    except Exception as e:
        print(f"FAIL: morning init crashed: {e}")


if __name__ == "__main__":
    _run_tests()
