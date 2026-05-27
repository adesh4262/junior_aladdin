"""
Junior Aladdin - Mode Manager
=============================
Manage operational mode transitions: ALERT / PAPER / LIVE.

LIVE mode guardrails:
- Algo-ID must exist.
- Registered static IP must match current IP.
- ComplianceGuard must approve a synthetic LIVE validation order.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.execution.compliance import ComplianceGuard


class ModeManager:
    VALID_MODES = {"ALERT", "PAPER", "LIVE"}

    def __init__(self, initial_mode: Optional[str] = None) -> None:
        self._log = setup_logger("mode_manager")
        self._compliance = ComplianceGuard()

        env_mode = os.getenv("JUNIOR_ALADDIN_MODE")
        cfg_mode = Config.get("system", "mode", default="ALERT")
        requested = initial_mode if initial_mode is not None else (env_mode if env_mode is not None else cfg_mode)

        normalized = self._normalize_mode(requested)
        self._mode = normalized if normalized is not None else "ALERT"

    def get_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> bool:
        normalized = self._normalize_mode(mode)
        if normalized is None:
            self._log.warning("Mode change rejected", requested_mode=str(mode), reason="invalid_mode")
            return False

        if normalized == "LIVE":
            ok, reason = self._validate_live_requirements()
            if not ok:
                self._log.warning("LIVE mode rejected", reason=reason)
                return False

        prev = self._mode
        self._mode = normalized
        self._log.info("Mode updated", previous_mode=prev, current_mode=self._mode)
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "valid_modes": sorted(list(self.VALID_MODES)),
        }

    def _validate_live_requirements(self) -> tuple[bool, str]:
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

        probe_order = {
            "symbol": "NIFTY",
            "qty": 1,
            "price": 100.0,
            "direction": "BUY",
            "order_type": "LIMIT",
            "algo_id": algo_id,
        }

        decision = self._compliance.validate_order(
            order=probe_order,
            mode="LIVE",
            current_ip=current_ip,
            registered_ip=registered_ip,
        )
        if not decision.allow:
            return False, decision.reason
        return True, "ok"

    @classmethod
    def _normalize_mode(cls, mode: Any) -> Optional[str]:
        m = str(mode).strip().upper() if mode is not None else ""
        if m in cls.VALID_MODES:
            return m
        return None


def _run_tests() -> None:
    print("=" * 60)
    print(" JUNIOR ALADDIN - Mode Manager Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    print("[Test 1] Default mode from constructor...")
    mm = ModeManager(initial_mode="ALERT")
    if mm.get_mode() == "ALERT":
        print("  PASS: default mode ALERT")
        passed += 1
    else:
        print("  FAIL: expected ALERT")
        failed += 1

    print("\n[Test 2] Switch ALERT -> PAPER...")
    if mm.set_mode("PAPER") and mm.get_mode() == "PAPER":
        print("  PASS: switched to PAPER")
        passed += 1
    else:
        print("  FAIL: PAPER mode switch failed")
        failed += 1

    print("\n[Test 3] Reject invalid mode...")
    if (not mm.set_mode("INVALID")) and mm.get_mode() == "PAPER":
        print("  PASS: invalid mode rejected")
        passed += 1
    else:
        print("  FAIL: invalid mode handling failed")
        failed += 1

    print("\n[Test 4] LIVE mode blocked without env requirements...")
    old_algo = os.getenv("JUNIOR_ALADDIN_ALGO_ID")
    old_reg = os.getenv("JUNIOR_ALADDIN_REGISTERED_IP")
    old_cur = os.getenv("JUNIOR_ALADDIN_CURRENT_IP")

    try:
        if "JUNIOR_ALADDIN_ALGO_ID" in os.environ:
            del os.environ["JUNIOR_ALADDIN_ALGO_ID"]
        if "JUNIOR_ALADDIN_REGISTERED_IP" in os.environ:
            del os.environ["JUNIOR_ALADDIN_REGISTERED_IP"]
        if "JUNIOR_ALADDIN_CURRENT_IP" in os.environ:
            del os.environ["JUNIOR_ALADDIN_CURRENT_IP"]

        blocked = not mm.set_mode("LIVE")
        if blocked:
            print("  PASS: LIVE blocked when requirements missing")
            passed += 1
        else:
            print("  FAIL: LIVE should be blocked")
            failed += 1

        print("\n[Test 5] LIVE mode allowed with valid algo/ip...")
        os.environ["JUNIOR_ALADDIN_ALGO_ID"] = "JA-LIVE-001"
        os.environ["JUNIOR_ALADDIN_REGISTERED_IP"] = "1.1.1.1"
        os.environ["JUNIOR_ALADDIN_CURRENT_IP"] = "1.1.1.1"

        if mm.set_mode("LIVE") and mm.get_mode() == "LIVE":
            print("  PASS: LIVE mode enabled")
            passed += 1
        else:
            print("  FAIL: LIVE mode should be enabled")
            failed += 1

    finally:
        if old_algo is None:
            os.environ.pop("JUNIOR_ALADDIN_ALGO_ID", None)
        else:
            os.environ["JUNIOR_ALADDIN_ALGO_ID"] = old_algo

        if old_reg is None:
            os.environ.pop("JUNIOR_ALADDIN_REGISTERED_IP", None)
        else:
            os.environ["JUNIOR_ALADDIN_REGISTERED_IP"] = old_reg

        if old_cur is None:
            os.environ.pop("JUNIOR_ALADDIN_CURRENT_IP", None)
        else:
            os.environ["JUNIOR_ALADDIN_CURRENT_IP"] = old_cur

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
