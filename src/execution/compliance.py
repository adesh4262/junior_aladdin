"""
Junior Aladdin - Compliance Guard
=================================
PURPOSE:
Centralize execution compliance checks for the trading system.

This file exists because the roadmap explicitly expects:
    src/execution/compliance.py

RESPONSIBILITIES:
- enforce allowed order types
- require Algo-ID where needed
- validate static IP match for LIVE mode
- track orders-per-second ceiling
- provide a single compliance decision object for execution layer

IMPORTANT:
This module does NOT place orders.
It only validates whether an order is compliant.

CONNECTS TO:
- broker_interface / live broker
- paper broker
- execution engine
- captain / mode manager
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
import math
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

IST = timezone(timedelta(hours=5, minutes=30))
_logger = setup_logger("compliance_guard")


@dataclass
class ComplianceDecision:
    """
    Result of compliance validation.
    """
    allow: bool
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": self.allow,
            "reason": self.reason,
            "warnings": self.warnings,
        }


class ComplianceGuard:
    """
    Execution compliance validator.
    """

    def __init__(self):
        self._logger = _logger

        allowed_types_config = Config.get(
            "compliance",
            "allowed_order_types",
            default=["LIMIT", "STOPLOSS_LIMIT"],
        )
        if isinstance(allowed_types_config, list):
            normalized_allowed_types = [
                str(order_type).strip().upper()
                for order_type in allowed_types_config
                if str(order_type).strip()
            ]
            self._allowed_order_types = normalized_allowed_types or ["LIMIT", "STOPLOSS_LIMIT"]
        else:
            self._allowed_order_types = ["LIMIT", "STOPLOSS_LIMIT"]

        max_ops_config = Config.get(
            "compliance",
            "max_orders_per_second",
            default=9,
        )
        self._max_orders_per_second = self._safe_int(max_ops_config, default=9)
        if self._max_orders_per_second < 1:
            self._max_orders_per_second = 9

        self._recent_order_timestamps: deque = deque()

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except Exception:
            return default

    def _reject(
        self,
        reason: str,
        warnings: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> ComplianceDecision:
        payload: Dict[str, Any] = {"reason": reason}
        if extra:
            payload.update(extra)
        self._logger.warning("Compliance validation rejected", extra=payload)
        return ComplianceDecision(False, reason, warnings)

    # ------------------------------------------------------------------
    # Main validation
    # ------------------------------------------------------------------
    def validate_order(
        self,
        order: Dict[str, Any],
        mode: str = "PAPER",
        current_ip: Optional[str] = None,
        registered_ip: Optional[str] = None,
    ) -> ComplianceDecision:
        """
        Validate one order against compliance rules.
        """
        warnings: List[str] = []

        mode_value = str(mode or "").strip().upper()
        if not isinstance(order, dict):
            return self._reject(
                "invalid_order_format",
                warnings,
                extra={
                    "mode": mode_value,
                    "received_type": type(order).__name__,
                },
            )

        order_type_raw = order.get("order_type", "")
        algo_id_raw = order.get("algo_id", "")
        qty_raw = order.get("qty", 0)
        price_raw = order.get("price", 0)
        symbol_raw = order.get("symbol", "")
        direction_raw = order.get("direction", "")

        order_type = str(order_type_raw).strip().upper() if order_type_raw is not None else ""
        algo_id = str(algo_id_raw).strip() if algo_id_raw is not None else ""
        qty = self._safe_int(qty_raw, default=0)
        price = self._safe_float(price_raw, default=0.0)
        symbol = str(symbol_raw).strip() if symbol_raw is not None else ""
        direction = str(direction_raw).strip().upper() if direction_raw is not None else ""

        # Basic structure checks
        if not symbol:
            return self._reject("missing_symbol", warnings, extra={"mode": mode_value})
        if not order_type:
            return self._reject("missing_order_type", warnings, extra={"symbol": symbol, "mode": mode_value})
        if not direction:
            return self._reject("invalid_direction", warnings, extra={"symbol": symbol, "mode": mode_value})
        if qty <= 0:
            return self._reject(
                "invalid_qty",
                warnings,
                extra={"symbol": symbol, "qty": qty_raw, "mode": mode_value},
            )
        if direction not in ("BUY", "SELL"):
            return self._reject(
                "invalid_direction",
                warnings,
                extra={"symbol": symbol, "direction": direction_raw, "mode": mode_value},
            )
        if price <= 0:
            return self._reject(
                "invalid_price",
                warnings,
                extra={"symbol": symbol, "price": price_raw, "mode": mode_value},
            )

        # Order type restriction
        if order_type not in self._allowed_order_types:
            return self._reject(
                f"order_type_not_allowed:{order_type}",
                warnings,
                extra={"symbol": symbol, "mode": mode_value},
            )

        # LIVE mode strict checks
        if mode_value == "LIVE":
            if not algo_id:
                return self._reject("missing_algo_id", warnings, extra={"symbol": symbol, "mode": mode_value})

            if registered_ip and current_ip and registered_ip != current_ip:
                return self._reject(
                    "static_ip_mismatch",
                    warnings,
                    extra={
                        "symbol": symbol,
                        "mode": mode_value,
                        "registered_ip": registered_ip,
                        "current_ip": current_ip,
                    },
                )

            ops_ok = self._check_ops_limit()
            if not ops_ok:
                return self._reject(
                    "max_orders_per_second_exceeded",
                    warnings,
                    extra={"symbol": symbol, "mode": mode_value},
                )

        else:
            # Paper/alert warnings only
            if not algo_id:
                warnings.append("algo_id_missing_non_live")

        self._logger.info(
            "Compliance validation passed",
            extra={
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "direction": direction,
                "order_type": order_type,
                "mode": mode_value,
                "warnings": warnings,
            },
        )

        return ComplianceDecision(True, "", warnings)

    # ------------------------------------------------------------------
    # Internal rate limiting
    # ------------------------------------------------------------------
    def _check_ops_limit(self) -> bool:
        """
        Enforce max orders-per-second rule.
        """
        limit = self._safe_int(self._max_orders_per_second, default=1)
        if limit < 1:
            limit = 1

        now = datetime.now(IST)

        while self._recent_order_timestamps:
            age = (now - self._recent_order_timestamps[0]).total_seconds()
            if age > 1.0:
                self._recent_order_timestamps.popleft()
            else:
                break

        if len(self._recent_order_timestamps) >= limit:
            self._logger.warning(
                "OPS compliance breached",
                extra={
                    "orders_in_last_second": len(self._recent_order_timestamps),
                    "limit": limit,
                },
            )
            return False

        self._recent_order_timestamps.append(now)
        return True

    # ------------------------------------------------------------------
    # Helpers / status
    # ------------------------------------------------------------------
    def reset(self):
        self._recent_order_timestamps.clear()
        self._logger.info("Compliance guard reset")

    def get_status(self) -> Dict[str, Any]:
        return {
            "allowed_order_types": list(self._allowed_order_types),
            "max_orders_per_second": self._max_orders_per_second,
            "recent_orders_in_window": len(self._recent_order_timestamps),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Compliance Guard Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    guard = ComplianceGuard()

    print(" [Test 1] Valid PAPER order...")
    o1 = {
        "symbol": "NIFTY22450CE",
        "qty": 65,
        "price": 100.0,
        "direction": "BUY",
        "order_type": "LIMIT",
        "algo_id": "",
    }
    r1 = guard.validate_order(o1, mode="PAPER")
    if r1.allow:
        print(f" ✅ PAPER order allowed: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ PAPER order rejected: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Invalid order type blocked...")
    o2 = dict(o1)
    o2["order_type"] = "MARKET"
    r2 = guard.validate_order(o2, mode="LIVE", current_ip="1.1.1.1", registered_ip="1.1.1.1")
    if not r2.allow and "order_type_not_allowed" in r2.reason:
        print(f" ✅ Invalid type blocked: {r2.reason}")
        passed += 1
    else:
        print(f" ❌ Invalid type check failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Missing Algo-ID blocked in LIVE...")
    o3 = dict(o1)
    o3["order_type"] = "LIMIT"
    o3["algo_id"] = ""
    r3 = guard.validate_order(o3, mode="LIVE", current_ip="1.1.1.1", registered_ip="1.1.1.1")
    if not r3.allow and r3.reason == "missing_algo_id":
        print(f" ✅ Missing Algo-ID blocked")
        passed += 1
    else:
        print(f" ❌ Algo-ID check failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Static IP mismatch blocked...")
    o4 = dict(o1)
    o4["algo_id"] = "JA-001"
    r4 = guard.validate_order(o4, mode="LIVE", current_ip="2.2.2.2", registered_ip="1.1.1.1")
    if not r4.allow and r4.reason == "static_ip_mismatch":
        print(" ✅ Static IP mismatch blocked")
        passed += 1
    else:
        print(f" ❌ Static IP check failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Valid LIVE order passes...")
    guard.reset()
    o5 = dict(o1)
    o5["algo_id"] = "JA-001"
    r5 = guard.validate_order(o5, mode="LIVE", current_ip="1.1.1.1", registered_ip="1.1.1.1")
    if r5.allow:
        print(f" ✅ LIVE order allowed: {r5.to_dict()}")
        passed += 1
    else:
        print(f" ❌ LIVE order rejected: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] OPS limit enforcement...")
    guard.reset()
    allowed_count = 0
    blocked = False
    for i in range(guard._max_orders_per_second + 1):
        oi = {
            "symbol": f"NIFTY{i}",
            "qty": 1,
            "price": 100,
            "direction": "BUY",
            "order_type": "LIMIT",
            "algo_id": "JA-OPS",
        }
        ri = guard.validate_order(oi, mode="LIVE", current_ip="1.1.1.1", registered_ip="1.1.1.1")
        if ri.allow:
            allowed_count += 1
        else:
            if ri.reason == "max_orders_per_second_exceeded":
                blocked = True
                break
    if blocked:
        print(f" ✅ OPS limit blocked extra order after {allowed_count} allowed")
        passed += 1
    else:
        print(" ❌ OPS limit failed")
        failed += 1

    print("\n [Test 7] Invalid fields blocked...")
    bad = {
        "symbol": "",
        "qty": 0,
        "price": 0,
        "direction": "WHATEVER",
        "order_type": "LIMIT",
        "algo_id": "X",
    }
    r7 = guard.validate_order(bad, mode="PAPER")
    if not r7.allow:
        print(f" ✅ Invalid fields blocked: {r7.reason}")
        passed += 1
    else:
        print(f" ❌ Invalid fields should fail: {r7.to_dict()}")
        failed += 1

    print("\n [Test 8] Status and reset...")
    guard.reset()
    st = guard.get_status()
    if st["recent_orders_in_window"] == 0:
        print(f" ✅ Status/reset works: {st}")
        passed += 1
    else:
        print(f" ❌ Reset/status failed: {st}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Compliance Guard working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()