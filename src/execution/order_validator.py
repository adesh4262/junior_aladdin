"""
Junior Aladdin - Order Validator
================================
PURPOSE:
Validate raw order payloads before they are sent to broker/compliance layers.

This file exists because the roadmap explicitly expects:
    src/execution/order_validator.py

RESPONSIBILITIES:
- validate field presence
- validate numeric sanity
- validate direction / order type
- validate stop-loss trigger-price relationships
- produce clear error messages for upstream engines

CONNECTS TO:
- Compliance Guard
- Broker Interface / Paper Broker / Live Broker
- Captain / Execution engine
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, List

from src.utils.logger import setup_logger

_logger = setup_logger("order_validator")


@dataclass
class OrderValidationResult:
    valid: bool
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "warnings": self.warnings,
        }


class OrderValidator:
    """
    Validate order payload structure and execution sanity.
    """

    ALLOWED_DIRECTIONS = {"BUY", "SELL"}
    ALLOWED_ORDER_TYPES = {"LIMIT", "STOPLOSS_LIMIT"}

    def __init__(self):
        self._logger = _logger

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

    def _reject(self, reason: str, warnings: List[str]) -> OrderValidationResult:
        self._logger.warning(
            "Order validation rejected",
            extra={"reason": reason},
        )
        return OrderValidationResult(False, reason, warnings)

    def validate(self, order: Dict[str, Any]) -> OrderValidationResult:
        """
        Validate one order dict.

        Expected fields:
        - symbol
        - qty
        - price
        - direction
        - order_type

        For STOPLOSS_LIMIT:
        - trigger_price also required
        """
        warnings: List[str] = []

        if not isinstance(order, dict):
            return self._reject("order_not_dict", warnings)

        symbol = str(order.get("symbol", "")).strip()
        direction = str(order.get("direction", "")).strip().upper()
        order_type = str(order.get("order_type", "")).strip().upper()

        qty = order.get("qty", None)
        price = order.get("price", None)
        trigger_price = order.get("trigger_price", None)

        if not symbol:
            return self._reject("missing_symbol", warnings)

        qty_type_probe = self._safe_float(qty, default=float("nan"))
        qty = self._safe_int(qty, default=0)
        if math.isnan(qty_type_probe) or math.isinf(qty_type_probe):
            return self._reject("invalid_qty_type", warnings)

        if qty <= 0:
            return self._reject("qty_must_be_positive", warnings)

        price_type_probe = self._safe_float(price, default=float("nan"))
        price = self._safe_float(price, default=0.0)
        if math.isnan(price_type_probe) or math.isinf(price_type_probe):
            return self._reject("invalid_price_type", warnings)

        if price <= 0 or math.isnan(price) or math.isinf(price):
            return self._reject("price_must_be_positive", warnings)

        if direction not in self.ALLOWED_DIRECTIONS:
            return self._reject(f"invalid_direction:{direction}", warnings)

        if order_type not in self.ALLOWED_ORDER_TYPES:
            return self._reject(f"invalid_order_type:{order_type}", warnings)

        if order_type == "STOPLOSS_LIMIT":
            trigger_type_probe = self._safe_float(trigger_price, default=float("nan"))
            trigger_price = self._safe_float(trigger_price, default=0.0)
            if math.isnan(trigger_type_probe) or math.isinf(trigger_type_probe):
                return self._reject("invalid_trigger_price_type", warnings)

            if trigger_price <= 0 or math.isnan(trigger_price) or math.isinf(trigger_price):
                return self._reject("trigger_price_must_be_positive", warnings)

            trigger_ok = self._validate_trigger_logic(
                direction=direction,
                trigger_price=trigger_price,
                limit_price=price,
            )
            if not trigger_ok.valid:
                return trigger_ok

        # soft warnings
        if qty > 10000:
            warnings.append("very_large_quantity")

        if price < 1:
            warnings.append("very_low_price")

        result = OrderValidationResult(True, "", warnings)

        self._logger.info(
            "Order validation passed",
            extra={
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "direction": direction,
                "order_type": order_type,
                "warnings": warnings,
            },
        )

        return result

    def _validate_trigger_logic(
        self,
        direction: str,
        trigger_price: float,
        limit_price: float,
    ) -> OrderValidationResult:
        """
        Validate stop-loss-limit trigger logic.

        Basic practical assumptions:
        - BUY stop-loss-limit:
            trigger usually <= limit_price
        - SELL stop-loss-limit:
            trigger usually >= limit_price
        """
        warnings: List[str] = []

        if direction == "BUY":
            if trigger_price > limit_price:
                return self._reject("buy_sl_trigger_above_limit_price", warnings)

        elif direction == "SELL":
            if trigger_price < limit_price:
                return self._reject("sell_sl_trigger_below_limit_price", warnings)

        return OrderValidationResult(True, "", warnings)


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Order Validator Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    ov = OrderValidator()

    print(" [Test 1] Valid LIMIT BUY...")
    o1 = {
        "symbol": "NIFTY22450CE",
        "qty": 65,
        "price": 100.0,
        "direction": "BUY",
        "order_type": "LIMIT",
    }
    r1 = ov.validate(o1)
    if r1.valid:
        print(f" ✅ LIMIT BUY valid: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ LIMIT BUY invalid: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Valid SL-LIMIT SELL...")
    o2 = {
        "symbol": "NIFTY22450PE",
        "qty": 65,
        "price": 99.0,
        "trigger_price": 100.0,
        "direction": "SELL",
        "order_type": "STOPLOSS_LIMIT",
    }
    r2 = ov.validate(o2)
    if r2.valid:
        print(f" ✅ STOPLOSS_LIMIT SELL valid: {r2.to_dict()}")
        passed += 1
    else:
        print(f" ❌ STOPLOSS_LIMIT SELL invalid: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Missing symbol...")
    o3 = {
        "symbol": "",
        "qty": 65,
        "price": 100.0,
        "direction": "BUY",
        "order_type": "LIMIT",
    }
    r3 = ov.validate(o3)
    if not r3.valid and r3.reason == "missing_symbol":
        print(f" ✅ Missing symbol blocked")
        passed += 1
    else:
        print(f" ❌ Missing symbol check failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Bad quantity...")
    o4 = {
        "symbol": "X",
        "qty": 0,
        "price": 100.0,
        "direction": "BUY",
        "order_type": "LIMIT",
    }
    r4 = ov.validate(o4)
    if not r4.valid and r4.reason == "qty_must_be_positive":
        print(f" ✅ Bad qty blocked")
        passed += 1
    else:
        print(f" ❌ Qty validation failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Bad direction...")
    o5 = {
        "symbol": "X",
        "qty": 1,
        "price": 100.0,
        "direction": "UP",
        "order_type": "LIMIT",
    }
    r5 = ov.validate(o5)
    if not r5.valid and "invalid_direction" in r5.reason:
        print(f" ✅ Bad direction blocked")
        passed += 1
    else:
        print(f" ❌ Direction validation failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Bad order type...")
    o6 = {
        "symbol": "X",
        "qty": 1,
        "price": 100.0,
        "direction": "BUY",
        "order_type": "MARKET",
    }
    r6 = ov.validate(o6)
    if not r6.valid and "invalid_order_type" in r6.reason:
        print(f" ✅ Bad order type blocked")
        passed += 1
    else:
        print(f" ❌ Order type validation failed: {r6.to_dict()}")
        failed += 1

    print("\n [Test 7] Missing trigger on SL order...")
    o7 = {
        "symbol": "X",
        "qty": 1,
        "price": 100.0,
        "direction": "SELL",
        "order_type": "STOPLOSS_LIMIT",
    }
    r7 = ov.validate(o7)
    if not r7.valid and r7.reason == "invalid_trigger_price_type":
        print(f" ✅ Missing trigger blocked")
        passed += 1
    else:
        print(f" ❌ Trigger validation failed: {r7.to_dict()}")
        failed += 1

    print("\n [Test 8] Wrong BUY trigger relationship...")
    o8 = {
        "symbol": "X",
        "qty": 1,
        "price": 100.0,
        "trigger_price": 101.0,
        "direction": "BUY",
        "order_type": "STOPLOSS_LIMIT",
    }
    r8 = ov.validate(o8)
    if not r8.valid and r8.reason == "buy_sl_trigger_above_limit_price":
        print(f" ✅ BUY trigger relation blocked")
        passed += 1
    else:
        print(f" ❌ BUY trigger relation failed: {r8.to_dict()}")
        failed += 1

    print("\n [Test 9] Wrong SELL trigger relationship...")
    o9 = {
        "symbol": "X",
        "qty": 1,
        "price": 100.0,
        "trigger_price": 99.0,
        "direction": "SELL",
        "order_type": "STOPLOSS_LIMIT",
    }
    r9 = ov.validate(o9)
    if not r9.valid and r9.reason == "sell_sl_trigger_below_limit_price":
        print(f" ✅ SELL trigger relation blocked")
        passed += 1
    else:
        print(f" ❌ SELL trigger relation failed: {r9.to_dict()}")
        failed += 1

    print("\n [Test 10] Large qty warning...")
    o10 = {
        "symbol": "BIGQTY",
        "qty": 20000,
        "price": 100.0,
        "direction": "BUY",
        "order_type": "LIMIT",
    }
    r10 = ov.validate(o10)
    if r10.valid and "very_large_quantity" in r10.warnings:
        print(f" ✅ Large qty warning works: {r10.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Large qty warning failed: {r10.to_dict()}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Order Validator working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()