"""
Junior Aladdin - Angel One Broker Adapter
=========================================
PURPOSE:
Provide a live broker implementation for Angel One SmartAPI.

This file exists because the roadmap explicitly expects:
    src/execution/angel_one_broker.py

RESPONSIBILITIES:
- conform to BrokerInterface
- translate generic order requests into Angel One SmartAPI payloads
- expose order placement / modification / cancellation / positions
- remain thin and broker-specific, leaving validation/compliance outside

IMPORTANT:
This module assumes:
- AuthManager is already authenticated
- ComplianceGuard and OrderValidator are called BEFORE broker methods
- It does not decide whether an order is safe/compliant

CONNECTS TO:
- AuthManager
- BrokerInterface
- Compliance Guard
- Order Validator
- Execution layer / Captain
"""

from typing import Dict, List, Any, Optional

from src.execution.broker_interface import BrokerInterface
from src.utils.logger import setup_logger

_logger = setup_logger("angel_one_broker")


class AngelOneBroker(BrokerInterface):
    """
    Thin live broker adapter around Angel One SmartAPI.
    """

    def __init__(self, auth_manager):
        self._logger = _logger
        self._auth = auth_manager

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _api(self):
        api = self._auth.get_smart_api()
        if api is None:
            raise RuntimeError("SmartAPI unavailable or authentication failed")
        return api

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------
    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        direction: str,
        algo_id: str = "",
    ) -> str:
        api = self._api()

        payload = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": "",
            "transactiontype": direction.upper(),
            "exchange": "NFO",
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": str(price),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(qty),
            "tag": algo_id,
        }

        response = api.placeOrder(payload)

        order_id = self._extract_order_id(response)

        self._logger.info(
            "Angel One LIMIT order placed",
            extra={
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "direction": direction,
                "order_id": order_id,
            },
        )

        return order_id

    def place_sl_limit_order(
        self,
        symbol: str,
        qty: int,
        trigger: float,
        price: float,
        direction: str,
        algo_id: str = "",
    ) -> str:
        api = self._api()

        payload = {
            "variety": "STOPLOSS",
            "tradingsymbol": symbol,
            "symboltoken": "",
            "transactiontype": direction.upper(),
            "exchange": "NFO",
            "ordertype": "STOPLOSS_LIMIT",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": str(price),
            "triggerprice": str(trigger),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(qty),
            "tag": algo_id,
        }

        response = api.placeOrder(payload)

        order_id = self._extract_order_id(response)

        self._logger.info(
            "Angel One SL-LIMIT order placed",
            extra={
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "trigger": trigger,
                "direction": direction,
                "order_id": order_id,
            },
        )

        return order_id

    def modify_order(
        self,
        order_id: str,
        new_price: float,
        new_trigger: Optional[float] = None,
    ) -> bool:
        api = self._api()

        payload = {
            "orderid": order_id,
            "price": str(new_price),
        }

        if new_trigger is not None:
            payload["triggerprice"] = str(new_trigger)

        response = api.modifyOrder(payload)
        ok = self._response_success(response)

        self._logger.info(
            "Angel One order modified",
            extra={
                "order_id": order_id,
                "new_price": new_price,
                "new_trigger": new_trigger,
                "success": ok,
            },
        )

        return ok

    def cancel_order(self, order_id: str) -> bool:
        api = self._api()

        response = api.cancelOrder(order_id)
        ok = self._response_success(response)

        self._logger.info(
            "Angel One order cancelled",
            extra={"order_id": order_id, "success": ok},
        )

        return ok

    def get_positions(self) -> List[Dict[str, Any]]:
        api = self._api()
        response = api.position()

        if isinstance(response, dict) and response.get("status") and isinstance(response.get("data"), list):
            return response["data"]

        return []

    def get_orders(self) -> List[Dict[str, Any]]:
        api = self._api()
        response = api.orderBook()

        if isinstance(response, dict) and response.get("status") and isinstance(response.get("data"), list):
            return response["data"]

        return []

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        orders = self.get_orders()
        for o in orders:
            if str(o.get("orderid", "")) == str(order_id):
                return o
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_order_id(self, response: Any) -> str:
        """
        Normalize Angel One order placement response.
        """
        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            if "data" in response and isinstance(response["data"], dict):
                if "orderid" in response["data"]:
                    return str(response["data"]["orderid"])
            if "orderid" in response:
                return str(response["orderid"])

        raise RuntimeError(f"Unable to extract order_id from response: {response}")

    def _response_success(self, response: Any) -> bool:
        if isinstance(response, dict):
            return bool(response.get("status", False))
        # some endpoints may return string/other
        return response is not None


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Angel One Broker Adapter Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Mock auth / mock API
    # ------------------------------------------------------------------
    class MockAPI:
        def placeOrder(self, payload):
            return {"status": True, "data": {"orderid": "MOCK-ORDER-1"}}

        def modifyOrder(self, payload):
            return {"status": True}

        def cancelOrder(self, order_id):
            return {"status": True}

        def position(self):
            return {"status": True, "data": [{"symbol": "NIFTY", "qty": "65"}]}

        def orderBook(self):
            return {
                "status": True,
                "data": [
                    {"orderid": "MOCK-ORDER-1", "tradingsymbol": "NIFTY22450CE"},
                    {"orderid": "MOCK-ORDER-2", "tradingsymbol": "NIFTY22450PE"},
                ],
            }

    class MockAuth:
        def get_smart_api(self):
            return MockAPI()

    broker = AngelOneBroker(MockAuth())

    print(" [Test 1] LIMIT order placement...")
    try:
        oid1 = broker.place_limit_order("NIFTY22450CE", 65, 100.0, "BUY", "JA-001")
        if oid1 == "MOCK-ORDER-1":
            print(f" ✅ LIMIT order placed: {oid1}")
            passed += 1
        else:
            print(f" ❌ Unexpected order_id: {oid1}")
            failed += 1
    except Exception as e:
        print(f" ❌ LIMIT order failed: {e}")
        failed += 1

    print("\n [Test 2] SL-LIMIT order placement...")
    try:
        oid2 = broker.place_sl_limit_order("NIFTY22450PE", 65, 95.0, 94.0, "SELL", "JA-002")
        if oid2 == "MOCK-ORDER-1":
            print(f" ✅ SL-LIMIT order placed: {oid2}")
            passed += 1
        else:
            print(f" ❌ Unexpected order_id: {oid2}")
            failed += 1
    except Exception as e:
        print(f" ❌ SL-LIMIT order failed: {e}")
        failed += 1

    print("\n [Test 3] Modify order...")
    try:
        ok3 = broker.modify_order("MOCK-ORDER-1", 101.0, 99.0)
        if ok3:
            print(" ✅ Modify order works")
            passed += 1
        else:
            print(" ❌ Modify order failed")
            failed += 1
    except Exception as e:
        print(f" ❌ Modify order exception: {e}")
        failed += 1

    print("\n [Test 4] Cancel order...")
    try:
        ok4 = broker.cancel_order("MOCK-ORDER-1")
        if ok4:
            print(" ✅ Cancel order works")
            passed += 1
        else:
            print(" ❌ Cancel order failed")
            failed += 1
    except Exception as e:
        print(f" ❌ Cancel order exception: {e}")
        failed += 1

    print("\n [Test 5] Get positions...")
    try:
        pos = broker.get_positions()
        if isinstance(pos, list) and len(pos) == 1:
            print(f" ✅ Positions fetch works: {pos}")
            passed += 1
        else:
            print(f" ❌ Positions fetch failed: {pos}")
            failed += 1
    except Exception as e:
        print(f" ❌ Positions exception: {e}")
        failed += 1

    print("\n [Test 6] Get orders / single order...")
    try:
        orders = broker.get_orders()
        order = broker.get_order("MOCK-ORDER-2")
        if isinstance(orders, list) and len(orders) == 2 and order is not None:
            print(f" ✅ Orders fetch works, single order found: {order}")
            passed += 1
        else:
            print(f" ❌ Orders fetch/get_order failed: orders={orders}, order={order}")
            failed += 1
    except Exception as e:
        print(f" ❌ Orders exception: {e}")
        failed += 1

    print("\n [Test 7] Extract order_id helper failure path...")
    try:
        broker._extract_order_id({"bad": "format"})
        print(" ❌ Should have raised error")
        failed += 1
    except RuntimeError:
        print(" ✅ Invalid order response correctly rejected")
        passed += 1
    except Exception as e:
        print(f" ❌ Unexpected exception type: {e}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Angel One Broker Adapter working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()