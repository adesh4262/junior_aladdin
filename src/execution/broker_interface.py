"""
Junior Aladdin - Broker Interface
=================================
PURPOSE:
Define the standard abstract broker contract used by the system.

This file exists because the roadmap explicitly expects:
    src/execution/broker_interface.py

GOAL:
Any broker implementation (paper broker, Angel One live broker, future brokers)
must implement the same methods so the rest of the system can stay broker-agnostic.

CONNECTS TO:
- Paper Broker
- Angel One Broker
- Execution / Captain / Position Manager
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BrokerInterface(ABC):
    """
    Abstract broker interface for all execution backends.
    """

    @abstractmethod
    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        direction: str,
        algo_id: str = "",
    ) -> str:
        """
        Place a limit order and return order_id.
        """
        raise NotImplementedError

    @abstractmethod
    def place_sl_limit_order(
        self,
        symbol: str,
        qty: int,
        trigger: float,
        price: float,
        direction: str,
        algo_id: str = "",
    ) -> str:
        """
        Place a stop-loss-limit order and return order_id.
        """
        raise NotImplementedError

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        new_price: float,
        new_trigger: Optional[float] = None,
    ) -> bool:
        """
        Modify an open order.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        """
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return current open positions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Return all tracked orders.
        """
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Return one order by ID.
        """
        raise NotImplementedError


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Broker Interface Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # Basic import / abstract type existence test
    print(" [Test 1] Class exists...")
    try:
        cls = BrokerInterface
        print(f" ✅ BrokerInterface exists: {cls.__name__}")
        passed += 1
    except Exception as e:
        print(f" ❌ BrokerInterface missing: {e}")
        failed += 1

    # Abstract instantiation should fail
    print("\n [Test 2] Abstract class cannot be instantiated...")
    try:
        BrokerInterface()  # type: ignore
        print(" ❌ Abstract class should not instantiate")
        failed += 1
    except TypeError:
        print(" ✅ Abstract class instantiation blocked")
        passed += 1
    except Exception as e:
        print(f" ❌ Unexpected exception: {e}")
        failed += 1

    # Minimal implementation test
    print("\n [Test 3] Minimal implementation works...")
    try:
        class DummyBroker(BrokerInterface):
            def place_limit_order(self, symbol, qty, price, direction, algo_id=""):
                return "OID-1"

            def place_sl_limit_order(self, symbol, qty, trigger, price, direction, algo_id=""):
                return "OID-2"

            def modify_order(self, order_id, new_price, new_trigger=None):
                return True

            def cancel_order(self, order_id):
                return True

            def get_positions(self):
                return []

            def get_orders(self):
                return []

            def get_order(self, order_id):
                return {"order_id": order_id}

        b = DummyBroker()
        if (
            b.place_limit_order("ABC", 1, 100, "BUY") == "OID-1"
            and b.place_sl_limit_order("ABC", 1, 95, 94, "SELL") == "OID-2"
            and b.modify_order("OID-1", 101)
            and b.cancel_order("OID-2")
            and b.get_positions() == []
            and b.get_orders() == []
            and b.get_order("OID-1") == {"order_id": "OID-1"}
        ):
            print(" ✅ Minimal implementation satisfies interface")
            passed += 1
        else:
            print(" ❌ Dummy broker behavior mismatch")
            failed += 1
    except Exception as e:
        print(f" ❌ Minimal implementation failed: {e}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Broker Interface working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()
    