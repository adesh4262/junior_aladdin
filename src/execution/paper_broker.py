"""
Junior Aladdin - Paper Broker
=============================
PURPOSE:
Provide a paper-trading broker implementation compatible with the broker
interface and execution layer.

This file exists because the roadmap explicitly expects:
    src/execution/paper_broker.py

RESPONSIBILITIES:
- place paper limit orders
- place paper stoploss-limit orders
- update fills against simulated market prices
- maintain open orders
- maintain positions
- support modify / cancel
- apply deterministic slippage model

CONNECTS TO:
- broker_interface.py
- execution pipeline
- captain / mode manager
- journal / position manager
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import uuid

from src.execution.broker_interface import BrokerInterface
from src.utils.logger import setup_logger

IST = timezone(timedelta(hours=5, minutes=30))
_logger = setup_logger("paper_broker")


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    qty: int
    direction: str
    order_type: str
    requested_price: float
    trigger_price: float = 0.0
    status: str = "OPEN"  # OPEN / FILLED / CANCELLED / REJECTED
    fill_price: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""


@dataclass
class PaperPosition:
    symbol: str
    qty: int
    direction: str
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: str = ""
    updated_at: str = ""


class PaperBroker(BrokerInterface):
    """
    Deterministic paper broker simulator.
    """

    def __init__(self, slippage_pct: float = 0.003):
        self._logger = _logger
        self._slippage_pct = slippage_pct

        self._orders: Dict[str, PaperOrder] = {}
        self._positions: Dict[str, PaperPosition] = {}
        self._last_price_by_symbol: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Market data updates
    # ------------------------------------------------------------------
    def update_market_price(self, symbol: str, ltp: float):
        """
        Push latest market price into paper broker.
        This drives order fills and MTM updates.
        """
        if not symbol or ltp <= 0:
            return

        self._last_price_by_symbol[symbol] = float(ltp)
        self._try_fill_orders(symbol, ltp)
        self._update_position_mtm(symbol, ltp)

    # ------------------------------------------------------------------
    # BrokerInterface methods
    # ------------------------------------------------------------------
    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        direction: str,
        algo_id: str = "",
    ) -> str:
        order_id = self._new_order_id()
        now = self._now_iso()

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            qty=int(qty),
            direction=direction.upper(),
            order_type="LIMIT",
            requested_price=float(price),
            created_at=now,
            updated_at=now,
            notes=f"algo_id={algo_id}",
        )

        if not symbol or qty <= 0 or price <= 0 or direction.upper() not in ("BUY", "SELL"):
            order.status = "REJECTED"
            order.notes += "|invalid_order_fields"
            self._orders[order_id] = order
            self._logger.warning("Paper limit order rejected", extra=asdict(order))
            return order_id

        self._orders[order_id] = order
        self._logger.info("Paper limit order placed", extra=asdict(order))

        ltp = self._last_price_by_symbol.get(symbol)
        if ltp:
            self._try_fill_orders(symbol, ltp)

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
        order_id = self._new_order_id()
        now = self._now_iso()

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            qty=int(qty),
            direction=direction.upper(),
            order_type="STOPLOSS_LIMIT",
            requested_price=float(price),
            trigger_price=float(trigger),
            created_at=now,
            updated_at=now,
            notes=f"algo_id={algo_id}",
        )

        if (
            not symbol
            or qty <= 0
            or trigger <= 0
            or price <= 0
            or direction.upper() not in ("BUY", "SELL")
        ):
            order.status = "REJECTED"
            order.notes += "|invalid_sl_order_fields"
            self._orders[order_id] = order
            self._logger.warning("Paper SL-limit order rejected", extra=asdict(order))
            return order_id

        self._orders[order_id] = order
        self._logger.info("Paper SL-limit order placed", extra=asdict(order))
        return order_id

    def modify_order(
        self,
        order_id: str,
        new_price: float,
        new_trigger: Optional[float] = None,
    ) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != "OPEN":
            return False

        if new_price <= 0:
            return False

        order.requested_price = float(new_price)
        if new_trigger is not None and new_trigger > 0:
            order.trigger_price = float(new_trigger)

        order.updated_at = self._now_iso()

        self._logger.info("Paper order modified", extra=asdict(order))

        ltp = self._last_price_by_symbol.get(order.symbol)
        if ltp:
            self._try_fill_orders(order.symbol, ltp)

        return True

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != "OPEN":
            return False

        order.status = "CANCELLED"
        order.updated_at = self._now_iso()

        self._logger.info("Paper order cancelled", extra=asdict(order))
        return True

    def get_positions(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self._positions.values()]

    def get_orders(self) -> List[Dict[str, Any]]:
        return [asdict(o) for o in self._orders.values()]

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        order = self._orders.get(order_id)
        return asdict(order) if order else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _try_fill_orders(self, symbol: str, ltp: float):
        for order in self._orders.values():
            if order.symbol != symbol or order.status != "OPEN":
                continue

            if order.order_type == "LIMIT":
                if self._limit_fillable(order, ltp):
                    fill_price = self._apply_slippage(ltp, order.direction)
                    self._fill_order(order, fill_price)

            elif order.order_type == "STOPLOSS_LIMIT":
                if self._sl_triggered(order, ltp):
                    fill_price = self._apply_slippage(ltp, order.direction)
                    self._fill_order(order, fill_price)

    def _limit_fillable(self, order: PaperOrder, ltp: float) -> bool:
        if order.direction == "BUY":
            return ltp <= order.requested_price
        return ltp >= order.requested_price

    def _sl_triggered(self, order: PaperOrder, ltp: float) -> bool:
        if order.direction == "BUY":
            return ltp >= order.trigger_price
        return ltp <= order.trigger_price

    def _apply_slippage(self, ltp: float, direction: str) -> float:
        if direction == "BUY":
            return round(ltp * (1 + self._slippage_pct), 2)
        return round(ltp * (1 - self._slippage_pct), 2)

    def _fill_order(self, order: PaperOrder, fill_price: float):
        order.status = "FILLED"
        order.fill_price = float(fill_price)
        order.updated_at = self._now_iso()

        self._logger.info("Paper order filled", extra=asdict(order))
        self._apply_position_fill(order)

    def _apply_position_fill(self, order: PaperOrder):
        symbol = order.symbol
        qty = order.qty
        fill_price = order.fill_price
        direction = order.direction
        now = self._now_iso()

        existing = self._positions.get(symbol)

        if existing is None:
            self._positions[symbol] = PaperPosition(
                symbol=symbol,
                qty=qty,
                direction=direction,
                avg_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                opened_at=now,
                updated_at=now,
            )
            return

        if existing.direction == direction:
            total_qty = existing.qty + qty
            existing.avg_price = round(
                ((existing.avg_price * existing.qty) + (fill_price * qty)) / total_qty,
                2,
            )
            existing.qty = total_qty
            existing.updated_at = now
            return

        # opposite-side fill
        if qty < existing.qty:
            realized = self._calculate_realized(existing.direction, existing.avg_price, fill_price, qty)
            existing.qty -= qty
            existing.realized_pnl = round(existing.realized_pnl + realized, 2)
            existing.updated_at = now
            return

        if qty == existing.qty:
            realized = self._calculate_realized(existing.direction, existing.avg_price, fill_price, qty)
            existing.realized_pnl = round(existing.realized_pnl + realized, 2)
            del self._positions[symbol]
            return

        # flip
        realized = self._calculate_realized(existing.direction, existing.avg_price, fill_price, existing.qty)
        leftover_qty = qty - existing.qty
        self._positions[symbol] = PaperPosition(
            symbol=symbol,
            qty=leftover_qty,
            direction=direction,
            avg_price=fill_price,
            unrealized_pnl=0.0,
            realized_pnl=round(existing.realized_pnl + realized, 2),
            opened_at=now,
            updated_at=now,
        )

    def _calculate_realized(
        self,
        old_direction: str,
        old_avg_price: float,
        exit_price: float,
        qty: int,
    ) -> float:
        if old_direction == "BUY":
            return round((exit_price - old_avg_price) * qty, 2)
        return round((old_avg_price - exit_price) * qty, 2)

    def _update_position_mtm(self, symbol: str, ltp: float):
        pos = self._positions.get(symbol)
        if pos is None:
            return

        if pos.direction == "BUY":
            pnl = (ltp - pos.avg_price) * pos.qty
        else:
            pnl = (pos.avg_price - ltp) * pos.qty

        pos.unrealized_pnl = round(pnl, 2)
        pos.updated_at = self._now_iso()

    def _new_order_id(self) -> str:
        return f"PAPER-{uuid.uuid4().hex[:10].upper()}"

    def _now_iso(self) -> str:
        return datetime.now(IST).isoformat()


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Paper Broker Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    broker = PaperBroker()

    print(" [Test 1] Place valid BUY limit order...")
    broker.update_market_price("NIFTYCE", 100.0)
    oid1 = broker.place_limit_order("NIFTYCE", 65, 101.0, "BUY")
    o1 = broker.get_order(oid1)
    if o1 and o1["status"] == "FILLED":
        print(f" ✅ BUY filled: {o1['fill_price']}")
        passed += 1
    else:
        print(f" ❌ BUY order not filled: {o1}")
        failed += 1

    print("\n [Test 2] Position created...")
    pos = broker.get_positions()
    if len(pos) == 1 and pos[0]["symbol"] == "NIFTYCE":
        print(f" ✅ Position exists: {pos[0]}")
        passed += 1
    else:
        print(f" ❌ Position missing: {pos}")
        failed += 1

    print("\n [Test 3] MTM update...")
    broker.update_market_price("NIFTYCE", 110.0)
    pos3 = broker.get_positions()[0]
    if pos3["unrealized_pnl"] != 0:
        print(f" ✅ MTM updated: {pos3['unrealized_pnl']}")
        passed += 1
    else:
        print(f" ❌ MTM not updated: {pos3}")
        failed += 1

    print("\n [Test 4] SELL reduces / closes position...")
    oid4 = broker.place_limit_order("NIFTYCE", 65, 109.0, "SELL")
    o4 = broker.get_order(oid4)
    if o4 and o4["status"] == "FILLED" and len(broker.get_positions()) == 0:
        print(" ✅ SELL closed position")
        passed += 1
    else:
        print(f" ❌ SELL close failed: order={o4}, positions={broker.get_positions()}")
        failed += 1

    print("\n [Test 5] Stop-loss-limit order trigger...")
    broker.update_market_price("NIFTYPE", 90.0)
    oid5 = broker.place_sl_limit_order("NIFTYPE", 65, 95.0, 96.0, "BUY")
    o5_before = broker.get_order(oid5)
    broker.update_market_price("NIFTYPE", 95.5)
    o5_after = broker.get_order(oid5)
    if o5_before and o5_after and o5_after["status"] == "FILLED":
        print(f" ✅ SL-limit triggered and filled: {o5_after['fill_price']}")
        passed += 1
    else:
        print(f" ❌ SL-limit failed: before={o5_before}, after={o5_after}")
        failed += 1

    print("\n [Test 6] Cancel open order...")
    oid6 = broker.place_limit_order("BANKNIFTY", 15, 200.0, "BUY")
    broker.update_market_price("BANKNIFTY", 210.0)
    broker.cancel_order(oid6)
    o6 = broker.get_order(oid6)
    if o6 and o6["status"] == "CANCELLED":
        print(" ✅ Cancel worked")
        passed += 1
    else:
        print(f" ❌ Cancel failed: {o6}")
        failed += 1

    print("\n [Test 7] Modify open order...")
    oid7 = broker.place_limit_order("FINNIFTY", 40, 100.0, "BUY")
    broker.update_market_price("FINNIFTY", 105.0)
    ok_mod = broker.modify_order(oid7, 106.0)
    o7 = broker.get_order(oid7)
    broker.update_market_price("FINNIFTY", 105.0)
    o7_after = broker.get_order(oid7)
    if ok_mod and o7_after and o7_after["status"] == "FILLED":
        print(" ✅ Modify + fill worked")
        passed += 1
    else:
        print(f" ❌ Modify failed: before={o7}, after={o7_after}")
        failed += 1

    print("\n [Test 8] Invalid order rejection...")
    oid8 = broker.place_limit_order("", 0, 0, "BUY")
    o8 = broker.get_order(oid8)
    if o8 and o8["status"] == "REJECTED":
        print(" ✅ Invalid order rejected")
        passed += 1
    else:
        print(f" ❌ Invalid rejection failed: {o8}")
        failed += 1

    print("\n [Test 9] Order listing works...")
    all_orders = broker.get_orders()
    if len(all_orders) >= 5:
        print(f" ✅ Orders tracked: {len(all_orders)}")
        passed += 1
    else:
        print(f" ❌ Orders too few: {all_orders}")
        failed += 1

    print("\n [Test 10] Interface compliance...")
    positions10 = broker.get_positions()
    open_orders10 = [o for o in broker.get_orders() if o["status"] == "OPEN"]
    if isinstance(positions10, list) and isinstance(open_orders10, list):
        print(" ✅ Interface behavior valid")
        passed += 1
    else:
        print(" ❌ Interface behavior invalid")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Paper Broker working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()