"""
Junior Aladdin - Cost Calculator
================================
PURPOSE:
Compute transaction costs for trades so all P&L can be evaluated on a
cost-adjusted basis.

This file exists because the roadmap explicitly expects:
    src/risk/cost_calculator.py

COST COMPONENTS:
- Brokerage
- GST on brokerage
- STT on sell side
- Stamp duty
- Exchange charges

GOAL:
Provide transparent and reusable trade-cost logic for:
- journal
- risk engine
- performance analysis
- paper/live evaluation

CONNECTS TO:
- Risk Engine
- Journal
- Daily Summary
- Position Manager / Execution layer
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("cost_calculator")


@dataclass
class TradeCostBreakdown:
    brokerage_entry: float
    brokerage_exit: float
    brokerage_total: float
    gst_on_brokerage: float
    stt: float
    stamp_duty: float
    exchange_charges: float
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostCalculator:
    """
    Compute full trade cost breakdown and net P&L after costs.
    """

    def __init__(self):
        self._logger = _logger

        self._stt_sell_pct = Config.get("costs", "stt_sell_pct", default=0.000125)
        self._brokerage_per_order = Config.get("costs", "brokerage_per_order", default=20.0)
        self._gst_on_brokerage_pct = Config.get("costs", "gst_on_brokerage_pct", default=0.18)
        self._stamp_duty_pct = Config.get("costs", "stamp_duty_pct", default=0.00003)
        self._exchange_charges_pct = Config.get("costs", "exchange_charges_pct", default=0.0005)

    def calculate_trade_costs(
        self,
        entry_price: float,
        exit_price: float,
        qty: int,
    ) -> TradeCostBreakdown:
        """
        Calculate total transaction cost for one completed trade.
        """
        if entry_price <= 0 or exit_price <= 0 or qty <= 0:
            return TradeCostBreakdown(
                brokerage_entry=0.0,
                brokerage_exit=0.0,
                brokerage_total=0.0,
                gst_on_brokerage=0.0,
                stt=0.0,
                stamp_duty=0.0,
                exchange_charges=0.0,
                total_cost=0.0,
            )

        turnover_entry = entry_price * qty
        turnover_exit = exit_price * qty

        brokerage_entry = float(self._brokerage_per_order)
        brokerage_exit = float(self._brokerage_per_order)
        brokerage_total = brokerage_entry + brokerage_exit

        gst_on_brokerage = brokerage_total * self._gst_on_brokerage_pct

        # STT only on sell side
        sell_turnover = turnover_exit
        stt = sell_turnover * self._stt_sell_pct

        # stamp duty generally applied on buy side turnover
        stamp_duty = turnover_entry * self._stamp_duty_pct

        exchange_charges = (turnover_entry + turnover_exit) * self._exchange_charges_pct

        total_cost = (
            brokerage_total
            + gst_on_brokerage
            + stt
            + stamp_duty
            + exchange_charges
        )

        breakdown = TradeCostBreakdown(
            brokerage_entry=round(brokerage_entry, 2),
            brokerage_exit=round(brokerage_exit, 2),
            brokerage_total=round(brokerage_total, 2),
            gst_on_brokerage=round(gst_on_brokerage, 2),
            stt=round(stt, 2),
            stamp_duty=round(stamp_duty, 2),
            exchange_charges=round(exchange_charges, 2),
            total_cost=round(total_cost, 2),
        )

        self._logger.info(
            "Trade costs calculated",
            extra={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "total_cost": breakdown.total_cost,
            },
        )

        return breakdown

    def calculate_net_pnl(
        self,
        entry_price: float,
        exit_price: float,
        qty: int,
        direction: str,
    ) -> Dict[str, Any]:
        """
        Calculate gross and net P&L for a trade.
        """
        direction = str(direction).upper()

        if direction == "BUY":
            gross_pnl = (exit_price - entry_price) * qty
        elif direction == "SELL":
            gross_pnl = (entry_price - exit_price) * qty
        else:
            gross_pnl = 0.0

        costs = self.calculate_trade_costs(entry_price, exit_price, qty)
        net_pnl = gross_pnl - costs.total_cost

        result = {
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "cost_breakdown": costs.to_dict(),
        }

        self._logger.info(
            "Net PnL calculated",
            extra={
                "direction": direction,
                "gross_pnl": result["gross_pnl"],
                "net_pnl": result["net_pnl"],
                "total_cost": costs.total_cost,
            },
        )

        return result


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Cost Calculator Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    calc = CostCalculator()

    print(" [Test 1] Valid cost calculation...")
    c1 = calc.calculate_trade_costs(entry_price=100.0, exit_price=110.0, qty=65)
    if c1.total_cost > 0:
        print(f" ✅ Cost breakdown: {c1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Invalid cost result: {c1.to_dict()}")
        failed += 1

    print("\n [Test 2] BUY net PnL...")
    r2 = calc.calculate_net_pnl(
        entry_price=100.0,
        exit_price=110.0,
        qty=65,
        direction="BUY",
    )
    if r2["gross_pnl"] == 650.0 and r2["net_pnl"] < r2["gross_pnl"]:
        print(f" ✅ BUY PnL works: {r2}")
        passed += 1
    else:
        print(f" ❌ BUY PnL wrong: {r2}")
        failed += 1

    print("\n [Test 3] SELL net PnL...")
    r3 = calc.calculate_net_pnl(
        entry_price=110.0,
        exit_price=100.0,
        qty=65,
        direction="SELL",
    )
    if r3["gross_pnl"] == 650.0 and r3["net_pnl"] < r3["gross_pnl"]:
        print(f" ✅ SELL PnL works: {r3}")
        passed += 1
    else:
        print(f" ❌ SELL PnL wrong: {r3}")
        failed += 1

    print("\n [Test 4] Invalid input handling...")
    c4 = calc.calculate_trade_costs(0, 0, 0)
    if c4.total_cost == 0.0:
        print(f" ✅ Invalid input safely handled: {c4.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Invalid input handling failed: {c4.to_dict()}")
        failed += 1

    print("\n [Test 5] Direction fallback...")
    r5 = calc.calculate_net_pnl(
        entry_price=100.0,
        exit_price=110.0,
        qty=65,
        direction="UNKNOWN",
    )
    if r5["gross_pnl"] == 0.0:
        print(f" ✅ Unknown direction safely handled: {r5}")
        passed += 1
    else:
        print(f" ❌ Unknown direction handling failed: {r5}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Cost Calculator working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()