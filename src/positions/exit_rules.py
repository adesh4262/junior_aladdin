"""
Junior Aladdin - Exit Rules
===========================
PURPOSE:
Provide reusable exit-rule logic for open positions.

This file exists because the roadmap explicitly expects:
    src/positions/exit_rules.py

RESPONSIBILITIES:
- stop-loss hit detection
- target hit detection
- breakeven shift logic
- partial-exit logic
- trailing-stop logic
- time-exit logic
- confidence-decay exit logic
- regime-change tightening logic

This file does NOT manage broker orders directly.
It only evaluates what should happen to a position.

CONNECTS TO:
- Position Manager
- Risk Engine
- Captain
- Feature / regime / ML confidence updates
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ExitDecision:
    should_exit: bool
    exit_reason: str = ""
    new_sl: Optional[float] = None
    partial_exit_qty: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_exit": self.should_exit,
            "exit_reason": self.exit_reason,
            "new_sl": self.new_sl,
            "partial_exit_qty": self.partial_exit_qty,
            "warnings": self.warnings,
        }


class ExitRules:
    """
    Reusable position exit-rule evaluator.
    """

    def __init__(self):
        pass

    def check_stop_loss(
        self,
        direction: str,
        current_price: float,
        sl_price: float,
    ) -> ExitDecision:
        if sl_price <= 0 or current_price <= 0:
            return ExitDecision(False)

        direction = str(direction).upper()

        if direction == "BUY" and current_price <= sl_price:
            return ExitDecision(True, "stop_loss_hit")
        if direction == "SELL" and current_price >= sl_price:
            return ExitDecision(True, "stop_loss_hit")

        return ExitDecision(False)

    def check_target(
        self,
        direction: str,
        current_price: float,
        target_price: float,
    ) -> ExitDecision:
        if target_price <= 0 or current_price <= 0:
            return ExitDecision(False)

        direction = str(direction).upper()

        if direction == "BUY" and current_price >= target_price:
            return ExitDecision(True, "target_hit")
        if direction == "SELL" and current_price <= target_price:
            return ExitDecision(True, "target_hit")

        return ExitDecision(False)

    def check_breakeven_move(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        risk_points: float,
        atr: float,
        breakeven_trigger_ratio: float = 1.0,
    ) -> ExitDecision:
        """
        Move SL to breakeven after 1R in favor.
        """
        if (
            entry_price <= 0
            or current_price <= 0
            or risk_points <= 0
            or atr <= 0
        ):
            return ExitDecision(False)

        direction = str(direction).upper()
        trigger_distance = risk_points * breakeven_trigger_ratio

        if direction == "BUY":
            if current_price - entry_price >= trigger_distance:
                new_sl = round(entry_price + 0.1 * atr, 2)
                if new_sl > current_sl:
                    return ExitDecision(False, "move_to_breakeven", new_sl=new_sl)

        elif direction == "SELL":
            if entry_price - current_price >= trigger_distance:
                new_sl = round(entry_price - 0.1 * atr, 2)
                if current_sl <= 0 or new_sl < current_sl:
                    return ExitDecision(False, "move_to_breakeven", new_sl=new_sl)

        return ExitDecision(False)

    def check_partial_exit(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        qty: int,
        risk_points: float,
        partial_exit_trigger_ratio: float = 1.5,
    ) -> ExitDecision:
        """
        Partial exit after 1.5R in favor.
        """
        if qty <= 1 or risk_points <= 0:
            return ExitDecision(False)

        direction = str(direction).upper()
        trigger_distance = risk_points * partial_exit_trigger_ratio
        partial_qty = max(1, qty // 2)

        if direction == "BUY":
            if current_price - entry_price >= trigger_distance:
                return ExitDecision(False, "partial_exit", partial_exit_qty=partial_qty)

        elif direction == "SELL":
            if entry_price - current_price >= trigger_distance:
                return ExitDecision(False, "partial_exit", partial_exit_qty=partial_qty)

        return ExitDecision(False)

    def compute_trailing_stop(
        self,
        direction: str,
        current_price: float,
        current_sl: float,
        atr: float,
        trailing_atr_multiplier: float = 1.0,
    ) -> ExitDecision:
        """
        ATR-based trailing stop. Never move stop backward.
        """
        if current_price <= 0 or atr <= 0:
            return ExitDecision(False)

        direction = str(direction).upper()
        distance = atr * trailing_atr_multiplier

        if direction == "BUY":
            proposed = round(current_price - distance, 2)
            if proposed > current_sl:
                return ExitDecision(False, "trail_stop", new_sl=proposed)

        elif direction == "SELL":
            proposed = round(current_price + distance, 2)
            if current_sl <= 0 or proposed < current_sl:
                return ExitDecision(False, "trail_stop", new_sl=proposed)

        return ExitDecision(False)

    def check_time_exit(
        self,
        hold_minutes: float,
        max_hold_minutes: float,
    ) -> ExitDecision:
        if max_hold_minutes <= 0:
            return ExitDecision(False)
        if hold_minutes >= max_hold_minutes:
            return ExitDecision(True, "time_exit")
        return ExitDecision(False)

    def check_confidence_decay(
        self,
        ml_probability: Optional[float],
        threshold: float = 0.35,
    ) -> ExitDecision:
        """
        Exit if refreshed ML confidence drops too low.
        """
        if ml_probability is None:
            return ExitDecision(False)
        if ml_probability < threshold:
            return ExitDecision(True, "confidence_decay_exit")
        return ExitDecision(False)

    def check_regime_change_tightening(
        self,
        regime: str,
        direction: str,
        current_price: float,
        atr: float,
    ) -> ExitDecision:
        """
        Tighten stop if regime becomes CHOP or EVENT while in trade.
        """
        if current_price <= 0 or atr <= 0:
            return ExitDecision(False)

        regime = str(regime).upper()
        direction = str(direction).upper()

        if regime in ("CHOP", "EVENT"):
            if direction == "BUY":
                new_sl = round(current_price - 0.5 * atr, 2)
                return ExitDecision(False, "regime_tighten", new_sl=new_sl)
            elif direction == "SELL":
                new_sl = round(current_price + 0.5 * atr, 2)
                return ExitDecision(False, "regime_tighten", new_sl=new_sl)

        return ExitDecision(False)


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Exit Rules Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    rules = ExitRules()

    print(" [Test 1] Stop-loss BUY...")
    r1 = rules.check_stop_loss("BUY", 99, 100)
    if r1.should_exit and r1.exit_reason == "stop_loss_hit":
        print(f" ✅ BUY SL hit: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ BUY SL failed: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Stop-loss SELL...")
    r2 = rules.check_stop_loss("SELL", 101, 100)
    if r2.should_exit and r2.exit_reason == "stop_loss_hit":
        print(f" ✅ SELL SL hit: {r2.to_dict()}")
        passed += 1
    else:
        print(f" ❌ SELL SL failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Target hit BUY...")
    r3 = rules.check_target("BUY", 110, 108)
    if r3.should_exit and r3.exit_reason == "target_hit":
        print(f" ✅ BUY target hit: {r3.to_dict()}")
        passed += 1
    else:
        print(f" ❌ BUY target failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Breakeven move...")
    r4 = rules.check_breakeven_move(
        direction="BUY",
        entry_price=100,
        current_price=111,
        current_sl=95,
        risk_points=10,
        atr=5,
    )
    if r4.exit_reason == "move_to_breakeven" and r4.new_sl is not None:
        print(f" ✅ Breakeven move works: {r4.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Breakeven failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Partial exit...")
    r5 = rules.check_partial_exit(
        direction="BUY",
        entry_price=100,
        current_price=116,
        qty=10,
        risk_points=10,
    )
    if r5.exit_reason == "partial_exit" and r5.partial_exit_qty > 0:
        print(f" ✅ Partial exit works: {r5.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Partial exit failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Trailing stop...")
    r6 = rules.compute_trailing_stop(
        direction="BUY",
        current_price=120,
        current_sl=105,
        atr=10,
    )
    if r6.exit_reason == "trail_stop" and r6.new_sl is not None:
        print(f" ✅ Trailing stop works: {r6.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Trailing stop failed: {r6.to_dict()}")
        failed += 1

    print("\n [Test 7] Time exit...")
    r7 = rules.check_time_exit(hold_minutes=46, max_hold_minutes=45)
    if r7.should_exit and r7.exit_reason == "time_exit":
        print(f" ✅ Time exit works: {r7.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Time exit failed: {r7.to_dict()}")
        failed += 1

    print("\n [Test 8] Confidence decay...")
    r8 = rules.check_confidence_decay(ml_probability=0.30)
    if r8.should_exit and r8.exit_reason == "confidence_decay_exit":
        print(f" ✅ Confidence decay works: {r8.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Confidence decay failed: {r8.to_dict()}")
        failed += 1

    print("\n [Test 9] Regime tightening...")
    r9 = rules.check_regime_change_tightening(
        regime="CHOP",
        direction="BUY",
        current_price=120,
        atr=10,
    )
    if r9.exit_reason == "regime_tighten" and r9.new_sl is not None:
        print(f" ✅ Regime tightening works: {r9.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Regime tightening failed: {r9.to_dict()}")
        failed += 1

    print("\n [Test 10] Invalid inputs safe...")
    r10 = rules.check_partial_exit(
        direction="BUY",
        entry_price=100,
        current_price=105,
        qty=1,
        risk_points=10,
    )
    if not r10.should_exit and r10.partial_exit_qty == 0:
        print(f" ✅ Invalid edge case handled safely: {r10.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Edge case failed: {r10.to_dict()}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Exit Rules working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()