"""
Junior Aladdin - Position Manager
=================================
PURPOSE:
Manage open positions using reusable exit rules.

This file exists because the roadmap explicitly expects:
    src/positions/position_manager.py

RESPONSIBILITIES:
- register new positions
- update positions with latest market state
- apply stop-loss / target / breakeven / trailing / time / confidence exits
- maintain current stop / target state
- mark positions closed
- expose clean position status to Captain / dashboard / journal

IMPORTANT:
This module does NOT send real broker orders directly.
It decides what should happen to positions.
Execution layer can consume these decisions later.

CONNECTS TO:
- Exit Rules
- Risk Engine
- Captain
- Journal
- Execution
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import math
from typing import Dict, Any, Optional, List

from src.utils.logger import setup_logger

try:
    from src.positions.exit_rules import ExitRules, ExitDecision
    _EXIT_RULES_IMPORT_ERROR = None
except Exception as exc:
    ExitRules = None  # type: ignore[assignment]
    ExitDecision = Any  # type: ignore[assignment]
    _EXIT_RULES_IMPORT_ERROR = exc

IST = timezone(timedelta(hours=5, minutes=30))
_logger = setup_logger("position_manager")


@dataclass
class ManagedPosition:
    position_id: str
    symbol: str
    direction: str
    qty: int
    entry_price: float
    current_sl: float
    target_price: float
    entry_time: str
    status: str = "OPEN"  # OPEN / CLOSED
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    partial_exit_done: bool = False
    exit_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PositionManager:
    """
    Manage positions and evaluate exits.
    """

    def __init__(self):
        self._logger = _logger
        self._rules = None
        if ExitRules is None:
            self._logger.critical(
                "ExitRules import failed",
                extra={"error": str(_EXIT_RULES_IMPORT_ERROR)},
            )
        else:
            try:
                self._rules = ExitRules()
            except Exception as exc:
                self._logger.critical(
                    "ExitRules initialization failed",
                    extra={"error": str(exc)},
                )
        self._positions: Dict[str, ManagedPosition] = {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------
    def add_position(
        self,
        position_id: str,
        symbol: str,
        direction: str,
        qty: int,
        entry_price: float,
        sl_price: float,
        target_price: float,
        entry_time: Optional[str] = None,
    ) -> bool:
        """
        Register a new open position.
        """
        direction_value = str(direction or "").upper()
        qty_value = self._safe_int(qty, default=0)
        entry_price_value = self._safe_float(entry_price, default=0.0)
        sl_price_value = self._safe_float(sl_price, default=0.0)
        target_price_value = self._safe_float(target_price, default=0.0)

        failure_reason = ""
        if (
            not position_id
            or not symbol
            or direction_value not in ("BUY", "SELL")
        ):
            failure_reason = "invalid_identity_fields"
        elif qty_value <= 0 or qty_value > 1_000_000:
            failure_reason = "invalid_qty"
        elif entry_price_value <= 0 or entry_price_value > 10_000_000:
            failure_reason = "invalid_entry_price"
        elif sl_price_value <= 0 or sl_price_value > 10_000_000:
            failure_reason = "invalid_sl_price"
        elif target_price_value <= 0 or target_price_value > 10_000_000:
            failure_reason = "invalid_target_price"

        if failure_reason:
            self._logger.warning(
                "Position add blocked due to invalid inputs",
                extra={
                    "reason": failure_reason,
                    "position_id": position_id,
                    "symbol": symbol,
                    "direction": direction,
                    "qty": qty,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "target_price": target_price,
                },
            )
            return False

        if (
            math.isnan(entry_price_value)
            or math.isinf(entry_price_value)
            or math.isnan(sl_price_value)
            or math.isinf(sl_price_value)
            or math.isnan(target_price_value)
            or math.isinf(target_price_value)
        ):
            self._logger.warning(
                "Position add blocked due to non-finite price input",
                extra={
                    "position_id": position_id,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "target_price": target_price,
                },
            )
            return False

        if entry_time is None:
            entry_time = datetime.now(IST).isoformat()

        self._positions[position_id] = ManagedPosition(
            position_id=position_id,
            symbol=symbol,
            direction=direction_value,
            qty=qty_value,
            entry_price=entry_price_value,
            current_sl=sl_price_value,
            target_price=target_price_value,
            entry_time=entry_time,
            current_price=entry_price_value,
        )

        self._logger.info(
            "Position added",
            extra={
                "position_id": position_id,
                "symbol": symbol,
                "direction": direction_value,
                "qty": qty_value,
                "entry_price": entry_price_value,
                "sl_price": sl_price_value,
                "target_price": target_price_value,
            },
        )
        return True

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> bool:
        """
        Mark a position closed.
        """
        pos = self._positions.get(position_id)
        if pos is None or pos.status != "OPEN":
            return False

        exit_price_value = self._safe_float(exit_price, default=0.0)
        if exit_price_value <= 0:
            self._logger.warning(
                "Position close blocked due to invalid exit price",
                extra={
                    "position_id": position_id,
                    "exit_price": exit_price,
                },
            )
            return False

        pos.current_price = exit_price_value
        pos.realized_pnl = round(self._compute_pnl(pos, exit_price_value), 2)
        pos.unrealized_pnl = 0.0
        pos.status = "CLOSED"
        pos.exit_reason = exit_reason

        self._logger.info(
            "Position closed",
            extra={
                "position_id": position_id,
                "symbol": pos.symbol,
                "exit_price": exit_price_value,
                "exit_reason": exit_reason,
                "realized_pnl": pos.realized_pnl,
            },
        )
        return True

    # ------------------------------------------------------------------
    # Position update / exit evaluation
    # ------------------------------------------------------------------
    def update_position(
        self,
        position_id: str,
        current_price: float,
        hold_minutes: float,
        atr: Optional[float] = None,
        ml_probability: Optional[float] = None,
        regime: str = "UNKNOWN",
        max_hold_minutes: float = 45.0,
    ) -> Dict[str, Any]:
        """
        Apply all exit rules to one position and update state.
        """
        pos = self._positions.get(position_id)
        if pos is None:
            return {"error": "position_not_found"}

        if pos.status != "OPEN":
            return {"error": "position_not_open", "position": pos.to_dict()}

        if self._rules is None:
            self._logger.critical(
                "ExitRules unavailable during update",
                extra={"position_id": position_id},
            )
            return {"error": "exit_rules_unavailable", "position": pos.to_dict()}

        current_price_value = self._safe_float(current_price, default=0.0)
        if current_price_value <= 0:
            self._logger.warning(
                "Invalid current_price in update_position",
                extra={
                    "position_id": position_id,
                    "current_price": current_price,
                },
            )
            return {"error": "invalid_current_price", "position": pos.to_dict()}

        hold_minutes_parsed = self._safe_float(hold_minutes, default=float("nan"))
        if math.isnan(hold_minutes_parsed) or math.isinf(hold_minutes_parsed):
            self._logger.warning(
                "Invalid hold_minutes in update_position, defaulting to 0.0",
                extra={
                    "position_id": position_id,
                    "hold_minutes": hold_minutes,
                },
            )
            hold_minutes_value = 0.0
        else:
            hold_minutes_value = hold_minutes_parsed

        atr_parsed = self._safe_float(atr, default=float("nan")) if atr is not None else 0.0
        if atr is not None and (math.isnan(atr_parsed) or math.isinf(atr_parsed) or atr_parsed < 0):
            self._logger.warning(
                "Invalid atr in update_position, defaulting to 0.0",
                extra={
                    "position_id": position_id,
                    "atr": atr,
                },
            )
            atr_value = 0.0
        else:
            atr_value = atr_parsed

        max_hold_parsed = self._safe_float(max_hold_minutes, default=float("nan"))
        if math.isnan(max_hold_parsed) or math.isinf(max_hold_parsed) or max_hold_parsed <= 0:
            self._logger.warning(
                "Invalid max_hold_minutes in update_position, defaulting to 45.0",
                extra={
                    "position_id": position_id,
                    "max_hold_minutes": max_hold_minutes,
                },
            )
            max_hold_minutes_value = 45.0
        else:
            max_hold_minutes_value = max_hold_parsed

        ml_probability_value: Optional[float] = None
        if ml_probability is not None:
            ml_prob_parsed = self._safe_float(ml_probability, default=float("nan"))
            if math.isnan(ml_prob_parsed) or math.isinf(ml_prob_parsed):
                self._logger.warning(
                    "Invalid ml_probability in update_position, defaulting to None",
                    extra={
                        "position_id": position_id,
                        "ml_probability": ml_probability,
                    },
                )
            else:
                ml_probability_value = ml_prob_parsed

        regime_value = str(regime or "UNKNOWN")

        pos.current_price = current_price_value
        pos.unrealized_pnl = round(self._compute_pnl(pos, current_price_value), 2)

        self._update_excursions(pos, current_price_value)

        risk_points = abs(pos.entry_price - pos.current_sl)

        # 1. Hard stop-loss
        d1 = self._rules.check_stop_loss(pos.direction, current_price_value, pos.current_sl)
        if d1.should_exit:
            self.close_position(position_id, current_price_value, d1.exit_reason)
            return {"action": "EXIT", "decision": d1.to_dict(), "position": pos.to_dict()}

        # 2. Target
        d2 = self._rules.check_target(pos.direction, current_price_value, pos.target_price)
        if d2.should_exit:
            self.close_position(position_id, current_price_value, d2.exit_reason)
            return {"action": "EXIT", "decision": d2.to_dict(), "position": pos.to_dict()}

        # 3. Confidence decay
        d3 = self._rules.check_confidence_decay(ml_probability_value)
        if d3.should_exit:
            self.close_position(position_id, current_price_value, d3.exit_reason)
            return {"action": "EXIT", "decision": d3.to_dict(), "position": pos.to_dict()}

        # 4. Time exit
        d4 = self._rules.check_time_exit(hold_minutes_value, max_hold_minutes_value)
        if d4.should_exit:
            self.close_position(position_id, current_price_value, d4.exit_reason)
            return {"action": "EXIT", "decision": d4.to_dict(), "position": pos.to_dict()}

        # 5. Breakeven move
        if atr_value > 0:
            d5 = self._rules.check_breakeven_move(
                direction=pos.direction,
                entry_price=pos.entry_price,
                current_price=current_price_value,
                current_sl=pos.current_sl,
                risk_points=risk_points,
                atr=atr_value,
            )
            if d5.new_sl is not None:
                new_sl_value = self._safe_float(d5.new_sl, default=pos.current_sl)
                if self._is_valid_sl_update(pos, new_sl_value):
                    pos.current_sl = new_sl_value
                self._logger.info(
                    "Position moved to breakeven",
                    extra={
                        "position_id": position_id,
                        "new_sl": pos.current_sl,
                    },
                )
                return {"action": "MODIFY_SL", "decision": d5.to_dict(), "position": pos.to_dict()}

        # 6. Partial exit
        d6 = self._rules.check_partial_exit(
            direction=pos.direction,
            entry_price=pos.entry_price,
            current_price=current_price_value,
            qty=pos.qty,
            risk_points=risk_points,
        )
        if d6.partial_exit_qty > 0 and not pos.partial_exit_done:
            pos.partial_exit_done = True
            self._logger.info(
                "Partial exit recommended",
                extra={
                    "position_id": position_id,
                    "partial_exit_qty": d6.partial_exit_qty,
                },
            )
            return {"action": "PARTIAL_EXIT", "decision": d6.to_dict(), "position": pos.to_dict()}

        # 7. Regime tightening
        if atr_value > 0:
            d7 = self._rules.check_regime_change_tightening(
                regime=regime_value,
                direction=pos.direction,
                current_price=current_price_value,
                atr=atr_value,
            )
            if d7.new_sl is not None:
                new_sl_value = self._safe_float(d7.new_sl, default=0.0)
                if self._is_valid_sl_update(pos, new_sl_value):
                    pos.current_sl = new_sl_value
                    self._logger.info(
                        "Regime-based SL tightening applied",
                        extra={
                            "position_id": position_id,
                            "new_sl": pos.current_sl,
                            "regime": regime_value,
                        },
                    )
                    return {"action": "MODIFY_SL", "decision": d7.to_dict(), "position": pos.to_dict()}

        # 8. Trailing stop
        if atr_value > 0:
            d8 = self._rules.compute_trailing_stop(
                direction=pos.direction,
                current_price=current_price_value,
                current_sl=pos.current_sl,
                atr=atr_value,
            )
            if d8.new_sl is not None:
                new_sl_value = self._safe_float(d8.new_sl, default=0.0)
                if self._is_valid_sl_update(pos, new_sl_value):
                    pos.current_sl = new_sl_value
                    self._logger.info(
                        "Trailing stop updated",
                        extra={
                            "position_id": position_id,
                            "new_sl": pos.current_sl,
                        },
                    )
                    return {"action": "MODIFY_SL", "decision": d8.to_dict(), "position": pos.to_dict()}

        return {"action": "HOLD", "position": pos.to_dict()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_pnl(self, pos: ManagedPosition, price: float) -> float:
        entry_price = self._safe_float(pos.entry_price, default=0.0)
        current_price = self._safe_float(price, default=0.0)
        qty = self._safe_int(pos.qty, default=0)

        if entry_price <= 0 or current_price <= 0 or qty <= 0:
            return 0.0

        if pos.direction == "BUY":
            pnl = (current_price - entry_price) * qty
        else:
            pnl = (entry_price - current_price) * qty

        if math.isnan(pnl) or math.isinf(pnl):
            return 0.0
        return pnl

    def _update_excursions(self, pos: ManagedPosition, current_price: float):
        entry_price = self._safe_float(pos.entry_price, default=0.0)
        price = self._safe_float(current_price, default=0.0)
        if entry_price <= 0 or price <= 0:
            return

        favorable = 0.0
        adverse = 0.0

        if pos.direction == "BUY":
            favorable = price - entry_price
            adverse = entry_price - price
        else:
            favorable = entry_price - price
            adverse = price - entry_price

        if math.isnan(favorable) or math.isinf(favorable) or math.isnan(adverse) or math.isinf(adverse):
            return

        max_favorable = self._safe_float(pos.max_favorable_excursion, default=0.0)
        max_adverse = self._safe_float(pos.max_adverse_excursion, default=0.0)

        pos.max_favorable_excursion = round(max(max_favorable, favorable), 2)
        pos.max_adverse_excursion = round(max(max_adverse, adverse), 2)

    def _is_valid_sl_update(self, pos: ManagedPosition, new_sl: float) -> bool:
        """
        Ensure SL only moves in protective direction.
        """
        current_sl = self._safe_float(pos.current_sl, default=0.0)
        new_sl_value = self._safe_float(new_sl, default=0.0)
        if new_sl_value <= 0:
            return False

        if pos.direction == "BUY":
            return new_sl_value > current_sl
        return new_sl_value < current_sl or current_sl <= 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        pos = self._positions.get(position_id)
        return pos.to_dict() if pos else None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._positions.values() if p.status == "OPEN"]

    def get_all_positions(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._positions.values()]

    def reset(self):
        self._positions.clear()
        self._logger.info("Position Manager reset")


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Position Manager Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    pm = PositionManager()

    print(" [Test 1] Add position...")
    ok1 = pm.add_position(
        position_id="POS-1",
        symbol="NIFTY22450CE",
        direction="BUY",
        qty=65,
        entry_price=100.0,
        sl_price=95.0,
        target_price=110.0,
    )
    if ok1 and len(pm.get_open_positions()) == 1:
        print(f" ✅ Position added: {pm.get_open_positions()[0]}")
        passed += 1
    else:
        print(" ❌ Position add failed")
        failed += 1

    print("\n [Test 2] Hold update...")
    r2 = pm.update_position(
        position_id="POS-1",
        current_price=102.0,
        hold_minutes=5,
        atr=4.0,
        ml_probability=0.8,
        regime="TRENDING",
    )
    if r2["action"] in ("HOLD", "MODIFY_SL", "PARTIAL_EXIT"):
        print(f" ✅ Position update works: {r2['action']}")
        passed += 1
    else:
        print(f" ❌ Update failed: {r2}")
        failed += 1

    print("\n [Test 3] Target exit...")
    pm2 = PositionManager()
    pm2.add_position("POS-2", "NIFTY22450CE", "BUY", 65, 100.0, 95.0, 110.0)
    r3 = pm2.update_position(
        position_id="POS-2",
        current_price=111.0,
        hold_minutes=10,
        atr=4.0,
        ml_probability=0.8,
        regime="TRENDING",
    )
    if r3["action"] == "EXIT" and r3["decision"]["exit_reason"] == "target_hit":
        print(f" ✅ Target exit works: {r3}")
        passed += 1
    else:
        print(f" ❌ Target exit failed: {r3}")
        failed += 1

    print("\n [Test 4] Stop-loss exit...")
    pm3 = PositionManager()
    pm3.add_position("POS-3", "NIFTY22450PE", "SELL", 65, 100.0, 105.0, 90.0)
    r4 = pm3.update_position(
        position_id="POS-3",
        current_price=106.0,
        hold_minutes=10,
        atr=4.0,
        ml_probability=0.8,
        regime="TRENDING",
    )
    if r4["action"] == "EXIT" and r4["decision"]["exit_reason"] == "stop_loss_hit":
        print(f" ✅ Stop exit works: {r4}")
        passed += 1
    else:
        print(f" ❌ Stop exit failed: {r4}")
        failed += 1

    print("\n [Test 5] Confidence decay exit...")
    pm4 = PositionManager()
    pm4.add_position("POS-4", "NIFTY22450CE", "BUY", 65, 100.0, 95.0, 110.0)
    r5 = pm4.update_position(
        position_id="POS-4",
        current_price=101.0,
        hold_minutes=10,
        atr=4.0,
        ml_probability=0.2,
        regime="TRENDING",
    )
    if r5["action"] == "EXIT" and r5["decision"]["exit_reason"] == "confidence_decay_exit":
        print(f" ✅ Confidence exit works: {r5}")
        passed += 1
    else:
        print(f" ❌ Confidence exit failed: {r5}")
        failed += 1

    print("\n [Test 6] Breakeven / trailing / partial path...")
    pm5 = PositionManager()
    pm5.add_position("POS-5", "NIFTY22450CE", "BUY", 10, 100.0, 95.0, 130.0)
    r6 = pm5.update_position(
        position_id="POS-5",
        current_price=111.0,
        hold_minutes=10,
        atr=5.0,
        ml_probability=0.8,
        regime="TRENDING",
    )
    if r6["action"] in ("MODIFY_SL", "PARTIAL_EXIT", "HOLD"):
        print(f" ✅ Advanced path works: {r6['action']}")
        passed += 1
    else:
        print(f" ❌ Advanced path failed: {r6}")
        failed += 1

    print("\n [Test 7] Close position manually...")
    pm6 = PositionManager()
    pm6.add_position("POS-6", "NIFTY22450CE", "BUY", 65, 100.0, 95.0, 120.0)
    ok7 = pm6.close_position("POS-6", 108.0, "manual_close")
    pos7 = pm6.get_position("POS-6")
    if ok7 and pos7 and pos7["status"] == "CLOSED":
        print(f" ✅ Manual close works: {pos7}")
        passed += 1
    else:
        print(f" ❌ Manual close failed: {pos7}")
        failed += 1

    print("\n [Test 8] Invalid position update safe...")
    r8 = pm.update_position(
        position_id="NO-SUCH",
        current_price=100.0,
        hold_minutes=1,
        atr=2.0,
    )
    if r8.get("error") == "position_not_found":
        print(f" ✅ Missing position safe: {r8}")
        passed += 1
    else:
        print(f" ❌ Missing position failed: {r8}")
        failed += 1

    print("\n [Test 9] Reset...")
    pm.reset()
    if len(pm.get_all_positions()) == 0:
        print(" ✅ Reset works")
        passed += 1
    else:
        print(" ❌ Reset failed")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Position Manager working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()