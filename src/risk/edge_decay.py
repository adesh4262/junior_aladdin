"""
Junior Aladdin - Edge Decay Monitor
===================================
PURPOSE:
Track strategy edge deterioration over rolling trade windows.

This file exists because the roadmap explicitly expects:
    src/risk/edge_decay.py

RULES FROM PLAN:
- Track each strategy's win rate over rolling 20-trade windows
- If win rate drops below 40% for 20+ trades -> raise threshold by 5 points
- If win rate drops below 30% -> strategy PAUSED until self-learning review

This module does NOT itself rewrite strategy config live.
It produces recommendations / state flags for Captain / learning layer.

CONNECTS TO:
- Journal / trade history
- Strategy DNA
- Weekly self-learning review
- Captain
"""

from dataclasses import dataclass, field
from collections import deque
import math
from typing import Dict, List, Any, Optional

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("edge_decay")


@dataclass
class StrategyEdgeStatus:
    strategy_name: str
    trades_seen: int
    rolling_window_size: int
    rolling_win_rate: float
    rolling_profit_factor: float
    recommendation: str  # KEEP / RAISE_THRESHOLD / PAUSE
    threshold_adjustment: int = 0
    paused: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "trades_seen": self.trades_seen,
            "rolling_window_size": self.rolling_window_size,
            "rolling_win_rate": self.rolling_win_rate,
            "rolling_profit_factor": self.rolling_profit_factor,
            "recommendation": self.recommendation,
            "threshold_adjustment": self.threshold_adjustment,
            "paused": self.paused,
            "notes": self.notes,
        }


class EdgeDecayMonitor:
    """
    Monitors rolling performance for each strategy.
    """

    def __init__(self, window_size: int = 20):
        self._logger = _logger
        self._window_size = self._safe_int(window_size, default=20)
        if self._window_size <= 0:
            self._window_size = 20

        raise_threshold_raw = Config.get("edge_decay", "raise_threshold_below", default=0.40)
        pause_below_raw = Config.get("edge_decay", "pause_below", default=0.30)
        raise_by_raw = Config.get("edge_decay", "raise_by", default=5)

        self._raise_threshold_below = self._safe_float(raise_threshold_raw, default=0.40)
        self._pause_below = self._safe_float(pause_below_raw, default=0.30)
        self._raise_by = self._safe_int(raise_by_raw, default=5)
        if self._raise_by < 0:
            self._raise_by = 5

        self._history: Dict[str, deque] = {}

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
    # Data ingestion
    # ------------------------------------------------------------------
    def record_trade(
        self,
        strategy_name: str,
        pnl_rupees: float,
    ):
        """
        Record one completed trade outcome.
        """
        strategy = str(strategy_name).strip() if strategy_name is not None else ""
        if not strategy:
            self._logger.warning(
                "Skipped trade record due to invalid strategy name",
                extra={"strategy_name": strategy_name},
            )
            return

        pnl_probe = self._safe_float(pnl_rupees, default=float("nan"))
        if math.isnan(pnl_probe):
            self._logger.warning(
                "Invalid pnl_rupees received; defaulting to 0.0",
                extra={"strategy": strategy, "pnl_rupees": pnl_rupees},
            )
            pnl_value = 0.0
        else:
            pnl_value = pnl_probe

        if strategy not in self._history:
            self._history[strategy] = deque(maxlen=self._window_size)

        self._history[strategy].append(pnl_value)

        self._logger.info(
            "Trade recorded for edge monitor",
            extra={
                "strategy": strategy,
                "pnl_rupees": pnl_value,
                "window_count": len(self._history[strategy]),
            },
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_strategy(self, strategy_name: str) -> StrategyEdgeStatus:
        """
        Evaluate one strategy using its rolling trade history.
        """
        strategy = str(strategy_name).strip() if strategy_name is not None else ""
        history_raw = list(self._history.get(strategy, []))
        history: List[float] = []
        corrected_values = 0
        for value in history_raw:
            safe_value = self._safe_float(value, default=float("nan"))
            if math.isnan(safe_value):
                corrected_values += 1
                history.append(0.0)
            else:
                history.append(safe_value)

        if corrected_values > 0:
            self._logger.warning(
                "Non-finite values found in strategy history and normalized to 0.0",
                extra={
                    "strategy": strategy,
                    "corrected_values": corrected_values,
                },
            )

        trades_seen = len(history)

        if trades_seen == 0:
            return StrategyEdgeStatus(
                strategy_name=strategy,
                trades_seen=0,
                rolling_window_size=self._window_size,
                rolling_win_rate=0.0,
                rolling_profit_factor=0.0,
                recommendation="KEEP",
                threshold_adjustment=0,
                paused=False,
                notes=["no_trade_history"],
            )

        wins = [x for x in history if x > 0]
        losses = [x for x in history if x < 0]

        rolling_win_rate = len(wins) / trades_seen if trades_seen > 0 else 0.0

        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        elif gross_win > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0

        recommendation = "KEEP"
        adjustment = 0
        paused = False
        notes: List[str] = []

        # only enforce if enough data
        if trades_seen >= self._window_size:
            if rolling_win_rate < self._pause_below:
                recommendation = "PAUSE"
                adjustment = 0
                paused = True
                notes.append("rolling_win_rate_below_pause_threshold")
            elif rolling_win_rate < self._raise_threshold_below:
                recommendation = "RAISE_THRESHOLD"
                adjustment = self._raise_by
                paused = False
                notes.append("rolling_win_rate_below_raise_threshold")
            else:
                notes.append("edge_stable")
        else:
            notes.append("insufficient_window_history")

        result = StrategyEdgeStatus(
            strategy_name=strategy,
            trades_seen=trades_seen,
            rolling_window_size=self._window_size,
            rolling_win_rate=round(rolling_win_rate, 4),
            rolling_profit_factor=round(profit_factor, 4) if profit_factor != 999.0 else 999.0,
            recommendation=recommendation,
            threshold_adjustment=adjustment,
            paused=paused,
            notes=notes,
        )

        self._logger.info(
            "Strategy edge evaluated",
            extra=result.to_dict(),
        )

        return result

    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all tracked strategies.
        """
        return {
            strategy: self.evaluate_strategy(strategy).to_dict()
            for strategy in self._history.keys()
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def reset(self):
        self._history.clear()
        self._logger.info("Edge Decay Monitor reset")

    def get_status(self) -> Dict[str, Any]:
        return {
            "strategies_tracked": len(self._history),
            "window_size": self._window_size,
            "tracked_names": sorted(self._history.keys()),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Edge Decay Monitor Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    edm = EdgeDecayMonitor(window_size=20)

    print(" [Test 1] Empty strategy evaluation...")
    r1 = edm.evaluate_strategy("VWAP_PULLBACK")
    if r1.recommendation == "KEEP" and "no_trade_history" in r1.notes:
        print(f" ✅ Empty history handled: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Empty history failed: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Strong strategy stays KEEP...")
    edm.reset()
    for _ in range(14):
        edm.record_trade("VWAP_PULLBACK", 500)
    for _ in range(6):
        edm.record_trade("VWAP_PULLBACK", -200)
    r2 = edm.evaluate_strategy("VWAP_PULLBACK")
    if r2.recommendation == "KEEP" and r2.rolling_win_rate >= 0.40:
        print(f" ✅ Strong strategy kept: {r2.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Strong strategy evaluation failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Weak strategy raises threshold...")
    edm.reset()
    for _ in range(7):
        edm.record_trade("SR_REJECTION", 400)
    for _ in range(13):
        edm.record_trade("SR_REJECTION", -300)
    r3 = edm.evaluate_strategy("SR_REJECTION")
    if r3.recommendation == "RAISE_THRESHOLD" and r3.threshold_adjustment == 5:
        print(f" ✅ Raise-threshold trigger works: {r3.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Raise-threshold failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Very weak strategy pauses...")
    edm.reset()
    for _ in range(5):
        edm.record_trade("ORB", 300)
    for _ in range(15):
        edm.record_trade("ORB", -350)
    r4 = edm.evaluate_strategy("ORB")
    if r4.recommendation == "PAUSE" and r4.paused:
        print(f" ✅ Pause trigger works: {r4.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Pause trigger failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Insufficient history does not overreact...")
    edm.reset()
    for _ in range(3):
        edm.record_trade("FVG_RETEST", -200)
    r5 = edm.evaluate_strategy("FVG_RETEST")
    if r5.recommendation == "KEEP" and "insufficient_window_history" in r5.notes:
        print(f" ✅ Insufficient-history handling works: {r5.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Insufficient-history handling failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Evaluate all...")
    edm.record_trade("VWAP_PULLBACK", 400)
    edm.record_trade("STOP_HUNT_RECLAIM", -150)
    r6 = edm.evaluate_all()
    if "FVG_RETEST" in r6 and "VWAP_PULLBACK" in r6 and "STOP_HUNT_RECLAIM" in r6:
        print(f" ✅ evaluate_all works: keys={list(r6.keys())}")
        passed += 1
    else:
        print(f" ❌ evaluate_all failed: {r6}")
        failed += 1

    print("\n [Test 7] Status and reset...")
    st = edm.get_status()
    if st["strategies_tracked"] >= 1:
        print(f" ✅ Status works: {st}")
        passed += 1
    else:
        print(f" ❌ Status failed: {st}")
        failed += 1

    edm.reset()
    st2 = edm.get_status()
    if st2["strategies_tracked"] == 0:
        print(f" ✅ Reset works: {st2}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st2}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Edge Decay Monitor working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()