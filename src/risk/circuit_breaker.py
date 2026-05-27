"""
Junior Aladdin - Circuit Breaker
================================
PURPOSE:
Emergency account-protection layer that decides when the system must stop
trading and move to LOCKED / SAFE mode.

This file exists because the roadmap explicitly expects:
    src/risk/circuit_breaker.py

TRIGGERS (from plan):
1. Daily loss >= 2%
2. 3 consecutive losses + high tilt
3. Data gap / feed outage > 5 sec
4. Repeated broker/execution API failures

ACTIONS (downstream, outside this file):
- cancel pending orders
- close positions safely
- disable new trades
- require manual re-authorization

CONNECTS TO:
- Risk Engine
- Behavioral Sentinel
- Feed Health
- Captain
- Execution layer
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("circuit_breaker")


@dataclass
class CircuitBreakerDecision:
    """
    Output of breaker evaluation.
    """
    triggered: bool
    reason: str = ""
    severity: str = "NORMAL"  # NORMAL / CAUTION / SAFE / LOCKED
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "severity": self.severity,
            "warnings": self.warnings,
        }


class CircuitBreaker:
    """
    Central emergency-stop evaluator.
    """

    def __init__(self):
        self._logger = _logger
        max_daily_loss_pct_raw = Config.get("risk", "max_daily_loss_pct", default=0.02)
        max_consecutive_losses_raw = Config.get("risk", "max_consecutive_losses", default=3)
        tilt_reduce_threshold_raw = Config.get("behavioral", "tilt_reduce_threshold", default=70)

        self._max_daily_loss_pct = self._safe_float(max_daily_loss_pct_raw, default=0.02)
        if self._max_daily_loss_pct <= 0:
            self._max_daily_loss_pct = 0.02

        self._max_consecutive_losses = self._safe_int(max_consecutive_losses_raw, default=3)
        if self._max_consecutive_losses <= 0:
            self._max_consecutive_losses = 3

        self._tilt_reduce_threshold = self._safe_int(tilt_reduce_threshold_raw, default=70)

        self._data_gap_seconds = 5.0
        self._max_broker_errors = 3

        self._recent_broker_errors: int = 0

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

    def _trigger(
        self,
        reason: str,
        severity: str,
        warnings: List[str],
    ) -> CircuitBreakerDecision:
        self._logger.critical(
            "Circuit breaker triggered",
            extra={
                "reason": reason,
                "severity": severity,
                "warnings": list(warnings),
            },
        )
        return CircuitBreakerDecision(
            triggered=True,
            reason=reason,
            severity=severity,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Broker/API error tracking
    # ------------------------------------------------------------------
    def record_broker_error(self):
        self._recent_broker_errors += 1
        self._logger.warning(
            "Broker error recorded",
            extra={"recent_broker_errors": self._recent_broker_errors},
        )

    def reset_broker_errors(self):
        self._recent_broker_errors = 0
        self._logger.info("Broker error counter reset")

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        capital: float,
        daily_pnl: float,
        consecutive_losses: int,
        tilt_score: float,
        feed_lag_ms: float = 0.0,
        feed_health: str = "HEALTHY",
    ) -> CircuitBreakerDecision:
        """
        Evaluate whether breaker should trigger.
        """
        warnings: List[str] = []

        capital_value = self._safe_float(capital, default=0.0)
        daily_pnl_value = self._safe_float(daily_pnl, default=0.0)
        consecutive_losses_value = self._safe_int(consecutive_losses, default=0)
        tilt_score_value = self._safe_float(tilt_score, default=0.0)
        feed_lag_ms_value = self._safe_float(feed_lag_ms, default=0.0)
        feed_health_value = feed_health if isinstance(feed_health, str) else "HEALTHY"

        if capital_value <= 0:
            return self._trigger("invalid_capital", "LOCKED", warnings)

        daily_loss_pct = abs(min(daily_pnl_value, 0.0)) / capital_value

        # 1. Daily loss hard stop
        if daily_loss_pct >= self._max_daily_loss_pct:
            return self._trigger("daily_loss_limit_hit", "LOCKED", warnings)

        # 2. 3 consecutive losses + high tilt
        if consecutive_losses_value >= self._max_consecutive_losses and tilt_score_value > self._tilt_reduce_threshold:
            return self._trigger("consecutive_losses_plus_tilt", "LOCKED", warnings)

        # 3. Data gap / feed outage
        lag_seconds = feed_lag_ms_value / 1000.0
        if feed_health_value == "DOWN" or lag_seconds > self._data_gap_seconds:
            return self._trigger("data_gap_or_feed_down", "SAFE", warnings)

        # 4. Broker/API failures
        if self._recent_broker_errors >= self._max_broker_errors:
            return self._trigger("broker_api_failure_limit", "LOCKED", warnings)

        # Soft warnings
        if daily_loss_pct >= self._max_daily_loss_pct * 0.75:
            warnings.append("approaching_daily_loss_limit")

        if consecutive_losses_value >= 2:
            warnings.append("multiple_consecutive_losses")

        if feed_health_value == "STALE":
            warnings.append("feed_stale")

        return CircuitBreakerDecision(
            triggered=False,
            reason="",
            severity="NORMAL" if not warnings else "CAUTION",
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Status / reset
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "recent_broker_errors": self._recent_broker_errors,
            "max_broker_errors": self._max_broker_errors,
            "max_daily_loss_pct": self._max_daily_loss_pct,
        }

    def reset_daily(self):
        self.reset_broker_errors()
        self._logger.info("Circuit Breaker reset")


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Circuit Breaker Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    cb = CircuitBreaker()

    print(" [Test 1] Normal safe state...")
    r1 = cb.evaluate(
        capital=50000,
        daily_pnl=200,
        consecutive_losses=0,
        tilt_score=10,
        feed_lag_ms=50,
        feed_health="HEALTHY",
    )
    if not r1.triggered and r1.severity in ("NORMAL", "CAUTION"):
        print(f" ✅ Normal state works: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Normal state failed: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Daily loss hard stop...")
    r2 = cb.evaluate(
        capital=50000,
        daily_pnl=-1200,  # > 2%
        consecutive_losses=1,
        tilt_score=20,
        feed_lag_ms=50,
        feed_health="HEALTHY",
    )
    if r2.triggered and r2.reason == "daily_loss_limit_hit":
        print(f" ✅ Daily loss trigger works: {r2.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Daily loss trigger failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Consecutive losses + tilt trigger...")
    r3 = cb.evaluate(
        capital=50000,
        daily_pnl=-400,
        consecutive_losses=3,
        tilt_score=80,
        feed_lag_ms=50,
        feed_health="HEALTHY",
    )
    if r3.triggered and r3.reason == "consecutive_losses_plus_tilt":
        print(f" ✅ Tilt-loss trigger works: {r3.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Tilt-loss trigger failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Feed outage trigger...")
    r4 = cb.evaluate(
        capital=50000,
        daily_pnl=0,
        consecutive_losses=0,
        tilt_score=0,
        feed_lag_ms=7000,
        feed_health="DOWN",
    )
    if r4.triggered and r4.reason == "data_gap_or_feed_down":
        print(f" ✅ Feed outage trigger works: {r4.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Feed outage trigger failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Broker error trigger...")
    cb.reset_daily()
    cb.record_broker_error()
    cb.record_broker_error()
    cb.record_broker_error()
    r5 = cb.evaluate(
        capital=50000,
        daily_pnl=0,
        consecutive_losses=0,
        tilt_score=0,
        feed_lag_ms=50,
        feed_health="HEALTHY",
    )
    if r5.triggered and r5.reason == "broker_api_failure_limit":
        print(f" ✅ Broker error trigger works: {r5.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Broker trigger failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Warning state before hard stop...")
    cb.reset_daily()
    r6 = cb.evaluate(
        capital=50000,
        daily_pnl=-800,  # 1.6%, near 2%
        consecutive_losses=2,
        tilt_score=40,
        feed_lag_ms=50,
        feed_health="HEALTHY",
    )
    if not r6.triggered and r6.severity == "CAUTION":
        print(f" ✅ Warning state works: {r6.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Warning state failed: {r6.to_dict()}")
        failed += 1

    print("\n [Test 7] Reset...")
    cb.reset_daily()
    st7 = cb.get_status()
    if st7["recent_broker_errors"] == 0:
        print(f" ✅ Reset works: {st7}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st7}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Circuit Breaker working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()