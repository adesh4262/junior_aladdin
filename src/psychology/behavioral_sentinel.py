"""
Junior Aladdin - Behavioral Sentinel
====================================
PURPOSE:
Protect the system from psychologically bad or impulsive decisions.

This module simulates professional behavioral discipline:
- revenge detection
- FOMO prevention
- overtrading control
- tilt escalation
- live checklist gating

This is NOT a strategy.
It is a safety filter between scored opportunities and execution.

CORE FEATURES:
1. Pre-trade checklist
2. Revenge-trading cooldowns
3. FOMO score override
4. Lunch overtrading block
5. Daily overtrading block
6. Tilt score management
7. Automatic disable when emotional state is too poor

CONNECTS TO:
- Captain / execution approval
- Risk engine
- Dashboard behavioral panel
- Journal
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import math
from typing import Dict, List, Optional, Any, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

IST = timezone(timedelta(hours=5, minutes=30))
_logger = setup_logger("behavioral_sentinel")


@dataclass
class BehavioralDecision:
    """
    Result of behavioral review.
    """
    allow_trade: bool
    block_reason: str = ""
    tilt_score: float = 0.0
    score_override: Optional[float] = None
    cooldown_active: bool = False
    cooldown_minutes_left: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_trade": self.allow_trade,
            "block_reason": self.block_reason,
            "tilt_score": self.tilt_score,
            "score_override": self.score_override,
            "cooldown_active": self.cooldown_active,
            "cooldown_minutes_left": self.cooldown_minutes_left,
            "warnings": self.warnings,
        }


class BehavioralSentinel:
    """
    Behavioral safety engine.
    """

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

    def __init__(self):
        self._logger = _logger

        self._cooldown_1 = max(
            0,
            self._safe_int(Config.get("behavioral", "cooldown_after_1_loss_min", default=5), 5),
        )
        self._cooldown_2 = max(
            0,
            self._safe_int(Config.get("behavioral", "cooldown_after_2_losses_min", default=15), 15),
        )
        self._cooldown_3 = max(
            0,
            self._safe_int(Config.get("behavioral", "cooldown_after_3_losses_min", default=30), 30),
        )

        self._fomo_move_threshold = self._safe_float(
            Config.get("behavioral", "fomo_market_move_threshold", default=0.01),
            0.01,
        )
        self._fomo_min_score = self._safe_float(
            Config.get("behavioral", "fomo_min_score_override", default=78),
            78.0,
        )

        self._max_trades_in_lunch = max(
            0,
            self._safe_int(Config.get("behavioral", "max_trades_in_lunch", default=2), 2),
        )
        self._tilt_warning = max(
            0,
            self._safe_int(Config.get("behavioral", "tilt_warning_threshold", default=50), 50),
        )
        self._tilt_reduce = max(
            0,
            self._safe_int(Config.get("behavioral", "tilt_reduce_threshold", default=70), 70),
        )
        self._tilt_disable = max(
            0,
            self._safe_int(Config.get("behavioral", "tilt_disable_threshold", default=85), 85),
        )

        self._tilt_consecutive_loss_points = max(
            0,
            self._safe_int(
                Config.get("behavioral", "tilt_consecutive_loss_points", default=15),
                15,
            ),
        )
        self._tilt_negative_pnl_max_points = max(
            0,
            self._safe_int(
                Config.get("behavioral", "tilt_negative_pnl_max_points", default=20),
                20,
            ),
        )
        self._tilt_overtrading_max_points = max(
            0,
            self._safe_int(
                Config.get("behavioral", "tilt_overtrading_max_points", default=20),
                20,
            ),
        )
        self._tilt_revenge_attempt_points = max(
            0,
            self._safe_int(
                Config.get("behavioral", "tilt_revenge_attempt_points", default=15),
                15,
            ),
        )

        self._tilt_pnl_divisor = self._safe_float(
            Config.get("behavioral", "tilt_pnl_divisor", default=1000.0),
            1000.0,
        )
        if self._tilt_pnl_divisor <= 0:
            self._tilt_pnl_divisor = 1000.0

        self._last_loss_time: Optional[datetime] = None
        self._consecutive_losses: int = 0
        self._revenge_attempts: int = 0
        self._trades_today: int = 0
        self._lunch_trades_today: int = 0
        self._daily_pnl: float = 0.0
        self._cooldown_until: Optional[datetime] = None
        self._tilt_score: float = 0.0

    def _reject(
        self,
        block_reason: str,
        tilt_score: Optional[float] = None,
        score_override: Optional[float] = None,
        cooldown_active: bool = False,
        cooldown_minutes_left: float = 0.0,
        warnings: Optional[List[str]] = None,
    ) -> BehavioralDecision:
        warnings = list(warnings or [])
        self._logger.warning(
            "Behavioral trade blocked",
            extra={"block_reason": block_reason, "tilt_score": self._tilt_score},
        )
        return BehavioralDecision(
            allow_trade=False,
            block_reason=block_reason,
            tilt_score=self._tilt_score if tilt_score is None else tilt_score,
            score_override=score_override,
            cooldown_active=cooldown_active,
            cooldown_minutes_left=cooldown_minutes_left,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def evaluate_trade(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
        live_mode: bool = False,
        checklist: Optional[Dict[str, bool]] = None,
    ) -> BehavioralDecision:
        """
        Evaluate whether trade should be allowed behaviorally.
        """
        now = datetime.now(IST)
        warnings: List[str] = []

        self._recompute_tilt(context)

        cooldown_active, cooldown_left = self._get_cooldown_status(now)
        if cooldown_active:
            return self._reject(
                block_reason="cooldown_active",
                cooldown_active=True,
                cooldown_minutes_left=round(cooldown_left, 1),
                warnings=warnings,
            )

        # Hard disable on extreme tilt
        if self._tilt_score > self._tilt_disable:
            return self._reject(
                block_reason="tilt_disable",
                warnings=warnings,
            )

        # Live checklist gating
        if live_mode:
            checklist = checklist if isinstance(checklist, dict) else {}
            required = [
                "narrative_aligned",
                "score_valid",
                "microstructure_support",
                "risk_valid",
                "thesis_written",
            ]
            missing = [k for k in required if not checklist.get(k, False)]
            if missing:
                return self._reject(
                    block_reason=f"checklist_incomplete:{','.join(missing)}",
                    warnings=warnings,
                )

        # Lunch overtrading block
        session = str(context.get("session_phase", "UNKNOWN"))
        if session == "LUNCH_LULL" and self._lunch_trades_today >= self._max_trades_in_lunch:
            return self._reject(
                block_reason="lunch_overtrading_block",
                warnings=warnings,
            )

        # Daily overtrading block
        if self._trades_today >= 5:
            return self._reject(
                block_reason="daily_overtrading_block",
                warnings=warnings,
            )

        # FOMO detection
        score_override = self._check_fomo(opportunity, context)
        if score_override is not None:
            opp_score = self._safe_float(opportunity.get("raw_score"), 0.0)
            if opp_score < score_override:
                return self._reject(
                    block_reason="fomo_block",
                    score_override=score_override,
                    warnings=["market_extended_fomo_risk"],
                )
            warnings.append(f"fomo_override_active:{score_override}")

        # Soft warnings
        if self._tilt_score > self._tilt_warning:
            warnings.append("tilt_warning")
        if self._tilt_score > self._tilt_reduce:
            warnings.append("size_should_reduce")

        return BehavioralDecision(
            allow_trade=True,
            tilt_score=self._tilt_score,
            score_override=score_override,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Trade outcome updates
    # ------------------------------------------------------------------
    def record_trade_result(
        self,
        pnl_rupees: float,
        session_phase: str = "",
        was_loss: Optional[bool] = None,
    ):
        """
        Update internal behavioral state after a trade closes.
        """
        now = datetime.now(IST)
        pnl = self._safe_float(pnl_rupees, 0.0)
        self._trades_today += 1
        self._daily_pnl += pnl

        if session_phase == "LUNCH_LULL":
            self._lunch_trades_today += 1

        if was_loss is None:
            was_loss = pnl < 0

        if was_loss:
            self._last_loss_time = now
            self._consecutive_losses += 1
            self._apply_loss_cooldown(now)
        else:
            self._consecutive_losses = 0

        self._recompute_tilt({})

        self._logger.info(
            "Trade result recorded",
            extra={
                "pnl_rupees": pnl,
                "consecutive_losses": self._consecutive_losses,
                "daily_pnl": self._daily_pnl,
                "tilt_score": self._tilt_score,
            },
        )

    def record_revenge_attempt(self):
        """
        Call when user/system attempts trade during revenge conditions.
        """
        self._revenge_attempts += 1
        self._recompute_tilt({})

        self._logger.warning(
            "Revenge attempt recorded",
            extra={
                "revenge_attempts": self._revenge_attempts,
                "tilt_score": self._tilt_score,
            },
        )

    # ------------------------------------------------------------------
    # Checklist
    # ------------------------------------------------------------------
    def build_live_checklist(
        self,
        context: Dict[str, Any],
        opportunity: Dict[str, Any],
        thesis_written: bool = False,
    ) -> Dict[str, bool]:
        """
        Build the mandatory live pre-trade checklist.
        """
        narrative_label = str(context.get("narrative_label", "NEUTRAL"))
        micro = context.get("microstructure", {}) or {}
        spread_zscore = micro.get("spread_zscore")
        raw_score = self._safe_float(opportunity.get("raw_score"), 0.0)
        risk_points = self._safe_float(opportunity.get("risk_points"), 0.0)

        checklist = {
            "narrative_aligned": narrative_label != "EVENT_RISK",
            "score_valid": raw_score >= 65,
            "microstructure_support": spread_zscore is None
            or self._safe_float(spread_zscore, 999.0) < 2.0,
            "risk_valid": risk_points > 0,
            "thesis_written": thesis_written,
        }
        return checklist

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def _apply_loss_cooldown(self, now: datetime):
        if self._consecutive_losses == 1:
            minutes = self._cooldown_1
        elif self._consecutive_losses == 2:
            minutes = self._cooldown_2
        else:
            minutes = self._cooldown_3

        self._cooldown_until = now + timedelta(minutes=minutes)

    def _get_cooldown_status(self, now: datetime) -> Tuple[bool, float]:
        if self._cooldown_until is None:
            return False, 0.0
        seconds = (self._cooldown_until - now).total_seconds()
        if seconds <= 0:
            self._cooldown_until = None
            return False, 0.0
        return True, seconds / 60.0

    def _check_fomo(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> Optional[float]:
        """
        If market already moved too much too fast, require stronger score.
        """
        market_move = self._safe_float(context.get("market_move_30m_pct"), 0.0)
        price_vs_vwap = abs(self._safe_float(context.get("price_vs_vwap_pct"), 0.0))
        same_direction = bool(context.get("same_direction_as_recent_move", True))

        if (
            market_move > self._fomo_move_threshold
            and same_direction
            and price_vs_vwap > 1.5
        ):
            return float(self._fomo_min_score)

        return None

    def _recompute_tilt(self, context: Dict[str, Any]):
        """
        Tilt score 0-100.
        """
        score = 0.0

        # consecutive losses
        consecutive_component = min(self._consecutive_losses, 3) * self._tilt_consecutive_loss_points
        consecutive_component = min(
            consecutive_component,
            3 * self._tilt_consecutive_loss_points,
        )
        score += max(0.0, consecutive_component)

        # negative pnl worsening
        if self._daily_pnl < 0:
            daily_loss_fraction = min(abs(self._daily_pnl) / self._tilt_pnl_divisor, 1.0)
            daily_loss_fraction = max(0.0, daily_loss_fraction)
            pnl_component = daily_loss_fraction * self._tilt_negative_pnl_max_points
            score += min(pnl_component, self._tilt_negative_pnl_max_points)

        # overtrading
        overtrade_ratio = min(self._trades_today / 5.0, 1.0)
        overtrade_ratio = max(0.0, overtrade_ratio)
        overtrade_component = overtrade_ratio * self._tilt_overtrading_max_points
        score += min(overtrade_component, self._tilt_overtrading_max_points)

        # revenge attempts
        revenge_component = self._revenge_attempts * self._tilt_revenge_attempt_points
        score += max(0.0, revenge_component)

        self._tilt_score = round(min(score, 100.0), 1)

    # ------------------------------------------------------------------
    # Status / reset
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        cooldown_active, cooldown_left = self._get_cooldown_status(datetime.now(IST))
        return {
            "tilt_score": self._tilt_score,
            "consecutive_losses": self._consecutive_losses,
            "revenge_attempts": self._revenge_attempts,
            "trades_today": self._trades_today,
            "lunch_trades_today": self._lunch_trades_today,
            "daily_pnl": round(self._daily_pnl, 2),
            "cooldown_active": cooldown_active,
            "cooldown_minutes_left": round(cooldown_left, 1),
        }

    def reset_daily(self):
        self._last_loss_time = None
        self._consecutive_losses = 0
        self._revenge_attempts = 0
        self._trades_today = 0
        self._lunch_trades_today = 0
        self._daily_pnl = 0.0
        self._cooldown_until = None
        self._tilt_score = 0.0
        self._logger.info("Behavioral Sentinel reset")


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Behavioral Sentinel Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    bs = BehavioralSentinel()

    print(" [Test 1] Initial state...")
    st = bs.get_status()
    if st["tilt_score"] == 0 and st["trades_today"] == 0:
        print(f" ✅ Initial state clean: {st}")
        passed += 1
    else:
        print(f" ❌ Bad initial state: {st}")
        failed += 1

    print("\n [Test 2] Valid trade allowed...")
    opp = {"raw_score": 72, "risk_points": 10}
    ctx = {
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "market_move_30m_pct": 0.003,
        "price_vs_vwap_pct": 0.5,
        "same_direction_as_recent_move": True,
        "microstructure": {"spread_zscore": 0.5},
    }
    checklist = {
        "narrative_aligned": True,
        "score_valid": True,
        "microstructure_support": True,
        "risk_valid": True,
        "thesis_written": True,
    }
    d2 = bs.evaluate_trade(opp, ctx, live_mode=True, checklist=checklist)
    if d2.allow_trade:
        print(f" ✅ Trade allowed: {d2.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Should allow trade: {d2.to_dict()}")
        failed += 1

    print("\n [Test 3] Checklist incomplete blocks live trade...")
    bad_checklist = dict(checklist)
    bad_checklist["thesis_written"] = False
    d3 = bs.evaluate_trade(opp, ctx, live_mode=True, checklist=bad_checklist)
    if not d3.allow_trade and "checklist_incomplete" in d3.block_reason:
        print(f" ✅ Checklist block works: {d3.block_reason}")
        passed += 1
    else:
        print(f" ❌ Checklist block failed: {d3.to_dict()}")
        failed += 1

    print("\n [Test 4] First loss triggers cooldown...")
    bs.record_trade_result(-300, session_phase="GOLDEN_AM", was_loss=True)
    d4 = bs.evaluate_trade(opp, ctx, live_mode=False)
    if not d4.allow_trade and d4.cooldown_active:
        print(f" ✅ Cooldown active: {d4.cooldown_minutes_left:.1f} min")
        passed += 1
    else:
        print(f" ❌ Cooldown missing: {d4.to_dict()}")
        failed += 1

    print("\n [Test 5] Revenge attempt increases tilt...")
    bs.record_revenge_attempt()
    st5 = bs.get_status()
    if st5["revenge_attempts"] == 1 and st5["tilt_score"] > 0:
        print(f" ✅ Revenge attempt tracked: tilt={st5['tilt_score']}")
        passed += 1
    else:
        print(f" ❌ Revenge tracking failed: {st5}")
        failed += 1

    print("\n [Test 6] FOMO block...")
    bs.reset_daily()
    fomo_ctx = {
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "market_move_30m_pct": 0.02,
        "price_vs_vwap_pct": 1.8,
        "same_direction_as_recent_move": True,
        "microstructure": {"spread_zscore": 0.5},
    }
    weak_opp = {"raw_score": 68, "risk_points": 8}
    d6 = bs.evaluate_trade(weak_opp, fomo_ctx, live_mode=False)
    if not d6.allow_trade and d6.block_reason == "fomo_block":
        print(f" ✅ FOMO block works: override={d6.score_override}")
        passed += 1
    else:
        print(f" ❌ FOMO block failed: {d6.to_dict()}")
        failed += 1

    print("\n [Test 7] Lunch overtrading block...")
    bs.reset_daily()
    bs._lunch_trades_today = 2
    lunch_ctx = dict(ctx)
    lunch_ctx["session_phase"] = "LUNCH_LULL"
    d7 = bs.evaluate_trade(opp, lunch_ctx, live_mode=False)
    if not d7.allow_trade and d7.block_reason == "lunch_overtrading_block":
        print(" ✅ Lunch overtrading block works")
        passed += 1
    else:
        print(f" ❌ Lunch block failed: {d7.to_dict()}")
        failed += 1

    print("\n [Test 8] Daily overtrading block...")
    bs.reset_daily()
    bs._trades_today = 5
    d8 = bs.evaluate_trade(opp, ctx, live_mode=False)
    if not d8.allow_trade and d8.block_reason == "daily_overtrading_block":
        print(" ✅ Daily overtrading block works")
        passed += 1
    else:
        print(f" ❌ Daily block failed: {d8.to_dict()}")
        failed += 1

    print("\n [Test 9] Extreme tilt disables...")
    bs.reset_daily()
    bs._consecutive_losses = 3
    bs._revenge_attempts = 3
    bs._daily_pnl = -3000
    bs._trades_today = 5
    bs._recompute_tilt({})
    d9 = bs.evaluate_trade(opp, ctx, live_mode=False)
    if not d9.allow_trade and d9.block_reason == "tilt_disable":
        print(f" ✅ Tilt disable works: tilt={d9.tilt_score}")
        passed += 1
    else:
        print(f" ❌ Tilt disable failed: {d9.to_dict()}")
        failed += 1

    print("\n [Test 10] Reset daily...")
    bs.reset_daily()
    st10 = bs.get_status()
    if (
        st10["tilt_score"] == 0
        and st10["trades_today"] == 0
        and st10["revenge_attempts"] == 0
    ):
        print(f" ✅ Reset works: {st10}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st10}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Behavioral Sentinel working perfectly!")
        print(" ✅ Ready for next roadmap step.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()