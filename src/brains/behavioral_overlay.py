"""
Junior Aladdin - Behavioral Overlay
===================================
PURPOSE:
Apply behavioral and mature-trader filters on top of brain-generated
opportunities before they move further in the decision pipeline.

This file exists because the roadmap expects:
    src/brains/behavioral_overlay.py

RESPONSIBILITIES:
- run Behavioral Sentinel checks
- run Mature Trader rule checks
- merge both decisions into one overlay result
- filter opportunities safely
- provide a unified behavioral gate between brains and execution/scoring

CONNECTS TO:
- brain_base.py
- Behavioral Sentinel
- Mature Trader Rules
- Captain / execution approval flow
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from src.utils.logger import setup_logger
from src.psychology.behavioral_sentinel import BehavioralSentinel
from src.psychology.mature_trader import MatureTraderRules

_logger = setup_logger("behavioral_overlay")


@dataclass
class BehavioralOverlayDecision:
    original_count: int
    allowed_count: int
    blocked_count: int
    allowed_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    blocked_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "allowed_count": self.allowed_count,
            "blocked_count": self.blocked_count,
            "allowed_opportunities": self.allowed_opportunities,
            "blocked_opportunities": self.blocked_opportunities,
            "notes": self.notes,
        }


class BehavioralOverlay:
    """
    Unifies behavioral filtering over opportunities.
    """

    def __init__(self):
        self._logger = _logger
        self._sentinel = BehavioralSentinel()
        self._mature_rules = MatureTraderRules()

    def evaluate(
        self,
        opportunities: List[Dict[str, Any]],
        context: Dict[str, Any],
        live_mode: bool = False,
        checklist: Optional[Dict[str, bool]] = None,
    ) -> BehavioralOverlayDecision:
        """
        Filter a list of opportunities through behavioral layers.
        """
        allowed: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        notes: List[str] = []

        for opp in opportunities or []:
            sentinel_decision = self._sentinel.evaluate_trade(
                opportunity=opp,
                context=context,
                live_mode=live_mode,
                checklist=checklist,
            )

            if not sentinel_decision.allow_trade:
                blocked.append(
                    {
                        "opportunity": opp,
                        "block_source": "behavioral_sentinel",
                        "decision": sentinel_decision.to_dict(),
                    }
                )
                continue

            mature_decision = self._mature_rules.evaluate(
                opportunity=opp,
                context=context,
            )

            if not mature_decision.allow_trade:
                blocked.append(
                    {
                        "opportunity": opp,
                        "block_source": "mature_trader",
                        "decision": mature_decision.to_dict(),
                    }
                )
                continue

            merged = dict(opp)
            merged["_behavioral"] = {
                "sentinel": sentinel_decision.to_dict(),
                "mature_trader": mature_decision.to_dict(),
            }

            # If mature-trader warns, preserve those notes
            if mature_decision.warnings:
                notes.extend(mature_decision.warnings)

            if sentinel_decision.warnings:
                notes.extend(sentinel_decision.warnings)

            # Apply optional score override marker only; actual score mutation can happen later if desired
            if sentinel_decision.score_override is not None:
                merged["_behavioral"]["score_override"] = sentinel_decision.score_override

            allowed.append(merged)

        decision = BehavioralOverlayDecision(
            original_count=len(opportunities or []),
            allowed_count=len(allowed),
            blocked_count=len(blocked),
            allowed_opportunities=allowed,
            blocked_opportunities=blocked,
            notes=sorted(set(notes)),
        )

        self._logger.info(
            "Behavioral overlay evaluated",
            extra={
                "original_count": decision.original_count,
                "allowed_count": decision.allowed_count,
                "blocked_count": decision.blocked_count,
                "notes": decision.notes,
            },
        )

        return decision

    def record_trade_result(
        self,
        pnl_rupees: float,
        session_phase: str = "",
        was_loss: Optional[bool] = None,
    ):
        """
        Forward trade result to sentinel state.
        """
        self._sentinel.record_trade_result(
            pnl_rupees=pnl_rupees,
            session_phase=session_phase,
            was_loss=was_loss,
        )

    def record_revenge_attempt(self):
        self._sentinel.record_revenge_attempt()

    def record_fake_breakout(
        self,
        level: float,
        direction: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._mature_rules.record_fake_breakout(level, direction, metadata)

    def reset_daily(self):
        self._sentinel.reset_daily()
        self._mature_rules.reset_daily()
        self._logger.info("Behavioral Overlay reset")

    def get_status(self) -> Dict[str, Any]:
        return {
            "behavioral_sentinel": self._sentinel.get_status(),
            "mature_trader": self._mature_rules.get_status(),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Behavioral Overlay Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    overlay = BehavioralOverlay()

    print(" [Test 1] Basic clean opportunity passes...")
    opportunities = [
        {
            "strategy": "VWAP_PULLBACK",
            "direction": "BUY",
            "raw_score": 72,
            "risk_points": 10,
        }
    ]
    context = {
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "market_move_30m_pct": 0.003,
        "price_vs_vwap_pct": 0.5,
        "same_direction_as_recent_move": True,
        "microstructure": {"spread_zscore": 0.5},
        "features_1m": {
            "volume_ratio": 1.2,
            "macd_histogram": 2.0,
        },
        "key_levels": {"sr_zone_count": 2},
        "session_memory": {"failed_breakouts": []},
    }
    checklist = {
        "narrative_aligned": True,
        "score_valid": True,
        "microstructure_support": True,
        "risk_valid": True,
        "thesis_written": True,
    }
    r1 = overlay.evaluate(opportunities, context, live_mode=True, checklist=checklist)
    if r1.allowed_count == 1 and r1.blocked_count == 0:
        print(f" ✅ Clean opportunity passes: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Clean opportunity failed: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Checklist block works...")
    bad_checklist = dict(checklist)
    bad_checklist["thesis_written"] = False
    r2 = overlay.evaluate(opportunities, context, live_mode=True, checklist=bad_checklist)
    if r2.allowed_count == 0 and r2.blocked_count == 1:
        print(f" ✅ Checklist block works: {r2.blocked_opportunities[0]}")
        passed += 1
    else:
        print(f" ❌ Checklist block failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Mature trader block works...")
    bad_context = dict(context)
    bad_context["features_1m"] = {
        "volume_ratio": 0.4,
        "macd_histogram": 2.0,
    }
    r3 = overlay.evaluate(opportunities, bad_context, live_mode=False)
    if r3.allowed_count == 0 and r3.blocked_count == 1:
        print(f" ✅ Mature trader block works: {r3.blocked_opportunities[0]}")
        passed += 1
    else:
        print(f" ❌ Mature block failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Multiple opportunities filtering...")
    opps4 = [
        {
            "strategy": "VWAP_PULLBACK",
            "direction": "BUY",
            "raw_score": 72,
            "risk_points": 10,
        },
        {
            "strategy": "TREND_CONTINUATION",
            "direction": "BUY",
            "raw_score": 68,
            "risk_points": 10,
        },
    ]
    r4 = overlay.evaluate(opps4, context, live_mode=False)
    if r4.original_count == 2:
        print(f" ✅ Multiple opportunities handled: {r4.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Multiple handling failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Record trade result updates state...")
    overlay.record_trade_result(-300, session_phase="GOLDEN_AM", was_loss=True)
    st5 = overlay.get_status()
    if st5["behavioral_sentinel"]["consecutive_losses"] == 1:
        print(f" ✅ Trade result state update works: {st5}")
        passed += 1
    else:
        print(f" ❌ Trade result state update failed: {st5}")
        failed += 1

    print("\n [Test 6] Record fake breakout...")
    overlay.record_fake_breakout(23200, "BUY")
    st6 = overlay.get_status()
    if st6["mature_trader"]["recent_fake_breakouts"] == 1:
        print(f" ✅ Fake breakout memory works: {st6}")
        passed += 1
    else:
        print(f" ❌ Fake breakout memory failed: {st6}")
        failed += 1

    print("\n [Test 7] Reset daily...")
    overlay.reset_daily()
    st7 = overlay.get_status()
    if (
        st7["behavioral_sentinel"]["consecutive_losses"] == 0
        and st7["mature_trader"]["recent_fake_breakouts"] == 0
    ):
        print(f" ✅ Reset works: {st7}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st7}")
        failed += 1

    print("\n [Test 8] Empty opportunities safe...")
    r8 = overlay.evaluate([], context, live_mode=False)
    if r8.original_count == 0 and r8.allowed_count == 0:
        print(f" ✅ Empty opportunities safe: {r8.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Empty handling failed: {r8.to_dict()}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Behavioral Overlay working perfectly!")
        print(" ✅ Current-phase missing behavioral overlay completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()