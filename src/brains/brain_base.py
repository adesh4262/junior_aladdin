"""
Junior Aladdin - Brain Engine Base & Selector (Hardened Version)
================================================================
PURPOSE:
Coordinate the 4 trading brains and 2 overlays with stronger,
production-grade routing logic.

BRAINS:
1. STRUCTURAL
2. TACTICAL
3. INSTITUTIONAL
4. ADAPTIVE

OVERLAYS:
A. EVENT
B. BEHAVIORAL

This module does NOT execute trades.
It decides:
- which brains are active
- whether the system is restricted
- what direction is preferred
- how overlays modify behavior

This hardened version improves:
- stricter restriction handling
- more explicit event behavior
- better fallback logic
- cleaner regime/session selection
- safer direction preference derivation

CONNECTS TO:
- Narrative Engine
- Regime Engine
- Time Context Engine
- Feature Engine / MTF / Smart Money outputs
- Strategies
- Captain Orchestrator
"""

from dataclasses import dataclass, field
from typing import Dict, List

from src.utils.logger import setup_logger

_logger = setup_logger("brain_engine")


@dataclass
class BrainDecision:
    """
    Standardized output of the Brain Engine.
    """
    active_brains: List[str] = field(default_factory=list)
    event_overlay: bool = False
    behavioral_overlay: bool = True
    restricted: bool = False
    restriction_reason: str = ""
    preferred_direction: str = "BOTH"  # BUY / SELL / BOTH / NONE
    size_factor: float = 1.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "active_brains": self.active_brains,
            "event_overlay": self.event_overlay,
            "behavioral_overlay": self.behavioral_overlay,
            "restricted": self.restricted,
            "restriction_reason": self.restriction_reason,
            "preferred_direction": self.preferred_direction,
            "size_factor": self.size_factor,
            "notes": self.notes,
        }


class BrainEngine:
    """
    Select active brains based on regime, narrative, session, data quality,
    and feature-state alignment.

    HARD RULES:
    - Max 2 brains active at once
    - EVENT overlay reduces size aggressively
    - CHOP prefers ADAPTIVE only
    - Low data quality restricts all trading
    - Tactical-only session overrides normal selection
    """

    PRIORITY_ORDER = [
        "STRUCTURAL",
        "TACTICAL",
        "INSTITUTIONAL",
        "ADAPTIVE",
    ]

    def __init__(self):
        self._logger = _logger

    def select_brains(
        self,
        regime_data: Dict,
        narrative_data: Dict,
        time_context: Dict,
        feature_data: Dict,
        data_quality_score: float = 100.0,
    ) -> BrainDecision:
        """
        Main selector.

        Inputs:
        - regime_data: regime engine output
        - narrative_data: narrative engine output
        - time_context: time-context output
        - feature_data: selected context / meta features
        - data_quality_score: current feed/data quality

        Returns:
        BrainDecision
        """
        decision = BrainDecision()

        regime = str(regime_data.get("regime", "UNKNOWN"))
        narrative_label = str(narrative_data.get("narrative_label", "NEUTRAL"))
        weighted_mtf = float(feature_data.get("weighted_mtf", 0.0) or 0.0)
        smart_money_score = float(feature_data.get("sm_direction_score", 0.0) or 0.0)
        session_phase = str(time_context.get("session_phase", "UNKNOWN"))
        tactical_only = bool(time_context.get("tactical_only", False))
        trading_allowed = bool(time_context.get("trading_allowed", True))
        day_type = str(time_context.get("day_type", "UNKNOWN"))

        is_event_day = day_type == "EVENT_DAY" or narrative_label == "EVENT_RISK"

        # ------------------------------------------------------------------
        # 1. Hard restrictions
        # ------------------------------------------------------------------
        if not trading_allowed:
            decision.restricted = True
            decision.restriction_reason = f"Trading not allowed in session {session_phase}"
            decision.preferred_direction = "NONE"
            decision.notes.append("Session blocks new entries")
            return decision

        if data_quality_score < 60:
            decision.restricted = True
            decision.restriction_reason = f"Low data quality ({data_quality_score})"
            decision.preferred_direction = "NONE"
            decision.notes.append("SAFE mode recommended")
            return decision

        # ------------------------------------------------------------------
        # 2. Event overlay
        # ------------------------------------------------------------------
        if is_event_day:
            decision.event_overlay = True
            decision.size_factor *= 0.3
            decision.notes.append("Event overlay active: size reduced to 30%")

        # ------------------------------------------------------------------
        # 3. Preferred direction
        # ------------------------------------------------------------------
        decision.preferred_direction = self._derive_direction(
            narrative_label=narrative_label,
            weighted_mtf=weighted_mtf,
        )

        # ------------------------------------------------------------------
        # 4. Tactical-only session override
        # ------------------------------------------------------------------
        if tactical_only:
            decision.active_brains = ["TACTICAL"]
            decision.notes.append("Tactical-only session override")
            return decision

        # ------------------------------------------------------------------
        # 5. Regime-based base selection
        # ------------------------------------------------------------------
        active: List[str] = []

        if regime == "TRENDING":
            if abs(weighted_mtf) >= 4.5:
                active.append("STRUCTURAL")
            elif abs(weighted_mtf) >= 3.0:
                active.append("TACTICAL")

            if abs(smart_money_score) >= 30:
                active.append("INSTITUTIONAL")

        elif regime == "RANGE":
            active.append("STRUCTURAL")

            if session_phase in ("GOLDEN_PM", "INITIAL_BALANCE"):
                if abs(weighted_mtf) < 3.0:
                    active.append("TACTICAL")

        elif regime == "VOLATILE":
            if abs(weighted_mtf) >= 3.0:
                active.append("STRUCTURAL")
            active.append("TACTICAL")

        elif regime == "CHOP":
            active.append("ADAPTIVE")

        elif regime == "EVENT":
            decision.event_overlay = True
            active.append("TACTICAL")
            decision.size_factor *= 0.5
            decision.notes.append("Event regime: tactical only")

        else:
            # fallback mode
            if abs(weighted_mtf) >= 4.5:
                active.append("STRUCTURAL")
            elif abs(smart_money_score) >= 30:
                active.append("INSTITUTIONAL")
            elif abs(weighted_mtf) >= 2.0:
                active.append("TACTICAL")
            else:
                active.append("ADAPTIVE")

        # ------------------------------------------------------------------
        # 6. Priority ordering / dedupe
        # ------------------------------------------------------------------
        active = self._dedupe_by_priority(active)

        # ------------------------------------------------------------------
        # 7. Max 2 brains
        # ------------------------------------------------------------------
        if len(active) > 2:
            active = active[:2]
            decision.notes.append("Trimmed to max 2 brains")

        # ------------------------------------------------------------------
        # 8. If event overlay active, suppress structural/institutional excess
        # ------------------------------------------------------------------
        if decision.event_overlay:
            if "TACTICAL" in active:
                active = ["TACTICAL"]
            elif active:
                active = [active[0]]
            decision.notes.append("Event overlay narrowed active brains")

        decision.active_brains = active

        # ------------------------------------------------------------------
        # 9. Behavioral overlay always on
        # ------------------------------------------------------------------
        decision.behavioral_overlay = True

        # ------------------------------------------------------------------
        # 10. Final sanity fallback
        # ------------------------------------------------------------------
        if not decision.active_brains:
            decision.active_brains = ["ADAPTIVE"]
            decision.notes.append("Fallback to ADAPTIVE")

        self._logger.info(
            "Brain selection complete",
            extra={
                "regime": regime,
                "narrative": narrative_label,
                "weighted_mtf": weighted_mtf,
                "smart_money_score": smart_money_score,
                "session": session_phase,
                "brains": decision.active_brains,
                "preferred_direction": decision.preferred_direction,
                "size_factor": decision.size_factor,
                "event_overlay": decision.event_overlay,
            },
        )

        return decision

    def _derive_direction(self, narrative_label: str, weighted_mtf: float) -> str:
        """
        Derive preferred direction from macro + trend alignment.
        """
        bullish_narrative = narrative_label in ("MILD_BULLISH", "STRONG_BULLISH")
        bearish_narrative = narrative_label in ("MILD_BEARISH", "STRONG_BEARISH")

        if narrative_label == "EVENT_RISK":
            return "NONE"

        if weighted_mtf >= 3.0 and not bearish_narrative:
            return "BUY"

        if weighted_mtf <= -3.0 and not bullish_narrative:
            return "SELL"

        if bullish_narrative and weighted_mtf >= 0:
            return "BUY"

        if bearish_narrative and weighted_mtf <= 0:
            return "SELL"

        return "BOTH"

    def _dedupe_by_priority(self, brains: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []

        for brain in self.PRIORITY_ORDER:
            if brain in brains and brain not in seen:
                ordered.append(brain)
                seen.add(brain)

        return ordered


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Brain Engine Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    engine = BrainEngine()

    print(" [Test 1] TRENDING + strong MTF...")
    d1 = engine.select_brains(
        regime_data={"regime": "TRENDING"},
        narrative_data={"narrative_label": "MILD_BULLISH"},
        time_context={
            "session_phase": "GOLDEN_AM",
            "trading_allowed": True,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": 5.5, "sm_direction_score": 10},
        data_quality_score=85,
    )
    if "STRUCTURAL" in d1.active_brains:
        print(f" ✅ Brains={d1.active_brains}")
        passed += 1
    else:
        print(f" ❌ Unexpected brains={d1.active_brains}")
        failed += 1

    print("\n [Test 2] TRENDING + strong smart money...")
    d2 = engine.select_brains(
        regime_data={"regime": "TRENDING"},
        narrative_data={"narrative_label": "NEUTRAL"},
        time_context={
            "session_phase": "GOLDEN_PM",
            "trading_allowed": True,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": 4.8, "sm_direction_score": 40},
        data_quality_score=90,
    )
    if "STRUCTURAL" in d2.active_brains and "INSTITUTIONAL" in d2.active_brains:
        print(f" ✅ Brains={d2.active_brains}")
        passed += 1
    else:
        print(f" ❌ Unexpected brains={d2.active_brains}")
        failed += 1

    print("\n [Test 3] RANGE lunch tactical-only...")
    d3 = engine.select_brains(
        regime_data={"regime": "RANGE"},
        narrative_data={"narrative_label": "NEUTRAL"},
        time_context={
            "session_phase": "LUNCH_LULL",
            "trading_allowed": True,
            "tactical_only": True,
        },
        feature_data={"weighted_mtf": 1.0, "sm_direction_score": 0},
        data_quality_score=80,
    )
    if d3.active_brains == ["TACTICAL"]:
        print(f" ✅ Brains={d3.active_brains}")
        passed += 1
    else:
        print(f" ❌ Unexpected brains={d3.active_brains}")
        failed += 1

    print("\n [Test 4] VOLATILE regime...")
    d4 = engine.select_brains(
        regime_data={"regime": "VOLATILE"},
        narrative_data={"narrative_label": "MILD_BULLISH"},
        time_context={
            "session_phase": "GOLDEN_AM",
            "trading_allowed": True,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": 3.5, "sm_direction_score": 20},
        data_quality_score=88,
    )
    if "TACTICAL" in d4.active_brains:
        print(f" ✅ Brains={d4.active_brains}")
        passed += 1
    else:
        print(f" ❌ Unexpected brains={d4.active_brains}")
        failed += 1

    print("\n [Test 5] CHOP regime...")
    d5 = engine.select_brains(
        regime_data={"regime": "CHOP"},
        narrative_data={"narrative_label": "NEUTRAL"},
        time_context={
            "session_phase": "GOLDEN_PM",
            "trading_allowed": True,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": 0.5, "sm_direction_score": 0},
        data_quality_score=86,
    )
    if d5.active_brains == ["ADAPTIVE"]:
        print(f" ✅ Brains={d5.active_brains}")
        passed += 1
    else:
        print(f" ❌ Unexpected brains={d5.active_brains}")
        failed += 1

    print("\n [Test 6] EVENT day...")
    d6 = engine.select_brains(
        regime_data={"regime": "EVENT"},
        narrative_data={"narrative_label": "EVENT_RISK"},
        time_context={
            "session_phase": "INITIAL_BALANCE",
            "trading_allowed": True,
            "tactical_only": False,
            "day_type": "EVENT_DAY",
        },
        feature_data={"weighted_mtf": 0.0, "sm_direction_score": 0},
        data_quality_score=90,
    )
    if d6.event_overlay and "TACTICAL" in d6.active_brains and d6.size_factor < 1.0:
        print(f" ✅ Brains={d6.active_brains}, size_factor={d6.size_factor}")
        passed += 1
    else:
        print(f" ❌ Unexpected decision={d6.to_dict()}")
        failed += 1

    print("\n [Test 7] Trading blocked by session...")
    d7 = engine.select_brains(
        regime_data={"regime": "TRENDING"},
        narrative_data={"narrative_label": "MILD_BULLISH"},
        time_context={
            "session_phase": "LAST_MINUTES",
            "trading_allowed": False,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": 6.0, "sm_direction_score": 20},
        data_quality_score=90,
    )
    if d7.restricted and d7.preferred_direction == "NONE":
        print(f" ✅ Restricted: {d7.restriction_reason}")
        passed += 1
    else:
        print(f" ❌ Unexpected decision={d7.to_dict()}")
        failed += 1

    print("\n [Test 8] Data quality restriction...")
    d8 = engine.select_brains(
        regime_data={"regime": "TRENDING"},
        narrative_data={"narrative_label": "MILD_BULLISH"},
        time_context={
            "session_phase": "GOLDEN_AM",
            "trading_allowed": True,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": 6.0, "sm_direction_score": 20},
        data_quality_score=45,
    )
    if d8.restricted:
        print(" ✅ Restricted by data quality")
        passed += 1
    else:
        print(" ❌ Should be restricted")
        failed += 1

    print("\n [Test 9] Direction derivation...")
    d9 = engine.select_brains(
        regime_data={"regime": "TRENDING"},
        narrative_data={"narrative_label": "MILD_BEARISH"},
        time_context={
            "session_phase": "GOLDEN_PM",
            "trading_allowed": True,
            "tactical_only": False,
        },
        feature_data={"weighted_mtf": -4.0, "sm_direction_score": 0},
        data_quality_score=80,
    )
    if d9.preferred_direction == "SELL":
        print(f" ✅ Preferred direction={d9.preferred_direction}")
        passed += 1
    else:
        print(f" ❌ Unexpected preferred direction={d9.preferred_direction}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Brain Engine (Hardened) working perfectly!")
        print(" ✅ Ready for next roadmap step.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()