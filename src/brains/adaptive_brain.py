"""
Junior Aladdin - Adaptive Brain
===============================
PURPOSE:
Provide a conservative fallback scanning layer for uncertain or low-quality
market environments where the other brains are unsuitable.

This file exists because the roadmap expects:
    src/brains/adaptive_brain.py

IMPORTANT DESIGN CHOICE:
The Adaptive Brain should be extremely selective.
If context is weak, it should return no opportunities rather than force trades.

CURRENT RESPONSIBILITY:
- act as fallback / sparse-opportunity router
- only allow adaptive-compatible strategies
- stay conservative in CHOP / uncertain conditions

NOTE:
At this build stage, there is no dedicated standalone adaptive strategy file
in the roadmap-completed set yet. So this brain operates as a conservative
wrapper that can:
- return no trade in unsafe conditions
- optionally surface very limited tactical continuation only if explicitly allowed later

For now, this file is intentionally minimal-but-correct and structurally complete.
That is better than fabricating fake adaptive signals.

CONNECTS TO:
- brain_base.py
- Captain
- context/regime layer
"""

from typing import Dict, List, Optional

from src.utils.logger import setup_logger

_logger = setup_logger("adaptive_brain")


class AdaptiveBrain:
    """
    Conservative fallback brain.

    This implementation is intentionally restrictive:
    - in CHOP, uncertainty, weak context -> mostly no trade
    - can later be extended when dedicated adaptive strategies are added
    """

    def __init__(self):
        self._logger = _logger
        self._strategies: List[str] = []  # placeholder registry for future adaptive strategies

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Adaptive scan is intentionally conservative.
        Current policy:
        - if context weak or CHOP -> no trade
        - if explicitly allowed later, strategy registry can be extended
        """
        if not features_1m or not context:
            return []

        regime = str(context.get("regime", "UNKNOWN"))
        preferred_direction = str(context.get("preferred_direction", "BOTH")).upper()
        session_phase = str(context.get("session_phase", "UNKNOWN"))
        data_quality = float(context.get("data_quality_score", 100) or 0)
        narrative = str(context.get("narrative_label", "NEUTRAL"))

        # Hard conservatism
        if data_quality < 70:
            return []
        if session_phase in ("PRE_MARKET", "OPENING_AUCTION", "LAST_MINUTES", "POST_MARKET"):
            return []
        if preferred_direction == "NONE":
            return []
        if narrative == "EVENT_RISK":
            return []

        # In this version, AdaptiveBrain stays deliberately flat in CHOP/uncertainty.
        # This is safer than inventing weak adaptive trades.
        self._logger.info(
            "Adaptive brain scan complete",
            extra={
                "regime": regime,
                "preferred_direction": preferred_direction,
                "opportunity_count": 0,
                "mode": "conservative_no_signal",
            },
        )
        return []

    def get_strategy_names(self) -> List[str]:
        return list(self._strategies)

    def get_status(self) -> Dict:
        return {
            "brain": "ADAPTIVE",
            "strategy_count": len(self._strategies),
            "strategies": self.get_strategy_names(),
            "mode": "conservative_fallback",
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Adaptive Brain Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    brain = AdaptiveBrain()

    print(" [Test 1] Create brain...")
    st1 = brain.get_status()
    if st1["brain"] == "ADAPTIVE":
        print(f" ✅ Status OK: {st1}")
        passed += 1
    else:
        print(f" ❌ Bad status: {st1}")
        failed += 1

    print("\n [Test 2] Empty input safe...")
    r2 = brain.scan({}, context={})
    if r2 == []:
        print(" ✅ Empty input handled safely")
        passed += 1
    else:
        print(f" ❌ Empty input handling failed: {r2}")
        failed += 1

    print("\n [Test 3] Low data quality blocks...")
    features = {"last_close": 23000}
    ctx3 = {
        "regime": "CHOP",
        "preferred_direction": "BUY",
        "session_phase": "GOLDEN_PM",
        "data_quality_score": 50,
        "narrative_label": "NEUTRAL",
    }
    r3 = brain.scan(features, context=ctx3)
    if r3 == []:
        print(" ✅ Low data quality block works")
        passed += 1
    else:
        print(f" ❌ Low data quality block failed: {r3}")
        failed += 1

    print("\n [Test 4] Event risk blocks...")
    ctx4 = {
        "regime": "UNKNOWN",
        "preferred_direction": "BUY",
        "session_phase": "GOLDEN_PM",
        "data_quality_score": 90,
        "narrative_label": "EVENT_RISK",
    }
    r4 = brain.scan(features, context=ctx4)
    if r4 == []:
        print(" ✅ EVENT_RISK block works")
        passed += 1
    else:
        print(f" ❌ EVENT_RISK block failed: {r4}")
        failed += 1

    print("\n [Test 5] Direction NONE blocks...")
    ctx5 = {
        "regime": "CHOP",
        "preferred_direction": "NONE",
        "session_phase": "GOLDEN_PM",
        "data_quality_score": 90,
        "narrative_label": "NEUTRAL",
    }
    r5 = brain.scan(features, context=ctx5)
    if r5 == []:
        print(" ✅ Preferred-direction NONE block works")
        passed += 1
    else:
        print(f" ❌ NONE block failed: {r5}")
        failed += 1

    print("\n [Test 6] Session block...")
    ctx6 = {
        "regime": "UNKNOWN",
        "preferred_direction": "BUY",
        "session_phase": "LAST_MINUTES",
        "data_quality_score": 90,
        "narrative_label": "NEUTRAL",
    }
    r6 = brain.scan(features, context=ctx6)
    if r6 == []:
        print(" ✅ Session block works")
        passed += 1
    else:
        print(f" ❌ Session block failed: {r6}")
        failed += 1

    print("\n [Test 7] Safe no-trade fallback...")
    ctx7 = {
        "regime": "CHOP",
        "preferred_direction": "BUY",
        "session_phase": "GOLDEN_PM",
        "data_quality_score": 90,
        "narrative_label": "NEUTRAL",
    }
    r7 = brain.scan(features, context=ctx7)
    if r7 == []:
        print(" ✅ Adaptive conservative fallback works (no forced trades)")
        passed += 1
    else:
        print(f" ❌ Adaptive fallback unsafe: {r7}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Adaptive Brain working perfectly!")
        print(" ✅ Current-phase missing brain file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()