"""
Junior Aladdin - Institutional Brain
====================================
PURPOSE:
Coordinate all INSTITUTIONAL-brain strategies and return only the strongest
institutional-quality opportunities.

INSTITUTIONAL STRATEGIES OWNED:
- OI Wall Bounce
- FVG Retest

RESPONSIBILITIES:
- run only when institutional context is meaningful
- obey preferred direction from Brain Engine
- gather opportunities from institutional strategies
- rank and return strongest signals only
- fail safely on missing or weak inputs

CONNECTS TO:
- brain_base.py
- institutional strategies
- captain / scoring / trap pipeline
"""

from typing import Dict, List, Optional

from src.utils.logger import setup_logger
from src.strategies.oi_wall_bounce import OIWallBounceStrategy
from src.strategies.fvg_retest import FVGRetestStrategy

_logger = setup_logger("institutional_brain")


class InstitutionalBrain:
    """
    Runs all institutional strategies and returns strongest institutional opportunities.
    """

    def __init__(self):
        self._logger = _logger

        self._strategies = [
            OIWallBounceStrategy(),
            FVGRetestStrategy(),
        ]

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Scan institutional strategies and return ranked opportunities.
        """
        if not features_1m or not context:
            return []

        if not self._session_ok(context):
            return []

        if not self._regime_ok(context):
            return []

        if not self._institutional_context_ok(context):
            return []

        preferred_direction = str(context.get("preferred_direction", "BOTH")).upper()

        opportunities: List[Dict] = []

        for strategy in self._strategies:
            try:
                results = strategy.safe_scan(
                    features_1m=features_1m,
                    features_5m=features_5m,
                    features_15m=features_15m,
                    context=context,
                )

                for opp in results:
                    opp_dict = opp.to_dict() if hasattr(opp, "to_dict") else opp

                    if preferred_direction == "BUY" and opp_dict.get("direction") != "BUY":
                        continue
                    if preferred_direction == "SELL" and opp_dict.get("direction") != "SELL":
                        continue
                    if preferred_direction == "NONE":
                        continue

                    opportunities.append(opp_dict)

            except Exception as e:
                self._logger.error(
                    "Institutional strategy scan failure",
                    extra={
                        "strategy": getattr(strategy, "name", "UNKNOWN"),
                        "error": str(e),
                    },
                )

        ranked = self._rank_opportunities(opportunities)
        self._logger.info(
            "Institutional brain scan complete",
            extra={
                "opportunity_count": len(ranked),
                "preferred_direction": preferred_direction,
                "strategies_run": len(self._strategies),
            },
        )
        return ranked

    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank institutional opportunities.
        Institutional setups should be selective and high-conviction.
        """
        if not opportunities:
            return []

        opportunities.sort(
            key=lambda x: (
                float(x.get("raw_score", 0) or 0),
                float(x.get("risk_reward", 0) or 0),
            ),
            reverse=True,
        )

        return opportunities[:2]

    def _session_ok(self, context: Dict) -> bool:
        session = str(context.get("session_phase", "UNKNOWN"))
        return session in (
            "INITIAL_BALANCE",
            "GOLDEN_AM",
            "GOLDEN_PM",
            "CLOSING_SESSION",
        )

    def _regime_ok(self, context: Dict) -> bool:
        regime = str(context.get("regime", "UNKNOWN"))
        return regime not in ("CHOP", "EVENT")

    def _institutional_context_ok(self, context: Dict) -> bool:
        """
        Require some institutional evidence to justify using this brain.
        """
        options = context.get("options", {}) or {}
        sm5 = context.get("smart_money_5m", {}) or {}
        sm15 = context.get("smart_money_15m", {}) or {}

        ce_wall = options.get("highest_ce_oi_strike", 0)
        pe_wall = options.get("highest_pe_oi_strike", 0)
        sm_score_5 = sm5.get("sm_direction_score", 0)
        sm_score_15 = sm15.get("sm_direction_score", 0)
        total_fvgs_5 = sm5.get("total_fvgs", 0)
        total_fvgs_15 = sm15.get("total_fvgs", 0)

        if ce_wall or pe_wall:
            return True

        if abs(float(sm_score_5 or 0)) >= 20:
            return True

        if abs(float(sm_score_15 or 0)) >= 20:
            return True

        if int(total_fvgs_5 or 0) > 0 or int(total_fvgs_15 or 0) > 0:
            return True

        return False

    def get_strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]

    def get_status(self) -> Dict:
        return {
            "brain": "INSTITUTIONAL",
            "strategy_count": len(self._strategies),
            "strategies": self.get_strategy_names(),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Institutional Brain Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    brain = InstitutionalBrain()

    print(" [Test 1] Create brain...")
    st1 = brain.get_status()
    if st1["brain"] == "INSTITUTIONAL" and st1["strategy_count"] == 2:
        print(f" ✅ Status OK: {st1}")
        passed += 1
    else:
        print(f" ❌ Bad status: {st1}")
        failed += 1

    print("\n [Test 2] Strategy names...")
    names = set(brain.get_strategy_names())
    expected = {
        "OI_WALL_BOUNCE",
        "FVG_RETEST",
    }
    if names == expected:
        print(f" ✅ Institutional strategies registered: {names}")
        passed += 1
    else:
        print(f" ❌ Strategy registry mismatch: {names}")
        failed += 1

    features_1m = {
        "last_close": 23050.5,
        "high": 23058.0,
        "low": 23045.0,
        "rsi": 38.0,
        "atr": 15.0,
        "volume_ratio": 1.2,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.55,
        "upper_wick_ratio": 0.05,
    }

    context = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "preferred_direction": "BUY",
        "weighted_mtf": 1.0,
        "data_quality_score": 85,
        "microstructure": {"spread_zscore": 0.5},
        "options": {
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 8000000,
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 6000000,
            "pcr_oi": 1.1,
        },
        "volume_profile": {
            "poc": 23200,
            "poc_volume": 50000,
            "vah": 23280,
            "val": 23120,
        },
        "smart_money_5m": {
            "sm_direction_score": 30,
            "fvgs": [
                {
                    "direction": "BULLISH",
                    "top": 23060,
                    "bottom": 23050,
                    "gap_size": 10,
                    "status": "UNMITIGATED",
                    "index": 10,
                }
            ],
        },
        "smart_money_15m": {},
        "last_swing_high": 23250,
        "last_swing_low": 23000,
    }

    print("\n [Test 3] Valid institutional scan...")
    r3 = brain.scan(features_1m=features_1m, context=context)
    if isinstance(r3, list):
        print(f" ✅ Scan returned list, count={len(r3)}")
        passed += 1
    else:
        print(f" ❌ Scan did not return list: {r3}")
        failed += 1

    print("\n [Test 4] Preferred direction filter...")
    if all(x.get("direction") == "BUY" for x in r3):
        print(" ✅ Preferred direction filter works")
        passed += 1
    else:
        print(f" ❌ Unexpected direction mix: {r3}")
        failed += 1

    print("\n [Test 5] Missing institutional context blocks...")
    bad_ctx5 = {
        **context,
        "options": {},
        "smart_money_5m": {},
        "smart_money_15m": {},
    }
    r5 = brain.scan(features_1m=features_1m, context=bad_ctx5)
    if r5 == []:
        print(" ✅ Missing institutional context block works")
        passed += 1
    else:
        print(f" ❌ Institutional context block failed: {r5}")
        failed += 1

    print("\n [Test 6] Session block...")
    bad_ctx6 = {**context, "session_phase": "LUNCH_LULL"}
    r6 = brain.scan(features_1m=features_1m, context=bad_ctx6)
    if r6 == []:
        print(" ✅ Session block works")
        passed += 1
    else:
        print(f" ❌ Session block failed: {r6}")
        failed += 1

    print("\n [Test 7] Regime block...")
    bad_ctx7 = {**context, "regime": "CHOP"}
    r7 = brain.scan(features_1m=features_1m, context=bad_ctx7)
    if r7 == []:
        print(" ✅ Regime block works")
        passed += 1
    else:
        print(f" ❌ Regime block failed: {r7}")
        failed += 1

    print("\n [Test 8] Direction NONE block...")
    bad_ctx8 = {**context, "preferred_direction": "NONE"}
    r8 = brain.scan(features_1m=features_1m, context=bad_ctx8)
    if r8 == []:
        print(" ✅ Direction NONE block works")
        passed += 1
    else:
        print(f" ❌ Direction NONE block failed: {r8}")
        failed += 1

    print("\n [Test 9] Empty input safe...")
    r9 = brain.scan({}, context={})
    if r9 == []:
        print(" ✅ Empty input handled safely")
        passed += 1
    else:
        print(f" ❌ Empty input handling failed: {r9}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Institutional Brain working perfectly!")
        print(" ✅ Current-phase missing brain file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()