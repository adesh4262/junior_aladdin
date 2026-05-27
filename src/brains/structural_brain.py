"""
Junior Aladdin - Structural Brain
=================================
PURPOSE:
Coordinate all STRUCTURAL-brain strategies and return the best
high-quality structural opportunities.

This file exists because the roadmap expects:
    src/brains/structural_brain.py

STRUCTURAL STRATEGIES OWNED:
- VWAP Pullback
- Trend Continuation
- Opening Range Breakout
- S/R Rejection
- Volume Profile POC Reversal

RESPONSIBILITIES:
- run only in structurally appropriate contexts
- obey preferred direction from Brain Engine
- gather opportunities from structural strategies
- rank and return only strongest signals
- fail safely if context or features are incomplete

CONNECTS TO:
- brain_base.py
- structural strategies
- captain / scoring / trap pipeline
"""

from typing import Dict, List, Optional

from src.utils.logger import setup_logger
from src.strategies.vwap_pullback import VWAPPullbackStrategy
from src.strategies.trend_continuation import TrendContinuationStrategy
from src.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from src.strategies.sr_rejection import SRRejectionStrategy
from src.strategies.vol_profile_poc import VolumeProfilePOCStrategy

_logger = setup_logger("structural_brain")


class StructuralBrain:
    """
    Runs all structural strategies and returns strongest structural opportunities.
    """

    def __init__(self):
        self._logger = _logger

        self._strategies = [
            VWAPPullbackStrategy(),
            TrendContinuationStrategy(),
            OpeningRangeBreakoutStrategy(),
            SRRejectionStrategy(),
            VolumeProfilePOCStrategy(),
        ]

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Scan all structural strategies and return ranked opportunities.
        """
        if not features_1m or not context:
            return []

        if not self._session_ok(context):
            return []

        if not self._regime_ok(context):
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
                    "Structural strategy scan failure",
                    extra={
                        "strategy": getattr(strategy, "name", "UNKNOWN"),
                        "error": str(e),
                    },
                )

        ranked = self._rank_opportunities(opportunities)
        self._logger.info(
            "Structural brain scan complete",
            extra={
                "opportunity_count": len(ranked),
                "preferred_direction": preferred_direction,
                "strategies_run": len(self._strategies),
            },
        )
        return ranked

    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank structural opportunities by quality.
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

        # keep only strongest 3 to avoid clutter
        return opportunities[:3]

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

    def get_strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]

    def get_status(self) -> Dict:
        return {
            "brain": "STRUCTURAL",
            "strategy_count": len(self._strategies),
            "strategies": self.get_strategy_names(),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Structural Brain Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    brain = StructuralBrain()

    print(" [Test 1] Create brain...")
    st1 = brain.get_status()
    if st1["brain"] == "STRUCTURAL" and st1["strategy_count"] == 5:
        print(f" ✅ Status OK: {st1}")
        passed += 1
    else:
        print(f" ❌ Bad status: {st1}")
        failed += 1

    print("\n [Test 2] Strategy names...")
    names = set(brain.get_strategy_names())
    expected = {
        "VWAP_PULLBACK",
        "TREND_CONTINUATION",
        "OPENING_RANGE_BREAKOUT",
        "SR_REJECTION",
        "VOL_PROFILE_POC",
    }
    if names == expected:
        print(f" ✅ All structural strategies registered: {names}")
        passed += 1
    else:
        print(f" ❌ Strategy registry mismatch: {names}")
        failed += 1

    # Shared bullish test context
    features_1m = {
        "last_close": 23200.0,
        "high": 23208.0,
        "low": 23192.0,
        "vwap": 23198.0,
        "ema_9": 23201.0,
        "ema_21": 23197.0,
        "ema_50": 23170.0,
        "rsi": 45.0,
        "atr": 12.0,
        "volume_ratio": 0.85,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.40,
        "upper_wick_ratio": 0.05,
        "trend_direction": 1,
        "supertrend_direction": 1,
        "price_vs_vwap_pct": 0.01,
        "macd_histogram": 1.0,
        "macd_hist_slope": 0.8,
    }

    context = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "MILD_BULLISH",
        "weighted_mtf": 5.5,
        "preferred_direction": "BUY",
        "data_quality_score": 85,
        "microstructure": {"spread_zscore": 0.5},
        "last_swing_high": 23260.0,
        "last_swing_low": 23150.0,
        "key_levels": {
            "pdh": 23300,
            "pdl": 23050,
            "or_high": 23200,
            "or_low": 23100,
            "ib_high": 23250,
            "ib_low": 23080,
            "sr_zones": [
                {"level": 23050, "strength": 3, "type": "support"},
                {"level": 23300, "strength": 2, "type": "resistance"},
            ],
        },
        "volume_profile": {
            "poc": 23200,
            "poc_volume": 50000,
            "vah": 23280,
            "val": 23120,
            "va_width": 160,
        },
        "options": {
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 8000000,
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 6000000,
            "pcr_oi": 1.1,
        },
        "smart_money_5m": {"sm_direction_score": 20, "buy_side_pools": [], "sell_side_pools": []},
        "smart_money_15m": {},
        "session_memory": {"levels_defended": [23050], "failed_breakouts": []},
    }

    print("\n [Test 3] Valid structural scan...")
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

    print("\n [Test 5] Session block...")
    bad_ctx5 = {**context, "session_phase": "LUNCH_LULL"}
    r5 = brain.scan(features_1m=features_1m, context=bad_ctx5)
    if r5 == []:
        print(" ✅ Session block works")
        passed += 1
    else:
        print(f" ❌ Session block failed: {r5}")
        failed += 1

    print("\n [Test 6] Regime block...")
    bad_ctx6 = {**context, "regime": "CHOP"}
    r6 = brain.scan(features_1m=features_1m, context=bad_ctx6)
    if r6 == []:
        print(" ✅ Regime block works")
        passed += 1
    else:
        print(f" ❌ Regime block failed: {r6}")
        failed += 1

    print("\n [Test 7] Direction NONE block...")
    bad_ctx7 = {**context, "preferred_direction": "NONE"}
    r7 = brain.scan(features_1m=features_1m, context=bad_ctx7)
    if r7 == []:
        print(" ✅ Preferred-direction NONE block works")
        passed += 1
    else:
        print(f" ❌ Direction NONE block failed: {r7}")
        failed += 1

    print("\n [Test 8] Empty input safe...")
    r8 = brain.scan({}, context={})
    if r8 == []:
        print(" ✅ Empty input handled safely")
        passed += 1
    else:
        print(f" ❌ Empty input handling failed: {r8}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Structural Brain working perfectly!")
        print(" ✅ Current-phase missing brain file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()