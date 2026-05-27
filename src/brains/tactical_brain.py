"""
Junior Aladdin - Tactical Brain
===============================
PURPOSE:
Coordinate all TACTICAL-brain strategies and return only the best
short-horizon, high-velocity opportunities.

TACTICAL STRATEGIES OWNED:
- Stop Hunt Reclaim
- ATM Momentum Burst
- Failed Breakout Reversal
- Absorption Reversal
- Liquidity Sweep Reversal

RESPONSIBILITIES:
- run only in tactical-appropriate contexts
- obey preferred direction from Brain Engine
- gather opportunities from tactical strategies
- rank and return strongest opportunities only
- fail safely on missing or weak inputs

CONNECTS TO:
- brain_base.py
- tactical strategies
- captain / scoring / trap pipeline
"""

from typing import Dict, List, Optional

from src.utils.logger import setup_logger
from src.strategies.stop_hunt_reclaim import StopHuntReclaimStrategy
from src.strategies.atm_momentum import ATMMomentumBurstStrategy
from src.strategies.failed_breakout import FailedBreakoutReversalStrategy
from src.strategies.absorption_reversal import AbsorptionReversalStrategy
from src.strategies.liquidity_sweep import LiquiditySweepReversalStrategy

_logger = setup_logger("tactical_brain")


class TacticalBrain:
    """
    Runs all tactical strategies and returns strongest tactical opportunities.
    """

    def __init__(self):
        self._logger = _logger

        self._strategies = [
            StopHuntReclaimStrategy(),
            ATMMomentumBurstStrategy(),
            FailedBreakoutReversalStrategy(),
            AbsorptionReversalStrategy(),
            LiquiditySweepReversalStrategy(),
        ]

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Scan all tactical strategies and return ranked opportunities.
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
                    "Tactical strategy scan failure",
                    extra={
                        "strategy": getattr(strategy, "name", "UNKNOWN"),
                        "error": str(e),
                    },
                )

        ranked = self._rank_opportunities(opportunities)
        self._logger.info(
            "Tactical brain scan complete",
            extra={
                "opportunity_count": len(ranked),
                "preferred_direction": preferred_direction,
                "strategies_run": len(self._strategies),
            },
        )
        return ranked

    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank tactical opportunities by aggressiveness-adjusted quality.
        Tactical strategies should prefer:
        - higher score
        - strong RR
        - more selective outputs
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

        # Tactical brain should remain selective
        return opportunities[:3]

    def _session_ok(self, context: Dict) -> bool:
        """
        Tactical session gate.
        Tactical strategies can run in:
        - INITIAL_BALANCE
        - GOLDEN_AM
        - GOLDEN_PM
        They should not run in lunch / last minutes / premarket.
        """
        session = str(context.get("session_phase", "UNKNOWN"))
        return session in (
            "INITIAL_BALANCE",
            "GOLDEN_AM",
            "GOLDEN_PM",
        )

    def _regime_ok(self, context: Dict) -> bool:
        """
        Tactical strategies should not run in EVENT or CHOP by default.
        """
        regime = str(context.get("regime", "UNKNOWN"))
        return regime not in ("CHOP", "EVENT")

    def get_strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]

    def get_status(self) -> Dict:
        return {
            "brain": "TACTICAL",
            "strategy_count": len(self._strategies),
            "strategies": self.get_strategy_names(),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Tactical Brain Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    brain = TacticalBrain()

    print(" [Test 1] Create brain...")
    st1 = brain.get_status()
    if st1["brain"] == "TACTICAL" and st1["strategy_count"] == 5:
        print(f" ✅ Status OK: {st1}")
        passed += 1
    else:
        print(f" ❌ Bad status: {st1}")
        failed += 1

    print("\n [Test 2] Strategy names...")
    names = set(brain.get_strategy_names())
    expected = {
        "STOP_HUNT_RECLAIM",
        "ATM_MOMENTUM_BURST",
        "FAILED_BREAKOUT_REVERSAL",
        "ABSORPTION_REVERSAL",
        "LIQUIDITY_SWEEP_REVERSAL",
    }
    if names == expected:
        print(f" ✅ All tactical strategies registered: {names}")
        passed += 1
    else:
        print(f" ❌ Strategy registry mismatch: {names}")
        failed += 1

    # shared BUY tactical context
    features_1m = {
        "last_close": 23055.0,
        "high": 23060.0,
        "low": 23038.0,
        "rsi": 40.0,
        "rsi_slope": 3.0,
        "atr": 15.0,
        "volume_ratio": 1.8,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.60,
        "upper_wick_ratio": 0.05,
        "price_acceleration": 3.0,
        "roc_5": 0.08,
        "macd_histogram": 2.0,
        "macd_hist_slope": 1.2,
        "supertrend_direction": 1,
        "trend_direction": 0,
        "price_vs_vwap_pct": 0.2,
    }

    context = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "preferred_direction": "BUY",
        "weighted_mtf": 1.0,
        "data_quality_score": 85,
        "last_swing_high": 23200,
        "last_swing_low": 23050,
        "key_levels": {
            "pdh": 23300,
            "pdl": 23050,
            "sr_zones": [
                {"level": 23050, "strength": 3, "type": "support"},
                {"level": 23300, "strength": 2, "type": "resistance"},
            ],
        },
        "volume_profile": {"val": 23050, "vah": 23250, "poc": 23150, "poc_volume": 50000},
        "options": {
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 8000000,
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 6000000,
            "pcr_oi": 1.1,
        },
        "session_memory": {"levels_defended": [23050], "failed_breakouts": []},
        "microstructure": {
            "stop_hunt_detected": True,
            "stop_hunt_type": "BUY_HUNT",
            "stop_hunt_level": 23050,
            "absorption_detected": True,
            "absorption_direction": "BULLISH",
            "spread_zscore": 0.5,
        },
        "smart_money_5m": {
            "buy_side_pools": [{"level": 23050, "touches": 3}],
            "sell_side_pools": [],
            "last_choch_direction": "BULLISH",
        },
        "smart_money_15m": {},
    }

    print("\n [Test 3] Valid tactical scan...")
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
        print("\n 🎉 Tactical Brain working perfectly!")
        print(" ✅ Current-phase missing brain file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()