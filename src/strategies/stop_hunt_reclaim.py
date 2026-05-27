"""
Junior Aladdin - Stop Hunt Reclaim Strategy (Hardened Version)
==============================================================
PURPOSE:
Trade reversals after institutional stop hunts.

A stop hunt occurs when price deliberately sweeps beyond an obvious swing level,
triggers clustered stops, and then sharply reclaims back across that level.
This is one of the highest-edge tactical concepts because it aligns with
institutional accumulation/distribution after weak-hand liquidation.

This hardened version improves:
- true sweep validation
- reclaim quality checks
- stronger microstructure consistency
- better anti-noise logic
- stronger target realism
- safer spread and data-quality filters

BUY STOP HUNT CONDITIONS:
1. A swing support level exists
2. Price sweeps below it by a meaningful amount
3. Price reclaims back above it with quality
4. Rejection wick confirms defense
5. Volume is strong enough to imply real participation
6. Regime not CHOP / EVENT
7. Session allows tactical execution
8. Narrative not strongly against
9. ATR available
10. Spread/data quality acceptable
11. RSI not exhausted beyond usefulness
12. Optional microstructure stop-hunt confirmation improves quality

SELL CONDITIONS:
Mirror of BUY above swing high.

CONNECTS TO:
- Microstructure features
- Smart Money / swing structure
- StrategyBase / Opportunity
- StrategyQuality helper layer
- Tactical brain
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class StopHuntReclaimStrategy(StrategyBase):
    """
    Hardened tactical strategy for institutional stop sweeps.
    """

    @property
    def name(self) -> str:
        return "STOP_HUNT_RECLAIM"

    @property
    def brain(self) -> str:
        return "TACTICAL"

    def __init__(self):
        self._min_pierce_points = 3.0
        self._min_volume_ratio = 1.5
        self._max_volume_ratio = 4.0
        self._wick_body_ratio = 2.0
        self._sl_atr_mult = 0.1
        self._min_rr = 2.0

        self._rsi_buy_low = 30
        self._rsi_buy_high = 55
        self._rsi_sell_low = 45
        self._rsi_sell_high = 70

        super().__init__()
        self._logger = setup_logger("strategy_stop_hunt_reclaim")

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Opportunity]:
        if not features_1m or not context:
            return []

        opportunities: List[Opportunity] = []

        buy_opp = self._check_buy_hunt(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell_hunt(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:2]

    def _check_buy_hunt(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        low = self._safe_get(f1m, "low", 0)
        high = self._safe_get(f1m, "high", 0)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_high = ctx.get("last_swing_high")
        last_swing_low = ctx.get("last_swing_low")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        micro = ctx.get("microstructure", {})
        stop_hunt_detected = micro.get("stop_hunt_detected", False)
        stop_hunt_type = micro.get("stop_hunt_type", "NONE")
        stop_hunt_level = micro.get("stop_hunt_level", 0)

        if close <= 0 or atr is None or atr <= 0:
            return None

        swing_low = stop_hunt_level if stop_hunt_level and stop_hunt_level > 0 else (last_swing_low or 0)
        if swing_low <= 0:
            return None

        pierce_points = round(swing_low - low, 2) if low > 0 else 0.0
        hunt_from_micro = stop_hunt_detected and stop_hunt_type == "BUY_HUNT"
        hunt_from_candle = (
            low > 0
            and low < swing_low
            and pierce_points >= self._min_pierce_points
            and close > swing_low
        )

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "hunt_detected": hunt_from_micro or hunt_from_candle,
            "pierce_size_ok": pierce_points >= self._min_pierce_points,
            "reclaimed": close > swing_low,
            "reclaim_quality": self._reclaim_quality_buy(
                close=close,
                level=swing_low,
                high=high,
                low=low,
            ),
            "wick_rejection": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="BUY",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio,
                self._min_volume_ratio,
                self._max_volume_ratio,
            ),
            "rsi_ok": (rsi is None) or (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "atr_ok": atr > 0,
            "mtf_not_hostile": weighted_mtf > -3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(low - atr * self._sl_atr_mult - 1, 2)
        risk = abs(entry - sl)
        if risk <= 0:
            return None

        min_target = entry + risk * self._min_rr
        target = min_target

        if last_swing_high and last_swing_high > entry:
            target = max(min_target, min(last_swing_high, entry + risk * 3.0))

        target = round(target, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 60
        if hunt_from_micro:
            score += 10
        if volume_ratio is not None and volume_ratio >= 2.0:
            score += 5
        if rsi is not None and 35 <= rsi <= 45:
            score += 5
        if weighted_mtf > 0:
            score += 5
        if narrative in ("MILD_BULLISH", "STRONG_BULLISH"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5

        thesis = (
            f"Bullish stop hunt reclaim: swept {pierce_points:.1f}pts below swing low "
            f"{swing_low:.2f}, reclaimed above, volume={volume_ratio}x, dataQ={data_quality}"
        )

        return Opportunity(
            strategy=self.name,
            direction="BUY",
            entry_price=entry,
            sl_price=sl,
            target_price=target,
            raw_score=score,
            thesis=thesis,
            timeframe="1min",
            brain=self.brain,
            conditions_met=conditions,
        )

    def _check_sell_hunt(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        low = self._safe_get(f1m, "low", 0)
        high = self._safe_get(f1m, "high", 0)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_low = ctx.get("last_swing_low")
        last_swing_high = ctx.get("last_swing_high")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        micro = ctx.get("microstructure", {})
        stop_hunt_detected = micro.get("stop_hunt_detected", False)
        stop_hunt_type = micro.get("stop_hunt_type", "NONE")
        stop_hunt_level = micro.get("stop_hunt_level", 0)

        if close <= 0 or atr is None or atr <= 0:
            return None

        swing_high = stop_hunt_level if stop_hunt_level and stop_hunt_level > 0 else (last_swing_high or 0)
        if swing_high <= 0:
            return None

        pierce_points = round(high - swing_high, 2) if high > 0 else 0.0
        hunt_from_micro = stop_hunt_detected and stop_hunt_type == "SELL_HUNT"
        hunt_from_candle = (
            high > 0
            and high > swing_high
            and pierce_points >= self._min_pierce_points
            and close < swing_high
        )

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "hunt_detected": hunt_from_micro or hunt_from_candle,
            "pierce_size_ok": pierce_points >= self._min_pierce_points,
            "reclaimed": close < swing_high,
            "reclaim_quality": self._reclaim_quality_sell(
                close=close,
                level=swing_high,
                high=high,
                low=low,
            ),
            "wick_rejection": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="SELL",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio,
                self._min_volume_ratio,
                self._max_volume_ratio,
            ),
            "rsi_ok": (rsi is None) or (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "atr_ok": atr > 0,
            "mtf_not_hostile": weighted_mtf < 3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(high + atr * self._sl_atr_mult + 1, 2)
        risk = abs(entry - sl)
        if risk <= 0:
            return None

        min_target = entry - risk * self._min_rr
        target = min_target

        if last_swing_low and last_swing_low < entry:
            target = min(min_target, max(last_swing_low, entry - risk * 3.0))

        target = round(target, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 60
        if hunt_from_micro:
            score += 10
        if volume_ratio is not None and volume_ratio >= 2.0:
            score += 5
        if rsi is not None and 55 <= rsi <= 65:
            score += 5
        if weighted_mtf < 0:
            score += 5
        if narrative in ("MILD_BEARISH", "STRONG_BEARISH"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5

        thesis = (
            f"Bearish stop hunt reclaim: swept {pierce_points:.1f}pts above swing high "
            f"{swing_high:.2f}, reclaimed below, volume={volume_ratio}x, dataQ={data_quality}"
        )

        return Opportunity(
            strategy=self.name,
            direction="SELL",
            entry_price=entry,
            sl_price=sl,
            target_price=target,
            raw_score=score,
            thesis=thesis,
            timeframe="1min",
            brain=self.brain,
            conditions_met=conditions,
        )

    def _reclaim_quality_buy(self, close: float, level: float, high: float, low: float) -> bool:
        rng = max(0.0, high - low)
        if rng <= 0 or level <= 0:
            return False
        return close > level and ((close - level) / rng) >= 0.10

    def _reclaim_quality_sell(self, close: float, level: float, high: float, low: float) -> bool:
        rng = max(0.0, high - low)
        if rng <= 0 or level <= 0:
            return False
        return close < level and ((level - close) / rng) >= 0.10


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Stop Hunt Reclaim Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = StopHuntReclaimStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "STOP_HUNT_RECLAIM" and strategy.brain == "TACTICAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] Perfect BUY stop hunt...")
    buy_f = {
        "last_close": 23055.0,
        "low": 23038.0,
        "high": 23060.0,
        "rsi": 40.0,
        "atr": 15.0,
        "volume_ratio": 2.0,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.60,
        "upper_wick_ratio": 0.05,
    }
    buy_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "weighted_mtf": 1.0,
        "last_swing_high": 23200,
        "last_swing_low": 23050,
        "microstructure": {
            "stop_hunt_detected": True,
            "stop_hunt_type": "BUY_HUNT",
            "stop_hunt_level": 23050,
            "spread_zscore": 0.5,
        },
        "data_quality_score": 85,
    }
    r2 = strategy.safe_scan(buy_f, context=buy_ctx)
    if len(r2) >= 1 and r2[0].direction == "BUY":
        print(f" ✅ BUY signal @{r2[0].entry_price}, RR={r2[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] BUY from candle-detection only...")
    no_micro_ctx = {**buy_ctx, "microstructure": {"spread_zscore": 0.5}}
    r3 = strategy.safe_scan(buy_f, context=no_micro_ctx)
    if len(r3) >= 1 and r3[0].direction == "BUY":
        print(" ✅ Candle-based detection works")
        passed += 1
    else:
        print(" ❌ Should detect from candle sweep/reclaim")
        failed += 1

    print("\n [Test 4] No signal without sufficient pierce...")
    no_pierce = {**buy_f, "low": 23049.0}
    r4 = strategy.safe_scan(no_pierce, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (pierce too small)")
        passed += 1
    else:
        print(" ❌ Should require >=3pt pierce")
        failed += 1

    print("\n [Test 5] No signal without reclaim...")
    no_reclaim = {**buy_f, "last_close": 23045.0}
    r5 = strategy.safe_scan(no_reclaim, context=buy_ctx)
    if len(r5) == 0:
        print(" ✅ No signal (not reclaimed)")
        passed += 1
    else:
        print(" ❌ Must reclaim above level")
        failed += 1

    print("\n [Test 6] No signal with low volume...")
    low_vol = {**buy_f, "volume_ratio": 0.8}
    r6 = strategy.safe_scan(low_vol, context=buy_ctx)
    if len(r6) == 0:
        print(" ✅ No signal (low volume)")
        passed += 1
    else:
        print(" ❌ Should block low volume")
        failed += 1

    print("\n [Test 7] No signal in CHOP...")
    chop_ctx = {**buy_ctx, "regime": "CHOP"}
    r7 = strategy.safe_scan(buy_f, context=chop_ctx)
    if len(r7) == 0:
        print(" ✅ No signal (CHOP blocked)")
        passed += 1
    else:
        print(" ❌ Should block CHOP")
        failed += 1

    print("\n [Test 8] No signal with EVENT_RISK...")
    event_ctx = {**buy_ctx, "narrative_label": "EVENT_RISK"}
    r8 = strategy.safe_scan(buy_f, context=event_ctx)
    if len(r8) == 0:
        print(" ✅ No signal (EVENT_RISK blocked)")
        passed += 1
    else:
        print(" ❌ Should block EVENT_RISK")
        failed += 1

    print("\n [Test 9] SELL stop hunt...")
    sell_f = {
        "last_close": 23195.0,
        "high": 23215.0,
        "low": 23190.0,
        "rsi": 60.0,
        "atr": 15.0,
        "volume_ratio": 1.8,
        "candle_body_ratio": 0.10,
        "upper_wick_ratio": 0.55,
        "lower_wick_ratio": 0.05,
    }
    sell_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "NEUTRAL",
        "weighted_mtf": -1.0,
        "last_swing_low": 23100,
        "last_swing_high": 23200,
        "microstructure": {
            "stop_hunt_detected": True,
            "stop_hunt_type": "SELL_HUNT",
            "stop_hunt_level": 23200,
            "spread_zscore": 0.4,
        },
        "data_quality_score": 90,
    }
    r9 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r9) >= 1 and r9[0].direction == "SELL":
        print(f" ✅ SELL signal @{r9[0].entry_price}, RR={r9[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 10] No signal without swing levels...")
    no_swing_ctx = {**buy_ctx, "last_swing_low": None, "microstructure": {"spread_zscore": 0.5}}
    r10 = strategy.safe_scan(buy_f, context=no_swing_ctx)
    if len(r10) == 0:
        print(" ✅ No signal (no swing level to hunt)")
        passed += 1
    else:
        print(" ❌ Should require swing level")
        failed += 1

    print("\n [Test 11] Empty data...")
    r11 = strategy.safe_scan({})
    if len(r11) == 0:
        print(" ✅ No crash with empty input")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 12] Stats tracking...")
    stats = strategy.get_stats()
    if stats["scan_count"] >= 8:
        print(f" ✅ Scans={stats['scan_count']}, Signals={stats['signal_count']}")
        passed += 1
    else:
        print(f" ❌ Stats: {stats}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Stop Hunt Reclaim Strategy (Hardened) working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()