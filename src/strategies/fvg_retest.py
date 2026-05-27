"""
Junior Aladdin - FVG Retest Strategy (Hardened Version)
=======================================================
PURPOSE:
Trade high-quality retests of institutional Fair Value Gaps (FVGs).

A Fair Value Gap represents an imbalance left by aggressive price movement.
When price returns into a valid, still-meaningful FVG and rejects, it can
offer a strong continuation/reversal opportunity depending on context.

This hardened version improves:
- FVG selection quality
- preference for higher-timeframe and unmitigated gaps
- stronger zone proximity and reclaim logic
- data quality / spread protection
- trend-hostility protection
- target realism and reward/risk sanity

BUY FVG RETEST CONDITIONS:
1. A bullish active FVG exists
2. FVG is UNMITIGATED or PARTIALLY_MITIGATED
3. Price is inside / near the FVG zone
4. Rejection wick confirms demand
5. RSI supportive but not exhausted
6. Volume meaningful
7. Regime not CHOP / EVENT
8. Session allows execution
9. Narrative not strongly against
10. ATR available
11. Spread / data quality acceptable
12. MTF not strongly bearish
13. Enough room to target

SELL CONDITIONS:
Mirror of BUY using bearish FVG.

CONNECTS TO:
- Smart Money Concepts module
- StrategyBase / Opportunity
- StrategyQuality helper layer
- Institutional brain
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class FVGRetestStrategy(StrategyBase):
    """
    Institutional strategy for high-quality fair value gap retests.
    """

    @property
    def name(self) -> str:
        return "FVG_RETEST"

    @property
    def brain(self) -> str:
        return "INSTITUTIONAL"

    def __init__(self):
        self._zone_tolerance_pct = 0.10
        self._rsi_buy_low = 32
        self._rsi_buy_high = 60
        self._rsi_sell_low = 40
        self._rsi_sell_high = 68
        self._sl_atr_mult = 0.3
        self._min_rr = 1.5
        self._min_volume_ratio = 0.7
        self._max_volume_ratio = 3.0
        self._wick_body_ratio = 1.2
        self._min_gap_size = 5.0
        super().__init__()
        self._logger = setup_logger("strategy_fvg_retest")

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

        buy_opp = self._check_buy_fvg(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell_fvg(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:2]

    def _extract_active_fvgs(self, ctx: Dict) -> List[Dict]:
        """
        Collect active FVGs from 5m and 15m smart money outputs.
        """
        fvgs: List[Dict] = []

        for source_name in ("smart_money_5m", "smart_money_15m"):
            sm = ctx.get(source_name, {})
            source_fvgs = sm.get("fvgs", [])

            if not isinstance(source_fvgs, list):
                continue

            for fvg in source_fvgs:
                if not isinstance(fvg, dict):
                    continue

                direction = fvg.get("direction", "NONE")
                status = fvg.get("status", "UNKNOWN")
                top = fvg.get("top", 0)
                bottom = fvg.get("bottom", 0)
                gap_size = fvg.get("gap_size", 0)

                if (
                    top and bottom
                    and top > 0
                    and bottom > 0
                    and top > bottom
                    and direction in ("BULLISH", "BEARISH")
                    and status in ("UNMITIGATED", "PARTIALLY_MITIGATED")
                    and float(gap_size or 0) >= self._min_gap_size
                ):
                    fvgs.append(
                        {
                            **fvg,
                            "source_tf": source_name,
                        }
                    )

        fvgs.sort(
            key=lambda x: (
                1 if x.get("source_tf") == "smart_money_15m" else 0,
                1 if x.get("status") == "UNMITIGATED" else 0,
                float(x.get("gap_size", 0) or 0),
                int(x.get("index", 0) or 0),
            ),
            reverse=True,
        )
        return fvgs

    def _is_price_near_zone(self, price: float, top: float, bottom: float) -> bool:
        if price <= 0 or top <= 0 or bottom <= 0 or top <= bottom:
            return False

        if bottom <= price <= top:
            return True

        midpoint = (top + bottom) / 2.0
        zone_ref = midpoint if midpoint > 0 else top
        distance_pct = abs(price - zone_ref) / zone_ref * 100
        return distance_pct <= self._zone_tolerance_pct

    def _check_buy_fvg(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
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
        last_swing_high = ctx.get("last_swing_high")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None

        active_fvgs = self._extract_active_fvgs(ctx)
        bullish_fvgs = [f for f in active_fvgs if f.get("direction") == "BULLISH"]
        if not bullish_fvgs:
            return None

        chosen = None
        for fvg in bullish_fvgs:
            if self._is_price_near_zone(close, float(fvg["top"]), float(fvg["bottom"])):
                chosen = fvg
                break

        if chosen is None:
            return None

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.2),
            "session_ok": StrategyQuality.get_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "bullish_fvg_exists": True,
            "near_fvg_zone": True,
            "rejection_wick": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="BUY",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "rsi_ok": (rsi is None) or (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._min_volume_ratio, self._max_volume_ratio
            ),
            "atr_ok": atr > 0,
            "mtf_not_strongly_bearish": weighted_mtf > -3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(float(chosen["bottom"]) - atr * self._sl_atr_mult - 1, 2)

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

        score = 55
        if chosen.get("status") == "UNMITIGATED":
            score += 10
        if chosen.get("source_tf") == "smart_money_15m":
            score += 10
        if float(chosen.get("gap_size", 0) or 0) >= 10:
            score += 5
        if volume_ratio is not None and volume_ratio >= 1.2:
            score += 5
        if rsi is not None and 38 <= rsi <= 50:
            score += 5
        if weighted_mtf > 0:
            score += 5
        if regime in ("TRENDING", "RANGE"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5

        thesis = (
            f"BUY FVG retest: bullish {chosen.get('source_tf')} FVG "
            f"{chosen['bottom']:.0f}-{chosen['top']:.0f}, "
            f"status={chosen.get('status')}, gap={chosen.get('gap_size', 0)}, "
            f"dataQ={data_quality}"
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

    def _check_sell_fvg(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
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
        last_swing_low = ctx.get("last_swing_low")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None

        active_fvgs = self._extract_active_fvgs(ctx)
        bearish_fvgs = [f for f in active_fvgs if f.get("direction") == "BEARISH"]
        if not bearish_fvgs:
            return None

        chosen = None
        for fvg in bearish_fvgs:
            if self._is_price_near_zone(close, float(fvg["top"]), float(fvg["bottom"])):
                chosen = fvg
                break

        if chosen is None:
            return None

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.2),
            "session_ok": StrategyQuality.get_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "bearish_fvg_exists": True,
            "near_fvg_zone": True,
            "rejection_wick": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="SELL",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "rsi_ok": (rsi is None) or (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._min_volume_ratio, self._max_volume_ratio
            ),
            "atr_ok": atr > 0,
            "mtf_not_strongly_bullish": weighted_mtf < 3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(float(chosen["top"]) + atr * self._sl_atr_mult + 1, 2)

        risk = abs(sl - entry)
        if risk <= 0:
            return None

        min_target = entry - risk * self._min_rr
        target = min_target

        if last_swing_low and last_swing_low < entry:
            target = min(min_target, max(last_swing_low, entry - risk * 3.0))

        target = round(target, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 55
        if chosen.get("status") == "UNMITIGATED":
            score += 10
        if chosen.get("source_tf") == "smart_money_15m":
            score += 10
        if float(chosen.get("gap_size", 0) or 0) >= 10:
            score += 5
        if volume_ratio is not None and volume_ratio >= 1.2:
            score += 5
        if rsi is not None and 50 <= rsi <= 62:
            score += 5
        if weighted_mtf < 0:
            score += 5
        if regime in ("TRENDING", "RANGE"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5

        thesis = (
            f"SELL FVG retest: bearish {chosen.get('source_tf')} FVG "
            f"{chosen['bottom']:.0f}-{chosen['top']:.0f}, "
            f"status={chosen.get('status')}, gap={chosen.get('gap_size', 0)}, "
            f"dataQ={data_quality}"
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


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — FVG Retest Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = FVGRetestStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "FVG_RETEST" and strategy.brain == "INSTITUTIONAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] BUY bullish FVG retest...")
    buy_f = {
        "last_close": 23205.0,
        "low": 23198.0,
        "high": 23210.0,
        "rsi": 44.0,
        "atr": 15.0,
        "volume_ratio": 1.4,
        "candle_body_ratio": 0.10,
        "lower_wick_ratio": 0.45,
        "upper_wick_ratio": 0.05,
    }
    buy_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "MILD_BULLISH",
        "weighted_mtf": 2.0,
        "last_swing_high": 23300,
        "smart_money_5m": {
            "fvgs": [
                {
                    "direction": "BULLISH",
                    "top": 23210,
                    "bottom": 23200,
                    "gap_size": 10,
                    "status": "UNMITIGATED",
                    "index": 10,
                }
            ]
        },
        "smart_money_15m": {},
        "microstructure": {"spread_zscore": 0.5},
        "data_quality_score": 85,
    }
    r2 = strategy.safe_scan(buy_f, context=buy_ctx)
    if len(r2) >= 1 and r2[0].direction == "BUY":
        print(f" ✅ BUY signal @{r2[0].entry_price}, RR={r2[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] No signal without FVG...")
    no_fvg_ctx = {
        **buy_ctx,
        "smart_money_5m": {"fvgs": []},
        "smart_money_15m": {},
    }
    r3 = strategy.safe_scan(buy_f, context=no_fvg_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (no FVG)")
        passed += 1
    else:
        print(" ❌ Should require FVG")
        failed += 1

    print("\n [Test 4] No signal far from FVG zone...")
    far_f = {**buy_f, "last_close": 23350.0}
    r4 = strategy.safe_scan(far_f, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (far from FVG)")
        passed += 1
    else:
        print(" ❌ Should block far-from-zone")
        failed += 1

    print("\n [Test 5] SELL bearish FVG retest...")
    sell_f = {
        "last_close": 23195.0,
        "high": 23202.0,
        "low": 23190.0,
        "rsi": 56.0,
        "atr": 15.0,
        "volume_ratio": 1.5,
        "candle_body_ratio": 0.10,
        "upper_wick_ratio": 0.45,
        "lower_wick_ratio": 0.05,
    }
    sell_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "MILD_BEARISH",
        "weighted_mtf": -2.0,
        "last_swing_low": 23100,
        "smart_money_5m": {
            "fvgs": [
                {
                    "direction": "BEARISH",
                    "top": 23200,
                    "bottom": 23190,
                    "gap_size": 10,
                    "status": "UNMITIGATED",
                    "index": 11,
                }
            ]
        },
        "smart_money_15m": {},
        "microstructure": {"spread_zscore": 0.4},
        "data_quality_score": 90,
    }
    r5 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r5) >= 1 and r5[0].direction == "SELL":
        print(f" ✅ SELL signal @{r5[0].entry_price}, RR={r5[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 6] No signal in CHOP...")
    chop_ctx = {**buy_ctx, "regime": "CHOP"}
    r6 = strategy.safe_scan(buy_f, context=chop_ctx)
    if len(r6) == 0:
        print(" ✅ No signal (CHOP blocked)")
        passed += 1
    else:
        print(" ❌ Should block CHOP")
        failed += 1

    print("\n [Test 7] No signal with EVENT_RISK...")
    event_ctx = {**buy_ctx, "narrative_label": "EVENT_RISK"}
    r7 = strategy.safe_scan(buy_f, context=event_ctx)
    if len(r7) == 0:
        print(" ✅ No signal (EVENT_RISK blocked)")
        passed += 1
    else:
        print(" ❌ Should block EVENT_RISK")
        failed += 1

    print("\n [Test 8] Empty data...")
    r8 = strategy.safe_scan({})
    if len(r8) == 0:
        print(" ✅ No crash with empty input")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 9] Stats tracking...")
    stats = strategy.get_stats()
    if stats["scan_count"] >= 7:
        print(f" ✅ Scans={stats['scan_count']}, Signals={stats['signal_count']}")
        passed += 1
    else:
        print(f" ❌ Stats: {stats}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 FVG Retest Strategy (Hardened) working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()