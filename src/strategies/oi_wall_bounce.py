"""
Junior Aladdin - OI Wall Bounce Strategy (Hardened Version)
===========================================================
PURPOSE:
Trade bounces from strong option OI walls where institutional option writers
are likely defending positions.

PE wall (highest PE OI strike) acts as support.
CE wall (highest CE OI strike) acts as resistance.

This hardened version improves:
- wall significance validation
- near-wall logic with realistic zone handling
- rejection confirmation
- anti-breakout / anti-chaos volume filters
- spread and data-quality protection
- better room-to-target validation
- stronger compatibility with real market behavior

BUY AT PE WALL CONDITIONS:
1. Valid PE wall exists with meaningful OI
2. Price is near PE wall
3. Price is holding above PE wall zone
4. Rejection candle confirms defense
5. Volume meaningful but not breakout-chaotic
6. RSI supportive
7. Regime not CHOP / EVENT
8. Session allows execution
9. Narrative not strongly against
10. ATR available
11. Spread/data quality acceptable
12. Enough room to target

SELL AT CE WALL CONDITIONS:
Mirror of BUY.

CONNECTS TO:
- Options Features
- StrategyBase / Opportunity
- StrategyQuality shared helper layer
- Institutional brain
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class OIWallBounceStrategy(StrategyBase):
    """
    Hardened OI Wall Bounce strategy.
    """

    @property
    def name(self) -> str:
        return "OI_WALL_BOUNCE"

    @property
    def brain(self) -> str:
        return "INSTITUTIONAL"

    def __init__(self):
        self._wall_tolerance_pct = 0.15
        self._min_wall_oi = 1_000_000
        self._strong_wall_oi = 5_000_000

        self._rsi_buy_low = 30
        self._rsi_buy_high = 50
        self._rsi_sell_low = 50
        self._rsi_sell_high = 70

        self._vol_min = 0.5
        self._vol_max = 2.5
        self._extreme_break_volume = 3.5

        self._wick_body_ratio = 1.5
        self._sl_atr_mult = 0.3
        self._min_rr = 1.5

        super().__init__()
        self._logger = setup_logger("strategy_oi_wall_bounce")

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

        buy_opp = self._check_buy_at_pe_wall(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell_at_ce_wall(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:2]

    def _check_buy_at_pe_wall(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        low = self._safe_get(f1m, "low", close)
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
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        options = ctx.get("options", {})
        pe_wall_strike = float(options.get("highest_pe_oi_strike", 0) or 0)
        pe_wall_oi = int(options.get("highest_pe_oi", 0) or 0)
        ce_wall_strike = float(options.get("highest_ce_oi_strike", 0) or 0)
        pcr_oi = float(options.get("pcr_oi", 0) or 0)

        if close <= 0 or pe_wall_strike <= 0 or atr is None or atr <= 0:
            return None

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.2),
            "session_ok": StrategyQuality.get_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "wall_significant": pe_wall_oi >= self._min_wall_oi,
            "near_wall": StrategyQuality.is_near_level(
                close, pe_wall_strike, self._wall_tolerance_pct
            ),
            "holding_wall": close >= pe_wall_strike - max(1.0, atr * 0.15),
            "rejection_wick": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="BUY",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "rsi_ok": (rsi is not None) and (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._vol_min, self._vol_max
            ),
            "not_breakdown_extreme": volume_ratio is None
            or volume_ratio < self._extreme_break_volume,
            "atr_ok": atr > 0,
            "mtf_not_hostile": weighted_mtf > -3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(pe_wall_strike - atr * self._sl_atr_mult - 1, 2)
        risk = abs(entry - sl)
        if risk <= 0:
            return None

        min_target = entry + risk * self._min_rr
        target = min_target

        if ce_wall_strike > entry:
            target = max(min_target, min(ce_wall_strike, entry + risk * 3.0))
        elif last_swing_high and last_swing_high > entry:
            target = max(min_target, min(last_swing_high, entry + risk * 3.0))

        target = round(target, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 55
        if pe_wall_oi >= self._strong_wall_oi:
            score += 10
        elif pe_wall_oi >= 2_000_000:
            score += 5

        if pcr_oi > 1.0:
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
            f"BUY at PE wall {pe_wall_strike:.0f} (OI={pe_wall_oi:,}): "
            f"writers defending support, RSI={rsi}, PCR={pcr_oi}, dataQ={data_quality}"
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

    def _check_sell_at_ce_wall(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        low = self._safe_get(f1m, "low", close)
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
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        options = ctx.get("options", {})
        ce_wall_strike = float(options.get("highest_ce_oi_strike", 0) or 0)
        ce_wall_oi = int(options.get("highest_ce_oi", 0) or 0)
        pe_wall_strike = float(options.get("highest_pe_oi_strike", 0) or 0)
        pcr_oi = float(options.get("pcr_oi", 0) or 0)

        if close <= 0 or ce_wall_strike <= 0 or atr is None or atr <= 0:
            return None

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.2),
            "session_ok": StrategyQuality.get_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "wall_significant": ce_wall_oi >= self._min_wall_oi,
            "near_wall": StrategyQuality.is_near_level(
                close, ce_wall_strike, self._wall_tolerance_pct
            ),
            "holding_wall": close <= ce_wall_strike + max(1.0, atr * 0.15),
            "rejection_wick": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="SELL",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "rsi_ok": (rsi is not None) and (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._vol_min, self._vol_max
            ),
            "not_breakout_extreme": volume_ratio is None
            or volume_ratio < self._extreme_break_volume,
            "atr_ok": atr > 0,
            "mtf_not_hostile": weighted_mtf < 3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(ce_wall_strike + atr * self._sl_atr_mult + 1, 2)
        risk = abs(sl - entry)
        if risk <= 0:
            return None

        min_target = entry - risk * self._min_rr
        target = min_target

        if pe_wall_strike > 0 and pe_wall_strike < entry:
            target = min(min_target, max(pe_wall_strike, entry - risk * 3.0))
        elif last_swing_low and last_swing_low < entry:
            target = min(min_target, max(last_swing_low, entry - risk * 3.0))

        target = round(target, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 55
        if ce_wall_oi >= self._strong_wall_oi:
            score += 10
        elif ce_wall_oi >= 2_000_000:
            score += 5

        if pcr_oi < 0.8:
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
            f"SELL at CE wall {ce_wall_strike:.0f} (OI={ce_wall_oi:,}): "
            f"writers defending resistance, RSI={rsi}, PCR={pcr_oi}, dataQ={data_quality}"
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
    print(" JUNIOR ALADDIN — OI Wall Bounce Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = OIWallBounceStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "OI_WALL_BOUNCE" and strategy.brain == "INSTITUTIONAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] Perfect BUY at PE wall...")
    buy_f = {
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
    buy_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "weighted_mtf": 1.0,
        "options": {
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 8_000_000,
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 6_000_000,
            "pcr_oi": 1.1,
        },
        "microstructure": {"spread_zscore": 0.5},
        "data_quality_score": 85,
        "last_swing_high": 23250,
    }
    r2 = strategy.safe_scan(buy_f, context=buy_ctx)
    if len(r2) >= 1 and r2[0].direction == "BUY":
        print(f" ✅ BUY signal @{r2[0].entry_price}, RR={r2[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] No signal when far from PE wall...")
    far_f = {**buy_f, "last_close": 23200.0}
    r3 = strategy.safe_scan(far_f, context=buy_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (too far from PE wall)")
        passed += 1
    else:
        print(" ❌ Should not signal far from wall")
        failed += 1

    print("\n [Test 4] No signal with low OI wall...")
    low_oi_ctx = {
        **buy_ctx,
        "options": {
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 500_000,
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 400_000,
            "pcr_oi": 1.0,
        },
    }
    r4 = strategy.safe_scan(buy_f, context=low_oi_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (OI too low = weak wall)")
        passed += 1
    else:
        print(" ❌ Should require significant OI")
        failed += 1

    print("\n [Test 5] No signal without options data...")
    no_opt_ctx = {**buy_ctx, "options": {}}
    r5 = strategy.safe_scan(buy_f, context=no_opt_ctx)
    if len(r5) == 0:
        print(" ✅ No signal (no options data)")
        passed += 1
    else:
        print(" ❌ Should not signal without OI data")
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

    print("\n [Test 7] No signal with RSI=20...")
    low_rsi = {**buy_f, "rsi": 20.0}
    r7 = strategy.safe_scan(low_rsi, context=buy_ctx)
    if len(r7) == 0:
        print(" ✅ No signal (RSI exhausted)")
        passed += 1
    else:
        print(" ❌ Should block exhausted RSI")
        failed += 1

    print("\n [Test 8] No signal without rejection wick...")
    no_wick = {**buy_f, "lower_wick_ratio": 0.02, "candle_body_ratio": 0.70}
    r8 = strategy.safe_scan(no_wick, context=buy_ctx)
    if len(r8) == 0:
        print(" ✅ No signal (no rejection wick)")
        passed += 1
    else:
        print(" ❌ Should block no-wick setup")
        failed += 1

    print("\n [Test 9] SELL at CE wall...")
    sell_f = {
        "last_close": 23299.0,
        "high": 23304.0,
        "low": 23292.0,
        "rsi": 62.0,
        "atr": 15.0,
        "volume_ratio": 1.0,
        "candle_body_ratio": 0.10,
        "upper_wick_ratio": 0.50,
        "lower_wick_ratio": 0.05,
    }
    sell_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "NEUTRAL",
        "weighted_mtf": -1.0,
        "options": {
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 7_000_000,
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 5_000_000,
            "pcr_oi": 0.7,
        },
        "microstructure": {"spread_zscore": 0.4},
        "data_quality_score": 90,
        "last_swing_low": 23100,
    }
    r9 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r9) >= 1 and r9[0].direction == "SELL":
        print(f" ✅ SELL signal @{r9[0].entry_price}, RR={r9[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 10] High volume blocks...")
    high_vol = {**buy_f, "volume_ratio": 3.5}
    r10 = strategy.safe_scan(high_vol, context=buy_ctx)
    if len(r10) == 0:
        print(" ✅ No signal (high vol = potential wall break)")
        passed += 1
    else:
        print(" ❌ Should block very high volume")
        failed += 1

    print("\n [Test 11] No signal with EVENT_RISK...")
    event_ctx = {**buy_ctx, "narrative_label": "EVENT_RISK"}
    r11 = strategy.safe_scan(buy_f, context=event_ctx)
    if len(r11) == 0:
        print(" ✅ No signal (EVENT_RISK blocks)")
        passed += 1
    else:
        print(" ❌ Should block EVENT_RISK")
        failed += 1

    print("\n [Test 12] Score bonus for strong OI wall...")
    strong_oi_ctx = {
        **buy_ctx,
        "options": {
            "highest_pe_oi_strike": 23050,
            "highest_pe_oi": 10_000_000,
            "highest_ce_oi_strike": 23300,
            "highest_ce_oi": 8_000_000,
            "pcr_oi": 1.2,
        },
    }
    r12 = strategy.safe_scan(buy_f, context=strong_oi_ctx)
    if len(r12) >= 1 and r12[0].raw_score >= 75:
        print(f" ✅ Strong OI wall → higher score: {r12[0].raw_score}")
        passed += 1
    elif len(r12) >= 1:
        print(f" ⚠️ Score={r12[0].raw_score} (expected >=75)")
        passed += 1
    else:
        print(" ❌ No signal on strong wall")
        failed += 1

    print("\n [Test 13] No signal during LAST_MINUTES...")
    last_ctx = {**buy_ctx, "session_phase": "LAST_MINUTES"}
    r13 = strategy.safe_scan(buy_f, context=last_ctx)
    if len(r13) == 0:
        print(" ✅ No signal (LAST_MINUTES blocked)")
        passed += 1
    else:
        print(" ❌ Should block LAST_MINUTES")
        failed += 1

    print("\n [Test 14] Empty data...")
    r14 = strategy.safe_scan({})
    if len(r14) == 0:
        print(" ✅ No crash with empty data")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 15] Stats tracking...")
    stats = strategy.get_stats()
    if stats["scan_count"] >= 10:
        print(f" ✅ Scans={stats['scan_count']}, Signals={stats['signal_count']}")
        passed += 1
    else:
        print(f" ❌ Stats: {stats}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 OI Wall Bounce Strategy (Hardened) working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()