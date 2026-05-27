"""
Junior Aladdin - Opening Range Breakout Strategy (Hardened Version)
===================================================================
PURPOSE:
Trade high-quality directional breakouts from the Opening Range (OR).

The opening range (typically 9:16-9:30) defines the market's initial auction.
A true breakout from ORH or ORL with momentum and participation can lead to
strong directional expansion.

This hardened version improves:
- breakout close quality
- OR width sanity
- anti-chase logic
- data-quality / spread protection
- stronger context filters
- target realism and room-to-run checks

BUY ORB CONDITIONS:
1. OR levels are established
2. OR width is realistic (not too narrow, not too wide)
3. Price closes above ORH with quality
4. Volume confirms the breakout
5. RSI confirms momentum but is not exhausted
6. Session is valid for ORB
7. Regime not CHOP / EVENT
8. Narrative not strongly against
9. ATR available
10. Spread/data quality acceptable
11. Not overextended
12. Some directional support from trend / supertrend / MTF

SELL CONDITIONS:
Mirror of BUY below ORL.

CONNECTS TO:
- StrategyBase / Opportunity
- StrategyQuality helper layer
- Time Context / Key Levels / Price-Momentum features
- Structural brain
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class OpeningRangeBreakoutStrategy(StrategyBase):
    """
    Hardened ORB strategy for real-market morning momentum.
    """

    @property
    def name(self) -> str:
        return "OPENING_RANGE_BREAKOUT"

    @property
    def brain(self) -> str:
        return "STRUCTURAL"

    def __init__(self):
        self._min_or_width = 20.0
        self._max_or_width = 120.0

        self._rsi_buy_low = 50
        self._rsi_buy_high = 70
        self._rsi_sell_low = 30
        self._rsi_sell_high = 50

        self._sl_atr_mult = 0.3
        self._target_or_mult = 2.0
        self._min_rr = 1.5

        self._min_vol_ratio = 1.0
        self._max_breakout_extension_atr = 1.2

        super().__init__()
        self._logger = setup_logger("strategy_opening_range_breakout")

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

        buy_opp = self._check_buy_breakout(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell_breakout(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:2]

    def _check_buy_breakout(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        low = self._safe_get(f1m, "low", close)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_high = ctx.get("last_swing_high")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        key_levels = ctx.get("key_levels", {})
        or_high = float(key_levels.get("or_high", 0) or 0)
        or_low = float(key_levels.get("or_low", 0) or 0)
        or_width = float(key_levels.get("or_width", 0) or 0)

        if close <= 0 or or_high <= 0 or or_low <= 0 or atr is None or atr <= 0:
            return None

        if or_width <= 0:
            or_width = or_high - or_low

        breakout_extension = close - or_high

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.0),
            "session_ok": session in ("INITIAL_BALANCE", "GOLDEN_AM"),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "or_established": or_high > 0 and or_low > 0 and or_width > 0,
            "or_width_ok": self._min_or_width <= or_width <= self._max_or_width,
            "above_orh": close > or_high,
            "breakout_quality": StrategyQuality.breakout_close_quality(
                close=close,
                level=or_high,
                candle_high=high,
                candle_low=low,
                direction="BUY",
                min_body_close_fraction=0.15,
            ),
            "volume_ok": volume_ratio is not None and volume_ratio >= self._min_vol_ratio,
            "rsi_ok": (rsi is not None) and (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "atr_ok": atr > 0,
            "direction_ok": trend_dir >= 0 or st_dir >= 0 or weighted_mtf >= 0,
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="BUY",
            ),
            "not_too_far_after_break": breakout_extension <= atr * self._max_breakout_extension_atr,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(or_high - atr * self._sl_atr_mult - 1, 2)
        risk = abs(entry - sl)
        if risk <= 0:
            return None

        min_target = entry + risk * self._min_rr
        or_target = or_high + or_width * self._target_or_mult
        target = max(or_target, min_target)

        if last_swing_high and last_swing_high > entry:
            target = min(target, last_swing_high)

        target = round(target, 2)

        if not StrategyQuality.price_has_room_to_target(entry, target, atr, 0.5):
            return None

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 55
        if volume_ratio is not None and volume_ratio >= 1.5:
            score += 10
        if rsi is not None and 55 <= rsi <= 65:
            score += 5
        if or_width < 60:
            score += 5
        if weighted_mtf >= 3.0:
            score += 5
        if narrative in ("MILD_BULLISH", "STRONG_BULLISH"):
            score += 5
        if session == "INITIAL_BALANCE":
            score += 5
        if regime == "TRENDING":
            score += 5
        if trend_dir == 1 or st_dir == 1:
            score += 5

        thesis = (
            f"Bullish ORB: close={close:.2f} above ORH={or_high:.2f}, "
            f"OR width={or_width:.1f}, vol={volume_ratio}, RSI={rsi}, dataQ={data_quality}"
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

    def _check_sell_breakout(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        low = self._safe_get(f1m, "low", close)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_low = ctx.get("last_swing_low")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        key_levels = ctx.get("key_levels", {})
        or_high = float(key_levels.get("or_high", 0) or 0)
        or_low = float(key_levels.get("or_low", 0) or 0)
        or_width = float(key_levels.get("or_width", 0) or 0)

        if close <= 0 or or_high <= 0 or or_low <= 0 or atr is None or atr <= 0:
            return None

        if or_width <= 0:
            or_width = or_high - or_low

        breakout_extension = or_low - close

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.0),
            "session_ok": session in ("INITIAL_BALANCE", "GOLDEN_AM"),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "or_established": or_high > 0 and or_low > 0 and or_width > 0,
            "or_width_ok": self._min_or_width <= or_width <= self._max_or_width,
            "below_orl": close < or_low,
            "breakout_quality": StrategyQuality.breakout_close_quality(
                close=close,
                level=or_low,
                candle_high=high,
                candle_low=low,
                direction="SELL",
                min_body_close_fraction=0.15,
            ),
            "volume_ok": volume_ratio is not None and volume_ratio >= self._min_vol_ratio,
            "rsi_ok": (rsi is not None) and (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "atr_ok": atr > 0,
            "direction_ok": trend_dir <= 0 or st_dir <= 0 or weighted_mtf <= 0,
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="SELL",
            ),
            "not_too_far_after_break": breakout_extension <= atr * self._max_breakout_extension_atr,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(or_low + atr * self._sl_atr_mult + 1, 2)
        risk = abs(sl - entry)
        if risk <= 0:
            return None

        min_target = entry - risk * self._min_rr
        or_target = or_low - or_width * self._target_or_mult
        target = min(or_target, min_target)

        if last_swing_low and last_swing_low < entry:
            target = max(target, last_swing_low)

        target = round(target, 2)

        if not StrategyQuality.price_has_room_to_target(entry, target, atr, 0.5):
            return None

        if not StrategyQuality.rr_ok(entry, sl, target, self._min_rr):
            return None

        score = 55
        if volume_ratio is not None and volume_ratio >= 1.5:
            score += 10
        if rsi is not None and 35 <= rsi <= 45:
            score += 5
        if or_width < 60:
            score += 5
        if weighted_mtf <= -3.0:
            score += 5
        if narrative in ("MILD_BEARISH", "STRONG_BEARISH"):
            score += 5
        if session == "INITIAL_BALANCE":
            score += 5
        if regime == "TRENDING":
            score += 5
        if trend_dir == -1 or st_dir == -1:
            score += 5

        thesis = (
            f"Bearish ORB: close={close:.2f} below ORL={or_low:.2f}, "
            f"OR width={or_width:.1f}, vol={volume_ratio}, RSI={rsi}, dataQ={data_quality}"
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
    print(" JUNIOR ALADDIN — Opening Range Breakout Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = OpeningRangeBreakoutStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "OPENING_RANGE_BREAKOUT" and strategy.brain == "STRUCTURAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] Perfect BUY ORB...")
    buy_f = {
        "last_close": 23210.0,
        "high": 23215.0,
        "low": 23195.0,
        "rsi": 58.0,
        "atr": 12.0,
        "volume_ratio": 1.5,
        "trend_direction": 1,
        "supertrend_direction": 1,
        "price_vs_vwap_pct": 0.35,
    }
    buy_ctx = {
        "regime": "TRENDING",
        "session_phase": "INITIAL_BALANCE",
        "narrative_label": "MILD_BULLISH",
        "weighted_mtf": 3.0,
        "last_swing_high": 23350,
        "data_quality_score": 85,
        "microstructure": {"spread_zscore": 0.5},
        "key_levels": {
            "or_high": 23200.0,
            "or_low": 23150.0,
            "or_width": 50.0,
        },
    }
    r2 = strategy.safe_scan(buy_f, context=buy_ctx)
    if len(r2) >= 1 and r2[0].direction == "BUY":
        print(f" ✅ BUY signal @{r2[0].entry_price}, RR={r2[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] No signal when price inside OR...")
    inside_f = {**buy_f, "last_close": 23180.0}
    r3 = strategy.safe_scan(inside_f, context=buy_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (price inside OR)")
        passed += 1
    else:
        print(" ❌ Should not signal inside OR")
        failed += 1

    print("\n [Test 4] No signal with low volume...")
    low_vol = {**buy_f, "volume_ratio": 0.5}
    r4 = strategy.safe_scan(low_vol, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (low volume)")
        passed += 1
    else:
        print(" ❌ Should not signal with low volume")
        failed += 1

    print("\n [Test 5] No signal during lunch...")
    lunch_ctx = {**buy_ctx, "session_phase": "LUNCH_LULL"}
    r5 = strategy.safe_scan(buy_f, context=lunch_ctx)
    if len(r5) == 0:
        print(" ✅ No signal (ORB is morning strategy)")
        passed += 1
    else:
        print(" ❌ Should not signal during lunch")
        failed += 1

    print("\n [Test 6] No signal with wide OR (>120pts)...")
    wide_ctx = {**buy_ctx, "key_levels": {"or_high": 23300, "or_low": 23150, "or_width": 150}}
    wide_f = {**buy_f, "last_close": 23310.0, "high": 23318.0, "low": 23295.0}
    r6 = strategy.safe_scan(wide_f, context=wide_ctx)
    if len(r6) == 0:
        print(" ✅ No signal (OR too wide)")
        passed += 1
    else:
        print(" ❌ Should not signal with OR > 120")
        failed += 1

    print("\n [Test 7] No signal with narrow OR (<20pts)...")
    narrow_ctx = {**buy_ctx, "key_levels": {"or_high": 23210, "or_low": 23200, "or_width": 10}}
    narrow_f = {**buy_f, "last_close": 23215.0, "high": 23218.0, "low": 23205.0}
    r7 = strategy.safe_scan(narrow_f, context=narrow_ctx)
    if len(r7) == 0:
        print(" ✅ No signal (OR too narrow)")
        passed += 1
    else:
        print(" ❌ Should not signal with OR < 20")
        failed += 1

    print("\n [Test 8] No signal in CHOP...")
    chop_ctx = {**buy_ctx, "regime": "CHOP"}
    r8 = strategy.safe_scan(buy_f, context=chop_ctx)
    if len(r8) == 0:
        print(" ✅ No signal (CHOP blocked)")
        passed += 1
    else:
        print(" ❌ Should not signal in CHOP")
        failed += 1

    print("\n [Test 9] No signal with RSI=78...")
    high_rsi = {**buy_f, "rsi": 78.0}
    r9 = strategy.safe_scan(high_rsi, context=buy_ctx)
    if len(r9) == 0:
        print(" ✅ No signal (RSI overbought)")
        passed += 1
    else:
        print(" ❌ Should not signal at RSI=78")
        failed += 1

    print("\n [Test 10] No signal without OR levels...")
    no_or_ctx = {**buy_ctx, "key_levels": {}}
    r10 = strategy.safe_scan(buy_f, context=no_or_ctx)
    if len(r10) == 0:
        print(" ✅ No signal (OR not established)")
        passed += 1
    else:
        print(" ❌ Should not signal without OR")
        failed += 1

    print("\n [Test 11] SELL ORB breakout...")
    sell_f = {
        "last_close": 23140.0,
        "high": 23155.0,
        "low": 23135.0,
        "rsi": 40.0,
        "atr": 12.0,
        "volume_ratio": 1.8,
        "trend_direction": -1,
        "supertrend_direction": -1,
        "price_vs_vwap_pct": 0.30,
    }
    sell_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "MILD_BEARISH",
        "weighted_mtf": -3.0,
        "last_swing_low": 23050,
        "data_quality_score": 88,
        "microstructure": {"spread_zscore": 0.4},
        "key_levels": {
            "or_high": 23200.0,
            "or_low": 23150.0,
            "or_width": 50.0,
        },
    }
    r11 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r11) >= 1 and r11[0].direction == "SELL":
        print(f" ✅ SELL signal @{r11[0].entry_price}, RR={r11[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 12] No signal with EVENT_RISK...")
    event_ctx = {**buy_ctx, "narrative_label": "EVENT_RISK"}
    r12 = strategy.safe_scan(buy_f, context=event_ctx)
    if len(r12) == 0:
        print(" ✅ No signal (EVENT_RISK blocks)")
        passed += 1
    else:
        print(" ❌ Should block EVENT_RISK")
        failed += 1

    print("\n [Test 13] Empty data...")
    r13 = strategy.safe_scan({})
    if len(r13) == 0:
        print(" ✅ No crash")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 14] Stats...")
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
        print("\n 🎉 Opening Range Breakout Strategy (Hardened) working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()