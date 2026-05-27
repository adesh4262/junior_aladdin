"""
Junior Aladdin - Trend Continuation Strategy (Hardened Version)
==============================================================
PURPOSE:
Enter continuation moves after a controlled retracement in a confirmed trend.

This strategy is for the 2nd/3rd wave continuation:
- trend already established
- pullback occurs into structure
- momentum resumes in original direction
- enough room remains for continuation

BUY CONDITIONS:
1. Trend direction bullish
2. EMA stack bullish
3. Price retraced near EMA21
4. Price reclaimed above EMA9 and holds
5. RSI in continuation band
6. MACD histogram positive and not weakening badly
7. Supertrend confirms
8. Volume on retracement controlled
9. MTF supportive
10. Session supports structural trade
11. Regime supports continuation
12. Narrative not strongly against
13. Spread/data quality acceptable
14. Not overextended
15. Enough room to target

SELL CONDITIONS:
Mirror of BUY.

CONNECTS TO:
- StrategyBase / Opportunity
- StrategyQuality shared helper layer
- Trend, momentum, volatility, microstructure, context
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class TrendContinuationStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "TREND_CONTINUATION"

    @property
    def brain(self) -> str:
        return "STRUCTURAL"

    def __init__(self):
        self._ema_pullback_pct = 0.20
        self._rsi_buy_low = 45
        self._rsi_buy_high = 65
        self._rsi_sell_low = 35
        self._rsi_sell_high = 55

        self._vol_min = 0.55
        self._vol_max = 1.25
        self._sl_atr_mult = 0.15
        self._min_rr = 1.5
        self._mtf_threshold = 3.0

        super().__init__()
        self._logger = setup_logger("strategy_trend_continuation")

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

        buy_opp = self._check_buy(features_1m, features_5m, features_15m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell(features_1m, features_5m, features_15m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        return opportunities

    def _check_buy(
        self,
        f1m: Dict,
        f5m: Optional[Dict],
        f15m: Optional[Dict],
        ctx: Dict,
    ) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        low = self._safe_get(f1m, "low", close)
        ema9 = self._safe_get(f1m, "ema_9")
        ema21 = self._safe_get(f1m, "ema_21")
        ema50 = self._safe_get(f1m, "ema_50")
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        macd_hist = self._safe_get(f1m, "macd_histogram")
        macd_slope = self._safe_get(f1m, "macd_hist_slope")
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_high = ctx.get("last_swing_high")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None
        if ema9 is None or ema21 is None or ema50 is None:
            return None

        ema21_dist_pct = abs(close - ema21) / ema21 * 100 if ema21 > 0 else 999

        conditions = {
            "data_quality_ok": data_quality >= 60,
            "spread_ok": spread_zscore is None or spread_zscore < 2.0,
            "session_ok": StrategyQuality.structural_session_ok(ctx),
            "regime_ok": regime in ("TRENDING", "VOLATILE", "UNKNOWN"),
            "narrative_ok": narrative not in ("STRONG_BEARISH", "EVENT_RISK"),
            "trend_bullish": trend_dir == 1,
            "ema_aligned": ema9 > ema21 > ema50,
            "near_ema21": ema21_dist_pct <= self._ema_pullback_pct,
            "above_ema9": close >= ema9,
            "rsi_ok": rsi is not None and self._rsi_buy_low <= rsi <= self._rsi_buy_high,
            "volume_ok": volume_ratio is not None and self._vol_min <= volume_ratio <= self._vol_max,
            "supertrend_ok": st_dir >= 0,
            "macd_ok": macd_hist is not None and macd_hist > 0,
            "mtf_ok": weighted_mtf >= self._mtf_threshold,
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="BUY",
            ),
        }

        if not self._all_conditions(conditions):
            return None

        sl = round(min(low, ema50) - atr * self._sl_atr_mult - 1, 2)
        risk = abs(close - sl)
        if risk <= 0:
            return None

        min_target = close + risk * self._min_rr
        target = min_target
        if last_swing_high and last_swing_high > close:
            target = max(min_target, min(last_swing_high, close + risk * 3.0))
        target = round(target, 2)

        if not StrategyQuality.rr_ok(close, sl, target, self._min_rr):
            return None

        score = 55
        if trend_dir == 1 and st_dir == 1:
            score += 10
        if rsi is not None and 50 <= rsi <= 58:
            score += 5
        if macd_slope is not None and macd_slope > 0:
            score += 5
        if weighted_mtf >= 5.0:
            score += 5
        if narrative in ("MILD_BULLISH", "STRONG_BULLISH"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if regime == "TRENDING":
            score += 5
        if volume_ratio is not None and volume_ratio <= 1.0:
            score += 5

        thesis = (
            f"BUY continuation: EMA stack bullish, pullback to EMA21, "
            f"RSI={rsi}, MACD={macd_hist}, MTF={weighted_mtf}, dataQ={data_quality}"
        )

        return Opportunity(
            strategy=self.name,
            direction="BUY",
            entry_price=close,
            sl_price=sl,
            target_price=target,
            raw_score=score,
            thesis=thesis,
            timeframe="1min",
            brain=self.brain,
            conditions_met=conditions,
        )

    def _check_sell(
        self,
        f1m: Dict,
        f5m: Optional[Dict],
        f15m: Optional[Dict],
        ctx: Dict,
    ) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        low = self._safe_get(f1m, "low", close)
        ema9 = self._safe_get(f1m, "ema_9")
        ema21 = self._safe_get(f1m, "ema_21")
        ema50 = self._safe_get(f1m, "ema_50")
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        macd_hist = self._safe_get(f1m, "macd_histogram")
        macd_slope = self._safe_get(f1m, "macd_hist_slope")
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_low = ctx.get("last_swing_low")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None
        if ema9 is None or ema21 is None or ema50 is None:
            return None

        ema21_dist_pct = abs(close - ema21) / ema21 * 100 if ema21 > 0 else 999

        conditions = {
            "data_quality_ok": data_quality >= 60,
            "spread_ok": spread_zscore is None or spread_zscore < 2.0,
            "session_ok": StrategyQuality.structural_session_ok(ctx),
            "regime_ok": regime in ("TRENDING", "VOLATILE", "UNKNOWN"),
            "narrative_ok": narrative not in ("STRONG_BULLISH", "EVENT_RISK"),
            "trend_bearish": trend_dir == -1,
            "ema_aligned": ema9 < ema21 < ema50,
            "near_ema21": ema21_dist_pct <= self._ema_pullback_pct,
            "below_ema9": close <= ema9,
            "rsi_ok": rsi is not None and self._rsi_sell_low <= rsi <= self._rsi_sell_high,
            "volume_ok": volume_ratio is not None and self._vol_min <= volume_ratio <= self._vol_max,
            "supertrend_ok": st_dir <= 0,
            "macd_ok": macd_hist is not None and macd_hist < 0,
            "mtf_ok": weighted_mtf <= -self._mtf_threshold,
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="SELL",
            ),
        }

        if not self._all_conditions(conditions):
            return None

        sl = round(max(high, ema50) + atr * self._sl_atr_mult + 1, 2)
        risk = abs(sl - close)
        if risk <= 0:
            return None

        min_target = close - risk * self._min_rr
        target = min_target
        if last_swing_low and last_swing_low < close:
            target = min(min_target, max(last_swing_low, close - risk * 3.0))
        target = round(target, 2)

        if not StrategyQuality.rr_ok(close, sl, target, self._min_rr):
            return None

        score = 55
        if trend_dir == -1 and st_dir == -1:
            score += 10
        if rsi is not None and 42 <= rsi <= 50:
            score += 5
        if macd_slope is not None and macd_slope < 0:
            score += 5
        if weighted_mtf <= -5.0:
            score += 5
        if narrative in ("MILD_BEARISH", "STRONG_BEARISH"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if regime == "TRENDING":
            score += 5
        if volume_ratio is not None and volume_ratio <= 1.0:
            score += 5

        thesis = (
            f"SELL continuation: EMA stack bearish, pullback to EMA21, "
            f"RSI={rsi}, MACD={macd_hist}, MTF={weighted_mtf}, dataQ={data_quality}"
        )

        return Opportunity(
            strategy=self.name,
            direction="SELL",
            entry_price=close,
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
    print(" JUNIOR ALADDIN — Trend Continuation Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = TrendContinuationStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "TREND_CONTINUATION" and strategy.brain == "STRUCTURAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] Perfect BUY continuation...")
    buy_f = {
        "last_close": 23280.0,
        "high": 23286.0,
        "low": 23272.0,
        "ema_9": 23278.0,
        "ema_21": 23275.0,
        "ema_50": 23240.0,
        "rsi": 55.0,
        "atr": 12.0,
        "volume_ratio": 0.8,
        "trend_direction": 1,
        "supertrend_direction": 1,
        "macd_histogram": 5.0,
        "macd_hist_slope": 1.5,
        "price_vs_vwap_pct": 0.15,
    }
    buy_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "MILD_BULLISH",
        "weighted_mtf": 5.0,
        "last_swing_high": 23350,
        "data_quality_score": 85,
        "microstructure": {"spread_zscore": 0.5},
    }
    r2 = strategy.safe_scan(buy_f, context=buy_ctx)
    buys = [x for x in r2 if x.direction == "BUY"]
    if buys:
        print(f" ✅ BUY signal @{buys[0].entry_price}, RR={buys[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] Block low data quality...")
    bad_ctx = {**buy_ctx, "data_quality_score": 40}
    r3 = strategy.safe_scan(buy_f, context=bad_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (low data quality)")
        passed += 1
    else:
        print(" ❌ Should block low quality")
        failed += 1

    print("\n [Test 4] Block far from EMA21...")
    far_f = {**buy_f, "last_close": 23350.0, "high": 23356.0, "low": 23340.0}
    r4 = strategy.safe_scan(far_f, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (too far from EMA21)")
        passed += 1
    else:
        print(" ❌ Should block far pullback")
        failed += 1

    print("\n [Test 5] Perfect SELL continuation...")
    sell_f = {
        "last_close": 23120.0,
        "high": 23128.0,
        "low": 23112.0,
        "ema_9": 23122.0,
        "ema_21": 23125.0,
        "ema_50": 23160.0,
        "rsi": 42.0,
        "atr": 12.0,
        "volume_ratio": 0.7,
        "trend_direction": -1,
        "supertrend_direction": -1,
        "macd_histogram": -6.0,
        "macd_hist_slope": -2.0,
        "price_vs_vwap_pct": 0.12,
    }
    sell_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "MILD_BEARISH",
        "weighted_mtf": -5.5,
        "last_swing_low": 23050,
        "data_quality_score": 90,
        "microstructure": {"spread_zscore": 0.4},
    }
    r5 = strategy.safe_scan(sell_f, context=sell_ctx)
    sells = [x for x in r5 if x.direction == "SELL"]
    if sells:
        print(f" ✅ SELL signal @{sells[0].entry_price}, RR={sells[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 6] Empty data...")
    r6 = strategy.safe_scan({})
    if len(r6) == 0:
        print(" ✅ No crash with empty input")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 7] Stats tracking...")
    stats = strategy.get_stats()
    if stats["scan_count"] >= 5:
        print(f" ✅ Scans={stats['scan_count']}, Signals={stats['signal_count']}")
        passed += 1
    else:
        print(f" ❌ Stats: {stats}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Trend Continuation Strategy (Hardened) working perfectly!")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()