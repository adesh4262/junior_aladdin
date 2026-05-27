"""
Junior Aladdin - VWAP Pullback Strategy (Hardened Version)
=========================================================
PURPOSE:
Detect high-quality pullbacks to VWAP in established directional trends.

This is one of the highest-edge structural strategies in the system,
but only when:
- the trend is already real
- the pullback is controlled
- the reclaim is high quality
- there is enough room to the next objective
- liquidity and context remain supportive

BUY CONDITIONS (all must be true):
1. Regime supports directional trading
2. Trend direction bullish
3. EMA stack bullish
4. Price near VWAP within tight pullback tolerance
5. Price reclaims above VWAP with quality close
6. RSI cooled but not dead
7. Pullback volume controlled (not capitulation, not dead)
8. Rejection wick confirms demand
9. MTF alignment supportive
10. Narrative not strongly against
11. Session structurally tradeable
12. Spread / data quality acceptable
13. Not overextended from VWAP
14. Enough room to target

SELL CONDITIONS:
Mirror of BUY.

RISK MODEL:
- SL below pullback low / VWAP support structure with ATR buffer
- Target uses nearest structure high/low or RR projection
- Must maintain minimum acceptable RR

CONNECTS TO:
- StrategyBase / Opportunity
- StrategyQuality shared hardened validation
- Price / Momentum / Volatility / Microstructure / Context layers
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class VWAPPullbackStrategy(StrategyBase):
    """
    Hardened VWAP pullback strategy for structural entries.
    """

    @property
    def name(self) -> str:
        return "VWAP_PULLBACK"

    @property
    def brain(self) -> str:
        return "STRUCTURAL"

    def __init__(self):
        self._pullback_pct = 0.15
        self._rsi_buy_low = 35
        self._rsi_buy_high = 55
        self._rsi_sell_low = 45
        self._rsi_sell_high = 65

        self._min_volume_ratio = 0.55
        self._max_volume_ratio = 1.25
        self._wick_body_ratio = 1.8
        self._mtf_threshold = 4.5
        self._sl_atr_mult = 0.2
        self._min_rr = 1.5

        super().__init__()
        self._logger = setup_logger("strategy_vwap_pullback")

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
        vwap = self._safe_get(f1m, "vwap", 0)
        ema9 = self._safe_get(f1m, "ema_9")
        ema21 = self._safe_get(f1m, "ema_21")
        ema50 = self._safe_get(f1m, "ema_50")
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_high = ctx.get("last_swing_high")
        data_quality = ctx.get("data_quality_score", 100)

        if close <= 0 or vwap <= 0 or atr is None or atr <= 0:
            return None
        if ema9 is None or ema21 is None or ema50 is None:
            return None

        vwap_distance_pct = abs(close - vwap) / vwap * 100 if vwap > 0 else 999

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.0),
            "session_ok": StrategyQuality.structural_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("MILD_BEARISH", "STRONG_BEARISH", "EVENT_RISK")
            ),
            "trend_bullish": trend_dir == 1,
            "ema_aligned": ema9 > ema21 > ema50,
            "near_vwap": vwap_distance_pct <= self._pullback_pct,
            "above_vwap": close >= vwap,
            "reclaim_quality": StrategyQuality.breakout_close_quality(
                close=close,
                level=vwap,
                candle_high=high,
                candle_low=low,
                direction="BUY",
                min_body_close_fraction=0.12,
            ),
            "rsi_ok": rsi is not None and self._rsi_buy_low <= rsi <= self._rsi_buy_high,
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio,
                self._min_volume_ratio,
                self._max_volume_ratio,
            ),
            "wick_ok": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="BUY",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "supertrend_ok": st_dir >= 0,
            "mtf_ok": StrategyQuality.hard_direction_filter(weighted_mtf, "BUY", self._mtf_threshold),
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="BUY",
            ),
            "macd_not_weak": (macd_hist is None) or (macd_hist > -3),
        }

        if not self._all_conditions(conditions):
            return None

        structure_low = min(low, vwap, ema21)
        sl = round(structure_low - atr * self._sl_atr_mult - 1, 2)
        risk = abs(close - sl)
        if risk <= 0:
            return None

        min_target = close + risk * self._min_rr
        target = min_target

        if last_swing_high and last_swing_high > close:
            target = max(min_target, min(last_swing_high, close + risk * 3.0))

        if not StrategyQuality.price_has_room_to_target(close, target, atr, 0.5):
            return None

        target = round(target, 2)
        if not StrategyQuality.rr_ok(close, sl, target, self._min_rr):
            return None

        score = 55
        if weighted_mtf >= 6.0:
            score += 10
        if 40 <= (rsi or 0) <= 50:
            score += 5
        if volume_ratio is not None and volume_ratio <= 0.9:
            score += 5
        if regime == "TRENDING":
            score += 5
        if narrative in ("MILD_BULLISH", "STRONG_BULLISH"):
            score += 5
        if session == "GOLDEN_AM":
            score += 5
        if close > ema9 > ema21:
            score += 5

        thesis = (
            f"Bullish VWAP pullback: close={close:.2f}, VWAP={vwap:.2f}, "
            f"dist={vwap_distance_pct:.3f}%, RSI={rsi}, MTF={weighted_mtf}, "
            f"dataQ={data_quality}"
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
        vwap = self._safe_get(f1m, "vwap", 0)
        ema9 = self._safe_get(f1m, "ema_9")
        ema21 = self._safe_get(f1m, "ema_21")
        ema50 = self._safe_get(f1m, "ema_50")
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_low = ctx.get("last_swing_low")
        data_quality = ctx.get("data_quality_score", 100)

        if close <= 0 or vwap <= 0 or atr is None or atr <= 0:
            return None
        if ema9 is None or ema21 is None or ema50 is None:
            return None

        vwap_distance_pct = abs(close - vwap) / vwap * 100 if vwap > 0 else 999

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.0),
            "session_ok": StrategyQuality.structural_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("MILD_BULLISH", "STRONG_BULLISH", "EVENT_RISK")
            ),
            "trend_bearish": trend_dir == -1,
            "ema_aligned": ema9 < ema21 < ema50,
            "near_vwap": vwap_distance_pct <= self._pullback_pct,
            "below_vwap": close <= vwap,
            "reclaim_quality": StrategyQuality.breakout_close_quality(
                close=close,
                level=vwap,
                candle_high=high,
                candle_low=low,
                direction="SELL",
                min_body_close_fraction=0.12,
            ),
            "rsi_ok": rsi is not None and self._rsi_sell_low <= rsi <= self._rsi_sell_high,
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio,
                self._min_volume_ratio,
                self._max_volume_ratio,
            ),
            "wick_ok": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="SELL",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "supertrend_ok": st_dir <= 0,
            "mtf_ok": StrategyQuality.hard_direction_filter(weighted_mtf, "SELL", self._mtf_threshold),
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="SELL",
            ),
            "macd_not_weak": (macd_hist is None) or (macd_hist < 3),
        }

        if not self._all_conditions(conditions):
            return None

        structure_high = max(high, vwap, ema21)
        sl = round(structure_high + atr * self._sl_atr_mult + 1, 2)
        risk = abs(sl - close)
        if risk <= 0:
            return None

        min_target = close - risk * self._min_rr
        target = min_target

        if last_swing_low and last_swing_low < close:
            target = min(min_target, max(last_swing_low, close - risk * 3.0))

        if not StrategyQuality.price_has_room_to_target(close, target, atr, 0.5):
            return None

        target = round(target, 2)
        if not StrategyQuality.rr_ok(close, sl, target, self._min_rr):
            return None

        score = 55
        if weighted_mtf <= -6.0:
            score += 10
        if 50 <= (rsi or 100) <= 60:
            score += 5
        if volume_ratio is not None and volume_ratio <= 0.9:
            score += 5
        if regime == "TRENDING":
            score += 5
        if narrative in ("MILD_BEARISH", "STRONG_BEARISH"):
            score += 5
        if session == "GOLDEN_AM":
            score += 5
        if close < ema9 < ema21:
            score += 5

        thesis = (
            f"Bearish VWAP pullback: close={close:.2f}, VWAP={vwap:.2f}, "
            f"dist={vwap_distance_pct:.3f}%, RSI={rsi}, MTF={weighted_mtf}, "
            f"dataQ={data_quality}"
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
    print(" JUNIOR ALADDIN — VWAP Pullback Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = VWAPPullbackStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "VWAP_PULLBACK" and strategy.brain == "STRUCTURAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong name/brain")
        failed += 1

    print("\n [Test 2] Perfect BUY setup...")
    buy_features = {
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
    }
    buy_context = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "MILD_BULLISH",
        "weighted_mtf": 5.5,
        "last_swing_high": 23260.0,
        "data_quality_score": 85,
        "microstructure": {"spread_zscore": 0.5},
    }
    r2 = strategy.safe_scan(buy_features, context=buy_context)
    if len(r2) >= 1 and r2[0].direction == "BUY":
        print(f" ✅ BUY signal @{r2[0].entry_price}, RR={r2[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] Block low data quality...")
    bad_q_ctx = {**buy_context, "data_quality_score": 40}
    r3 = strategy.safe_scan(buy_features, context=bad_q_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (low data quality)")
        passed += 1
    else:
        print(" ❌ Should block low quality")
        failed += 1

    print("\n [Test 4] Block overextension...")
    ext_features = {**buy_features, "price_vs_vwap_pct": 1.5, "rsi": 80}
    r4 = strategy.safe_scan(ext_features, context=buy_context)
    if len(r4) == 0:
        print(" ✅ No signal (overextended)")
        passed += 1
    else:
        print(" ❌ Should block overextended move")
        failed += 1

    print("\n [Test 5] Perfect SELL setup...")
    sell_features = {
        "last_close": 23100.0,
        "high": 23108.0,
        "low": 23092.0,
        "vwap": 23102.0,
        "ema_9": 23099.0,
        "ema_21": 23103.0,
        "ema_50": 23130.0,
        "rsi": 55.0,
        "atr": 12.0,
        "volume_ratio": 0.80,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.05,
        "upper_wick_ratio": 0.40,
        "trend_direction": -1,
        "supertrend_direction": -1,
        "price_vs_vwap_pct": 0.01,
        "macd_histogram": -1.0,
    }
    sell_context = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "MILD_BEARISH",
        "weighted_mtf": -5.5,
        "last_swing_low": 23040.0,
        "data_quality_score": 88,
        "microstructure": {"spread_zscore": 0.4},
    }
    r5 = strategy.safe_scan(sell_features, context=sell_context)
    if len(r5) >= 1 and r5[0].direction == "SELL":
        print(f" ✅ SELL signal @{r5[0].entry_price}, RR={r5[0].risk_reward}")
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
        print("\n 🎉 VWAP Pullback Strategy (Hardened) working perfectly!")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()