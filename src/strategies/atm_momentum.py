"""
Junior Aladdin - ATM Momentum Burst Strategy (Hardened Version)
===============================================================
PURPOSE:
Capture fast directional momentum bursts when price acceleration,
volume confirmation, and directional alignment expand together.

This is a TACTICAL brain strategy:
- fast entry
- fast exit
- strong filters
- avoid weak, late, noisy, illiquid bursts

WHY THIS HARDENED VERSION EXISTS:
Earlier versions had a hidden contradiction:
- SL tied to ATR
- target capped too tightly
- RR sometimes became mathematically impossible
This version fixes that and also adds:
- stronger momentum confirmation
- anti-chase protection
- liquidity sanity
- cooldown enforcement
- adaptive but realistic tactical target construction

BUY CONDITIONS:
1. Positive acceleration
2. RSI in bullish momentum zone
3. RSI slope rising or unavailable
4. Volume above threshold
5. MACD histogram positive
6. Supertrend confirms
7. Spread not abnormally wide
8. Session suitable for tactical execution
9. Regime not CHOP / EVENT
10. Narrative not strongly against
11. ATR available
12. ROC positive or unavailable
13. Candle body strong enough
14. Not overextended from VWAP
15. Data quality acceptable

SELL CONDITIONS:
Mirror of BUY.

RISK MODEL:
- SL = 0.8 * ATR
- target distance = adaptive tactical target
- minimum practical RR enforced
- tactical target allowed to expand above 8 points if needed to preserve valid RR

CONNECTS TO:
- StrategyBase / Opportunity
- StrategyQuality helper layer
- Momentum features
- Microstructure spread quality
- Time / regime / narrative context
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger

IST = timezone(timedelta(hours=5, minutes=30))


class ATMMomentumBurstStrategy(StrategyBase):
    """
    Tactical momentum strategy for high-quality directional bursts.
    """

    @property
    def name(self) -> str:
        return "ATM_MOMENTUM_BURST"

    @property
    def brain(self) -> str:
        return "TACTICAL"

    def __init__(self):
        # RSI momentum zones
        self._rsi_buy_min = 55
        self._rsi_buy_max = 78
        self._rsi_sell_min = 22
        self._rsi_sell_max = 45

        # Confirmation thresholds
        self._min_volume_ratio = 1.3
        self._min_body_ratio = 0.4

        # Risk model
        self._sl_atr_mult = 0.8
        self._target_points_min = 3.0
        self._target_points_soft_cap = 8.0
        self._target_points_hard_cap = 20.0
        self._min_rr = 1.5

        # Cooldown
        self._cooldown_seconds = 180
        self._last_signal_time: Optional[datetime] = None

        super().__init__()
        self._logger = setup_logger("strategy_atm_momentum")

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Opportunity]:
        if not features_1m or not context:
            return []

        if self._is_in_cooldown():
            return []

        opportunities: List[Opportunity] = []

        buy_opp = self._check_buy_burst(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)
            self._last_signal_time = datetime.now(IST)
            return opportunities

        sell_opp = self._check_sell_burst(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)
            self._last_signal_time = datetime.now(IST)

        return opportunities

    def _is_in_cooldown(self) -> bool:
        if self._last_signal_time is None:
            return False
        now = datetime.now(IST)
        elapsed = (now - self._last_signal_time).total_seconds()
        return elapsed < self._cooldown_seconds

    def _build_target_distance(self, risk: float, atr: float) -> float:
        """
        Tactical target builder:
        1. Start with tactical soft target range (3-8)
        2. If RR is too low, expand to preserve minimum RR
        3. Hard-cap at 20 points so strategy remains tactical
        """
        if risk <= 0 or atr <= 0:
            return 0.0

        base_target = max(
            self._target_points_min,
            min(self._target_points_soft_cap, atr * 0.8),
        )

        rr_target = risk * self._min_rr
        target_distance = max(base_target, rr_target)

        return round(min(target_distance, self._target_points_hard_cap), 2)

    def _check_buy_burst(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        rsi = self._safe_get(f1m, "rsi")
        rsi_slope = self._safe_get(f1m, "rsi_slope")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        macd_hist = self._safe_get(f1m, "macd_histogram")
        macd_slope = self._safe_get(f1m, "macd_hist_slope")
        price_accel = self._safe_get(f1m, "price_acceleration")
        roc_5 = self._safe_get(f1m, "roc_5")
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        micro = ctx.get("microstructure", {})
        spread_zscore = micro.get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.0),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "acceleration": price_accel is not None and price_accel > 0,
            "rsi_momentum": rsi is not None and self._rsi_buy_min <= rsi <= self._rsi_buy_max,
            "rsi_rising": (rsi_slope is None) or (rsi_slope > 0),
            "volume_ok": volume_ratio is not None and volume_ratio >= self._min_volume_ratio,
            "macd_ok": macd_hist is not None and macd_hist > 0,
            "supertrend_ok": st_dir >= 0,
            "atr_ok": atr > 0,
            "roc_ok": (roc_5 is None) or (roc_5 > 0),
            "body_ok": body_ratio >= self._min_body_ratio,
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="BUY",
            ),
            "trend_supportive": trend_dir >= 0,
            "mtf_not_hostile": weighted_mtf > -2.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(close - atr * self._sl_atr_mult, 2)
        risk = abs(entry - sl)
        if risk <= 0:
            return None

        target_points = self._build_target_distance(risk, atr)
        if target_points <= 0:
            return None

        target = round(entry + target_points, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, 1.0):
            return None

        score = 50
        if volume_ratio is not None and volume_ratio >= 2.0:
            score += 10
        if rsi is not None and 60 <= rsi <= 70:
            score += 5
        if macd_slope is not None and macd_slope > 0:
            score += 5
        if price_accel is not None and price_accel > 1:
            score += 5
        if weighted_mtf >= 3.0:
            score += 5
        if narrative in ("MILD_BULLISH", "STRONG_BULLISH"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if regime == "TRENDING":
            score += 5
        if trend_dir == 1:
            score += 5

        thesis = (
            f"Bullish ATM momentum burst: accel={price_accel}, RSI={rsi}, "
            f"RSI_slope={rsi_slope}, vol={volume_ratio}x, MACD={macd_hist}, "
            f"target={target_points:.1f}pts, dataQ={data_quality}"
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

    def _check_sell_burst(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        rsi = self._safe_get(f1m, "rsi")
        rsi_slope = self._safe_get(f1m, "rsi_slope")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        macd_hist = self._safe_get(f1m, "macd_histogram")
        macd_slope = self._safe_get(f1m, "macd_hist_slope")
        price_accel = self._safe_get(f1m, "price_acceleration")
        roc_5 = self._safe_get(f1m, "roc_5")
        st_dir = self._safe_get(f1m, "supertrend_direction", 0)
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        trend_dir = self._safe_get(f1m, "trend_direction", 0)
        price_vs_vwap_pct = self._safe_get(f1m, "price_vs_vwap_pct", 0.0)

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        micro = ctx.get("microstructure", {})
        spread_zscore = micro.get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.0),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "acceleration": price_accel is not None and price_accel < 0,
            "rsi_momentum": rsi is not None and self._rsi_sell_min <= rsi <= self._rsi_sell_max,
            "rsi_falling": (rsi_slope is None) or (rsi_slope < 0),
            "volume_ok": volume_ratio is not None and volume_ratio >= self._min_volume_ratio,
            "macd_ok": macd_hist is not None and macd_hist < 0,
            "supertrend_ok": st_dir <= 0,
            "atr_ok": atr > 0,
            "roc_ok": (roc_5 is None) or (roc_5 < 0),
            "body_ok": body_ratio >= self._min_body_ratio,
            "not_overextended": StrategyQuality.overextension_filter(
                price_vs_vwap_pct=price_vs_vwap_pct,
                rsi=rsi,
                direction="SELL",
            ),
            "trend_supportive": trend_dir <= 0,
            "mtf_not_hostile": weighted_mtf < 2.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(close + atr * self._sl_atr_mult, 2)
        risk = abs(sl - entry)
        if risk <= 0:
            return None

        target_points = self._build_target_distance(risk, atr)
        if target_points <= 0:
            return None

        target = round(entry - target_points, 2)

        if not StrategyQuality.rr_ok(entry, sl, target, 1.0):
            return None

        score = 50
        if volume_ratio is not None and volume_ratio >= 2.0:
            score += 10
        if rsi is not None and 30 <= rsi <= 40:
            score += 5
        if macd_slope is not None and macd_slope < 0:
            score += 5
        if price_accel is not None and price_accel < -1:
            score += 5
        if weighted_mtf <= -3.0:
            score += 5
        if narrative in ("MILD_BEARISH", "STRONG_BEARISH"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if regime == "TRENDING":
            score += 5
        if trend_dir == -1:
            score += 5

        thesis = (
            f"Bearish ATM momentum burst: accel={price_accel}, RSI={rsi}, "
            f"RSI_slope={rsi_slope}, vol={volume_ratio}x, MACD={macd_hist}, "
            f"target={target_points:.1f}pts, dataQ={data_quality}"
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
    print(" JUNIOR ALADDIN — ATM Momentum Burst Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = ATMMomentumBurstStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "ATM_MOMENTUM_BURST" and strategy.brain == "TACTICAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] Perfect BUY momentum burst...")
    buy_f = {
        "last_close": 23220.0,
        "rsi": 62.0,
        "rsi_slope": 5.0,
        "atr": 12.0,
        "volume_ratio": 1.8,
        "macd_histogram": 6.0,
        "macd_hist_slope": 2.0,
        "price_acceleration": 3.0,
        "roc_5": 0.08,
        "supertrend_direction": 1,
        "candle_body_ratio": 0.65,
        "trend_direction": 1,
        "price_vs_vwap_pct": 0.25,
    }
    buy_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "MILD_BULLISH",
        "weighted_mtf": 4.0,
        "microstructure": {"spread_zscore": 0.5},
        "data_quality_score": 85,
    }
    strategy._last_signal_time = None
    results = strategy.safe_scan(buy_f, context=buy_ctx)
    if len(results) >= 1:
        opp = results[0]
        print(f" ✅ Signal: {opp.direction} @{opp.entry_price}")
        print(f" SL={opp.sl_price}, TGT={opp.target_price}, RR={opp.risk_reward}")
        print(f" Score={opp.raw_score}")
        passed += 1

        if opp.direction == "BUY":
            print(" ✅ Direction correct")
            passed += 1
        else:
            print(" ❌ Expected BUY")
            failed += 1

        if opp.risk_reward >= 1.0:
            print(f" ✅ RR={opp.risk_reward}")
            passed += 1
        else:
            print(" ❌ RR too low")
            failed += 1

        pts = abs(opp.target_price - opp.entry_price)
        if pts >= 3:
            print(f" ✅ Target {pts:.1f}pts valid for tactical burst")
            passed += 1
        else:
            print(" ❌ Target too small")
            failed += 1

        if all(opp.conditions_met.values()):
            print(f" ✅ All {len(opp.conditions_met)} conditions met")
            passed += 1
        else:
            false_c = {k: v for k, v in opp.conditions_met.items() if not v}
            print(f" ❌ Failed: {false_c}")
            failed += 1
    else:
        print(" ❌ No signal")
        failed += 5

    print("\n [Test 3] No signal with RSI=45...")
    strategy._last_signal_time = None
    low_rsi = {**buy_f, "rsi": 45.0}
    r3 = strategy.safe_scan(low_rsi, context=buy_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (RSI below momentum threshold)")
        passed += 1
    else:
        print(" ❌ Should not signal with low RSI")
        failed += 1

    print("\n [Test 4] No signal with low volume...")
    strategy._last_signal_time = None
    low_vol = {**buy_f, "volume_ratio": 0.8}
    r4 = strategy.safe_scan(low_vol, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (low volume)")
        passed += 1
    else:
        print(" ❌ Should not signal with low volume")
        failed += 1

    print("\n [Test 5] No signal with negative acceleration...")
    strategy._last_signal_time = None
    neg_accel = {**buy_f, "price_acceleration": -2.0}
    r5 = strategy.safe_scan(neg_accel, context=buy_ctx)
    if len(r5) == 0:
        print(" ✅ No signal (negative acceleration)")
        passed += 1
    else:
        print(" ❌ Should not signal with negative acceleration")
        failed += 1

    print("\n [Test 6] No signal with MACD<0...")
    strategy._last_signal_time = None
    neg_macd = {**buy_f, "macd_histogram": -3.0}
    r6 = strategy.safe_scan(neg_macd, context=buy_ctx)
    if len(r6) == 0:
        print(" ✅ No signal (MACD against)")
        passed += 1
    else:
        print(" ❌ Should not signal with MACD<0")
        failed += 1

    print("\n [Test 7] No signal during LUNCH_LULL...")
    strategy._last_signal_time = None
    lunch_ctx = {**buy_ctx, "session_phase": "LUNCH_LULL"}
    r7 = strategy.safe_scan(buy_f, context=lunch_ctx)
    if len(r7) == 0:
        print(" ✅ No signal (lunch blocked)")
        passed += 1
    else:
        print(" ❌ Should not signal during lunch")
        failed += 1

    print("\n [Test 8] No signal in CHOP...")
    strategy._last_signal_time = None
    chop_ctx = {**buy_ctx, "regime": "CHOP"}
    r8 = strategy.safe_scan(buy_f, context=chop_ctx)
    if len(r8) == 0:
        print(" ✅ No signal (CHOP blocked)")
        passed += 1
    else:
        print(" ❌ Should not signal in CHOP")
        failed += 1

    print("\n [Test 9] No signal with wide spread...")
    strategy._last_signal_time = None
    wide_spread_ctx = {**buy_ctx, "microstructure": {"spread_zscore": 3.0}}
    r9 = strategy.safe_scan(buy_f, context=wide_spread_ctx)
    if len(r9) == 0:
        print(" ✅ No signal (spread widening = no liquidity)")
        passed += 1
    else:
        print(" ❌ Should not signal with wide spread")
        failed += 1

    print("\n [Test 10] No signal with small body...")
    strategy._last_signal_time = None
    small_body = {**buy_f, "candle_body_ratio": 0.1}
    r10 = strategy.safe_scan(small_body, context=buy_ctx)
    if len(r10) == 0:
        print(" ✅ No signal (small body = no conviction)")
        passed += 1
    else:
        print(" ❌ Should not signal with small body")
        failed += 1

    print("\n [Test 11] Cooldown enforcement...")
    strategy._last_signal_time = datetime.now(IST)
    r11 = strategy.safe_scan(buy_f, context=buy_ctx)
    if len(r11) == 0:
        print(" ✅ No signal (cooldown active)")
        passed += 1
    else:
        print(" ❌ Cooldown failed")
        failed += 1

    print("\n [Test 12] SELL momentum burst...")
    strategy._last_signal_time = None
    sell_f = {
        "last_close": 23180.0,
        "rsi": 35.0,
        "rsi_slope": -4.0,
        "atr": 12.0,
        "volume_ratio": 1.6,
        "macd_histogram": -5.0,
        "macd_hist_slope": -1.5,
        "price_acceleration": -3.0,
        "roc_5": -0.06,
        "supertrend_direction": -1,
        "candle_body_ratio": 0.60,
        "trend_direction": -1,
        "price_vs_vwap_pct": 0.22,
    }
    sell_ctx = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "MILD_BEARISH",
        "weighted_mtf": -4.0,
        "microstructure": {},
        "data_quality_score": 90,
    }
    r12 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r12) >= 1 and r12[0].direction == "SELL":
        print(f" ✅ SELL signal @{r12[0].entry_price}, RR={r12[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 13] No signal with EVENT_RISK...")
    strategy._last_signal_time = None
    event_ctx = {**buy_ctx, "narrative_label": "EVENT_RISK"}
    r13 = strategy.safe_scan(buy_f, context=event_ctx)
    if len(r13) == 0:
        print(" ✅ No signal (EVENT_RISK blocks)")
        passed += 1
    else:
        print(" ❌ Should block EVENT_RISK")
        failed += 1

    print("\n [Test 14] Empty data...")
    strategy._last_signal_time = None
    r14 = strategy.safe_scan({})
    if len(r14) == 0:
        print(" ✅ No crash with empty data")
        passed += 1
    else:
        print(" ❌ Should return empty on empty input")
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
        print("\n 🎉 ATM Momentum Burst Strategy (Hardened) working perfectly!")
        print(" ✅ Step 40 batch COMPLETE (4 strategies).")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()