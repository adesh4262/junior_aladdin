"""
Junior Aladdin - Absorption Reversal Strategy
=============================================
PURPOSE:
Trade reversals after institutional absorption is detected.

Absorption means:
- large volume trades occur
- price does NOT move much in that direction
- a bigger player is absorbing the flow
This often appears near support/resistance and can lead to sharp reversal.

This is a TACTICAL strategy.

BUY ABSORPTION CONDITIONS:
1. Absorption detected by microstructure
2. Absorption direction is bullish
3. Price is near support / defended area / VAL / PDL if available
4. RSI not exhausted
5. Volume meaningful
6. Regime not CHOP
7. Session allows tactical trading
8. Narrative not strongly against
9. ATR available
10. Spread not severely widened

SELL ABSORPTION CONDITIONS:
Mirror of BUY.

RISK MODEL:
- BUY SL below local low / support minus small ATR buffer
- SELL SL above local high / resistance plus small ATR buffer
- Target = opposite level or minimum RR projection

CONNECTS TO:
- Microstructure features (absorption_detected, absorption_direction)
- Key levels / volume profile / session memory
- Tactical brain
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class AbsorptionReversalStrategy(StrategyBase):
    """
    Tactical reversal strategy using absorption signals.
    """

    @property
    def name(self) -> str:
        return "ABSORPTION_REVERSAL"

    @property
    def brain(self) -> str:
        return "TACTICAL"

    def __init__(self):
        self._rsi_buy_low = 28
        self._rsi_buy_high = 52
        self._rsi_sell_low = 48
        self._rsi_sell_high = 72

        self._min_volume_ratio = 1.2
        self._max_volume_ratio = 4.0

        self._sl_atr_mult = 0.2
        self._min_rr = 1.5
        self._level_tolerance_pct = 0.15

        super().__init__()
        self._logger = setup_logger("strategy_absorption_reversal")

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

        buy_opp = self._check_buy_absorption(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell_absorption(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:2]

    # ------------------------------------------------------------------
    # Support / Resistance collection
    # ------------------------------------------------------------------
    def _collect_support_levels(self, ctx: Dict) -> List[float]:
        levels: List[float] = []

        key_levels = ctx.get("key_levels", {})
        vp = ctx.get("volume_profile", {})
        options = ctx.get("options", {})
        session_memory = ctx.get("session_memory", {})

        for val in (
            key_levels.get("pdl"),
            key_levels.get("or_low"),
            key_levels.get("ib_low"),
            vp.get("val"),
            options.get("highest_pe_oi_strike"),
        ):
            if val and val > 0:
                levels.append(float(val))

        for zone in key_levels.get("sr_zones", []):
            if isinstance(zone, dict) and zone.get("type") == "support":
                lvl = zone.get("level", 0)
                if lvl > 0:
                    levels.append(float(lvl))

        for lvl in session_memory.get("levels_defended", []):
            if lvl and lvl > 0:
                levels.append(float(lvl))

        return sorted(set(round(x, 2) for x in levels))

    def _collect_resistance_levels(self, ctx: Dict) -> List[float]:
        levels: List[float] = []

        key_levels = ctx.get("key_levels", {})
        vp = ctx.get("volume_profile", {})
        options = ctx.get("options", {})

        for val in (
            key_levels.get("pdh"),
            key_levels.get("or_high"),
            key_levels.get("ib_high"),
            vp.get("vah"),
            options.get("highest_ce_oi_strike"),
        ):
            if val and val > 0:
                levels.append(float(val))

        for zone in key_levels.get("sr_zones", []):
            if isinstance(zone, dict) and zone.get("type") == "resistance":
                lvl = zone.get("level", 0)
                if lvl > 0:
                    levels.append(float(lvl))

        return sorted(set(round(x, 2) for x in levels))

    def _is_near_any_level(self, price: float, levels: List[float]) -> bool:
        if price <= 0 or not levels:
            return False
        for level in levels:
            if StrategyQuality.is_near_level(price, level, self._level_tolerance_pct):
                return True
        return False

    def _nearest_level(self, price: float, levels: List[float]) -> Optional[float]:
        if price <= 0 or not levels:
            return None
        return min(levels, key=lambda x: abs(price - x))

    # ------------------------------------------------------------------
    # BUY
    # ------------------------------------------------------------------
    def _check_buy_absorption(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        low = self._safe_get(f1m, "low", 0)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        last_swing_high = ctx.get("last_swing_high")

        micro = ctx.get("microstructure", {})
        absorption_detected = micro.get("absorption_detected", False)
        absorption_direction = micro.get("absorption_direction", "NONE")
        spread_zscore = micro.get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None

        support_levels = self._collect_support_levels(ctx)
        near_support = self._is_near_any_level(close, support_levels)
        nearest_support = self._nearest_level(close, support_levels)

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "absorption_detected": absorption_detected,
            "direction_bullish": absorption_direction == "BULLISH",
            "near_support": near_support,
            "rsi_ok": (rsi is None) or (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "volume_ok": (
                volume_ratio is not None
                and self._min_volume_ratio <= volume_ratio <= self._max_volume_ratio
            ),
            "atr_ok": atr > 0,
            "spread_not_broken": (spread_zscore is None) or (spread_zscore < 2.5),
            "macd_not_collapsing": (macd_hist is None) or (macd_hist > -8),
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        reference_level = nearest_support if nearest_support is not None else low
        sl = round(min(low, reference_level) - atr * self._sl_atr_mult - 1, 2)

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
        if near_support:
            score += 10
        if volume_ratio is not None and volume_ratio >= 2.0:
            score += 5
        if rsi is not None and 32 <= rsi <= 45:
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5
        if narrative in ("MILD_BULLISH", "STRONG_BULLISH"):
            score += 5
        if nearest_support is not None:
            score += 5

        thesis = (
            f"BUY absorption reversal: bullish absorption confirmed, "
            f"price near support {nearest_support if nearest_support else 'local'}, "
            f"vol={volume_ratio}x"
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

    # ------------------------------------------------------------------
    # SELL
    # ------------------------------------------------------------------
    def _check_sell_absorption(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", 0)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        last_swing_low = ctx.get("last_swing_low")

        micro = ctx.get("microstructure", {})
        absorption_detected = micro.get("absorption_detected", False)
        absorption_direction = micro.get("absorption_direction", "NONE")
        spread_zscore = micro.get("spread_zscore")

        if close <= 0 or atr is None or atr <= 0:
            return None

        resistance_levels = self._collect_resistance_levels(ctx)
        near_resistance = self._is_near_any_level(close, resistance_levels)
        nearest_resistance = self._nearest_level(close, resistance_levels)

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "absorption_detected": absorption_detected,
            "direction_bearish": absorption_direction == "BEARISH",
            "near_resistance": near_resistance,
            "rsi_ok": (rsi is None) or (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "volume_ok": (
                volume_ratio is not None
                and self._min_volume_ratio <= volume_ratio <= self._max_volume_ratio
            ),
            "atr_ok": atr > 0,
            "spread_not_broken": (spread_zscore is None) or (spread_zscore < 2.5),
            "macd_not_exploding": (macd_hist is None) or (macd_hist < 8),
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        reference_level = nearest_resistance if nearest_resistance is not None else high
        sl = round(max(high, reference_level) + atr * self._sl_atr_mult + 1, 2)

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
        if near_resistance:
            score += 10
        if volume_ratio is not None and volume_ratio >= 2.0:
            score += 5
        if rsi is not None and 55 <= rsi <= 68:
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5
        if narrative in ("MILD_BEARISH", "STRONG_BEARISH"):
            score += 5
        if nearest_resistance is not None:
            score += 5

        thesis = (
            f"SELL absorption reversal: bearish absorption confirmed, "
            f"price near resistance {nearest_resistance if nearest_resistance else 'local'}, "
            f"vol={volume_ratio}x"
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
    print(" JUNIOR ALADDIN — Absorption Reversal Strategy Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = AbsorptionReversalStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "ABSORPTION_REVERSAL" and strategy.brain == "TACTICAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] BUY bullish absorption near support...")
    buy_f = {
        "last_close": 23055.0,
        "low": 23040.0,
        "rsi": 40.0,
        "atr": 15.0,
        "volume_ratio": 2.0,
        "macd_histogram": -1.0,
    }
    buy_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "last_swing_high": 23200,
        "key_levels": {
            "pdl": 23050,
            "or_low": 23100,
            "ib_low": 23080,
            "sr_zones": [{"level": 23050, "strength": 3, "type": "support"}],
        },
        "volume_profile": {"val": 23050},
        "options": {"highest_pe_oi_strike": 23000},
        "session_memory": {"levels_defended": [23050]},
        "microstructure": {
            "absorption_detected": True,
            "absorption_direction": "BULLISH",
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

    print("\n [Test 3] No signal without absorption...")
    no_abs_ctx = {**buy_ctx, "microstructure": {"absorption_detected": False}}
    r3 = strategy.safe_scan(buy_f, context=no_abs_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (absorption missing)")
        passed += 1
    else:
        print(" ❌ Should require absorption")
        failed += 1

    print("\n [Test 4] No signal in CHOP...")
    chop_ctx = {**buy_ctx, "regime": "CHOP"}
    r4 = strategy.safe_scan(buy_f, context=chop_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (CHOP blocked)")
        passed += 1
    else:
        print(" ❌ Should block CHOP")
        failed += 1

    print("\n [Test 5] SELL bearish absorption near resistance...")
    sell_f = {
        "last_close": 23195.0,
        "high": 23210.0,
        "rsi": 60.0,
        "atr": 15.0,
        "volume_ratio": 1.8,
        "macd_histogram": 1.0,
    }
    sell_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "NEUTRAL",
        "last_swing_low": 23100,
        "key_levels": {
            "pdh": 23200,
            "or_high": 23200,
            "sr_zones": [{"level": 23200, "strength": 3, "type": "resistance"}],
        },
        "volume_profile": {"vah": 23200},
        "options": {"highest_ce_oi_strike": 23200},
        "microstructure": {
            "absorption_detected": True,
            "absorption_direction": "BEARISH",
            "spread_zscore": 0.4,
        },
        "data_quality_score": 90,
    }
    r5 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r5) >= 1 and r5[0].direction == "SELL":
        print(f" ✅ SELL signal @{r5[0].entry_price}, RR={r5[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
        failed += 1

    print("\n [Test 6] No signal with EVENT_RISK...")
    event_ctx = {**buy_ctx, "narrative_label": "EVENT_RISK"}
    r6 = strategy.safe_scan(buy_f, context=event_ctx)
    if len(r6) == 0:
        print(" ✅ No signal (EVENT_RISK blocked)")
        passed += 1
    else:
        print(" ❌ Should block EVENT_RISK")
        failed += 1

    print("\n [Test 7] Empty data...")
    r7 = strategy.safe_scan({})
    if len(r7) == 0:
        print(" ✅ No crash with empty input")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 8] Stats tracking...")
    stats = strategy.get_stats()
    if stats["scan_count"] >= 6:
        print(f" ✅ Scans={stats['scan_count']}, Signals={stats['signal_count']}")
        passed += 1
    else:
        print(f" ❌ Stats: {stats}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Absorption Reversal Strategy working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()