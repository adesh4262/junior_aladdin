"""
Junior Aladdin - Failed Breakout Reversal Strategy (Hardened Version)
=====================================================================
PURPOSE:
Trade reversals after fake breakouts beyond important levels.

A failed breakout happens when price pushes above resistance or below support,
attracts breakout traders, then quickly returns back inside the range.
This traps weak hands and often creates a strong reversal move.

This hardened version improves:
- stronger candidate level selection
- proper reclaim quality checks
- stronger trap-style validation
- data quality and spread protection
- better volume sanity
- more realistic target/room checks

BUY FAILED BREAKDOWN CONDITIONS:
1. A valid support level exists near price
2. Price sweeps below support by meaningful pierce
3. Price reclaims back above support with quality close
4. Lower wick confirms rejection
5. Volume is meaningful but not breakdown-chaotic
6. Regime not CHOP / EVENT
7. Session allows tactical execution
8. Narrative not strongly against
9. ATR available
10. RSI not exhausted
11. Spread / data quality acceptable

SELL CONDITIONS:
Mirror of BUY at resistance.

CONNECTS TO:
- Key Levels / S&R zones
- OR / IB / prior-day levels
- Volume Profile
- OI walls
- Session memory
- StrategyBase / Opportunity
- StrategyQuality helper layer
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class FailedBreakoutReversalStrategy(StrategyBase):
    """
    Hardened tactical reversal strategy for fake breakouts / fake breakdowns.
    """

    @property
    def name(self) -> str:
        return "FAILED_BREAKOUT_REVERSAL"

    @property
    def brain(self) -> str:
        return "TACTICAL"

    def __init__(self):
        self._min_pierce_points = 3.0
        self._level_tolerance_pct = 0.12
        self._wick_body_ratio = 1.8
        self._sl_atr_mult = 0.2
        self._min_rr = 1.8

        self._rsi_buy_low = 28
        self._rsi_buy_high = 55
        self._rsi_sell_low = 45
        self._rsi_sell_high = 72

        self._min_volume_ratio = 0.7
        self._max_volume_ratio = 3.0
        self._extreme_break_volume = 3.5
        self._merge_distance = 8.0

        super().__init__()
        self._logger = setup_logger("strategy_failed_breakout")

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Opportunity]:
        if not features_1m or not context:
            return []

        close = self._safe_get(features_1m, "last_close", 0)
        if close <= 0:
            return []

        opportunities: List[Opportunity] = []

        support_levels = self._collect_support_levels(context)
        resistance_levels = self._collect_resistance_levels(context)

        for level_info in support_levels:
            level = float(level_info.get("price", 0))
            if level <= 0:
                continue
            if not StrategyQuality.is_near_level(close, level, self._level_tolerance_pct):
                continue
            opp = self._check_buy_failed_breakdown(features_1m, context, level_info)
            if opp:
                opportunities.append(opp)

        for level_info in resistance_levels:
            level = float(level_info.get("price", 0))
            if level <= 0:
                continue
            if not StrategyQuality.is_near_level(close, level, self._level_tolerance_pct):
                continue
            opp = self._check_sell_failed_breakout(features_1m, context, level_info)
            if opp:
                opportunities.append(opp)

        opportunities = self._dedupe_opportunities(opportunities)
        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:3]

    # ------------------------------------------------------------------
    # Levels
    # ------------------------------------------------------------------
    def _collect_support_levels(self, ctx: Dict) -> List[Dict]:
        levels: List[Dict] = []

        key_levels = ctx.get("key_levels", {})
        volume_profile = ctx.get("volume_profile", {})
        options = ctx.get("options", {})
        session_memory = ctx.get("session_memory", {})

        def add(price, source, strength=2):
            if price and price > 0:
                levels.append(
                    {
                        "price": float(price),
                        "type": "support",
                        "source": source,
                        "strength": int(strength),
                    }
                )

        add(key_levels.get("pdl"), "PDL", 3)
        add(key_levels.get("or_low"), "ORL", 2)
        add(key_levels.get("ib_low"), "IBL", 2)
        add(volume_profile.get("val"), "VAL", 2)
        add(options.get("highest_pe_oi_strike"), "PE_WALL", 3)

        for zone in key_levels.get("sr_zones", []):
            if isinstance(zone, dict) and zone.get("type") == "support":
                add(zone.get("level", 0), "SR_ZONE", zone.get("strength", 2))

        for lvl in session_memory.get("levels_defended", []):
            add(lvl, "DEFENDED", 4)

        return self._cluster_levels(levels)

    def _collect_resistance_levels(self, ctx: Dict) -> List[Dict]:
        levels: List[Dict] = []

        key_levels = ctx.get("key_levels", {})
        volume_profile = ctx.get("volume_profile", {})
        options = ctx.get("options", {})

        def add(price, source, strength=2):
            if price and price > 0:
                levels.append(
                    {
                        "price": float(price),
                        "type": "resistance",
                        "source": source,
                        "strength": int(strength),
                    }
                )

        add(key_levels.get("pdh"), "PDH", 3)
        add(key_levels.get("or_high"), "ORH", 2)
        add(key_levels.get("ib_high"), "IBH", 2)
        add(volume_profile.get("vah"), "VAH", 2)
        add(options.get("highest_ce_oi_strike"), "CE_WALL", 3)

        for zone in key_levels.get("sr_zones", []):
            if isinstance(zone, dict) and zone.get("type") == "resistance":
                add(zone.get("level", 0), "SR_ZONE", zone.get("strength", 2))

        return self._cluster_levels(levels)

    def _cluster_levels(self, levels: List[Dict]) -> List[Dict]:
        if not levels:
            return []

        levels = sorted(
            [x for x in levels if x.get("price", 0) > 0],
            key=lambda x: (x["type"], x["price"]),
        )

        clustered: List[Dict] = []
        for level in levels:
            if not clustered:
                clustered.append(
                    {
                        "price": level["price"],
                        "type": level["type"],
                        "members": [level["price"]],
                        "sources": {level["source"]},
                        "strength": int(level.get("strength", 1)),
                    }
                )
                continue

            last = clustered[-1]
            if (
                last["type"] == level["type"]
                and abs(last["price"] - level["price"]) <= self._merge_distance
            ):
                last["members"].append(level["price"])
                last["sources"].add(level["source"])
                last["strength"] = max(last["strength"], int(level.get("strength", 1)))
                last["price"] = round(sum(last["members"]) / len(last["members"]), 2)
            else:
                clustered.append(
                    {
                        "price": level["price"],
                        "type": level["type"],
                        "members": [level["price"]],
                        "sources": {level["source"]},
                        "strength": int(level.get("strength", 1)),
                    }
                )

        result: List[Dict] = []
        for c in clustered:
            result.append(
                {
                    "price": c["price"],
                    "type": c["type"],
                    "source": "|".join(sorted(c["sources"])),
                    "source_confluence": len(c["sources"]),
                    "strength": c["strength"],
                }
            )
        return result

    def _dedupe_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        if not opportunities:
            return []

        selected: List[Opportunity] = []
        for opp in opportunities:
            keep = True
            for existing in selected:
                if (
                    existing.direction == opp.direction
                    and abs(existing.entry_price - opp.entry_price) <= 8
                ):
                    keep = False
                    break
            if keep:
                selected.append(opp)
        return selected

    # ------------------------------------------------------------------
    # BUY
    # ------------------------------------------------------------------
    def _check_buy_failed_breakdown(
        self,
        f1m: Dict,
        ctx: Dict,
        level_info: Dict,
    ) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        low = self._safe_get(f1m, "low", 0)
        high = self._safe_get(f1m, "high", 0)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        last_swing_high = ctx.get("last_swing_high")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        level = float(level_info["price"])
        source = str(level_info.get("source", "UNKNOWN"))
        strength = int(level_info.get("strength", 1))
        confluence = int(level_info.get("source_confluence", 1))

        if close <= 0 or low <= 0 or atr is None or atr <= 0:
            return None

        pierce_points = round(level - low, 2)

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "near_support": StrategyQuality.is_near_level(close, level, self._level_tolerance_pct),
            "pierced_below": low < level,
            "pierce_size_ok": pierce_points >= self._min_pierce_points,
            "reclaimed_support": close > level,
            "reclaim_quality": self._reclaim_quality_buy(close, level, high, low),
            "wick_ok": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="BUY",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._min_volume_ratio, self._max_volume_ratio
            ),
            "not_breakdown_extreme": volume_ratio is None or volume_ratio < self._extreme_break_volume,
            "rsi_ok": (rsi is None) or (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "atr_ok": atr > 0,
            "macd_not_weak": (macd_hist is None) or (macd_hist > -8),
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
        if strength >= 3:
            score += 10
        if confluence >= 2:
            score += 10
        if any(x in source for x in ("PDL", "VAL", "PE_WALL", "DEFENDED")):
            score += 5
        if volume_ratio is not None and volume_ratio >= 1.5:
            score += 5
        if rsi is not None and 32 <= rsi <= 45:
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5

        thesis = (
            f"BUY failed breakdown at support {level:.2f} "
            f"({source}, confluence={confluence}, strength={strength}), "
            f"reclaim confirmed, dataQ={data_quality}"
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
    def _check_sell_failed_breakout(
        self,
        f1m: Dict,
        ctx: Dict,
        level_info: Dict,
    ) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        low = self._safe_get(f1m, "low", 0)
        high = self._safe_get(f1m, "high", 0)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        last_swing_low = ctx.get("last_swing_low")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        level = float(level_info["price"])
        source = str(level_info.get("source", "UNKNOWN"))
        strength = int(level_info.get("strength", 1))
        confluence = int(level_info.get("source_confluence", 1))

        if close <= 0 or high <= 0 or atr is None or atr <= 0:
            return None

        pierce_points = round(high - level, 2)

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "near_resistance": StrategyQuality.is_near_level(close, level, self._level_tolerance_pct),
            "pierced_above": high > level,
            "pierce_size_ok": pierce_points >= self._min_pierce_points,
            "reclaimed_below": close < level,
            "reclaim_quality": self._reclaim_quality_sell(close, level, high, low),
            "wick_ok": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="SELL",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._min_volume_ratio, self._max_volume_ratio
            ),
            "not_breakout_extreme": volume_ratio is None or volume_ratio < self._extreme_break_volume,
            "rsi_ok": (rsi is None) or (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "atr_ok": atr > 0,
            "macd_not_weak": (macd_hist is None) or (macd_hist < 8),
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
        if strength >= 3:
            score += 10
        if confluence >= 2:
            score += 10
        if any(x in source for x in ("PDH", "VAH", "CE_WALL")):
            score += 5
        if volume_ratio is not None and volume_ratio >= 1.5:
            score += 5
        if rsi is not None and 55 <= rsi <= 68:
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5

        thesis = (
            f"SELL failed breakout at resistance {level:.2f} "
            f"({source}, confluence={confluence}, strength={strength}), "
            f"reclaim confirmed, dataQ={data_quality}"
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
    print(" JUNIOR ALADDIN — Failed Breakout Reversal Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = FailedBreakoutReversalStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "FAILED_BREAKOUT_REVERSAL" and strategy.brain == "TACTICAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] BUY failed breakdown at support...")
    buy_f = {
        "last_close": 23055.0,
        "low": 23038.0,
        "high": 23060.0,
        "rsi": 40.0,
        "atr": 15.0,
        "volume_ratio": 1.8,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.60,
        "upper_wick_ratio": 0.05,
        "macd_histogram": -2.0,
    }
    buy_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "last_swing_high": 23200,
        "last_swing_low": 23050,
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
            "val": 23050,
            "vah": 23250,
        },
        "options": {
            "highest_pe_oi_strike": 23000,
            "highest_ce_oi_strike": 23400,
        },
        "session_memory": {
            "levels_defended": [23050],
        },
        "microstructure": {
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

    print("\n [Test 3] No signal without sufficient pierce...")
    no_pierce = {**buy_f, "low": 23049.0}
    r3 = strategy.safe_scan(no_pierce, context=buy_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (pierce too small)")
        passed += 1
    else:
        print(" ❌ Should block weak pierce")
        failed += 1

    print("\n [Test 4] No signal without reclaim...")
    no_reclaim = {**buy_f, "last_close": 23045.0}
    r4 = strategy.safe_scan(no_reclaim, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (not reclaimed)")
        passed += 1
    else:
        print(" ❌ Must reclaim level")
        failed += 1

    print("\n [Test 5] No signal with low volume...")
    low_vol = {**buy_f, "volume_ratio": 0.5}
    r5 = strategy.safe_scan(low_vol, context=buy_ctx)
    if len(r5) == 0:
        print(" ✅ No signal (volume too weak)")
        passed += 1
    else:
        print(" ❌ Should block low volume")
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

    print("\n [Test 7] SELL failed breakout at resistance...")
    sell_f = {
        "last_close": 23195.0,
        "high": 23215.0,
        "low": 23190.0,
        "rsi": 60.0,
        "atr": 15.0,
        "volume_ratio": 1.7,
        "candle_body_ratio": 0.10,
        "upper_wick_ratio": 0.55,
        "lower_wick_ratio": 0.05,
        "macd_histogram": 2.0,
    }
    sell_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "NEUTRAL",
        "last_swing_low": 23100,
        "last_swing_high": 23200,
        "key_levels": {
            "pdh": 23200,
            "sr_zones": [
                {"level": 23200, "strength": 3, "type": "resistance"},
            ],
        },
        "volume_profile": {
            "val": 23120,
            "vah": 23280,
        },
        "options": {
            "highest_ce_oi_strike": 23200,
            "highest_pe_oi_strike": 23050,
        },
        "microstructure": {
            "spread_zscore": 0.4,
        },
        "data_quality_score": 90,
    }
    r7 = strategy.safe_scan(sell_f, context=sell_ctx)
    if len(r7) >= 1 and r7[0].direction == "SELL":
        print(f" ✅ SELL signal @{r7[0].entry_price}, RR={r7[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No SELL signal")
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

    print("\n [Test 9] No signal during LAST_MINUTES...")
    last_ctx = {**buy_ctx, "session_phase": "LAST_MINUTES"}
    r9 = strategy.safe_scan(buy_f, context=last_ctx)
    if len(r9) == 0:
        print(" ✅ No signal (LAST_MINUTES blocked)")
        passed += 1
    else:
        print(" ❌ Should block LAST_MINUTES")
        failed += 1

    print("\n [Test 10] Empty data...")
    r10 = strategy.safe_scan({})
    if len(r10) == 0:
        print(" ✅ No crash with empty input")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 11] Stats tracking...")
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
        print("\n 🎉 Failed Breakout Reversal Strategy (Hardened) working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()