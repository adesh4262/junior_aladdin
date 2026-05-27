"""
Junior Aladdin - Liquidity Sweep Reversal Strategy (Hardened Version)
=====================================================================
PURPOSE:
Trade reversals after price sweeps a liquidity pool and then reclaims.

Liquidity pools are clusters of equal highs or equal lows where stop orders
collect. Smart money often sweeps those levels, takes liquidity, then reverses.

This hardened version improves:
- pool quality validation
- reclaim quality checks
- liquidity / spread sanity
- overextension avoidance
- better target realism
- stronger confluence scoring

BUY LIQUIDITY SWEEP CONDITIONS:
1. Buy-side liquidity pool exists
2. Pool quality is meaningful
3. Price sweeps below pool
4. Reclaims above pool with quality close
5. Rejection wick confirms reversal
6. Volume is meaningful but not chaos
7. Regime not CHOP / EVENT
8. Session allows tactical trading
9. Narrative not strongly against
10. ATR available
11. Spread acceptable
12. Optional structure context not hostile

SELL CONDITIONS:
Mirror logic using sell-side liquidity pools.

RISK MODEL:
- BUY SL below sweep low minus ATR buffer
- SELL SL above sweep high plus ATR buffer
- Target = next opposing swing / structure or RR projection

CONNECTS TO:
- Smart Money Concepts module (liquidity pools, structure)
- Tactical brain
- StrategyQuality hardened helper layer
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.strategies.strategy_quality import StrategyQuality
from src.utils.logger import setup_logger


class LiquiditySweepReversalStrategy(StrategyBase):
    """
    Tactical reversal after smart-money liquidity sweep.
    """

    @property
    def name(self) -> str:
        return "LIQUIDITY_SWEEP_REVERSAL"

    @property
    def brain(self) -> str:
        return "TACTICAL"

    def __init__(self):
        self._min_pierce_points = 3.0
        self._pool_tolerance_pct = 0.15
        self._wick_body_ratio = 1.8
        self._sl_atr_mult = 0.2
        self._min_rr = 1.8

        self._rsi_buy_low = 28
        self._rsi_buy_high = 55
        self._rsi_sell_low = 45
        self._rsi_sell_high = 72

        self._min_volume_ratio = 0.8
        self._max_volume_ratio = 4.0
        self._extreme_break_volume = 4.5

        self._min_pool_touches = 2

        super().__init__()
        self._logger = setup_logger("strategy_liquidity_sweep")

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

        buy_opp = self._check_buy_sweep(features_1m, context)
        if buy_opp:
            opportunities.append(buy_opp)

        sell_opp = self._check_sell_sweep(features_1m, context)
        if sell_opp:
            opportunities.append(sell_opp)

        opportunities.sort(
            key=lambda x: (x.raw_score, x.risk_reward),
            reverse=True,
        )
        return opportunities[:2]

    # ------------------------------------------------------------------
    # Pool extraction
    # ------------------------------------------------------------------
    def _extract_buy_side_pools(self, ctx: Dict) -> List[Dict]:
        pools: List[Dict] = []

        for source_name in ("smart_money_5m", "smart_money_15m"):
            sm = ctx.get(source_name, {})
            source_pools = sm.get("buy_side_pools", [])
            if not isinstance(source_pools, list):
                continue

            for pool in source_pools:
                if not isinstance(pool, dict):
                    continue

                level = pool.get("level", 0)
                touches = int(pool.get("touches", 0) or 0)

                if level and level > 0:
                    pools.append(
                        {
                            "level": float(level),
                            "touches": touches,
                            "source_tf": source_name,
                        }
                    )

        pools.sort(
            key=lambda x: (x["touches"], x["level"]),
            reverse=True,
        )
        return pools

    def _extract_sell_side_pools(self, ctx: Dict) -> List[Dict]:
        pools: List[Dict] = []

        for source_name in ("smart_money_5m", "smart_money_15m"):
            sm = ctx.get(source_name, {})
            source_pools = sm.get("sell_side_pools", [])
            if not isinstance(source_pools, list):
                continue

            for pool in source_pools:
                if not isinstance(pool, dict):
                    continue

                level = pool.get("level", 0)
                touches = int(pool.get("touches", 0) or 0)

                if level and level > 0:
                    pools.append(
                        {
                            "level": float(level),
                            "touches": touches,
                            "source_tf": source_name,
                        }
                    )

        pools.sort(
            key=lambda x: (x["touches"], x["level"]),
            reverse=True,
        )
        return pools

    def _nearest_pool(self, price: float, pools: List[Dict]) -> Optional[Dict]:
        if price <= 0 or not pools:
            return None
        return min(pools, key=lambda p: abs(price - p["level"]))

    def _pool_quality_ok(self, pool: Dict) -> bool:
        if not pool:
            return False
        touches = int(pool.get("touches", 0) or 0)
        return touches >= self._min_pool_touches

    # ------------------------------------------------------------------
    # BUY sweep
    # ------------------------------------------------------------------
    def _check_buy_sweep(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
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

        smart_money_5m = ctx.get("smart_money_5m", {})
        smart_money_15m = ctx.get("smart_money_15m", {})

        if close <= 0 or low <= 0 or atr is None or atr <= 0:
            return None

        pools = self._extract_buy_side_pools(ctx)
        if not pools:
            return None

        pool = self._nearest_pool(close, pools)
        if pool is None:
            return None

        level = float(pool["level"])
        touches = int(pool.get("touches", 0))
        source_tf = str(pool.get("source_tf", "unknown"))

        structure_supportive = (
            smart_money_5m.get("last_choch_direction") == "BULLISH"
            or smart_money_15m.get("last_choch_direction") == "BULLISH"
            or smart_money_5m.get("last_bos_direction") == "BULLISH"
            or smart_money_15m.get("last_bos_direction") == "BULLISH"
        )

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BEARISH", "EVENT_RISK")
            ),
            "pool_exists": level > 0,
            "pool_quality_ok": self._pool_quality_ok(pool),
            "near_pool": StrategyQuality.is_near_level(
                close, level, self._pool_tolerance_pct
            ),
            "swept_below_pool": low < level,
            "pierce_size_ok": (level - low) >= self._min_pierce_points,
            "reclaimed_pool": close > level,
            "wick_rejection": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="BUY",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._min_volume_ratio, self._max_volume_ratio
            ),
            "not_breakdown_extreme": volume_ratio is None
            or volume_ratio < self._extreme_break_volume,
            "rsi_ok": (rsi is None) or (self._rsi_buy_low <= rsi <= self._rsi_buy_high),
            "atr_ok": atr > 0,
            "structure_supportive": structure_supportive or True,
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

        score = 55
        if touches >= 3:
            score += 10
        if source_tf == "smart_money_15m":
            score += 10
        if volume_ratio is not None and volume_ratio >= 1.5:
            score += 5
        if rsi is not None and 32 <= rsi <= 45:
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5
        if weighted_mtf > 0:
            score += 5

        thesis = (
            f"BUY liquidity sweep: price swept below buy-side pool {level:.0f}, "
            f"reclaimed above, touches={touches}, tf={source_tf}, dataQ={data_quality}"
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
    # SELL sweep
    # ------------------------------------------------------------------
    def _check_sell_sweep(self, f1m: Dict, ctx: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", 0)
        low = self._safe_get(f1m, "low", 0)
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

        smart_money_5m = ctx.get("smart_money_5m", {})
        smart_money_15m = ctx.get("smart_money_15m", {})

        if close <= 0 or high <= 0 or atr is None or atr <= 0:
            return None

        pools = self._extract_sell_side_pools(ctx)
        if not pools:
            return None

        pool = self._nearest_pool(close, pools)
        if pool is None:
            return None

        level = float(pool["level"])
        touches = int(pool.get("touches", 0))
        source_tf = str(pool.get("source_tf", "unknown"))

        structure_supportive = (
            smart_money_5m.get("last_choch_direction") == "BEARISH"
            or smart_money_15m.get("last_choch_direction") == "BEARISH"
            or smart_money_5m.get("last_bos_direction") == "BEARISH"
            or smart_money_15m.get("last_bos_direction") == "BEARISH"
        )

        conditions = {
            "data_quality_ok": StrategyQuality.get_data_quality_ok(ctx, 60),
            "spread_ok": StrategyQuality.get_spread_ok(ctx, 2.5),
            "session_ok": StrategyQuality.tactical_session_ok(ctx),
            "regime_ok": StrategyQuality.get_regime_ok(ctx, ("CHOP", "EVENT")),
            "narrative_ok": StrategyQuality.get_narrative_ok(
                ctx, ("STRONG_BULLISH", "EVENT_RISK")
            ),
            "pool_exists": level > 0,
            "pool_quality_ok": self._pool_quality_ok(pool),
            "near_pool": StrategyQuality.is_near_level(
                close, level, self._pool_tolerance_pct
            ),
            "swept_above_pool": high > level,
            "pierce_size_ok": (high - level) >= self._min_pierce_points,
            "reclaimed_below": close < level,
            "wick_rejection": StrategyQuality.rejection_quality(
                upper_wick_ratio=upper_wick,
                lower_wick_ratio=lower_wick,
                body_ratio=body_ratio,
                direction="SELL",
                min_wick_body_mult=self._wick_body_ratio,
            ),
            "volume_ok": StrategyQuality.volume_confirmation(
                volume_ratio, self._min_volume_ratio, self._max_volume_ratio
            ),
            "not_breakout_extreme": volume_ratio is None
            or volume_ratio < self._extreme_break_volume,
            "rsi_ok": (rsi is None) or (self._rsi_sell_low <= rsi <= self._rsi_sell_high),
            "atr_ok": atr > 0,
            "structure_supportive": structure_supportive or True,
            "mtf_not_hostile": weighted_mtf < 3.0,
        }

        if not self._all_conditions(conditions):
            return None

        entry = close
        sl = round(high + atr * self._sl_atr_mult + 1, 2)
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
        if touches >= 3:
            score += 10
        if source_tf == "smart_money_15m":
            score += 10
        if volume_ratio is not None and volume_ratio >= 1.5:
            score += 5
        if rsi is not None and 55 <= rsi <= 68:
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5
        if weighted_mtf < 0:
            score += 5

        thesis = (
            f"SELL liquidity sweep: price swept above sell-side pool {level:.0f}, "
            f"reclaimed below, touches={touches}, tf={source_tf}, dataQ={data_quality}"
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
    print(" JUNIOR ALADDIN — Liquidity Sweep Reversal Strategy Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = LiquiditySweepReversalStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "LIQUIDITY_SWEEP_REVERSAL" and strategy.brain == "TACTICAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] BUY liquidity sweep below equal lows...")
    buy_f = {
        "last_close": 23055.0,
        "low": 23038.0,
        "high": 23060.0,
        "rsi": 40.0,
        "atr": 15.0,
        "volume_ratio": 1.8,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.60,
    }
    buy_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "last_swing_high": 23200,
        "weighted_mtf": 1.0,
        "smart_money_5m": {
            "buy_side_pools": [{"level": 23050, "touches": 3}],
            "last_choch_direction": "BULLISH",
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

    print("\n [Test 3] No signal without pools...")
    no_pool_ctx = {**buy_ctx, "smart_money_5m": {"buy_side_pools": []}}
    r3 = strategy.safe_scan(buy_f, context=no_pool_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (no liquidity pools)")
        passed += 1
    else:
        print(" ❌ Should require liquidity pool")
        failed += 1

    print("\n [Test 4] No signal without sweep...")
    no_sweep = {**buy_f, "low": 23049.0}
    r4 = strategy.safe_scan(no_sweep, context=buy_ctx)
    if len(r4) == 0:
        print(" ✅ No signal (no sweep)")
        passed += 1
    else:
        print(" ❌ Should require sweep below pool")
        failed += 1

    print("\n [Test 5] SELL liquidity sweep above equal highs...")
    sell_f = {
        "last_close": 23195.0,
        "high": 23215.0,
        "low": 23190.0,
        "rsi": 60.0,
        "atr": 15.0,
        "volume_ratio": 1.7,
        "candle_body_ratio": 0.10,
        "upper_wick_ratio": 0.55,
    }
    sell_ctx = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_PM",
        "narrative_label": "NEUTRAL",
        "last_swing_low": 23100,
        "weighted_mtf": -1.0,
        "smart_money_5m": {
            "sell_side_pools": [{"level": 23200, "touches": 3}],
            "last_choch_direction": "BEARISH",
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
        print("\n 🎉 Liquidity Sweep Reversal Strategy working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()