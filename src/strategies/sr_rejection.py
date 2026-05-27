"""
Junior Aladdin - S/R Rejection Strategy (Clustered-Level Strong Version)
========================================================================
PURPOSE:
Trade high-quality reversals from proven support/resistance levels.

This version fixes the core architectural issue:
- raw levels from many sources are clustered
- source confluence is preserved
- cluster strength boosts score
- cluster strength does NOT wrongly block the setup

BUY CONDITIONS:
1. Price is near clustered support
2. Rejection wick confirms defense
3. Close holds support area
4. RSI is supportive but not exhausted
5. Volume confirms but is not breakdown-extreme
6. Regime supports bounce
7. Session allows execution
8. Narrative not strongly against
9. Data quality and spread acceptable
10. Enough room to target

SELL CONDITIONS:
Mirror of BUY.

CONNECTS TO:
- StrategyBase / Opportunity
- key_levels / options / volume_profile / session_memory
- context regime / session / narrative / microstructure
"""

from typing import Dict, List, Optional, Set

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.utils.logger import setup_logger


class SRRejectionStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "SR_REJECTION"

    @property
    def brain(self) -> str:
        return "STRUCTURAL"

    def __init__(self):
        self._level_tolerance_pct = 0.08
        self._cluster_distance = 8.0

        self._wick_body_ratio = 2.0
        self._support_hold_atr_frac = 0.15
        self._resistance_hold_atr_frac = 0.15

        self._rsi_buy_low = 28
        self._rsi_buy_high = 48
        self._rsi_sell_low = 52
        self._rsi_sell_high = 72

        self._vol_min = 0.6
        self._vol_max = 2.2
        self._too_high_break_volume = 2.8

        self._sl_atr_mult = 0.2
        self._min_rr = 1.5

        super().__init__()
        self._logger = setup_logger("strategy_sr_rejection")

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

        raw_levels = self._collect_levels(context)
        if not raw_levels:
            return []

        clustered_levels = self._cluster_levels(raw_levels)

        opportunities: List[Opportunity] = []

        for cluster in clustered_levels:
            level_type = cluster["type"]
            level_price = cluster["price"]

            if not self._is_near_level(close, level_price):
                continue

            if level_type == "support":
                opp = self._check_buy_at_support(features_1m, context, cluster)
                if opp:
                    opportunities.append(opp)

            elif level_type == "resistance":
                opp = self._check_sell_at_resistance(features_1m, context, cluster)
                if opp:
                    opportunities.append(opp)

        opportunities = self._dedupe_opportunities(opportunities)
        opportunities.sort(key=lambda x: (x.raw_score, x.risk_reward), reverse=True)
        return opportunities[:3]

    def _collect_levels(self, ctx: Dict) -> List[Dict]:
        levels: List[Dict] = []

        key_levels = ctx.get("key_levels", {})
        options = ctx.get("options", {})
        volume_profile = ctx.get("volume_profile", {})
        session_memory = ctx.get("session_memory", {})

        pdh = key_levels.get("pdh", 0)
        pdl = key_levels.get("pdl", 0)
        if pdh > 0:
            levels.append({"price": float(pdh), "type": "resistance", "source": "PDH"})
        if pdl > 0:
            levels.append({"price": float(pdl), "type": "support", "source": "PDL"})

        for name, level_type in (
            ("or_high", "resistance"),
            ("or_low", "support"),
            ("ib_high", "resistance"),
            ("ib_low", "support"),
        ):
            value = key_levels.get(name, 0)
            if value and value > 0:
                levels.append({"price": float(value), "type": level_type, "source": name.upper()})

        for zone in key_levels.get("sr_zones", []):
            if isinstance(zone, dict) and zone.get("level", 0) > 0:
                levels.append({
                    "price": float(zone["level"]),
                    "type": zone.get("type", "support"),
                    "source": "SR_ZONE",
                })

        ce_wall = options.get("highest_ce_oi_strike", 0)
        pe_wall = options.get("highest_pe_oi_strike", 0)
        if ce_wall > 0:
            levels.append({"price": float(ce_wall), "type": "resistance", "source": "CE_WALL"})
        if pe_wall > 0:
            levels.append({"price": float(pe_wall), "type": "support", "source": "PE_WALL"})

        vah = volume_profile.get("vah", 0)
        val_price = volume_profile.get("val", 0)
        if vah > 0:
            levels.append({"price": float(vah), "type": "resistance", "source": "VAH"})
        if val_price > 0:
            levels.append({"price": float(val_price), "type": "support", "source": "VAL"})

        for defended in session_memory.get("levels_defended", []):
            if defended and defended > 0:
                levels.append({"price": float(defended), "type": "support", "source": "DEFENDED"})

        return levels

    def _cluster_levels(self, levels: List[Dict]) -> List[Dict]:
        """
        Cluster nearby levels but preserve source confluence.
        """
        if not levels:
            return []

        cleaned = [x for x in levels if x.get("price", 0) > 0]
        cleaned.sort(key=lambda x: (x["type"], x["price"]))

        support_levels = [x for x in cleaned if x["type"] == "support"]
        resistance_levels = [x for x in cleaned if x["type"] == "resistance"]

        return self._cluster_by_type(support_levels, "support") + self._cluster_by_type(
            resistance_levels, "resistance"
        )

    def _cluster_by_type(self, levels: List[Dict], level_type: str) -> List[Dict]:
        if not levels:
            return []

        clusters: List[Dict] = []

        for lvl in sorted(levels, key=lambda x: x["price"]):
            if not clusters:
                clusters.append({
                    "price": lvl["price"],
                    "type": level_type,
                    "members": [lvl["price"]],
                    "sources": {lvl["source"]},
                    "source_confluence": 1,
                })
                continue

            last = clusters[-1]
            if abs(last["price"] - lvl["price"]) <= self._cluster_distance:
                last["members"].append(lvl["price"])
                last["sources"].add(lvl["source"])
                last["source_confluence"] = len(last["sources"])
                last["price"] = round(sum(last["members"]) / len(last["members"]), 2)
            else:
                clusters.append({
                    "price": lvl["price"],
                    "type": level_type,
                    "members": [lvl["price"]],
                    "sources": {lvl["source"]},
                    "source_confluence": 1,
                })

        # convert sets to strings
        for c in clusters:
            c["source"] = "|".join(sorted(c["sources"]))
        return clusters

    def _dedupe_opportunities(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        if not opportunities:
            return []

        selected: List[Opportunity] = []
        for opp in sorted(opportunities, key=lambda x: (x.raw_score, x.risk_reward), reverse=True):
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

    def _is_near_level(self, price: float, level: float) -> bool:
        if price <= 0 or level <= 0:
            return False
        return abs(price - level) / level * 100 <= self._level_tolerance_pct

    def _support_hold(self, close: float, level: float, atr: float) -> bool:
        tolerance = max(1.0, atr * self._support_hold_atr_frac)
        return close >= level - tolerance

    def _resistance_hold(self, close: float, level: float, atr: float) -> bool:
        tolerance = max(1.0, atr * self._resistance_hold_atr_frac)
        return close <= level + tolerance

    def _wick_rejection_buy(self, lower_wick: float, body_ratio: float) -> bool:
        if lower_wick <= 0:
            return False
        if body_ratio <= 0.12:
            return lower_wick >= 0.20
        return lower_wick >= body_ratio * self._wick_body_ratio

    def _wick_rejection_sell(self, upper_wick: float, body_ratio: float) -> bool:
        if upper_wick <= 0:
            return False
        if body_ratio <= 0.12:
            return upper_wick >= 0.20
        return upper_wick >= body_ratio * self._wick_body_ratio

    def _base_quality_ok(
        self,
        regime: str,
        session: str,
        narrative: str,
        data_quality: float,
        spread_zscore: Optional[float],
    ) -> bool:
        if data_quality < 60:
            return False
        if spread_zscore is not None and spread_zscore >= 2.2:
            return False
        if regime in ("CHOP", "EVENT"):
            return False
        if narrative in ("EVENT_RISK",):
            return False
        if session in ("PRE_MARKET", "OPENING_AUCTION", "POST_MARKET", "LAST_MINUTES"):
            return False
        return True

    def _check_buy_at_support(self, f1m: Dict, ctx: Dict, cluster: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        low = self._safe_get(f1m, "low", close)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        lower_wick = self._safe_get(f1m, "lower_wick_ratio", 0)
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_high = ctx.get("last_swing_high")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        level = float(cluster["price"])
        confluence = int(cluster.get("source_confluence", 1))
        source = str(cluster.get("source", "UNKNOWN"))

        if close <= 0 or atr is None or atr <= 0:
            return None

        conditions = {
            "base_quality_ok": self._base_quality_ok(regime, session, narrative, data_quality, spread_zscore),
            "near_level": self._is_near_level(close, level),
            "wick_rejection": self._wick_rejection_buy(lower_wick, body_ratio),
            "support_hold": self._support_hold(close, level, atr),
            "rsi_ok": rsi is not None and self._rsi_buy_low <= rsi <= self._rsi_buy_high,
            "volume_ok": volume_ratio is not None and self._vol_min <= volume_ratio <= self._vol_max,
            "not_breakdown_extreme": volume_ratio is None or volume_ratio < self._too_high_break_volume,
            "macd_not_collapsing": macd_hist is None or macd_hist > -6,
            "mtf_not_strongly_bearish": weighted_mtf > -3.0,
        }

        if not self._all_conditions(conditions):
            return None

        sl = round(min(low, level) - atr * self._sl_atr_mult - 1, 2)
        risk = abs(close - sl)
        if risk <= 0:
            return None

        min_target = close + risk * self._min_rr
        target = min_target
        if last_swing_high and last_swing_high > close:
            target = max(min_target, min(last_swing_high, close + risk * 3.0))
        target = round(target, 2)

        rr = (target - close) / risk
        if rr < self._min_rr:
            return None

        score = 60
        if confluence >= 2:
            score += 10
        if confluence >= 3:
            score += 5
        if any(x in source for x in ("PDL", "PE_WALL", "DEFENDED", "VAL")):
            score += 10
        if rsi is not None and 32 <= rsi <= 42:
            score += 5
        if weighted_mtf > 0:
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if volume_ratio is not None and 0.8 <= volume_ratio <= 1.5:
            score += 5

        thesis = (
            f"BUY rejection at support {level:.2f} ({source}, confluence={confluence}) "
            f"RSI={rsi}, RR={round(rr, 2)}"
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

    def _check_sell_at_resistance(self, f1m: Dict, ctx: Dict, cluster: Dict) -> Optional[Opportunity]:
        close = self._safe_get(f1m, "last_close", 0)
        high = self._safe_get(f1m, "high", close)
        rsi = self._safe_get(f1m, "rsi")
        atr = self._safe_get(f1m, "atr")
        volume_ratio = self._safe_get(f1m, "volume_ratio")
        body_ratio = self._safe_get(f1m, "candle_body_ratio", 0)
        upper_wick = self._safe_get(f1m, "upper_wick_ratio", 0)
        macd_hist = self._safe_get(f1m, "macd_histogram")

        regime = ctx.get("regime", "UNKNOWN")
        session = ctx.get("session_phase", "")
        narrative = ctx.get("narrative_label", "NEUTRAL")
        weighted_mtf = ctx.get("weighted_mtf", 0)
        last_swing_low = ctx.get("last_swing_low")
        data_quality = float(ctx.get("data_quality_score", 100) or 0)
        spread_zscore = ctx.get("microstructure", {}).get("spread_zscore")

        level = float(cluster["price"])
        confluence = int(cluster.get("source_confluence", 1))
        source = str(cluster.get("source", "UNKNOWN"))

        if close <= 0 or atr is None or atr <= 0:
            return None

        conditions = {
            "base_quality_ok": self._base_quality_ok(regime, session, narrative, data_quality, spread_zscore),
            "near_level": self._is_near_level(close, level),
            "wick_rejection": self._wick_rejection_sell(upper_wick, body_ratio),
            "resistance_hold": self._resistance_hold(close, level, atr),
            "rsi_ok": rsi is not None and self._rsi_sell_low <= rsi <= self._rsi_sell_high,
            "volume_ok": volume_ratio is not None and self._vol_min <= volume_ratio <= self._vol_max,
            "not_breakout_extreme": volume_ratio is None or volume_ratio < self._too_high_break_volume,
            "macd_not_exploding": macd_hist is None or macd_hist < 6,
            "mtf_not_strongly_bullish": weighted_mtf < 3.0,
        }

        if not self._all_conditions(conditions):
            return None

        sl = round(max(high, level) + atr * self._sl_atr_mult + 1, 2)
        risk = abs(sl - close)
        if risk <= 0:
            return None

        min_target = close - risk * self._min_rr
        target = min_target
        if last_swing_low and last_swing_low < close:
            target = min(min_target, max(last_swing_low, close - risk * 3.0))
        target = round(target, 2)

        rr = (close - target) / risk
        if rr < self._min_rr:
            return None

        score = 60
        if confluence >= 2:
            score += 10
        if confluence >= 3:
            score += 5
        if any(x in source for x in ("PDH", "CE_WALL", "VAH")):
            score += 10
        if rsi is not None and 58 <= rsi <= 68:
            score += 5
        if weighted_mtf < 0:
            score += 5
        if session in ("GOLDEN_AM", "GOLDEN_PM", "INITIAL_BALANCE"):
            score += 5
        if regime in ("RANGE", "TRENDING"):
            score += 5
        if volume_ratio is not None and 0.8 <= volume_ratio <= 1.5:
            score += 5

        thesis = (
            f"SELL rejection at resistance {level:.2f} ({source}, confluence={confluence}) "
            f"RSI={rsi}, RR={round(rr, 2)}"
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
    print(" JUNIOR ALADDIN — S/R Rejection Strategy Test (Clustered-Level Fixed)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = SRRejectionStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "SR_REJECTION" and strategy.brain == "STRUCTURAL":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong name/brain")
        failed += 1

    print("\n [Test 2] BUY at PDL support...")
    buy_features = {
        "last_close": 23050.5,
        "high": 23058.0,
        "low": 23045.0,
        "rsi": 35.0,
        "atr": 15.0,
        "volume_ratio": 1.2,
        "candle_body_ratio": 0.12,
        "lower_wick_ratio": 0.55,
        "upper_wick_ratio": 0.08,
        "macd_histogram": -2.0,
    }
    buy_context = {
        "regime": "RANGE",
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "weighted_mtf": 1.0,
        "last_swing_high": 23200,
        "data_quality_score": 85,
        "microstructure": {"spread_zscore": 0.6},
        "key_levels": {
            "pdh": 23300,
            "pdl": 23050,
            "pdc": 23180,
            "or_high": 23200,
            "or_low": 23100,
            "ib_high": 23250,
            "ib_low": 23080,
            "sr_zones": [
                {"level": 23050, "strength": 3, "type": "support"},
                {"level": 23300, "strength": 2, "type": "resistance"},
            ],
        },
        "options": {"highest_ce_oi_strike": 23400, "highest_pe_oi_strike": 23000},
        "volume_profile": {"poc": 23150, "vah": 23250, "val": 23050},
        "session_memory": {"levels_defended": [23050], "failed_breakouts": []},
    }
    r2 = strategy.safe_scan(buy_features, context=buy_context)
    buys = [x for x in r2 if x.direction == "BUY"]
    if buys:
        print(f" ✅ BUY signal @{buys[0].entry_price}, RR={buys[0].risk_reward}")
        passed += 1
    else:
        print(" ❌ No BUY signal")
        failed += 1

    print("\n [Test 3] No signal with low data quality...")
    bad_ctx = {**buy_context, "data_quality_score": 40}
    r3 = strategy.safe_scan(buy_features, context=bad_ctx)
    if len(r3) == 0:
        print(" ✅ No signal (low data quality)")
        passed += 1
    else:
        print(" ❌ Should block low quality")
        failed += 1

    print("\n [Test 4] No signal with no wick...")
    no_wick = {**buy_features, "lower_wick_ratio": 0.05}
    r4 = strategy.safe_scan(no_wick, context=buy_context)
    if len(r4) == 0:
        print(" ✅ No signal (no rejection wick)")
        passed += 1
    else:
        print(" ❌ Should block no rejection wick")
        failed += 1

    print("\n [Test 5] SELL at PDH resistance...")
    sell_features = {
        "last_close": 23299.0,
        "high": 23304.0,
        "low": 23292.0,
        "rsi": 65.0,
        "atr": 15.0,
        "volume_ratio": 1.0,
        "candle_body_ratio": 0.10,
        "upper_wick_ratio": 0.50,
        "lower_wick_ratio": 0.05,
        "macd_histogram": 2.0,
    }
    sell_context = {
        **buy_context,
        "last_swing_low": 23100,
    }
    r5 = strategy.safe_scan(sell_features, context=sell_context)
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
        print("\n 🎉 S/R Rejection Strategy (Clustered-Level Fixed) working perfectly!")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()