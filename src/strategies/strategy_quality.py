"""
Junior Aladdin - Shared Strategy Quality Filters
===============================================
PURPOSE:
Provide production-grade shared validation logic for all strategies.

This module hardens every strategy by centralizing:
- location quality checks
- anti-chase / anti-late-entry logic
- spread / liquidity sanity
- data quality gating
- reclaim / breakout body quality
- support/resistance proximity helpers
- duplicate-level handling helpers
- overextension filters
- tactical and structural session gating

WHY THIS EXISTS:
Passing unit tests is not enough. Real-market failures often come from:
- marginal signals near levels
- wide spread / poor liquidity
- choppy or overextended conditions
- weak closes beyond levels
- duplicate overlapping signals

This module gives strategies a professional-grade set of reusable guards.

CONNECTS TO:
- All strategies
- Trap detection / scoring later
- Brain / context / feature engine outputs
"""

from typing import Dict, List, Optional, Tuple


class StrategyQuality:
    """
    Shared static helpers for hardening strategy logic.
    """

    @staticmethod
    def get_data_quality_ok(context: Dict, min_score: float = 60.0) -> bool:
        score = float(context.get("data_quality_score", 100.0) or 0.0)
        return score >= min_score

    @staticmethod
    def get_spread_ok(context: Dict, max_zscore: float = 2.0) -> bool:
        micro = context.get("microstructure", {})
        spread_z = micro.get("spread_zscore")
        return spread_z is None or spread_z < max_zscore

    @staticmethod
    def get_session_ok(
        context: Dict,
        blocked_sessions: Optional[Tuple[str, ...]] = None,
    ) -> bool:
        if blocked_sessions is None:
            blocked_sessions = (
                "PRE_MARKET",
                "OPENING_AUCTION",
                "OR_FORMATION",
                "LAST_MINUTES",
                "POST_MARKET",
            )
        session = context.get("session_phase", "")
        return session not in blocked_sessions

    @staticmethod
    def get_narrative_ok(
        context: Dict,
        disallowed_labels: Optional[Tuple[str, ...]] = None,
    ) -> bool:
        if disallowed_labels is None:
            disallowed_labels = ("EVENT_RISK",)
        label = context.get("narrative_label", "NEUTRAL")
        return label not in disallowed_labels

    @staticmethod
    def get_regime_ok(
        context: Dict,
        blocked_regimes: Optional[Tuple[str, ...]] = None,
    ) -> bool:
        if blocked_regimes is None:
            blocked_regimes = ("CHOP",)
        regime = context.get("regime", "UNKNOWN")
        return regime not in blocked_regimes

    @staticmethod
    def is_near_level(price: float, level: float, tolerance_pct: float) -> bool:
        if price <= 0 or level <= 0:
            return False
        return abs(price - level) / level * 100 <= tolerance_pct

    @staticmethod
    def nearest_level(price: float, levels: List[float]) -> Optional[float]:
        cleaned = [float(x) for x in levels if x and x > 0]
        if price <= 0 or not cleaned:
            return None
        return min(cleaned, key=lambda x: abs(price - x))

    @staticmethod
    def dedupe_levels(
        levels: List[Dict],
        merge_distance: float = 8.0,
    ) -> List[Dict]:
        """
        Merge overlapping levels into stronger representative levels.
        """
        if not levels:
            return []

        cleaned = [x for x in levels if isinstance(x, dict) and x.get("price", 0) > 0]
        if not cleaned:
            return []

        cleaned.sort(key=lambda x: x["price"])
        merged: List[Dict] = []

        for level in cleaned:
            if not merged:
                merged.append(level.copy())
                continue

            last = merged[-1]
            if abs(level["price"] - last["price"]) <= merge_distance:
                total_strength = last.get("strength", 1) + level.get("strength", 1)
                avg_price = (
                    last["price"] * last.get("strength", 1)
                    + level["price"] * level.get("strength", 1)
                ) / total_strength
                last["price"] = round(avg_price, 2)
                last["strength"] = total_strength
                sources = set(str(last.get("source", "")).split("|"))
                sources.add(str(level.get("source", "")))
                last["source"] = "|".join(sorted(s for s in sources if s))
            else:
                merged.append(level.copy())

        return merged

    @staticmethod
    def rejection_quality(
        upper_wick_ratio: float,
        lower_wick_ratio: float,
        body_ratio: float,
        direction: str,
        min_wick_body_mult: float = 1.5,
    ) -> bool:
        """
        Check rejection candle quality for the intended direction.
        """
        if body_ratio < 0:
            return False

        if direction.upper() == "BUY":
            if body_ratio < 0.12:
                return lower_wick_ratio > 0.25
            return lower_wick_ratio >= body_ratio * min_wick_body_mult

        if direction.upper() == "SELL":
            if body_ratio < 0.12:
                return upper_wick_ratio > 0.25
            return upper_wick_ratio >= body_ratio * min_wick_body_mult

        return False

    @staticmethod
    def breakout_close_quality(
        close: float,
        level: float,
        candle_high: Optional[float],
        candle_low: Optional[float],
        direction: str,
        min_body_close_fraction: float = 0.2,
    ) -> bool:
        """
        Ensure breakout/reclaim is not just a marginal tick beyond level.
        """
        if close <= 0 or level <= 0 or candle_high is None or candle_low is None:
            return False

        rng = candle_high - candle_low
        if rng <= 0:
            return False

        if direction.upper() == "BUY":
            if close <= level:
                return False
            return (close - level) / rng >= min_body_close_fraction

        if direction.upper() == "SELL":
            if close >= level:
                return False
            return (level - close) / rng >= min_body_close_fraction

        return False

    @staticmethod
    def overextension_filter(
        price_vs_vwap_pct: Optional[float],
        rsi: Optional[float],
        direction: str,
    ) -> bool:
        """
        Avoid chasing already overextended moves.
        """
        pvp = abs(price_vs_vwap_pct or 0.0)

        if direction.upper() == "BUY":
            if pvp > 1.2:
                return False
            if rsi is not None and rsi >= 78:
                return False
            return True

        if direction.upper() == "SELL":
            if pvp > 1.2:
                return False
            if rsi is not None and rsi <= 22:
                return False
            return True

        return True

    @staticmethod
    def volume_confirmation(
        volume_ratio: Optional[float],
        min_ratio: float,
        max_ratio: float,
    ) -> bool:
        if volume_ratio is None:
            return False
        return min_ratio <= volume_ratio <= max_ratio

    @staticmethod
    def rr_ok(entry: float, sl: float, target: float, min_rr: float = 1.0) -> bool:
        risk = abs(entry - sl)
        reward = abs(target - entry)
        if risk <= 0:
            return False
        return (reward / risk) >= min_rr

    @staticmethod
    def tactical_session_ok(context: Dict) -> bool:
        session = context.get("session_phase", "")
        return session not in (
            "PRE_MARKET",
            "OPENING_AUCTION",
            "OR_FORMATION",
            "LUNCH_LULL",
            "LAST_MINUTES",
            "POST_MARKET",
        )

    @staticmethod
    def structural_session_ok(context: Dict) -> bool:
        session = context.get("session_phase", "")
        return session in (
            "INITIAL_BALANCE",
            "GOLDEN_AM",
            "GOLDEN_PM",
            "CLOSING_SESSION",
        )

    @staticmethod
    def event_overlay_block(context: Dict) -> bool:
        return context.get("narrative_label") == "EVENT_RISK"

    @staticmethod
    def soft_direction_filter(
        weighted_mtf: float,
        direction: str,
        threshold: float = 0.0,
    ) -> bool:
        if direction.upper() == "BUY":
            return weighted_mtf >= threshold
        if direction.upper() == "SELL":
            return weighted_mtf <= -threshold
        return True

    @staticmethod
    def hard_direction_filter(
        weighted_mtf: float,
        direction: str,
        threshold: float,
    ) -> bool:
        if direction.upper() == "BUY":
            return weighted_mtf >= threshold
        if direction.upper() == "SELL":
            return weighted_mtf <= -threshold
        return False

    @staticmethod
    def price_has_room_to_target(
        entry: float,
        target: float,
        atr: Optional[float],
        min_multiple: float = 0.5,
    ) -> bool:
        if entry <= 0 or target <= 0 or atr is None or atr <= 0:
            return False
        return abs(target - entry) >= atr * min_multiple


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Strategy Quality Helpers Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    print(" [Test 1] Level proximity...")
    if StrategyQuality.is_near_level(23052, 23050, 0.1):
        print(" ✅ Near level works")
        passed += 1
    else:
        print(" ❌ Near level failed")
        failed += 1

    print("\n [Test 2] Rejection quality...")
    if StrategyQuality.rejection_quality(0.05, 0.50, 0.12, "BUY"):
        print(" ✅ BUY rejection quality works")
        passed += 1
    else:
        print(" ❌ BUY rejection quality failed")
        failed += 1

    if StrategyQuality.rejection_quality(0.55, 0.05, 0.12, "SELL"):
        print(" ✅ SELL rejection quality works")
        passed += 1
    else:
        print(" ❌ SELL rejection quality failed")
        failed += 1

    print("\n [Test 3] Breakout close quality...")
    if StrategyQuality.breakout_close_quality(23210, 23200, 23215, 23195, "BUY"):
        print(" ✅ BUY breakout quality works")
        passed += 1
    else:
        print(" ❌ BUY breakout quality failed")
        failed += 1

    print("\n [Test 4] Overextension filter...")
    if StrategyQuality.overextension_filter(0.6, 65, "BUY"):
        print(" ✅ Normal extension accepted")
        passed += 1
    else:
        print(" ❌ Normal extension blocked incorrectly")
        failed += 1

    if not StrategyQuality.overextension_filter(1.5, 80, "BUY"):
        print(" ✅ Overextended buy blocked")
        passed += 1
    else:
        print(" ❌ Overextended buy not blocked")
        failed += 1

    print("\n [Test 5] RR check...")
    if StrategyQuality.rr_ok(100, 95, 110, 1.5):
        print(" ✅ RR check works")
        passed += 1
    else:
        print(" ❌ RR check failed")
        failed += 1

    print("\n [Test 6] Level dedupe...")
    levels = [
        {"price": 23050, "strength": 2, "source": "PDL"},
        {"price": 23054, "strength": 3, "source": "VAL"},
        {"price": 23200, "strength": 1, "source": "ORH"},
    ]
    merged = StrategyQuality.dedupe_levels(levels, merge_distance=5)
    if len(merged) == 2:
        print(" ✅ Level dedupe works")
        passed += 1
    else:
        print(f" ❌ Dedupe failed: {merged}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Strategy Quality Helpers working perfectly!")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()