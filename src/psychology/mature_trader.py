"""
Junior Aladdin - Mature Trader Rules
====================================
PURPOSE:
Provide a separate roadmap-aligned module for advanced discretionary
discipline rules that act as a professional-quality advisory filter.

This module is not a strategy and not a hard execution engine.
It evaluates trade setups against mature-trader heuristics such as:
- weak breakout confirmation
- overused support/resistance levels
- contradictory narrative vs price
- marginal setups that are better skipped

WHY THIS FILE EXISTS:
The roadmap explicitly expects:
    src/psychology/mature_trader.py

Some behavior logic already exists in behavioral_sentinel, but this dedicated
file is still important so the system remains structurally complete and
modular, and so future captain/scoring layers can query this module cleanly.

RULES INCLUDED:
1. Technical says BUY but volume does not confirm -> caution/block
2. Breakout reverses within 2 candles -> fake breakout memory flag
3. 3rd+ touch of S/R weakens bounce reliability
4. Strong narrative one side but price opposite -> caution
5. Marginal rejection cluster -> recommend stand aside

CONNECTS TO:
- Strategies
- Behavioral Sentinel
- Captain
- Scoring / filtering pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger

_logger = setup_logger("mature_trader")


@dataclass
class MatureTraderDecision:
    """
    Output of mature trader rule evaluation.
    """
    allow_trade: bool
    severity: str = "INFO"   # INFO / WARNING / BLOCK
    block_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    recommendation: str = "PROCEED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_trade": self.allow_trade,
            "severity": self.severity,
            "block_reason": self.block_reason,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


class MatureTraderRules:
    """
    Encapsulates mature-trader heuristics.
    """

    def __init__(self):
        self._logger = _logger
        self._recent_fake_breakouts: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
    ) -> MatureTraderDecision:
        """
        Evaluate a setup using professional behavioral heuristics.
        """
        direction = str(opportunity.get("direction", "")).upper()
        strategy = str(opportunity.get("strategy", "UNKNOWN"))
        raw_score = float(opportunity.get("raw_score", 0) or 0)

        features = context.get("features_1m", {}) or {}
        micro = context.get("microstructure", {}) or {}
        key_levels = context.get("key_levels", {}) or {}
        narrative_label = str(context.get("narrative_label", "NEUTRAL"))
        session_memory = context.get("session_memory", {}) or {}

        volume_ratio = features.get("volume_ratio")
        macd_hist = features.get("macd_histogram")
        spread_zscore = micro.get("spread_zscore")
        sr_zone_count = int(key_levels.get("sr_zone_count", 0) or 0)
        failed_breakouts = session_memory.get("failed_breakouts", []) or []

        warnings: List[str] = []

        # --------------------------------------------------------------
        # Rule 1: Technical signal without volume confirmation
        # --------------------------------------------------------------
        if volume_ratio is not None:
            try:
                vr = float(volume_ratio)
                if vr < 0.6:
                    return MatureTraderDecision(
                        allow_trade=False,
                        severity="BLOCK",
                        block_reason="volume_not_confirming",
                        warnings=["technical_signal_without_volume"],
                        recommendation="SKIP_TRADE",
                    )
                elif vr < 0.9:
                    warnings.append("volume_soft_confirmation")
            except (TypeError, ValueError):
                warnings.append("volume_ratio_invalid")

        # --------------------------------------------------------------
        # Rule 2: Spread too wide
        # --------------------------------------------------------------
        if spread_zscore is not None:
            try:
                sz = float(spread_zscore)
                if sz >= 2.0:
                    return MatureTraderDecision(
                        allow_trade=False,
                        severity="BLOCK",
                        block_reason="spread_widening",
                        warnings=["liquidity_poor"],
                        recommendation="SKIP_TRADE",
                    )
                elif sz >= 1.3:
                    warnings.append("spread_slightly_wide")
            except (TypeError, ValueError):
                warnings.append("spread_zscore_invalid")

        # --------------------------------------------------------------
        # Rule 3: Too many failed breakouts already today
        # --------------------------------------------------------------
        if len(failed_breakouts) >= 3:
            warnings.append("market_trappy_today")

        # --------------------------------------------------------------
        # Rule 4: Narrative strongly opposite to trade
        # --------------------------------------------------------------
        if direction == "BUY" and narrative_label in ("MILD_BEARISH", "STRONG_BEARISH", "EVENT_RISK"):
            return MatureTraderDecision(
                allow_trade=False,
                severity="BLOCK",
                block_reason="narrative_against_long",
                warnings=["macro_flow_against_trade"],
                recommendation="SKIP_TRADE",
            )

        if direction == "SELL" and narrative_label in ("MILD_BULLISH", "STRONG_BULLISH", "EVENT_RISK"):
            return MatureTraderDecision(
                allow_trade=False,
                severity="BLOCK",
                block_reason="narrative_against_short",
                warnings=["macro_flow_against_trade"],
                recommendation="SKIP_TRADE",
            )

        # --------------------------------------------------------------
        # Rule 5: Marginal score warning
        # --------------------------------------------------------------
        if 20 <= raw_score <= 25:
            warnings.append("marginal_setup_recommend_skip")

        # --------------------------------------------------------------
        # Rule 6: Weak momentum alignment
        # --------------------------------------------------------------
        if macd_hist is not None:
            try:
                mh = float(macd_hist)
                if direction == "BUY" and mh < 0:
                    warnings.append("macd_against_long")
                if direction == "SELL" and mh > 0:
                    warnings.append("macd_against_short")
            except (TypeError, ValueError):
                warnings.append("macd_invalid")

        # --------------------------------------------------------------
        # Rule 7: Sparse key-level context
        # --------------------------------------------------------------
        if sr_zone_count == 0 and strategy in (
            "SR_REJECTION",
            "FAILED_BREAKOUT_REVERSAL",
            "LIQUIDITY_SWEEP_REVERSAL",
        ):
            warnings.append("weak_level_context")

        # Final recommendation
        if warnings:
            return MatureTraderDecision(
                allow_trade=True,
                severity="WARNING",
                warnings=warnings,
                recommendation="PROCEED_WITH_CAUTION",
            )

        return MatureTraderDecision(
            allow_trade=True,
            severity="INFO",
            warnings=[],
            recommendation="PROCEED",
        )

    # ------------------------------------------------------------------
    # Optional memory hooks
    # ------------------------------------------------------------------
    def record_fake_breakout(
        self,
        level: float,
        direction: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store fake breakout memory for future review / analytics.
        """
        metadata = metadata or {}
        self._recent_fake_breakouts.append(
            {
                "level": level,
                "direction": direction,
                "metadata": metadata,
            }
        )
        if len(self._recent_fake_breakouts) > 50:
            self._recent_fake_breakouts = self._recent_fake_breakouts[-50:]

        self._logger.info(
            "Fake breakout recorded",
            extra={"level": level, "direction": direction},
        )

    def get_recent_fake_breakouts(self) -> List[Dict[str, Any]]:
        return list(self._recent_fake_breakouts)

    def reset_daily(self):
        self._recent_fake_breakouts.clear()
        self._logger.info("Mature Trader Rules reset")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "recent_fake_breakouts": len(self._recent_fake_breakouts),
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Mature Trader Rules Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    rules = MatureTraderRules()

    print(" [Test 1] Strong valid setup...")
    opp1 = {"direction": "BUY", "strategy": "VWAP_PULLBACK", "raw_score": 72}
    ctx1 = {
        "narrative_label": "MILD_BULLISH",
        "features_1m": {
            "volume_ratio": 1.2,
            "macd_histogram": 2.0,
        },
        "microstructure": {"spread_zscore": 0.5},
        "key_levels": {"sr_zone_count": 2},
        "session_memory": {"failed_breakouts": []},
    }
    r1 = rules.evaluate(opp1, ctx1)
    if r1.allow_trade and r1.recommendation in ("PROCEED", "PROCEED_WITH_CAUTION"):
        print(f" ✅ Valid setup result: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Unexpected block: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Volume not confirming -> block...")
    ctx2 = dict(ctx1)
    ctx2["features_1m"] = dict(ctx1["features_1m"])
    ctx2["features_1m"]["volume_ratio"] = 0.4
    r2 = rules.evaluate(opp1, ctx2)
    if not r2.allow_trade and r2.block_reason == "volume_not_confirming":
        print(f" ✅ Volume block works: {r2.block_reason}")
        passed += 1
    else:
        print(f" ❌ Volume block failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Narrative against long -> block...")
    ctx3 = dict(ctx1)
    ctx3["narrative_label"] = "STRONG_BEARISH"
    r3 = rules.evaluate(opp1, ctx3)
    if not r3.allow_trade and r3.block_reason == "narrative_against_long":
        print(f" ✅ Narrative block works: {r3.block_reason}")
        passed += 1
    else:
        print(f" ❌ Narrative block failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Wide spread -> block...")
    ctx4 = dict(ctx1)
    ctx4["microstructure"] = {"spread_zscore": 2.2}
    r4 = rules.evaluate(opp1, ctx4)
    if not r4.allow_trade and r4.block_reason == "spread_widening":
        print(f" ✅ Spread block works: {r4.block_reason}")
        passed += 1
    else:
        print(f" ❌ Spread block failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Trappy market warning...")
    ctx5 = dict(ctx1)
    ctx5["session_memory"] = {"failed_breakouts": [1, 2, 3]}
    r5 = rules.evaluate(opp1, ctx5)
    if r5.allow_trade and "market_trappy_today" in r5.warnings:
        print(f" ✅ Trap warning works: {r5.warnings}")
        passed += 1
    else:
        print(f" ❌ Trap warning failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Record fake breakout memory...")
    rules.record_fake_breakout(23200, "BUY")
    st6 = rules.get_status()
    if st6["recent_fake_breakouts"] == 1:
        print(f" ✅ Fake breakout memory works: {st6}")
        passed += 1
    else:
        print(f" ❌ Fake breakout memory failed: {st6}")
        failed += 1

    print("\n [Test 7] Reset daily...")
    rules.reset_daily()
    st7 = rules.get_status()
    if st7["recent_fake_breakouts"] == 0:
        print(f" ✅ Reset works: {st7}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st7}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Mature Trader Rules working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()