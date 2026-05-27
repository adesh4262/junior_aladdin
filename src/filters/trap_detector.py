"""
Junior Aladdin - Trap Detection Engine
=====================================
PURPOSE:
Evaluate every strategy opportunity before scoring and determine whether the
setup is likely a market trap.

This engine is one of the strongest edges in the system because markets are
designed to trap weak/late participants. The goal is to reject weak breakouts,
late entries, contradictory OI moves, poor liquidity setups, and trap-prone
time windows before the scoring engine wastes capital on them.

TRAP SCORE (0-100) COMPONENTS:
1. Breakout volume weakness
2. Time-of-day bias
3. Session-memory failed breakout history
4. OI contradiction
5. RSI extreme
6. Spread widening
7. Wick dominance on breakout candle

ACTIONS:
0-25   -> No adjustment
26-40  -> Penalty -10
41-50  -> Penalty -20
51-70  -> REJECT
71-100 -> REJECT + mark trap zone for rest of session

CONNECTS TO:
- Strategy outputs (Opportunity)
- Options Features
- Microstructure Features
- Session Memory
- Scoring Engine (consumes trap result)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger

_logger = setup_logger("trap_detector")


@dataclass
class TrapAssessment:
    """
    Standard output of trap detection.
    """
    trap_score: int
    trap_probability: float
    action: str
    score_penalty: int
    reject: bool
    reasons: List[str] = field(default_factory=list)
    trap_zone_level: Optional[float] = None
    trap_zone_activated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trap_score": self.trap_score,
            "trap_probability": self.trap_probability,
            "action": self.action,
            "score_penalty": self.score_penalty,
            "reject": self.reject,
            "reasons": self.reasons,
            "trap_zone_level": self.trap_zone_level,
            "trap_zone_activated": self.trap_zone_activated,
        }


class TrapDetector:
    """
    Evaluate opportunities for trap characteristics.

    Usage:
        detector = TrapDetector()
        result = detector.evaluate(opportunity_dict, context)
    """

    def __init__(self):
        self._logger = _logger
        self._trap_zones: List[float] = []
        self._zone_tolerance_points = 5.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        opportunity: Dict,
        context: Optional[Dict] = None,
    ) -> TrapAssessment:
        """
        Evaluate one opportunity.

        Args:
            opportunity: strategy output dict / Opportunity.to_dict()
            context: unified context dict containing:
                - session_phase
                - session_memory
                - options
                - microstructure
                - key_levels
                - features_1m (optional)
                - feed_health / data quality (optional)

        Returns:
            TrapAssessment
        """
        context = context or {}

        # --------------------------------------------------------------
        # REQUIRED FIX #1 & #5: Safe parsing and fail-closed direction
        # --------------------------------------------------------------
        if opportunity is None or not isinstance(opportunity, dict):
            self._logger.error(
                "Invalid opportunity input (not a dict)",
                extra={"opportunity_type": str(type(opportunity))},
            )
            return TrapAssessment(
                trap_score=100,
                trap_probability=1.0,
                action="REJECT",
                score_penalty=0,
                reject=True,
                reasons=["invalid_opportunity_input"],
                trap_zone_level=None,
                trap_zone_activated=False,
            )

        try:
            entry_raw = opportunity.get("entry_price", None)
            entry = float(entry_raw)  # may raise
        except Exception:
            self._logger.error(
                "Invalid opportunity input (entry_price parse failed)",
                extra={"entry_price": opportunity.get("entry_price", None)},
            )
            return TrapAssessment(
                trap_score=100,
                trap_probability=1.0,
                action="REJECT",
                score_penalty=0,
                reject=True,
                reasons=["invalid_entry_price"],
                trap_zone_level=None,
                trap_zone_activated=False,
            )

        if entry <= 0:
            self._logger.error(
                "Invalid opportunity input (entry_price <= 0)",
                extra={"entry_price": entry},
            )
            return TrapAssessment(
                trap_score=100,
                trap_probability=1.0,
                action="REJECT",
                score_penalty=0,
                reject=True,
                reasons=["invalid_entry_price"],
                trap_zone_level=None,
                trap_zone_activated=False,
            )

        try:
            direction_raw = opportunity.get("direction", None)
            direction = str(direction_raw).strip().upper() if direction_raw is not None else ""
        except Exception:
            direction = ""

        if direction == "LONG":
            direction = "BUY"
        elif direction == "SHORT":
            direction = "SELL"

        if direction not in {"BUY", "SELL"}:
            self._logger.error(
                "Invalid opportunity direction",
                extra={"direction": opportunity.get("direction", None)},
            )
            return TrapAssessment(
                trap_score=100,
                trap_probability=1.0,
                action="REJECT",
                score_penalty=0,
                reject=True,
                reasons=["invalid_direction"],
                trap_zone_level=None,
                trap_zone_activated=False,
            )

        if opportunity.get("direction") != direction:
            opportunity = dict(opportunity)
            opportunity["direction"] = direction

        raw_reasons: List[str] = []
        trap_score = 0

        # 1. breakout volume weakness
        s, r = self._score_breakout_volume(opportunity, context)
        trap_score += s
        raw_reasons.extend(r)

        # 2. time-of-day bias
        s, r = self._score_time_of_day(context)
        trap_score += s
        raw_reasons.extend(r)

        # 3. session-memory failed breakouts
        s, r = self._score_session_memory(context)
        trap_score += s
        raw_reasons.extend(r)

        # 4. OI contradiction
        s, r = self._score_oi_contradiction(opportunity, context)
        trap_score += s
        raw_reasons.extend(r)

        # 5. RSI extreme
        s, r = self._score_rsi_extreme(opportunity, context)
        trap_score += s
        raw_reasons.extend(r)

        # 6. spread widening
        s, r = self._score_spread(context)
        trap_score += s
        raw_reasons.extend(r)

        # 7. wick dominance
        s, r = self._score_wick_dominance(opportunity, context)
        trap_score += s
        raw_reasons.extend(r)

        # 8. repeated trap zone reuse
        s, r = self._score_known_trap_zone(entry)
        trap_score += s
        raw_reasons.extend(r)

        trap_score = max(0, min(100, int(round(trap_score))))
        trap_probability = round(trap_score / 100.0, 3)

        action, penalty, reject, activate_zone = self._map_action(trap_score)

        # --------------------------------------------------------------
        # REQUIRED FIX #4: Trap zone activation consistency
        # --------------------------------------------------------------
        trap_zone_level = None
        if activate_zone:
            if entry > 0:
                trap_zone_level = entry
                self._register_trap_zone(trap_zone_level)
            else:
                activate_zone = False

        result = TrapAssessment(
            trap_score=trap_score,
            trap_probability=trap_probability,
            action=action,
            score_penalty=penalty,
            reject=reject,
            reasons=raw_reasons,
            trap_zone_level=trap_zone_level,
            trap_zone_activated=activate_zone,
        )

        self._logger.info(
            "Trap evaluation complete",
            extra={
                "strategy": opportunity.get("strategy", "UNKNOWN"),
                "direction": direction,
                "entry_price": entry,
                "trap_score": trap_score,
                "action": action,
                "reject": reject,
                "reasons": raw_reasons[:6],
            },
        )

        return result

    def get_trap_zones(self) -> List[float]:
        return list(self._trap_zones)

    def reset_daily(self):
        self._trap_zones.clear()
        self._logger.info("Trap Detector reset")

    # ------------------------------------------------------------------
    # Scoring Components
    # ------------------------------------------------------------------
    def _score_breakout_volume(self, opportunity: Dict, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        volume_ratio = self._extract_feature(context, "volume_ratio")
        is_breakout_style = self._is_breakout_style_strategy(opportunity)

        if is_breakout_style and volume_ratio is not None:
            if volume_ratio < 0.8:
                score += 25
                reasons.append("breakout_volume_weak_<0.8")
            elif volume_ratio < 1.0:
                score += 15
                reasons.append("breakout_volume_soft_<1.0")

        return score, reasons

    def _score_time_of_day(self, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        session = context.get("session_phase", "")
        if session == "LUNCH_LULL":
            score += 20
            reasons.append("trap_prone_lunch")
        elif session == "CLOSING_SESSION":
            score += 15
            reasons.append("trap_prone_close")
        elif session == "INITIAL_BALANCE":
            score += 10
            reasons.append("initial_balance_noise")

        return score, reasons

    def _score_session_memory(self, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        session_memory = context.get("session_memory", {}) or {}
        failed_breakouts = session_memory.get("failed_breakouts", [])

        # REQUIRED FIX #3: session memory type safety
        if isinstance(failed_breakouts, (list, tuple)):
            trap_count = len(failed_breakouts)
        else:
            trap_count = 0

        if trap_count >= 2:
            score += 15
            reasons.append("multiple_failed_breakouts_today")
        elif trap_count >= 1:
            score += 8
            reasons.append("prior_failed_breakout_today")

        return score, reasons

    def _safe_float(self, value: Any, field_name: str) -> float:
        """
        Safe float conversion helper for noisy external data.

        Returns 0.0 on any conversion failure and logs a debug message.
        """
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception as e:
            self._logger.debug(
                "Float conversion failed; defaulting to 0.0",
                extra={
                    "field_name": field_name,
                    "value": str(value),
                    "error": str(e),
                },
            )
            return 0.0

    def _score_oi_contradiction(self, opportunity: Dict, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        options = context.get("options", {}) or {}
        entry = float(opportunity.get("entry_price", 0) or 0)
        direction = str(opportunity.get("direction", "")).upper()

        # REQUIRED FIX #2: Safe OI wall parsing
        ce_wall = self._safe_float(options.get("highest_ce_oi_strike", 0), "highest_ce_oi_strike")
        pe_wall = self._safe_float(options.get("highest_pe_oi_strike", 0), "highest_pe_oi_strike")

        if direction == "BUY":
            if ce_wall > 0 and abs(ce_wall - entry) <= 30:
                score += 15
                reasons.append("buying_into_ce_wall")
        elif direction == "SELL":
            if pe_wall > 0 and abs(pe_wall - entry) <= 30:
                score += 15
                reasons.append("selling_into_pe_wall")

        return score, reasons

    def _score_rsi_extreme(self, opportunity: Dict, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        rsi = self._extract_feature(context, "rsi")
        direction = str(opportunity.get("direction", "")).upper()

        if rsi is None:
            return 0, reasons

        if direction == "BUY" and rsi > 70:
            score += 10
            reasons.append("buy_rsi_overbought")
        elif direction == "SELL" and rsi < 30:
            score += 10
            reasons.append("sell_rsi_oversold")

        return score, reasons

    def _score_spread(self, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        micro = context.get("microstructure", {}) or {}
        spread_zscore = micro.get("spread_zscore")

        if spread_zscore is not None:
            try:
                spread_zscore = float(spread_zscore)
                if spread_zscore > 1.5:
                    score += 10
                    reasons.append("spread_widening")
            except (TypeError, ValueError):
                pass

        return score, reasons

    def _score_wick_dominance(self, opportunity: Dict, context: Dict) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        body_ratio = self._extract_feature(context, "candle_body_ratio")
        upper_wick_ratio = self._extract_feature(context, "upper_wick_ratio")
        lower_wick_ratio = self._extract_feature(context, "lower_wick_ratio")
        direction = str(opportunity.get("direction", "")).upper()

        if body_ratio is None:
            return 0, reasons

        wick_ratio = 0.0
        if direction == "BUY":
            wick_ratio = upper_wick_ratio or 0.0
        elif direction == "SELL":
            wick_ratio = lower_wick_ratio or 0.0

        if wick_ratio > 0.6:
            score += 5
            reasons.append("breakout_wick_dominance")

        return score, reasons

    def _score_known_trap_zone(self, entry_price: float) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        score = 0

        if entry_price <= 0:
            return score, reasons

        for level in self._trap_zones:
            if abs(entry_price - level) <= self._zone_tolerance_points:
                score += 15
                reasons.append("revisiting_known_trap_zone")
                break

        return score, reasons

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_feature(self, context: Dict, key: str) -> Optional[float]:
        """
        Search for a feature across common context containers.
        """
        for container_key in ("features_1m", "features", "price_momentum", "microstructure"):
            container = context.get(container_key)
            if isinstance(container, dict) and key in container:
                value = container.get(key)
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

        micro = context.get("microstructure", {})
        if isinstance(micro, dict) and key in micro:
            try:
                return float(micro.get(key))
            except (TypeError, ValueError):
                return None

        return None

    def _is_breakout_style_strategy(self, opportunity: Dict) -> bool:
        strategy_name = str(opportunity.get("strategy", "")).upper()
        breakout_keywords = (
            "BREAKOUT",
            "ORB",
            "MOMENTUM",
            "CONTINUATION",
        )
        return any(k in strategy_name for k in breakout_keywords)

    def _map_action(self, trap_score: int) -> Tuple[str, int, bool, bool]:
        """
        Map trap score to action.
        Returns: (action, penalty, reject, activate_zone)
        """
        if trap_score <= 25:
            return "ALLOW", 0, False, False
        if trap_score <= 40:
            return "PENALIZE_LIGHT", -10, False, False
        if trap_score <= 50:
            return "PENALIZE_HEAVY", -20, False, False
        if trap_score <= 70:
            return "REJECT", 0, True, False
        return "REJECT_AND_MARK_ZONE", 0, True, True

    def _register_trap_zone(self, level: float):
        if level <= 0:
            return

        for existing in self._trap_zones:
            if abs(existing - level) <= self._zone_tolerance_points:
                return

        self._trap_zones.append(round(level, 2))
        self._logger.warning(
            "Trap zone registered",
            extra={"level": level, "zones": self._trap_zones},
        )


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Trap Detector Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    detector = TrapDetector()

    print(" [Test 1] Clean low-trap setup...")
    opp = {
        "strategy": "VWAP_PULLBACK",
        "direction": "BUY",
        "entry_price": 23200.0,
    }
    ctx = {
        "session_phase": "GOLDEN_AM",
        "session_memory": {"failed_breakouts": []},
        "options": {"highest_ce_oi_strike": 23450, "highest_pe_oi_strike": 23050},
        "microstructure": {
            "spread_zscore": 0.4,
            "upper_wick_ratio": 0.10,
            "lower_wick_ratio": 0.25,
            "candle_body_ratio": 0.40,
        },
        "features_1m": {"rsi": 48, "volume_ratio": 0.9},
    }
    r1 = detector.evaluate(opp, ctx)
    if r1.trap_score <= 25 and not r1.reject:
        print(f" ✅ trap_score={r1.trap_score}, action={r1.action}")
        passed += 1
    else:
        print(f" ❌ Unexpected trap result: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Breakout with weak volume...")
    opp2 = {
        "strategy": "OPENING_RANGE_BREAKOUT",
        "direction": "BUY",
        "entry_price": 23300.0,
    }
    ctx2 = {
        "session_phase": "INITIAL_BALANCE",
        "session_memory": {"failed_breakouts": []},
        "options": {"highest_ce_oi_strike": 23310},
        "microstructure": {
            "spread_zscore": 0.5,
            "upper_wick_ratio": 0.65,
            "lower_wick_ratio": 0.05,
            "candle_body_ratio": 0.15,
        },
        "features_1m": {"rsi": 72, "volume_ratio": 0.7},
    }
    r2 = detector.evaluate(opp2, ctx2)
    if r2.trap_score >= 40:
        print(f" ✅ trap_score={r2.trap_score}, reasons={r2.reasons}")
        passed += 1
    else:
        print(f" ❌ Trap score too low: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Session memory trap penalty...")
    ctx3 = {
        "session_phase": "GOLDEN_PM",
        "session_memory": {"failed_breakouts": [23200, 23250]},
        "options": {},
        "microstructure": {"spread_zscore": 0.2},
        "features_1m": {"rsi": 50, "volume_ratio": 1.1},
    }
    opp3 = {"strategy": "TREND_CONTINUATION", "direction": "BUY", "entry_price": 23240}
    r3 = detector.evaluate(opp3, ctx3)
    if any("multiple_failed_breakouts_today" == x for x in r3.reasons):
        print(f" ✅ session-memory penalty applied, score={r3.trap_score}")
        passed += 1
    else:
        print(f" ❌ Missing session-memory penalty: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Known trap zone re-entry...")
    detector.reset_daily()
    high_trap = TrapAssessment(
        trap_score=75,
        trap_probability=0.75,
        action="REJECT_AND_MARK_ZONE",
        score_penalty=0,
        reject=True,
        reasons=["demo"],
        trap_zone_level=23300.0,
        trap_zone_activated=True,
    )
    detector._register_trap_zone(23300.0)

    opp4 = {"strategy": "SR_REJECTION", "direction": "BUY", "entry_price": 23303.0}
    ctx4 = {
        "session_phase": "GOLDEN_AM",
        "session_memory": {"failed_breakouts": []},
        "options": {},
        "microstructure": {"spread_zscore": 0.2},
        "features_1m": {"rsi": 42, "volume_ratio": 1.0},
    }
    r4 = detector.evaluate(opp4, ctx4)
    if any("revisiting_known_trap_zone" == x for x in r4.reasons):
        print(f" ✅ known trap-zone penalty applied, score={r4.trap_score}")
        passed += 1
    else:
        print(f" ❌ Missing trap-zone penalty: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Hard reject threshold...")
    opp5 = {
        "strategy": "ORB_BREAKOUT",
        "direction": "BUY",
        "entry_price": 23300.0,
    }
    ctx5 = {
        "session_phase": "LUNCH_LULL",
        "session_memory": {"failed_breakouts": [1, 2, 3]},
        "options": {"highest_ce_oi_strike": 23310},
        "microstructure": {
            "spread_zscore": 2.1,
            "upper_wick_ratio": 0.8,
            "lower_wick_ratio": 0.1,
            "candle_body_ratio": 0.12,
        },
        "features_1m": {"rsi": 74, "volume_ratio": 0.6},
    }
    r5 = detector.evaluate(opp5, ctx5)
    if r5.reject:
        print(f" ✅ reject=True, trap_score={r5.trap_score}, action={r5.action}")
        passed += 1
    else:
        print(f" ❌ Should reject: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Reset daily...")
    detector.reset_daily()
    if detector.get_trap_zones() == []:
        print(" ✅ trap zones cleared")
        passed += 1
    else:
        print(" ❌ reset failed")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Trap Detector working perfectly!")
        print(" ✅ Ready for next roadmap step.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()