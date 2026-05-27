"""
Junior Aladdin - Opportunity Scorer
===================================
PURPOSE:
Score every post-strategy, post-trap opportunity using the 10-factor
confluence model from the system plan.

FACTORS:
1. technical_structure   (14%)
2. location_quality      (14%)
3. momentum              (12%)
4. options_pressure      (12%)
5. regime_match          (10%)
6. mtf_alignment         (10%)
7. smart_money           (8%)
8. narrative_fit         (7%)
9. time_context          (7%)
10. trap_safety          (6%)

FEATURES:
- regime-adaptive weight modifiers
- hard rejection rules
- confluence bonus
- triple-confluence bonus
- score normalization and diagnostics

CONNECTS TO:
- Strategies (Opportunity input)
- Trap Detector
- Narrative / Regime / Time Context / Feature Engine outputs
- Brain Engine / Captain / Risk Engine
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, List, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("opportunity_scorer")

_NAN = float("nan")

_BLUEPRINT_DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "technical_structure": 0.14,
    "location_quality": 0.14,
    "momentum": 0.12,
    "options_pressure": 0.12,
    "regime_match": 0.10,
    "mtf_alignment": 0.10,
    "smart_money": 0.08,
    "narrative_fit": 0.07,
    "time_context": 0.07,
    "trap_safety": 0.06,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safe float conversion with non-finite guard (NaN/Inf).
    """
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off", ""):
            return False
    return default


def _extract_mtf_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the phase-3 MTF payload out of either flat or nested context shapes.

    Supports:
    - top-level keys like weighted_mtf / mtf_trap_zone
    - context["mtf"] from FeatureEngine
    - context["mtf_alignment"] from Captain-style payloads
    """
    payload: Dict[str, Any] = {}
    if not isinstance(context, dict):
        return payload

    direct_keys = (
        "weighted_mtf",
        "mtf_trap_zone",
        "confluence_bonus_applied",
        "trend_strength_with_confluence",
    )
    for key in direct_keys:
        value = context.get(key)
        if value is not None:
            payload[key] = value

    nested_candidates = []
    for key in ("mtf", "mtf_alignment"):
        node = context.get(key)
        if isinstance(node, dict):
            nested_candidates.append(node)

    for candidate in nested_candidates:
        for key in direct_keys:
            if key not in payload and key in candidate and candidate.get(key) is not None:
                payload[key] = candidate.get(key)

    return payload


@dataclass
class ScoredOpportunity:
    """
    Standard scored opportunity wrapper.
    """
    opportunity: Dict[str, Any]
    factor_scores: Dict[str, float]
    final_score: float
    regime_used: str
    weights_used: Dict[str, float]
    bonuses: Dict[str, float] = field(default_factory=dict)
    hard_reject: bool = False
    reject_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity": self.opportunity,
            "factor_scores": self.factor_scores,
            "final_score": self.final_score,
            "regime_used": self.regime_used,
            "weights_used": self.weights_used,
            "bonuses": self.bonuses,
            "hard_reject": self.hard_reject,
            "reject_reason": self.reject_reason,
        }


class OpportunityScorer:
    """
    10-factor confluence scoring engine with regime-adaptive weighting.
    """

    FACTOR_ORDER = [
        "technical_structure",
        "location_quality",
        "momentum",
        "options_pressure",
        "regime_match",
        "mtf_alignment",
        "smart_money",
        "narrative_fit",
        "time_context",
        "trap_safety",
    ]

    def __init__(self):
        self._logger = _logger

        self._base_weights = Config.get(
            "scoring",
            "base_weights",
            default=dict(_BLUEPRINT_DEFAULT_BASE_WEIGHTS),
        )

        # Sanitize base weights from config
        if not isinstance(self._base_weights, dict):
            self._base_weights = dict(_BLUEPRINT_DEFAULT_BASE_WEIGHTS)

        for k, v in list(self._base_weights.items()):
            default_weight = _BLUEPRINT_DEFAULT_BASE_WEIGHTS.get(k, 0.0)
            self._base_weights[k] = _safe_float(v, default_weight)

        self._regime_modifiers = Config.get(
            "scoring",
            "regime_modifiers",
            default={},
        )

        self._min_total_score = Config.get(
            "scoring",
            "min_total_score",
            default=58,
        )
        self._min_component_score = Config.get(
            "scoring",
            "min_component_score",
            default=15,
        )

        # REQUIRED FIX #2: Sanitize config thresholds
        self._min_total_score = _safe_float(self._min_total_score, 58.0)
        self._min_component_score = _safe_float(self._min_component_score, 15.0)
        self._min_total_score = min(self._min_total_score, 52.0)

        self._mtf_trap_zone_multiplier = _safe_float(
            Config.get("scoring", "mtf_trap_zone_multiplier", default=0.8),
            0.8,
        )
        self._mtf_trap_zone_multiplier = max(0.5, min(1.0, self._mtf_trap_zone_multiplier))

        self._mtf_confluence_bonus = _safe_float(
            Config.get("scoring", "mtf_confluence_bonus", default=4.0),
            4.0,
        )
        self._mtf_confluence_bonus = max(0.0, min(10.0, self._mtf_confluence_bonus))

        self._mtf_confluence_bonus_threshold = _safe_float(
            Config.get("scoring", "mtf_confluence_bonus_threshold", default=65.0),
            65.0,
        )
        self._mtf_confluence_bonus_threshold = max(0.0, min(100.0, self._mtf_confluence_bonus_threshold))

        self._confluence_bonus_2 = Config.get(
            "scoring",
            "confluence_bonus_2_strategies",
            default=10,
        )
        self._triple_confluence_bonus = Config.get(
            "scoring",
            "triple_confluence_bonus",
            default=15,
        )

    @staticmethod
    def _safe_float_strict(value: Any, field_name: str) -> Tuple[float, bool]:
        """
        Strict float parsing for critical fields.
        Returns: (parsed_value, is_error)
        If conversion fails OR yields NaN/Inf => (0.0, True).
        """
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return 0.0, True
            return v, False
        except Exception:
            return 0.0, True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score_opportunity(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ScoredOpportunity:
        """
        Score one opportunity end-to-end.
        """
        # Context validation
        if context is None or not isinstance(context, dict) or not context:
            return ScoredOpportunity(
                opportunity=opportunity if isinstance(opportunity, dict) else {},
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_context",
            )

        # Opportunity validation
        if opportunity is None or not isinstance(opportunity, dict):
            return ScoredOpportunity(
                opportunity={},
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_opportunity_input",
            )

        # Direction whitelist
        direction = str(opportunity.get("direction", "")).strip().upper()
        if direction == "LONG":
            direction = "BUY"
        elif direction == "SHORT":
            direction = "SELL"
        if direction not in ("BUY", "SELL"):
            return ScoredOpportunity(
                opportunity=opportunity,
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_direction",
            )

        if opportunity.get("direction") != direction:
            opportunity = dict(opportunity)
            opportunity["direction"] = direction

        # Critical numeric validation: entry_price
        entry = _safe_float(opportunity.get("entry_price"), 0.0)
        if entry <= 0:
            return ScoredOpportunity(
                opportunity=opportunity,
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_entry_price",
            )

        # ------------------------------------------------------------------
        # REQUIRED FIX #1: Strict parsing for critical context fields
        # ------------------------------------------------------------------
        narrative_fit_factor, err = self._safe_float_strict(
            context.get("narrative_fit_factor", 0.8), "narrative_fit_factor"
        )
        if err:
            return ScoredOpportunity(
                opportunity=opportunity,
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_narrative_fit_factor",
            )

        trap_probability, err = self._safe_float_strict(
            context.get("trap_probability", 0.0), "trap_probability"
        )
        if err:
            return ScoredOpportunity(
                opportunity=opportunity,
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_trap_probability",
            )

        data_quality_score, err = self._safe_float_strict(
            context.get("data_quality_score", 100.0), "data_quality_score"
        )
        if err:
            return ScoredOpportunity(
                opportunity=opportunity,
                factor_scores={},
                final_score=0.0,
                regime_used="UNKNOWN",
                weights_used={},
                hard_reject=True,
                reject_reason="invalid_data_quality_score",
            )

        # Use a shallow copy to ensure downstream reads the validated numeric values.
        context = dict(context)
        context["narrative_fit_factor"] = narrative_fit_factor
        context["trap_probability"] = trap_probability
        context["data_quality_score"] = data_quality_score

        regime = str(context.get("regime", "UNKNOWN"))
        weights = self._build_regime_weights(regime, context)

        factor_scores = {
            "technical_structure": self._score_technical_structure(opportunity, context),
            "location_quality": self._score_location(opportunity, context),
            "momentum": self._score_momentum(opportunity, context),
            "options_pressure": self._score_options(opportunity, context),
            "regime_match": self._score_regime_match(opportunity, context),
            "mtf_alignment": self._score_mtf(opportunity, context),
            "smart_money": self._score_smart_money(opportunity, context),
            "narrative_fit": self._score_narrative(opportunity, context),
            "time_context": self._score_time(opportunity, context),
            "trap_safety": self._score_trap_safety(opportunity, context),
        }

        hard_reject, reject_reason = self._check_hard_rejection(
            opportunity,
            context,
            factor_scores,
        )
        bonuses = self._compute_bonuses(opportunity, context, factor_scores)

        weighted_total = 0.0
        for k in self.FACTOR_ORDER:
            weighted_total += factor_scores[k] * weights[k]

        final_score = round(weighted_total + sum(bonuses.values()), 2)
        final_score = max(0.0, min(100.0, final_score))

        # Force score to zero on hard reject
        if hard_reject:
            final_score = 0.0
            bonuses.clear()

        # Apply min total score threshold
        if not hard_reject and final_score < self._min_total_score:
            hard_reject = True
            reject_reason = f"below_min_total_score:{final_score}"
            final_score = 0.0
            bonuses.clear()

        result = ScoredOpportunity(
            opportunity=opportunity,
            factor_scores=factor_scores,
            final_score=final_score,
            regime_used=regime,
            weights_used=weights,
            bonuses=bonuses,
            hard_reject=hard_reject,
            reject_reason=reject_reason,
        )

        extra = {
            "strategy": opportunity.get("strategy", "UNKNOWN"),
            "direction": opportunity.get("direction", "UNKNOWN"),
            "final_score": final_score,
            "hard_reject": hard_reject,
            "regime": regime,
        }
        if hard_reject:
            extra["reject_reason"] = reject_reason

        self._logger.info("Opportunity scored", extra=extra)

        return result

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    def _build_regime_weights(self, regime: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply regime modifiers and renormalize to 1.0.
        """
        weights = dict(self._base_weights)

        regime_key = regime.lower() if isinstance(regime, str) else "unknown"
        mods = self._regime_modifiers.get(regime_key, {})

        if isinstance(mods, dict):
            for factor, delta in mods.items():
                if factor in weights and factor != "min_score_override":
                    weights[factor] = max(0.0, weights[factor] + _safe_float(delta, 0.0))

        if bool(context.get("is_expiry_day", False)):
            expiry_mods = self._regime_modifiers.get("expiry", {})
            if isinstance(expiry_mods, dict):
                for factor, delta in expiry_mods.items():
                    if factor in weights:
                        weights[factor] = max(0.0, weights[factor] + _safe_float(delta, 0.0))

        total = sum(weights.values())
        if total <= 0:
            n = len(self.FACTOR_ORDER)
            return {k: 1.0 / n for k in self.FACTOR_ORDER}

        normalized = {k: round(v / total, 6) for k, v in weights.items()}

        for k in self.FACTOR_ORDER:
            if k not in normalized:
                normalized[k] = 0.001
                self._logger.warning(
                    "Missing weight key after normalization; injecting default",
                    extra={"missing_key": k, "injected_value": 0.001, "regime": regime},
                )

        # REQUIRED FIX #3: Renormalize after injection
        total2 = sum(normalized.values())
        if total2 > 0:
            normalized = {k: round(v / total2, 6) for k, v in normalized.items()}

        return normalized

    # ------------------------------------------------------------------
    # Hard rejection rules
    # ------------------------------------------------------------------
    def _check_hard_rejection(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
        factor_scores: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Hard rejection rules from the plan.
        Order matters:
        1. explicit blocked session / time
        2. narrative fit zero
        3. trap too high
        4. data quality too low
        5. spread too wide
        6. weak component floor
        """
        narrative_fit_factor = _safe_float(context.get("narrative_fit_factor", 0.8), 0.0)
        trap_probability = _safe_float(context.get("trap_probability", 0.0), 0.0)
        data_quality = _safe_float(context.get("data_quality_score", 100.0), 0.0)

        micro = context.get("microstructure", {})
        if not isinstance(micro, dict):
            micro = {}

        spread_zscore = micro.get("spread_zscore")
        session_phase = str(context.get("session_phase", "")).upper().strip()

        if session_phase in (
            "PRE_MARKET",
            "OPENING_AUCTION",
            "OR_FORMATION",
            "LAST_MINUTES",
            "POST_MARKET",
        ):
            return True, f"time_context_blocked:{session_phase}"

        if narrative_fit_factor == 0.0:
            return True, "narrative_fit_zero"

        if trap_probability > 0.50:
            return True, f"trap_probability_high:{trap_probability}"

        if data_quality < 60:
            return True, f"data_quality_low:{data_quality}"

        if spread_zscore is not None:
            if _safe_float(spread_zscore, 0.0) > 2.0:
                return True, f"spread_zscore_high:{spread_zscore}"

        for factor_name, score in factor_scores.items():
            component_floor = self._min_component_score
            if factor_name == "smart_money":
                component_floor = min(component_floor, 10.0)

            if score < component_floor:
                return True, f"component_below_min:{factor_name}={score}"

        return False, ""

    # ------------------------------------------------------------------
    # Bonuses
    # ------------------------------------------------------------------
    def _compute_bonuses(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
        factor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        bonuses: Dict[str, float] = {}

        directional_confluence = _safe_int(context.get("same_direction_signals", 1), 1)
        if directional_confluence >= 2:
            bonuses["multi_strategy_confluence"] = _safe_float(self._confluence_bonus_2, 0.0)

        technical_ok = factor_scores.get("technical_structure", 0) >= 70
        narrative_ok = factor_scores.get("narrative_fit", 0) >= 70
        smart_money_ok = factor_scores.get("smart_money", 0) >= 70
        if technical_ok and narrative_ok and smart_money_ok:
            bonuses["triple_confluence"] = _safe_float(self._triple_confluence_bonus, 0.0)

        return bonuses

    # ------------------------------------------------------------------
    # Factor scoring
    # ------------------------------------------------------------------
    def _score_technical_structure(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        raw_score = _safe_float(opportunity.get("raw_score", 50), 50.0)
        rr = _safe_float(opportunity.get("risk_reward", 1.0), 1.0)

        score = min(100.0, raw_score * 0.8 + min(rr * 10, 20))
        return round(score, 2)

    def _score_location(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        direction = str(opportunity.get("direction", "")).upper()
        entry = _safe_float(opportunity.get("entry_price", 0), 0.0)

        key_levels = context.get("key_levels", {}) or {}
        volume_profile = context.get("volume_profile", {}) or {}
        options = context.get("options", {}) or {}
        smart_money_5m = context.get("smart_money_5m", {}) or {}
        smart_money_15m = context.get("smart_money_15m", {}) or {}

        if not isinstance(key_levels, dict):
            key_levels = {}
        if not isinstance(volume_profile, dict):
            volume_profile = {}
        if not isinstance(options, dict):
            options = {}
        if not isinstance(smart_money_5m, dict):
            smart_money_5m = {}
        if not isinstance(smart_money_15m, dict):
            smart_money_15m = {}

        if entry <= 0:
            return 0.0

        score = 10.0
        nearby_levels: List[float] = []

        for k in ("pdh", "pdl", "or_high", "or_low", "ib_high", "ib_low"):
            fv = _safe_float(key_levels.get(k, 0), 0.0)
            if fv > 0:
                nearby_levels.append(fv)

        sr_zones = key_levels.get("sr_zones", [])
        if isinstance(sr_zones, list):
            for z in sr_zones:
                if isinstance(z, dict):
                    lvl = _safe_float(z.get("level", 0), 0.0)
                    if lvl > 0:
                        nearby_levels.append(lvl)

        for k in ("poc", "vah", "val"):
            fv = _safe_float(volume_profile.get(k, 0), 0.0)
            if fv > 0:
                nearby_levels.append(fv)

        for k in ("highest_ce_oi_strike", "highest_pe_oi_strike"):
            fv = _safe_float(options.get(k, 0), 0.0)
            if fv > 0:
                nearby_levels.append(fv)

        if nearby_levels:
            nearest = min(nearby_levels, key=lambda x: abs(entry - x))
            dist = abs(entry - nearest)
            if dist <= 5:
                score += 35
            elif dist <= 15:
                score += 20
            elif dist <= 30:
                score += 10
            else:
                score += 0

        for sm in (smart_money_5m, smart_money_15m):
            nearest_dir = str(sm.get("nearest_fvg_direction", "NONE")).upper().strip()
            nearest_dist = _safe_float(sm.get("nearest_fvg_distance", 999), 999.0)
            ob_dir = str(sm.get("nearest_ob_direction", "NONE")).upper().strip()

            if direction == "BUY":
                if nearest_dir == "BULLISH" and nearest_dist <= 15:
                    score += 10
                if ob_dir == "BULLISH":
                    score += 10
            elif direction == "SELL":
                if nearest_dir == "BEARISH" and nearest_dist <= 15:
                    score += 10
                if ob_dir == "BEARISH":
                    score += 10

        return round(min(score, 100.0), 2)

    def _score_momentum(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        direction = str(opportunity.get("direction", "")).upper()
        f1 = context.get("features_1m", {}) or {}
        if not isinstance(f1, dict):
            f1 = {}

        rsi = _safe_float(f1.get("rsi"), _NAN)
        rsi_slope = _safe_float(f1.get("rsi_slope"), _NAN)
        macd_hist = _safe_float(f1.get("macd_histogram"), _NAN)
        macd_slope = _safe_float(f1.get("macd_hist_slope"), _NAN)
        roc_5 = _safe_float(f1.get("roc_5"), _NAN)

        score = 5.0

        if direction == "BUY":
            if 45 <= rsi <= 65:
                score += 20
            elif 35 <= rsi <= 75:
                score += 8
            if rsi_slope > 0:
                score += 10
            if macd_hist > 0:
                score += 15
            if macd_slope > 0:
                score += 10
            if roc_5 > 0:
                score += 10
        else:
            if 35 <= rsi <= 55:
                score += 20
            elif 25 <= rsi <= 65:
                score += 8
            if rsi_slope < 0:
                score += 10
            if macd_hist < 0:
                score += 15
            if macd_slope < 0:
                score += 10
            if roc_5 < 0:
                score += 10

        return round(min(score, 100.0), 2)

    def _score_options(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        direction = str(opportunity.get("direction", "")).upper()
        opt = context.get("options", {}) or {}
        if not isinstance(opt, dict):
            opt = {}

        score = 10.0

        pcr = _safe_float(opt.get("pcr_oi", 0), 0.0)
        synthetic_premium = _safe_float(opt.get("synthetic_premium", 0), 0.0)
        gex_regime = str(opt.get("gex_regime", "NEUTRAL")).upper().strip()

        if direction == "BUY":
            if pcr > 1.0:
                score += 20
            elif pcr > 0.85:
                score += 10
            if synthetic_premium >= 0:
                score += 15
            if gex_regime == "NEGATIVE":
                score += 10
        else:
            if pcr < 0.8 and pcr > 0:
                score += 20
            elif pcr < 1.0 and pcr > 0:
                score += 10
            if synthetic_premium <= 0:
                score += 15
            if gex_regime == "NEGATIVE":
                score += 10

        return round(min(score, 100.0), 2)

    def _score_regime_match(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        strategy = str(opportunity.get("strategy", "")).upper()
        regime = str(context.get("regime", "UNKNOWN")).upper().strip()

        score = 20.0

        if strategy in ("VWAP_PULLBACK", "TREND_CONTINUATION", "OPENING_RANGE_BREAKOUT"):
            if regime == "TRENDING":
                score = 90
            elif regime == "VOLATILE":
                score = 60
            elif regime == "RANGE":
                score = 35
            else:
                score = 10

        elif strategy in ("SR_REJECTION", "VOL_PROFILE_POC", "OI_WALL_BOUNCE"):
            if regime == "RANGE":
                score = 90
            elif regime == "TRENDING":
                score = 65
            elif regime == "VOLATILE":
                score = 40
            else:
                score = 10

        elif strategy in (
            "STOP_HUNT_RECLAIM",
            "FAILED_BREAKOUT_REVERSAL",
            "ABSORPTION_REVERSAL",
            "LIQUIDITY_SWEEP_REVERSAL",
            "ATM_MOMENTUM_BURST",
        ):
            if regime in ("TRENDING", "RANGE", "VOLATILE"):
                score = 80
            elif regime == "CHOP":
                score = 10
            else:
                score = 40

        elif strategy in ("FVG_RETEST",):
            if regime in ("TRENDING", "RANGE"):
                score = 80
            elif regime == "VOLATILE":
                score = 60
            else:
                score = 15

        elif strategy in ("PRE_EVENT_STRADDLE",):
            if regime == "EVENT":
                score = 95
            else:
                score = 25

        return round(score, 2)

    def _score_mtf(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        direction = str(opportunity.get("direction", "")).upper()
        mtf_ctx = _extract_mtf_context(context)
        weighted_mtf = _safe_float(mtf_ctx.get("weighted_mtf", context.get("weighted_mtf", 0)), 0.0)
        mtf_trap_zone = _coerce_bool(mtf_ctx.get("mtf_trap_zone", context.get("mtf_trap_zone", False)), False)
        confluence_bonus_applied = _coerce_bool(
            mtf_ctx.get("confluence_bonus_applied", context.get("confluence_bonus_applied", False)),
            False,
        )
        trend_strength_with_confluence = _safe_float(
            mtf_ctx.get(
                "trend_strength_with_confluence",
                context.get("trend_strength_with_confluence", 0),
            ),
            0.0,
        )

        score = 10.0

        if direction == "BUY":
            if weighted_mtf >= 6.0:
                score = 95
            elif weighted_mtf >= 4.5:
                score = 85
            elif weighted_mtf >= 3.0:
                score = 70
            elif weighted_mtf >= 1.0:
                score = 45
            elif weighted_mtf > -1.0:
                score = 25
            else:
                score = 5
        else:
            if weighted_mtf <= -6.0:
                score = 95
            elif weighted_mtf <= -4.5:
                score = 85
            elif weighted_mtf <= -3.0:
                score = 70
            elif weighted_mtf <= -1.0:
                score = 45
            elif weighted_mtf < 1.0:
                score = 25
            else:
                score = 5

        if mtf_trap_zone:
            score *= self._mtf_trap_zone_multiplier
        elif confluence_bonus_applied and trend_strength_with_confluence >= self._mtf_confluence_bonus_threshold:
            score = min(100.0, score + self._mtf_confluence_bonus)

        return round(score, 2)

    def _score_smart_money(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        direction = str(opportunity.get("direction", "")).upper()
        sm5 = context.get("smart_money_5m", {}) or {}
        sm15 = context.get("smart_money_15m", {}) or {}
        if not isinstance(sm5, dict):
            sm5 = {}
        if not isinstance(sm15, dict):
            sm15 = {}

        score = 10.0

        structure_dir_5 = str(sm5.get("structure_direction", "NEUTRAL")).upper().strip()
        structure_dir_15 = str(sm15.get("structure_direction", "NEUTRAL")).upper().strip()
        sm_score_5 = _safe_float(sm5.get("sm_direction_score", 0), 0.0)
        sm_score_15 = _safe_float(sm15.get("sm_direction_score", 0), 0.0)

        if direction == "BUY":
            if structure_dir_5 == "BULLISH":
                score += 15
            if structure_dir_15 == "BULLISH":
                score += 15
            if sm_score_5 > 20:
                score += 10
            if sm_score_15 > 20:
                score += 10
        else:
            if structure_dir_5 == "BEARISH":
                score += 15
            if structure_dir_15 == "BEARISH":
                score += 15
            if sm_score_5 < -20:
                score += 10
            if sm_score_15 < -20:
                score += 10

        return round(min(score, 100.0), 2)

    def _score_narrative(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        fit_factor = _safe_float(context.get("narrative_fit_factor", 0.8), 0.0)
        score = fit_factor / 1.2 * 100 if fit_factor > 0 else 0
        return round(min(max(score, 0.0), 100.0), 2)

    def _score_time(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        session = str(context.get("session_phase", "UNKNOWN")).upper().strip()
        size_mult = _safe_float(context.get("size_multiplier", 0), 0.0)

        if session in ("GOLDEN_AM", "GOLDEN_PM"):
            score = 90
        elif session == "INITIAL_BALANCE":
            score = 70
        elif session == "CLOSING_SESSION":
            score = 45
        elif session == "LUNCH_LULL":
            score = 20
        elif session in ("PRE_MARKET", "OPENING_AUCTION", "OR_FORMATION", "LAST_MINUTES", "POST_MARKET"):
            score = 0
        else:
            score = max(0.0, min(size_mult * 100, 70))

        return round(score, 2)

    def _score_trap_safety(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> float:
        trap_probability = _safe_float(context.get("trap_probability", 0.0), 0.0)
        score = (1.0 - trap_probability) * 100
        return round(min(max(score, 0.0), 100.0), 2)


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Opportunity Scorer Test (Corrected)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    scorer = OpportunityScorer()

    print(" [Test 1] Clean structural BUY setup...")
    opp1 = {
        "strategy": "VWAP_PULLBACK",
        "direction": "BUY",
        "entry_price": 23200.0,
        "raw_score": 75,
        "risk_reward": 2.5,
    }
    ctx1 = {
        "regime": "TRENDING",
        "session_phase": "GOLDEN_AM",
        "size_multiplier": 1.0,
        "narrative_fit_factor": 1.0,
        "trap_probability": 0.10,
        "data_quality_score": 85,
        "weighted_mtf": 5.5,
        "same_direction_signals": 2,
        "features_1m": {
            "rsi": 48,
            "volume_ratio": 0.9,
            "macd_histogram": 2.0,
            "macd_hist_slope": 0.8,
            "roc_5": 0.06,
            "candle_body_ratio": 0.45,
            "upper_wick_ratio": 0.1,
            "lower_wick_ratio": 0.3,
        },
        "microstructure": {"spread_zscore": 0.4},
        "key_levels": {
            "pdh": 23300,
            "pdl": 23050,
            "sr_zones": [{"level": 23200, "strength": 3, "type": "support"}],
        },
        "volume_profile": {"poc": 23205, "vah": 23280, "val": 23120},
        "options": {
            "pcr_oi": 1.1,
            "synthetic_premium": 5,
            "gex_regime": "NEGATIVE",
            "highest_ce_oi_strike": 23450,
            "highest_pe_oi_strike": 23050,
        },
        "smart_money_5m": {
            "structure_direction": "BULLISH",
            "sm_direction_score": 30,
            "nearest_fvg_direction": "BULLISH",
            "nearest_fvg_distance": 8,
            "nearest_ob_direction": "BULLISH",
        },
        "smart_money_15m": {
            "structure_direction": "BULLISH",
            "sm_direction_score": 35,
            "nearest_ob_direction": "BULLISH",
        },
    }
    r1 = scorer.score_opportunity(opp1, ctx1)
    if not r1.hard_reject and r1.final_score >= 58:
        print(f" ✅ final_score={r1.final_score}, bonuses={r1.bonuses}")
        passed += 1
    else:
        print(f" ❌ Unexpected score result: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Hard reject from trap probability...")
    ctx2 = dict(ctx1)
    ctx2["trap_probability"] = 0.8
    r2 = scorer.score_opportunity(opp1, ctx2)
    if r2.hard_reject and "trap_probability_high" in r2.reject_reason:
        print(f" ✅ hard reject: {r2.reject_reason}")
        passed += 1
    else:
        print(f" ❌ Trap reject failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Hard reject from blocked session...")
    ctx3 = dict(ctx1)
    ctx3["session_phase"] = "LAST_MINUTES"
    r3 = scorer.score_opportunity(opp1, ctx3)
    if r3.hard_reject and "time_context_blocked" in r3.reject_reason:
        print(f" ✅ blocked session reject: {r3.reject_reason}")
        passed += 1
    else:
        print(f" ❌ Session reject failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Regime-adaptive scoring effect...")
    opp4 = {
        "strategy": "SR_REJECTION",
        "direction": "BUY",
        "entry_price": 23050.0,
        "raw_score": 85,
        "risk_reward": 2.5,
    }
    ctx4 = dict(ctx1)
    ctx4["regime"] = "RANGE"
    r4 = scorer.score_opportunity(opp4, ctx4)
    if r4.factor_scores["regime_match"] >= 80:
        print(f" ✅ range strategy in RANGE gets strong regime score={r4.factor_scores['regime_match']}")
        passed += 1
    else:
        print(f" ❌ Regime scoring weak: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Empty/minimal context safety...")
    opp5 = {
        "strategy": "UNKNOWN",
        "direction": "BUY",
        "entry_price": 100,
        "raw_score": 50,
        "risk_reward": 1.0,
    }
    ctx5 = {
        "regime": "UNKNOWN",
        "session_phase": "GOLDEN_AM",
        "size_multiplier": 1.0,
        "narrative_fit_factor": 0.8,
        "trap_probability": 0.0,
        "data_quality_score": 80,
        "weighted_mtf": 0.0,
        "features_1m": {},
        "microstructure": {},
        "key_levels": {},
        "volume_profile": {},
        "options": {},
        "smart_money_5m": {},
        "smart_money_15m": {},
    }
    r5 = scorer.score_opportunity(opp5, ctx5)
    if isinstance(r5.final_score, float):
        print(f" ✅ minimal context safe, final_score={r5.final_score}")
        passed += 1
    else:
        print(" ❌ Minimal context unsafe")
        failed += 1

    print("\n [Test 6] Component floor hard reject...")
    ctx6 = dict(ctx1)
    ctx6["features_1m"] = {
        "rsi": 90,
        "volume_ratio": 0.2,
        "macd_histogram": -5,
        "macd_hist_slope": -3,
        "roc_5": -0.1,
        "candle_body_ratio": 0.1,
        "upper_wick_ratio": 0.7,
        "lower_wick_ratio": 0.05,
    }
    r6 = scorer.score_opportunity(opp1, ctx6)
    if r6.hard_reject:
        print(f" ✅ component floor reject: {r6.reject_reason}")
        passed += 1
    else:
        print(" ❌ Expected component reject")
        failed += 1

    print("\n [Test 7] Phase-3 MTF confluence bonus...")
    ctx7 = dict(ctx1)
    ctx7["mtf"] = {
        "weighted_mtf": 5.5,
        "mtf_trap_zone": False,
        "confluence_bonus_applied": True,
        "trend_strength_with_confluence": 78.0,
    }
    r7 = scorer.score_opportunity(opp1, ctx7)
    if r7.factor_scores["mtf_alignment"] > r1.factor_scores["mtf_alignment"]:
        print(f" ✅ phase-3 confluence lift: {r1.factor_scores['mtf_alignment']} -> {r7.factor_scores['mtf_alignment']}")
        passed += 1
    else:
        print(f" ❌ Expected confluence bonus to lift mtf_alignment: base={r1.factor_scores['mtf_alignment']}, phase3={r7.factor_scores['mtf_alignment']}")
        failed += 1

    print("\n [Test 8] Phase-3 MTF trap zone attenuation...")
    ctx8 = dict(ctx7)
    ctx8["mtf"] = dict(ctx7["mtf"])
    ctx8["mtf"]["mtf_trap_zone"] = True
    r8 = scorer.score_opportunity(opp1, ctx8)
    if r8.factor_scores["mtf_alignment"] < r7.factor_scores["mtf_alignment"]:
        print(f" ✅ trap zone reduced mtf_alignment: {r7.factor_scores['mtf_alignment']} -> {r8.factor_scores['mtf_alignment']}")
        passed += 1
    else:
        print(f" ❌ Expected trap-zone attenuation: phase3={r7.factor_scores['mtf_alignment']}, trap={r8.factor_scores['mtf_alignment']}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Opportunity Scorer (Corrected) working perfectly!")
        print(" ✅ Ready for next roadmap step.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()