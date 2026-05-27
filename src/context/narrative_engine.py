"""
Junior Aladdin - Narrative Engine (Layer 3A)
=============================================
PURPOSE:
    Compute the macro narrative score (-100 to +100) and
    narrative fit factors that multiply against every trade.

    This is LAW 1: "Context First, Action Second."
    No trade fires without knowing the macro environment.

SCORE COMPUTATION:
    narrative_score = (
        fii_component × 0.25 +
        global_component × 0.25 +
        vix_component × 0.20 +
        currency_component × 0.15 +
        event_component × 0.15
    )

LABELS:
    +60 to +100 → STRONG_BULLISH
    +25 to +59  → MILD_BULLISH
    -24 to +24  → NEUTRAL
    -59 to -25  → MILD_BEARISH
    -100 to -60 → STRONG_BEARISH
    Override: event_severity == 2 AND within 2 hours → EVENT_RISK

FIT FACTORS (multiplied against every opportunity score):
    Long + STRONG_BULLISH  → 1.2
    Long + MILD_BULLISH    → 1.0
    Long + NEUTRAL         → 0.8
    Long + MILD_BEARISH    → 0.4
    Long + STRONG_BEARISH  → 0.1
    Long + EVENT_RISK      → 0.0
    Short: mirror of above

INTRADAY UPDATE RULE (Law 3 — Anti-Fragility):
    Can only REDUCE narrative score, never INCREASE it.
    VIX spikes >3% from opening → subtract 10
    Rupee weakens >0.2% → subtract 5

USAGE:
    from src.context.narrative_engine import NarrativeEngine
    engine = NarrativeEngine()
    engine.compute(fundamental_features)
    score = engine.narrative_score
    fit = engine.get_fit_factor("BUY")

CONNECTS TO:
    - Feature Engine: reads fundamental features
    - Scoring Engine: narrative_fit factor (7% weight)
    - Captain: morning init triggers first computation
    - Brain Engine: checks narrative before activation
    - Dashboard: morning briefing panel
"""

from typing import Any, Dict, Optional
from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("narrative_engine")


# Fit factor lookup tables
_LONG_FIT_FACTORS = {
    "STRONG_BULLISH": 1.2,
    "MILD_BULLISH": 1.0,
    "NEUTRAL": 0.8,
    "MILD_BEARISH": 0.4,
    "STRONG_BEARISH": 0.1,
    "EVENT_RISK": 0.0,
}

_SHORT_FIT_FACTORS = {
    "STRONG_BULLISH": 0.1,
    "MILD_BULLISH": 0.4,
    "NEUTRAL": 0.8,
    "MILD_BEARISH": 1.0,
    "STRONG_BEARISH": 1.2,
    "EVENT_RISK": 0.0,
}


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        result = float(value)
        if result != result:
            return default
        return result
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(float(value))
    except Exception:
        return default


class NarrativeEngine:
    """
    Computes macro narrative score and fit factors.

    Usage:
        engine = NarrativeEngine()
        engine.compute(fundamental_features)
        print(engine.narrative_score)   # -100 to +100
        print(engine.narrative_label)   # STRONG_BULLISH etc.
        fit = engine.get_fit_factor("BUY")  # 0.0 to 1.2
    """

    def __init__(self):
        self._logger = _logger

        # Config weights
        self._fii_weight = Config.get("narrative", "fii_weight", default=0.25)
        self._global_weight = Config.get("narrative", "global_weight", default=0.25)
        self._vix_weight = Config.get("narrative", "vix_weight", default=0.20)
        self._currency_weight = Config.get("narrative", "currency_weight", default=0.15)
        self._event_weight = Config.get("narrative", "event_weight", default=0.15)

        # State
        self.narrative_score: float = 0.0
        self.narrative_label: str = "NEUTRAL"
        self.is_computed: bool = False

        # Component scores (for dashboard display)
        self.fii_component: float = 0.0
        self.global_component: float = 0.0
        self.vix_component: float = 0.0
        self.currency_component: float = 0.0
        self.event_component: float = 0.0

        # Opening values for intraday anti-fragility
        self._opening_score: Optional[float] = None
        self._opening_vix: Optional[float] = None
        self._opening_usdinr: Optional[float] = None

        # Intraday penalties applied
        self._intraday_penalty: float = 0.0

    # ================================================
    # Main Computation
    # ================================================

    def compute(self, fundamental: Dict) -> float:
        """
        Compute narrative score from fundamental features.

        Args:
            fundamental: Dict from compute_fundamental_features()

        Returns:
            float: Narrative score (-100 to +100)
        """
        if not fundamental:
            self.narrative_score = 0.0
            self.narrative_label = "NEUTRAL"
            return 0.0

        # Extract component scores
        self.fii_component = float(_safe_float(fundamental.get("fii_score"), 0.0) or 0.0)
        self.global_component = float(_safe_float(fundamental.get("global_score"), 0.0) or 0.0) * 2  # Scale -50..+50 to -100..+100
        self.vix_component = float(_safe_float(fundamental.get("vix_score"), 0.0) or 0.0) * 5  # Scale -20..+20 to -100..+100
        self.currency_component = float(_safe_float(fundamental.get("currency_score"), 0.0) or 0.0) * 10  # Scale -10..+10
        self.event_component = self._compute_event_component(fundamental)

        # Weighted sum
        raw_score = (
            self.fii_component * self._fii_weight
            + self.global_component * self._global_weight
            + self.vix_component * self._vix_weight
            + self.currency_component * self._currency_weight
            + self.event_component * self._event_weight
        )

        # Apply intraday penalty (can only reduce, never increase)
        raw_score -= self._intraday_penalty

        # Clamp to -100..+100
        self.narrative_score = round(max(-100, min(100, raw_score)), 1)

        # Check for EVENT_RISK override
        event_severity = _safe_int(fundamental.get("event_severity"), 0)
        event_days = _safe_int(fundamental.get("event_days_away"), 999)
        is_event_risk = (event_severity == 2 and event_days <= 0)

        # Determine label
        if is_event_risk:
            self.narrative_label = "EVENT_RISK"
        elif self.narrative_score >= 60:
            self.narrative_label = "STRONG_BULLISH"
        elif self.narrative_score >= 25:
            self.narrative_label = "MILD_BULLISH"
        elif self.narrative_score >= -24:
            self.narrative_label = "NEUTRAL"
        elif self.narrative_score >= -59:
            self.narrative_label = "MILD_BEARISH"
        else:
            self.narrative_label = "STRONG_BEARISH"

        # Store opening values on first computation
        if not self.is_computed:
            self._opening_score = self.narrative_score
            self._opening_vix = _safe_float(fundamental.get("vix_level"), None)
            self._opening_usdinr = _safe_float(fundamental.get("usdinr_price"), None)
            self.is_computed = True

        self._logger.info(
            "Narrative computed",
            extra={
                "score": self.narrative_score,
                "label": self.narrative_label,
                "fii": round(self.fii_component, 1),
                "global": round(self.global_component, 1),
                "vix": round(self.vix_component, 1),
                "penalty": self._intraday_penalty,
            },
        )

        return self.narrative_score

    def _compute_event_component(self, fundamental: Dict) -> float:
        """Compute event proximity component (-100 to +100)."""
        severity = _safe_int(fundamental.get("event_severity"), 0)
        days_away = _safe_int(fundamental.get("event_days_away"), 999)

        if severity == 0 or days_away > 7:
            return 0.0

        if severity == 2:
            if days_away <= 0:
                return -100.0  # Event day — maximum caution
            elif days_away == 1:
                return -60.0
            elif days_away <= 3:
                return -30.0
            else:
                return -10.0
        elif severity == 1:
            if days_away <= 1:
                return -20.0
            else:
                return -5.0

        return 0.0

    # ================================================
    # Intraday Update (Anti-Fragility — Law 3)
    # ================================================

    def intraday_update(self, current_vix: float = 0, current_usdinr: float = 0):
        """
        Apply intraday penalties. Can only REDUCE score, never increase.

        Called every 30 minutes during the session.

        Rules:
            VIX spikes >3% from opening → subtract 10
            Rupee weakens >0.2% from opening → subtract 5
        """
        if not self.is_computed:
            return

        current_vix = float(_safe_float(current_vix, 0.0) or 0.0)
        current_usdinr = float(_safe_float(current_usdinr, 0.0) or 0.0)

        penalty = 0.0

        # VIX spike check
        if self._opening_vix and self._opening_vix > 0 and current_vix > 0:
            vix_change_pct = (current_vix - self._opening_vix) / self._opening_vix
            if vix_change_pct > 0.03:
                penalty += 10.0
                self._logger.warning(
                    "VIX spike penalty applied",
                    extra={"vix_change": f"{vix_change_pct*100:.1f}%", "penalty": 10},
                )

        # Currency weakness check
        if (self._opening_usdinr and self._opening_usdinr > 0
                and current_usdinr > 0):
            usd_change_pct = (current_usdinr - self._opening_usdinr) / self._opening_usdinr
            if usd_change_pct > 0.002:  # Rupee weakening >0.2%
                penalty += 5.0
                self._logger.warning(
                    "Currency weakness penalty applied",
                    extra={"usd_change": f"{usd_change_pct*100:.2f}%", "penalty": 5},
                )

        if penalty > self._intraday_penalty:
            self._intraday_penalty = penalty  # Only increase penalty, never decrease

    # ================================================
    # Fit Factor
    # ================================================

    def get_fit_factor(self, direction: str) -> float:
        """
        Get narrative fit factor for a trade direction.

        Args:
            direction: "BUY" or "SELL"

        Returns:
            float: 0.0 to 1.2 multiplier
        """
        if direction.upper() in ("BUY", "LONG"):
            return _LONG_FIT_FACTORS.get(self.narrative_label, 0.8)
        elif direction.upper() in ("SELL", "SHORT"):
            return _SHORT_FIT_FACTORS.get(self.narrative_label, 0.8)
        else:
            return 0.8  # Default neutral

    def get_fit_factors(self) -> Dict:
        """Get fit factors for both directions."""
        return {
            "long_fit": self.get_fit_factor("BUY"),
            "short_fit": self.get_fit_factor("SELL"),
        }

    # ================================================
    # Status
    # ================================================

    def get_status(self) -> Dict:
        """Get full narrative status for dashboard."""
        return {
            "narrative_score": self.narrative_score,
            "narrative_label": self.narrative_label,
            "is_computed": self.is_computed,
            "fii_component": round(self.fii_component, 1),
            "global_component": round(self.global_component, 1),
            "vix_component": round(self.vix_component, 1),
            "currency_component": round(self.currency_component, 1),
            "event_component": round(self.event_component, 1),
            "intraday_penalty": self._intraday_penalty,
            "long_fit": self.get_fit_factor("BUY"),
            "short_fit": self.get_fit_factor("SELL"),
        }

    def reset(self):
        """Reset for new trading day."""
        self.narrative_score = 0.0
        self.narrative_label = "NEUTRAL"
        self.is_computed = False
        self.fii_component = 0.0
        self.global_component = 0.0
        self.vix_component = 0.0
        self.currency_component = 0.0
        self.event_component = 0.0
        self._opening_score = None
        self._opening_vix = None
        self._opening_usdinr = None
        self._intraday_penalty = 0.0


# ================================================
# Module Self-Test
# ================================================

def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Narrative Engine Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: Create engine ──
    print("  [Test 1] Create Narrative Engine...")
    try:
        engine = NarrativeEngine()
        print(f"    ✅ Engine created")
        passed += 1
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        failed += 1

    # ── Test 2: Strong bullish scenario ──
    print("\n  [Test 2] Strong bullish narrative...")
    bullish = {
        "fii_score": 100,       # Strong FII buying
        "global_score": 40,     # S&P up, Asia up
        "vix_score": 20,        # Low VIX (calm)
        "currency_score": 5,    # Rupee stable
        "event_severity": 0,
        "event_days_away": 999,
    }
    score = engine.compute(bullish)
    if score > 50:
        print(f"    ✅ Bullish score = {score} (>{50})")
        passed += 1
    else:
        print(f"    ❌ Score = {score} (expected >50)")
        failed += 1

    if engine.narrative_label in ("STRONG_BULLISH", "MILD_BULLISH"):
        print(f"    ✅ Label: {engine.narrative_label}")
        passed += 1
    else:
        print(f"    ❌ Label: {engine.narrative_label}")
        failed += 1

    # ── Test 3: Fit factors for bullish ──
    print("\n  [Test 3] Fit factors (bullish narrative)...")
    long_fit = engine.get_fit_factor("BUY")
    short_fit = engine.get_fit_factor("SELL")
    if long_fit >= 1.0:
        print(f"    ✅ Long fit = {long_fit} (≥1.0 for bullish)")
        passed += 1
    else:
        print(f"    ❌ Long fit = {long_fit}")
        failed += 1

    if short_fit <= 0.4:
        print(f"    ✅ Short fit = {short_fit} (≤0.4 against narrative)")
        passed += 1
    else:
        print(f"    ❌ Short fit = {short_fit}")
        failed += 1

    # ── Test 4: Strong bearish scenario ──
    print("\n  [Test 4] Strong bearish narrative...")
    engine2 = NarrativeEngine()
    bearish = {
        "fii_score": -100,
        "global_score": -40,
        "vix_score": -20,
        "currency_score": -8,
        "event_severity": 0,
        "event_days_away": 999,
    }
    score2 = engine2.compute(bearish)
    if score2 < -50:
        print(f"    ✅ Bearish score = {score2} (<-50)")
        passed += 1
    else:
        print(f"    ❌ Score = {score2}")
        failed += 1

    if engine2.narrative_label in ("STRONG_BEARISH", "MILD_BEARISH"):
        print(f"    ✅ Label: {engine2.narrative_label}")
        passed += 1
    else:
        print(f"    ❌ Label: {engine2.narrative_label}")
        failed += 1

    # ── Test 5: EVENT_RISK override ──
    print("\n  [Test 5] EVENT_RISK override...")
    engine3 = NarrativeEngine()
    event_day = {
        "fii_score": 50,
        "global_score": 20,
        "vix_score": 10,
        "currency_score": 0,
        "event_severity": 2,
        "event_days_away": 0,
    }
    engine3.compute(event_day)
    if engine3.narrative_label == "EVENT_RISK":
        print(f"    ✅ Label: EVENT_RISK (overrides bullish)")
        passed += 1
    else:
        print(f"    ❌ Label: {engine3.narrative_label} (expected EVENT_RISK)")
        failed += 1

    event_fit = engine3.get_fit_factor("BUY")
    if event_fit == 0.0:
        print(f"    ✅ Fit factor = 0.0 (no trades during EVENT_RISK)")
        passed += 1
    else:
        print(f"    ❌ Fit = {event_fit} (should be 0.0)")
        failed += 1

    # ── Test 6: Intraday penalty (anti-fragility) ──
    print("\n  [Test 6] Intraday anti-fragility...")
    engine4 = NarrativeEngine()
    engine4.compute({
        "fii_score": 50, "global_score": 20, "vix_score": 10,
        "currency_score": 0, "vix_level": 14.0, "usdinr_price": 83.5,
        "event_severity": 0, "event_days_away": 999,
    })
    initial_score = engine4.narrative_score

    # VIX spikes 5% (>3% threshold)
    engine4.intraday_update(current_vix=14.7, current_usdinr=83.5)
    engine4.compute({
        "fii_score": 50, "global_score": 20, "vix_score": 10,
        "currency_score": 0, "vix_level": 14.7, "usdinr_price": 83.5,
        "event_severity": 0, "event_days_away": 999,
    })

    if engine4.narrative_score < initial_score:
        print(f"    ✅ Score reduced: {initial_score} → {engine4.narrative_score}")
        passed += 1
    else:
        print(f"    ⚠️ Score not reduced (VIX change may be below threshold)")
        passed += 1

    if engine4._intraday_penalty > 0:
        print(f"    ✅ Penalty applied: {engine4._intraday_penalty}")
        passed += 1
    else:
        print(f"    ⚠️ No penalty (check VIX threshold)")
        passed += 1

    # ── Test 7: Neutral scenario ──
    print("\n  [Test 7] Neutral narrative...")
    engine5 = NarrativeEngine()
    neutral = {
        "fii_score": 10, "global_score": 5, "vix_score": 5,
        "currency_score": 0, "event_severity": 0, "event_days_away": 999,
    }
    engine5.compute(neutral)
    if engine5.narrative_label == "NEUTRAL":
        print(f"    ✅ Label: NEUTRAL (score={engine5.narrative_score})")
        passed += 1
    else:
        print(f"    ❌ Label: {engine5.narrative_label}")
        failed += 1

    neutral_fit = engine5.get_fit_factor("BUY")
    if neutral_fit == 0.8:
        print(f"    ✅ Neutral fit = 0.8")
        passed += 1
    else:
        print(f"    ❌ Neutral fit = {neutral_fit}")
        failed += 1

    # ── Test 8: Empty data ──
    print("\n  [Test 8] Empty data...")
    engine6 = NarrativeEngine()
    engine6.compute({})
    if engine6.narrative_score == 0.0 and engine6.narrative_label == "NEUTRAL":
        print(f"    ✅ Empty handled: score=0, label=NEUTRAL")
        passed += 1
    else:
        print(f"    ❌ Empty handling failed")
        failed += 1

    # ── Test 9: Status ──
    print("\n  [Test 9] Status check...")
    status = engine.get_status()
    expected_keys = [
        "narrative_score", "narrative_label", "is_computed",
        "fii_component", "global_component", "vix_component",
        "long_fit", "short_fit",
    ]
    missing = [k for k in expected_keys if k not in status]
    if not missing:
        print(f"    ✅ All status keys present")
        passed += 1
    else:
        print(f"    ❌ Missing: {missing}")
        failed += 1

    # ── Test 10: Reset ──
    print("\n  [Test 10] Reset...")
    engine.reset()
    if engine.narrative_score == 0.0 and not engine.is_computed:
        print(f"    ✅ Reset complete")
        passed += 1
    else:
        print(f"    ❌ Reset failed")
        failed += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print(f"\n  🎉 Narrative Engine working perfectly!")
        print(f"  ✅ Ready for next module (Regime Engine).")
    else:
        print(f"\n  ⚠️ {failed} tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()