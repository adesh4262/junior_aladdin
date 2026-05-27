"""
Junior Aladdin - Risk Engine
============================
PURPOSE:
Final capital-preservation layer before execution.

This module validates whether a trade should be allowed from a portfolio-risk
point of view and computes safe size recommendations using the plan's hard rules.

HARD RULES:
- Max risk per trade: 0.5% of capital
- Max daily loss: 2% -> LOCKED
- Max trades/day: 5
- Max consecutive losses before pause: 3

SEQUENTIAL SIZE MULTIPLIERS:
1. GARCH / volatility multiplier
2. Drawdown multiplier
3. Tilt multiplier
4. Session multiplier
5. Expiry multiplier
6. Score multiplier

CONNECTS TO:
- Behavioral Sentinel
- Opportunity Scorer
- Time Context
- Captain
- Position Manager
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Any, Optional, List, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("risk_engine")


@dataclass
class RiskDecision:
    """
    Final risk decision for one opportunity.
    """
    allow_trade: bool
    block_reason: str = ""
    recommended_lots: int = 0
    recommended_qty: int = 0
    estimated_risk_rupees: float = 0.0
    max_allowed_risk_rupees: float = 0.0
    size_multipliers: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_trade": self.allow_trade,
            "block_reason": self.block_reason,
            "recommended_lots": self.recommended_lots,
            "recommended_qty": self.recommended_qty,
            "estimated_risk_rupees": self.estimated_risk_rupees,
            "max_allowed_risk_rupees": self.max_allowed_risk_rupees,
            "size_multipliers": self.size_multipliers,
            "warnings": self.warnings,
        }


class RiskEngine:
    """
    Portfolio risk engine with hard limits and adaptive sizing.
    """

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_float_strict(value: Any, default: float) -> Tuple[float, bool]:
        if value is None:
            return default, True
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default, False
            return v, True
        except Exception:
            return default, False

    def __init__(self):
        self._logger = _logger

        self._lot_size = self._safe_float(
            Config.get("market", "lot_size", default=65),
            65.0,
        )

        self._max_risk_per_trade_pct = self._safe_float(
            Config.get("risk", "max_risk_per_trade_pct", default=0.005),
            0.005,
        )
        self._max_daily_loss_pct = self._safe_float(
            Config.get("risk", "max_daily_loss_pct", default=0.02),
            0.02,
        )
        self._max_trades_per_day = self._safe_int(
            Config.get("risk", "max_trades_per_day", default=5),
            5,
        )
        self._max_consecutive_losses = self._safe_int(
            Config.get("risk", "max_consecutive_losses", default=3),
            3,
        )

        self._drawdown_caution_pct = self._safe_float(
            Config.get("risk", "drawdown_caution_pct", default=0.03),
            0.03,
        )
        self._drawdown_reduce_pct = self._safe_float(
            Config.get("risk", "drawdown_reduce_pct", default=0.05),
            0.05,
        )
        self._drawdown_half_pct = self._safe_float(
            Config.get("risk", "drawdown_half_pct", default=0.08),
            0.08,
        )
        self._drawdown_lock_pct = self._safe_float(
            Config.get("risk", "drawdown_lock_pct", default=0.10),
            0.10,
        )

    def _block_decision(self, block_reason: str, **kwargs: Any) -> RiskDecision:
        self._logger.warning(
            "Risk trade blocked",
            extra={"block_reason": block_reason},
        )
        return RiskDecision(
            allow_trade=False,
            block_reason=block_reason,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RiskDecision:
        """
        Evaluate one scored opportunity and return safe size / approval.
        """
        if not isinstance(context, dict):
            return self._block_decision("invalid_context")

        if not isinstance(opportunity, dict):
            return self._block_decision("invalid_opportunity")

        capital, capital_valid = self._safe_float_strict(context.get("capital"), 50000.0)
        daily_pnl = self._safe_float(context.get("daily_pnl"), 0.0)
        trades_today = self._safe_int(context.get("trades_today"), 0)
        consecutive_losses = self._safe_int(context.get("consecutive_losses"), 0)
        drawdown_pct = self._safe_float(context.get("drawdown_pct"), 0.0)
        tilt_score = self._safe_float(context.get("tilt_score"), 0.0)
        score = self._safe_float(
            opportunity.get("final_score", opportunity.get("raw_score")),
            0.0,
        )
        session_phase = str(context.get("session_phase", "UNKNOWN"))
        is_expiry_day = bool(context.get("is_expiry_day", False))
        expiry_size_factor = self._safe_float(context.get("expiry_size_factor"), 1.0)

        entry_price = self._safe_float(opportunity.get("entry_price"), 0.0)
        sl_price = self._safe_float(opportunity.get("sl_price"), 0.0)
        risk_points = abs(entry_price - sl_price)

        warnings: List[str] = []

        if (not capital_valid) or capital <= 0:
            return self._block_decision("invalid_capital")

        if entry_price <= 0 or sl_price <= 0 or risk_points <= 0:
            return self._block_decision("invalid_entry_or_sl")

        # --------------------------------------------------------------
        # Hard blocks
        # --------------------------------------------------------------
        if trades_today >= self._max_trades_per_day:
            return self._block_decision("max_trades_reached")

        if consecutive_losses >= self._max_consecutive_losses:
            return self._block_decision("max_consecutive_losses_reached")

        if abs(min(daily_pnl, 0.0)) >= capital * self._max_daily_loss_pct:
            return self._block_decision("daily_loss_limit_hit")

        if drawdown_pct >= self._drawdown_lock_pct:
            return self._block_decision("drawdown_lock")

        max_allowed_risk_rupees = round(capital * self._max_risk_per_trade_pct, 2)

        # --------------------------------------------------------------
        # Sequential multipliers
        # --------------------------------------------------------------
        size_multipliers = {
            "volatility": self._volatility_multiplier(context, warnings),
            "drawdown": self._drawdown_multiplier(drawdown_pct, warnings),
            "tilt": self._tilt_multiplier(tilt_score, warnings),
            "session": self._session_multiplier(session_phase, warnings),
            "expiry": self._expiry_multiplier(is_expiry_day, expiry_size_factor, warnings),
            "score": self._score_multiplier(score, warnings),
        }

        # --- GARCH HIGH VOLATILITY MULTIPLIER (NEW) ---
        if context.get("garch_high_vol"):
            garch_mult = float(Config.get("risk", "garch_high_vol_multiplier", default=0.5))
            size_multipliers["garch"] = garch_mult
            warnings.append("garch_high_volatility")
        # ---------------------------------------------

        combined_multiplier = 1.0
        for value in size_multipliers.values():
            combined_multiplier *= value

        # --------------------------------------------------------------
        # Base lot sizing
        # --------------------------------------------------------------
        risk_per_lot = risk_points * self._lot_size
        if risk_per_lot <= 0:
            return self._block_decision("invalid_risk_per_lot")

        raw_base_lots = max_allowed_risk_rupees / risk_per_lot
        adjusted_lots = int(raw_base_lots * combined_multiplier)

        recommended_lots = max(1, adjusted_lots) if raw_base_lots > 0 else 0
        recommended_qty = recommended_lots * self._lot_size
        estimated_risk_rupees = round(recommended_lots * risk_per_lot, 2)

        # --------------------------------------------------------------
        # Final validation
        # --------------------------------------------------------------
        if recommended_lots <= 0:
            return self._block_decision(
                "sizing_result_zero",
                max_allowed_risk_rupees=max_allowed_risk_rupees,
                size_multipliers=size_multipliers,
                warnings=warnings,
            )

        # strict rule: if even 1 lot breaches allowed risk, block
        if risk_per_lot > max_allowed_risk_rupees * 1.05:
            return self._block_decision(
                "one_lot_exceeds_risk_budget",
                max_allowed_risk_rupees=max_allowed_risk_rupees,
                estimated_risk_rupees=round(risk_per_lot, 2),
                size_multipliers=size_multipliers,
                warnings=warnings,
            )

        if estimated_risk_rupees > max_allowed_risk_rupees:
            warnings.append("recommended_size_near_limit")

        decision = RiskDecision(
            allow_trade=True,
            recommended_lots=recommended_lots,
            recommended_qty=recommended_qty,
            estimated_risk_rupees=estimated_risk_rupees,
            max_allowed_risk_rupees=max_allowed_risk_rupees,
            size_multipliers=size_multipliers,
            warnings=warnings,
        )

        self._logger.info(
            "Risk evaluation complete",
            extra={
                "strategy": opportunity.get("strategy", "UNKNOWN"),
                "entry_price": entry_price,
                "sl_price": sl_price,
                "risk_points": risk_points,
                "lots": recommended_lots,
                "estimated_risk_rupees": estimated_risk_rupees,
                "allow_trade": decision.allow_trade,
                "warnings": warnings,
            },
        )

        return decision

    # ------------------------------------------------------------------
    # Multipliers
    # ------------------------------------------------------------------
    def _volatility_multiplier(self, context: Dict[str, Any], warnings: List[str]) -> float:
        result = 1.0
        atr_percentile = context.get("atr_percentile")
        if atr_percentile is not None:
            try:
                val = self._safe_float(atr_percentile, 0.0)
                if val > 90:
                    warnings.append("very_high_volatility")
                    result = 0.5
                elif val > 80:
                    warnings.append("high_volatility")
                    result = 0.75
            except Exception:
                pass
        return max(0.0, min(1.0, result))

    def _drawdown_multiplier(self, drawdown_pct: float, warnings: List[str]) -> float:
        result = 1.0
        if drawdown_pct >= self._drawdown_half_pct:
            warnings.append("drawdown_half_size")
            result = 0.5
        elif drawdown_pct >= self._drawdown_reduce_pct:
            warnings.append("drawdown_reduced_size")
            result = 0.75
        elif drawdown_pct >= self._drawdown_caution_pct:
            warnings.append("drawdown_caution")
            result = 0.9
        return max(0.0, min(1.0, result))

    def _tilt_multiplier(self, tilt_score: float, warnings: List[str]) -> float:
        result = 1.0
        if tilt_score > 85:
            warnings.append("tilt_extreme")
            result = 0.0
        elif tilt_score > 70:
            warnings.append("tilt_half_size")
            result = 0.5
        elif tilt_score > 50:
            warnings.append("tilt_reduce")
            result = 0.75
        return max(0.0, min(1.0, result))

    def _session_multiplier(self, session_phase: str, warnings: List[str]) -> float:
        result = 1.0
        if session_phase in ("LUNCH_LULL", "CLOSING_SESSION"):
            warnings.append("session_size_reduced")
            result = 0.5
        return max(0.0, min(1.0, result))

    def _expiry_multiplier(
        self,
        is_expiry_day: bool,
        expiry_size_factor: float,
        warnings: List[str],
    ) -> float:
        result = 1.0
        if is_expiry_day:
            warnings.append("expiry_size_adjustment")
            result = max(0.1, min(expiry_size_factor, 1.0))
        return max(0.0, min(1.0, result))

    def _score_multiplier(self, score: float, warnings: List[str]) -> float:
        result = 0.4
        if score > 80:
            result = 1.0
        elif score >= 70:
            result = 0.8
        elif score >= 60:
            result = 0.6
        else:
            warnings.append("low_score_reduced")
        return max(0.0, min(1.0, result))

    # ------------------------------------------------------------------
    # Status / metadata
    # ------------------------------------------------------------------
    def get_limits(self) -> Dict[str, Any]:
        return {
            "lot_size": self._lot_size,
            "max_risk_per_trade_pct": self._max_risk_per_trade_pct,
            "max_daily_loss_pct": self._max_daily_loss_pct,
            "max_trades_per_day": self._max_trades_per_day,
            "max_consecutive_losses": self._max_consecutive_losses,
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Risk Engine Test (Corrected)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    engine = RiskEngine()

    # baseline VALID trade:
    # 3 points risk * 65 = 195 rupees < 250 max risk
    valid_opp = {
        "strategy": "VWAP_PULLBACK",
        "entry_price": 23200.0,
        "sl_price": 23197.0,
        "final_score": 82,
    }

    base_ctx = {
        "capital": 50000,
        "daily_pnl": 0,
        "trades_today": 1,
        "consecutive_losses": 0,
        "drawdown_pct": 0.01,
        "tilt_score": 10,
        "session_phase": "GOLDEN_AM",
        "is_expiry_day": False,
        "expiry_size_factor": 1.0,
        "atr_percentile": 50,
    }

    print(" [Test 1] Clean valid trade...")
    r1 = engine.evaluate(valid_opp, base_ctx)
    if r1.allow_trade and r1.recommended_lots >= 1:
        print(
            f" allow_trade={r1.allow_trade}, "
            f"lots={r1.recommended_lots}, risk={r1.estimated_risk_rupees}"
        )
        passed += 1
    else:
        print(f" Unexpected decision: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Block when one lot exceeds risk budget...")
    high_risk_opp = {
        "strategy": "VWAP_PULLBACK",
        "entry_price": 23200.0,
        "sl_price": 23180.0,  # 20 * 65 = 1300
        "final_score": 85,
    }
    r2 = engine.evaluate(high_risk_opp, base_ctx)
    if not r2.allow_trade and r2.block_reason == "one_lot_exceeds_risk_budget":
        print(f" blocked: {r2.block_reason}")
        passed += 1
    else:
        print(f" Should block: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Block on max trades/day...")
    ctx3 = dict(base_ctx)
    ctx3["trades_today"] = 5
    r3 = engine.evaluate(valid_opp, ctx3)
    if not r3.allow_trade and r3.block_reason == "max_trades_reached":
        print(f" blocked: {r3.block_reason}")
        passed += 1
    else:
        print(f" Max-trades block failed: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Block on daily loss...")
    ctx4 = dict(base_ctx)
    ctx4["daily_pnl"] = -1200  # > 2% of 50k
    r4 = engine.evaluate(valid_opp, ctx4)
    if not r4.allow_trade and r4.block_reason == "daily_loss_limit_hit":
        print(f" blocked: {r4.block_reason}")
        passed += 1
    else:
        print(f" Daily-loss block failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Drawdown reduction works...")
    ctx5 = dict(base_ctx)
    ctx5["drawdown_pct"] = 0.06
    r5 = engine.evaluate(valid_opp, ctx5)
    if r5.allow_trade and r5.size_multipliers.get("drawdown") == 0.75:
        print(f" drawdown multiplier={r5.size_multipliers['drawdown']}")
        passed += 1
    else:
        print(f" Drawdown sizing failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Tilt reduction works...")
    ctx6 = dict(base_ctx)
    ctx6["tilt_score"] = 75
    r6 = engine.evaluate(valid_opp, ctx6)
    if r6.allow_trade and r6.size_multipliers.get("tilt") == 0.5:
        print(f" tilt multiplier={r6.size_multipliers['tilt']}")
        passed += 1
    else:
        print(f" Tilt sizing failed: {r6.to_dict()}")
        failed += 1

    print("\n [Test 7] Lunch session reduction works...")
    ctx7 = dict(base_ctx)
    ctx7["session_phase"] = "LUNCH_LULL"
    r7 = engine.evaluate(valid_opp, ctx7)
    if r7.allow_trade and r7.size_multipliers.get("session") == 0.5:
        print(f" session multiplier={r7.size_multipliers['session']}")
        passed += 1
    else:
        print(f" Session sizing failed: {r7.to_dict()}")
        failed += 1

    print("\n [Test 8] Expiry reduction works...")
    ctx8 = dict(base_ctx)
    ctx8["is_expiry_day"] = True
    ctx8["expiry_size_factor"] = 0.7
    r8 = engine.evaluate(valid_opp, ctx8)
    if r8.allow_trade and r8.size_multipliers.get("expiry") == 0.7:
        print(f" expiry multiplier={r8.size_multipliers['expiry']}")
        passed += 1
    else:
        print(f" Expiry sizing failed: {r8.to_dict()}")
        failed += 1

    print("\n [Test 9] Low score reduces size...")
    opp9 = dict(valid_opp)
    opp9["final_score"] = 62
    r9 = engine.evaluate(opp9, base_ctx)
    if r9.allow_trade and r9.size_multipliers.get("score") == 0.6:
        print(f" score multiplier={r9.size_multipliers['score']}")
        passed += 1
    else:
        print(f" Score multiplier failed: {r9.to_dict()}")
        failed += 1

    print("\n [Test 10] Invalid entry/sl blocked...")
    opp10 = {"strategy": "X", "entry_price": 0, "sl_price": 0, "final_score": 80}
    r10 = engine.evaluate(opp10, base_ctx)
    if not r10.allow_trade and r10.block_reason == "invalid_entry_or_sl":
        print(f" blocked: {r10.block_reason}")
        passed += 1
    else:
        print(f" Invalid-entry block failed: {r10.to_dict()}")
        failed += 1

    print("\n [Test 11] Limits structure...")
    limits = engine.get_limits()
    if limits["lot_size"] == 65 and limits["max_trades_per_day"] == 5:
        print(f" limits={limits}")
        passed += 1
    else:
        print(f" Limits wrong: {limits}")
        failed += 1

    print("\n [Test 12] Invalid context data blocked safely...")
    ctx12 = dict(base_ctx)
    ctx12["capital"] = "not_a_number"
    r12 = engine.evaluate(valid_opp, ctx12)
    if not r12.allow_trade and r12.block_reason == "invalid_capital":
        print(f" blocked: {r12.block_reason}")
        passed += 1
    else:
        print(f" Invalid-context block failed: {r12.to_dict()}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n Risk Engine (Corrected) working perfectly!")
        print(" Ready for next roadmap step.")
    else:
        print(f"\n {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()