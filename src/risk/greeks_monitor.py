"""
Junior Aladdin - Greeks Monitor
===============================
PURPOSE:
Monitor portfolio-level Greek exposure from current open positions.

This file exists because the roadmap explicitly expects:
    src/risk/greeks_monitor.py

It provides a clean risk-layer interface for:
- net delta
- net gamma
- optional vega / theta aggregation
- directional blocking recommendations
- concentration warnings

WHY THIS MODULE MATTERS:
A system may have individually valid trades but still become dangerous at the
portfolio level if too many positions stack in one direction.

This module does NOT size or execute.
It only evaluates current exposure and produces warnings / blocks.

CONNECTS TO:
- Risk Engine
- Options Features / option chain
- Position Manager
- Captain / dashboard
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Any, Optional

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("greeks_monitor")


@dataclass
class GreeksExposure:
    """
    Standard portfolio Greek exposure report.
    """
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    gross_positions: int = 0
    block_new_longs: bool = False
    block_new_shorts: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_theta": self.net_theta,
            "net_vega": self.net_vega,
            "gross_positions": self.gross_positions,
            "block_new_longs": self.block_new_longs,
            "block_new_shorts": self.block_new_shorts,
            "warnings": self.warnings,
        }


class GreeksMonitor:
    """
    Aggregates portfolio Greeks and determines directional overexposure.
    """

    def __init__(self):
        self._logger = _logger
        delta_block_threshold_raw = Config.get("risk", "delta_block_threshold", default=500.0)
        gamma_warning_threshold_raw = Config.get("risk", "gamma_warning_threshold", default=50.0)

        self._delta_block_threshold = self._safe_float(delta_block_threshold_raw, default=500.0)
        if self._delta_block_threshold <= 0:
            self._delta_block_threshold = 500.0

        self._gamma_warning_threshold = self._safe_float(gamma_warning_threshold_raw, default=50.0)
        if self._gamma_warning_threshold <= 0:
            self._gamma_warning_threshold = 50.0

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

    def evaluate(
        self,
        open_positions: List[Dict[str, Any]],
        option_chain: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        lot_size: Optional[int] = None,
    ) -> GreeksExposure:
        """
        Evaluate current portfolio Greek exposure.

        Expected position fields:
        - symbol
        - qty
        - direction ("BUY"/"SELL")
        - strike (optional if symbol lookup not available)
        - option_type ("CE"/"PE") optional
        - expiry optional

        Option chain expected:
        {
            strike: {
                "ce": {"delta","gamma","theta","vega",...},
                "pe": {...}
            }
        }
        """
        lot_size = lot_size or Config.get("market", "lot_size", default=65)

        if not isinstance(open_positions, list):
            warnings = ["invalid_open_positions_type"]
            self._logger.warning(
                "Invalid open_positions type in Greeks monitor",
                extra={"received_type": type(open_positions).__name__},
            )
            return GreeksExposure(
                net_delta=0.0,
                net_gamma=0.0,
                net_theta=0.0,
                net_vega=0.0,
                gross_positions=0,
                block_new_longs=False,
                block_new_shorts=False,
                warnings=warnings,
            )

        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        warnings: List[str] = []

        for pos in open_positions:
            if not isinstance(pos, dict):
                warning = "skipped_position_invalid_type:UNKNOWN"
                warnings.append(warning)
                self._logger.warning(
                    "Skipping invalid position type in Greeks monitor",
                    extra={"warning": warning, "received_type": type(pos).__name__},
                )
                continue

            symbol = str(pos.get("symbol", "UNKNOWN"))
            qty = self._safe_float(pos.get("qty", 0), default=0.0)
            direction = str(pos.get("direction", "BUY")).strip().upper()
            strike = self._safe_int(pos.get("strike", 0), default=0)
            option_type = str(pos.get("option_type", "")).strip().upper()

            if qty <= 0 or strike <= 0 or option_type not in ("CE", "PE"):
                warning = f"skipped_position_missing_greek_keys:{symbol}"
                warnings.append(warning)
                self._logger.warning(
                    "Skipping position with missing/invalid Greek keys",
                    extra={
                        "warning": warning,
                        "symbol": symbol,
                        "qty": qty,
                        "strike": strike,
                        "option_type": option_type,
                    },
                )
                continue

            side_mult = 1.0 if direction == "BUY" else -1.0

            greek_row = self._get_greek_row(option_chain, strike, option_type)
            if greek_row is None:
                warning = f"missing_option_chain_greeks:{symbol}"
                warnings.append(warning)
                self._logger.warning(
                    "Missing option chain Greeks for position",
                    extra={"warning": warning, "symbol": symbol, "strike": strike, "option_type": option_type},
                )
                continue

            delta = self._safe_float(greek_row.get("delta", 0.0), default=0.0)
            gamma = self._safe_float(greek_row.get("gamma", 0.0), default=0.0)
            theta = self._safe_float(greek_row.get("theta", 0.0), default=0.0)
            vega = self._safe_float(greek_row.get("vega", 0.0), default=0.0)

            # qty already expected as actual quantity, not lots
            mult = qty * side_mult

            net_delta += delta * mult
            net_gamma += gamma * mult
            net_theta += theta * mult
            net_vega += vega * mult

        if math.isnan(net_delta) or math.isinf(net_delta):
            self._logger.warning("Non-finite net_delta detected; resetting to 0.0")
            warnings.append("non_finite_net_delta_reset")
            net_delta = 0.0

        if math.isnan(net_gamma) or math.isinf(net_gamma):
            self._logger.warning("Non-finite net_gamma detected; resetting to 0.0")
            warnings.append("non_finite_net_gamma_reset")
            net_gamma = 0.0

        if math.isnan(net_theta) or math.isinf(net_theta):
            self._logger.warning("Non-finite net_theta detected; resetting to 0.0")
            warnings.append("non_finite_net_theta_reset")
            net_theta = 0.0

        if math.isnan(net_vega) or math.isinf(net_vega):
            self._logger.warning("Non-finite net_vega detected; resetting to 0.0")
            warnings.append("non_finite_net_vega_reset")
            net_vega = 0.0

        net_delta = round(net_delta, 2)
        net_gamma = round(net_gamma, 4)
        net_theta = round(net_theta, 2)
        net_vega = round(net_vega, 2)

        block_new_longs = net_delta > self._delta_block_threshold
        block_new_shorts = net_delta < -self._delta_block_threshold

        if block_new_longs:
            warnings.append("delta_too_long_block_new_longs")
        if block_new_shorts:
            warnings.append("delta_too_short_block_new_shorts")

        if abs(net_gamma) > self._gamma_warning_threshold:
            warnings.append("high_gamma_exposure")

        exposure = GreeksExposure(
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            gross_positions=len(open_positions or []),
            block_new_longs=block_new_longs,
            block_new_shorts=block_new_shorts,
            warnings=warnings,
        )

        self._logger.info(
            "Greeks exposure computed",
            extra={
                "net_delta": net_delta,
                "net_gamma": net_gamma,
                "net_theta": net_theta,
                "net_vega": net_vega,
                "gross_positions": exposure.gross_positions,
                "block_new_longs": block_new_longs,
                "block_new_shorts": block_new_shorts,
                "warnings": warnings,
            },
        )

        return exposure

    def _get_greek_row(
        self,
        option_chain: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
        strike: int,
        option_type: str,
    ) -> Optional[Dict[str, Any]]:
        if not option_chain:
            return None

        strike_row = option_chain.get(strike)
        if not isinstance(strike_row, dict):
            return None

        side_key = "ce" if option_type == "CE" else "pe"
        row = strike_row.get(side_key)
        if not isinstance(row, dict):
            return None

        return row


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Greeks Monitor Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    gm = GreeksMonitor()

    test_chain = {
        22450: {
            "ce": {"delta": 0.52, "gamma": 0.003, "theta": -8.5, "vega": 12.3},
            "pe": {"delta": -0.48, "gamma": 0.003, "theta": -7.8, "vega": 11.9},
        },
        22500: {
            "ce": {"delta": 0.40, "gamma": 0.0025, "theta": -7.0, "vega": 10.1},
            "pe": {"delta": -0.60, "gamma": 0.0027, "theta": -8.2, "vega": 12.8},
        },
    }

    print(" [Test 1] Empty positions...")
    r1 = gm.evaluate([], test_chain)
    if r1.net_delta == 0 and r1.gross_positions == 0:
        print(f" ✅ Empty portfolio handled: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Empty portfolio failed: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Single long CE position...")
    positions2 = [
        {"symbol": "NIFTY22450CE", "qty": 65, "direction": "BUY", "strike": 22450, "option_type": "CE"}
    ]
    r2 = gm.evaluate(positions2, test_chain)
    if r2.net_delta > 0 and r2.gross_positions == 1:
        print(f" ✅ Long CE exposure valid: delta={r2.net_delta}, gamma={r2.net_gamma}")
        passed += 1
    else:
        print(f" ❌ Long CE exposure wrong: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Single long PE position...")
    positions3 = [
        {"symbol": "NIFTY22450PE", "qty": 65, "direction": "BUY", "strike": 22450, "option_type": "PE"}
    ]
    r3 = gm.evaluate(positions3, test_chain)
    if r3.net_delta < 0 and r3.gross_positions == 1:
        print(f" ✅ Long PE exposure valid: delta={r3.net_delta}")
        passed += 1
    else:
        print(f" ❌ Long PE exposure wrong: {r3.to_dict()}")
        failed += 1

    print("\n [Test 4] Short CE flips sign...")
    positions4 = [
        {"symbol": "NIFTY22450CE", "qty": 65, "direction": "SELL", "strike": 22450, "option_type": "CE"}
    ]
    r4 = gm.evaluate(positions4, test_chain)
    if r4.net_delta < 0:
        print(f" ✅ Short CE flips delta sign: {r4.net_delta}")
        passed += 1
    else:
        print(f" ❌ Short CE sign wrong: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Portfolio aggregation...")
    positions5 = [
        {"symbol": "NIFTY22450CE", "qty": 65, "direction": "BUY", "strike": 22450, "option_type": "CE"},
        {"symbol": "NIFTY22450PE", "qty": 65, "direction": "BUY", "strike": 22450, "option_type": "PE"},
    ]
    r5 = gm.evaluate(positions5, test_chain)
    if r5.gross_positions == 2:
        print(f" ✅ Aggregation works: delta={r5.net_delta}, gamma={r5.net_gamma}, positions={r5.gross_positions}")
        passed += 1
    else:
        print(f" ❌ Aggregation failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Missing chain data warning...")
    positions6 = [
        {"symbol": "NIFTY99999CE", "qty": 65, "direction": "BUY", "strike": 99999, "option_type": "CE"}
    ]
    r6 = gm.evaluate(positions6, test_chain)
    if "missing_option_chain_greeks:NIFTY99999CE" in r6.warnings:
        print(f" ✅ Missing chain warning works: {r6.warnings}")
        passed += 1
    else:
        print(f" ❌ Missing chain warning failed: {r6.to_dict()}")
        failed += 1

    print("\n [Test 7] Missing keys warning...")
    positions7 = [
        {"symbol": "BROKENPOS", "qty": 65, "direction": "BUY"}
    ]
    r7 = gm.evaluate(positions7, test_chain)
    if "skipped_position_missing_greek_keys:BROKENPOS" in r7.warnings:
        print(f" ✅ Missing-key warning works: {r7.warnings}")
        passed += 1
    else:
        print(f" ❌ Missing-key warning failed: {r7.to_dict()}")
        failed += 1

    print("\n [Test 8] Long delta block...")
    positions8 = [
        {"symbol": "NIFTY22450CE", "qty": 2000, "direction": "BUY", "strike": 22450, "option_type": "CE"}
    ]
    r8 = gm.evaluate(positions8, test_chain)
    if r8.block_new_longs:
        print(f" ✅ Long block works: delta={r8.net_delta}")
        passed += 1
    else:
        print(f" ❌ Long block failed: {r8.to_dict()}")
        failed += 1

    print("\n [Test 9] Short delta block...")
    positions9 = [
        {"symbol": "NIFTY22450PE", "qty": 2000, "direction": "BUY", "strike": 22450, "option_type": "PE"}
    ]
    r9 = gm.evaluate(positions9, test_chain)
    if r9.block_new_shorts:
        print(f" ✅ Short block works: delta={r9.net_delta}")
        passed += 1
    else:
        print(f" ❌ Short block failed: {r9.to_dict()}")
        failed += 1

    print("\n [Test 10] High gamma warning...")
    high_gamma_chain = {
        22450: {
            "ce": {"delta": 0.52, "gamma": 1.0, "theta": -8.5, "vega": 12.3},
            "pe": {"delta": -0.48, "gamma": 1.0, "theta": -7.8, "vega": 11.9},
        }
    }
    positions10 = [
        {"symbol": "NIFTY22450CE", "qty": 100, "direction": "BUY", "strike": 22450, "option_type": "CE"}
    ]
    r10 = gm.evaluate(positions10, high_gamma_chain)
    if "high_gamma_exposure" in r10.warnings:
        print(f" ✅ High gamma warning works: {r10.warnings}")
        passed += 1
    else:
        print(f" ❌ High gamma warning failed: {r10.to_dict()}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Greeks Monitor working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()