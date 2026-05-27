"""
Strategy Engine (Layer 6)

Central orchestrator for all 13 trading strategies.

Strict boundaries:
- Delegates to existing strategy classes only (no trading logic here).
- No trap detection, scoring, risk sizing, or execution.
- Stateless across scans (no session memory / caches).
- Must never raise exceptions from scan(); all errors are caught and logged.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Type


try:
    from src.utils.logger import setup_logger  # type: ignore
except Exception as e:  # pragma: no cover
    # Phase 1 hardening: system must not run without the proper structured logger.
    raise ImportError(
        "Failed to import src.utils.logger.setup_logger. "
        "StrategyEngine requires the project's structured logger to be available."
    ) from e


log = setup_logger("strategy_engine")  # type: ignore[misc]


# StrategyBase / Opportunity types
try:
    from src.strategies.strategy_base import Opportunity, StrategyBase  # type: ignore
except Exception as e:  # pragma: no cover
    log.error(
        "Failed to import StrategyBase/Opportunity from src.strategies.strategy_base",
        error=str(e),
        traceback=traceback.format_exc(),
    )
    StrategyBase = object  # type: ignore[assignment,misc]
    Opportunity = Any  # type: ignore[misc,assignment]


def _safe_import(path: str, cls_name: str) -> Optional[Type[Any]]:
    """
    Import `cls_name` from module `path`.

    If import fails, logs the error and returns None.
    """
    try:
        module_obj = __import__(path, fromlist=[cls_name])
        cls = getattr(module_obj, cls_name)

        # Optional: improved observability on successful import
        try:
            log.debug(
                "Strategy import succeeded",
                module_path=path,
                class_name=cls_name,
            )
        except Exception:
            pass

        return cls
    except Exception as e:
        log.error(
            "Strategy import failed",
            module_path=path,  # avoid reserved 'module' field collision
            class_name=cls_name,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        return None


_VWAPPullbackStrategy = _safe_import(
    "src.strategies.vwap_pullback", "VWAPPullbackStrategy"
)
_TrendContinuationStrategy = _safe_import(
    "src.strategies.trend_continuation", "TrendContinuationStrategy"
)
_OpeningRangeBreakoutStrategy = _safe_import(
    "src.strategies.opening_range_breakout", "OpeningRangeBreakoutStrategy"
)
_SRRejectionStrategy = _safe_import("src.strategies.sr_rejection", "SRRejectionStrategy")
_VolumeProfilePOCStrategy = _safe_import(
    "src.strategies.vol_profile_poc", "VolumeProfilePOCStrategy"
)
_StopHuntReclaimStrategy = _safe_import(
    "src.strategies.stop_hunt_reclaim", "StopHuntReclaimStrategy"
)
_OIWallBounceStrategy = _safe_import(
    "src.strategies.oi_wall_bounce", "OIWallBounceStrategy"
)
_ATMMomentumStrategy = _safe_import(
    "src.strategies.atm_momentum", "ATMMomentumBurstStrategy"
)
_FailedBreakoutReversalStrategy = _safe_import(
    "src.strategies.failed_breakout", "FailedBreakoutReversalStrategy"
)
_AbsorptionReversalStrategy = _safe_import(
    "src.strategies.absorption_reversal", "AbsorptionReversalStrategy"
)
_FVGRetestStrategy = _safe_import("src.strategies.fvg_retest", "FVGRetestStrategy")
_LiquiditySweepReversalStrategy = _safe_import(
    "src.strategies.liquidity_sweep", "LiquiditySweepReversalStrategy"
)
_PreEventStraddleStrategy = _safe_import(
    "src.strategies.pre_event_straddle", "PreEventStraddleStrategy"
)


_StrategyClass = Type[Any]


def _safe_repr(value: Any, limit: int = 200) -> str:
    try:
        s = repr(value)
    except Exception:
        s = "<unrepr-able>"
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_valid_opportunity(obj: Any) -> bool:
    """
    Validate an opportunity object to protect downstream layers.

    Valid items must be either:
      a) an instance of Opportunity (from strategy_base), OR
      b) a dict with required keys AND semantic checks.

    Semantic checks (dict):
      - entry_price, sl_price, target_price > 0
      - raw_score between 0 and 100 inclusive
      - direction is BUY or SELL (case-insensitive)
    """
    # Case A: Opportunity instance
    try:
        if isinstance(Opportunity, type) and isinstance(obj, Opportunity):  # type: ignore[arg-type]
            # Best-effort semantic validation if fields exist; otherwise accept as valid.
            try:
                direction = getattr(obj, "direction", None)
                if direction is not None:
                    d = str(direction).strip().upper()
                    if d == "LONG":
                        d = "BUY"
                    elif d == "SHORT":
                        d = "SELL"
                    if d not in {"BUY", "SELL"}:
                        return False

                entry_price = getattr(obj, "entry_price", None)
                sl_price = getattr(obj, "sl_price", None)
                target_price = getattr(obj, "target_price", None)
                raw_score = getattr(obj, "raw_score", None)

                # If any of these are present, validate them.
                if entry_price is not None and (not _is_number(entry_price) or float(entry_price) <= 0):
                    return False
                if sl_price is not None and (not _is_number(sl_price) or float(sl_price) <= 0):
                    return False
                if target_price is not None and (not _is_number(target_price) or float(target_price) <= 0):
                    return False
                if raw_score is not None:
                    if not _is_number(raw_score):
                        return False
                    rs = float(raw_score)
                    if rs < 0 or rs > 100:
                        return False
            except Exception:
                # Never fail validation due to attribute access errors; consider it invalid instead of raising.
                return False

            return True
    except Exception:
        # If Opportunity isn't a real runtime type, fall through to dict validation only.
        pass

    # Case B: dict with required keys and semantic checks
    if isinstance(obj, dict):
        required_keys = {
            "strategy",
            "direction",
            "entry_price",
            "sl_price",
            "target_price",
            "raw_score",
        }
        if not required_keys.issubset(set(obj.keys())):
            return False

        direction = obj.get("direction")
        d = str(direction).strip().upper() if direction is not None else ""
        if d == "LONG":
            d = "BUY"
        elif d == "SHORT":
            d = "SELL"
        if d not in {"BUY", "SELL"}:
            return False

        entry_price = obj.get("entry_price")
        sl_price = obj.get("sl_price")
        target_price = obj.get("target_price")
        raw_score = obj.get("raw_score")

        if not _is_number(entry_price) or float(entry_price) <= 0:
            return False
        if not _is_number(sl_price) or float(sl_price) <= 0:
            return False
        if not _is_number(target_price) or float(target_price) <= 0:
            return False
        if not _is_number(raw_score):
            return False

        rs = float(raw_score)
        if rs < 0 or rs > 100:
            return False

        return True

    return False


class StrategyEngine:
    """
    Central scanning orchestrator for all concrete strategies.

    Responsibilities:
    - Maintain a single registry of strategy instances.
    - Scan strategies safely and collect Opportunity objects.
    - Provide filters by brain and direction.
    - Ensure scan() never raises.

    This class must not:
    - Perform trap detection, scoring, risk sizing, or execution.
    - Keep state between scans (beyond a static registry of strategy instances).
    """

    def __init__(self) -> None:
        self._strategies: List[StrategyBase] = []
        self._register_strategies()

    def _register_strategies(self) -> None:
        """
        Instantiate and register all available strategy classes.

        Notes:
        - Missing imports are already logged during module load via _safe_import().
        - Constructor failures are caught and logged here; registry continues.
        """
        strategy_classes: Tuple[Optional[_StrategyClass], ...] = (
            _VWAPPullbackStrategy,
            _TrendContinuationStrategy,
            _OpeningRangeBreakoutStrategy,
            _SRRejectionStrategy,
            _VolumeProfilePOCStrategy,
            _StopHuntReclaimStrategy,
            _OIWallBounceStrategy,
            _ATMMomentumStrategy,
            _FailedBreakoutReversalStrategy,
            _AbsorptionReversalStrategy,
            _FVGRetestStrategy,
            _LiquiditySweepReversalStrategy,
            _PreEventStraddleStrategy,
        )

        for cls in strategy_classes:
            if cls is None:
                continue

            try:
                instance = cls()
                self._strategies.append(instance)
            except Exception as e:
                log.error(
                    "Strategy instantiation failed",
                    class_name=getattr(cls, "__name__", str(cls)),
                    error=str(e),
                    traceback=traceback.format_exc(),
                )

        log.info("Strategy registry initialized", registered_count=len(self._strategies))

        if len(self._strategies) == 0:
            log.critical("Strategy registry is empty; no strategies available.")

    @property
    def strategies(self) -> Tuple[StrategyBase, ...]:
        """Read-only view of registered strategies."""
        return tuple(self._strategies)

    def get_registered_strategies(self) -> List[Dict[str, str]]:
        """
        Return lightweight registry metadata for audit/introspection.

        Each entry includes name, brain, and class to keep reporting stable.
        """
        registered: List[Dict[str, str]] = []
        for strat in self._strategies:
            try:
                name = getattr(strat, "name", None)
                name_s = str(name).strip() if name is not None else strat.__class__.__name__
            except Exception:
                name_s = strat.__class__.__name__

            try:
                brain = getattr(strat, "brain", None)
                brain_s = str(brain).strip() if brain is not None else "UNKNOWN"
            except Exception:
                brain_s = "UNKNOWN"

            registered.append(
                {
                    "name": name_s,
                    "brain": brain_s,
                    "class": strat.__class__.__name__,
                }
            )

        return registered

    def scan(
        self,
        features_1m: Dict[str, Any],
        features_5m: Optional[Dict[str, Any]] = None,
        features_15m: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        brain_filter: Optional[List[str]] = None,
        direction_filter: Optional[str] = None,
    ) -> List[Opportunity]:
        """
        Scan appropriate strategies and return a combined list of Opportunity objects.

        Error handling contract:
            - This method never raises.
            - Invalid/missing context => warning + [].
            - Missing/empty features_1m => debug + proceed (context-only strategies allowed).
            - Per-strategy exceptions => error + continue.
        """
        try:
            if len(self._strategies) == 0:
                try:
                    log.error("StrategyEngine.scan called but strategy registry is empty")
                except Exception:
                    pass
                return []

            # STRICT FILTER VALIDATION (Fail-Closed)
            if brain_filter is not None and not isinstance(brain_filter, list):
                try:
                    log.error(
                        "Invalid brain_filter type; expected list[str]. Failing closed.",
                        brain_filter_type=str(type(brain_filter)),
                    )
                except Exception:
                    pass
                return []

            if direction_filter is not None:
                normalized_direction_check = str(direction_filter).strip().upper()
                if normalized_direction_check not in {"BUY", "SELL"}:
                    try:
                        log.error(
                            "Invalid direction_filter; expected BUY/SELL. Failing closed.",
                            direction_filter=direction_filter,
                        )
                    except Exception:
                        pass
                    return []

            # Proceed even if missing/empty to allow context-only strategies.
            if features_1m is None or not isinstance(features_1m, dict) or not features_1m:
                try:
                    log.debug(
                        "features_1m missing; some strategies may not produce signals",
                        features_1m_type=str(type(features_1m)),
                        features_1m_empty=isinstance(features_1m, dict) and not bool(features_1m),
                    )
                except Exception:
                    pass
                features_1m = {} if not isinstance(features_1m, dict) else features_1m

            if context is None or not isinstance(context, dict) or not context:
                try:
                    log.warning("StrategyEngine.scan called with invalid context", value=context)
                except Exception:
                    pass
                return []

            normalized_brains: Optional[Set[str]] = None
            if brain_filter is not None:
                normalized_brains = {str(b).strip().upper() for b in brain_filter if str(b).strip()}
                if not normalized_brains:
                    normalized_brains = None

            normalized_direction: Optional[str] = None
            if direction_filter is not None:
                normalized_direction = str(direction_filter).strip().upper()

            try:
                log.info(
                    "Strategy scan started",
                    registered_strategies=len(self._strategies),
                    brain_filter=list(normalized_brains) if normalized_brains is not None else None,
                    direction_filter=normalized_direction,
                )
            except Exception:
                pass

            opportunities: List[Opportunity] = []
            any_strategy_matched_brain = False

            for strategy in self._strategies:
                strategy_name = strategy.__class__.__name__
                try:
                    # brain filtering (case-insensitive)
                    if normalized_brains is not None:
                        strat_brain = getattr(strategy, "brain", None)
                        if strat_brain is None:
                            continue

                        strat_brain_u = str(strat_brain).strip().upper()
                        if strat_brain_u not in normalized_brains:
                            continue
                        any_strategy_matched_brain = True

                    safe_scan = getattr(strategy, "safe_scan", None)
                    if safe_scan is None or not callable(safe_scan):
                        try:
                            log.error(
                                "Strategy missing callable safe_scan; skipping",
                                strategy_class=strategy_name,
                            )
                        except Exception:
                            pass
                        continue

                    result = safe_scan(features_1m, features_5m, features_15m, context)

                    if result is None:
                        continue

                    # OUTPUT VALIDATION: only accept valid Opportunity objects/dicts.
                    if isinstance(result, (list, tuple)):
                        for item in result:
                            if _is_valid_opportunity(item):
                                opportunities.append(item)
                            else:
                                try:
                                    log.warning(
                                        "Invalid opportunity object discarded",
                                        strategy_class=strategy_name,
                                        opportunity_repr=_safe_repr(item),
                                    )
                                except Exception:
                                    pass
                    else:
                        if _is_valid_opportunity(result):
                            opportunities.append(result)
                        else:
                            try:
                                log.warning(
                                    "Invalid opportunity object discarded",
                                    strategy_class=strategy_name,
                                    opportunity_repr=_safe_repr(result),
                                )
                            except Exception:
                                pass

                except Exception as e:
                    try:
                        log.error(
                            "Strategy scan failed (guarded)",
                            strategy_class=strategy_name,
                            error=str(e),
                            traceback=traceback.format_exc(),
                        )
                    except Exception:
                        pass
                    continue

            if normalized_brains is not None and not any_strategy_matched_brain:
                try:
                    log.debug(
                        "No strategies matched brain_filter (ignored as per contract)",
                        brain_filter=list(normalized_brains),
                    )
                except Exception:
                    pass

            # Direction filter applied on final list (case-insensitive)
            if normalized_direction is not None:
                filtered: List[Opportunity] = []
                malformed_logged = 0
                malformed_log_limit = 20

                for opp in opportunities:
                    try:
                        opp_dir = getattr(opp, "direction", None)
                        if opp_dir is None and isinstance(opp, dict):
                            opp_dir = opp.get("direction")

                        if opp_dir is None:
                            continue

                        if str(opp_dir).strip().upper() == normalized_direction:
                            filtered.append(opp)
                    except Exception as e:
                        if malformed_logged < malformed_log_limit:
                            malformed_logged += 1
                            try:
                                log.warning(
                                    "Malformed opportunity encountered during direction filtering; discarded",
                                    error=str(e),
                                    opportunity_repr=_safe_repr(opp),
                                )
                            except Exception:
                                pass
                        continue

                opportunities = filtered

            try:
                log.info("Strategy scan completed", opportunities=len(opportunities))
            except Exception:
                pass

            return opportunities

        except Exception as e:
            try:
                log.error(
                    "StrategyEngine.scan failed (top-level guarded)",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            except Exception:
                pass
            return []


if __name__ == "__main__":
    try:
        engine = StrategyEngine()
        print(f"Registered strategies: {len(engine.strategies)}")

        if len(engine.strategies) != 13:
            print(f"ERROR: Expected 13 strategies, got {len(engine.strategies)}")
            sys.exit(1)

        # Minimal dummy inputs to ensure scan() never raises
        features_1m_dummy: Dict[str, Any] = {"last_close": 100.0, "atr": 1.0}
        context_dummy: Dict[str, Any] = {"regime": "UNKNOWN"}

        _ = engine.scan(features_1m=features_1m_dummy, context=context_dummy)

        # Second mock scan with empty features_1m should not raise (and typically returns empty).
        result_empty = engine.scan(features_1m={}, context=context_dummy)
        if not isinstance(result_empty, list):
            print("ERROR: scan() did not return a list")
            sys.exit(1)

        print("Strategy Engine self-test PASSED.")
    except Exception:
        print("ERROR: Strategy Engine self-test FAILED.")
        traceback.print_exc()
        sys.exit(1)