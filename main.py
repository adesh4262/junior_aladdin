# FILE: main.py
"""
Junior Aladdin — Main Bootstrap Orchestrator
==========================================

(This file is large; only minimal critical fixes applied per audit)

CRITICAL FIXES APPLIED (Builder Task 1):
- Use DataEngine private attribute `_candle_builder` (not `candle_builder`) wherever CandleBuilder is accessed.
- Pass `token=` into CandleBuilder.get_last_closed(...) for token-based candle retrieval.
- Ensure health heartbeat candle-count visibility uses `_candle_builder` as well.

All architecture, adapters, and pipeline structure preserved.
"""

from __future__ import annotations

import argparse
import inspect
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


# -----------------------------------------------------------------------------
# Ensure project root is importable (fixes "No module named 'src'" in non-packaged runs)
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Runtime Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeArgs:
    mode: str
    config_path: str
    confirm_live: bool
    health_interval_sec: float
    candle_poll_interval_sec: float


class StartupError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
class JuniorAladdinOrchestrator:
    """
    Production-oriented bootstrap engine.

    This class owns:
    - component lifecycle
    - runtime candle-close loop
    - health monitoring
    - shutdown semantics
    """

    def __init__(self, args: RuntimeArgs):
        self._args = args

        # Lazy-import logger and config so sys.path patch above is applied first
        from src.utils.config_loader import Config
        from src.utils.logger import setup_logger

        # 1) Config
        self._config = Config
        self._config.load(args.config_path)

        # 2) Logger
        self._log = setup_logger("main")
        self._log.info("Configuration loaded", config_path=args.config_path, mode=args.mode)

        self._validate_config()
        self._validate_mode()

        # Shutdown control
        self._stop_event = threading.Event()
        self._started = False
        self._shutdown_lock = threading.Lock()

        # Components (filled in initialize_components)
        self.state = None

        self.feed_health = None
        self.tick_validator = None
        self.candle_builder = None
        self.option_chain_poller = None
        self.data_engine = None

        self.feature_engine = None
        self.narrative_engine = None
        self.regime_engine = None
        self.time_context_engine = None
        self.market_dna_engine = None

        self.captain_engine = None
        self.strategy_engine = None
        self.trap_detector = None
        self.opportunity_scorer = None

        self.ml_filter = None
        self.anomaly_detector = None
        self.garch_forecaster = None

        self.risk_engine = None
        self.order_validator = None
        self.broker = None  # paper or live broker

        # Candle-close tracking
        self._last_1m_closed_ts = None

        # Heartbeat tracking
        self._last_health_log_mono = 0.0

        # Dashboard control-plane publisher (Part 1 stabilization).
        # What changed: backend now owns the shared-memory heartbeat and
        # kill-switch resources that the dashboard already expects.
        self._control_plane = None

        # Dashboard IPC bridge (Part 2A stabilization).
        # Scope: real snapshot transport + real command channel only.
        self._dashboard_ipc = None

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_config(self) -> None:
        required = [
            ("market", "index"),
            ("market", "lot_size"),
            ("market", "market_open"),
            ("market", "market_close"),
            ("compliance", "allowed_order_types"),
        ]
        missing = []
        for keys in required:
            v = self._config.get(*keys, default=None)
            if v is None:
                missing.append(".".join(keys))
        if missing:
            raise StartupError(f"Missing required config keys: {missing}")

        allowed = self._config.get("compliance", "allowed_order_types", default=[])
        if not isinstance(allowed, list) or not allowed:
            raise StartupError("Invalid config: compliance.allowed_order_types must be a non-empty list")

        # Safety: ensure forbidden order types are not enabled by config
        forbidden = {"MARKET", "IOC"}
        if any(str(x).upper() in forbidden for x in allowed):
            raise StartupError("Config violation: forbidden order types present in allowed_order_types")

        self._log.info(
            "Config validated",
            index=self._config.get("market", "index"),
            lot_size=self._config.get("market", "lot_size"),
            expiry_day=self._config.get("market", "expiry_day", default=None),
        )

    def _validate_mode(self) -> None:
        mode = self._args.mode.upper().strip()
        if mode not in {"OBSERVE", "PAPER", "LIVE"}:
            raise StartupError(f"Invalid mode: {self._args.mode}. Use observe|paper|live.")

        if mode == "LIVE" and not self._args.confirm_live:
            raise StartupError("LIVE mode requires --confirm-live (explicit acknowledgement).")

        if mode == "LIVE":
            # Fail-fast if .env not present for credentials (actual validation is inside auth/broker layers)
            env_path = _PROJECT_ROOT / ".env"
            if not env_path.exists():
                raise StartupError("LIVE mode requires .env with broker credentials; .env not found.")

    # -------------------------------------------------------------------------
    # Component construction helpers
    # -------------------------------------------------------------------------
    def _construct(self, cls: Any, candidates: Dict[str, Any], *, name: str) -> Any:
        """
        Construct a class instance using dependency injection-like behavior:
        - uses signature introspection to pass only supported kwargs
        - fails fast if required args are missing

        This avoids tight coupling to evolving constructors while remaining strict.
        """
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            # fallback: try zero-arg
            self._log.warning("No signature available; constructing without args", component=name)
            return cls()

        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param_name in candidates:
                kwargs[param_name] = candidates[param_name]
            else:
                if param.default is inspect._empty:
                    raise StartupError(f"Cannot construct {name}: missing required arg '{param_name}'")
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise StartupError(f"Failed constructing {name} with kwargs={list(kwargs.keys())}: {e}") from e

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def initialize_components(self) -> None:
        """
        Initialize components in strict order.
        """
        self._log.info("Initializing components...")

        # Imports inside method to reduce side-effects and keep startup controlled
        from src.core.market_state import MarketState
        from src.core.feed_health import FeedHealthMonitor
        from src.core.tick_validator import TickValidator
        from src.core.candle_builder import CandleBuilder
        from src.core.option_chain_poller import OptionChainPoller
        from src.core.data_engine import DataEngine, _InstrumentMapperAdapter
        from src.core.instrument_mapper import InstrumentMapper
        from src.core.auth_manager import AuthManager

        from src.features.feature_engine import FeatureEngine

        from src.context.narrative_engine import NarrativeEngine
        from src.context.regime_engine import RegimeEngine
        from src.context.time_context import TimeContextEngine
        from src.dna.market_dna import MarketDNAEngine

        from src.captain.captain_engine import CaptainEngine

        from src.strategies.strategy_engine import StrategyEngine
        from src.filters.trap_detector import TrapDetector
        from src.filters.opportunity_scorer import OpportunityScorer

        from src.ml.lightgbm_filter import LightGBMFilter
        from src.ml.anomaly_detector import AnomalyDetector
        from src.ml.garch_forecaster import GARCHForecaster

        from src.risk.risk_engine import RiskEngine
        from src.execution.order_validator import OrderValidator
        from src.execution.paper_broker import PaperBroker
        from src.core.dashboard_control_plane import DashboardControlPlanePublisher
        from src.core.dashboard_ipc import DashboardIpcBridge

        # 3) MarketState
        self.state = MarketState()
        # Attempt to set mode if supported
        try:
            if hasattr(self.state, "update"):
                self.state.update(mode=self._args.mode.upper())
        except Exception:
            pass

        # 4) FeedHealth
        self.feed_health = FeedHealthMonitor()

        # 5) AuthManager (needed by OptionChainPoller and DataEngine)
        self._auth_manager = AuthManager()

        # 6) InstrumentMapper (needed by TickValidator, OptionChainPoller and DataEngine)
        nifty_token = str(self._config.get("market", "nifty_spot_token", default="99926000"))
        vix_token = str(self._config.get("market", "india_vix_token", default="26017"))
        base_mapper = InstrumentMapper()
        self._mapper_adapter = _InstrumentMapperAdapter(base_mapper, nifty_token, vix_token)

        # 7) TickValidator (requires instrument_mapper)
        self.tick_validator = TickValidator(self._mapper_adapter)

        # 8) CandleBuilder
        self.candle_builder = CandleBuilder()

        # 9) OptionChainPoller (requires auth_manager and instrument_mapper)
        self.option_chain_poller = OptionChainPoller(self._auth_manager, self._mapper_adapter)

        # 10) DataEngine (dependency-injected where supported)
        candidates = {
            "market_state": self.state,
            "feed_health_monitor": self.feed_health,
            "tick_validator": self.tick_validator,
            "candle_builder": self.candle_builder,
            "option_chain_poller": self.option_chain_poller,
            "instrument_mapper": self._mapper_adapter,
            "auth_manager": self._auth_manager,
        }
        self.data_engine = self._construct(DataEngine, candidates, name="DataEngine")

        # 11) FeatureEngine
        self.feature_engine = FeatureEngine()

        # 12) Context engines
        self.narrative_engine = NarrativeEngine()
        self.regime_engine = RegimeEngine()
        self.time_context_engine = TimeContextEngine()
        try:
            self.market_dna_engine = MarketDNAEngine()
        except Exception as exc:
            self._log.error("MarketDNAEngine initialization failed", error=str(exc))
            self.market_dna_engine = None

        # 13) CaptainEngine
        self.captain_engine = self._construct(CaptainEngine, {"market_state": self.state}, name="CaptainEngine")

        # 14) Brain layer: instantiate so it is import-verified (captain uses internally)
        self._initialize_brains_if_required()

        # 15) StrategyEngine
        self.strategy_engine = StrategyEngine()

        # 16) ML components (reject-only / optional)
        self.ml_filter = LightGBMFilter() if LightGBMFilter is not None else None
        self.anomaly_detector = AnomalyDetector() if AnomalyDetector is not None else None
        self.garch_forecaster = GARCHForecaster() if GARCHForecaster is not None else None

        # 17) RiskEngine (mandatory)
        self.risk_engine = RiskEngine()

        # 18) Execution layer
        self.order_validator = OrderValidator()

        mode = self._args.mode.upper()
        if mode in {"OBSERVE", "PAPER"}:
            self.broker = PaperBroker()
        elif mode == "LIVE":
            from src.execution.angel_one_broker import AngelOneBroker  # delayed import
            self.broker = AngelOneBroker()

        # 19) TrapDetector
        self.trap_detector = TrapDetector()

        # 20) OpportunityScorer
        self.opportunity_scorer = OpportunityScorer()

        # 21) Dashboard control plane publisher
        # Why this initialization exists:
        # The dashboard already checks for a backend heartbeat shared memory
        # segment and a kill-switch block. Creating the publisher here keeps the
        # control-plane truth owned by the backend/orchestrator instead of
        # leaving the dashboard to wait forever on resources that never exist.
        self._control_plane = DashboardControlPlanePublisher(
            heartbeat_name=str(self._config.get("dashboard", "backend_heartbeat_name", default="ja_backend_heartbeat")),
            kill_switch_name=str(self._config.get("dashboard", "kill_switch_name", default="junior_aladdin_kill_switch")),
            mode=self._args.mode.upper(),
            compliance_state=1,
        )

        # 22) Dashboard IPC bridge
        # Why this initialization exists:
        # Part 2A requires the backend to publish real HOT/WARM/COLD frames and
        # consume dashboard commands over a real cross-process channel.  This
        # bridge owns only that transport layer and does not change trading
        # engines or dashboard UI logic.
        self._dashboard_ipc = DashboardIpcBridge(
            host=str(self._config.get("dashboard", "ipc_host", default="127.0.0.1")),
            snapshot_port=int(self._config.get("dashboard", "snapshot_port", default=18765)),
            command_port=int(self._config.get("dashboard", "command_port", default=18766)),
            hot_interval_ms=int(self._config.get("dashboard", "hot_interval_ms", default=200)),
            warm_interval_ms=int(self._config.get("dashboard", "warm_interval_ms", default=1000)),
            cold_interval_ms=int(self._config.get("dashboard", "cold_interval_ms", default=5000)),
        )

        self._log.info("Components initialized successfully")

    def _initialize_brains_if_required(self) -> None:
        from src.brains.structural_brain import StructuralBrain
        from src.brains.tactical_brain import TacticalBrain
        from src.brains.institutional_brain import InstitutionalBrain
        from src.brains.adaptive_brain import AdaptiveBrain

        structural = StructuralBrain()
        tactical = TacticalBrain()
        institutional = InstitutionalBrain()
        adaptive = AdaptiveBrain()

        cap = self.captain_engine

        try:
            if hasattr(cap, "set_brains") and callable(getattr(cap, "set_brains")):
                cap.set_brains(
                    {
                        "STRUCTURAL": structural,
                        "TACTICAL": tactical,
                        "INSTITUTIONAL": institutional,
                        "ADAPTIVE": adaptive,
                    }
                )
                self._log.info("Brains injected into CaptainEngine via set_brains()")
            elif hasattr(cap, "register_brain") and callable(getattr(cap, "register_brain")):
                cap.register_brain("STRUCTURAL", structural)
                cap.register_brain("TACTICAL", tactical)
                cap.register_brain("INSTITUTIONAL", institutional)
                cap.register_brain("ADAPTIVE", adaptive)
                self._log.info("Brains injected into CaptainEngine via register_brain()")
            else:
                self._log.info("CaptainEngine brain injection not required (no injection API detected)")
        except Exception as e:
            raise StartupError(f"CaptainEngine brain wiring failed: {e}") from e

    # -------------------------------------------------------------------------
    # Lifecycle: start/stop
    # -------------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        if self.data_engine is None:
            raise StartupError("start() called before initialize_components()")

        if self._control_plane is not None:
            self._log.info("Starting dashboard control plane...")
            control_ok = self._call_start(self._control_plane, "DashboardControlPlane")
            if control_ok is False:
                raise StartupError("Dashboard control plane failed to start")
            try:
                self._control_plane.publish_state(reason="backend_bootstrap_started")
            except Exception as e:
                self._log.warning("Dashboard control plane initial publish failed", error=str(e))

        if self._dashboard_ipc is not None:
            self._log.info("Starting dashboard IPC bridge...")
            ipc_ok = self._call_start(self._dashboard_ipc, "DashboardIpcBridge")
            if ipc_ok is False:
                raise StartupError("Dashboard IPC bridge failed to start")

        self._log.info("Starting DataEngine...")
        started_ok = self._call_start(self.data_engine, "DataEngine")
        if started_ok is False:
            raise StartupError("DataEngine failed to start (returned False)")

        self._started = True
        try:
            if self._control_plane is not None:
                self._control_plane.publish_state(reason="backend_started")
                self._control_plane.publish_heartbeat()
        except Exception as e:
            self._log.warning("Dashboard control plane post-start publish failed", error=str(e))
        self._log.info("System started", mode=self._args.mode.upper())

    def _call_start(self, component: Any, name: str) -> Optional[bool]:
        try:
            if hasattr(component, "start") and callable(getattr(component, "start")):
                rv = component.start()
                self._log.info(f"{name} start() called", returned=rv)
                return rv if isinstance(rv, bool) else True
            self._log.warning(f"{name} has no start(); assuming passive component")
            return True
        except Exception as e:
            self._log.error(f"{name} start failed", error=str(e))
            raise

    # -------------------------------------------------------------------------
    # Pipeline publication helpers (Issue #1)
    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_pipeline_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None
        return None

    @classmethod
    def _serialize_for_market_state(cls, value: Any, *, _depth: int = 0, _max_depth: int = 6) -> Any:
        """Convert runtime pipeline objects into MarketState-safe structures.

        Why this exists:
        The main pipeline emits a mix of dataclasses, enums, dicts, and lists.
        MarketState expects plain Python containers that can be deep-copied and
        later surfaced to the dashboard without leaking live references.
        """
        if _depth > _max_depth:
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(k): cls._serialize_for_market_state(v, _depth=_depth + 1, _max_depth=_max_depth)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._serialize_for_market_state(v, _depth=_depth + 1, _max_depth=_max_depth) for v in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return cls._serialize_for_market_state(to_dict(), _depth=_depth + 1, _max_depth=_max_depth)
            except Exception:
                return str(value)
        if hasattr(value, "__dict__"):
            try:
                return cls._serialize_for_market_state(dict(value.__dict__), _depth=_depth + 1, _max_depth=_max_depth)
            except Exception:
                return str(value)
        return str(value)

    @classmethod
    def _extract_captain_brain_state(cls, captain_result: Any) -> Tuple[list, dict]:
        active_brains: list = []
        brain_confidence: dict = {}

        if captain_result is None:
            return active_brains, brain_confidence

        if isinstance(captain_result, list):
            active_brains = [str(x) for x in captain_result if x is not None]
            return active_brains, brain_confidence

        allowed = getattr(captain_result, "allowed_brains", None)
        if isinstance(allowed, list):
            active_brains = [str(x) for x in allowed if x is not None]

        consensus_strength = getattr(captain_result, "consensus_strength", None)
        context_score = getattr(captain_result, "context_score", None)
        risk_level = getattr(captain_result, "risk_level", None)
        decision = getattr(captain_result, "decision", None)
        if decision is not None:
            try:
                brain_confidence["captain_confidence_score"] = float(getattr(decision, "confidence_score", 0.0) or 0.0)
            except Exception:
                pass
        try:
            if consensus_strength is not None:
                brain_confidence["consensus_strength"] = float(consensus_strength)
            if context_score is not None:
                brain_confidence["context_score"] = float(context_score)
            if risk_level is not None:
                brain_confidence["risk_level"] = float(risk_level)
        except Exception:
            pass
        return active_brains, brain_confidence

    @classmethod
    def _build_market_state_payload(
        cls,
        *,
        candle_ts: Any,
        spot_price: Any,
        candles_by_tf: Dict[str, Any],
        feature_bundle: Dict[str, Any],
        context: Dict[str, Any],
        captain_result: Any,
        raw_opportunities: list,
        trapped_opportunities: list,
        scored_opportunities: list,
        ml_filtered: list,
        behavioral_filtered: list,
        approved_opportunities: list,
    ) -> Dict[str, Any]:
        per_tf = feature_bundle.get("per_tf", {}) or {}
        mtf_payload = feature_bundle.get("mtf", {}) or {}
        features_state = {
            "1min": cls._serialize_for_market_state(per_tf.get("1min", {})),
            "3min": cls._serialize_for_market_state(per_tf.get("3min", {})),
            "5min": cls._serialize_for_market_state(per_tf.get("5min", {})),
            "15min": cls._serialize_for_market_state(per_tf.get("15min", {})),
        }
        if isinstance(mtf_payload, dict) and mtf_payload:
            serialized_mtf = cls._serialize_for_market_state(mtf_payload)
            features_state["mtf"] = serialized_mtf
            features_state["mtf_alignment"] = serialized_mtf

        smart_money_state = {
            "5min": cls._serialize_for_market_state(feature_bundle.get("smart_money_5m", {})),
            "15min": cls._serialize_for_market_state(feature_bundle.get("smart_money_15m", {})),
        }

        def _normalize_tf(tf: str) -> str:
            mapping = {"1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m"}
            return mapping.get(str(tf), str(tf))

        mtf_candles: Dict[str, Any] = {}
        if isinstance(candles_by_tf, dict):
            for tf, rows in candles_by_tf.items():
                if not isinstance(rows, list):
                    continue
                normalized_rows = []
                for candle in rows[-5000:]:
                    if not isinstance(candle, dict):
                        continue
                    normalized_rows.append({
                        "timestamp": cls._serialize_for_market_state(candle.get("timestamp")),
                        "open": candle.get("open"),
                        "high": candle.get("high"),
                        "low": candle.get("low"),
                        "close": candle.get("close"),
                        "volume": candle.get("volume"),
                    })
                mtf_candles[_normalize_tf(str(tf))] = normalized_rows

        def _compute_vwap_bands(rows: list) -> Dict[str, list]:
            if not rows:
                return {}
            vwap = []
            upper1 = []
            lower1 = []
            upper2 = []
            lower2 = []
            cum_v = 0.0
            cum_pv = 0.0
            cum_p2v = 0.0
            for candle in rows:
                try:
                    h = float(candle.get("high"))
                    l = float(candle.get("low"))
                    c = float(candle.get("close"))
                    vol = float(candle.get("volume") or 0.0)
                    tp = (h + l + c) / 3.0
                    cum_v += max(0.0, vol)
                    cum_pv += tp * max(0.0, vol)
                    cum_p2v += (tp * tp) * max(0.0, vol)
                    if cum_v <= 0.0:
                        vw = None
                        sd = None
                    else:
                        vw = cum_pv / cum_v
                        mean_sq = cum_p2v / cum_v
                        var = max(0.0, mean_sq - (vw * vw))
                        sd = var ** 0.5
                except Exception:
                    vw = None
                    sd = None
                vwap.append(vw)
                upper1.append((vw + sd) if vw is not None and sd is not None else None)
                lower1.append((vw - sd) if vw is not None and sd is not None else None)
                upper2.append((vw + 2.0 * sd) if vw is not None and sd is not None else None)
                lower2.append((vw - 2.0 * sd) if vw is not None and sd is not None else None)
            return {
                "vwap": vwap,
                "upper1": upper1,
                "lower1": lower1,
                "upper2": upper2,
                "lower2": lower2,
            }

        vwap_bands = {tf: _compute_vwap_bands(rows) for tf, rows in mtf_candles.items() if rows}

        key_levels_state = cls._serialize_for_market_state(feature_bundle.get("key_levels", {}))
        or_high = None
        or_low = None
        ib_high = None
        ib_low = None
        ib_width = None
        if isinstance(key_levels_state, dict):
            or_high = key_levels_state.get("or_high")
            or_low = key_levels_state.get("or_low")
            ib_high = key_levels_state.get("ib_high")
            ib_low = key_levels_state.get("ib_low")
            ib_width = key_levels_state.get("ib_width")

        active_brains, brain_confidence = cls._extract_captain_brain_state(captain_result)

        timestamp_value = cls._coerce_pipeline_datetime(candle_ts)
        payload = {
            "timestamp": timestamp_value,
            "spot": float(spot_price or 0.0),
            "features": features_state,
            "options_features": cls._serialize_for_market_state(feature_bundle.get("options", {})),
            "microstructure": cls._serialize_for_market_state(feature_bundle.get("microstructure", {})),
            "key_levels": key_levels_state,
            "smart_money": smart_money_state,
            "mtf_candles": mtf_candles,
            "candles_by_tf": mtf_candles,
            "vwap_bands": vwap_bands,
            "or_levels": {"high": or_high, "low": or_low},
            "ib_levels": {"high": ib_high, "low": ib_low},
            "or_high": or_high,
            "or_low": or_low,
            "ib_high": ib_high,
            "ib_low": ib_low,
            "ib_width": ib_width,
            "active_timeframe": "5m",
            "timeframe": "5m",
            # Issue #3: Feed health diagnostics published to MarketState
            "feed_health": str(context.get("feed_health") or "UNKNOWN"),
            "data_quality_score": float(context.get("data_quality_score") or 0.0),
            "ticks_per_second": float(context.get("ticks_per_second") or 0.0),
            "feed_lag_ms": float(context.get("feed_lag_ms") or 0.0),
            "using_fallback": bool(context.get("using_fallback", False)),
            "kill_switch_state": str(context.get("kill_switch_state") or "UNKNOWN"),
            "narrative_score": float(context.get("narrative_score") or 0.0),
            "narrative_label": str(context.get("narrative_label") or "NEUTRAL"),
            "narrative_fit_factors": cls._serialize_for_market_state(context.get("narrative_fit_factors", {})),
            "regime": str(context.get("regime") or "UNKNOWN"),
            "regime_confidence": float(context.get("regime_confidence") or 0.0),
            "regime_transition_prob": float(context.get("regime_transition_prob") or 0.0),
            "session_phase": str(context.get("session_phase") or "UNKNOWN"),
            "session_size_multiplier": float(context.get("session_size_multiplier") or 0.0),
            "day_type": str(context.get("day_type") or "UNKNOWN"),
            "day_personality": cls._serialize_for_market_state(
                context.get("day_personality", {"day_type": str(context.get("day_type") or "UNKNOWN")})
            ),
            "historical_match_score": float(context.get("historical_match_score") or 0.0),
            "session_memory": cls._serialize_for_market_state(context.get("session_memory", {})),
            "active_brains": active_brains,
            "brain_confidence": brain_confidence,
            "raw_opportunities": cls._serialize_for_market_state(raw_opportunities),
            "trapped_opportunities": cls._serialize_for_market_state(trapped_opportunities),
            "scored_opportunities": cls._serialize_for_market_state(scored_opportunities),
            "ml_filtered": cls._serialize_for_market_state(ml_filtered),
            "behavioral_filtered": cls._serialize_for_market_state(behavioral_filtered),
            "approved_opportunities": cls._serialize_for_market_state(approved_opportunities),
        }
        return payload

    def _publish_market_state_payload(self, payload: Dict[str, Any]) -> None:
        if self.state is None or not hasattr(self.state, "update"):
            return
        try:
            self.state.update(**payload)
        except Exception as e:
            self._log.error("MarketState pipeline publication failed", error=str(e), payload_keys=list(payload.keys()))

    @staticmethod
    def _status_to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        data = getattr(value, "__dict__", None)
        if isinstance(data, dict):
            return dict(data)
        return {}

    @staticmethod
    def _format_health_timestamp(value: Any) -> str:
        if value is None:
            return "never"
        if isinstance(value, str):
            text = value.strip()
            return text if text else "never"
        if isinstance(value, datetime):
            try:
                return value.isoformat() if value.tzinfo is not None else f"{value.isoformat()}Z"
            except Exception:
                return str(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                numeric = float(value)
                if numeric <= 0.0:
                    return "never"
                if numeric > 1_000_000_000_000.0:
                    numeric /= 1_000_000_000.0
                return f"{datetime.utcfromtimestamp(numeric).isoformat()}Z"
            except Exception:
                return str(value)
        return str(value)

    @classmethod
    def _engine_health_row(cls, *, alive: Any, status: str, last_heartbeat: Any, last_error: Any = "") -> Dict[str, Any]:
        return {
            "alive": bool(alive),
            "status": str(status or "UNKNOWN").upper(),
            "last_heartbeat": cls._format_health_timestamp(last_heartbeat),
            "last_error": str(last_error or ""),
        }

    def stop(self, reason: str) -> None:
        with self._shutdown_lock:
            if self._stop_event.is_set():
                return
            self._stop_event.set()

        self._log.warning("Shutdown initiated", reason=reason)

        emergency_reason = str(reason or "")
        emergency_requested = emergency_reason.lower().startswith("kill_switch:") or "emergency" in emergency_reason.lower()
        try:
            if self._control_plane is not None:
                self._control_plane.publish_state(
                    reason=reason,
                    emergency_stop_flag=1 if emergency_requested else 0,
                    live_approval_flag=0 if self._args.mode.upper() == "LIVE" else None,
                )
        except Exception as e:
            self._log.warning("Dashboard control plane shutdown publish failed", error=str(e))

        self._safe_stop_component(self.broker, "Broker")
        self._safe_stop_component(self.data_engine, "DataEngine")
        self._safe_stop_component(self._dashboard_ipc, "DashboardIpcBridge")
        if self._control_plane is not None:
            try:
                self._control_plane.stop(reason=reason)
                self._log.info("DashboardControlPlane stopped via stop(reason=...)")
            except Exception as e:
                self._log.error("DashboardControlPlane stop error", error=str(e))

        self._started = False
        self._log.info("Shutdown completed", reason=reason)

    def _safe_stop_component(self, component: Any, name: str) -> None:
        if component is None:
            return
        try:
            if hasattr(component, "stop") and callable(getattr(component, "stop")):
                component.stop()
                self._log.info(f"{name} stopped via stop()")
            elif hasattr(component, "disconnect") and callable(getattr(component, "disconnect")):
                component.disconnect()
                self._log.info(f"{name} stopped via disconnect()")
            elif hasattr(component, "close") and callable(getattr(component, "close")):
                component.close()
                self._log.info(f"{name} closed via close()")
            else:
                self._log.debug(f"{name} has no stop/close API; skipping")
        except Exception as e:
            self._log.error(f"{name} stop error", error=str(e))

    # -------------------------------------------------------------------------
    # Runtime loop
    # -------------------------------------------------------------------------
    def run_forever(self) -> int:
        if not self._started:
            raise StartupError("run_forever() called before start()")

        self._log.info("Runtime loop started")

        poll_interval = max(0.05, float(self._args.candle_poll_interval_sec))
        health_interval = max(1.0, float(self._args.health_interval_sec))

        while not self._stop_event.is_set():
            try:
                emergency_reason = self._poll_dashboard_control_plane()
                if emergency_reason is not None:
                    self._log.critical("Dashboard kill-switch emergency observed by backend", reason=emergency_reason)
                    self.stop(reason=f"kill_switch:{emergency_reason}")
                    return 1

                now_m = time.monotonic()
                self._service_dashboard_ipc(now_m)
                if (now_m - self._last_health_log_mono) >= health_interval:
                    self._last_health_log_mono = now_m
                    self._emit_health()

                last_closed = self._get_last_closed_1m()
                if last_closed is not None:
                    ts = last_closed.get("timestamp")
                    if ts is not None and ts != self._last_1m_closed_ts:
                        self._last_1m_closed_ts = ts
                        self._process_candle_close(last_closed)

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                self.stop(reason="KeyboardInterrupt")
                return 0
            except Exception as e:
                self._log.error("Fatal error in runtime loop", error=str(e))
                self.stop(reason=f"fatal_runtime_error: {e}")
                return 1

        return 0

    # -------------------------------------------------------------------------
    # Pipeline cycle
    # -------------------------------------------------------------------------
    def _get_last_closed_1m(self) -> Optional[Dict[str, Any]]:
        cb = None

        # CHANGE 1: DataEngine uses private attribute `_candle_builder`
        if self.data_engine is not None and hasattr(self.data_engine, "_candle_builder"):
            cb = getattr(self.data_engine, "_candle_builder")

        if cb is None:
            cb = self.candle_builder
        if cb is None:
            return None

        try:
            # CHANGE 2: CandleBuilder.get_last_closed requires token
            if hasattr(cb, "get_last_closed") and callable(getattr(cb, "get_last_closed")):
                nifty_token = str(self._config.get("market", "nifty_spot_token", default="99926000"))
                return cb.get_last_closed("1min", token=nifty_token)

            candles = getattr(cb, "candles", None)
            if isinstance(candles, dict) and candles.get("1min"):
                return list(candles["1min"])[-1]
        except Exception as e:
            self._log.warning("Failed reading last closed 1-min candle", error=str(e))

        return None

    def _process_candle_close(self, candle_1m: Dict[str, Any]) -> None:
        candle_ts = candle_1m.get("timestamp")
        spot_price = candle_1m.get("close")

        self._log.info("Pipeline cycle started", candle_ts=str(candle_ts), spot=spot_price)

        # CHANGE 3: DataEngine uses private attribute `_candle_builder`
        cb = getattr(self.data_engine, "_candle_builder", None) if self.data_engine is not None else None
        cb = cb or self.candle_builder

        candles_by_tf: Dict[str, Any] = {}
        try:
            candles_dict = getattr(cb, "candles", {})
            for tf in ("1min", "3min", "5min", "15min"):
                candles_by_tf[tf] = list(candles_dict.get(tf, [])) if isinstance(candles_dict, dict) else []
        except Exception as e:
            self._log.error("Candle extraction failed", error=str(e), candle_ts=str(candle_ts))
            return

        option_chain = {}
        market_depth = {}
        previous_close = 0.0
        try:
            if self.state is not None and hasattr(self.state, "snapshot"):
                snap = self.state.snapshot()
                option_chain = snap.get("option_chain") or {}
                market_depth = snap.get("market_depth") or {}
                if spot_price is None:
                    spot_price = snap.get("spot", 0.0)
                previous_close = snap.get("previous_close", 0.0) or 0.0
        except Exception:
            option_chain = {}
            market_depth = {}

        try:
            feature_bundle = self.feature_engine.compute_all(
                candles_by_tf=candles_by_tf,
                option_chain=option_chain,
                market_depth=market_depth,
                spot_price=float(spot_price or 0.0),
                skip_freshness_check=False,  # live runtime must keep False
            )
        except Exception as e:
            self._log.error("FeatureEngine compute_all failed", error=str(e), candle_ts=str(candle_ts))
            return

        context: Dict[str, Any] = self._build_context(feature_bundle=feature_bundle, candle_ts=candle_ts)
        context["spot"] = float(spot_price or 0.0)
        context["previous_close"] = float(previous_close or 0.0)
        context["opening_price"] = float(candle_1m.get("open") or 0.0)

        try:
            self._update_context_engines(context)
        except Exception as e:
            self._log.error("Context update failed", error=str(e), candle_ts=str(candle_ts))
            return

        try:
            brain_filter = self._captain_step(context)
        except Exception as e:
            self._log.error("CaptainEngine step failed", error=str(e), candle_ts=str(candle_ts))
            return

        try:
            opportunities = self._scan_strategies(context, brain_filter=brain_filter)
        except Exception as e:
            self._log.error("StrategyEngine scan failed", error=str(e), candle_ts=str(candle_ts))
            return

        raw_stage: list = []
        trapped_stage: list = []
        scored_stage: list = []
        ml_stage: list = []
        behavioral_stage: list = []
        approved_stage: list = []

        for opp in opportunities or []:
            try:
                opp_d = opp.to_dict() if hasattr(opp, "to_dict") else dict(getattr(opp, "__dict__", {}) or {})
            except Exception:
                opp_d = {}
            raw_stage.append(opp_d)

        if not opportunities:
            self._publish_market_state_payload(
                self._build_market_state_payload(
                    candle_ts=candle_ts,
                    spot_price=spot_price,
                    candles_by_tf=candles_by_tf,
                    feature_bundle=feature_bundle,
                    context=context,
                    captain_result=brain_filter,
                    raw_opportunities=raw_stage,
                    trapped_opportunities=trapped_stage,
                    scored_opportunities=scored_stage,
                    ml_filtered=ml_stage,
                    behavioral_filtered=behavioral_stage,
                    approved_opportunities=approved_stage,
                )
            )
            self._log.info("No opportunities", candle_ts=str(candle_ts))
            self._log.info("Pipeline cycle finished", candle_ts=str(candle_ts), opportunities=0)
            return

        self._log.info("Opportunities generated", count=len(opportunities), candle_ts=str(candle_ts))

        passed = 0
        rejected = 0
        trap_rejected = 0
        score_rejected = 0
        ml_rejected = 0
        risk_rejected = 0

        for idx, opp in enumerate(opportunities):
            try:
                opp_d = opp.to_dict() if hasattr(opp, "to_dict") else dict(getattr(opp, "__dict__", {}) or {})
            except Exception:
                opp_d = {}

            trap_assessment = self.trap_detector.evaluate(opp_d, context)
            trap_payload = self._serialize_for_market_state(trap_assessment)
            trap_reject = bool(
                getattr(trap_assessment, "reject", False)
                or (isinstance(trap_assessment, dict) and trap_assessment.get("reject"))
            )
            if trap_reject:
                trapped_stage.append({
                    "opportunity": opp_d,
                    "trap": trap_payload,
                    "stage": "trap_reject",
                })
                trap_rejected += 1
                rejected += 1
                self._log.debug("Opportunity rejected by trap", idx=idx)
                continue

            scored = self.opportunity_scorer.score_opportunity(opp_d, context)
            scored_payload = self._serialize_for_market_state(scored.to_dict() if hasattr(scored, "to_dict") else scored)
            hard_reject = bool(getattr(scored, "hard_reject", False) or (isinstance(scored, dict) and scored.get("hard_reject")))
            if hard_reject:
                score_rejected += 1
                rejected += 1
                self._log.debug("Opportunity rejected by scoring", idx=idx)
                continue
            scored_stage.append(scored_payload)

            if self.anomaly_detector is not None:
                if self._ml_anomaly_blocks(scored, context):
                    ml_rejected += 1
                    rejected += 1
                    self._log.debug("Opportunity rejected by anomaly detector", idx=idx)
                    continue

            if self.ml_filter is not None:
                if self._ml_quality_blocks(scored, context):
                    ml_rejected += 1
                    rejected += 1
                    self._log.debug("Opportunity rejected by ML filter", idx=idx)
                    continue

            ml_stage.append(scored_payload)
            # Behavioral sentinel is not yet integrated in this runtime path.
            # Preserve post-ML survivors here so the dashboard pipeline surface
            # reflects the last successful gate before risk/execution.
            behavioral_stage.append(scored_payload)

            approved, exec_plan = self._risk_approve(scored, context)
            if not approved:
                risk_rejected += 1
                rejected += 1
                self._log.debug("Opportunity rejected by risk engine", idx=idx)
                continue

            approved_stage.append({
                "scored": scored_payload,
                "execution_plan": self._serialize_for_market_state(exec_plan),
                "stage": "risk_approved",
            })

            if self._args.mode.upper() == "OBSERVE":
                self._log.info("OBSERVE mode: trade approved but not executed", plan=exec_plan)
                passed += 1
                continue

            if self._args.mode.upper() in {"PAPER", "LIVE"}:
                ok = self._execute(exec_plan, context)
                if ok:
                    passed += 1
                else:
                    rejected += 1

        self._publish_market_state_payload(
            self._build_market_state_payload(
                candle_ts=candle_ts,
                spot_price=spot_price,
                candles_by_tf=candles_by_tf,
                feature_bundle=feature_bundle,
                context=context,
                captain_result=brain_filter,
                raw_opportunities=raw_stage,
                trapped_opportunities=trapped_stage,
                scored_opportunities=scored_stage,
                ml_filtered=ml_stage,
                behavioral_filtered=behavioral_stage,
                approved_opportunities=approved_stage,
            )
        )

        self._log.info(
            "Pipeline cycle finished",
            candle_ts=str(candle_ts),
            opportunities=len(opportunities),
            passed=passed,
            rejected=rejected,
            trap_rejected=trap_rejected,
            score_rejected=score_rejected,
            ml_rejected=ml_rejected,
            risk_rejected=risk_rejected,
        )

    # -------------------------------------------------------------------------
    # Context helpers (UNCHANGED)
    # -------------------------------------------------------------------------
    def _build_context(self, *, feature_bundle: Dict[str, Any], candle_ts: Any) -> Dict[str, Any]:
        per_tf = feature_bundle.get("per_tf", {}) or {}
        mtf = feature_bundle.get("mtf", {}) or {}
        weighted_mtf = 0.0
        if isinstance(mtf, dict):
            weighted_mtf = float(mtf.get("weighted_mtf", 0.0) or 0.0)

        context = {
            "candle_ts": candle_ts,
            "regime": None,
            "session_phase": None,
            "narrative_label": None,
            "narrative_score": None,
            "data_quality_score": 0.0,
            "ticks_per_second": 0.0,
            "feed_lag_ms": 0.0,
            "using_fallback": False,
            "weighted_mtf": weighted_mtf,
            "features_1m": per_tf.get("1min", {}) or {},
            "features_5m": per_tf.get("5min", {}) or {},
            "features_15m": per_tf.get("15min", {}) or {},
            "key_levels": feature_bundle.get("key_levels", {}) or {},
            "volume_profile": feature_bundle.get("volume_profile", {}) or {},
            "options": feature_bundle.get("options", {}) or {},
            "smart_money_5m": feature_bundle.get("smart_money_5m", {}) or {},
            "smart_money_15m": feature_bundle.get("smart_money_15m", {}) or {},
            "microstructure": feature_bundle.get("microstructure", {}) or {},
            "fundamental": feature_bundle.get("fundamental", {}) or {},
            "meta": feature_bundle.get("meta", {}) or {},
        }

        try:
            if self.feed_health is not None and hasattr(self.feed_health, "check"):
                health, dq = self.feed_health.check()
                context["feed_health"] = health
                context["data_quality_score"] = float(dq or 0.0)
                # Issue #3: Populate additional feed health diagnostics
                try:
                    context["ticks_per_second"] = float(self.feed_health.get_tps())
                except Exception:
                    pass
                try:
                    if hasattr(self.feed_health, "should_enter_safe_mode"):
                        context["using_fallback"] = bool(self.feed_health.should_enter_safe_mode)
                except Exception:
                    pass
                try:
                    from datetime import datetime, timezone, timedelta
                    _IST = timezone(timedelta(hours=5, minutes=30))
                    if hasattr(self.feed_health, "last_tick_time") and self.feed_health.last_tick_time is not None:
                        now = datetime.now(_IST)
                        lag_ms = max(0.0, (now - self.feed_health.last_tick_time).total_seconds() * 1000.0)
                        context["feed_lag_ms"] = lag_ms
                except Exception:
                    pass
        except Exception:
            pass

        # Issue #3: Populate kill_switch_state from control plane
        try:
            if self._control_plane is not None:
                ks_state = self._control_plane.read_kill_switch_state(default_empty=True)
                if isinstance(ks_state, dict):
                    emergency = int(ks_state.get("emergency_stop_flag", 0))
                    if emergency == 1:
                        context["kill_switch_state"] = "TRIGGERED"
                    else:
                        context["kill_switch_state"] = "ARMED"
                else:
                    context["kill_switch_state"] = context.get("kill_switch_state", "UNKNOWN")
            else:
                context["kill_switch_state"] = context.get("kill_switch_state", "UNKNOWN")
        except Exception:
            context["kill_switch_state"] = context.get("kill_switch_state", "UNKNOWN")

        return context

    def _update_context_engines(self, context: Dict[str, Any]) -> None:
        fundamental = context.get("fundamental", {}) or {}
        if not isinstance(fundamental, dict):
            fundamental = {}

        if self.narrative_engine is not None:
            try:
                score = self.narrative_engine.compute(fundamental)  # type: ignore[misc]
                context["narrative_score"] = float(score or 0.0)
                context["narrative_label"] = getattr(self.narrative_engine, "narrative_label", "NEUTRAL")
                if hasattr(self.narrative_engine, "get_fit_factors"):
                    context["narrative_fit_factors"] = self.narrative_engine.get_fit_factors()  # type: ignore[misc]
            except Exception as exc:
                self._log.error("NarrativeEngine compute failed", error=str(exc))

        if self.regime_engine is not None:
            try:
                rout = self.regime_engine.classify(  # type: ignore[misc]
                    context.get("features_1m", {}) or {},
                    context.get("features_5m", {}) or {},
                    context.get("features_15m", {}) or {},
                    event_data=fundamental,
                    vix_data=fundamental,
                )
            except Exception as exc:
                self._log.error("RegimeEngine classify failed", error=str(exc))
                rout = None
            if isinstance(rout, dict):
                context["regime"] = rout.get("regime")
                context["regime_confidence"] = rout.get("confidence")
                context["regime_transition_prob"] = rout.get("transition_prob", rout.get("transition_probability"))

        if self.time_context_engine is not None and hasattr(self.time_context_engine, "get_context"):
            try:
                tout = self.time_context_engine.get_context(  # type: ignore[misc]
                    current_time=self._coerce_pipeline_datetime(context.get("candle_ts")),
                    key_levels=context.get("key_levels", {}) or {},
                    event_data=fundamental,
                    vix_data=fundamental,
                    gap_pct=fundamental.get("gap_pct"),
                )
            except Exception as exc:
                self._log.error("TimeContextEngine get_context failed", error=str(exc))
                tout = None
            if isinstance(tout, dict):
                context["session_phase"] = tout.get("session_phase")
                context["session_size_multiplier"] = tout.get("size_multiplier")
                context["day_type"] = tout.get("day_type")

        market_dna_engine = getattr(self, "market_dna_engine", None)
        if market_dna_engine is not None and hasattr(market_dna_engine, "compute_fingerprint"):
            try:
                key_levels = context.get("key_levels", {}) or {}
                if not isinstance(key_levels, dict):
                    key_levels = {}

                fundamental = context.get("fundamental", {}) or {}
                if not isinstance(fundamental, dict):
                    fundamental = {}

                previous_close = float(context.get("previous_close", 0.0) or 0.0)
                opening_price = float(context.get("opening_price", 0.0) or 0.0)
                current_price = float(context.get("spot", 0.0) or 0.0)

                gap_pct = key_levels.get("gap_pct")
                if gap_pct is None:
                    gap_pct = fundamental.get("gap_pct")
                if gap_pct is None and previous_close > 0.0 and opening_price > 0.0:
                    gap_pct = ((opening_price - previous_close) / previous_close) * 100.0

                opening_volume_ratio = key_levels.get("opening_volume_ratio", fundamental.get("opening_volume_ratio", 1.0))
                try:
                    gap_pct_value = float(gap_pct or 0.0)
                except Exception:
                    gap_pct_value = 0.0
                try:
                    opening_volume_ratio_value = float(opening_volume_ratio or 1.0)
                except Exception:
                    opening_volume_ratio_value = 1.0

                fingerprint = market_dna_engine.compute_fingerprint(
                    key_levels=key_levels,
                    fundamental_data=fundamental,
                    regime=str(context.get("regime") or "UNKNOWN").upper(),
                    gap_pct=gap_pct_value,
                    previous_close=previous_close,
                    opening_price=opening_price,
                    current_price=current_price,
                    opening_volume_ratio=opening_volume_ratio_value,
                )

                context["day_personality"] = {
                    "day_type": str(context.get("day_type") or "UNKNOWN").upper(),
                    "regime_at_1015": fingerprint.get("regime_at_1015", str(context.get("regime") or "UNKNOWN")),
                    "gap_direction": fingerprint.get("gap_direction", 0),
                    "gap_magnitude": fingerprint.get("gap_magnitude", 0.0),
                    "date": fingerprint.get("date"),
                }
                context["historical_match_score"] = float(market_dna_engine.get_historical_match_score() or 0.0)
                context["session_memory"] = market_dna_engine.get_session_memory()
            except Exception as exc:
                self._log.error("MarketDNAEngine compute failed", error=str(exc))

        context.setdefault("day_personality", {"day_type": str(context.get("day_type") or "UNKNOWN").upper()})
        context.setdefault("historical_match_score", 0.0)
        context.setdefault(
            "session_memory",
            {
                "levels_defended": [],
                "levels_broken": [],
                "failed_breakouts": [],
                "traps_detected": 0,
                "dominant_direction_morning": "",
                "momentum_decay_started": False,
                "largest_move_size": 0.0,
                "volume_profile_shift": "STABLE",
            },
        )

        context.setdefault("regime", context.get("regime") or "UNKNOWN")
        context.setdefault("session_phase", context.get("session_phase") or "UNKNOWN")
        context.setdefault("narrative_label", context.get("narrative_label") or "NEUTRAL")
        context.setdefault("narrative_fit_factor", 1.0)
        context.setdefault("narrative_fit_factors", context.get("narrative_fit_factors") or {})

    # -------------------------------------------------------------------------
    # Adapters (UNCHANGED)
    # -------------------------------------------------------------------------
    def _captain_step(self, context: Dict[str, Any]) -> Optional[Any]:
        cap = self.captain_engine
        if cap is None:
            raise StartupError("CaptainEngine is not initialized")

        if hasattr(cap, "on_new_candle") and callable(getattr(cap, "on_new_candle")):
            return cap.on_new_candle(context)  # type: ignore[misc]
        if hasattr(cap, "step") and callable(getattr(cap, "step")):
            return cap.step(context)  # type: ignore[misc]
        if hasattr(cap, "update") and callable(getattr(cap, "update")):
            return cap.update(context)  # type: ignore[misc]

        raise StartupError("CaptainEngine has no step/update/on_new_candle method")

    def _scan_strategies(self, context: Dict[str, Any], brain_filter: Optional[Any]) -> list:
        se = self.strategy_engine
        if se is None:
            raise StartupError("StrategyEngine not initialized")

        f1 = context.get("features_1m", {}) or {}
        f5 = context.get("features_5m", {}) or {}
        f15 = context.get("features_15m", {}) or {}

        # FIX: Extract active_brains from CaptainCycleResult if needed
        brain_list = brain_filter
        if brain_filter is not None and hasattr(brain_filter, 'active_brains'):
            brain_list = brain_filter.active_brains
        elif brain_filter is not None and not isinstance(brain_filter, list):
            brain_list = None  # fallback to None for invalid types

        try:
            return se.scan(features_1m=f1, features_5m=f5, features_15m=f15, context=context, brain_filter=brain_list)  # type: ignore[misc]
        except TypeError:
            return se.scan(features_1m=f1, features_5m=f5, features_15m=f15, context=context)  # type: ignore[misc]

    def _ml_anomaly_blocks(self, scored: Any, context: Dict[str, Any]) -> bool:
        ad = self.anomaly_detector
        if ad is None:
            return False

        if hasattr(ad, "score") and callable(getattr(ad, "score")):
            s = ad.score(scored, context)  # type: ignore[misc]
            try:
                s = float(s)
            except Exception:
                return False
            pause_th = float(self._config.get("ml", "anomaly_pause_threshold", default=0.7))
            safe_th = float(self._config.get("ml", "anomaly_safe_threshold", default=0.9))
            if s >= safe_th:
                self._log.error("Anomaly SAFE threshold breached; blocking", score=s)
                return True
            if s >= pause_th:
                self._log.warning("Anomaly pause threshold breached; blocking", score=s)
                return True
        elif hasattr(ad, "should_block") and callable(getattr(ad, "should_block")):
            return bool(ad.should_block(scored, context))  # type: ignore[misc]
        return False

    def _ml_quality_blocks(self, scored: Any, context: Dict[str, Any]) -> bool:
        lf = self.ml_filter
        if lf is None:
            return False

        if hasattr(lf, "evaluate") and callable(getattr(lf, "evaluate")):
            res = lf.evaluate(scored, context)  # type: ignore[misc]
            if isinstance(res, dict):
                return bool(res.get("reject", False))
            return False
        if hasattr(lf, "should_reject") and callable(getattr(lf, "should_reject")):
            return bool(lf.should_reject(scored, context))  # type: ignore[misc]
        return False

    def _risk_approve(self, scored: Any, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        self._log.debug("Risk Engine evaluation started")
        re = self.risk_engine
        if re is None:
            raise StartupError("RiskEngine not initialized")

        out = None
        method_used = None
        if hasattr(re, "assess") and callable(getattr(re, "assess")):
            out = re.assess(scored, context, self.state)  # type: ignore[misc]
            method_used = "assess"
        elif hasattr(re, "evaluate") and callable(getattr(re, "evaluate")):
            out = re.evaluate(scored, context, self.state)  # type: ignore[misc]
            method_used = "evaluate"
        elif hasattr(re, "size_and_validate") and callable(getattr(re, "size_and_validate")):
            out = re.size_and_validate(scored, context, self.state)  # type: ignore[misc]
            method_used = "size_and_validate"
        else:
            raise StartupError("RiskEngine has no assess/evaluate/size_and_validate method")

        self._log.debug(f"Risk Engine called via {method_used}")

        if isinstance(out, dict):
            approved = bool(out.get("approved", out.get("allow", False)))
            plan = out.get("plan") if isinstance(out.get("plan"), dict) else out
            self._log.debug(f"Risk Engine result: approved={approved}")
            return approved, plan if isinstance(plan, dict) else {}
        if isinstance(out, tuple) and len(out) == 2:
            approved, plan = out
            self._log.debug(f"Risk Engine result: approved={approved}")
            return bool(approved), plan if isinstance(plan, dict) else {}
        self._log.debug("Risk Engine returned unexpected format")
        return False, {}

    def _execute(self, exec_plan: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not exec_plan:
            return False

        ov = self.order_validator
        if ov is not None and hasattr(ov, "validate") and callable(getattr(ov, "validate")):
            try:
                ok, reason = ov.validate(exec_plan)  # type: ignore[misc]
                if not ok:
                    self._log.warning("Order validation failed", reason=reason, plan=exec_plan)
                    return False
            except Exception as e:
                self._log.error("Order validation exception", error=str(e))
                return False

        broker = self.broker
        if broker is None:
            self._log.error("No broker configured for execution")
            return False

        try:
            if hasattr(broker, "execute") and callable(getattr(broker, "execute")):
                res = broker.execute(exec_plan, context)  # type: ignore[misc]
                return bool(res) if isinstance(res, bool) else True

            symbol = exec_plan.get("symbol")
            qty = exec_plan.get("qty")
            price = exec_plan.get("price")
            direction = exec_plan.get("direction")
            algo_id = exec_plan.get("algo_id", self._config.get("compliance", "algo_id", default=""))

            if symbol and qty and price and direction and hasattr(broker, "place_limit_order"):
                oid = broker.place_limit_order(symbol=symbol, qty=int(qty), price=float(price), direction=direction, algo_id=algo_id)  # type: ignore[misc]
                self._log.info("Order placed", order_id=str(oid), plan=exec_plan)
                return True

            self._log.warning("Broker API mismatch; cannot execute plan", plan=exec_plan)
            return False

        except Exception as e:
            self._log.error("Execution failed", error=str(e), plan=exec_plan)
            return False

    # -------------------------------------------------------------------------
    # Health / heartbeat
    # -------------------------------------------------------------------------
    def _emit_health(self) -> None:
        now = datetime.now().astimezone()
        de_status = {}
        try:
            if self.data_engine is not None and hasattr(self.data_engine, "get_status"):
                de_status = self.data_engine.get_status()  # type: ignore[misc]
        except Exception as e:
            de_status = {"error": str(e)}

        ws_status = {}
        try:
            ws = getattr(self.data_engine, "_ws", None) if self.data_engine is not None else None
            if ws is not None and hasattr(ws, "get_status"):
                ws_status = self._status_to_dict(ws.get_status())
        except Exception as e:
            ws_status = {"error": str(e)}

        candle_counts = {}
        try:
            # CRITICAL VISIBILITY FIX: DataEngine stores CandleBuilder at `_candle_builder`
            cb = getattr(self.data_engine, "_candle_builder", None) if self.data_engine is not None else None
            cb = cb or self.candle_builder
            if cb is not None and hasattr(cb, "candles"):
                candles = getattr(cb, "candles", {})
                if isinstance(candles, dict):
                    candle_counts = {k: len(v) for k, v in candles.items() if hasattr(v, "__len__")}
        except Exception as e:
            candle_counts = {"error": str(e)}

        fe_status = {}
        try:
            if self.feature_engine is not None and hasattr(self.feature_engine, "get_status"):
                fe_status = self.feature_engine.get_status()  # type: ignore[misc]
        except Exception as e:
            fe_status = {"error": str(e)}

        oc_status = {}
        try:
            if self.option_chain_poller is not None and hasattr(self.option_chain_poller, "get_status"):
                oc_status = self.option_chain_poller.get_status()  # type: ignore[misc]
        except Exception as e:
            oc_status = {"error": str(e)}

        cp_status = {}
        try:
            if self._control_plane is not None and hasattr(self._control_plane, "current_status"):
                cp_status = self._status_to_dict(self._control_plane.current_status())
        except Exception as e:
            cp_status = {"error": str(e)}

        ipc_status = {}
        try:
            if self._dashboard_ipc is not None and hasattr(self._dashboard_ipc, "get_status"):
                ipc_status = self._dashboard_ipc.get_status()  # type: ignore[misc]
        except Exception as e:
            ipc_status = {"error": str(e)}

        existing_engine_health = {}
        try:
            if self.state is not None and hasattr(self.state, "snapshot"):
                snap = self.state.snapshot()
                if isinstance(snap, dict):
                    maybe_engine_health = snap.get("engine_health", {})
                    if isinstance(maybe_engine_health, dict):
                        existing_engine_health = dict(maybe_engine_health)
        except Exception:
            existing_engine_health = {}

        engine_health = dict(existing_engine_health)

        if isinstance(de_status, dict):
            data_alive = bool(de_status.get("is_running", False))
            feed_health = str(de_status.get("feed_health", "UNKNOWN") or "UNKNOWN").upper()
            warnings = []
            if feed_health in {"DOWN", "STALE"}:
                warnings.append(f"feed_health={feed_health}")
            if bool(de_status.get("validator_degraded", False)):
                warnings.append("validator_degraded")
            if not bool(de_status.get("ws_connected", False)):
                warnings.append("websocket_disconnected")
            if bool(de_status.get("using_fallback", False)):
                warnings.append("using_fallback")
            status = "ERROR" if not data_alive else ("WARNING" if warnings else "OK")
            engine_health["data_engine"] = self._engine_health_row(
                alive=data_alive,
                status=status,
                last_heartbeat=now,
                last_error="; ".join(warnings),
            )

        if isinstance(ws_status, dict) and ws_status:
            feed_alive = bool(ws_status.get("is_connected", False))
            feed_warnings = []
            if bool(ws_status.get("feed_permanently_dead", False)):
                feed_warnings.append(str(ws_status.get("permanent_failure_reason") or "feed_permanently_dead"))
            elif bool(ws_status.get("feed_stale", False)):
                feed_warnings.append("feed_stale")
            if ws_status.get("last_error"):
                feed_warnings.append(str(ws_status.get("last_error")))
            feed_status = "ERROR" if bool(ws_status.get("feed_permanently_dead", False)) else ("WARNING" if feed_warnings else "OK")
            engine_health["websocket_feed"] = self._engine_health_row(
                alive=feed_alive,
                status=feed_status,
                last_heartbeat=ws_status.get("last_tick_time") or ws_status.get("last_connect_time") or now,
                last_error="; ".join([item for item in feed_warnings if item]),
            )

        if isinstance(fe_status, dict) and fe_status:
            total_errors = int(fe_status.get("total_errors", 0) or 0)
            feature_status = "WARNING" if total_errors > 0 else "OK"
            engine_health["feature_engine"] = self._engine_health_row(
                alive=True,
                status=feature_status,
                last_heartbeat=now,
                last_error=(f"errors={total_errors}" if total_errors > 0 else ""),
            )

        if isinstance(oc_status, dict) and oc_status:
            consecutive_failures = int(oc_status.get("consecutive_failures", 0) or 0)
            cooldown_until = oc_status.get("cooldown_until")
            option_last_error = str(oc_status.get("last_error") or "")
            option_status = "ERROR" if cooldown_until and consecutive_failures >= 3 else ("WARNING" if consecutive_failures > 0 or option_last_error else "OK")
            engine_health["option_chain_poller"] = self._engine_health_row(
                alive=True,
                status=option_status,
                last_heartbeat=oc_status.get("last_poll_time") or oc_status.get("last_success_time") or now,
                last_error=option_last_error or (f"consecutive_failures={consecutive_failures}" if consecutive_failures > 0 else ""),
            )

        if isinstance(cp_status, dict) and cp_status:
            emergency_flag = int(cp_status.get("emergency_stop_flag", 0) or 0)
            control_status = "ERROR" if emergency_flag else "OK"
            engine_health["control_plane"] = self._engine_health_row(
                alive=bool(cp_status.get("heartbeat_timestamp_ns", 0)),
                status=control_status,
                last_heartbeat=cp_status.get("heartbeat_timestamp_ns") or now,
                last_error=str(cp_status.get("safe_lock_reason") or ""),
            )

        if isinstance(ipc_status, dict) and ipc_status:
            engine_health["dashboard_ipc_bridge"] = self._engine_health_row(
                alive=True,
                status="OK",
                last_heartbeat=now,
                last_error="",
            )

        # Publish sampled engine health back into MarketState so the dashboard
        # sees the full matrix instead of only the captain row.
        try:
            if self.state is not None and hasattr(self.state, "update") and engine_health:
                self.state.update(engine_health=engine_health)
        except Exception as e:
            self._log.warning("Engine health publication failed", error=str(e), rows=list(engine_health.keys()))

        cap_state = None
        try:
            cap = self.captain_engine
            if cap is not None:
                if hasattr(cap, "get_state") and callable(getattr(cap, "get_state")):
                    cap_state = cap.get_state()  # type: ignore[misc]
                elif hasattr(cap, "state"):
                    cap_state = getattr(cap, "state")
        except Exception:
            cap_state = None

        self._log.info(
            "Heartbeat",
            mode=self._args.mode.upper(),
            data_engine=de_status,
            candles=candle_counts,
            feature_engine=fe_status,
            captain_state=cap_state,
        )

        try:
            if self._control_plane is not None:
                state_name = None
                if isinstance(cap_state, dict):
                    state_name = cap_state.get("system_state")
                if state_name is None and self.state is not None and hasattr(self.state, "snapshot"):
                    snap = self.state.snapshot()
                    if isinstance(snap, dict):
                        state_name = snap.get("system_state")
                self._control_plane.publish_state(reason=str(state_name or "heartbeat"))
                self._control_plane.publish_heartbeat()
        except Exception as e:
            self._log.warning("Dashboard control plane heartbeat publish failed", error=str(e))

    def _poll_dashboard_control_plane(self) -> Optional[str]:
        if self._control_plane is None:
            return None
        try:
            return self._control_plane.poll_emergency_request()
        except Exception as e:
            self._log.warning("Dashboard control plane emergency poll failed", error=str(e))
            return None

    def _service_dashboard_ipc(self, now_mono: float) -> None:
        if self._dashboard_ipc is None:
            return

        try:
            snap = self.state.snapshot() if (self.state is not None and hasattr(self.state, "snapshot")) else {}
            if not isinstance(snap, dict):
                snap = {}
            self._dashboard_ipc.publish_due_frames(snap, now_mono=now_mono)
        except Exception as e:
            self._log.warning("Dashboard IPC snapshot publish failed", error=str(e))

        try:
            commands = self._dashboard_ipc.poll_commands(limit=32)
        except Exception as e:
            self._log.warning("Dashboard IPC command poll failed", error=str(e))
            commands = []

        for command in commands:
            self._handle_dashboard_command(command)

    def _handle_dashboard_command(self, command: Dict[str, Any]) -> None:
        if not isinstance(command, dict):
            return
        command_type = str(command.get("type") or command.get("command") or "").strip().lower()
        if not command_type:
            self._log.warning("Dashboard IPC command missing type", payload=command)
            return

        if command_type == "emergency_stop":
            reason = str(command.get("reason") or command.get("source") or "dashboard_command")
            self._log.critical("Dashboard IPC emergency command received", reason=reason)
            try:
                if self._control_plane is not None:
                    self._control_plane.publish_state(reason=f"dashboard_command:{reason}", emergency_stop_flag=1)
            except Exception as e:
                self._log.warning("Dashboard control plane publish for emergency command failed", error=str(e))
            self.stop(reason=f"dashboard_command:{reason}")
            return

        # Command channel is now real, but future commands must still be
        # introduced in roadmap order. Unknown commands are logged and ignored.
        self._log.info("Dashboard IPC command received and ignored (not yet handled)", command_type=command_type, payload=command)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args() -> RuntimeArgs:
    p = argparse.ArgumentParser(description="Junior Aladdin — Main Orchestrator")
    p.add_argument(
        "--mode",
        required=True,
        choices=["observe", "paper", "live"],
        help="System mode: observe (no trades), paper (simulated), live (real orders; requires --confirm-live).",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required to enable LIVE mode. Without this flag, LIVE will refuse to start.",
    )
    p.add_argument("--health-interval-sec", type=float, default=5.0, help="Heartbeat log interval seconds.")
    p.add_argument("--candle-poll-interval-sec", type=float, default=0.25, help="Polling interval for candle close detection.")
    ns = p.parse_args()

    return RuntimeArgs(
        mode=str(ns.mode),
        config_path=str(ns.config),
        confirm_live=bool(ns.confirm_live),
        health_interval_sec=float(ns.health_interval_sec),
        candle_poll_interval_sec=float(ns.candle_poll_interval_sec),
    )


def _install_signal_handlers(orchestrator: JuniorAladdinOrchestrator) -> None:
    def _handler(signum, _frame):
        orchestrator.stop(reason=f"signal_{signum}")

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


def main() -> int:
    args = _parse_args()

    cfg_path = Path(args.config_path)
    if not cfg_path.exists():
        print(f"[FATAL] config file not found: {cfg_path}")
        return 2

    try:
        orch = JuniorAladdinOrchestrator(args)
        _install_signal_handlers(orch)

        orch.initialize_components()
        orch.start()

        return orch.run_forever()

    except StartupError as e:
        print(f"[STARTUP ERROR] {e}")
        return 2
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"[FATAL] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())