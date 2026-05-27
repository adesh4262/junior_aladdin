from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
from collections import deque
import threading

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    from datetime import timezone
    IST = timezone(timedelta(hours=5, minutes=30))

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.core.market_state import MarketState
from src.features.mtf_alignment import compute_mtf_alignment
from src.strategies.strategy_engine import StrategyEngine
from src.filters.trap_detector import TrapDetector
from src.filters.opportunity_scorer import OpportunityScorer
from src.risk.risk_engine import RiskEngine

# --- ML FILTER (NEW) ---
from src.ml.lightgbm_filter import LightGBMFilter, MLFilterDecision
from src.ml.garch_forecaster import GARCHForecaster
from src.ml.anomaly_detector import AnomalyDetector, AnomalyDecision
from src.ml.regime_classifier_backup import RegimeClassifierBackup, RegimePrediction


Direction = str  # "LONG" | "SHORT"


@dataclass(frozen=True)
class Opportunity:
    direction: Direction
    score: float = 0.0
    brain: str = "unknown"
    strategy: str = "unknown"
    symbol: str = "unknown"
    qty: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BrainSignal:
    brain: str
    direction: Optional[Direction]
    confidence: float


@dataclass(frozen=True)
class CaptainDecision:
    system_state: str
    confidence_score: float
    allow_trading: bool
    reason: str
    probe_mode: bool = False


@dataclass(frozen=True)
class DecisionMemoryEntry:
    timestamp: datetime
    system_state: str
    confidence_score: float
    opportunities_total: int
    approved: int
    vetoed: int


@dataclass(frozen=True)
class CaptainCycleResult:
    decision: CaptainDecision
    data_ok: bool
    feed_ok: bool
    risk_ok: bool
    context_score: float
    risk_level: float
    consensus_direction: Optional[Direction]
    consensus_strength: float
    suppressed_brains: List[str]
    allowed_brains: List[str]
    size_hint_multiplier: float
    opportunities_total: int
    trades_approved: int
    veto_count: int
    approved: List[Opportunity]
    vetoed: List[Tuple[Opportunity, str]]


class CaptainEngine:
    VALID_SYSTEM_STATES = {"BOOT", "OBSERVE", "ACTIVE", "CAUTIOUS", "SAFE", "LOCKED", "SHUTDOWN"}
    VALID_FEED_HEALTH = {"HEALTHY", "DELAYED", "STALE", "DOWN"}

    BRAIN_WEIGHTS = {
        "institutional_brain": 1.0,
        "structural_brain": 0.8,
        "tactical_brain": 0.5,
        "adaptive_brain": 0.4,
        "institutional": 1.0,
        "structural": 0.8,
        "tactical": 0.5,
        "adaptive": 0.4,
    }

    def __init__(self, market_state: MarketState) -> None:
        if market_state is None:
            raise ValueError("CaptainEngine requires a valid MarketState reference.")
        self._ms = market_state
        self._log = setup_logger("captain")

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Timings
        self._heartbeat_interval_sec = float(Config.get("captain", "heartbeat_interval_sec", default=5.0))
        self._market_open = self._parse_hhmm(Config.get("market", "market_open", default="09:15"), default=time(9, 15))
        self._market_close = self._parse_hhmm(Config.get("market", "market_close", default="15:30"), default=time(15, 30))
        self._observe_until = self._parse_hhmm(Config.get("captain", "observe_until", default="09:30"), default=time(9, 30))

        # Thresholds
        self._base_thr_observe_to_active = float(Config.get("captain", "base_threshold_observe_to_active", default=75.0))
        self._base_thr_cautious_to_active = float(Config.get("captain", "base_threshold_cautious_to_active", default=80.0))
        self._base_thr_safe_to_cautious = float(Config.get("captain", "base_threshold_safe_to_cautious", default=60.0))
        self._thr_active_to_cautious = float(Config.get("captain", "threshold_active_to_cautious", default=55.0))
        self._thr_active_to_safe = float(Config.get("captain", "threshold_active_to_safe", default=45.0))

        # Dwell
        self._min_dwell_cautious_to_active_sec = int(Config.get("captain", "min_dwell_cautious_to_active_sec", default=180))
        self._min_dwell_safe_to_cautious_sec = int(Config.get("captain", "min_dwell_safe_to_cautious_sec", default=300))
        self._min_dwell_before_upgrades_sec = int(Config.get("captain", "min_dwell_before_upgrades_sec", default=60))

        # SAFE trap prevention
        self._safe_trap_grace_min = int(Config.get("captain", "safe_trap_grace_min", default=30))
        self._relax_step_points = float(Config.get("captain", "relax_step_points", default=2.0))
        self._relax_step_every_min = int(Config.get("captain", "relax_step_every_min", default=5))
        self._relax_floor_observe_to_active = float(Config.get("captain", "relax_floor_observe_to_active", default=65.0))
        self._relax_floor_cautious_to_active = float(Config.get("captain", "relax_floor_cautious_to_active", default=70.0))
        self._relax_floor_safe_to_cautious = float(Config.get("captain", "relax_floor_safe_to_cautious", default=55.0))

        # Probe trades
        self._probe_interval_min = int(Config.get("captain", "probe_interval_min", default=15))
        self._probe_size_multiplier = float(Config.get("captain", "probe_size_multiplier", default=0.25))
        self._probe_requires_consensus = bool(Config.get("captain", "probe_requires_consensus", default=True))
        self._probe_min_consensus_strength = float(Config.get("captain", "probe_min_consensus_strength", default=55.0))
        self._probe_recovery_boost_min = int(Config.get("captain", "probe_recovery_boost_min", default=15))

        # Throttles
        self._max_trades_per_minute = int(Config.get("captain", "max_trades_per_minute", default=3))
        self._cooldown_after_loss_sec = int(Config.get("captain", "cooldown_after_loss_sec", default=60))
        self._veto_burst_count = int(Config.get("captain", "veto_burst_count", default=3))
        self._veto_burst_window_sec = int(Config.get("captain", "veto_burst_window_sec", default=300))
        self._veto_burst_safe_sec = int(Config.get("captain", "veto_burst_safe_sec", default=600))

        # Final veto safety
        self._final_veto_min_dqs = float(Config.get("captain", "final_veto_min_dqs", default=70.0))
        self._volatile_mtf_min_abs = float(Config.get("captain", "volatile_mtf_min_abs", default=4.0))

        # Adaptive adjustments
        self._dd_improving_bonus = float(Config.get("captain", "dd_improving_bonus", default=-2.0))
        self._dd_worsening_penalty = float(Config.get("captain", "dd_worsening_penalty", default=4.0))
        self._blocked_rate_bonus = float(Config.get("captain", "blocked_rate_bonus", default=-2.0))
        self._blocked_rate_trigger = float(Config.get("captain", "blocked_rate_trigger", default=0.80))
        self._context_override_consensus_min = float(Config.get("captain", "context_override_consensus_min", default=70.0))
        self._context_override_max_boost = float(Config.get("captain", "context_override_max_boost", default=10.0))

        # Metrics window
        self._metrics_window_sec = int(Config.get("captain", "metrics_window_sec", default=300))

        # Risk limits
        self._max_daily_loss_pct = float(Config.get("risk", "max_daily_loss_pct", default=0.02))
        self._drawdown_lock_pct = float(Config.get("risk", "drawdown_lock_pct", default=0.10))

        # Qty guard
        lot_size = int(Config.get("market", "lot_size", default=65))
        self._max_qty_per_trade = int(Config.get("captain", "max_qty_per_trade", default=lot_size))

        # Internal state
        snap = self._safe_snapshot()
        init_state = snap.get("system_state")
        self._current_state = init_state if isinstance(init_state, str) and init_state in self.VALID_SYSTEM_STATES else "BOOT"

        # Critical: seed state_entered_at from MarketState.timestamp if present to avoid dwell anomalies
        ts = self._coerce_datetime(snap.get("timestamp"))
        self._state_entered_at = ts if isinstance(ts, datetime) else datetime.now(IST)

        self._decision_memory: Deque[DecisionMemoryEntry] = deque(maxlen=300)
        self._approval_times: Deque[datetime] = deque(maxlen=5000)
        self._veto_times: Deque[datetime] = deque(maxlen=5000)
        self._dd_history: Deque[Tuple[datetime, float]] = deque(maxlen=2000)
        self._opp_stats_window: Deque[Tuple[datetime, int, int, int]] = deque(maxlen=5000)

        self._loss_cooldown_until: Optional[datetime] = None
        self._last_consecutive_losses: Optional[int] = None
        self._last_any_approval_at: Optional[datetime] = None

        # Probe tracking
        self._probe_pending: bool = False
        self._last_probe_attempt_at: Optional[datetime] = None
        self._recovery_boost_until: Optional[datetime] = None

        # Forced SAFE
        self._forced_safe_until: Optional[datetime] = None
        self._forced_safe_reason: str = ""

        # Manual overrides internal
        self._manual_lock: bool = False
        self._manual_shutdown: bool = False

        # Pipeline engines (instantiate once)
        self._strategy_engine = StrategyEngine()
        self._trap_detector = TrapDetector()
        self._opportunity_scorer = OpportunityScorer()
        self._risk_engine = RiskEngine()
        self._garch_forecaster = GARCHForecaster()
        if not bool(getattr(self._garch_forecaster, "_fitted", False)):
            try:
                self._garch_forecaster.fit(lookback_days=int(Config.get("ml", "garch_lookback_days", default=60)))
            except Exception as exc:
                self._log.warning("GARCH forecaster fit skipped/failed", error=str(exc))
        self._log.info("GARCH Forecaster initialized", status=self._garch_forecaster.get_status())

        # --- ML FILTER (NEW) ---
        self.ml_filter = LightGBMFilter()
        self._log.info("ML Filter initialized")

        # --- ANOMALY DETECTOR (NEW) ---
        self.anomaly_detector = AnomalyDetector()
        if not self.anomaly_detector._fitted:
            self.anomaly_detector.fit_from_parquet()
        self._log.info("Anomaly Detector initialized")

        # --- XGBOOST REGIME BACKUP (NEW) ---
        self.regime_backup = RegimeClassifierBackup()
        if not self.regime_backup._fitted:
            self.regime_backup.fit_from_parquet()
        self._log.info("XGBoost Regime Backup initialized")

    # ------------------------- Manual Controls -------------------------

    def set_manual_lock(self, value: bool) -> None:
        with self._lock:
            self._manual_lock = bool(value)
            self._log.warning("Manual lock set", manual_lock=self._manual_lock)

    def set_manual_shutdown(self, value: bool) -> None:
        with self._lock:
            self._manual_shutdown = bool(value)
            self._log.warning("Manual shutdown set", manual_shutdown=self._manual_shutdown)

    def clear_manual_overrides(self) -> None:
        with self._lock:
            self._manual_lock = False
            self._manual_shutdown = False
            self._log.info("Manual overrides cleared")

    # ------------------------- Probe Outcome Feedback -------------------------

    def notify_trade_outcome(
        self,
        pnl_rupees: float,
        timestamp: Optional[datetime] = None,
        was_probe: bool = False,
        within_risk_limits: bool = True,
    ) -> None:
        ts = self._coerce_datetime(timestamp) or datetime.now(IST)
        with self._lock:
            if not was_probe:
                return
            self._probe_pending = False
            success = (float(pnl_rupees) >= 0.0) or bool(within_risk_limits)
            if success:
                self._recovery_boost_until = ts + timedelta(minutes=max(1, self._probe_recovery_boost_min))
                self._log.info(
                    "Probe trade success -> recovery boost",
                    pnl_rupees=pnl_rupees,
                    recovery_boost_until=self._recovery_boost_until,
                )
            else:
                self._forced_safe_until = ts + timedelta(minutes=10)
                self._forced_safe_reason = "probe_failed"
                self._log.warning(
                    "Probe trade failed -> forced SAFE",
                    pnl_rupees=pnl_rupees,
                    forced_safe_until=self._forced_safe_until,
                )

    # ------------------------- Heartbeat (optional) -------------------------

    def start(self) -> None:
        with self._lock:
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._log.warning("Captain heartbeat already running")
                return
            self._stop_event.clear()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="captain_heartbeat", daemon=True)
            self._heartbeat_thread.start()
            self._log.info("Captain started", heartbeat_interval_sec=self._heartbeat_interval_sec)

    def stop(self, timeout_sec: float = 2.0) -> None:
        with self._lock:
            self._stop_event.set()
        th = self._heartbeat_thread
        if th and th.is_alive():
            th.join(timeout=timeout_sec)
        self._log.info("Captain stopped")

    def _heartbeat_loop(self) -> None:
        self._log.info("Captain heartbeat loop started")
        while not self._stop_event.is_set():
            try:
                now = datetime.now(IST)
                snap = self._safe_snapshot()
                state = snap.get("system_state")
                state = state if isinstance(state, str) and state in self.VALID_SYSTEM_STATES else self._current_state
                decision = CaptainDecision(
                    system_state=state,
                    confidence_score=0.0,
                    allow_trading=False,
                    reason="heartbeat",
                    probe_mode=False,
                )
                self._publish(
                    decision,
                    now,
                    0,
                    0,
                    0,
                    {
                        "observe_to_active": self._base_thr_observe_to_active,
                        "cautious_to_active": self._base_thr_cautious_to_active,
                        "safe_to_cautious": self._base_thr_safe_to_cautious,
                    },
                )
            except Exception as e:  # pragma: no cover
                self._log.error("Captain heartbeat error", error=str(e))
            self._stop_event.wait(max(1.0, float(self._heartbeat_interval_sec)))
        self._log.info("Captain heartbeat loop stopped")

    # ------------------------- Main Cycle -------------------------

    def step(
        self,
        opportunities: Optional[List[Union[Opportunity, Dict[str, Any]]]] = None,
        brain_signals: Optional[List[Union[BrainSignal, Dict[str, Any]]]] = None,
        now: Optional[datetime] = None,
    ) -> CaptainCycleResult:
        ts_now = self._coerce_datetime(now) or datetime.now(IST)

        with self._lock:
            snap = self._safe_snapshot()
            self._update_loss_cooldown(snap, ts_now)

            # Normalize externally supplied opportunities (if provided).
            opps: List[Opportunity] = []
            if opportunities:
                opps = [self._normalize_opportunity(o) for o in opportunities]
                opps = [o for o in opps if o is not None]

            layer = self._evaluate_layers_priority(snap, brain_signals, ts_now)
            dyn_thr = self._compute_dynamic_thresholds(layer, ts_now)

            consensus_dir, consensus_strength, suppressed_brains = self._brain_weighted_consensus(brain_signals)

            # Internal orchestration path: generate opportunities when none are supplied.
            if not opps and not opportunities:
                scan_brains = self._allowed_brains_for_scan(self._current_state)
                opps = self._generate_internal_opportunities(
                    snap=snap,
                    layer=layer,
                    now=ts_now,
                    allowed_brains=scan_brains,
                )

            opportunities_present = len(opps) > 0

            decision = self._decide_state(
                layer=layer,
                dyn_thr=dyn_thr,
                snap=snap,
                now=ts_now,
                opportunities_present=opportunities_present,
                consensus_strength=consensus_strength,
            )

            # If SAFE-probe got enabled in this step and we still have no opportunities,
            # do one probe-time generation pass.
            if not opps and not opportunities and decision.system_state == "SAFE" and decision.probe_mode:
                scan_brains = self._allowed_brains_for_scan(decision.system_state)
                opps = self._generate_internal_opportunities(
                    snap=snap,
                    layer=layer,
                    now=ts_now,
                    allowed_brains=scan_brains,
                )
                opportunities_present = len(opps) > 0

            allowed_brains = self._allowed_brains_for_state(decision.system_state, decision.probe_mode)

            # Rate-limit probe attempts even if veto happens
            if decision.system_state == "SAFE" and decision.probe_mode:
                self._last_probe_attempt_at = ts_now

            vetoed: List[Tuple[Opportunity, str]] = []

            # Suppress minority brain opps
            if suppressed_brains and opps:
                kept: List[Opportunity] = []
                for o in opps:
                    if o.brain in suppressed_brains:
                        vetoed.append((o, f"suppressed_brain({o.brain})"))
                        self._veto_times.append(ts_now)
                    else:
                        kept.append(o)
                opps = kept

            approved: List[Opportunity] = []
            for o in opps:
                ok, reason = self.final_veto(
                    opportunity=o,
                    decision=decision,
                    layer=layer,
                    dyn_thresholds=dyn_thr,
                    snap=snap,
                    now=ts_now,
                    consensus_direction=consensus_dir,
                    consensus_strength=consensus_strength,
                    allowed_brains=allowed_brains,
                )
                if ok:
                    approved.append(self._tag_probe(o, decision.probe_mode))
                    self._approval_times.append(ts_now)
                    self._last_any_approval_at = ts_now
                    if decision.probe_mode:
                        self._probe_pending = True
                else:
                    vetoed.append((o, reason))
                    self._veto_times.append(ts_now)

            if approved:
                approved = self._apply_trade_frequency_cap(approved, ts_now, vetoed)

            opportunities_total = len(opps)
            approved_count = len(approved)
            veto_count = len(vetoed)
            self._opp_stats_window.append((ts_now, opportunities_total, approved_count, veto_count))

            self._apply_veto_burst_safe(ts_now)

            self._decision_memory.append(
                DecisionMemoryEntry(
                    timestamp=ts_now,
                    system_state=decision.system_state,
                    confidence_score=float(decision.confidence_score),
                    opportunities_total=opportunities_total,
                    approved=approved_count,
                    vetoed=veto_count,
                )
            )

            self._publish(decision, ts_now, opportunities_total, approved_count, veto_count, dyn_thr)

            size_hint_multiplier = self._size_hint_multiplier(decision, layer, consensus_strength)

            return CaptainCycleResult(
                decision=decision,
                data_ok=layer.get("data_ok", False),
                feed_ok=layer.get("feed_ok", False),
                risk_ok=layer.get("risk_ok", False),
                context_score=float(layer.get("context_score", 0.0)),
                risk_level=float(layer.get("risk_level", 0.0)),
                consensus_direction=consensus_dir,
                consensus_strength=float(consensus_strength),
                suppressed_brains=suppressed_brains,
                allowed_brains=allowed_brains,
                size_hint_multiplier=size_hint_multiplier,
                opportunities_total=opportunities_total,
                trades_approved=approved_count,
                veto_count=veto_count,
                approved=approved,
                vetoed=vetoed,
            )

    # ========================= Layer Evaluation (Priority: Risk > Data > Feed > Context > Brains) =========================

    def _evaluate_layers_priority(self, snap: Dict[str, Any], brain_signals: Optional[List[Union[BrainSignal, Dict[str, Any]]]], now: datetime) -> Dict[str, Any]:
        risk = self._risk_layer(snap, now)
        if not risk["risk_ok"]:
            return {
                **risk,
                "data_ok": False,
                "data_score": 0.0,
                "feed_ok": False,
                "feed_health": "DOWN",
                "feed_score": 0.0,
                "context_score": 0.0,
                "regime": self._safe_str(snap.get("regime"), "UNKNOWN").upper(),
                "narrative_label": self._safe_str(snap.get("narrative_label"), "UNKNOWN").upper(),
                "weighted_mtf": self._extract_weighted_mtf_from_features(snap.get("features")),
                "brain_score": 0.0,
                "confidence_score": 0.0,
            }

        data = self._data_layer(snap, now)
        if not data["data_ok"]:
            return {
                **risk,
                **data,
                "feed_ok": False,
                "feed_health": self._safe_feed_health(snap.get("feed_health")),
                "feed_score": 0.0,
                "context_score": 0.0,
                "regime": self._safe_str(snap.get("regime"), "UNKNOWN").upper(),
                "narrative_label": self._safe_str(snap.get("narrative_label"), "UNKNOWN").upper(),
                "weighted_mtf": self._extract_weighted_mtf_from_features(snap.get("features")),
                "brain_score": 0.0,
                "confidence_score": 0.0,
            }

        feed = self._feed_layer(snap)
        if not feed["feed_ok"]:
            return {
                **risk,
                **data,
                **feed,
                "context_score": 0.0,
                "regime": self._safe_str(snap.get("regime"), "UNKNOWN").upper(),
                "narrative_label": self._safe_str(snap.get("narrative_label"), "UNKNOWN").upper(),
                "weighted_mtf": self._extract_weighted_mtf_from_features(snap.get("features")),
                "brain_score": 0.0,
                "confidence_score": 0.0,
            }

        context = self._context_layer(snap)
        brain_score = self._brain_score_avg(brain_signals)

        consensus_dir, consensus_strength, _ = self._brain_weighted_consensus(brain_signals)
        context_score = float(context["context_score"])
        if (
            consensus_strength >= self._context_override_consensus_min
            and context["regime"] not in {"CHOP", "EVENT"}
            and context["narrative_label"] != "EVENT_RISK"
            and context_score < 55.0
        ):
            boost = min(self._context_override_max_boost, (consensus_strength - self._context_override_consensus_min) / 3.0)
            context_score = float(max(0.0, min(100.0, context_score + boost)))

        confidence = self._weighted_score(
            [
                ("risk", float(risk["risk_level"]), 0.30),
                ("data", float(data["data_score"]), 0.20),
                ("feed", float(feed["feed_score"]), 0.15),
                ("context", float(context_score), 0.20),
                ("brains", float(brain_score), 0.15),
            ]
        )

        return {
            **risk,
            **data,
            **feed,
            **context,
            "context_score": context_score,
            "brain_score": float(brain_score),
            "confidence_score": float(confidence),
            "weighted_consensus_direction": consensus_dir,
            "weighted_consensus_strength": float(consensus_strength),
        }

    def _risk_layer(self, snap: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        if self._manual_shutdown:
            return {"risk_ok": False, "risk_level": 0.0, "risk_reason": "manual_shutdown(internal)", "risk_trajectory": "worsening", "circuit_breaker": False}
        if self._manual_lock:
            return {"risk_ok": False, "risk_level": 0.0, "risk_reason": "manual_lock(internal)", "risk_trajectory": "worsening", "circuit_breaker": False}

        eh = snap.get("engine_health")
        engine_health = eh if isinstance(eh, dict) else {}
        risk_engine = engine_health.get("risk_engine") if isinstance(engine_health.get("risk_engine"), dict) else {}

        circuit_breaker = bool(risk_engine.get("circuit_breaker", False))
        if circuit_breaker:
            return {"risk_ok": False, "risk_level": 0.0, "risk_reason": "circuit_breaker", "risk_trajectory": "worsening", "circuit_breaker": True}

        dd_raw = snap.get("drawdown_pct")
        dd = float(dd_raw) if isinstance(dd_raw, (int, float)) else 0.0
        dd = max(0.0, dd)

        daily_pnl_raw = snap.get("daily_pnl")
        daily_pnl = float(daily_pnl_raw) if isinstance(daily_pnl_raw, (int, float)) else 0.0

        capital_raw = snap.get("capital")
        capital = float(capital_raw) if isinstance(capital_raw, (int, float)) else None

        daily_loss_pct = 0.0
        if capital is not None and capital > 0 and daily_pnl < 0:
            daily_loss_pct = (-daily_pnl) / capital

        if daily_loss_pct >= self._max_daily_loss_pct:
            return {"risk_ok": False, "risk_level": 0.0, "risk_reason": f"daily_loss_pct>={self._max_daily_loss_pct:.3f}", "risk_trajectory": "worsening", "circuit_breaker": False}

        if dd >= self._drawdown_lock_pct:
            return {"risk_ok": False, "risk_level": 0.0, "risk_reason": f"drawdown_lock>={self._drawdown_lock_pct:.3f}", "risk_trajectory": "worsening", "circuit_breaker": False}

        self._dd_history.append((now, dd))
        dd_vel = self._drawdown_velocity_per_min(now)
        traj = "flat"
        if dd_vel is not None:
            if dd_vel > 0.002:
                traj = "worsening"
            elif dd_vel < -0.001:
                traj = "improving"

        risk_level = 100.0
        risk_level -= min(60.0, dd * 600.0)
        if traj == "worsening":
            risk_level -= 15.0
        elif traj == "improving":
            risk_level += 5.0
        risk_level = max(0.0, min(100.0, risk_level))

        return {"risk_ok": True, "risk_level": float(risk_level), "risk_reason": "", "risk_trajectory": traj, "circuit_breaker": False}

    def _data_layer(self, snap: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        dqs_raw = snap.get("data_quality_score")
        dqs = float(dqs_raw) if isinstance(dqs_raw, (int, float)) else None

        spot_ts = snap.get("timestamp")
        spot_age = self._age_seconds(spot_ts, now)

        eh = snap.get("engine_health")
        engine_health = eh if isinstance(eh, dict) else {}
        fe_age = self._engine_age_seconds(engine_health, "feature_engine", now)
        oc_age = self._engine_age_seconds(engine_health, "option_chain_poller", now)

        data_ok = True
        penalty = 0.0

        if dqs is None:
            data_ok = False
            dqs = 0.0
            penalty += 50.0
        else:
            if dqs < 40.0:
                data_ok = False
                penalty += 50.0
            elif dqs < 60.0:
                penalty += 20.0

        if spot_age is None or spot_age > 15.0:
            data_ok = False
            penalty += 50.0
        elif spot_age > 5.0:
            penalty += 15.0

        if fe_age is not None and fe_age > 60.0:
            penalty += 10.0
        if oc_age is not None and oc_age > 90.0:
            penalty += 5.0

        data_score = max(0.0, min(100.0, float(dqs) - penalty))
        return {"data_ok": data_ok, "data_score": float(data_score), "data_quality_score": float(dqs)}

    def _feed_layer(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        feed_health = self._safe_feed_health(snap.get("feed_health"))
        feed_ok = feed_health in {"HEALTHY", "DELAYED"}
        feed_score = {"HEALTHY": 100.0, "DELAYED": 70.0, "STALE": 20.0, "DOWN": 0.0}[feed_health]
        return {"feed_ok": feed_ok, "feed_health": feed_health, "feed_score": float(feed_score)}

    def _context_layer(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        regime = self._safe_str(snap.get("regime"), "UNKNOWN").upper()
        narrative = self._safe_str(snap.get("narrative_label"), "UNKNOWN").upper()
        weighted_mtf = self._extract_weighted_mtf_from_features(snap.get("features"))

        score = 50.0
        if regime == "TRENDING":
            score += 18.0
        elif regime == "RANGE":
            score += 10.0
        elif regime == "VOLATILE":
            score -= 8.0
        elif regime == "CHOP":
            score -= 28.0
        elif regime == "EVENT":
            score -= 35.0
        else:
            score -= 5.0

        if narrative == "EVENT_RISK":
            score -= 50.0
        elif "STRONG_" in narrative:
            score += 8.0
        elif "MILD_" in narrative:
            score += 3.0

        score += min(22.0, abs(float(weighted_mtf)) * 3.0)
        score = max(0.0, min(100.0, score))
        return {"context_score": float(score), "regime": regime, "narrative_label": narrative, "weighted_mtf": float(weighted_mtf)}

    # ========================= Dynamic thresholds =========================
    # (unchanged)
    def _compute_dynamic_thresholds(self, layer: Dict[str, Any], now: datetime) -> Dict[str, float]:
        thr_obs = float(self._base_thr_observe_to_active)
        thr_c2a = float(self._base_thr_cautious_to_active)
        thr_s2c = float(self._base_thr_safe_to_cautious)

        traj = layer.get("risk_trajectory", "flat")
        if traj == "improving":
            thr_obs += self._dd_improving_bonus
            thr_c2a += self._dd_improving_bonus
            thr_s2c += self._dd_improving_bonus * 0.5
        elif traj == "worsening":
            thr_obs += self._dd_worsening_penalty
            thr_c2a += self._dd_worsening_penalty
            thr_s2c += self._dd_worsening_penalty * 0.5

        blocked_rate, _ = self._compute_recent_rates(now, window_sec=self._metrics_window_sec)
        if blocked_rate is not None and blocked_rate >= self._blocked_rate_trigger:
            thr_obs += self._blocked_rate_bonus
            thr_c2a += self._blocked_rate_bonus
            thr_s2c += self._blocked_rate_bonus * 0.5

        relax = self._compute_relaxation_points(now)
        thr_obs -= relax
        thr_c2a -= relax
        thr_s2c -= relax * 0.5

        thr_obs = max(self._relax_floor_observe_to_active, thr_obs)
        thr_c2a = max(self._relax_floor_cautious_to_active, thr_c2a)
        thr_s2c = max(self._relax_floor_safe_to_cautious, thr_s2c)

        thr_obs = float(max(0.0, min(100.0, thr_obs)))
        thr_c2a = float(max(0.0, min(100.0, thr_c2a)))
        thr_s2c = float(max(0.0, min(100.0, thr_s2c)))

        return {"observe_to_active": thr_obs, "cautious_to_active": thr_c2a, "safe_to_cautious": thr_s2c}

    def _compute_relaxation_points(self, now: datetime) -> float:
        if self._current_state not in {"SAFE", "CAUTIOUS"}:
            return 0.0
        last_approval = self._last_any_approval_at or self._state_entered_at
        minutes_since = max(0.0, (now - last_approval).total_seconds() / 60.0)
        if minutes_since < float(self._safe_trap_grace_min):
            return 0.0
        extra = minutes_since - float(self._safe_trap_grace_min)
        step_every = max(1.0, float(self._relax_step_every_min))
        steps = int(extra // step_every) + 1
        return float(steps) * float(self._relax_step_points)

    # ========================= State Machine =========================
    # (unchanged)
    def _decide_state(
        self,
        layer: Dict[str, Any],
        dyn_thr: Dict[str, float],
        snap: Dict[str, Any],
        now: datetime,
        opportunities_present: bool,
        consensus_strength: float,
    ) -> CaptainDecision:
        t_local = self._local_time(now)
        if not self._is_market_hours(t_local):
            if t_local < self._market_open:
                return CaptainDecision("BOOT", float(layer.get("confidence_score", 0.0)), False, "pre_market", probe_mode=False)
            return CaptainDecision("SHUTDOWN", float(layer.get("confidence_score", 0.0)), False, "market_closed", probe_mode=False)

        if self._manual_shutdown:
            return CaptainDecision("SHUTDOWN", 0.0, False, "manual_shutdown(internal)", probe_mode=False)
        if self._manual_lock:
            return CaptainDecision("LOCKED", 0.0, False, "manual_lock(internal)", probe_mode=False)

        if self._forced_safe_until and now < self._forced_safe_until:
            return CaptainDecision("SAFE", float(layer.get("confidence_score", 0.0)), False, f"forced_safe({self._forced_safe_reason})", probe_mode=False)

        # Priority gates
        if not layer.get("risk_ok", False):
            if layer.get("circuit_breaker", False):
                return CaptainDecision("LOCKED", 0.0, False, "circuit_breaker", probe_mode=False)
            return CaptainDecision("SAFE", 0.0, False, f"risk_not_ok({layer.get('risk_reason','')})", probe_mode=False)
        if not layer.get("data_ok", False):
            return CaptainDecision("SAFE", 0.0, False, "data_not_ok", probe_mode=False)
        if not layer.get("feed_ok", False):
            return CaptainDecision("SAFE", 0.0, False, f"feed_not_ok({layer.get('feed_health')})", probe_mode=False)

        baseline = "OBSERVE" if t_local < self._observe_until else "ACTIVE"
        confidence = float(layer.get("confidence_score", 0.0))
        regime = str(layer.get("regime", "UNKNOWN")).upper()
        dwell_sec = self._time_in_current_state_sec(now)

        recovery_boost = (self._recovery_boost_until is not None and now < self._recovery_boost_until)

        # Probe mode eligibility only when we actually have opportunities to test
        probe_mode = False
        if self._current_state == "SAFE" and opportunities_present:
            probe_mode = self._should_allow_probe(layer, now)

        # BOOT/OBSERVE
        if self._current_state in {"BOOT", "OBSERVE"}:
            if baseline == "OBSERVE":
                return self._transition("OBSERVE", confidence, False, "observe_window", False, now)
            if dwell_sec >= self._min_dwell_before_upgrades_sec and confidence >= dyn_thr["observe_to_active"]:
                return self._transition("ACTIVE", confidence, True, "observe_to_active", False, now)
            return self._transition("OBSERVE", confidence, False, "observe_hold", False, now)

        # LOCKED
        if self._current_state == "LOCKED":
            return CaptainDecision("LOCKED", confidence, False, "locked", probe_mode=False)

        # ACTIVE
        if self._current_state == "ACTIVE":
            if confidence < self._thr_active_to_safe or regime in {"CHOP", "EVENT"}:
                return self._transition("SAFE", confidence, False, "active_to_safe", False, now)
            if confidence < self._thr_active_to_cautious:
                return self._transition("CAUTIOUS", confidence, False, "active_to_cautious", False, now)
            if self._loss_cooldown_until and now < self._loss_cooldown_until:
                return CaptainDecision("ACTIVE", confidence, False, "loss_cooldown", probe_mode=False)
            return CaptainDecision("ACTIVE", confidence, True, "active", probe_mode=False)

        # CAUTIOUS
        if self._current_state == "CAUTIOUS":
            if confidence < self._thr_active_to_safe or regime in {"CHOP", "EVENT"}:
                return self._transition("SAFE", confidence, False, "cautious_to_safe", False, now)
            if dwell_sec >= self._min_dwell_cautious_to_active_sec and confidence >= dyn_thr["cautious_to_active"]:
                if self._loss_cooldown_until and now < self._loss_cooldown_until:
                    return CaptainDecision("CAUTIOUS", confidence, False, "loss_cooldown", probe_mode=False)
                return self._transition("ACTIVE", confidence, True, "cautious_to_active", False, now)
            return CaptainDecision("CAUTIOUS", confidence, False, "cautious_hold", probe_mode=False)

        # SAFE (SMART SAFE)
        if self._current_state == "SAFE":
            if probe_mode:
                if self._loss_cooldown_until and now < self._loss_cooldown_until:
                    return CaptainDecision("SAFE", confidence, False, "loss_cooldown", probe_mode=False)
                return CaptainDecision("SAFE", confidence, True, "safe_probe_trade", probe_mode=True)

            if recovery_boost and dwell_sec >= 60.0 and confidence >= (dyn_thr["safe_to_cautious"] - 3.0):
                return self._transition("CAUTIOUS", confidence, False, "safe_to_cautious(recovery_boost)", False, now)

            open_positions = snap.get("open_positions")
            has_open_positions = isinstance(open_positions, list) and len(open_positions) > 0
            if not has_open_positions and dwell_sec >= self._min_dwell_safe_to_cautious_sec and confidence >= dyn_thr["safe_to_cautious"]:
                return self._transition("CAUTIOUS", confidence, False, "safe_to_cautious", False, now)

            return CaptainDecision("SAFE", confidence, False, "safe_hold", probe_mode=False)

        return CaptainDecision("SAFE", confidence, False, "fallback_safe", probe_mode=False)

    def _transition(self, new_state: str, confidence: float, allow: bool, reason: str, probe_mode: bool, now: datetime) -> CaptainDecision:
        if new_state not in self.VALID_SYSTEM_STATES:
            self._log.critical("Invalid transition target; forcing SAFE", requested=new_state)
            new_state = "SAFE"
            allow = False
            reason = "invalid_transition_target"
            probe_mode = False

        if new_state != self._current_state:
            self._current_state = new_state
            self._state_entered_at = now
            if new_state == "ACTIVE":
                self._recovery_boost_until = None
                self._forced_safe_until = None
                self._forced_safe_reason = ""
                self._probe_pending = False

        return CaptainDecision(new_state, float(confidence), bool(allow), reason, probe_mode=bool(probe_mode))

    # ========================= Probe Logic =========================
    # (unchanged)
    def _should_allow_probe(self, layer: Dict[str, Any], now: datetime) -> bool:
        if self._forced_safe_until and now < self._forced_safe_until:
            return False
        if self._probe_pending:
            return False
        if self._last_probe_attempt_at is not None:
            if now - self._last_probe_attempt_at < timedelta(minutes=max(1, self._probe_interval_min)):
                return False
        regime = str(layer.get("regime", "UNKNOWN")).upper()
        if regime in {"CHOP", "EVENT"}:
            return False
        dqs = float(layer.get("data_quality_score", 0.0))
        if dqs < max(60.0, self._final_veto_min_dqs - 5.0):
            return False
        return True

    # ========================= Brain Weighted Consensus =========================
    # (unchanged)
    def _brain_weighted_consensus(self, brain_signals: Optional[List[Union[BrainSignal, Dict[str, Any]]]]) -> Tuple[Optional[Direction], float, List[str]]:
        if not brain_signals:
            return None, 0.0, []

        parsed: List[BrainSignal] = []
        for bs in brain_signals:
            try:
                if isinstance(bs, BrainSignal):
                    parsed.append(bs)
                elif isinstance(bs, dict):
                    d = bs.get("direction")
                    parsed.append(
                        BrainSignal(
                            brain=str(bs.get("brain", "unknown")),
                            direction=d if d in {"LONG", "SHORT"} else None,
                            confidence=float(bs.get("confidence", 0.0)),
                        )
                    )
            except Exception:
                continue

        parsed = [p for p in parsed if p.direction in {"LONG", "SHORT"}]
        if not parsed:
            return None, 0.0, []

        long_w = 0.0
        short_w = 0.0
        long_brains: List[str] = []
        short_brains: List[str] = []

        for p in parsed:
            w = self.BRAIN_WEIGHTS.get(p.brain, self.BRAIN_WEIGHTS.get(p.brain.replace("_brain", ""), 0.4))
            conf = max(0.0, min(100.0, float(p.confidence)))
            wc = w * conf
            if p.direction == "LONG":
                long_w += wc
                long_brains.append(p.brain)
            else:
                short_w += wc
                short_brains.append(p.brain)

        if long_w == short_w:
            return None, 0.0, []

        consensus = "LONG" if long_w > short_w else "SHORT"
        minority = short_brains if consensus == "LONG" else long_brains
        minority = sorted(list(set(minority)))

        total = max(1e-6, long_w + short_w)
        gap = abs(long_w - short_w)
        strength = (gap / total) * 100.0
        magnitude = min(100.0, total / 2.0)
        strength = 0.6 * strength + 0.4 * (magnitude * 0.5)
        strength = float(max(0.0, min(100.0, strength)))

        return consensus, strength, minority

    def _brain_score_avg(self, brain_signals: Optional[List[Union[BrainSignal, Dict[str, Any]]]]) -> float:
        if not brain_signals:
            return 50.0
        confs: List[float] = []
        for bs in brain_signals:
            try:
                if isinstance(bs, BrainSignal):
                    confs.append(float(bs.confidence))
                elif isinstance(bs, dict):
                    confs.append(float(bs.get("confidence", 0.0)))
            except Exception:
                continue
        if not confs:
            return 40.0
        return float(max(0.0, min(100.0, sum(confs) / max(1, len(confs)))))

    def _allowed_brains_for_state(self, system_state: str, probe_mode: bool) -> List[str]:
        if system_state == "ACTIVE":
            return ["structural_brain", "tactical_brain", "institutional_brain", "adaptive_brain"]
        if system_state == "CAUTIOUS":
            return ["structural_brain", "institutional_brain"]
        if system_state == "SAFE" and probe_mode:
            return ["structural_brain"]
        return []

    def _allowed_brains_for_scan(self, system_state: str) -> List[str]:
        if system_state == "ACTIVE":
            return ["structural_brain", "tactical_brain", "institutional_brain", "adaptive_brain"]
        if system_state == "CAUTIOUS":
            return ["structural_brain", "institutional_brain"]
        if system_state == "SAFE":
            return ["structural_brain"]
        return []

    @staticmethod
    def _strategy_brain_filter_from_allowed(allowed_brains: List[str]) -> List[str]:
        mapping = {
            "structural_brain": "STRUCTURAL",
            "tactical_brain": "TACTICAL",
            "institutional_brain": "INSTITUTIONAL",
            "adaptive_brain": "ADAPTIVE",
        }
        out: List[str] = []
        for b in allowed_brains:
            mapped = mapping.get(str(b).strip().lower())
            if mapped:
                out.append(mapped)
        return sorted(list(set(out)))

    def _resolve_mtf_payload(
        self,
        *,
        features: Dict[str, Any],
        per_tf: Dict[str, Dict[str, Any]],
        snap: Dict[str, Any],
        key_levels: Dict[str, Any],
        fundamental: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve the phase-3 MTF payload from live feature inputs.

        Priority:
          1) Existing payload in features["mtf"] or features["mtf_alignment"]
          2) Compute full payload from per-TF feature blocks

        The computed payload is merged last so current live data wins over partial leftovers.
        """
        payload: Dict[str, Any] = {}

        if isinstance(features, dict):
            for key in ("mtf", "mtf_alignment"):
                node = features.get(key)
                if isinstance(node, dict) and node:
                    payload.update(dict(node))

        mtf_input: Dict[str, Dict[str, Any]] = {}
        for tf in ("1min", "3min", "5min", "15min"):
            node = per_tf.get(tf)
            if isinstance(node, dict) and node:
                mtf_input[tf] = dict(node)

        if mtf_input:
            vix_source = fundamental.get("vix_level", snap.get("vix_level", snap.get("vix_at_entry", None)))
            vix_level = self._safe_float_value(vix_source, 0.0)
            if not isinstance(vix_level, (int, float)) or vix_level <= 0:
                vix_level = None

            session_phase = self._safe_str(snap.get("session_phase"), "PRE_MARKET").upper()

            try:
                computed = compute_mtf_alignment(
                    mtf_input,
                    vix_level=vix_level,
                    session_phase=session_phase,
                    key_levels=key_levels,
                )
                if isinstance(computed, dict) and computed:
                    payload.update(computed)
            except Exception as exc:
                self._log.warning(
                    "Captain MTF compute failed; using existing payload only",
                    error=str(exc),
                )

        return payload

    def _generate_internal_opportunities(
        self,
        snap: Dict[str, Any],
        layer: Dict[str, Any],
        now: datetime,
        allowed_brains: List[str],
    ) -> List[Opportunity]:
        if not allowed_brains:
            return []

        features = snap.get("features") if isinstance(snap.get("features"), dict) else {}
        features_1m = features.get("1min") if isinstance(features.get("1min"), dict) else {}
        features_5m = features.get("5min") if isinstance(features.get("5min"), dict) else {}
        features_15m = features.get("15min") if isinstance(features.get("15min"), dict) else {}

        context = self._build_pipeline_context(snap, layer)

        strategy_brain_filter = self._strategy_brain_filter_from_allowed(allowed_brains)

        try:
            raw_opps = self._strategy_engine.scan(
                features_1m=features_1m,
                features_5m=features_5m,
                features_15m=features_15m,
                context=context,
                brain_filter=strategy_brain_filter,
            )
        except Exception as e:
            self._log.error("Internal strategy scan failed", error=str(e))
            return []

        generated: List[Opportunity] = []
        for raw in raw_opps:
            raw_dict = self._strategy_opportunity_to_dict(raw)
            if raw_dict is None:
                continue

            trap = self._trap_detector.evaluate(raw_dict, context)
            if bool(getattr(trap, "reject", False)):
                continue

            scoring_context = dict(context)
            scoring_context["trap_probability"] = self._safe_float_value(getattr(trap, "trap_probability", 0.0), 0.0)
            scoring_context["narrative_fit_factor"] = self._narrative_fit_factor_for_direction(raw_dict.get("direction"), snap)

            scored = self._opportunity_scorer.score_opportunity(raw_dict, scoring_context)
            if bool(getattr(scored, "hard_reject", True)):
                continue

            scored_dict = dict(raw_dict)
            scored_dict["final_score"] = self._safe_float_value(
                getattr(scored, "final_score", raw_dict.get("raw_score", 0.0)),
                0.0,
            )

            # ------------------- ML FILTER GATE (NEW) -------------------
            ml_decision = self.ml_filter.evaluate(scored_dict, context)
            scored_dict["ml_probability"] = float(ml_decision.probability)
            scored_dict["ml_action"] = str(ml_decision.action)
            scored_dict["ml_reduce_size"] = bool(ml_decision.reduce_size)
            scored_dict["ml_rejection_reason"] = str(ml_decision.rejection_reason)

            if ml_decision.action == "REJECT":
                self._log.info(
                    "Trade rejected by ML filter",
                    direction=self._safe_str(scored_dict.get("direction"), ""),
                    strategy=self._safe_str(scored_dict.get("strategy"), "unknown"),
                    brain=self._safe_str(scored_dict.get("brain"), "unknown"),
                    symbol=self._safe_str(scored_dict.get("symbol"), "unknown"),
                    final_score=float(scored_dict.get("final_score", 0.0) or 0.0),
                    ml_probability=float(ml_decision.probability),
                    ml_action=str(ml_decision.action),
                    ml_rejection_reason=str(ml_decision.rejection_reason),
                )
                continue
            # ------------------------------------------------------------

            # ------------------- ANOMALY DETECTOR (NEW) -------------------
            anomaly_decision = self.anomaly_detector.evaluate(scored_dict, context)
            scored_dict["anomaly_score"] = float(anomaly_decision.anomaly_score)
            scored_dict["anomaly_action"] = str(anomaly_decision.action)

            if anomaly_decision.pause_recommended:
                self._forced_safe_until = max(
                    self._forced_safe_until or now,
                    now + timedelta(minutes=5)
                )
                self._forced_safe_reason = "anomaly_pause"
                self._log.warning(
                    "Anomaly detected — entering PAUSE",
                    anomaly_score=anomaly_decision.anomaly_score,
                    pause_minutes=5,
                )
                continue

            if anomaly_decision.safe_recommended:
                self._forced_safe_until = max(
                    self._forced_safe_until or now,
                    now + timedelta(minutes=10)
                )
                self._forced_safe_reason = "anomaly_safe"
                self._log.warning(
                    "Anomaly detected — entering SAFE",
                    anomaly_score=anomaly_decision.anomaly_score,
                    safe_minutes=10,
                )
                continue
            # --------------------------------------------------------------

            # ----------- XGBOOST REGIME BACKUP (NEW) -----------
            rule_regime = context.get("regime", "UNKNOWN")
            regime_pred = self.regime_backup.predict(scored_dict, context, rule_regime)
            scored_dict["xgb_regime"] = str(regime_pred.predicted_regime)
            scored_dict["xgb_confidence"] = float(regime_pred.confidence)
            scored_dict["xgb_agrees_with_rule"] = bool(regime_pred.agrees_with_rule)

            if not regime_pred.agrees_with_rule and rule_regime != "UNKNOWN":
                self._log.warning(
                    "XGBoost regime mismatch — reducing size",
                    rule_regime=rule_regime,
                    xgb_regime=regime_pred.predicted_regime,
                    xgb_confidence=round(regime_pred.confidence, 3),
                    size_multiplier=regime_pred.size_multiplier,
                )
            # ----------------------------------------------------

            # Risk engine propagation (NEW)
            risk_context = dict(context)
            risk_context["ml_reduce_size"] = bool(scored_dict.get("ml_reduce_size", False))

            risk_decision = self._risk_engine.evaluate(scored_dict, risk_context)
            if not bool(getattr(risk_decision, "allow_trade", False)):
                continue

            captain_opp = self._to_captain_opportunity(
                raw_opp=scored_dict,  # include ML fields + final_score in meta
                scored=scored,
                trap=trap,
                risk_decision=risk_decision,
                now=now,
                snap=snap,
                ml_decision=ml_decision,
            )
            if captain_opp is not None:
                generated.append(captain_opp)

        return generated

    def _build_pipeline_context(self, snap: Dict[str, Any], layer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline context contract for downstream engines.

        Includes required compatibility keys for ML filter:
          - smart_money_5m / smart_money_15m
          - per_tf (1min/5min/15min dicts) + features_1m/5m/15m (backward compat)
        """
        session_memory = snap.get("session_memory") if isinstance(snap.get("session_memory"), dict) else {}
        options_features = snap.get("options_features") if isinstance(snap.get("options_features"), dict) else {}
        microstructure = snap.get("microstructure") if isinstance(snap.get("microstructure"), dict) else {}
        key_levels = snap.get("key_levels") if isinstance(snap.get("key_levels"), dict) else {}
        smart_money = snap.get("smart_money") if isinstance(snap.get("smart_money"), dict) else {}
        volume_profile = snap.get("volume_profile") if isinstance(snap.get("volume_profile"), dict) else {}
        fundamental = snap.get("fundamental") if isinstance(snap.get("fundamental"), dict) else {}

        # Per-timeframe features (Captain stores inside snap["features"])
        per_tf: Dict[str, Dict[str, Any]] = {}
        features = snap.get("features") if isinstance(snap.get("features"), dict) else {}
        if isinstance(features.get("1min"), dict):
            per_tf["1min"] = features.get("1min", {})
        if isinstance(features.get("5min"), dict):
            per_tf["5min"] = features.get("5min", {})
        if isinstance(features.get("15min"), dict):
            per_tf["15min"] = features.get("15min", {})

        sm5 = smart_money.get("5min") if isinstance(smart_money.get("5min"), dict) else {}
        sm15 = smart_money.get("15min") if isinstance(smart_money.get("15min"), dict) else {}

        mtf_payload = self._resolve_mtf_payload(
            features=features,
            per_tf=per_tf,
            snap=snap,
            key_levels=key_levels,
            fundamental=fundamental,
        )

        garch_status: Dict[str, Any] = {}
        garch_high_vol = False
        try:
            garch_status = self._garch_forecaster.get_status() if self._garch_forecaster is not None else {}
            garch_high_vol = bool(garch_status.get("is_high_vol_regime", False))
        except Exception as exc:
            self._log.warning("Captain GARCH status unavailable", error=str(exc))
            garch_status = {}
            garch_high_vol = False

        if mtf_payload:
            mtf_alignment_signal = int(mtf_payload.get("dominant_direction", 0) or 0)
            weighted_mtf_default = mtf_payload.get("weighted_mtf", self._extract_weighted_mtf_from_features(features))
        else:
            mtf_alignment_signal = 0
            weighted_mtf_default = self._extract_weighted_mtf_from_features(features)

        weighted_mtf_value = self._safe_float_value(layer.get("weighted_mtf", weighted_mtf_default), 0.0)

        return {
            "regime": self._safe_str(snap.get("regime"), "UNKNOWN").upper(),
            "narrative_label": self._safe_str(snap.get("narrative_label"), "UNKNOWN").upper(),
            "narrative_score": self._safe_float_value(snap.get("narrative_score", 0.0), 0.0),
            "session_phase": self._safe_str(snap.get("session_phase"), "PRE_MARKET").upper(),
            "session_memory": session_memory,
            "options": options_features,
            "microstructure": microstructure,
            "key_levels": key_levels,
            "volume_profile": volume_profile,
            "fundamental": fundamental,
            "smart_money": smart_money,
            # Context fix keys (MANDATORY)
            "smart_money_5m": sm5,
            "smart_money_15m": sm15,
            # Features structure (Captain actual)
            "per_tf": per_tf,
            # Phase-3 MTF payload (full, not just weighted_mtf)
            "mtf": mtf_payload,
            "mtf_alignment": mtf_payload,
            "mtf_alignment_signal": mtf_alignment_signal,
            "mtf_label": str(mtf_payload.get("mtf_label", "NEUTRAL") if isinstance(mtf_payload, dict) else "NEUTRAL"),
            "mtf_trap_zone": bool(mtf_payload.get("mtf_trap_zone", False) if isinstance(mtf_payload, dict) else False),
            "confluence_bonus_applied": bool(mtf_payload.get("confluence_bonus_applied", False) if isinstance(mtf_payload, dict) else False),
            "confluence_bonus_reason": mtf_payload.get("confluence_bonus_reason") if isinstance(mtf_payload, dict) else None,
            "trend_strength_with_confluence": self._safe_float_value(
                mtf_payload.get("trend_strength_with_confluence", 0.0) if isinstance(mtf_payload, dict) else 0.0,
                0.0,
            ),
            "tf_details": mtf_payload.get("tf_details", {}) if isinstance(mtf_payload, dict) else {},
            "tf_states": mtf_payload.get("tf_states", {}) if isinstance(mtf_payload, dict) else {},
            "tf_confidences": mtf_payload.get("tf_confidences", {}) if isinstance(mtf_payload, dict) else {},
            "mtf_warnings": mtf_payload.get("warnings", []) if isinstance(mtf_payload, dict) else [],
            "garch_high_vol": bool(garch_high_vol),
            "garch_forecast": self._safe_float_value(garch_status.get("last_forecast"), 0.0) if garch_status else 0.0,
            "garch_median_volatility": self._safe_float_value(garch_status.get("median_volatility"), 0.0) if garch_status else 0.0,
            "garch_model_type": str(garch_status.get("model_type", "NONE")) if garch_status else "NONE",
            "garch_status": garch_status,
            # Backward compatibility keys
            "features_1m": per_tf.get("1min", {}),
            "features_5m": per_tf.get("5min", {}),
            "features_15m": per_tf.get("15min", {}),
            "weighted_mtf": weighted_mtf_value,
            "data_quality_score": self._safe_float_value(layer.get("data_quality_score", snap.get("data_quality_score", 0.0)), 0.0),
            "capital": self._safe_float_value(snap.get("capital"), 50000.0),
            "daily_pnl": self._safe_float_value(snap.get("daily_pnl"), 0.0),
            "trades_today": self._safe_int_value(snap.get("trades_today"), 0),
            "consecutive_losses": self._safe_int_value(snap.get("consecutive_losses"), 0),
            "drawdown_pct": self._safe_float_value(snap.get("drawdown_pct"), 0.0),
            "tilt_score": self._safe_float_value(snap.get("tilt_score"), 0.0),
            "is_expiry_day": self._safe_str(snap.get("day_type"), "").upper() == "EXPIRY_DAY",
            "expiry_size_factor": 1.0,
            "mode": self._safe_str(snap.get("mode"), "ALERT").upper(),
        }

    @staticmethod
    def _strategy_opportunity_to_dict(raw_opp: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw_opp, dict):
            return dict(raw_opp)
        to_dict = getattr(raw_opp, "to_dict", None)
        if callable(to_dict):
            try:
                data = to_dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                return None
        return None

    def _to_captain_opportunity(
        self,
        raw_opp: Dict[str, Any],
        scored: Any,
        trap: Any,
        risk_decision: Any,
        now: datetime,
        snap: Dict[str, Any],
        ml_decision: Optional[MLFilterDecision] = None,
    ) -> Optional[Opportunity]:
        direction = self._normalize_direction(raw_opp.get("direction"))
        if direction is None:
            return None

        score = self._safe_float_value(getattr(scored, "final_score", raw_opp.get("final_score", raw_opp.get("raw_score", 0.0))), 0.0)
        brain = self._normalize_brain(raw_opp.get("brain"))
        strategy = str(raw_opp.get("strategy", "unknown"))
        symbol = str(raw_opp.get("symbol") or Config.get("market", "index", default="NIFTY"))

        qty_raw = getattr(risk_decision, "recommended_qty", None)
        qty_i = self._safe_int_value(qty_raw, 0)
        qty = qty_i if qty_i > 0 else None

        meta: Dict[str, Any] = {
            "pipeline_source": "captain_internal",
            "raw_direction": raw_opp.get("direction"),
            "trap_score": self._safe_int_value(getattr(trap, "trap_score", 0), 0),
            "risk_lots": self._safe_int_value(getattr(risk_decision, "recommended_lots", 0), 0),
            "risk_qty": self._safe_int_value(getattr(risk_decision, "recommended_qty", 0), 0),
            "estimated_risk_rupees": self._safe_float_value(getattr(risk_decision, "estimated_risk_rupees", 0.0), 0.0),
            "timestamp": now.isoformat(),
            "mode": self._safe_str(snap.get("mode"), "ALERT").upper(),
        }

        # Journal propagation (best-effort via meta)
        if ml_decision is not None:
            meta.update(
                {
                    "ml_probability": float(ml_decision.probability),
                    "ml_action": str(ml_decision.action),
                    "ml_reduce_size": bool(ml_decision.reduce_size),
                    "ml_rejection_reason": str(ml_decision.rejection_reason),
                }
            )
        else:
            # If already attached in raw_opp dict, carry forward
            for k in ("ml_probability", "ml_action", "ml_reduce_size", "ml_rejection_reason"):
                if k in raw_opp:
                    meta[k] = raw_opp.get(k)

        return Opportunity(
            direction=direction,
            score=score,
            brain=brain,
            strategy=strategy,
            symbol=symbol,
            qty=qty,
            meta=meta,
        )

    # --- rest of file unchanged below this line ---
    # (No changes were required by the task outside ML integration points)

    def _narrative_fit_factor_for_direction(self, direction: Any, snap: Dict[str, Any]) -> float:
        d = self._normalize_direction(direction)
        fits = snap.get("narrative_fit_factors") if isinstance(snap.get("narrative_fit_factors"), dict) else {}

        if d == "LONG":
            v = fits.get("long_fit")
            if isinstance(v, (int, float)):
                return self._safe_float_value(v, 0.8)
        if d == "SHORT":
            v = fits.get("short_fit")
            if isinstance(v, (int, float)):
                return self._safe_float_value(v, 0.8)

        label = self._safe_str(snap.get("narrative_label"), "NEUTRAL").upper()
        long_map = {
            "STRONG_BULLISH": 1.2,
            "MILD_BULLISH": 1.0,
            "NEUTRAL": 0.8,
            "MILD_BEARISH": 0.4,
            "STRONG_BEARISH": 0.1,
            "EVENT_RISK": 0.0,
        }
        short_map = {
            "STRONG_BULLISH": 0.1,
            "MILD_BULLISH": 0.4,
            "NEUTRAL": 0.8,
            "MILD_BEARISH": 1.0,
            "STRONG_BEARISH": 1.2,
            "EVENT_RISK": 0.0,
        }
        if d == "SHORT":
            return short_map.get(label, 0.8)
        return long_map.get(label, 0.8)

    @staticmethod
    def _normalize_direction(direction: Any) -> Optional[str]:
        d = str(direction).strip().upper() if direction is not None else ""
        if d in {"LONG", "BUY"}:
            return "LONG"
        if d in {"SHORT", "SELL"}:
            return "SHORT"
        return None

    @staticmethod
    def _normalize_brain(brain: Any) -> str:
        b = str(brain).strip().lower() if brain is not None else ""
        mapping = {
            "structural": "structural_brain",
            "structural_brain": "structural_brain",
            "tactical": "tactical_brain",
            "tactical_brain": "tactical_brain",
            "institutional": "institutional_brain",
            "institutional_brain": "institutional_brain",
            "adaptive": "adaptive_brain",
            "adaptive_brain": "adaptive_brain",
        }
        return mapping.get(b, "unknown")

    @staticmethod
    def _safe_float_value(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int_value(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    # ========================= Final Veto =========================
    # (unchanged)
    def final_veto(
        self,
        opportunity: Opportunity,
        decision: CaptainDecision,
        layer: Dict[str, Any],
        dyn_thresholds: Dict[str, float],
        snap: Dict[str, Any],
        now: datetime,
        consensus_direction: Optional[Direction],
        consensus_strength: float,
        allowed_brains: List[str],
    ) -> Tuple[bool, str]:
        if decision.system_state != "ACTIVE":
            if not (decision.system_state == "SAFE" and decision.probe_mode and decision.allow_trading):
                return False, f"state_not_active({decision.system_state})"
        if not decision.allow_trading:
            return False, "trading_not_allowed_by_captain"

        if self._loss_cooldown_until and now < self._loss_cooldown_until:
            return False, "loss_cooldown"

        dqs = float(layer.get("data_quality_score", 0.0))
        if dqs < float(self._final_veto_min_dqs):
            return False, f"data_quality<{self._final_veto_min_dqs:.0f}"

        if opportunity.brain and opportunity.brain != "unknown":
            if allowed_brains and opportunity.brain not in allowed_brains:
                return False, f"brain_not_allowed({opportunity.brain})"

        regime = str(layer.get("regime", "UNKNOWN")).upper()
        narrative = str(layer.get("narrative_label", "UNKNOWN")).upper()
        weighted_mtf = float(layer.get("weighted_mtf", 0.0))

        if regime in {"CHOP", "EVENT"}:
            return False, f"regime_{regime.lower()}_veto"
        if narrative == "EVENT_RISK":
            return False, "narrative_event_risk"
        if self._narrative_opposes(narrative, opportunity.direction):
            return False, f"narrative_opposes({narrative})"
        if regime == "VOLATILE" and abs(weighted_mtf) < float(self._volatile_mtf_min_abs):
            return False, "volatile_no_direction"

        if decision.system_state == "SAFE" and decision.probe_mode:
            if self._probe_requires_consensus:
                if consensus_direction is None:
                    return False, "probe_requires_consensus"
                if consensus_strength < float(self._probe_min_consensus_strength):
                    return False, f"probe_weak_consensus({consensus_strength:.1f})"
                if opportunity.direction != consensus_direction:
                    return False, "probe_direction_mismatch"
            open_positions = snap.get("open_positions")
            if isinstance(open_positions, list) and len(open_positions) > 0:
                return False, "probe_blocked_open_positions"

        if decision.system_state == "ACTIVE" and consensus_direction in {"LONG", "SHORT"}:
            if opportunity.direction in {"LONG", "SHORT"} and opportunity.direction != consensus_direction:
                return False, f"contradicts_consensus({consensus_direction})"

        mode = snap.get("mode")
        mode_s = mode if isinstance(mode, str) else "ALERT"
        if mode_s != "LIVE":
            return False, f"mode_not_live({mode_s})"
        if not self._compliance_ok_from_engine_health(snap):
            return False, "compliance_unknown_or_not_ok"

        if opportunity.qty is not None:
            if not isinstance(opportunity.qty, int) or opportunity.qty <= 0:
                return False, "invalid_qty"
            if self._max_qty_per_trade > 0 and opportunity.qty > self._max_qty_per_trade:
                return False, f"qty_exceeds_max({opportunity.qty}>{self._max_qty_per_trade})"

        cap = int(max(0, self._max_trades_per_minute))
        if cap == 0:
            return False, "trade_frequency_cap(cap=0)"
        if self._count_trades_last_60s(now) >= cap:
            return False, "trade_frequency_cap"

        if opportunity.direction not in {"LONG", "SHORT"}:
            return False, "invalid_direction"
        if not isinstance(opportunity.score, (int, float)):
            return False, "invalid_score_type"

        return True, "approved"

    # ========================= Veto burst SAFE =========================
    # (unchanged)
    def _apply_veto_burst_safe(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=max(1, self._veto_burst_window_sec))
        while self._veto_times and self._veto_times[0] < cutoff:
            self._veto_times.popleft()

        if len(self._veto_times) >= int(max(1, self._veto_burst_count)):
            self._forced_safe_until = now + timedelta(seconds=max(60, self._veto_burst_safe_sec))
            self._forced_safe_reason = f"veto_burst({len(self._veto_times)})"
            self._log.warning("Veto burst -> forced SAFE", forced_safe_until=self._forced_safe_until, reason=self._forced_safe_reason)
            self._veto_times.clear()

    # ========================= Loss cooldown =========================
    # (unchanged)
    def _update_loss_cooldown(self, snap: Dict[str, Any], now: datetime) -> None:
        cl_raw = snap.get("consecutive_losses")
        cl = int(cl_raw) if isinstance(cl_raw, int) and cl_raw >= 0 else 0
        if self._last_consecutive_losses is None:
            self._last_consecutive_losses = cl
            return
        if cl > self._last_consecutive_losses:
            self._loss_cooldown_until = now + timedelta(seconds=max(1, self._cooldown_after_loss_sec))
            self._log.warning("Loss detected -> cooldown started", consecutive_losses=cl, cooldown_until=self._loss_cooldown_until)
        self._last_consecutive_losses = cl

    # ========================= Publishing / Metrics / Utilities =========================
    # (unchanged; remainder of file exactly as before)

    def _publish(self, decision: CaptainDecision, now: datetime, opportunities_total: int, approved_count: int, veto_count: int, dyn_thr: Dict[str, float]) -> None:
        if decision.system_state not in self.VALID_SYSTEM_STATES:
            self._log.critical("Refusing to publish invalid system_state", system_state=decision.system_state)
            return

        blocked_rate, approval_ratio = self._compute_recent_rates(now, window_sec=self._metrics_window_sec)
        if blocked_rate is None:
            blocked_rate = 0.0
        if approval_ratio is None:
            approval_ratio = 0.0

        snap = self._safe_snapshot()
        eh = snap.get("engine_health")
        if not isinstance(eh, dict):
            eh = {}
        eh2 = dict(eh)

        eh2["captain"] = {
            "alive": True,
            "last_heartbeat": now,
            "last_state": decision.system_state,
            "reason": decision.reason,
            "consecutive_errors": 0,
            "last_error": "",
            "time_in_current_state_sec": float(self._time_in_current_state_sec(now)),
            "trades_blocked_rate": float(max(0.0, min(1.0, blocked_rate))),
            "approval_ratio": float(max(0.0, min(1.0, approval_ratio))),
            "probe_mode": bool(decision.probe_mode),
            "effective_threshold_observe_to_active": float(dyn_thr.get("observe_to_active", self._base_thr_observe_to_active)),
            "effective_threshold_cautious_to_active": float(dyn_thr.get("cautious_to_active", self._base_thr_cautious_to_active)),
            "effective_threshold_safe_to_cautious": float(dyn_thr.get("safe_to_cautious", self._base_thr_safe_to_cautious)),
        }

        try:
            self._ms.update(system_state=decision.system_state, engine_health=eh2)
        except Exception as e:
            self._log.critical("MarketState.update rejected Captain publish (CONTRACT FAILURE)", error=str(e), attempted_state=decision.system_state)

    def _compute_recent_rates(self, now: datetime, window_sec: int = 300) -> Tuple[Optional[float], Optional[float]]:
        if not self._opp_stats_window:
            return None, None
        cutoff = now - timedelta(seconds=max(1, window_sec))
        while self._opp_stats_window and self._opp_stats_window[0][0] < cutoff:
            self._opp_stats_window.popleft()

        total_opp = sum(x[1] for x in self._opp_stats_window)
        total_appr = sum(x[2] for x in self._opp_stats_window)
        total_veto = sum(x[3] for x in self._opp_stats_window)

        if total_opp <= 0:
            return 0.0, 0.0
        return float(total_veto) / float(total_opp), float(total_appr) / float(total_opp)

    def _count_trades_last_60s(self, now: datetime) -> int:
        cutoff = now - timedelta(seconds=60)
        while self._approval_times and self._approval_times[0] < cutoff:
            self._approval_times.popleft()
        return len(self._approval_times)

    def _apply_trade_frequency_cap(self, approved: List[Opportunity], now: datetime, vetoed: List[Tuple[Opportunity, str]]) -> List[Opportunity]:
        cap = int(max(0, self._max_trades_per_minute))
        if cap <= 0:
            for o in approved:
                vetoed.append((o, "trade_frequency_cap(cap=0)"))
                self._veto_times.append(now)
            return []

        current = self._count_trades_last_60s(now)
        remaining = max(0, cap - current)
        if remaining <= 0:
            for o in approved:
                vetoed.append((o, "trade_frequency_cap"))
                self._veto_times.append(now)
            return []

        if len(approved) <= remaining:
            return approved

        keep = approved[:remaining]
        for o in approved[remaining:]:
            vetoed.append((o, "trade_frequency_cap"))
            self._veto_times.append(now)
        return keep

    def _size_hint_multiplier(self, decision: CaptainDecision, layer: Dict[str, Any], consensus_strength: float) -> float:
        base = max(0.0, min(1.0, float(layer.get("risk_level", 50.0)) / 100.0))
        if decision.system_state == "ACTIVE":
            mult = base
        elif decision.system_state == "CAUTIOUS":
            mult = base * 0.5
        elif decision.system_state == "SAFE" and decision.probe_mode:
            mult = base * max(0.05, min(1.0, float(self._probe_size_multiplier)))
        else:
            mult = 0.0
        if decision.system_state == "ACTIVE" and consensus_strength < 35.0:
            mult *= 0.8
        return float(max(0.0, min(1.0, mult)))

    def _tag_probe(self, opp: Opportunity, probe_mode: bool) -> Opportunity:
        if not probe_mode:
            return opp
        meta = dict(opp.meta) if isinstance(opp.meta, dict) else {}
        meta["probe"] = True
        return Opportunity(direction=opp.direction, score=opp.score, brain=opp.brain, strategy=opp.strategy, symbol=opp.symbol, qty=opp.qty, meta=meta)

    def _extract_weighted_mtf_from_features(self, features: Any) -> float:
        if not isinstance(features, dict):
            return 0.0
        for key in ("mtf", "mtf_alignment"):
            node = features.get(key)
            if isinstance(node, dict):
                wm = node.get("weighted_mtf")
                if isinstance(wm, (int, float)):
                    return float(wm)
        node = features.get("mtf_alignment")
        if isinstance(node, dict):
            wm = node.get("weighted_mtf")
            if isinstance(wm, (int, float)):
                return float(wm)
        wm2 = features.get("weighted_mtf")
        if isinstance(wm2, (int, float)):
            return float(wm2)
        for v in features.values():
            if isinstance(v, dict):
                wm3 = v.get("weighted_mtf")
                if isinstance(wm3, (int, float)):
                    return float(wm3)
        return 0.0

    @staticmethod
    def _narrative_opposes(narrative_label: str, direction: Direction) -> bool:
        lab = narrative_label.upper().strip() if isinstance(narrative_label, str) else ""
        if direction == "LONG" and "BEARISH" in lab:
            return True
        if direction == "SHORT" and "BULLISH" in lab:
            return True
        return False

    def _drawdown_velocity_per_min(self, now: datetime) -> Optional[float]:
        if len(self._dd_history) < 2:
            return None
        cutoff = now - timedelta(minutes=5)
        oldest = None
        for ts, dd in self._dd_history:
            if ts >= cutoff:
                oldest = (ts, dd)
                break
        if oldest is None:
            oldest = self._dd_history[0]
        dt_min = max(0.001, (now - oldest[0]).total_seconds() / 60.0)
        latest_dd = self._dd_history[-1][1]
        return (latest_dd - oldest[1]) / dt_min

    def _time_in_current_state_sec(self, now: datetime) -> float:
        try:
            return max(0.0, (now - self._state_entered_at).total_seconds())
        except Exception:
            return 0.0

    def _safe_feed_health(self, value: Any) -> str:
        if isinstance(value, str) and value in self.VALID_FEED_HEALTH:
            return value
        return "DOWN"

    @staticmethod
    def _safe_str(value: Any, default: str) -> str:
        return value if isinstance(value, str) else default

    def _normalize_opportunity(self, o: Union[Opportunity, Dict[str, Any]]) -> Optional[Opportunity]:
        try:
            if isinstance(o, Opportunity):
                return o
            if isinstance(o, dict):
                d = self._normalize_direction(o.get("direction"))
                if d not in {"LONG", "SHORT"}:
                    return None
                qty_i_raw = self._safe_int_value(o.get("qty"), 0)
                qty_i = qty_i_raw if qty_i_raw > 0 else None
                meta = o.get("meta") if isinstance(o.get("meta"), dict) else None
                score = o.get("score", o.get("final_score", o.get("raw_score", 0.0)))
                score_f = float(score) if isinstance(score, (int, float)) else 0.0
                return Opportunity(
                    direction=d,
                    score=score_f,
                    brain=self._normalize_brain(o.get("brain", "unknown")),
                    strategy=str(o.get("strategy", "unknown")),
                    symbol=str(o.get("symbol", "unknown")),
                    qty=qty_i,
                    meta=meta,
                )
        except Exception:
            return None
        return None

    @staticmethod
    def _weighted_score(components: List[Tuple[str, float, float]]) -> float:
        total_w = sum(max(0.0, float(w)) for _, _, w in components)
        if total_w <= 0:
            return 0.0
        s = 0.0
        for _, score, w in components:
            s += max(0.0, min(100.0, float(score))) * (max(0.0, float(w)) / total_w)
        return float(max(0.0, min(100.0, s)))

    def _compliance_ok_from_engine_health(self, snap: Dict[str, Any]) -> bool:
        eh = snap.get("engine_health")
        if not isinstance(eh, dict):
            return False
        for k in ("compliance_engine", "compliance"):
            node = eh.get(k)
            if isinstance(node, dict) and node.get("ok") is True:
                return True
        return False

    def _safe_snapshot(self) -> Dict[str, Any]:
        try:
            snap = self._ms.snapshot()
            if not isinstance(snap, dict):
                self._log.critical("MarketState.snapshot() returned non-dict", snapshot_type=str(type(snap)))
                return {}
            return snap
        except Exception as e:
            self._log.critical("MarketState.snapshot() failed", error=str(e))
            return {}

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=IST)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=IST)
            except Exception:
                return None
        return None

    def _age_seconds(self, ts: Any, now: datetime) -> Optional[float]:
        dt = self._coerce_datetime(ts)
        if dt is None:
            return None
        try:
            return max(0.0, (now - dt).total_seconds())
        except Exception:
            return None

    def _engine_age_seconds(self, engine_health: Dict[str, Any], engine_name: str, now: datetime) -> Optional[float]:
        if not isinstance(engine_health, dict):
            return None
        node = engine_health.get(engine_name)
        if not isinstance(node, dict):
            return None
        return self._age_seconds(node.get("last_heartbeat"), now)

    @staticmethod
    def _parse_hhmm(value: Any, default: time) -> time:
        if not isinstance(value, str):
            return default.replace(tzinfo=None)
        try:
            hh, mm = value.strip().split(":")
            return time(int(hh), int(mm), tzinfo=None)
        except Exception:
            return default.replace(tzinfo=None)

    @staticmethod
    def _local_time(now: datetime) -> time:
        dt = now.astimezone(IST) if now.tzinfo is not None else now.replace(tzinfo=IST)
        return dt.time().replace(tzinfo=None)

    def _is_market_hours(self, t_local: time) -> bool:
        return self._market_open <= t_local <= self._market_close

    def get_state(self) -> Dict[str, Any]:
        """Return current captain state for heartbeat/monitoring."""
        return {
            "system_state": self._state,
            "mode": self._market_state.mode if self._market_state else None,
            "daily_pnl": self._market_state.daily_pnl if self._market_state else 0.0,
            "drawdown_pct": self._market_state.drawdown_pct if self._market_state else 0.0,
            "trades_today": self._trades_today,
            "consecutive_losses": self._consecutive_losses,
            "last_decision_time": self._last_decision_time.isoformat() if self._last_decision_time else None,
            "last_decision_state": self._decision_memory.get("last_state") if self._decision_memory else None,
            "ml_loaded": self._lightgbm_loaded,
            "ml_filter_active": self._lightgbm_loaded and not self._lightgbm_fallback,
            "garch_loaded": self._garch is not None,
            "anomaly_detector_loaded": self._anomaly_detector is not None,
        }


# Backward-compatible alias for existing imports.
Captain = CaptainEngine


# =============================== SELF TEST ===============================
# (unchanged)
def _build_minimal_candles_dict() -> Dict[str, Any]:
    return {
        "1min": deque(maxlen=400),
        "3min": deque(maxlen=140),
        "5min": deque(maxlen=80),
        "15min": deque(maxlen=30),
    }


def _assert_no_forbidden_top_level_fields(snap: Dict[str, Any]) -> None:
    forbidden = {
        "captain_reason",
        "captain_brain_constraint",
        "compliance_ok",
        "features_last_update",
        "websocket_connected",
        "portfolio_delta",
        "correlation_concentration",
        "manual_lock",
        "manual_shutdown",
        "circuit_breaker",
        "mtf_alignment",
    }
    present = sorted([k for k in forbidden if k in snap])
    assert not present, f"Forbidden top-level fields present in MarketState: {present}"


def _freshen_market_state(ms: MarketState, when: datetime) -> None:
    snap = ms.snapshot()
    spot = snap.get("spot")
    spot_f = float(spot) if isinstance(spot, (int, float)) else 24500.0

    eh = snap.get("engine_health")
    eh2 = dict(eh) if isinstance(eh, dict) else {}
    for k in ("data_engine", "feature_engine", "option_chain_poller", "risk_engine", "compliance_engine"):
        node = dict(eh2.get(k, {})) if isinstance(eh2.get(k, {}), dict) else {}
        node["alive"] = True
        node["last_heartbeat"] = when
        eh2[k] = node
    ce = dict(eh2.get("compliance_engine", {}))
    ce["ok"] = True
    ce["last_heartbeat"] = when
    eh2["compliance_engine"] = ce

    ms.update(spot=spot_f, timestamp=when, engine_health=eh2)


def run_self_test() -> None:
    log = setup_logger("captain_self_test")
    ms = MarketState()

    now_real = datetime.now(IST)
    t0 = now_real.replace(hour=10, minute=0, second=0, microsecond=0)

    engine_health = {
        "data_engine": {"alive": True, "last_heartbeat": t0},
        "feature_engine": {"alive": True, "last_heartbeat": t0},
        "option_chain_poller": {"alive": True, "last_heartbeat": t0},
        "risk_engine": {"alive": True, "last_heartbeat": t0, "circuit_breaker": False},
        "compliance_engine": {"alive": True, "last_heartbeat": t0, "ok": True},
    }

    ms.update(
        spot=24500.0,
        timestamp=t0,
        feed_health="HEALTHY",
        data_quality_score=72.0,
        system_state="SAFE",
        engine_health=engine_health,
        candles=_build_minimal_candles_dict(),
        regime="RANGE",
        narrative_label="NEUTRAL",
        features={"mtf_alignment": {"weighted_mtf": 3.0}},
        daily_pnl=0.0,
        capital=50000.0,
        drawdown_pct=0.03,
        consecutive_losses=0,
        mode="LIVE",
        open_positions=[],
    )

    snap0 = ms.snapshot()
    _assert_no_forbidden_top_level_fields(snap0)
    log.info("Seed OK")

    captain = CaptainEngine(ms)

    # Strong wiring probe: Captain context must surface the full MTF payload.
    probe_features = {
        "1min": {"last_close": 100.0, "vwap": 99.0, "ema_9": 101.0, "ema_21": 100.0, "rsi": 60.0},
        "3min": {"last_close": 100.0, "vwap": 99.0, "ema_9": 101.0, "ema_21": 100.0, "rsi": 60.0},
        "5min": {"last_close": 100.0, "vwap": 99.0, "ema_9": 101.0, "ema_21": 100.0, "rsi": 60.0},
        "15min": {"last_close": 100.0, "vwap": 99.0, "ema_9": 101.0, "ema_21": 100.0, "rsi": 60.0},
    }
    probe_snap = {
        "regime": "RANGE",
        "narrative_label": "NEUTRAL",
        "session_phase": "GOLDEN_AM",
        "features": probe_features,
        "key_levels": {"pdh": 101.0},
        "smart_money": {"5min": {}, "15min": {}},
        "fundamental": {"vix_level": 15.0},
    }
    probe_ctx = captain._build_pipeline_context(probe_snap, {"data_quality_score": 72.0})
    assert isinstance(probe_ctx.get("mtf"), dict), "Captain context missing mtf payload"
    assert probe_ctx["mtf"].get("mtf_label") == "STRONG_BULLISH", probe_ctx["mtf"]
    assert probe_ctx.get("mtf_alignment_signal") == 1, probe_ctx
    assert probe_ctx.get("mtf_trap_zone") is False, probe_ctx
    assert probe_ctx.get("weighted_mtf", 0.0) > 0, probe_ctx
    assert "garch_high_vol" in probe_ctx and isinstance(probe_ctx.get("garch_status"), dict), probe_ctx

    brain_signals_weight_test = [
        {"brain": "institutional_brain", "direction": "SHORT", "confidence": 70},
        {"brain": "structural_brain", "direction": "LONG", "confidence": 85},
        {"brain": "tactical_brain", "direction": "LONG", "confidence": 80},
    ]
    consensus_dir, consensus_strength, suppressed = captain._brain_weighted_consensus(brain_signals_weight_test)
    assert consensus_dir == "LONG"
    log.info("Brain weighting PASS", consensus=consensus_dir, strength=consensus_strength, suppressed=suppressed)

    opp = {"direction": "LONG", "score": 80, "brain": "structural_brain", "strategy": "VWAP Pullback", "symbol": "NIFTY", "qty": 65}

    _freshen_market_state(ms, t0)
    res1 = captain.step(opportunities=[opp], brain_signals=brain_signals_weight_test, now=t0)
    log.info("SAFE probe attempt 1", state=res1.decision.system_state, approved=len(res1.approved), veto=len(res1.vetoed), probe_mode=res1.decision.probe_mode)

    stronger_brains = [
        {"brain": "institutional_brain", "direction": "LONG", "confidence": 80},
        {"brain": "structural_brain", "direction": "LONG", "confidence": 85},
    ]
    t1 = t0 + timedelta(minutes=15)
    _freshen_market_state(ms, t1)
    res1b = captain.step(opportunities=[opp], brain_signals=stronger_brains, now=t1)
    assert res1b.approved, "Expected probe approval with stronger consensus (and fresh timestamps)"
    assert res1b.decision.system_state == "SAFE" and res1b.decision.probe_mode is True
    log.info("SAFE probe approved PASS", approved=len(res1b.approved), size_hint=res1b.size_hint_multiplier)

    captain.notify_trade_outcome(pnl_rupees=120.0, timestamp=t1 + timedelta(minutes=1), was_probe=True, within_risk_limits=True)

    t2 = t1 + timedelta(minutes=2)
    _freshen_market_state(ms, t2)
    res2 = captain.step(opportunities=None, brain_signals=stronger_brains, now=t2)
    log.info("Probe recovery step", state=res2.decision.system_state, reason=res2.decision.reason)

    snapF = ms.snapshot()
    _assert_no_forbidden_top_level_fields(snapF)

    print("\nCAPTAIN SELF-TEST: PASS")
    log.info("Captain self-test PASSED")


if __name__ == "__main__":
    run_self_test()