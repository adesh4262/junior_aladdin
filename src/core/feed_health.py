"""
Junior Aladdin - Feed Health Monitor (FULL INSTITUTIONAL GRADE)
==============================================================
FILE: src/core/feed_health.py

Fix in this revision:
- HARD DOWN conditions bypass hysteresis (institutional safety):
  * broker latency > broker_down_ms => immediate DOWN
  * queue latency > queue_down_ms   => immediate DOWN
  * ws permanently dead (WS mode)   => immediate DOWN
  * lag_ms > eff_down_ms            => immediate DOWN

All existing logic remains intact; we only add a hard-down override layer.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any, List, Deque

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

IST = timezone(timedelta(hours=5, minutes=30))


# -----------------------------------------------------------------------------
# Helpers (defensive)
# -----------------------------------------------------------------------------
_EPS = 1e-9


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any, default: Optional[str] = None) -> Optional[str]:
    if v is None:
        return default
    try:
        return str(v)
    except Exception:
        return default


def _now_ist() -> datetime:
    return datetime.now(IST)


def _parse_hhmm(s: Any, default_h: int, default_m: int) -> Tuple[int, int]:
    try:
        parts = str(s).split(":")
        if len(parts) != 2:
            return default_h, default_m
        return int(parts[0]), int(parts[1])
    except Exception:
        return default_h, default_m


def _market_phase_at(dt: datetime) -> str:
    """
    Returns PRE_MARKET / MARKET / POST_MARKET by time boundaries.
    """
    try:
        open_str = Config.get("market", "market_open", default="09:15")
        close_str = Config.get("market", "market_close", default="15:30")
        oh, om = _parse_hhmm(open_str, 9, 15)
        ch, cm = _parse_hhmm(close_str, 15, 30)

        start = dt.replace(hour=oh, minute=om, second=0, microsecond=0)
        end = dt.replace(hour=ch, minute=cm, second=0, microsecond=0)

        if dt < start:
            return "PRE_MARKET"
        if dt > end:
            return "POST_MARKET"
        return "MARKET"
    except Exception:
        return "MARKET"


def _health_rank(h: str) -> int:
    """
    Lower is better.
    PRE_MARKET/POST_MARKET are handled separately.
    """
    order = {"HEALTHY": 0, "DELAYED": 1, "STALE": 2, "DOWN": 3}
    return order.get(h, 99)


# -----------------------------------------------------------------------------
# Configuration container
# -----------------------------------------------------------------------------
@dataclass
class FeedHealthConfig:
    delay_ms: int
    stale_ms: int
    down_ms: int

    worsen_confirmations: int

    # SAFE mode
    safe_quality_threshold: float
    safe_quality_consec: int
    safe_down_seconds: float
    safe_broker_latency_ms: float
    safe_broker_latency_consec: int

    # websocket metrics thresholds
    broker_down_ms: float
    queue_stale_ms: float
    queue_down_ms: float
    drop_warn_per_min: int

    # option chain thresholds
    option_fresh_sec: float
    option_ok_sec: float

    # REST expectations
    rest_expected_max_tps: float
    rest_expected_gap_ms: float

    # parse failure degrade
    parse_fail_rate_stale_pct: float


def _load_cfg() -> FeedHealthConfig:
    delay_ms = int(Config.get("data", "feed_delay_threshold_ms", default=1000))
    stale_ms = int(Config.get("data", "feed_stale_threshold_ms", default=3000))
    down_ms = int(Config.get("data", "feed_down_threshold_ms", default=5000))

    worsen_confirmations = int(Config.get("data", "feed_health_worsen_confirmations", default=3))
    worsen_confirmations = max(1, worsen_confirmations)

    safe_quality_threshold = float(Config.get("data", "feed_health_safe_quality_threshold", default=40.0))
    safe_quality_consec = int(Config.get("data", "feed_health_safe_quality_consec", default=5))
    safe_quality_consec = max(1, safe_quality_consec)

    safe_down_seconds = float(Config.get("data", "feed_health_safe_down_seconds", default=10.0))
    safe_down_seconds = max(1.0, safe_down_seconds)

    safe_broker_latency_ms = float(Config.get("data", "feed_health_safe_broker_latency_ms", default=10000.0))
    safe_broker_latency_consec = int(Config.get("data", "feed_health_safe_broker_latency_consec", default=3))
    safe_broker_latency_consec = max(1, safe_broker_latency_consec)

    broker_down_ms = float(Config.get("data", "feed_health_broker_down_ms", default=5000.0))
    queue_stale_ms = float(Config.get("data", "feed_health_queue_stale_ms", default=1500.0))
    queue_down_ms = float(Config.get("data", "feed_health_queue_down_ms", default=5000.0))
    drop_warn_per_min = int(Config.get("data", "feed_health_drop_warn_per_min", default=500))

    option_fresh_sec = float(Config.get("data", "feed_health_option_fresh_sec", default=90.0))
    option_ok_sec = float(Config.get("data", "feed_health_option_ok_sec", default=180.0))

    rest_expected_max_tps = float(Config.get("data", "feed_health_rest_expected_max_tps", default=0.5))
    rest_expected_gap_ms = float(Config.get("data", "feed_health_rest_expected_gap_ms", default=2500.0))

    parse_fail_rate_stale_pct = float(Config.get("data", "feed_health_parse_fail_rate_stale_pct", default=50.0))

    return FeedHealthConfig(
        delay_ms=delay_ms,
        stale_ms=stale_ms,
        down_ms=down_ms,
        worsen_confirmations=worsen_confirmations,
        safe_quality_threshold=safe_quality_threshold,
        safe_quality_consec=safe_quality_consec,
        safe_down_seconds=safe_down_seconds,
        safe_broker_latency_ms=safe_broker_latency_ms,
        safe_broker_latency_consec=safe_broker_latency_consec,
        broker_down_ms=broker_down_ms,
        queue_stale_ms=queue_stale_ms,
        queue_down_ms=queue_down_ms,
        drop_warn_per_min=drop_warn_per_min,
        option_fresh_sec=option_fresh_sec,
        option_ok_sec=option_ok_sec,
        rest_expected_max_tps=rest_expected_max_tps,
        rest_expected_gap_ms=rest_expected_gap_ms,
        parse_fail_rate_stale_pct=parse_fail_rate_stale_pct,
    )


# -----------------------------------------------------------------------------
# Feed Health Monitor
# -----------------------------------------------------------------------------
class FeedHealthMonitor:
    """
    Monitor feed health and compute a practical data-quality score.
    """

    def __init__(self):
        self._logger = setup_logger("feed_health")
        self._cfg = _load_cfg()

        self._feed_mode: str = "WEBSOCKET"  # or "REST"
        self._rest_poll_interval_sec: Optional[float] = None

        self.last_tick_time: Optional[datetime] = None
        self.last_option_update: Optional[datetime] = None

        self.current_health: str = "DOWN"
        self.data_quality_score: float = 0.0

        self._ws_metrics: Dict[str, Any] = {}
        self._ws_metrics_last_update: Optional[datetime] = None

        self._tick_times: deque = deque(maxlen=500)
        self._health_history: deque = deque(maxlen=300)

        self._recent_spikes: int = 0
        self._last_spike_reset_time: Optional[datetime] = None

        self._has_received_tick: bool = False
        self._has_received_option_update: bool = False
        self._last_state_change_time: Optional[datetime] = None

        self._pending_worse_state: Optional[str] = None
        self._pending_worse_count: int = 0

        self.should_enter_safe_mode: bool = False
        self.safe_mode_reasons: List[str] = []
        self._low_quality_streak: int = 0
        self._high_broker_latency_streak: int = 0
        self._down_since: Optional[datetime] = None

        self._last_dropped_ticks_seen: Optional[int] = None
        self._drop_increments_per_check: deque = deque(maxlen=60)
        self._drop_time_window: Deque[Tuple[datetime, int]] = deque(maxlen=600)

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------
    def set_feed_mode(self, mode: str, rest_poll_interval_sec: Optional[float] = None):
        m = _safe_str(mode, default="WEBSOCKET")
        if m is None:
            m = "WEBSOCKET"
        m = m.strip().upper()
        if m not in ("WEBSOCKET", "REST"):
            self._logger.warning("Invalid feed mode; keeping current", requested=mode, current=self._feed_mode)
            return

        with self._lock:
            self._feed_mode = m
            self._rest_poll_interval_sec = rest_poll_interval_sec

        self._logger.info("Feed mode set", mode=m, rest_poll_interval_sec=rest_poll_interval_sec)

    def get_feed_mode(self) -> str:
        with self._lock:
            return self._feed_mode

    # ------------------------------------------------------------------
    # Feed event updates
    # ------------------------------------------------------------------
    def on_tick(self, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = _now_ist()
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)

        with self._lock:
            self.last_tick_time = timestamp
            self._tick_times.append(timestamp)
            self._has_received_tick = True
            if self._last_spike_reset_time is None:
                self._last_spike_reset_time = timestamp

    def on_option_update(self, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = _now_ist()
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)

        with self._lock:
            self.last_option_update = timestamp
            self._has_received_option_update = True

    def on_spike(self):
        with self._lock:
            self._recent_spikes += 1

    def reset_spikes(self):
        with self._lock:
            self._recent_spikes = 0
            self._last_spike_reset_time = _now_ist()

    # ------------------------------------------------------------------
    # Main health evaluation
    # ------------------------------------------------------------------
    def check(
        self,
        now: Optional[datetime] = None,
        ws_status: Optional[Dict[str, Any]] = None,
        vix_level: Optional[float] = None,
    ) -> Tuple[str, float]:
        try:
            if now is None:
                now = _now_ist()
            elif now.tzinfo is None:
                now = now.replace(tzinfo=IST)

            # Market hours gating
            phase = _market_phase_at(now)
            if phase != "MARKET":
                with self._lock:
                    self.current_health = phase
                    self.data_quality_score = 100.0
                    self.should_enter_safe_mode = False
                    self.safe_mode_reasons = []
                    self._pending_worse_state = None
                    self._pending_worse_count = 0
                return phase, 100.0

            ws_metrics = self._ingest_ws_status(ws_status, now)
            self._auto_reset_spikes(now)

            eff_delay_ms, eff_stale_ms, eff_down_ms = self._effective_thresholds(vix_level=vix_level)

            tick_gap_ms = self._compute_tick_gap_ms(now)
            option_age_sec = self._compute_option_age_sec(now)
            tps = self.get_tps()

            prelim_health, reasons = self._classify_health_institutional(
                now=now,
                tick_gap_ms=tick_gap_ms,
                eff_delay_ms=eff_delay_ms,
                eff_stale_ms=eff_stale_ms,
                eff_down_ms=eff_down_ms,
                ws_metrics=ws_metrics,
            )

            # NEW: Hard-down bypass hysteresis (institutional safety)
            if self._is_hard_down(prelim_health, reasons):
                final_health = "DOWN"
                # Reset hysteresis pending to avoid sticking
                with self._lock:
                    self._pending_worse_state = None
                    self._pending_worse_count = 0
            else:
                final_health = self._apply_hysteresis(prelim_health, now)

            score = self._compute_quality_score_hardened(
                health=final_health,
                option_age_sec=option_age_sec,
                tps=tps,
                ws_metrics=ws_metrics,
                health_reasons=reasons,
            )

            self._update_safe_mode_recommendation(
                now=now,
                health=final_health,
                score=score,
                ws_metrics=ws_metrics,
            )

            with self._lock:
                prev_health = self.current_health
                self.current_health = final_health
                self.data_quality_score = score

            if final_health != prev_health:
                with self._lock:
                    self._last_state_change_time = now
                log_func = self._logger.info if final_health == "HEALTHY" else self._logger.warning
                log_func(
                    "Feed health changed",
                    prev=prev_health,
                    new=final_health,
                    tick_gap_ms=round(tick_gap_ms, 0),
                    quality_score=score,
                    tps=round(tps, 2),
                    option_age_sec=round(option_age_sec, 1),
                    recent_spikes=self._recent_spikes,
                    feed_mode=self.get_feed_mode(),
                    broker_latency_ms=ws_metrics.get("avg_broker_latency_ms"),
                    queue_latency_ms=ws_metrics.get("avg_queue_latency_ms"),
                    dropped_ticks=ws_metrics.get("dropped_ticks"),
                    queue_size=ws_metrics.get("queue_size"),
                    drop_rate_per_min=ws_metrics.get("drop_rate_per_min"),
                    reasons=reasons[:4],
                    safe_mode=self.should_enter_safe_mode,
                    safe_reasons=self.safe_mode_reasons[:3],
                )

            with self._lock:
                self._health_history.append(
                    {
                        "time": now,
                        "health": final_health,
                        "score": score,
                        "tick_gap_ms": round(tick_gap_ms, 0),
                        "tps": round(tps, 2),
                        "option_age_sec": round(option_age_sec, 1),
                        "spikes": self._recent_spikes,
                        "mode": self._feed_mode,
                        "ws": ws_metrics,
                        "safe": self.should_enter_safe_mode,
                    }
                )

            return final_health, score

        except Exception as e:
            self._logger.error("FeedHealthMonitor.check exception; returning DOWN", error=str(e))
            with self._lock:
                self.current_health = "DOWN"
                self.data_quality_score = 0.0
                self.should_enter_safe_mode = True
                self.safe_mode_reasons = ["check_exception"]
            return "DOWN", 0.0

    def _is_hard_down(self, prelim_health: str, reasons: List[str]) -> bool:
        """
        HARD DOWN conditions must bypass hysteresis.
        """
        if prelim_health != "DOWN":
            return False
        hard_markers = (
            "broker_latency>",
            "queue_latency>",
            "ws_feed_permanently_dead",
            "lag>",
        )
        for r in reasons:
            if any(r.startswith(m) for m in hard_markers):
                return True
        return False

    # ------------------------------------------------------------------
    # WS status integration
    # ------------------------------------------------------------------
    def _ingest_ws_status(self, ws_status: Optional[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
        if not isinstance(ws_status, dict):
            return {}

        avg_broker = _safe_float(ws_status.get("avg_broker_latency_ms"), default=None)
        avg_queue = _safe_float(ws_status.get("avg_queue_latency_ms"), default=None)
        queue_size = _safe_int(ws_status.get("queue_size"), default=None)
        dropped_ticks = _safe_int(ws_status.get("dropped_ticks"), default=None)
        feed_dead = bool(ws_status.get("feed_permanently_dead", False))
        ws_connected = bool(ws_status.get("is_connected", False))
        parse_fail_rate = _safe_float(ws_status.get("parse_fail_rate_pct_window"), default=None)

        drop_rate_per_min = None
        if dropped_ticks is not None:
            with self._lock:
                if self._last_dropped_ticks_seen is None:
                    self._last_dropped_ticks_seen = dropped_ticks
                    inc = 0
                else:
                    inc = max(0, dropped_ticks - self._last_dropped_ticks_seen)
                    self._last_dropped_ticks_seen = dropped_ticks

                self._drop_increments_per_check.append(inc)
                self._drop_time_window.append((now, inc))

                cutoff = now - timedelta(seconds=60)
                while self._drop_time_window and self._drop_time_window[0][0] < cutoff:
                    self._drop_time_window.popleft()
                drop_rate_per_min = sum(x[1] for x in self._drop_time_window)

        metrics = {
            "avg_broker_latency_ms": avg_broker,
            "avg_queue_latency_ms": avg_queue,
            "queue_size": queue_size,
            "dropped_ticks": dropped_ticks,
            "drop_rate_per_min": drop_rate_per_min,
            "feed_permanently_dead": feed_dead,
            "ws_connected": ws_connected,
            "parse_fail_rate_pct": parse_fail_rate,
        }

        with self._lock:
            self._ws_metrics = metrics
            self._ws_metrics_last_update = now

        return metrics

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------
    def _effective_thresholds(self, vix_level: Optional[float]) -> Tuple[float, float, float]:
        v = _safe_float(vix_level, default=None)
        if v is None:
            scale = 1.0
        else:
            scale = 1.0 + max(0.0, (v - 15.0) / 100.0)
            scale = _clamp(scale, 1.0, 2.5)

        delay = float(self._cfg.delay_ms) * scale
        stale = float(self._cfg.stale_ms) * scale
        down = float(self._cfg.down_ms) * scale

        if self.get_feed_mode() == "REST":
            rest_gap = float(self._cfg.rest_expected_gap_ms)
            delay = max(delay, rest_gap * 0.8)
            stale = max(stale, rest_gap * 1.3)
            down = max(down, rest_gap * 2.0)

        return delay, stale, down

    def _compute_tick_gap_ms(self, now: datetime) -> float:
        with self._lock:
            lt = self.last_tick_time
        if lt is None:
            return 999999.0
        return max(0.0, (now - lt).total_seconds() * 1000.0)

    def _compute_option_age_sec(self, now: datetime) -> float:
        with self._lock:
            lo = self.last_option_update
        if lo is None:
            return 999.0
        return max(0.0, (now - lo).total_seconds())

    def _classify_health_institutional(
        self,
        now: datetime,
        tick_gap_ms: float,
        eff_delay_ms: float,
        eff_stale_ms: float,
        eff_down_ms: float,
        ws_metrics: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []

        with self._lock:
            has_tick = self._has_received_tick

        if not has_tick and (not ws_metrics):
            return "DOWN", ["no_tick_received"]

        mode = self.get_feed_mode()

        if mode == "WEBSOCKET" and bool(ws_metrics.get("feed_permanently_dead", False)):
            return "DOWN", ["ws_feed_permanently_dead"]

        broker_lat_ms = _safe_float(ws_metrics.get("avg_broker_latency_ms"), default=None)
        queue_lat_ms = _safe_float(ws_metrics.get("avg_queue_latency_ms"), default=None)
        parse_fail_rate = _safe_float(ws_metrics.get("parse_fail_rate_pct"), default=None)

        # broker latency hard DOWN
        if broker_lat_ms is not None and broker_lat_ms > float(self._cfg.broker_down_ms):
            return "DOWN", [f"broker_latency>{self._cfg.broker_down_ms}ms"]

        # queue backpressure hard down
        if queue_lat_ms is not None and queue_lat_ms > float(self._cfg.queue_down_ms):
            return "DOWN", [f"queue_latency>{self._cfg.queue_down_ms}ms"]

        if queue_lat_ms is not None and queue_lat_ms > float(self._cfg.queue_stale_ms):
            reasons.append("queue_backpressure")

        if parse_fail_rate is not None and parse_fail_rate >= float(self._cfg.parse_fail_rate_stale_pct):
            reasons.append("high_parse_fail_rate")

        drop_rate = _safe_int(ws_metrics.get("drop_rate_per_min"), default=None)
        if drop_rate is not None and drop_rate > int(self._cfg.drop_warn_per_min):
            reasons.append("high_drop_rate_per_min")

        lag_ms = broker_lat_ms if broker_lat_ms is not None else tick_gap_ms
        reasons.append("using_broker_latency" if broker_lat_ms is not None else "using_tick_gap")

        if not has_tick and mode == "REST":
            return "STALE", ["rest_no_tick_yet"]

        if lag_ms > eff_down_ms:
            return "DOWN", reasons + [f"lag>{eff_down_ms:.0f}ms"]
        if lag_ms > eff_stale_ms:
            return "STALE", reasons + [f"lag>{eff_stale_ms:.0f}ms"]
        if lag_ms > eff_delay_ms:
            return "DELAYED", reasons + [f"lag>{eff_delay_ms:.0f}ms"]

        if "queue_backpressure" in reasons or "high_parse_fail_rate" in reasons:
            return "STALE", reasons
        if "high_drop_rate_per_min" in reasons:
            return "DELAYED", reasons

        return "HEALTHY", reasons

    def _apply_hysteresis(self, prelim_health: str, now: datetime) -> str:
        with self._lock:
            current = self.current_health

        if current not in ("HEALTHY", "DELAYED", "STALE", "DOWN"):
            with self._lock:
                self._pending_worse_state = None
                self._pending_worse_count = 0
            return prelim_health

        # improvements immediate
        if _health_rank(prelim_health) < _health_rank(current):
            with self._lock:
                self._pending_worse_state = None
                self._pending_worse_count = 0
            return prelim_health

        if prelim_health == current:
            with self._lock:
                self._pending_worse_state = None
                self._pending_worse_count = 0
            return current

        # worsening needs confirmations
        with self._lock:
            if self._pending_worse_state != prelim_health:
                self._pending_worse_state = prelim_health
                self._pending_worse_count = 1
            else:
                self._pending_worse_count += 1

            if self._pending_worse_count >= self._cfg.worsen_confirmations:
                self._pending_worse_state = None
                self._pending_worse_count = 0
                return prelim_health

        return current

    def _compute_quality_score_hardened(
        self,
        health: str,
        option_age_sec: float,
        tps: float,
        ws_metrics: Dict[str, Any],
        health_reasons: List[str],
    ) -> float:
        score = 0.0

        if health == "HEALTHY":
            score += 40
        elif health == "DELAYED":
            score += 20
        elif health == "STALE":
            score += 5

        with self._lock:
            has_opt = self._has_received_option_update
            spikes = self._recent_spikes
            has_tick = self._has_received_tick

        if has_opt:
            if option_age_sec < float(self._cfg.option_fresh_sec):
                score += 20
            elif option_age_sec < float(self._cfg.option_ok_sec):
                score += 10

        if has_tick:
            if spikes == 0:
                score += 20
            elif spikes <= 2:
                score += 10

        if self.get_feed_mode() == "REST":
            score += 10
        else:
            if tps > 1.0:
                score += 10
            elif tps > 0.5:
                score += 5

        broker_lat = _safe_float(ws_metrics.get("avg_broker_latency_ms"), default=None)
        queue_lat = _safe_float(ws_metrics.get("avg_queue_latency_ms"), default=None)
        drop_rate = _safe_int(ws_metrics.get("drop_rate_per_min"), default=None)
        qsize = _safe_int(ws_metrics.get("queue_size"), default=None)
        parse_fail_rate = _safe_float(ws_metrics.get("parse_fail_rate_pct"), default=None)

        if broker_lat is not None:
            if broker_lat > 3000:
                score -= 10
            elif broker_lat > 1500:
                score -= 5

        if queue_lat is not None:
            if queue_lat > 2000:
                score -= 8
            elif queue_lat > 1000:
                score -= 4

        if parse_fail_rate is not None and parse_fail_rate >= float(self._cfg.parse_fail_rate_stale_pct):
            score -= 8

        if drop_rate is not None:
            if drop_rate > 800:
                score -= 10
            elif drop_rate > 300:
                score -= 6
            elif drop_rate > 100:
                score -= 3

        if qsize is not None:
            qmax = int(Config.get("data", "websocket_tick_queue_max", default=5000))
            if qmax > 0 and qsize > 0.8 * qmax:
                score -= 8

        return round(float(_clamp(score, 0.0, 100.0)), 2)

    def _update_safe_mode_recommendation(self, now: datetime, health: str, score: float, ws_metrics: Dict[str, Any]):
        reasons: List[str] = []
        broker_lat = _safe_float(ws_metrics.get("avg_broker_latency_ms"), default=None)

        if health == "DOWN":
            with self._lock:
                if self._down_since is None:
                    self._down_since = now
                down_since = self._down_since
            down_dur = (now - down_since).total_seconds() if down_since else 0.0
            if down_dur >= float(self._cfg.safe_down_seconds):
                reasons.append(f"DOWN>{self._cfg.safe_down_seconds}s")
        else:
            with self._lock:
                self._down_since = None

        if score < float(self._cfg.safe_quality_threshold):
            with self._lock:
                self._low_quality_streak += 1
                streak = self._low_quality_streak
            if streak >= int(self._cfg.safe_quality_consec):
                reasons.append(f"quality<{self._cfg.safe_quality_threshold} for {streak} checks")
        else:
            with self._lock:
                self._low_quality_streak = 0

        if broker_lat is not None and broker_lat > float(self._cfg.safe_broker_latency_ms):
            with self._lock:
                self._high_broker_latency_streak += 1
                streak2 = self._high_broker_latency_streak
            if streak2 >= int(self._cfg.safe_broker_latency_consec):
                reasons.append(f"broker_latency>{self._cfg.safe_broker_latency_ms}ms for {streak2} checks")
        else:
            with self._lock:
                self._high_broker_latency_streak = 0

        if self.get_feed_mode() == "REST" and health in ("STALE", "DOWN"):
            reasons.append("REST_mode_degraded")

        with self._lock:
            self.safe_mode_reasons = reasons
            self.should_enter_safe_mode = bool(reasons)

    def _auto_reset_spikes(self, now: datetime):
        with self._lock:
            if self._last_spike_reset_time is None:
                self._last_spike_reset_time = now
                return
            elapsed = (now - self._last_spike_reset_time).total_seconds()
            if elapsed >= 300:
                self._recent_spikes = 0
                self._last_spike_reset_time = now

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_tps(self, window_sec: float = 5.0) -> float:
        with self._lock:
            times = list(self._tick_times)
        if len(times) < 2:
            return 0.0
        end = times[-1]
        start_cut = end - timedelta(seconds=float(max(1.0, window_sec)))
        count = 0
        for t in reversed(times):
            if t >= start_cut:
                count += 1
            else:
                break
        span = (end - start_cut).total_seconds()
        if span <= 0:
            return 0.0
        return round(count / span, 2)

    def reset(self):
        with self._lock:
            self.last_tick_time = None
            self.last_option_update = None
            self.current_health = "DOWN"
            self.data_quality_score = 0.0
            self._tick_times.clear()
            self._health_history.clear()
            self._recent_spikes = 0
            self._has_received_tick = False
            self._has_received_option_update = False
            self._last_spike_reset_time = None
            self._last_state_change_time = None
            self._pending_worse_state = None
            self._pending_worse_count = 0
            self.should_enter_safe_mode = False
            self.safe_mode_reasons = []
            self._low_quality_streak = 0
            self._high_broker_latency_streak = 0
            self._down_since = None
            self._ws_metrics = {}
            self._ws_metrics_last_update = None
            self._last_dropped_ticks_seen = None
            self._drop_increments_per_check.clear()
            self._drop_time_window.clear()

        self._logger.info("Feed Health Monitor reset")


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Feed Health Monitor Test (Institutional Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    m = FeedHealthMonitor()

    dt_post = datetime(2026, 4, 1, 16, 0, 0, tzinfo=IST)
    h, s = m.check(now=dt_post)
    if h == "POST_MARKET" and s == 100.0:
        print(" ✅ POST_MARKET gating ok")
        passed += 1
    else:
        print(f" ❌ POST_MARKET gating failed: {h}, {s}")
        failed += 1

    base = datetime(2026, 4, 1, 10, 0, 0, tzinfo=IST)
    m.reset()
    m.on_tick(base)
    h1, _ = m.check(now=base)

    delayed = base + timedelta(milliseconds=m._cfg.delay_ms + 50)
    h2, _ = m.check(now=delayed)
    if h2 == "HEALTHY":
        print(" ✅ Hysteresis prevents single worsening")
        passed += 1
    else:
        print(f" ❌ Hysteresis failed: {h1}->{h2}")
        failed += 1

    # This must be immediate DOWN now (bypass hysteresis)
    ws_status = {"avg_broker_latency_ms": 6000, "is_connected": True, "feed_permanently_dead": False}
    h3, _ = m.check(now=base + timedelta(seconds=1), ws_status=ws_status)
    if h3 == "DOWN":
        print(" ✅ Broker latency forces immediate DOWN (bypass hysteresis)")
        passed += 1
    else:
        print(f" ❌ Expected DOWN, got {h3}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()