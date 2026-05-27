# FILE: src/core/websocket_manager.py

"""
Junior Aladdin - WebSocket Manager (FULL INSTITUTIONAL GRADE)
=============================================================
UPGRADE: Connection Manager -> Tick Ingress Pipeline

PURPOSE:
    Manage Angel One SmartAPI WebSocket connection with institutional-grade
    safety, observability, and backpressure-aware tick ingestion.

PRESERVED SAFETY FEATURES:
    - Manual shutdown hard-cuts pipeline & disables reconnects
    - Generation gating: stale callbacks ignored
    - Epoch gating: stale socket callbacks ignored within same generation
    - Exponential backoff reconnects with bounded delay
    - Permanent failure semantics after max retries
    - Watchdog stale detection (zombie connection recovery)
    - Strict token refresh validation
    - Parse failure circuit breaker
    - Instrument-aware price scaling map (token->scaling_factor)
    - Dynamic re-subscription API
    - Feed health callbacks

PHASE 5 PIPELINE HARDENING (NEW):
    - Bounded tick queue (producer-consumer decoupling)
    - Backpressure: drop oldest when queue full, track dropped_ticks
    - Consumer thread with intelligent batching
    - Per-instrument rate limiting / admission control
    - Latency telemetry:
        broker_latency_ms (received_at - exchange_timestamp)
        queue_latency_ms (dequeue_time - enqueue_time)
        processing_latency_ms (callback duration)
    - Subscription healing (missing tokens -> resubscribe / reconnect)

IMPORTANT:
    WebSocket thread MUST NOT call _on_tick directly.
    It only parses and enqueues ticks.

PATCH (OI SUPPORT):
    SmartAPI WebSocket v2 can emit Open Interest updates for option instruments.
    These fields are not guaranteed in every tick and may arrive with different key names.
    We now extract OI-related fields (oi/open_interest/openInterest/prev_oi/oi_change)
    and include them in normalized tick dicts when present.

"""

from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional, Any, Tuple, Deque
from collections import deque

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

IST = timezone(timedelta(hours=5, minutes=30))
_EPS = 1e-9


@dataclass(frozen=True)
class TickEnvelope:
    """
    Envelope stored in tick queue.
    - tick: normalized tick dict
    - enqueued_at: local time when added to queue (IST)
    """
    tick: Dict[str, Any]
    enqueued_at: datetime


class WebSocketManager:
    """
    Institutional-grade WebSocket + Tick Ingress Pipeline for SmartWebSocketV2.
    """

    def __init__(
        self,
        auth_manager,
        instrument_mapper,
        on_tick_callback: Callable[[Dict[str, Any]], None],
        on_connect_callback: Optional[Callable[[], None]] = None,
        on_disconnect_callback: Optional[Callable[[], None]] = None,
        # Batch callback (optional): if provided, consumer will call this for batches
        on_batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        # Feed health callbacks
        on_feed_stale: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_feed_restored: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_feed_permanent_failure: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._logger = setup_logger("websocket_manager")

        self._auth = auth_manager
        self._mapper = instrument_mapper

        self._on_tick = on_tick_callback
        self._on_batch = on_batch_callback
        self._on_connect = on_connect_callback
        self._on_disconnect = on_disconnect_callback

        self._on_feed_stale = on_feed_stale
        self._on_feed_restored = on_feed_restored
        self._on_feed_permanent_failure = on_feed_permanent_failure

        # WS lifecycle
        self._sws = None
        self._ws_thread: Optional[threading.Thread] = None

        # Producer/consumer pipeline
        self._tick_queue_max = int(Config.get("data", "websocket_tick_queue_max", default=5000))
        self._tick_queue: "queue.Queue[TickEnvelope]" = queue.Queue(maxsize=max(1000, self._tick_queue_max))
        self._consumer_thread: Optional[threading.Thread] = None
        self._consumer_stop = threading.Event()

        # Batching config
        self._batch_enabled = bool(Config.get("data", "websocket_batch_enabled", default=True))
        self._max_batch_size = int(Config.get("data", "websocket_max_batch_size", default=50))
        self._max_batch_interval_ms = int(Config.get("data", "websocket_max_batch_interval_ms", default=100))
        self._max_batch_size = max(1, self._max_batch_size)
        self._max_batch_interval_ms = max(10, self._max_batch_interval_ms)

        # Rate limiting config (admission control)
        self._rate_limit_enabled = bool(Config.get("data", "websocket_rate_limit_enabled", default=True))
        self._default_min_interval_ms = float(Config.get("data", "websocket_rate_limit_default_ms", default=50.0))  # 20 ticks/sec per token
        self._default_min_interval_ms = max(0.0, self._default_min_interval_ms)
        self._min_interval_by_token: Dict[str, float] = {}  # optional overrides
        self._last_processed_monotonic_by_token: Dict[str, float] = {}
        self._rate_limited_ticks: int = 0

        # Latest cache (even for rate-limited ticks)
        self._latest_tick_by_token: Dict[str, Dict[str, Any]] = {}

        # Backpressure metrics
        self._dropped_ticks: int = 0
        self._last_drop_log: Optional[datetime] = None
        self._drop_log_suppress_sec = float(Config.get("data", "websocket_drop_log_suppress_sec", default=10.0))

        # Threading / gating
        # Re-entrant lock avoids startup deadlocks when helper methods re-enter lock scope.
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._manual_shutdown = False
        self._shutdown_complete = True

        # Reconnect controls
        self._max_retries = int(Config.get("data", "websocket_max_retries", default=5))
        self._reconnect_delay = float(Config.get("data", "websocket_reconnect_delay_sec", default=2))
        self._max_backoff = float(Config.get("data", "websocket_max_backoff_sec", default=60))

        # Heartbeat watchdog
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self._watchdog_interval_sec = float(Config.get("data", "websocket_watchdog_interval_sec", default=2))
        self._heartbeat_timeout_sec = float(Config.get("data", "websocket_heartbeat_timeout_sec", default=5))
        self._subscription_no_tick_warn_sec = float(Config.get("data", "websocket_subscription_no_tick_warn_sec", default=30))

        # --- PATCH: connection-established guard for watchdog heartbeat stale detection ---
        self._connection_established_time: Optional[float] = None

        # Subscription healing
        self._healing_check_interval_sec = float(Config.get("data", "websocket_healing_check_interval_sec", default=60))
        self._resubscribe_threshold_sec = float(Config.get("data", "websocket_resubscribe_threshold_sec", default=90))
        self._resubscribe_min_gap_sec = float(Config.get("data", "websocket_resubscribe_min_gap_sec", default=60))
        self._last_resubscribe_attempt: Dict[str, datetime] = {}

        # Public state
        self.is_connected: bool = False
        self.reconnect_count: int = 0
        self.total_ticks_received: int = 0
        self.last_tick_time: Optional[datetime] = None
        self.last_connect_time: Optional[datetime] = None
        self.last_error: str = ""

        # Permanent failure
        self._feed_permanently_dead: bool = False
        self._permanent_failure_reason: str = ""
        self._permanent_failure_event = threading.Event()

        # Subscription tracking
        self._subscribed_tokens: List[Dict] = []
        self._subscribed_token_set: set[str] = set()
        self._subscription_requested_at: Optional[datetime] = None
        self._seen_tokens: set[str] = set()
        self._last_seen_by_token: Dict[str, datetime] = {}
        self._subscription_warned_missing_once: bool = False
        self._token_exchange_type: Dict[str, int] = {}  # token -> exchangeType 1/2

        # Price scaling
        self._token_scaling: Dict[str, float] = {}

        # Parse failure CB
        self._parse_outcomes: Deque[bool] = deque(maxlen=60)
        self._parse_total_count: int = 0
        self._parse_failure_count: int = 0
        self._parse_error_last_log: Dict[str, datetime] = {}
        self._parse_error_suppress_sec = float(Config.get("data", "websocket_parse_error_suppress_sec", default=60))

        # Tick rate + latencies (rolling stats)
        self._tick_times: Deque[datetime] = deque(maxlen=200)
        self._lat_broker_ms: Deque[float] = deque(maxlen=100)
        self._lat_queue_ms: Deque[float] = deque(maxlen=100)
        self._lat_processing_ms: Deque[float] = deque(maxlen=100)

        # Feed stale state
        self._feed_stale: bool = False
        self._feed_stale_since: Optional[datetime] = None
        self._last_stale_trigger_gen_epoch: Tuple[int, int] = (-1, -1)

        # Reconnect re-entry protection
        self._reconnect_lock = threading.Lock()

        # Generation + epoch gating
        self._generation: int = 0
        self._active_generation: int = 0
        self._epoch: int = 0
        self._active_epoch: int = 0

        # Healing scheduler
        self._last_healing_check: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def connect(
        self,
        spot_price: float = 24500.0,
        subscription_list: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Start websocket in background thread.
        Producer-consumer threads are started here.
        """
        with self._lock:
            self._feed_permanently_dead = False
            self._permanent_failure_reason = ""
            self._permanent_failure_event.clear()

            if self._ws_thread and self._ws_thread.is_alive():
                self._logger.warning("WebSocket thread already running")
                return True

            self._stop_event.clear()
            self._consumer_stop.clear()
            self._manual_shutdown = False
            self._shutdown_complete = False

            # Reset metrics/state for this run
            self.reconnect_count = 0
            self.last_error = ""
            self.total_ticks_received = 0
            self.last_tick_time = None
            self._connection_established_time = None  # PATCH: reset guard timestamp
            self._tick_times.clear()

            self._parse_outcomes.clear()
            self._parse_total_count = 0
            self._parse_failure_count = 0

            self._lat_broker_ms.clear()
            self._lat_queue_ms.clear()
            self._lat_processing_ms.clear()

            self._dropped_ticks = 0
            self._rate_limited_ticks = 0
            self._latest_tick_by_token.clear()
            self._last_processed_monotonic_by_token.clear()

            self._feed_stale = False
            self._feed_stale_since = None
            self._subscription_warned_missing_once = False
            self._seen_tokens.clear()
            self._last_seen_by_token.clear()
            self._last_resubscribe_attempt.clear()

            self._drain_tick_queue_locked()

            # Pre-check credentials (no broker import)
            if not self._credentials_present():
                self._logger.error("Missing websocket credentials (pre-check). connect() aborted.")
                self._shutdown_complete = True
                return False

            # Compute subscriptions
            if isinstance(subscription_list, list):
                self._subscribed_tokens = [
                    it for it in subscription_list
                    if isinstance(it, dict) and it.get("token") is not None
                ]
            else:
                self._subscribed_tokens = self._safe_get_subscription_list(spot_price)
            self._subscribed_token_set = set(str(x.get("token", "")) for x in self._subscribed_tokens if isinstance(x, dict))
            self._token_scaling = self._build_token_scaling_map(self._subscribed_tokens)
            self._token_exchange_type = self._build_token_exchange_type_map(self._subscribed_tokens)

            # Generation & epoch gating
            self._generation += 1
            generation = self._generation
            self._active_generation = generation

            self._epoch += 1
            epoch = self._epoch
            self._active_epoch = epoch

            # Start consumer & watchdog
            self._start_consumer_if_needed()
            self._start_watchdog_if_needed()

            self._ws_thread = threading.Thread(
                target=self._run_websocket,
                args=(generation, epoch),
                daemon=True,
                name="WebSocket-Thread",
            )
            self._ws_thread.start()

            self._logger.info(
                "WebSocket connection thread started",
                tokens_to_subscribe=len(self._subscribed_tokens),
                queue_max=self._tick_queue_max,
                generation=generation,
                epoch=epoch,
            )
            return True

    def disconnect(self):
        """
        Manual stop; deterministic cut-off.
        """
        with self._lock:
            self._logger.info("Disconnecting WebSocket", total_ticks=self.total_ticks_received)

            self._manual_shutdown = True
            self._stop_event.set()
            self._consumer_stop.set()
            self.is_connected = False
            self._connection_established_time = None  # PATCH: clear guard timestamp
            self.reconnect_count = 0

            self._active_generation = -1
            self._active_epoch = -1

            ws_ref = self._sws
            ws_thread_ref = self._ws_thread
            consumer_ref = self._consumer_thread

            self._watchdog_stop.set()
            watchdog_ref = self._watchdog_thread

        if ws_ref is not None:
            try:
                ws_ref.close_connection()
            except Exception as e:
                self._logger.debug("WebSocket close warning", error=str(e))

        if ws_thread_ref and ws_thread_ref.is_alive():
            ws_thread_ref.join(timeout=5)

        if consumer_ref and consumer_ref.is_alive():
            consumer_ref.join(timeout=5)

        if watchdog_ref and watchdog_ref.is_alive():
            watchdog_ref.join(timeout=3)

        with self._lock:
            self._sws = None
            ws_alive = ws_thread_ref.is_alive() if ws_thread_ref else False
            consumer_alive = consumer_ref.is_alive() if consumer_ref else False
            w_alive = watchdog_ref.is_alive() if watchdog_ref else False
            self._shutdown_complete = (not ws_alive) and (not consumer_alive) and (not w_alive)
            self.is_connected = False
            self._drain_tick_queue_locked()

        self._logger.info(
            "WebSocket disconnect sequence finished",
            ws_thread_alive=ws_alive,
            consumer_thread_alive=consumer_alive,
            watchdog_alive=w_alive,
            shutdown_complete=self._shutdown_complete,
        )

    def update_subscriptions(self, spot_price: float) -> bool:
        """
        Dynamic subscription update.
        """
        with self._lock:
            if self._manual_shutdown or self._stop_event.is_set():
                self._logger.warning("update_subscriptions ignored due to shutdown")
                return False
            if self._feed_permanently_dead:
                self._logger.warning("update_subscriptions ignored due to permanent failure")
                return False

            new_list = self._safe_get_subscription_list(spot_price)
            new_set = set(str(x.get("token", "")) for x in new_list if isinstance(x, dict))

            if new_set == self._subscribed_token_set and len(new_set) > 0:
                self._logger.debug("Subscriptions unchanged; no update needed", token_count=len(new_set))
                return True

            old_set = set(self._subscribed_token_set)
            self._logger.info("Updating subscriptions", old=len(old_set), new=len(new_set))

            sws = self._sws
            generation = self._active_generation
            epoch = self._active_epoch

            self._subscribed_tokens = new_list
            self._subscribed_token_set = new_set
            self._token_scaling = self._build_token_scaling_map(new_list)
            self._token_exchange_type = self._build_token_exchange_type_map(new_list)

            self._subscription_requested_at = datetime.now(IST)
            self._subscription_warned_missing_once = False
            self._seen_tokens.clear()
            self._last_seen_by_token.clear()

        if sws is None or not self._is_callback_active(generation, epoch):
            return True

        token_list = self._build_token_list_for_api(new_list)

        try:
            if hasattr(sws, "unsubscribe") and old_set:
                old_token_list = self._build_token_list_for_api([{"exchange": "nfo_cm", "token": t} for t in old_set])
                try:
                    sws.unsubscribe("unique_correlation_id", old_token_list)  # type: ignore[attr-defined]
                    self._logger.info("Unsubscribed old tokens", old=len(old_set))
                except Exception as e:
                    self._logger.warning("Unsubscribe failed; continuing with subscribe", error=str(e))

            sws.subscribe("unique_correlation_id", 3, token_list)
            self._logger.info("Subscribed updated token list", total=len(new_set))
            return True

        except Exception as e:
            self._logger.error("Dynamic subscription update failed; forcing reconnect", error=str(e))
            self._request_reconnect(reason="subscription_update_failed")
            return False

    def wait_for_permanent_failure(self, timeout_sec: Optional[float] = None) -> bool:
        return self._permanent_failure_event.wait(timeout=timeout_sec)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            ws_alive = self._ws_thread.is_alive() if self._ws_thread else False
            consumer_alive = self._consumer_thread.is_alive() if self._consumer_thread else False
            watchdog_alive = self._watchdog_thread.is_alive() if self._watchdog_thread else False

            now = datetime.now(IST)
            tick_gap_sec = None
            if self.last_tick_time:
                tick_gap_sec = (now - self.last_tick_time).total_seconds()

            tps = self._estimate_ticks_per_second_locked()

            parse_window = list(self._parse_outcomes)
            parse_fail_rate = None
            if len(parse_window) >= 10:
                parse_fail_rate = round((parse_window.count(False) / max(len(parse_window), 1)) * 100.0, 2)

            qsize = self._tick_queue.qsize()

            return {
                "is_connected": self.is_connected,
                "reconnect_count": self.reconnect_count,
                "total_ticks_received": self.total_ticks_received,
                "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None,
                "last_connect_time": self.last_connect_time.isoformat() if self.last_connect_time else None,
                "last_error": self.last_error,
                "subscribed_tokens": len(self._subscribed_tokens),
                "manual_shutdown": self._manual_shutdown,
                "feed_permanently_dead": self._feed_permanently_dead,
                "permanent_failure_reason": self._permanent_failure_reason,
                "feed_stale": self._feed_stale,
                "feed_stale_since": self._feed_stale_since.isoformat() if self._feed_stale_since else None,
                "tick_gap_sec": tick_gap_sec,
                "ticks_per_second_est": tps,
                "parse_total": self._parse_total_count,
                "parse_failures": self._parse_failure_count,
                "parse_fail_rate_pct_window": parse_fail_rate,

                # pipeline metrics
                "queue_size": qsize,
                "queue_max": self._tick_queue_max,
                "dropped_ticks": self._dropped_ticks,
                "rate_limited_ticks": self._rate_limited_ticks,
                "avg_broker_latency_ms": self._avg_locked(self._lat_broker_ms),
                "avg_queue_latency_ms": self._avg_locked(self._lat_queue_ms),
                "avg_processing_latency_ms": self._avg_locked(self._lat_processing_ms),

                # threads
                "thread_alive": ws_alive,
                "ws_thread_alive": ws_alive,
                "consumer_thread_alive": consumer_alive,
                "watchdog_alive": watchdog_alive,
                "shutdown_complete": self._shutdown_complete,

                # gating
                "generation": self._generation,
                "active_generation": self._active_generation,
                "epoch": self._epoch,
                "active_epoch": self._active_epoch,
            }

    # ------------------------------------------------------------------
    # Internal: WS Thread
    # ------------------------------------------------------------------
    def _run_websocket(self, generation: int, epoch: int):
        """
        Build SmartWebSocketV2 inside thread; bind callbacks; connect blocking.
        """
        try:
            if not self._is_callback_active(generation, epoch):
                return

            ok, sws = self._build_sws_instance(generation=generation)
            if not ok or sws is None:
                self._mark_permanent_failure(reason="sws_build_failed", generation=generation, epoch=epoch)
                return

            with self._lock:
                self._sws = sws

            self._bind_callbacks(generation, epoch)
            sws.connect()

        except Exception as e:
            if (
                not self._stop_event.is_set()
                and not self._manual_shutdown
                and self._is_callback_active(generation, epoch)
                and not self._feed_permanently_dead
            ):
                self.last_error = str(e)
                self._logger.error("WebSocket thread error", error=str(e), generation=generation, epoch=epoch)
                self._attempt_reconnect(generation, epoch, reason="thread_exception")

    def _build_sws_instance(self, generation: int) -> Tuple[bool, Any]:
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError as e:
            self._logger.error("SmartWebSocketV2 import failed", error=str(e), generation=generation)
            return False, None

        auth_token = getattr(self._auth, "auth_token", None)
        api_key = getattr(self._auth, "_api_key", None)
        client_id = getattr(self._auth, "_client_id", None)
        feed_token = getattr(self._auth, "feed_token", None)

        if not all([auth_token, api_key, client_id, feed_token]):
            self._logger.error(
                "Missing websocket credentials",
                has_auth_token=bool(auth_token),
                has_api_key=bool(api_key),
                has_client_id=bool(client_id),
                has_feed_token=bool(feed_token),
                generation=generation,
            )
            return False, None

        # Disable broker SDK auto-reconnect; this manager already implements
        # generation-gated reconnect logic and needs deterministic shutdown.
        return True, SmartWebSocketV2(
            auth_token,
            api_key,
            client_id,
            feed_token,
            max_retry_attempt=0,
        )

    def _bind_callbacks(self, generation: int, epoch: int):
        if self._sws is None:
            return

        def on_open(wsapp):
            self._on_open(wsapp, generation, epoch)

        def on_data(wsapp, message):
            self._on_data(wsapp, message, generation, epoch)

        def on_error(wsapp, error):
            self._on_error(wsapp, error, generation, epoch)

        def on_close(wsapp):
            self._on_close(wsapp, generation, epoch)

        self._sws.on_open = on_open
        self._sws.on_data = on_data
        self._sws.on_error = on_error
        self._sws.on_close = on_close

    def _on_open(self, wsapp, generation: int, epoch: int):
        if not self._is_callback_active(generation, epoch):
            return

        with self._lock:
            self.is_connected = True
            self.reconnect_count = 0
            self.last_connect_time = datetime.now(IST)
            self._connection_established_time = time.time()  # PATCH: set establishment time for watchdog guard
            self.last_error = ""
            self._subscription_requested_at = datetime.now(IST)
            self._subscription_warned_missing_once = False
            self._seen_tokens.clear()
            self._last_seen_by_token.clear()

            self._parse_outcomes.clear()
            self._parse_total_count = 0
            self._parse_failure_count = 0

        self._logger.info("WebSocket connected", generation=generation, epoch=epoch)

        token_list = self._build_token_list_for_api(self._subscribed_tokens)
        if token_list and self._sws is not None:
            try:
                self._sws.subscribe("unique_correlation_id", 3, token_list)
                self._logger.info(
                    "Subscribed to instruments",
                    total=sum(len(x.get("tokens", [])) for x in token_list),
                    generation=generation,
                    epoch=epoch,
                )
            except Exception as e:
                self.last_error = str(e)
                self._logger.error("Subscription failed", error=str(e), generation=generation, epoch=epoch)

        self._mark_feed_restored_if_needed()

        if self._on_connect:
            try:
                self._on_connect()
            except Exception as e:
                self._logger.debug("on_connect callback failed", error=str(e))

    def _on_data(self, wsapp, message, generation: int, epoch: int):
        """
        Producer: parse -> enqueue. MUST NOT call _on_tick directly.
        """
        if not self._is_callback_active(generation, epoch):
            return

        now = datetime.now(IST)
        with self._lock:
            self.total_ticks_received += 1
            self.last_tick_time = now
            self._tick_times.append(now)

        tick = None
        try:
            tick = self._parse_tick(message, now)
        except Exception as e:
            self._record_parse_outcome(False)
            self._log_parse_error_rate_limited(message, error=str(e), where="_parse_tick_exception")
            self._parse_failure_circuit_breaker(generation, epoch)
            return

        if tick is None:
            self._record_parse_outcome(False)
            self._parse_failure_circuit_breaker(generation, epoch)
            return

        self._record_parse_outcome(True)

        # Subscription seen tracking (cheap)
        tok = str(tick.get("token", ""))
        with self._lock:
            if tok:
                self._seen_tokens.add(tok)
                self._last_seen_by_token[tok] = now
                self._latest_tick_by_token[tok] = tick  # always update cache

        # Enqueue with backpressure
        env = TickEnvelope(tick=tick, enqueued_at=now)
        self._enqueue_tick(env)

        # Restore stale marker if needed (feed restored even if consumer is slow)
        self._mark_feed_restored_if_needed()

    def _on_error(self, wsapp, error, generation: int, epoch: int):
        if not self._is_callback_active(generation, epoch):
            return
        self.last_error = str(error)
        self.is_connected = False
        self._logger.error("WebSocket error", error=str(error), type=type(error).__name__, generation=generation, epoch=epoch)

    def _on_close(self, wsapp, generation: int, epoch: int):
        self.is_connected = False

        if not self._is_callback_active(generation, epoch):
            self._logger.info(
                "WebSocket close ignored (stale/manual)",
                generation=generation, epoch=epoch,
                active_generation=self._active_generation,
                active_epoch=self._active_epoch,
            )
            return

        self._logger.warning("WebSocket disconnected", total_ticks=self.total_ticks_received, generation=generation, epoch=epoch)

        if self._on_disconnect:
            try:
                self._on_disconnect()
            except Exception as e:
                self._logger.debug("on_disconnect callback failed", error=str(e))

        self._attempt_reconnect(generation, epoch, reason="on_close")

    # ------------------------------------------------------------------
    # Tick Queue + Consumer (PHASE 5)
    # ------------------------------------------------------------------
    def _start_consumer_if_needed(self):
        if self._consumer_thread and self._consumer_thread.is_alive():
            return
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="WebSocket-Consumer",
        )
        self._consumer_thread.start()
        self._logger.info(
            "Tick consumer thread started",
            batch_enabled=self._batch_enabled,
            max_batch_size=self._max_batch_size,
            max_batch_interval_ms=self._max_batch_interval_ms,
            rate_limit_enabled=self._rate_limit_enabled,
            default_min_interval_ms=self._default_min_interval_ms,
            queue_max=self._tick_queue_max,
        )

    def _enqueue_tick(self, env: TickEnvelope):
        """
        Backpressure policy:
          - bounded queue
          - on full: drop OLDEST item, increment dropped_ticks, rate-limited warning
        """
        try:
            self._tick_queue.put_nowait(env)
            return
        except queue.Full:
            # Drop oldest
            dropped = False
            try:
                _ = self._tick_queue.get_nowait()
                dropped = True
            except queue.Empty:
                dropped = False

            try:
                self._tick_queue.put_nowait(env)
            except queue.Full:
                # extremely rare: keep dropping
                dropped = True

            with self._lock:
                if dropped:
                    self._dropped_ticks += 1
                now = datetime.now(IST)
                should_log = (self._last_drop_log is None) or ((now - self._last_drop_log).total_seconds() >= self._drop_log_suppress_sec)
                if should_log:
                    self._last_drop_log = now
                    self._logger.warning(
                        "Tick queue full; dropping oldest tick (backpressure)",
                        queue_max=self._tick_queue_max,
                        queue_size=self._tick_queue.qsize(),
                        dropped_ticks=self._dropped_ticks,
                    )

    def _consumer_loop(self):
        """
        Consumer thread:
          - Pull ticks from queue
          - Batch if enabled
          - Apply rate limiting
          - Track latencies
          - Call _on_batch (if provided) or _on_tick per tick
        """
        while not self._consumer_stop.is_set():
            try:
                # If no batching, process one by one
                if not self._batch_enabled or self._on_batch is None:
                    try:
                        env = self._tick_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    self._process_tick(env)
                    continue

                # Batch mode
                batch: List[TickEnvelope] = []
                start_mono = time.monotonic()
                deadline = start_mono + (self._max_batch_interval_ms / 1000.0)

                while len(batch) < self._max_batch_size:
                    timeout = max(0.0, deadline - time.monotonic())
                    if timeout <= 0 and batch:
                        break
                    try:
                        env = self._tick_queue.get(timeout=min(0.5, timeout) if timeout > 0 else 0.0)
                        batch.append(env)
                    except queue.Empty:
                        if batch:
                            break
                        break

                if not batch:
                    continue

                # Process batch: compute per-tick latencies + rate limiting decisions.
                ticks_to_send: List[Dict[str, Any]] = []
                for env in batch:
                    tick = env.tick
                    if self._should_process_tick(tick):
                        self._record_latencies(tick=tick, enqueued_at=env.enqueued_at)
                        ticks_to_send.append(tick)
                    else:
                        with self._lock:
                            self._rate_limited_ticks += 1

                if not ticks_to_send:
                    continue

                t0 = time.perf_counter()
                try:
                    self._on_batch(ticks_to_send)  # type: ignore[misc]
                except Exception as e:
                    self._logger.warning("Batch callback error", error=str(e), batch_size=len(ticks_to_send))
                finally:
                    t1 = time.perf_counter()
                    self._push_processing_latency_ms((t1 - t0) * 1000.0)

            except Exception as e:
                self._logger.error("Consumer loop error", error=str(e))

    def _process_tick(self, env: TickEnvelope):
        tick = env.tick
        if not self._should_process_tick(tick):
            with self._lock:
                self._rate_limited_ticks += 1
            return

        self._record_latencies(tick=tick, enqueued_at=env.enqueued_at)

        t0 = time.perf_counter()
        try:
            self._on_tick(tick)
        except Exception as e:
            self._logger.warning("Tick callback error", error=str(e), tick_token=str(tick.get("token", "")))
        finally:
            t1 = time.perf_counter()
            self._push_processing_latency_ms((t1 - t0) * 1000.0)

    def _should_process_tick(self, tick: Dict[str, Any]) -> bool:
        """
        Per-instrument rate limiting:
          - If disabled -> True
          - Else enforce min interval per token in ms using monotonic clock
        """
        if not self._rate_limit_enabled:
            return True

        token = str(tick.get("token", "") or "")
        if not token:
            return True

        min_ms = self._min_interval_by_token.get(token, self._default_min_interval_ms)
        if min_ms <= 0:
            return True

        now_m = time.monotonic()
        last_m = self._last_processed_monotonic_by_token.get(token)
        if last_m is None:
            self._last_processed_monotonic_by_token[token] = now_m
            return True

        if (now_m - last_m) * 1000.0 < float(min_ms):
            return False

        self._last_processed_monotonic_by_token[token] = now_m
        return True

    # ------------------------------------------------------------------
    # Latency Telemetry (PHASE 5)
    # ------------------------------------------------------------------
    def _record_latencies(self, tick: Dict[str, Any], enqueued_at: datetime):
        """
        broker_latency_ms = received_at - exchange_timestamp_dt
        queue_latency_ms  = now - enqueued_at
        """
        now = datetime.now(IST)

        # queue latency
        q_ms = (now - enqueued_at).total_seconds() * 1000.0
        with self._lock:
            self._lat_queue_ms.append(float(q_ms))

        # broker latency
        ex_dt = tick.get("exchange_timestamp_dt")
        if isinstance(ex_dt, datetime):
            recv_at = tick.get("received_at")
            if isinstance(recv_at, datetime):
                b_ms = (recv_at - ex_dt).total_seconds() * 1000.0
            else:
                b_ms = (now - ex_dt).total_seconds() * 1000.0
            with self._lock:
                self._lat_broker_ms.append(float(b_ms))

    def _push_processing_latency_ms(self, ms: float):
        with self._lock:
            self._lat_processing_ms.append(float(ms))

    # ------------------------------------------------------------------
    # Reconnect + failure semantics (preserved)
    # ------------------------------------------------------------------
    def _attempt_reconnect(self, generation: int, epoch: int, reason: str):
        if not self._is_callback_active(generation, epoch):
            return

        if not self._reconnect_lock.acquire(blocking=False):
            self._logger.debug("Reconnect already in progress; skipping", reason=reason, generation=generation, epoch=epoch)
            return

        try:
            while True:
                if not self._is_callback_active(generation, epoch):
                    return

                self.reconnect_count += 1
                if self.reconnect_count > self._max_retries:
                    self._mark_permanent_failure(reason=f"max_retries_exhausted({self._max_retries})", generation=generation, epoch=epoch)
                    return

                delay = self._reconnect_delay * (2 ** (self.reconnect_count - 1))
                delay = min(delay, self._max_backoff)

                self._logger.info("Reconnecting after backoff", delay_sec=delay, attempt=self.reconnect_count, reason=reason)
                time.sleep(delay)

                if not self._is_callback_active(generation, epoch):
                    return

                if not self._refresh_tokens_strict():
                    self._mark_permanent_failure(reason="token_refresh_failed", generation=generation, epoch=epoch)
                    return

                ok, sws = self._build_sws_instance(generation=generation)
                if not ok or sws is None:
                    self.last_error = "SmartWebSocketV2 build failed"
                    continue

                with self._lock:
                    self._sws = sws
                    self._epoch += 1
                    new_epoch = self._epoch
                    self._active_epoch = new_epoch

                    self._parse_outcomes.clear()
                    self._parse_total_count = 0
                    self._parse_failure_count = 0
                    self._subscription_requested_at = datetime.now(IST)
                    self._subscription_warned_missing_once = False
                    self._seen_tokens.clear()
                    self._last_seen_by_token.clear()
                    self._drain_tick_queue_locked()

                self._bind_callbacks(generation, new_epoch)

                try:
                    if self._sws is not None and self._is_callback_active(generation, new_epoch):
                        self._logger.info("Reconnect connect() starting", generation=generation, epoch=new_epoch)
                        self._sws.connect()
                        return
                except Exception as e:
                    self.last_error = str(e)
                    self._logger.error("Reconnection connect() failed", error=str(e), attempt=self.reconnect_count)
        finally:
            try:
                self._reconnect_lock.release()
            except Exception:
                pass

    def _mark_permanent_failure(self, reason: str, generation: int, epoch: int):
        with self._lock:
            self._feed_permanently_dead = True
            self._permanent_failure_reason = reason
            self.is_connected = False
            self.last_error = reason

            self._active_generation = -1
            self._active_epoch = -1

            self._stop_event.set()
            self._consumer_stop.set()
            self._watchdog_stop.set()
            self._permanent_failure_event.set()

            ws_ref = self._sws

        self._logger.critical("Feed permanently dead", reason=reason, generation=generation, epoch=epoch, max_retries=self._max_retries)

        if self._on_feed_permanent_failure:
            try:
                self._on_feed_permanent_failure(self.get_status())
            except Exception as e:
                self._logger.debug("on_feed_permanent_failure callback failed", error=str(e))

        if ws_ref is not None:
            try:
                ws_ref.close_connection()
            except Exception as e:
                self._logger.debug("Permanent failure close warning", error=str(e))

    def _refresh_tokens_strict(self) -> bool:
        if not hasattr(self._auth, "refresh_session"):
            return self._credentials_present()
        try:
            rv = self._auth.refresh_session()
        except Exception as e:
            self._logger.critical("Token refresh exception", error=str(e))
            return False
        if isinstance(rv, bool) and rv is False:
            self._logger.critical("Token refresh returned False")
            return False
        if not self._credentials_present():
            self._logger.critical("Credentials missing after refresh")
            return False
        return True

    def _credentials_present(self) -> bool:
        auth_token = getattr(self._auth, "auth_token", None)
        api_key = getattr(self._auth, "_api_key", None)
        client_id = getattr(self._auth, "_client_id", None)
        feed_token = getattr(self._auth, "feed_token", None)
        return bool(auth_token and api_key and client_id and feed_token)

    def _is_callback_active(self, generation: int, epoch: int) -> bool:
        return (
            not self._manual_shutdown
            and not self._stop_event.is_set()
            and generation == self._active_generation
            and epoch == self._active_epoch
            and not self._feed_permanently_dead
        )

    # ------------------------------------------------------------------
    # Watchdog (stale + healing)
    # ------------------------------------------------------------------
    def _start_watchdog_if_needed(self):
        with self._lock:
            if self._watchdog_thread and self._watchdog_thread.is_alive():
                return
            self._watchdog_stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="WebSocket-Watchdog",
            )
            self._watchdog_thread.start()
            self._logger.info("WebSocket watchdog started", interval_sec=self._watchdog_interval_sec, timeout_sec=self._heartbeat_timeout_sec)

    def _watchdog_loop(self):
        while not self._watchdog_stop.is_set():
            try:
                time.sleep(max(0.5, float(self._watchdog_interval_sec)))

                with self._lock:
                    if self._manual_shutdown or self._stop_event.is_set() or self._feed_permanently_dead:
                        return
                    generation = self._active_generation
                    epoch = self._active_epoch
                    is_conn = self.is_connected
                    last_tick = self.last_tick_time
                    conn_established_time = self._connection_established_time  # PATCH: guard read

                    sub_at = self._subscription_requested_at
                    subscribed = set(self._subscribed_token_set)
                    seen = set(self._seen_tokens)
                    warned_once = self._subscription_warned_missing_once

                    last_heal = self._last_healing_check

                if not self._is_market_hours_now():
                    continue

                now = datetime.now(IST)

                # Heartbeat stale detection
                if is_conn:
                    # PATCH: Guard against early stale-trigger right after connect()
                    # Conditions:
                    #   - connection established >= 10 seconds ago
                    #   - at least one tick has been observed (last_tick is not None)
                    if conn_established_time is not None and (time.time() - float(conn_established_time)) > 10.0:
                        if last_tick is not None:
                            gap = (now - last_tick).total_seconds()
                            if gap > float(self._heartbeat_timeout_sec):
                                if (generation, epoch) != self._last_stale_trigger_gen_epoch:
                                    self._last_stale_trigger_gen_epoch = (generation, epoch)
                                    self._mark_feed_stale(now, gap_sec=gap, generation=generation, epoch=epoch)
                                    self._request_reconnect(reason="heartbeat_stale")
                            else:
                                if self._feed_stale:
                                    self._mark_feed_restored(now, reason="watchdog_tick_gap_recovered")
                        else:
                            # No ticks seen yet; skip stale check (subscription warning/healing covers this case)
                            pass
                    else:
                        # Connection just started; give it time to receive first ticks
                        pass

                # Passive subscription warning
                if is_conn and sub_at is not None and subscribed and not warned_once:
                    age = (now - sub_at).total_seconds()
                    if age >= float(self._subscription_no_tick_warn_sec):
                        missing = subscribed - seen
                        if missing:
                            self._logger.warning(
                                "Subscription tokens not seen after subscribe window",
                                missing_count=len(missing),
                                subscribed_count=len(subscribed),
                                seen_count=len(seen),
                                sample_missing=list(sorted(list(missing)))[:10],
                            )
                            with self._lock:
                                self._subscription_warned_missing_once = True

                # Auto-healing subscription (every healing interval)
                if is_conn:
                    if last_heal is None or (now - last_heal).total_seconds() >= float(self._healing_check_interval_sec):
                        with self._lock:
                            self._last_healing_check = now
                        self._auto_heal_subscriptions(now)

            except Exception as e:
                self._logger.error("Watchdog loop error", error=str(e))

    def _auto_heal_subscriptions(self, now: datetime):
        """
        Auto-heal missing tokens:
          If a subscribed token hasn't been seen for > resubscribe_threshold_sec,
          attempt partial resubscribe; fallback to reconnect.
        """
        with self._lock:
            subscribed = set(self._subscribed_token_set)
            last_seen = dict(self._last_seen_by_token)
            sws = self._sws
            gen = self._active_generation
            ep = self._active_epoch

        if sws is None or not self._is_callback_active(gen, ep):
            return

        missing: List[str] = []
        for tok in subscribed:
            ls = last_seen.get(tok)
            if ls is None:
                missing.append(tok)
                continue
            if (now - ls).total_seconds() > float(self._resubscribe_threshold_sec):
                missing.append(tok)

        if not missing:
            return

        # Respect per-token resubscribe cooldown
        eligible: List[str] = []
        with self._lock:
            for tok in missing:
                last_try = self._last_resubscribe_attempt.get(tok)
                if last_try is None or (now - last_try).total_seconds() >= float(self._resubscribe_min_gap_sec):
                    eligible.append(tok)
                    self._last_resubscribe_attempt[tok] = now

        if not eligible:
            return

        self._logger.warning("Auto-healing missing subscription tokens", missing_count=len(eligible), sample=eligible[:10])

        # Try partial resubscribe
        if self._resubscribe_tokens(eligible):
            return

        # Fallback to reconnect
        self._logger.error("Partial resubscribe not supported/failed; forcing reconnect")
        self._request_reconnect(reason="subscription_heal_failed")

    def _resubscribe_tokens(self, tokens: List[str]) -> bool:
        """
        Attempt to subscribe to specific tokens without full reconnect.
        Returns True on success, False otherwise.
        """
        with self._lock:
            sws = self._sws
            token_exchange = dict(self._token_exchange_type)
            gen = self._active_generation
            ep = self._active_epoch

        if sws is None or not self._is_callback_active(gen, ep):
            return False

        # Build token_list
        nse = [t for t in tokens if token_exchange.get(t, 2) == 1]
        nfo = [t for t in tokens if token_exchange.get(t, 2) == 2]

        token_list = []
        if nse:
            token_list.append({"exchangeType": 1, "tokens": nse})
        if nfo:
            token_list.append({"exchangeType": 2, "tokens": nfo})

        if not token_list:
            return False

        try:
            sws.subscribe("heal_correlation_id", 3, token_list)
            self._logger.info("Partial resubscribe sent", nse=len(nse), nfo=len(nfo))
            return True
        except Exception as e:
            self._logger.warning("Partial resubscribe failed", error=str(e))
            return False

    def _mark_feed_stale(self, now: datetime, gap_sec: float, generation: int, epoch: int):
        with self._lock:
            if self._feed_stale:
                return
            self._feed_stale = True
            self._feed_stale_since = now

        self._logger.warning("Feed stale detected", gap_sec=round(gap_sec, 3), timeout_sec=self._heartbeat_timeout_sec)

        if self._on_feed_stale:
            try:
                self._on_feed_stale(self.get_status())
            except Exception as e:
                self._logger.debug("on_feed_stale callback failed", error=str(e))

    def _mark_feed_restored_if_needed(self):
        with self._lock:
            if not self._feed_stale:
                return
        self._mark_feed_restored(datetime.now(IST), reason="tick_received")

    def _mark_feed_restored(self, now: datetime, reason: str):
        with self._lock:
            if not self._feed_stale:
                return
            self._feed_stale = False
            self._feed_stale_since = None
            self._last_stale_trigger_gen_epoch = (-1, -1)

        self._logger.info("Feed restored", reason=reason)

        if self._on_feed_restored:
            try:
                self._on_feed_restored(self.get_status())
            except Exception as e:
                self._logger.debug("on_feed_restored callback failed", error=str(e))

    def _request_reconnect(self, reason: str):
        with self._lock:
            if self._manual_shutdown or self._stop_event.is_set() or self._feed_permanently_dead:
                return
            ws_ref = self._sws
            gen = self._active_generation
            ep = self._active_epoch

        if ws_ref is None:
            return

        self._logger.warning("Forcing reconnect by closing connection", reason=reason, generation=gen, epoch=ep)
        try:
            ws_ref.close_connection()
        except Exception as e:
            self._logger.debug("close_connection error during reconnect request", error=str(e))

    # ------------------------------------------------------------------
    # Subscription list / scaling
    # ------------------------------------------------------------------
    def _safe_get_subscription_list(self, spot_price: float) -> List[Dict]:
        try:
            lst = self._mapper.get_subscription_list(spot_price)
            if not isinstance(lst, list):
                self._logger.error("InstrumentMapper returned non-list subscription list", type=str(type(lst)))
                return []
            out = []
            for it in lst:
                if isinstance(it, dict) and it.get("token") is not None:
                    out.append(it)
            return out
        except Exception as e:
            self._logger.error("Failed to get subscription list", error=str(e))
            return []

    def _build_token_list_for_api(self, subscribed_tokens: List[Dict]) -> List[Dict]:
        nse_tokens: List[str] = []
        nfo_tokens: List[str] = []
        for item in subscribed_tokens:
            if not isinstance(item, dict):
                continue
            exchange = item.get("exchange", "")
            token = str(item.get("token", ""))
            if not token:
                continue
            if exchange == "nse_cm":
                nse_tokens.append(token)
            elif exchange == "nfo_cm":
                nfo_tokens.append(token)
            else:
                ex_type = item.get("exchangeType", None)
                if ex_type == 1:
                    nse_tokens.append(token)
                elif ex_type == 2:
                    nfo_tokens.append(token)

        token_list = []
        if nse_tokens:
            token_list.append({"exchangeType": 1, "tokens": nse_tokens})
        if nfo_tokens:
            token_list.append({"exchangeType": 2, "tokens": nfo_tokens})
        return token_list

    def _build_token_exchange_type_map(self, subscribed_tokens: List[Dict]) -> Dict[str, int]:
        mp: Dict[str, int] = {}
        for item in subscribed_tokens:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", ""))
            if not token:
                continue
            exchange = item.get("exchange", "")
            if exchange == "nse_cm":
                mp[token] = 1
            elif exchange == "nfo_cm":
                mp[token] = 2
            else:
                ex_type = item.get("exchangeType", None)
                if ex_type in (1, 2):
                    mp[token] = int(ex_type)
                else:
                    mp[token] = 2
        return mp

    def _build_token_scaling_map(self, subscribed_tokens: List[Dict]) -> Dict[str, float]:
        scaling: Dict[str, float] = {}
        for item in subscribed_tokens:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", ""))
            if not token:
                continue

            sf = item.get("scaling_factor", None)
            try:
                if sf is not None:
                    sfv = float(sf)
                    if sfv > 0:
                        scaling[token] = sfv
                        continue
            except Exception:
                pass

            try:
                if hasattr(self._mapper, "get_scaling_factor"):
                    ex = item.get("exchange", None)
                    sf2 = self._mapper.get_scaling_factor(token, ex)  # type: ignore[attr-defined]
                    sf2v = float(sf2)
                    if sf2v > 0:
                        scaling[token] = sf2v
                        continue
            except Exception:
                pass

            exchange = item.get("exchange", "")
            if exchange == "nse_cm":
                scaling[token] = 100.0
            elif exchange == "nfo_cm":
                scaling[token] = 1.0
            else:
                ex_type = item.get("exchangeType", None)
                scaling[token] = 100.0 if ex_type == 1 else 1.0
        return scaling

    # ------------------------------------------------------------------
    # Parse Tick + Price Scaling + Latency extraction
    # ------------------------------------------------------------------
    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or isinstance(value, bool):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int:
        try:
            if value is None:
                return 0
            return int(float(value))
        except Exception:
            return 0

    def _safe_int_or_none(self, value: Any) -> Optional[int]:
        """
        Safer version for optional fields (OI etc).
        Returns None if conversion fails, instead of 0 (to avoid false "available" state).
        """
        try:
            if value is None or isinstance(value, bool):
                return None
            # Handle numeric strings like "12345.0"
            iv = int(float(value))
            if iv < 0:
                return None
            return iv
        except Exception:
            return None

    def _first_present_value(self, message: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in message:
                v = message.get(k)
                if v is not None:
                    return v
        return None

    def _scale_price(self, token: str, raw: Any, exchange_type: Optional[int] = None) -> float:
        v = self._safe_float(raw)
        if v is None:
            return 0.0
        with self._lock:
            sf = self._token_scaling.get(token)
        if sf is None:
            sf = 100.0 if exchange_type == 1 else 1.0
        if sf <= 0:
            sf = 1.0
        return round(float(v) / float(sf), 2)

    def _parse_exchange_timestamp(self, raw_ts: Any) -> Optional[datetime]:
        """
        Attempt to parse broker exchange_timestamp:
        - int/float epoch in ms or sec
        - ISO string
        - else None
        """
        if raw_ts is None:
            return None
        try:
            if isinstance(raw_ts, (int, float)) and not isinstance(raw_ts, bool):
                v = float(raw_ts)
                if v > 1e12:
                    return datetime.fromtimestamp(v / 1000.0, tz=IST)
                if v > 1e9:
                    return datetime.fromtimestamp(v, tz=IST)
                return None
            s = str(raw_ts)
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST)
                return dt.astimezone(IST)
            except Exception:
                return None
        except Exception:
            return None

    def _parse_tick(self, message: Any, received_at: datetime) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict):
            return None

        token = str(message.get("token", "") or "")
        if not token:
            # Without token we cannot route OI/price updates.
            return None

        exchange_type = message.get("exchange_type", message.get("exchangeType", None))
        try:
            exchange_type_int = int(exchange_type) if exchange_type is not None else None
        except Exception:
            exchange_type_int = None

        try:
            ltp_raw = message.get("last_traded_price", message.get("ltp", message.get("lastPrice", 0)))
            open_raw = message.get("open_price_of_the_day", message.get("open", 0))
            high_raw = message.get("high_price_of_the_day", message.get("high", 0))
            low_raw = message.get("low_price_of_the_day", message.get("low", 0))
            close_raw = message.get("closed_price", message.get("close_price", message.get("close", 0)))
            vol_raw = message.get("volume_trade_for_the_day", message.get("volume", 0))

            ex_ts_raw = message.get("exchange_timestamp", message.get("exchangeTimestamp", None))
            ex_ts_dt = self._parse_exchange_timestamp(ex_ts_raw)

            ltp = self._scale_price(token, ltp_raw, exchange_type=exchange_type_int)
            open_price = self._scale_price(token, open_raw, exchange_type=exchange_type_int)
            high = self._scale_price(token, high_raw, exchange_type=exchange_type_int)
            low = self._scale_price(token, low_raw, exchange_type=exchange_type_int)
            close = self._scale_price(token, close_raw, exchange_type=exchange_type_int)

            try:
                volume = int(float(vol_raw or 0))
            except Exception:
                volume = 0

            best_5_buy = []
            best_5_sell = []
            for i in range(5):
                buy_price = self._scale_price(token, message.get(f"best_5_buy_price_{i+1}", 0), exchange_type=exchange_type_int)
                buy_qty = self._safe_int(message.get(f"best_5_buy_qty_{i+1}", 0))
                if buy_price > 0:
                    best_5_buy.append({"price": buy_price, "qty": buy_qty})

                sell_price = self._scale_price(token, message.get(f"best_5_sell_price_{i+1}", 0), exchange_type=exchange_type_int)
                sell_qty = self._safe_int(message.get(f"best_5_sell_qty_{i+1}", 0))
                if sell_price > 0:
                    best_5_sell.append({"price": sell_price, "qty": sell_qty})

            tick: Dict[str, Any] = {
                "token": token,
                "ltp": ltp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "best_5_buy": best_5_buy,
                "best_5_sell": best_5_sell,
                "exchange_timestamp": ex_ts_raw if ex_ts_raw is not None else "",
                "exchange_timestamp_dt": ex_ts_dt,
                "received_at": received_at,
                "subscription_mode": message.get("subscription_mode", message.get("subscriptionMode", 0)),
                "exchange_type": exchange_type_int or 0,
                "sequence_number": message.get("sequence_number", message.get("sequenceNumber", 0)),
            }

            # ---------------------------
            # PATCH: OI field extraction
            # ---------------------------
            oi_raw = self._first_present_value(
                message,
                keys=[
                    "oi",
                    "open_interest",
                    "openInterest",
                    "openInterestValue",
                    "open_interest_value",
                    "openInt",
                    "openIntValue",
                ],
            )
            if oi_raw is not None:
                oi = self._safe_int_or_none(oi_raw)
                if oi is not None:
                    tick["oi"] = oi

            prev_oi_raw = self._first_present_value(
                message,
                keys=[
                    "prev_oi",
                    "previous_oi",
                    "prevOpenInterest",
                    "previousOpenInterest",
                    "prevOpenInterestValue",
                ],
            )
            if prev_oi_raw is not None:
                prev_oi = self._safe_int_or_none(prev_oi_raw)
                if prev_oi is not None:
                    tick["prev_oi"] = prev_oi

            oi_change_raw = self._first_present_value(
                message,
                keys=[
                    "oi_change",
                    "oiChange",
                    "openInterestChange",
                    "open_interest_change",
                    "changeinOpenInterest",
                    "changeInOpenInterest",
                    "changeInOI",
                ],
            )
            if oi_change_raw is not None:
                oi_change = self._safe_int_or_none(oi_change_raw)
                if oi_change is not None:
                    tick["oi_change"] = oi_change

            return tick

        except Exception as e:
            self._log_parse_error_rate_limited(message, error=str(e), where="_parse_tick")
            return None

    # ------------------------------------------------------------------
    # Parse failure circuit breaker
    # ------------------------------------------------------------------
    def _record_parse_outcome(self, success: bool):
        with self._lock:
            self._parse_total_count += 1
            if not success:
                self._parse_failure_count += 1
            self._parse_outcomes.append(bool(success))

    def _parse_failure_circuit_breaker(self, generation: int, epoch: int):
        with self._lock:
            window = list(self._parse_outcomes)
        if len(window) < 60:
            return
        failures = window.count(False)
        rate = failures / max(len(window), 1)
        if rate > 0.50 and self._is_callback_active(generation, epoch):
            self._logger.critical("Parse failure circuit breaker triggered", failure_rate_pct=round(rate * 100.0, 2), window=len(window))
            self._request_reconnect(reason="parse_failure_circuit_breaker")

    def _log_parse_error_rate_limited(self, message: Any, error: str, where: str):
        try:
            keys = tuple(sorted(list(message.keys()))) if isinstance(message, dict) else ("non_dict",)
            sig = f"{where}|{hash(keys)}"
            now = datetime.now(IST)

            with self._lock:
                last = self._parse_error_last_log.get(sig)

            if last is not None and (now - last).total_seconds() < float(self._parse_error_suppress_sec):
                return

            raw_trunc = str(message)[:200]
            self._logger.error("Tick parse failure", where=where, error=error, signature=sig, raw_message_trunc=raw_trunc)

            with self._lock:
                self._parse_error_last_log[sig] = now
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers: queue drain + avg + TPS
    # ------------------------------------------------------------------
    def _drain_tick_queue_locked(self):
        try:
            while True:
                _ = self._tick_queue.get_nowait()
        except queue.Empty:
            pass

    def _avg_locked(self, dq: Deque[float]) -> Optional[float]:
        if not dq:
            return None
        return round(sum(dq) / max(len(dq), 1), 3)

    def _estimate_ticks_per_second_locked(self) -> float:
        if len(self._tick_times) < 2:
            return 0.0
        span = (self._tick_times[-1] - self._tick_times[0]).total_seconds()
        if span <= 0:
            return 0.0
        return round(len(self._tick_times) / span, 3)

    # ------------------------------------------------------------------
    # Market hours helper (no external dependency)
    # ------------------------------------------------------------------
    def _is_market_hours_now(self) -> bool:
        try:
            now = datetime.now(IST)
            open_str = Config.get("market", "market_open", default="09:15")
            close_str = Config.get("market", "market_close", default="15:30")
            oh, om = [int(x) for x in str(open_str).split(":")]
            ch, cm = [int(x) for x in str(close_str).split(":")]
            start = now.replace(hour=oh, minute=om, second=0, microsecond=0)
            end = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
            return start <= now <= end
        except Exception:
            return True


# =====================================================================================
# Module Self-Test (Structural + Pipeline)
# =====================================================================================

def _run_tests():
    """
    Structural tests only (no live connect).
    Validates queue/backpressure logic + scaling + basic telemetry mechanics.
    """
    print("=" * 60)
    print(" JUNIOR ALADDIN — WebSocket Manager Test (Pipeline Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    class DummyAuth:
        auth_token = "a"
        feed_token = "b"
        _api_key = "k"
        _client_id = "c"

        def refresh_session(self):
            return True

    class DummyMapper:
        def get_subscription_list(self, spot_price):
            return [
                {"exchange": "nse_cm", "token": "99926000"},
                {"exchange": "nfo_cm", "token": "51234"},
            ]

    # Test 1: queue backpressure
    print(" [Test 1] Bounded queue + drop-oldest policy...")
    ws = WebSocketManager(DummyAuth(), DummyMapper(), lambda t: None)
    ws._tick_queue = queue.Queue(maxsize=5)  # force tiny queue
    ws._tick_queue_max = 5

    now = datetime.now(IST)
    for _i in range(10):
        ws._enqueue_tick(TickEnvelope(tick={"token": "99926000", "received_at": now, "exchange_timestamp_dt": None}, enqueued_at=now))
    if ws._tick_queue.qsize() <= 5:
        print("  ✅ Queue bounded")
        passed += 1
    else:
        print("  ❌ Queue unbounded")
        failed += 1

    if ws._dropped_ticks > 0:
        print(f"  ✅ Dropped ticks tracked: {ws._dropped_ticks}")
        passed += 1
    else:
        print("  ❌ Dropped ticks not tracked")
        failed += 1

    # Test 2: scaling
    print("\n [Test 2] Instrument-aware scaling...")
    scaling = ws._build_token_scaling_map(DummyMapper().get_subscription_list(24500))
    ws._token_scaling = scaling
    spot = ws._scale_price("99926000", 2233140, exchange_type=1)
    opt = ws._scale_price("51234", 105.5, exchange_type=2)
    if abs(spot - 22331.40) < 0.01 and abs(opt - 105.5) < 0.01:
        print("  ✅ Scaling correct")
        passed += 1
    else:
        print(f"  ❌ Scaling wrong: spot={spot}, opt={opt}")
        failed += 1

    # Test 3: exchange timestamp parsing
    print("\n [Test 3] Exchange timestamp parsing...")
    dt_ms = ws._parse_exchange_timestamp(1710000000000)  # ms epoch
    dt_iso = ws._parse_exchange_timestamp("2026-03-25 09:15:00+05:30")
    if isinstance(dt_ms, datetime) and isinstance(dt_iso, datetime):
        print("  ✅ Exchange timestamp parsing works")
        passed += 1
    else:
        print("  ❌ Exchange timestamp parsing failed")
        failed += 1

    # Test 4: OI extraction parsing
    print("\n [Test 4] OI field extraction...")
    msg = {
        "token": "51234",
        "exchangeType": 2,
        "last_traded_price": 123.45,
        "open_interest": "987654",
        "prev_oi": 980000,
        "oi_change": 7654,
        "volume_trade_for_the_day": 1000,
        "exchange_timestamp": 1710000000000,
    }
    tick = ws._parse_tick(msg, received_at=datetime.now(IST))
    if isinstance(tick, dict) and tick.get("oi") == 987654 and tick.get("prev_oi") == 980000 and tick.get("oi_change") == 7654:
        print("  ✅ OI fields parsed into tick")
        passed += 1
    else:
        print(f"  ❌ OI fields missing/incorrect: tick={tick}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()