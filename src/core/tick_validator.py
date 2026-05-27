"""
Junior Aladdin - Tick Validator (Layer 1) — INSTITUTIONAL-GRADE (Hardened)

FIRST LINE OF DEFENSE for live market data integrity.

Core pipeline (fixed order for in-market ticks):
1) TOKEN VALIDATION + SPEC RESOLUTION (with TTL cache + stale fallback)
2) TIMESTAMP PARSING + TZ NORMALIZATION (IST-aware)
3) TRADING DAY + MARKET HOURS FILTER (holiday-aware)
4) PRICE PARSING + PRICE BOUNDS + TICK SIZE MULTIPLE CHECK
5) ORDERING / DUPLICATE TOLERANCE (thread-safe)
6) FEED GAP + FEED HEALTH + GAP-BASED BASELINE RESET
7) SPIKE DETECTION (dynamic threshold)
8) STATE UPDATE
9) OUTPUT (includes feed_health + latency flags)

validate() NEVER raises; returns dict (valid) or None (rejected).
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from src.utils.config_loader import Config
from src.utils.logger import setup_logger

# helpers
from src.utils.helpers import IST, is_market_hours, ist_now  # type: ignore

try:
    from src.utils.helpers import is_trading_day  # type: ignore
except Exception:  # pragma: no cover
    is_trading_day = None  # type: ignore


_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class InstrumentSpec:
    """
    Normalized instrument specification resolved from instrument_mapper.

    Fields are best-effort; missing min/max/tick_size can be None and will fall back to config defaults.
    expiry: for derivatives; if provided and < tick date => reject.
    """
    token: str
    symbol: str
    instrument_class: str  # e.g., "INDEX", "STOCK", "FUTURE", "OPTION"
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    tick_size: Optional[float] = None
    expiry: Optional[date] = None

    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return self.min_price, self.max_price


@dataclass
class _CachedSpecEntry:
    spec: InstrumentSpec
    cached_at_ist: datetime


@dataclass
class _PerTokenState:
    last_timestamp: Optional[datetime] = None  # IST tz-aware
    last_price: Optional[float] = None
    last_volume: Optional[int] = None  # may be -1 for missing volume
    last_feed_health: Optional[str] = None

    # Logging rate limit
    last_feed_health_log_time: Optional[datetime] = None
    last_feed_health_logged_class: Optional[str] = None

    # For adaptive spike threshold
    ret_window: Deque[float] = None  # pct changes (absolute), e.g. 0.12 means 0.12%
    ret_sum: float = 0.0
    ret_sumsq: float = 0.0

    def __post_init__(self) -> None:
        if self.ret_window is None:
            self.ret_window = deque()


@dataclass(frozen=True)
class ValidatedTick:
    timestamp: datetime  # tz-aware IST
    token: str
    symbol: str
    instrument_class: str
    ltp: float
    volume: int  # -1 means "missing volume data"
    is_spike: bool
    feed_gap_sec: float
    feed_health: str  # HEALTHY / DELAYED / STALE / DOWN / FIRST_TICK
    latency_sec: float
    is_stale: bool
    reconnect_burst: bool
    same_timestamp_update: bool
    tick_size: Optional[float]
    dynamic_spike_threshold_pct: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "token": self.token,
            "symbol": self.symbol,
            "instrument_class": self.instrument_class,
            "ltp": self.ltp,
            "volume": self.volume,
            "is_spike": self.is_spike,
            "feed_gap_sec": self.feed_gap_sec,
            "feed_health": self.feed_health,
            "latency_sec": self.latency_sec,
            "is_stale": self.is_stale,
            "reconnect_burst": self.reconnect_burst,
            "same_timestamp_update": self.same_timestamp_update,
            "tick_size": self.tick_size,
            "dynamic_spike_threshold_pct": self.dynamic_spike_threshold_pct,
        }


class TickValidator:
    """
    Institutional-grade tick validator with multi-instrument awareness.

    Constructor requires instrument_mapper. The mapper can be any object that can resolve token -> spec.
    Supported mapper patterns (best-effort):
    - method: get_instrument_spec(token) -> dict|InstrumentSpec
    - method: get_spec(token) -> dict|InstrumentSpec
    - attribute: token_specs / token_spec_map / token_map : dict[token] -> dict|InstrumentSpec
    """

    # 1. Add a class-level constant for the default index spec (fallback).
    _DEFAULT_INDEX_SPEC = {
        "symbol": "NIFTY",
        "instrument_class": "INDEX",
        "min_price": 10000.0,
        "max_price": 50000.0,
        "tick_size": 0.05,
        "expiry": None,
    }

    # Rejection reasons
    _R_TOKEN_MISSING = "token_missing"
    _R_TOKEN_UNKNOWN = "token_unknown"
    _R_TOKEN_EXPIRED = "token_expired"
    _R_INVALID_TYPE = "invalid_type"
    _R_MISSING_FIELDS = "missing_fields"
    _R_BAD_TIMESTAMP = "bad_timestamp"
    _R_FUTURE_TIMESTAMP = "future_timestamp"
    _R_NON_TRADING_DAY = "non_trading_day"  # F3
    _R_OUTSIDE_MARKET_HOURS = "outside_market_hours"
    _R_BAD_LTP = "bad_ltp"
    _R_PRICE_OUT_OF_BOUNDS = "price_out_of_bounds"
    _R_PRICE_NOT_MULTIPLE_TICK = "price_not_multiple_of_tick_size"  # F7
    _R_OUT_OF_ORDER = "out_of_order"
    _R_TRUE_DUPLICATE = "true_duplicate"
    _R_OTHER = "other"

    def __init__(self, instrument_mapper, engine_name: str = "tick_validator") -> None:
        self._instrument_mapper = instrument_mapper
        self._log = setup_logger(engine_name)
        self._lock = threading.Lock()

        # per-token state
        self._states: Dict[str, _PerTokenState] = {}

        # --- F2: instrument spec cache ---
        self._spec_cache: Dict[str, _CachedSpecEntry] = {}
        self._spec_ttl_sec: int = int(Config.get("tick_validator", "spec_cache_ttl_sec", default=60))

        # callbacks
        self._on_feed_health_change: Optional[Callable[..., None]] = None

        # --- F4: circuit breaker callback ---
        self._on_circuit_breaker: Optional[Callable[..., None]] = None
        self._cb_threshold: int = int(Config.get("tick_validator", "cb_threshold", default=50))
        self._cb_last_call_time: Optional[datetime] = None
        self._cb_call_interval_sec: float = 60.0

        # --- F8: multi-token monitored health callback ---
        nifty_token = str(Config.get("market", "nifty_spot_token", default="99926000"))
        self._spot_token = nifty_token
        self._health_monitored_tokens = {nifty_token}

        # --- F5: volume overflow warn rate-limit per token ---
        self._overflow_warning_interval_sec: float = float(
            Config.get("tick_validator", "overflow_warning_interval_sec", default=60)
        )
        self._last_overflow_warn_by_token: Dict[str, datetime] = {}

        # --- F6: naive timestamp assumption ---
        self._assume_naive_is_utc: bool = bool(Config.get("tick_validator", "assume_naive_is_utc", default=False))
        self._last_naive_ts_debug_time: Optional[datetime] = None
        self._naive_ts_debug_interval_sec: float = 60.0

        # --- F7: tick size enforcement ---
        self._enforce_tick_size: bool = bool(Config.get("tick_validator", "enforce_tick_size", default=True))

        # Global counters / stats (lock protected)
        self._total_seen: int = 0
        self._total_valid: int = 0

        self._rejected_token_missing: int = 0
        self._rejected_token_unknown: int = 0
        self._rejected_token_expired: int = 0

        self._rejected_missing_fields: int = 0
        self._rejected_bad_ltp: int = 0
        self._rejected_bad_timestamp: int = 0
        self._rejected_future_timestamp: int = 0
        self._rejected_non_trading_day: int = 0
        self._rejected_outside_hours: int = 0
        self._rejected_price_bounds: int = 0
        self._rejected_tick_size: int = 0
        self._rejected_out_of_order: int = 0
        self._rejected_true_duplicates: int = 0
        self._rejected_other: int = 0

        self._flagged_spikes: int = 0
        self._spike_baseline_resets: int = 0
        self._reconnect_bursts: int = 0

        self._consecutive_rejections: int = 0
        self._last_corruption_log_time: Optional[datetime] = None

        # Rate-limit outside-hours logs (global) to avoid overnight spam
        self._last_outside_hours_log_time: Optional[datetime] = None

        # Config thresholds
        self._delay_sec: float = max(0.0, self._cfg_float(("data", "feed_delay_threshold_ms"), default=1000.0, ms_to_sec=True))
        self._stale_sec: float = max(0.0, self._cfg_float(("data", "feed_stale_threshold_ms"), default=3000.0, ms_to_sec=True))
        self._down_sec: float = max(0.0, self._cfg_float(("data", "feed_down_threshold_ms"), default=5000.0, ms_to_sec=True))

        # Spike settings
        self._base_spike_pct_default: float = self._cfg_float(("tick_validator", "spike_pct"), default=2.0)
        self._spike_std_window: int = int(self._cfg_float(("tick_validator", "spike_std_window"), default=50.0))
        self._spike_std_min_count: int = int(self._cfg_float(("tick_validator", "spike_std_min_count"), default=20.0))
        self._spike_std_mult: float = self._cfg_float(("tick_validator", "spike_std_mult"), default=4.0)

        # Large gap baseline reset + reconnect guard
        self._reset_gap_sec: float = self._cfg_float(("tick_validator", "reset_gap_sec"), default=300.0)
        self._reconnect_guard_sec: float = self._cfg_float(("tick_validator", "reconnect_guard_sec"), default=10.0)

        # Future timestamp guard + latency stale flag
        self._future_guard_min: float = self._cfg_float(("tick_validator", "future_guard_minutes"), default=5.0)
        self._max_latency_sec: float = self._cfg_float(("tick_validator", "max_latency_sec"), default=5.0)

        self._emit_log(
            "info",
            "TickValidator initialized",
            delay_sec=self._delay_sec,
            stale_sec=self._stale_sec,
            down_sec=self._down_sec,
            reset_gap_sec=self._reset_gap_sec,
            reconnect_guard_sec=self._reconnect_guard_sec,
            base_spike_pct_default=self._base_spike_pct_default,
            spike_std_window=self._spike_std_window,
            spike_std_min_count=self._spike_std_min_count,
            spike_std_mult=self._spike_std_mult,
            future_guard_minutes=self._future_guard_min,
            max_latency_sec=self._max_latency_sec,
            spec_cache_ttl_sec=self._spec_ttl_sec,
            cb_threshold=self._cb_threshold,
            assume_naive_is_utc=self._assume_naive_is_utc,
            enforce_tick_size=self._enforce_tick_size,
            overflow_warning_interval_sec=self._overflow_warning_interval_sec,
            health_monitored_tokens=sorted(list(self._health_monitored_tokens)),
        )

    # ------------------------- Public API -------------------------

    def set_feed_health_callback(self, callback: Optional[Callable[..., None]]) -> None:
        """Callback invoked on feed health class changes for monitored tokens (F8)."""
        self._on_feed_health_change = callback

    def add_health_monitored_token(self, token: str) -> None:
        """F8: Add token to health monitored set."""
        try:
            t = str(token).strip()
            if not t:
                return
            with self._lock:
                self._health_monitored_tokens.add(t)
        except Exception:
            return

    def set_circuit_breaker_callback(self, callback: Optional[Callable[..., None]]) -> None:
        """F4: Set callback invoked on excessive consecutive rejections."""
        self._on_circuit_breaker = callback

    def reset_daily(self) -> None:
        with self._lock:
            self._states.clear()
            self._spec_cache.clear()

            self._total_seen = 0
            self._total_valid = 0

            self._rejected_token_missing = 0
            self._rejected_token_unknown = 0
            self._rejected_token_expired = 0

            self._rejected_missing_fields = 0
            self._rejected_bad_ltp = 0
            self._rejected_bad_timestamp = 0
            self._rejected_future_timestamp = 0
            self._rejected_non_trading_day = 0
            self._rejected_outside_hours = 0
            self._rejected_price_bounds = 0
            self._rejected_tick_size = 0
            self._rejected_out_of_order = 0
            self._rejected_true_duplicates = 0
            self._rejected_other = 0

            self._flagged_spikes = 0
            self._spike_baseline_resets = 0
            self._reconnect_bursts = 0

            self._consecutive_rejections = 0
            self._last_corruption_log_time = None
            self._last_outside_hours_log_time = None

            self._cb_last_call_time = None
            self._last_naive_ts_debug_time = None
            self._last_overflow_warn_by_token.clear()

        self._emit_log("info", "TickValidator daily reset complete")

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_seen": self._total_seen,
                "total_valid": self._total_valid,
                "rejected_token_missing": self._rejected_token_missing,
                "rejected_token_unknown": self._rejected_token_unknown,
                "rejected_token_expired": self._rejected_token_expired,
                "rejected_missing_fields": self._rejected_missing_fields,
                "rejected_bad_ltp": self._rejected_bad_ltp,
                "rejected_bad_timestamp": self._rejected_bad_timestamp,
                "rejected_future_timestamp": self._rejected_future_timestamp,
                "rejected_non_trading_day": self._rejected_non_trading_day,
                "rejected_outside_hours": self._rejected_outside_hours,
                "rejected_price_bounds": self._rejected_price_bounds,
                "rejected_tick_size": self._rejected_tick_size,
                "rejected_out_of_order": self._rejected_out_of_order,
                "rejected_true_duplicates": self._rejected_true_duplicates,
                "rejected_other": self._rejected_other,
                "flagged_spikes": self._flagged_spikes,
                "spike_baseline_resets": self._spike_baseline_resets,
                "reconnect_bursts": self._reconnect_bursts,
                "consecutive_rejections": self._consecutive_rejections,
                "tokens_tracked": len(self._states),
                "spec_cache_size": len(self._spec_cache),
                "health_monitored_tokens": len(self._health_monitored_tokens),
            }

    def validate(self, raw_tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate a raw tick dict. MUST NEVER RAISE.
        Returns validated tick dict or None.
        """
        now_ist = ist_now()

        reject_reason: Optional[str] = None
        reject_details: Dict[str, Any] = {}

        token: Optional[str] = None
        spec: Optional[InstrumentSpec] = None

        timestamp_ist: Optional[datetime] = None
        ltp: Optional[float] = None
        volume: int = -1

        tick_sz: Optional[float] = None

        # Pre-parse path (no state mutation)
        try:
            if not isinstance(raw_tick, dict):
                reject_reason = self._R_INVALID_TYPE
                reject_details = {"raw_type": str(type(raw_tick))}
            else:
                token = self._extract_token(raw_tick)
                if token is None:
                    reject_reason = self._R_TOKEN_MISSING
                    reject_details = {"keys": list(raw_tick.keys())[:50]}
                else:
                    # 2. Inside validate(), after extracting token string, do:
                    spec = self._get_instrument_spec(token)
                    if spec is None:
                        # Allow known spot/index tokens with a hardcoded default spec
                        if token == self._spot_token or (self._spot_token is not None and token == self._spot_token):
                            spec = self._default_index_spec_for_token(token)
                        else:
                            # still reject unknown tokens
                            reject_reason = self._R_TOKEN_UNKNOWN
                            reject_details = {"token": token}
                    if reject_reason is None:
                        ts_raw = self._extract_timestamp(raw_tick)
                        if ts_raw is None:
                            reject_reason = self._R_MISSING_FIELDS
                            reject_details = {"token": token, "missing": "exchange_timestamp"}
                        else:
                            timestamp_ist = self._parse_timestamp_to_ist(ts_raw, now_ist=now_ist)
                            if timestamp_ist is None:
                                reject_reason = self._R_BAD_TIMESTAMP
                                reject_details = {"token": token, "ts_raw": str(ts_raw)}
                            else:
                                # Expiry check if provided
                                if spec.expiry is not None and timestamp_ist.date() > spec.expiry:
                                    reject_reason = self._R_TOKEN_EXPIRED
                                    reject_details = {
                                        "token": token,
                                        "symbol": spec.symbol,
                                        "expiry": spec.expiry.isoformat(),
                                        "tick_date": timestamp_ist.date().isoformat(),
                                    }
                                else:
                                    # Future guard
                                    future_limit = now_ist + timedelta(minutes=self._future_guard_min)
                                    if timestamp_ist > future_limit:
                                        reject_reason = self._R_FUTURE_TIMESTAMP
                                        reject_details = {
                                            "token": token,
                                            "timestamp": timestamp_ist.isoformat(),
                                            "now": now_ist.isoformat(),
                                            "future_guard_min": self._future_guard_min,
                                        }
                                    else:
                                        # F3: holiday-aware trading day check
                                        if not self._is_trading_day_safe(timestamp_ist.date()):
                                            reject_reason = self._R_NON_TRADING_DAY
                                            reject_details = {"token": token, "date": timestamp_ist.date().isoformat()}
                                        else:
                                            # Market hours check
                                            if not is_market_hours(timestamp_ist):
                                                reject_reason = self._R_OUTSIDE_MARKET_HOURS
                                                reject_details = {"token": token, "timestamp": timestamp_ist.isoformat()}
                                            else:
                                                ltp_raw = self._extract_ltp(raw_tick)
                                                if ltp_raw is None:
                                                    reject_reason = self._R_MISSING_FIELDS
                                                    reject_details = {"token": token, "missing": "ltp"}
                                                else:
                                                    ltp = self._safe_float(ltp_raw)
                                                    if ltp is None:
                                                        reject_reason = self._R_BAD_LTP
                                                        reject_details = {"token": token, "ltp_raw": str(ltp_raw)}
                                                    else:
                                                        min_p, max_p, tick_sz = self._bounds_and_tick_size(spec)
                                                        if min_p is not None and ltp < min_p:
                                                            reject_reason = self._R_PRICE_OUT_OF_BOUNDS
                                                            reject_details = {"token": token, "symbol": spec.symbol, "ltp": float(ltp), "min_price": float(min_p)}
                                                        elif max_p is not None and ltp > max_p:
                                                            reject_reason = self._R_PRICE_OUT_OF_BOUNDS
                                                            reject_details = {"token": token, "symbol": spec.symbol, "ltp": float(ltp), "max_price": float(max_p)}
                                                        else:
                                                            # F7: tick size multiple validation
                                                            if self._enforce_tick_size and tick_sz is not None and tick_sz > 0:
                                                                if not self._is_multiple_of_tick_size(float(ltp), float(tick_sz)):
                                                                    reject_reason = self._R_PRICE_NOT_MULTIPLE_TICK
                                                                    reject_details = {"token": token, "symbol": spec.symbol, "ltp": float(ltp), "tick_size": float(tick_sz)}
                                                            if reject_reason is None:
                                                                # Volume semantics: missing volume -> -1
                                                                vol_raw, vol_present = self._extract_volume_with_presence(raw_tick)
                                                                if not vol_present:
                                                                    volume = -1
                                                                else:
                                                                    volume = self._safe_int_volume(vol_raw, token=token, now_ist=now_ist, default=-1)
                                                                    if volume < 0:
                                                                        volume = 0
        except Exception as e:
            reject_reason = self._R_OTHER
            reject_details = {"error": str(e)}

        # Stateful section
        feed_health_callback_event: Optional[Tuple[str, str, str]] = None  # (token, old, new)
        cb_event: Optional[Dict[str, Any]] = None
        corruption_payload: Optional[Dict[str, Any]] = None
        outside_hours_should_log: bool = False

        same_timestamp_update: bool = False
        reconnect_burst: bool = False
        dynamic_threshold_pct: float = self._base_spike_pct_default
        is_spike: bool = False
        feed_gap_sec: float = 0.0
        feed_health: str = "FIRST_TICK"
        latency_sec: float = 0.0
        is_stale: bool = False

        return_out: Optional[Dict[str, Any]] = None

        try:
            with self._lock:
                self._total_seen += 1

                if reject_reason is not None:
                    self._consecutive_rejections += 1

                    if reject_reason == self._R_TOKEN_MISSING:
                        self._rejected_token_missing += 1
                    elif reject_reason == self._R_TOKEN_UNKNOWN:
                        self._rejected_token_unknown += 1
                    elif reject_reason == self._R_TOKEN_EXPIRED:
                        self._rejected_token_expired += 1
                    elif reject_reason == self._R_MISSING_FIELDS:
                        self._rejected_missing_fields += 1
                    elif reject_reason == self._R_BAD_LTP:
                        self._rejected_bad_ltp += 1
                    elif reject_reason == self._R_BAD_TIMESTAMP:
                        self._rejected_bad_timestamp += 1
                    elif reject_reason == self._R_FUTURE_TIMESTAMP:
                        self._rejected_future_timestamp += 1
                    elif reject_reason == self._R_NON_TRADING_DAY:
                        self._rejected_non_trading_day += 1
                    elif reject_reason == self._R_OUTSIDE_MARKET_HOURS:
                        self._rejected_outside_hours += 1
                        if self._last_outside_hours_log_time is None or (now_ist - self._last_outside_hours_log_time).total_seconds() >= 60.0:
                            self._last_outside_hours_log_time = now_ist
                            outside_hours_should_log = True
                    elif reject_reason == self._R_PRICE_OUT_OF_BOUNDS:
                        self._rejected_price_bounds += 1
                    elif reject_reason == self._R_PRICE_NOT_MULTIPLE_TICK:
                        self._rejected_tick_size += 1
                    elif reject_reason == self._R_OUT_OF_ORDER:
                        self._rejected_out_of_order += 1
                    elif reject_reason == self._R_TRUE_DUPLICATE:
                        self._rejected_true_duplicates += 1
                    else:
                        self._rejected_other += 1

                    # F4: circuit breaker on excessive rejections
                    if self._cb_threshold > 0 and self._consecutive_rejections >= self._cb_threshold and self._on_circuit_breaker is not None:
                        if self._cb_last_call_time is None or (now_ist - self._cb_last_call_time).total_seconds() >= self._cb_call_interval_sec:
                            self._cb_last_call_time = now_ist
                            cb_event = {"event": "excessive_rejections", "count": int(self._consecutive_rejections)}

                    if reject_reason not in (self._R_OUTSIDE_MARKET_HOURS, self._R_NON_TRADING_DAY):
                        corruption_payload = self._maybe_corruption_payload_locked(now_ist)

                    return_out = None

                else:
                    assert token is not None and spec is not None and timestamp_ist is not None and ltp is not None

                    state = self._states.get(token)
                    if state is None:
                        state = _PerTokenState()
                        self._states[token] = state

                    latency_sec = max(0.0, (now_ist - timestamp_ist).total_seconds())
                    is_stale = bool(latency_sec > self._max_latency_sec)

                    prev_ts = state.last_timestamp
                    prev_price = state.last_price
                    prev_volume = state.last_volume

                    # Ordering / duplicate tolerance
                    if prev_ts is not None:
                        if timestamp_ist < prev_ts:
                            self._consecutive_rejections += 1
                            self._rejected_out_of_order += 1
                            corruption_payload = self._maybe_corruption_payload_locked(now_ist)
                            return_out = None
                        elif timestamp_ist == prev_ts:
                            price_changed = (prev_price is None) or (float(ltp) != float(prev_price))
                            volume_changed = (prev_volume is None) or (int(volume) != int(prev_volume))
                            if price_changed or volume_changed:
                                same_timestamp_update = True
                            else:
                                self._consecutive_rejections += 1
                                self._rejected_true_duplicates += 1
                                corruption_payload = self._maybe_corruption_payload_locked(now_ist)
                                return_out = None

                    if return_out is None and prev_ts is not None and timestamp_ist == prev_ts and not same_timestamp_update:
                        # stop early for true duplicate
                        pass
                    else:
                        # Gap classification
                        if prev_ts is None:
                            feed_gap_sec = 0.0
                            feed_health = "FIRST_TICK"
                        else:
                            feed_gap_sec = max(0.0, (timestamp_ist - prev_ts).total_seconds())
                            feed_health = self._classify_gap(feed_gap_sec)

                        # Reconnect burst (do not reject, but annotate)
                        if prev_ts is not None and feed_gap_sec > self._reconnect_guard_sec:
                            reconnect_burst = True
                            self._reconnect_bursts += 1
                            prev_price = None  # avoid spike compare to old baseline

                        # F1: Spike baseline reset timing (IMMEDIATE before processing current tick)
                        if prev_ts is not None and feed_gap_sec > self._reset_gap_sec:
                            self._spike_baseline_resets += 1
                            # Reset state baseline immediately so current tick is not spike-compared against old baseline.
                            state.last_price = None
                            prev_price = None

                        # Dynamic spike threshold
                        dynamic_threshold_pct = self._dynamic_spike_threshold_pct_locked(spec, state)

                        # Spike detection
                        is_spike = False
                        if prev_price is not None and prev_price > 0:
                            jump_pct = abs(float(ltp) - float(prev_price)) / float(prev_price) * 100.0
                            if jump_pct > dynamic_threshold_pct:
                                is_spike = True
                                self._flagged_spikes += 1

                        # Update adaptive stats window only when we have a real baseline compare
                        if prev_price is not None and prev_price > 0:
                            chg_pct = abs(float(ltp) - float(prev_price)) / float(prev_price) * 100.0
                            self._update_return_window_locked(state, chg_pct)

                        # Update state for next tick baseline
                        state.last_timestamp = timestamp_ist
                        state.last_price = float(ltp)
                        state.last_volume = int(volume)

                        # F8: feed health callback for any monitored token
                        if token in self._health_monitored_tokens:
                            old_h = state.last_feed_health
                            if old_h is None:
                                state.last_feed_health = feed_health
                            else:
                                if feed_health != old_h:
                                    state.last_feed_health = feed_health
                                    feed_health_callback_event = (token, old_h, feed_health)

                        self._total_valid += 1
                        self._consecutive_rejections = 0

                        return_out = ValidatedTick(
                            timestamp=timestamp_ist,
                            token=token,
                            symbol=spec.symbol,
                            instrument_class=spec.instrument_class,
                            ltp=float(ltp),
                            volume=int(volume),
                            is_spike=bool(is_spike),
                            feed_gap_sec=float(feed_gap_sec),
                            feed_health=str(feed_health),
                            latency_sec=float(latency_sec),
                            is_stale=bool(is_stale),
                            reconnect_burst=bool(reconnect_burst),
                            same_timestamp_update=bool(same_timestamp_update),
                            tick_size=tick_sz,
                            dynamic_spike_threshold_pct=float(dynamic_threshold_pct),
                        ).as_dict()

        except Exception as e:  # pragma: no cover
            self._emit_exception("TickValidator internal failure", error=str(e))
            return None

        # Callbacks + logging outside lock
        if reject_reason is not None:
            if reject_reason == self._R_OUTSIDE_MARKET_HOURS:
                if outside_hours_should_log:
                    self._emit_log("debug", "Tick rejected: outside market hours", **reject_details)
                return None

            if reject_reason == self._R_NON_TRADING_DAY:
                self._emit_log("warning", "Tick rejected: non trading day", reason=reject_reason, **reject_details)
                # do not treat as corruption
                return None

            if reject_reason == self._R_FUTURE_TIMESTAMP:
                self._emit_log("critical", "Tick rejected: future timestamp beyond guard", **reject_details)
            elif reject_reason in (self._R_TOKEN_MISSING, self._R_TOKEN_UNKNOWN, self._R_TOKEN_EXPIRED):
                self._emit_log("error", "Tick rejected: token/spec validation failed", reason=reject_reason, **reject_details)
            elif reject_reason == self._R_PRICE_OUT_OF_BOUNDS:
                self._emit_log("error", "Tick rejected: price out of bounds", reason=reject_reason, **reject_details)
            elif reject_reason == self._R_PRICE_NOT_MULTIPLE_TICK:
                self._emit_log("error", "Tick rejected: price not multiple of tick size", reason=reject_reason, **reject_details)
            elif reject_reason in (self._R_BAD_TIMESTAMP, self._R_BAD_LTP, self._R_MISSING_FIELDS):
                self._emit_log("error", "Tick rejected during parsing", reason=reject_reason, **reject_details)
            elif reject_reason in (self._R_OUT_OF_ORDER, self._R_TRUE_DUPLICATE):
                self._emit_log("warning", "Tick rejected: ordering/duplicate", reason=reject_reason, **reject_details)
            else:
                self._emit_log("error", "Tick rejected: other", reason=reject_reason, **reject_details)

        if feed_health_callback_event is not None and self._on_feed_health_change is not None:
            tok, old_h, new_h = feed_health_callback_event
            try:
                self._on_feed_health_change(token=tok, old=old_h, new=new_h)  # type: ignore[misc]
            except TypeError:
                try:
                    self._on_feed_health_change(tok, old_h, new_h)  # type: ignore[misc]
                except Exception:
                    pass
            except Exception:
                pass

        if cb_event is not None and self._on_circuit_breaker is not None:
            try:
                self._on_circuit_breaker(cb_event["event"], count=cb_event["count"])  # type: ignore[misc]
            except TypeError:
                try:
                    self._on_circuit_breaker(cb_event["event"])  # type: ignore[misc]
                except Exception:
                    pass
            except Exception:
                pass

        if corruption_payload is not None:
            self._emit_log("critical", "Possible feed corruption (consecutive rejections > 10)", **corruption_payload)

        if return_out is not None and return_out.get("is_spike", False):
            self._emit_log(
                "warning",
                "Price spike detected (tick accepted but flagged)",
                token=return_out.get("token"),
                symbol=return_out.get("symbol"),
                ltp=return_out.get("ltp"),
                dynamic_spike_threshold_pct=return_out.get("dynamic_spike_threshold_pct"),
                timestamp=return_out.get("timestamp").isoformat() if isinstance(return_out.get("timestamp"), datetime) else str(return_out.get("timestamp")),
                feed_health=return_out.get("feed_health"),
                reconnect_burst=return_out.get("reconnect_burst"),
            )

        return return_out

    # ------------------------- Builder additions -------------------------

    def _get_instrument_spec(self, token: str) -> Optional[InstrumentSpec]:
        """
        Wrapper for spec resolution (used by validate()).
        """
        try:
            tok = str(token).strip()
            if not tok:
                return None
            return self._resolve_spec(tok, now_ist=ist_now())
        except Exception:
            return None

    def _default_index_spec_for_token(self, token: str) -> InstrumentSpec:
        """
        Converts _DEFAULT_INDEX_SPEC dict into InstrumentSpec for the given token.
        """
        d = self._DEFAULT_INDEX_SPEC
        return InstrumentSpec(
            token=str(token),
            symbol=str(d.get("symbol", "NIFTY")),
            instrument_class=str(d.get("instrument_class", "INDEX")).upper(),
            min_price=float(d.get("min_price", 10000.0)),
            max_price=float(d.get("max_price", 50000.0)),
            tick_size=float(d.get("tick_size", 0.05)),
            expiry=None,
        )

    # ------------------------- F2 Spec cache resolution -------------------------

    def _resolve_spec(self, token: str, now_ist: datetime) -> Optional[InstrumentSpec]:
        # Cache hit
        try:
            entry = self._spec_cache.get(token)
            if entry is not None:
                age = (now_ist - entry.cached_at_ist).total_seconds()
                if age < float(self._spec_ttl_sec):
                    return entry.spec
        except Exception:
            pass

        # Cache miss or stale; query mapper
        resolved: Optional[InstrumentSpec] = None
        mapper_error: Optional[str] = None
        try:
            resolved = self._resolve_spec_from_mapper(token)
        except Exception as e:
            mapper_error = str(e)
            resolved = None

        if resolved is not None:
            # Update cache
            try:
                self._spec_cache[token] = _CachedSpecEntry(spec=resolved, cached_at_ist=now_ist)
            except Exception:
                pass
            return resolved

        # Mapper returned None or errored; if stale cache exists, use it (do not reject tick)
        if entry is not None:
            self._emit_log(
                "warning",
                "Instrument spec resolution failed; using stale cached spec",
                token=token,
                mapper_error=mapper_error,
                cache_age_sec=round((now_ist - entry.cached_at_ist).total_seconds(), 3),
                ttl_sec=self._spec_ttl_sec,
            )
            return entry.spec

        return None

    def _resolve_spec_from_mapper(self, token: str) -> Optional[InstrumentSpec]:
        """
        Best-effort resolution of token -> InstrumentSpec from instrument_mapper.
        MAY raise; caller handles.
        """
        m = self._instrument_mapper
        candidate: Any = None

        if hasattr(m, "get_instrument_spec") and callable(getattr(m, "get_instrument_spec")):
            candidate = m.get_instrument_spec(token)
        elif hasattr(m, "get_spec") and callable(getattr(m, "get_spec")):
            candidate = m.get_spec(token)
        else:
            for attr in ("token_specs", "token_spec_map", "token_map", "instrument_specs"):
                if hasattr(m, attr):
                    mp = getattr(m, attr)
                    if isinstance(mp, dict):
                        candidate = mp.get(token)
                        break

        if candidate is None:
            return None

        if isinstance(candidate, InstrumentSpec):
            return candidate

        if isinstance(candidate, dict):
            symbol = str(candidate.get("symbol") or candidate.get("tradingsymbol") or candidate.get("name") or token)
            instrument_class = str(candidate.get("instrument_class") or candidate.get("type") or candidate.get("segment") or "UNKNOWN").upper()

            min_price = self._safe_float(candidate.get("min_price"))
            max_price = self._safe_float(candidate.get("max_price"))
            tick_size = self._safe_float(candidate.get("tick_size"))

            expiry_val = candidate.get("expiry") or candidate.get("expiry_date")
            expiry_dt = self._parse_expiry_date(expiry_val)

            return InstrumentSpec(
                token=str(token),
                symbol=symbol,
                instrument_class=instrument_class,
                min_price=min_price,
                max_price=max_price,
                tick_size=tick_size,
                expiry=expiry_dt,
            )

        return None

    @staticmethod
    def _parse_expiry_date(x: Any) -> Optional[date]:
        try:
            if x is None:
                return None
            if isinstance(x, date) and not isinstance(x, datetime):
                return x
            if isinstance(x, datetime):
                return x.date()
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return None
                try:
                    dt = datetime.fromisoformat(s)
                    return dt.date()
                except Exception:
                    pass
                for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
                    try:
                        dt2 = datetime.strptime(s, fmt)
                        return dt2.date()
                    except Exception:
                        continue
            return None
        except Exception:
            return None

    # ------------------------- Bounds + Tick size defaults -------------------------

    def _bounds_and_tick_size(self, spec: InstrumentSpec) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        min_p, max_p = spec.bounds()
        tick_sz = spec.tick_size

        if min_p is None or max_p is None or tick_sz is None:
            cls = (spec.instrument_class or "UNKNOWN").upper()
            sym = (spec.symbol or "").upper()

            if "BANKNIFTY" in sym:
                dmin = self._cfg_float(("tick_validator", "bounds_banknifty_min"), default=1000.0)
                dmax = self._cfg_float(("tick_validator", "bounds_banknifty_max"), default=100000.0)
                dtick = self._cfg_float(("tick_validator", "tick_banknifty"), default=0.05)
            elif "FINNIFTY" in sym:
                dmin = self._cfg_float(("tick_validator", "bounds_finnifty_min"), default=1000.0)
                dmax = self._cfg_float(("tick_validator", "bounds_finnifty_max"), default=100000.0)
                dtick = self._cfg_float(("tick_validator", "tick_finnifty"), default=0.05)
            elif "NIFTY" in sym:
                dmin = self._cfg_float(("tick_validator", "bounds_nifty_min"), default=10000.0)
                dmax = self._cfg_float(("tick_validator", "bounds_nifty_max"), default=50000.0)
                dtick = self._cfg_float(("tick_validator", "tick_nifty"), default=0.05)
            else:
                if cls == "STOCK":
                    dmin = self._cfg_float(("tick_validator", "bounds_stock_min"), default=1.0)
                    dmax = self._cfg_float(("tick_validator", "bounds_stock_max"), default=500000.0)
                    dtick = self._cfg_float(("tick_validator", "tick_stock"), default=0.05)
                elif cls in ("FUTURE", "FUT"):
                    dmin = self._cfg_float(("tick_validator", "bounds_future_min"), default=1.0)
                    dmax = self._cfg_float(("tick_validator", "bounds_future_max"), default=1000000.0)
                    dtick = self._cfg_float(("tick_validator", "tick_future"), default=0.05)
                elif cls in ("OPTION", "OPT"):
                    dmin = self._cfg_float(("tick_validator", "bounds_option_min"), default=0.0)
                    dmax = self._cfg_float(("tick_validator", "bounds_option_max"), default=1000000.0)
                    dtick = self._cfg_float(("tick_validator", "tick_option"), default=0.05)
                else:
                    dmin = self._cfg_float(("tick_validator", "bounds_index_min"), default=1.0)
                    dmax = self._cfg_float(("tick_validator", "bounds_index_max"), default=1000000.0)
                    dtick = self._cfg_float(("tick_validator", "tick_index"), default=0.05)

            if min_p is None:
                min_p = dmin
            if max_p is None:
                max_p = dmax
            if tick_sz is None:
                tick_sz = dtick

        return min_p, max_p, tick_sz

    # ------------------------- Adaptive spike threshold -------------------------

    def _dynamic_spike_threshold_pct_locked(self, spec: InstrumentSpec, state: _PerTokenState) -> float:
        base = self._base_spike_pct_default

        cls = (spec.instrument_class or "UNKNOWN").upper()
        mult = 1.0
        if cls == "INDEX":
            mult = self._cfg_float(("tick_validator", "spike_mult_index"), default=1.0)
        elif cls == "STOCK":
            mult = self._cfg_float(("tick_validator", "spike_mult_stock"), default=1.0)
        elif cls in ("FUTURE", "FUT"):
            mult = self._cfg_float(("tick_validator", "spike_mult_future"), default=1.0)
        elif cls in ("OPTION", "OPT"):
            mult = self._cfg_float(("tick_validator", "spike_mult_option"), default=1.0)
        else:
            mult = self._cfg_float(("tick_validator", "spike_mult_unknown"), default=1.0)

        base *= float(mult)

        n = len(state.ret_window)
        if n < self._spike_std_min_count:
            return max(0.0, base)

        mean = state.ret_sum / max(n, 1)
        var = max(0.0, (state.ret_sumsq / max(n, 1)) - (mean * mean))
        std = math.sqrt(var)

        adaptive = float(self._spike_std_mult) * std
        return max(0.0, max(base, adaptive))

    def _update_return_window_locked(self, state: _PerTokenState, chg_pct: float) -> None:
        try:
            if self._spike_std_window <= 1:
                return
            w = state.ret_window

            while len(w) >= self._spike_std_window:
                old = w.popleft()
                state.ret_sum -= old
                state.ret_sumsq -= old * old

            v = float(chg_pct)
            if not math.isfinite(v) or v < 0:
                return

            w.append(v)
            state.ret_sum += v
            state.ret_sumsq += v * v
        except Exception:
            return

    # ------------------------- Feed health helpers -------------------------

    def _classify_gap(self, feed_gap_sec: float) -> str:
        if feed_gap_sec > self._down_sec:
            return "DOWN"
        if feed_gap_sec > self._stale_sec:
            return "STALE"
        if feed_gap_sec > self._delay_sec:
            return "DELAYED"
        return "HEALTHY"

    def _maybe_corruption_payload_locked(self, now_ist: datetime) -> Optional[Dict[str, Any]]:
        if self._consecutive_rejections <= 10:
            return None

        if self._last_corruption_log_time is None:
            self._last_corruption_log_time = now_ist
            return {"consecutive_rejections": self._consecutive_rejections, "timestamp": now_ist.isoformat()}

        if (now_ist - self._last_corruption_log_time).total_seconds() >= 60.0:
            self._last_corruption_log_time = now_ist
            return {"consecutive_rejections": self._consecutive_rejections, "timestamp": now_ist.isoformat()}

        return None

    # ------------------------- Parsing helpers -------------------------

    @staticmethod
    def _extract_token(raw_tick: Dict[str, Any]) -> Optional[str]:
        try:
            tok = raw_tick.get("token", raw_tick.get("symboltoken", raw_tick.get("symbolToken", raw_tick.get("instrument_token"))))
            if tok is None:
                return None
            s = str(tok).strip()
            return s if s else None
        except Exception:
            return None

    @staticmethod
    def _extract_ltp(raw_tick: Dict[str, Any]) -> Any:
        if "ltp" in raw_tick:
            return raw_tick.get("ltp")
        return raw_tick.get("LTP", raw_tick.get("last_price", raw_tick.get("lastPrice", raw_tick.get("price"))))

    @staticmethod
    def _extract_timestamp(raw_tick: Dict[str, Any]) -> Any:
        if "exchange_timestamp" in raw_tick:
            return raw_tick.get("exchange_timestamp")
        return raw_tick.get("exchangeTime", raw_tick.get("timestamp", raw_tick.get("time")))

    @staticmethod
    def _extract_volume_with_presence(raw_tick: Dict[str, Any]) -> Tuple[Any, bool]:
        for k in ("volume", "vol", "v", "tradedVolume", "qty"):
            if k in raw_tick:
                val = raw_tick.get(k)
                if val is None:
                    return None, False
                return val, True
        return None, False

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                f = float(x)
                if f != f:
                    return None
                return f
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return None
                s = s.replace(",", "")
                f = float(s)
                if f != f:
                    return None
                return f
            f = float(x)
            if f != f:
                return None
            return f
        except Exception:
            return None

    def _safe_int_volume(self, x: Any, *, token: str, now_ist: datetime, default: int = -1) -> int:
        """
        F5: Volume overflow warning + clamp.
        - If value exceeds INT32_MAX OR OverflowError occurs => warn (rate-limited per token) and clamp.
        """
        if x is None:
            return int(default)

        raw_preview = None
        try:
            raw_preview = str(x)[:120]
        except Exception:
            raw_preview = "<unrepr>"

        def _warn_once() -> None:
            last = self._last_overflow_warn_by_token.get(token)
            if last is None or (now_ist - last).total_seconds() >= self._overflow_warning_interval_sec:
                self._last_overflow_warn_by_token[token] = now_ist
                self._emit_log(
                    "warning",
                    "Volume overflow/clamp detected",
                    token=token,
                    raw_value=raw_preview,
                    clamp_to=_INT32_MAX,
                )

        try:
            if isinstance(x, bool):
                return int(default)

            if isinstance(x, int):
                if x > _INT32_MAX:
                    _warn_once()
                    return _INT32_MAX
                if x < -_INT32_MAX:
                    _warn_once()
                    return -_INT32_MAX
                return int(x)

            if isinstance(x, float):
                if x != x:
                    return int(default)
                if x > _INT32_MAX:
                    _warn_once()
                    return _INT32_MAX
                if x < -_INT32_MAX:
                    _warn_once()
                    return -_INT32_MAX
                return int(x)

            if isinstance(x, str):
                s = x.strip().replace(",", "")
                if not s:
                    return int(default)

                sign = -1 if s.startswith("-") else 1
                body = s[1:] if s[0] in "+-" else s
                # Extremely long numeric strings => overflow
                if body.isdigit() and len(body) > 10:
                    _warn_once()
                    return _INT32_MAX * sign

                try:
                    v = int(s)
                except ValueError:
                    v = int(float(s))
                except OverflowError:
                    _warn_once()
                    return _INT32_MAX * sign

                if v > _INT32_MAX:
                    _warn_once()
                    return _INT32_MAX
                if v < -_INT32_MAX:
                    _warn_once()
                    return -_INT32_MAX
                return int(v)

            v = int(x)
            if v > _INT32_MAX:
                _warn_once()
                return _INT32_MAX
            if v < -_INT32_MAX:
                _warn_once()
                return -_INT32_MAX
            return int(v)

        except OverflowError:
            _warn_once()
            return _INT32_MAX
        except Exception:
            return int(default)

    def _cfg_float(self, *paths: Tuple[str, str], default: float, ms_to_sec: bool = False) -> float:
        val: Any = None
        for section, key in paths:
            v = Config.get(section, key, default=None)
            if v is not None:
                val = v
                break
        if val is None:
            val = default
        f = self._safe_float(val)
        if f is None:
            f = float(default)
        return float(f) / 1000.0 if ms_to_sec else float(f)

    def _parse_timestamp_to_ist(self, ts_raw: Any, *, now_ist: datetime) -> Optional[datetime]:
        """
        F6: naive timestamp assumption warning + config assume_naive_is_utc.

        - If naive encountered, logs debug:
          "Assuming naive timestamp is IST; if feed uses UTC, set assume_naive_is_utc=True in config"
        - If assume_naive_is_utc=True, treat naive as UTC and convert to IST.
        """
        try:
            if ts_raw is None:
                return None

            def _debug_naive_once() -> None:
                if self._last_naive_ts_debug_time is None or (now_ist - self._last_naive_ts_debug_time).total_seconds() >= self._naive_ts_debug_interval_sec:
                    self._last_naive_ts_debug_time = now_ist
                    self._emit_log(
                        "debug",
                        "Assuming naive timestamp is IST; if feed uses UTC, set assume_naive_is_utc=True in config",
                        assume_naive_is_utc=self._assume_naive_is_utc,
                    )

            if isinstance(ts_raw, datetime):
                if ts_raw.tzinfo is None:
                    _debug_naive_once()
                    if self._assume_naive_is_utc:
                        return ts_raw.replace(tzinfo=timezone.utc).astimezone(IST)
                    return ts_raw.replace(tzinfo=IST)
                return ts_raw.astimezone(IST)

            # epoch numeric
            if isinstance(ts_raw, (int, float)):
                try:
                    iv = int(ts_raw)
                except Exception:
                    return None
                if iv <= 0:
                    return None
                digits = len(str(abs(iv)))
                if digits == 13:
                    sec = iv / 1000.0
                elif digits == 10:
                    sec = float(iv)
                else:
                    return None
                return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(IST)

            # string
            if isinstance(ts_raw, str):
                s = ts_raw.strip()
                if not s:
                    return None

                if s.isdigit():
                    if len(s) == 13:
                        sec = int(s) / 1000.0
                    elif len(s) == 10:
                        sec = float(int(s))
                    else:
                        return None
                    return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(IST)

                s_iso = s.replace("Z", "+00:00")
                dt: Optional[datetime] = None
                try:
                    dt = datetime.fromisoformat(s_iso)
                except Exception:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                        try:
                            dt = datetime.strptime(s, fmt)
                            break
                        except Exception:
                            dt = None
                    if dt is None:
                        return None

                if dt.tzinfo is None:
                    _debug_naive_once()
                    if self._assume_naive_is_utc:
                        return dt.replace(tzinfo=timezone.utc).astimezone(IST)
                    return dt.replace(tzinfo=IST)
                return dt.astimezone(IST)

            return None
        except Exception:
            return None

    # ------------------------- Trading day helper -------------------------

    def _is_trading_day_safe(self, d: date) -> bool:
        """
        F3: holiday-aware trading day check.
        If helpers.is_trading_day exists, use it.
        Else fallback to weekend + economic_calendar.json severity=0 holiday events.
        """
        try:
            if is_trading_day is not None:
                return bool(is_trading_day(d))  # type: ignore[misc]
        except Exception:
            pass

        # fallback: weekend check + optional holiday calendar JSON (best-effort)
        try:
            if d.weekday() >= 5:
                return False
            cal_path = str(Config.get("historical", "economic_calendar_path", default="data/calendar/economic_calendar.json"))
            cal_path = cal_path.replace("\\", "/")
            if os.path.exists(cal_path):
                with open(cal_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                events = obj.get("events", [])
                if isinstance(events, list):
                    ds = d.isoformat()
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        if str(ev.get("date", "")).strip() == ds:
                            # treat severity 0 as holiday (as per sample calendar)
                            sev = ev.get("severity", None)
                            name = str(ev.get("name", "")).lower()
                            if sev == 0 or "holiday" in name:
                                return False
            return True
        except Exception:
            # safest fallback: weekend only
            return d.weekday() < 5

    # ------------------------- Tick size multiple check -------------------------

    @staticmethod
    def _is_multiple_of_tick_size(ltp: float, tick_sz: float) -> bool:
        """
        F7: tick size validation using modulo remainder tolerance.
        """
        if tick_sz <= 0:
            return True
        # floating-safe remainder
        r = abs(math.fmod(ltp, tick_sz))
        # if close to 0 or close to tick_sz, treat as multiple
        eps = 1e-9
        if r <= eps or (tick_sz - r) <= eps:
            return True
        # reject only if genuinely between
        if r > eps and r < (tick_sz - eps):
            return False
        return True

    # ------------------------- Logging shim -------------------------

    def _emit_log(self, level: str, msg: str, **fields: Any) -> None:
        try:
            fn = getattr(self._log, level, None)
            if fn is None:
                return
            try:
                fn(msg, **fields)
            except TypeError:
                # fallback stringify
                parts = []
                for k in sorted(fields.keys()):
                    try:
                        parts.append(f"{k}={fields[k]!r}")
                    except Exception:
                        parts.append(f"{k}=<unrepr>")
                fn(f"{msg} | " + ", ".join(parts))
        except Exception:
            return

    def _emit_exception(self, msg: str, **fields: Any) -> None:
        try:
            fn = getattr(self._log, "exception", None)
            if fn is None:
                self._emit_log("error", msg, **fields)
                return
            try:
                fn(msg, **fields)
            except TypeError:
                parts = []
                for k in sorted(fields.keys()):
                    try:
                        parts.append(f"{k}={fields[k]!r}")
                    except Exception:
                        parts.append(f"{k}=<unrepr>")
                fn(f"{msg} | " + ", ".join(parts))
        except Exception:
            return


# ------------------------- Self-test (Acceptance) -------------------------

def _dt_to_epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _find_recent_in_market_datetime(max_days_back: int = 14) -> datetime:
    """
    Find a recent IST date at 10:00 that is in market hours and trading day (best-effort).
    """
    base = ist_now()
    for i in range(max_days_back + 1):
        d = (base - timedelta(days=i)).date()
        dt = datetime(d.year, d.month, d.day, 10, 0, 0, tzinfo=IST)
        try:
            if is_market_hours(dt):
                return dt
        except Exception:
            continue
    # fallback: today 10:00
    d = base.date()
    return datetime(d.year, d.month, d.day, 10, 0, 0, tzinfo=IST)


if __name__ == "__main__":
    class _MockInstrumentMapper:
        def __init__(self):
            self.broken = False

        def get_instrument_spec(self, token: str) -> Dict[str, Any]:
            if self.broken:
                raise RuntimeError("Mapper intentionally broken")
            if str(token) == "99926000":
                return {
                    "symbol": "NIFTY",
                    "instrument_class": "INDEX",
                    "min_price": 10000.0,
                    "max_price": 50000.0,
                    "tick_size": 0.05,
                    "expiry": None,
                }
            return None

    print("Running TickValidator institutional hardening self-test...\n")

    mapper = _MockInstrumentMapper()
    tv = TickValidator(mapper)

    # F8: health monitored token
    tv.add_health_monitored_token("99926000")

    health_events: List[Tuple[str, str, str]] = []

    def feed_health_cb(*args, **kwargs):
        # expected call: token, old, new OR kwargs
        tok = kwargs.get("token") or (args[0] if len(args) > 0 else "?")
        old = kwargs.get("old") or (args[1] if len(args) > 1 else "?")
        new = kwargs.get("new") or (args[2] if len(args) > 2 else "?")
        health_events.append((str(tok), str(old), str(new)))

    cb_events: List[Dict[str, Any]] = []

    def circuit_cb(event: str, **kwargs):
        cb_events.append({"event": event, **kwargs})

    tv.set_feed_health_callback(feed_health_cb)
    tv.set_circuit_breaker_callback(circuit_cb)

    # Base time for ticks
    t0 = _find_recent_in_market_datetime()
    token = "99926000"

    # 1) F1 Spike baseline reset: gap > 5 min should not flag spike on first tick after gap
    tick1 = {"token": token, "ltp": 24500.00, "volume": 100, "exchange_timestamp": _dt_to_epoch_ms(t0)}
    out1 = tv.validate(tick1)
    gap_ts = t0 + timedelta(minutes=6)  # > reset_gap_sec default 300s
    tick2 = {"token": token, "ltp": 25000.00, "volume": 100, "exchange_timestamp": _dt_to_epoch_ms(gap_ts)}
    out2 = tv.validate(tick2)
    print("TEST1 Spike baseline reset:", "PASS" if (out2 and out2["is_spike"] is False) else "FAIL", "| is_spike=", (out2["is_spike"] if out2 else None))

    # 2) F3 Holiday rejection (uses calendar fallback if helpers missing)
    holiday_dt = datetime(2026, 4, 14, 10, 0, 0, tzinfo=IST)
    tick_h = {"token": token, "ltp": 24500.00, "volume": 10, "exchange_timestamp": _dt_to_epoch_ms(holiday_dt)}
    out_h = tv.validate(tick_h)
    # We can't directly read reject reason from output, but should be None; logs will show reason.
    print("TEST2 Holiday rejection:", "PASS" if out_h is None else "FAIL")

    # 3) F2 Cache fallback: populate cache, expire it, break mapper, still validate using stale cached spec
    # Force cache entry old
    with tv._lock:
        entry = tv._spec_cache.get(token)
        if entry:
            tv._spec_cache[token] = _CachedSpecEntry(spec=entry.spec, cached_at_ist=ist_now() - timedelta(seconds=tv._spec_ttl_sec + 1))
    mapper.broken = True
    tick3 = {"token": token, "ltp": 24501.00, "volume": 10, "exchange_timestamp": _dt_to_epoch_ms(t0 + timedelta(seconds=2))}
    out3 = tv.validate(tick3)
    print("TEST3 Spec cache stale fallback:", "PASS" if (out3 is not None) else "FAIL")

    # 4) F4 Circuit breaker after 50 consecutive rejections
    mapper.broken = False
    tv.reset_daily()
    tv.set_circuit_breaker_callback(circuit_cb)
    # produce 50 rejects by missing token
    for _ in range(int(Config.get("tick_validator", "cb_threshold", default=50))):
        tv.validate({"ltp": 1, "exchange_timestamp": _dt_to_epoch_ms(t0)})
    print("TEST4 Circuit breaker callback:", "PASS" if any(e.get("event") == "excessive_rejections" for e in cb_events) else "FAIL", "| events=", cb_events[-1] if cb_events else None)

    # 5) F5 Volume overflow clamp warning (hard to assert log; assert clamp)
    tv.reset_daily()
    huge_vol = "9" * 30
    tick5 = {"token": token, "ltp": 24500.00, "volume": huge_vol, "exchange_timestamp": _dt_to_epoch_ms(t0)}
    out5 = tv.validate(tick5)
    print("TEST5 Volume overflow clamp:", "PASS" if (out5 and out5["volume"] == _INT32_MAX) else "FAIL", "| volume=", (out5["volume"] if out5 else None))

    # 6) F7 Tick size multiple rejection
    # Use naive ISO timestamp (F6) so it parses; date is t0 date at 10:00
    naive_iso = f"{t0.date().isoformat()}T10:00:00"
    tick6 = {"token": token, "ltp": 24500.03, "volume": 10, "exchange_timestamp": naive_iso}
    out6 = tv.validate(tick6)
    print("TEST6 Tick size rejection:", "PASS" if out6 is None else "FAIL")

    # 7) F6 Naive timestamp assumption: accept naive ISO as IST (default assume_naive_is_utc=False)
    tick7 = {"token": token, "ltp": 24500.05, "volume": 10, "exchange_timestamp": naive_iso}
    out7 = tv.validate(tick7)
    print("TEST7 Naive timestamp accepted:", "PASS" if (out7 is not None) else "FAIL")

    # 8) F8 Multi-token feed health callback on health class changes
    tv.reset_daily()
    tv.add_health_monitored_token(token)
    tv.set_feed_health_callback(feed_health_cb)
    # First tick -> FIRST_TICK (no callback)
    tv.validate({"token": token, "ltp": 24500.00, "volume": 1, "exchange_timestamp": _dt_to_epoch_ms(t0)})
    # Next tick with 2-sec gap => DELAYED (since delay threshold default 1s) triggers callback
    tv.validate({"token": token, "ltp": 24500.05, "volume": 1, "exchange_timestamp": _dt_to_epoch_ms(t0 + timedelta(seconds=2))})
    print("TEST8 Feed health callback:", "PASS" if len(health_events) >= 1 else "FAIL", "| last_event=", health_events[-1] if health_events else None)

    print("\nSelf-test complete.\nStats:", tv.get_stats())