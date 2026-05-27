"""
Junior Aladdin - Option Chain Poller (Institutional Grade Hardened v2)
======================================================================

FILE: src/core/option_chain_poller.py

Round 1 Hardening (kept):
- thread safety (RLock + poll_lock)
- retries + backoff
- best-effort timeouts (executor + future timeouts)
- parallel fetching (optional)
- OI cache integration via update_from_tick/update_oi_from_tick
- IV validation bounds
- poll health telemetry + circuit breaker cooldown
- never-crash guarantees

Round 2 Intelligence (added):
1) Liquidity filter for IV computation (ltp>1, volume/oi present, spread sanity)
2) Strike relevance filtering (optional) - keep ATM always
3) OI classification uses underlying spot_change primarily
4) Anomaly detection on OI/volume vs previous values (10x jumps)
5) ATM CE/PE IV consistency check
6) Rate-limit awareness in parallel submission (max req/sec pacing)
7) Strike prioritization (ATM -> near ATM -> far)
8) Delta caching TODO (future)

IMPORTANT:
- We do NOT fabricate OI=0. Missing OI => None with data_quality flags.
- We do NOT compute IV on illiquid quotes. IV becomes None and iv_valid=False.

"""

from __future__ import annotations

import copy
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional, Any, Tuple

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.utils.helpers import round_to_strike, days_to_expiry
from src.utils.black_scholes import implied_volatility, compute_all_greeks

IST = timezone(timedelta(hours=5, minutes=30))


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


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class PollHealth:
    poll_count: int = 0
    last_poll_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_duration_ms: Optional[float] = None
    avg_duration_ms: Optional[float] = None  # EMA
    consecutive_failures: int = 0
    cooldown_until: Optional[datetime] = None
    last_error: str = ""
    last_success_ratio: Optional[float] = None


@dataclass(frozen=True)
class OICacheEntry:
    oi: Optional[int]
    volume: Optional[int]
    bid: Optional[float]
    ask: Optional[float]
    updated_at: datetime


class OptionChainPoller:
    """
    Polls NIFTY option chain from Angel One and computes IV/Greeks/OI classification.

    Usage:
        poller = OptionChainPoller(auth_manager, instrument_mapper)
        poller.update_from_tick(ws_tick)  # optional
        chain = poller.poll(spot_price=24500.0)
    """

    def __init__(self, auth_manager, instrument_mapper):
        self._logger = setup_logger("option_chain_poller")
        self._auth = auth_manager
        self._mapper = instrument_mapper

        # Market config
        self._strike_interval = int(Config.get("market", "strike_interval", default=50))
        self._strikes_range = int(Config.get("data", "option_strikes_range", default=5))
        self._risk_free_rate = float(Config.get("features", "risk_free_rate", default=0.065))

        # API robustness
        self._api_timeout_sec = float(Config.get("data", "option_poller_api_timeout_sec", default=2.0))
        self._api_retries = max(0, int(Config.get("data", "option_poller_api_max_retries", default=2)))
        self._api_backoff_base = float(Config.get("data", "option_poller_api_backoff_base_sec", default=0.5))

        # Parallelization / rate limiting
        self._parallel_enabled = bool(Config.get("data", "option_poller_parallel_enabled", default=True))
        self._max_workers = int(Config.get("data", "option_poller_max_workers", default=6))
        self._max_workers = max(1, min(self._max_workers, 16))

        # NEW: rate-limit awareness during submission
        self._max_rps = float(Config.get("data", "option_poller_max_requests_per_second", default=10))
        self._max_rps = max(1.0, min(self._max_rps, 50.0))
        # also cap workers to avoid excessive concurrent pressure
        self._max_workers = int(min(self._max_workers, max(2, int(self._max_rps))))

        # Poll budget
        self._poll_budget_sec = float(Config.get("data", "option_poller_poll_budget_sec", default=20.0))
        self._poll_budget_sec = max(5.0, self._poll_budget_sec)

        # OI cache staleness
        self._oi_stale_sec = float(Config.get("data", "option_poller_oi_stale_sec", default=60.0))
        self._oi_stale_sec = max(5.0, self._oi_stale_sec)

        # IV validation bounds
        self._iv_min = float(Config.get("data", "option_poller_iv_min", default=0.05))
        self._iv_max = float(Config.get("data", "option_poller_iv_max", default=2.0))

        # NEW: Liquidity filters for IV computation
        self._iv_min_ltp = float(Config.get("data", "option_poller_iv_min_ltp", default=1.0))
        self._iv_max_spread_ratio = float(Config.get("data", "option_poller_iv_max_spread_ratio", default=0.5))

        # NEW: strike relevance filter
        self._min_vol_oi_threshold = int(Config.get("data", "option_poller_min_volume_oi_threshold", default=1))
        self._filter_illiquid_strikes = bool(Config.get("data", "option_poller_filter_illiquid_strikes", default=False))

        # Circuit breaker
        self._cb_fail_threshold = int(Config.get("data", "option_poller_cb_consecutive_failures", default=5))
        self._cb_cooldown_sec = float(Config.get("data", "option_poller_cb_cooldown_sec", default=120.0))

        # State
        self._lock = threading.RLock()
        self._poll_lock = threading.Lock()

        self.current_chain: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.previous_chain: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.last_poll_time: Optional[datetime] = None
        self.poll_count: int = 0

        # OI cache: token -> OICacheEntry
        self._oi_cache: Dict[str, OICacheEntry] = {}

        # Poll health
        self._health = PollHealth()

        # Spot memory for OI classification context (NEW)
        self._previous_spot_price: Optional[float] = None

        # One-time warnings
        self._warned_timeout_config = False
        self._warned_no_oi_cache = False

        # TEMP (one-time): log raw REST LTP response keys for live verification; remove after confirmation
        self._raw_data_keys_logged = False

        # TODO (Phase 2 req #8): Delta caching / incremental update strategy

    # ------------------------------------------------------------------
    # External cache update API (DataEngine should call from websocket ticks)
    # ------------------------------------------------------------------
    def update_oi_from_tick(
        self,
        token: str,
        oi: Optional[int] = None,
        volume: Optional[int] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        if not token:
            return
        if timestamp is None:
            timestamp = datetime.now(IST)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)

        oi_i = _safe_int(oi, default=None)
        vol_i = _safe_int(volume, default=None)
        bid_f = _safe_float(bid, default=None)
        ask_f = _safe_float(ask, default=None)

        with self._lock:
            self._oi_cache[str(token)] = OICacheEntry(
                oi=oi_i,
                volume=vol_i,
                bid=bid_f,
                ask=ask_f,
                updated_at=timestamp,
            )

    def update_from_tick(self, tick: Dict[str, Any]):
        if not isinstance(tick, dict):
            return
        token = str(tick.get("token", "") or "")
        if not token:
            return

        ts = tick.get("received_at")
        timestamp = ts if isinstance(ts, datetime) else datetime.now(IST)

        oi = tick.get("open_interest", tick.get("oi", None))
        vol = tick.get("volume", None)

        bid = None
        ask = None
        b5b = tick.get("best_5_buy", None)
        b5s = tick.get("best_5_sell", None)
        if isinstance(b5b, list) and b5b:
            bid = _safe_float(b5b[0].get("price"), default=None) if isinstance(b5b[0], dict) else None
        if isinstance(b5s, list) and b5s:
            ask = _safe_float(b5s[0].get("price"), default=None) if isinstance(b5s[0], dict) else None

        self.update_oi_from_tick(token=token, oi=oi, volume=vol, bid=bid, ask=ask, timestamp=timestamp)

    # ------------------------------------------------------------------
    # Public Poll API
    # ------------------------------------------------------------------
    def poll(self, spot_price: float) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Fetch option chain for ATM ± range strikes.
        """
        if not self._poll_lock.acquire(blocking=False):
            self._logger.warning("Option chain poll skipped (previous poll still running)")
            with self._lock:
                return copy.deepcopy(self.current_chain)

        start_ts = datetime.now(IST)
        start_perf = time.perf_counter()

        try:
            with self._lock:
                cd_until = self._health.cooldown_until
            if cd_until is not None and start_ts < cd_until:
                self._logger.critical(
                    "Option poller in circuit-breaker cooldown; skipping poll",
                    cooldown_until=cd_until.isoformat(),
                    now=start_ts.isoformat(),
                )
                with self._lock:
                    return copy.deepcopy(self.current_chain)

            self._health.poll_count += 1
            self.poll_count = self._health.poll_count
            self._logger.info("Polling option chain", spot=spot_price, poll_count=self.poll_count)

            # previous chain snapshot
            with self._lock:
                self.previous_chain = copy.deepcopy(self.current_chain)

            prev_spot = self._previous_spot_price
            spot_change = (
                float(spot_price - prev_spot)
                if (isinstance(prev_spot, (int, float)) and prev_spot > 0)
                else 0.0
            )

            atm_strike = round_to_strike(spot_price, self._strike_interval)
            strikes: List[int] = [
                int(atm_strike + i * self._strike_interval)
                for i in range(-self._strikes_range, self._strikes_range + 1)
            ]

            # expiry
            current_expiry = None
            try:
                current_expiry = self._mapper.get_current_expiry()
            except Exception as e:
                self._logger.error("InstrumentMapper.get_current_expiry failed", error=str(e))

            if current_expiry is None:
                self._logger.error("No current expiry found")
                self._mark_poll_failure("no_expiry")
                with self._lock:
                    return copy.deepcopy(self.current_chain)

            expiry_date = current_expiry if isinstance(current_expiry, date) else None
            expiry_str = (
                current_expiry.strftime("%Y-%m-%d")
                if hasattr(current_expiry, "strftime")
                else str(current_expiry)
            )

            T = self._compute_T_years(expiry_date)

            # api
            smart_api = None
            try:
                smart_api = self._auth.get_smart_api()
            except Exception as e:
                self._logger.error("AuthManager.get_smart_api failed", error=str(e))

            if smart_api is None:
                self._logger.error("No API connection available")
                self._mark_poll_failure("no_api")
                with self._lock:
                    return copy.deepcopy(self.current_chain)

            self._configure_smartapi_timeouts(smart_api)

            # tasks prioritized
            tasks = self._build_prioritized_tasks(strikes=strikes, atm_strike=atm_strike)

            poll_deadline = start_perf + self._poll_budget_sec

            if self._parallel_enabled:
                quote_results = self._fetch_quotes_parallel(
                    smart_api=smart_api,
                    expiry_date=expiry_date,
                    expiry_str=expiry_str,
                    tasks=tasks,
                    poll_deadline=poll_deadline,
                )
            else:
                quote_results = self._fetch_quotes_sequential(
                    smart_api=smart_api,
                    expiry_date=expiry_date,
                    expiry_str=expiry_str,
                    tasks=tasks,
                    poll_deadline=poll_deadline,
                )

            # build chain
            new_chain: Dict[int, Dict[str, Dict[str, Any]]] = {}
            success_count = 0
            total_count = 0

            for strike in strikes:
                strike_data = {"ce": {}, "pe": {}}
                for opt_type in ("CE", "PE"):
                    total_count += 1
                    q = quote_results.get((strike, opt_type))

                    if q is None:
                        strike_data[opt_type.lower()] = self._empty_opt_data(
                            strike=strike, expiry_str=expiry_str, opt_type=opt_type, reason="quote_missing"
                        )
                        continue

                    opt_data = self._build_option_data(
                        strike=strike,
                        expiry_str=expiry_str,
                        opt_type=opt_type,
                        spot_price=spot_price,
                        spot_change=spot_change,
                        T=T,
                        quote=q,
                    )
                    if opt_data.get("data_available") is True:
                        success_count += 1
                    strike_data[opt_type.lower()] = opt_data

                new_chain[strike] = strike_data

            # Cross-leg IV consistency at ATM
            self._apply_atm_iv_consistency_check(new_chain, atm_strike)

            # Optional strike relevance filtering
            if self._filter_illiquid_strikes:
                new_chain = self._filter_chain_by_relevance(new_chain, atm_strike)

            # update health + state
            end_ts = datetime.now(IST)
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            success_ratio = success_count / max(total_count, 1)

            with self._lock:
                self.current_chain = new_chain
                self.last_poll_time = end_ts
                self._health.last_poll_time = end_ts
                self._health.last_duration_ms = duration_ms
                self._health.last_success_ratio = round(success_ratio, 4)

                if self._health.avg_duration_ms is None:
                    self._health.avg_duration_ms = duration_ms
                else:
                    alpha = 0.2
                    self._health.avg_duration_ms = (1 - alpha) * self._health.avg_duration_ms + alpha * duration_ms

                if success_ratio >= 0.50:
                    self._health.consecutive_failures = 0
                    self._health.last_success_time = end_ts
                    self._health.last_error = ""
                else:
                    self._health.consecutive_failures += 1
                    self._health.last_error = f"low_success_ratio:{success_ratio:.2f}"

                # update previous spot after completing poll
                self._previous_spot_price = float(spot_price) if spot_price and spot_price > 0 else self._previous_spot_price

            self._logger.info(
                "Option chain polled",
                total_strikes=len(new_chain),
                strikes_with_any_ltp=sum(
                    1
                    for s in new_chain.values()
                    if (s.get("ce", {}).get("ltp") not in (None, 0.0))
                    or (s.get("pe", {}).get("ltp") not in (None, 0.0))
                ),
                atm_strike=atm_strike,
                expiry=expiry_str,
                poll_count=self.poll_count,
                duration_ms=round(duration_ms, 1),
                success_ratio=round(success_ratio, 3),
                failures_in_row=self._health.consecutive_failures,
                parallel=self._parallel_enabled,
                max_workers=self._max_workers,
                max_rps=self._max_rps,
            )

            self._maybe_trip_circuit_breaker()
            return copy.deepcopy(new_chain)

        except Exception as e:
            self._logger.error("Option chain poll crashed; returning last known chain", error=str(e))
            self._mark_poll_failure(f"exception:{type(e).__name__}")
            self._maybe_trip_circuit_breaker()
            with self._lock:
                return copy.deepcopy(self.current_chain)

        finally:
            self._poll_lock.release()

    # ------------------------------------------------------------------
    # Tasks prioritization
    # ------------------------------------------------------------------
    def _build_prioritized_tasks(self, strikes: List[int], atm_strike: int) -> List[Tuple[int, str]]:
        tasks = [(s, "CE") for s in strikes] + [(s, "PE") for s in strikes]

        def pri(item: Tuple[int, str]) -> Tuple[int, int, int]:
            strike, opt_type = item
            dist_steps = int(abs(strike - atm_strike) / max(self._strike_interval, 1))
            if strike == atm_strike and opt_type == "CE":
                bucket = 0
            elif strike == atm_strike and opt_type == "PE":
                bucket = 1
            elif dist_steps <= 1:
                bucket = 2
            elif dist_steps <= 2:
                bucket = 3
            else:
                bucket = 4
            opt_ord = 0 if opt_type == "CE" else 1
            return (bucket, dist_steps, opt_ord)

        tasks.sort(key=pri)
        return tasks

    # ------------------------------------------------------------------
    # Parallel/Sequential fetch
    # ------------------------------------------------------------------
    def _fetch_quotes_parallel(
        self,
        smart_api,
        expiry_date: Optional[date],
        expiry_str: str,
        tasks: List[Tuple[int, str]],
        poll_deadline: float,
    ) -> Dict[Tuple[int, str], Dict[str, Any]]:
        results: Dict[Tuple[int, str], Dict[str, Any]] = {}

        ex: Optional[ThreadPoolExecutor] = None
        futures = []
        future_map: Dict[Any, Tuple[int, str]] = {}

        min_submit_interval = 1.0 / float(self._max_rps)
        next_submit_time = time.monotonic()

        try:
            ex = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="opt_poller")

            for strike, opt_type in tasks:
                if time.perf_counter() > poll_deadline:
                    self._logger.warning("Poll budget exceeded; stopping quote submission")
                    break

                now_m = time.monotonic()
                if now_m < next_submit_time:
                    time.sleep(min(0.05, next_submit_time - now_m))
                next_submit_time = max(next_submit_time + min_submit_interval, time.monotonic())

                fut = ex.submit(self._fetch_one_quote_with_retries, smart_api, strike, expiry_date, expiry_str, opt_type)
                futures.append(fut)
                future_map[fut] = (strike, opt_type)

            for fut in as_completed(futures, timeout=max(0.1, self._poll_budget_sec)):
                if time.perf_counter() > poll_deadline:
                    self._logger.warning("Poll budget exceeded; returning partial quotes")
                    break
                strike, opt_type = future_map.get(fut, (-1, ""))
                try:
                    q = fut.result(timeout=self._api_timeout_sec * max(1, (self._api_retries + 1)))
                    if q is not None:
                        results[(strike, opt_type)] = q
                except FuturesTimeoutError:
                    self._logger.warning("Quote future timed out", strike=strike, opt_type=opt_type)
                except Exception as e:
                    self._logger.debug("Quote future error", strike=strike, opt_type=opt_type, error=str(e)[:120])

        except Exception as e:
            self._logger.error("Parallel fetch failed; returning partial", error=str(e)[:120])

        finally:
            for fut in futures:
                if not fut.done():
                    try:
                        fut.cancel()
                    except Exception:
                        pass
            if ex is not None:
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    try:
                        ex.shutdown(wait=False)
                    except Exception:
                        pass

        return results

    def _fetch_quotes_sequential(
        self,
        smart_api,
        expiry_date: Optional[date],
        expiry_str: str,
        tasks: List[Tuple[int, str]],
        poll_deadline: float,
    ) -> Dict[Tuple[int, str], Dict[str, Any]]:
        results: Dict[Tuple[int, str], Dict[str, Any]] = {}
        min_submit_interval = 1.0 / float(self._max_rps)
        next_time = time.monotonic()

        for strike, opt_type in tasks:
            if time.perf_counter() > poll_deadline:
                self._logger.warning("Poll budget exceeded; returning partial quotes (sequential)")
                break

            now_m = time.monotonic()
            if now_m < next_time:
                time.sleep(min(0.05, next_time - now_m))
            next_time = max(next_time + min_submit_interval, time.monotonic())

            q = self._fetch_one_quote_with_retries(smart_api, strike, expiry_date, expiry_str, opt_type)
            if q is not None:
                results[(strike, opt_type)] = q

        return results

    # ------------------------------------------------------------------
    # One quote fetch with retries/backoff
    # ------------------------------------------------------------------
    def _fetch_one_quote_with_retries(
        self,
        smart_api,
        strike: int,
        expiry_date: Optional[date],
        expiry_str: str,
        opt_type: str,
    ) -> Optional[Dict[str, Any]]:
        token, symbol = self._resolve_token_symbol(strike, expiry_date, expiry_str, opt_type)
        if token is None or symbol is None:
            return None

        last_err = None
        for attempt in range(self._api_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = smart_api.ltpData("NFO", symbol, token)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                if not resp or not isinstance(resp, dict):
                    last_err = "resp_not_dict"
                    raise RuntimeError("ltpData response invalid")

                if not resp.get("status", False):
                    last_err = str(resp.get("message", "status_false"))[:120]
                    raise RuntimeError(f"ltpData status false: {last_err}")

                data = resp.get("data", {})
                if not isinstance(data, dict):
                    last_err = "data_not_dict"
                    raise RuntimeError("ltpData data invalid")

                # LTP extraction (hardened for SmartAPI v2). DO NOT fallback to 'close'.
                ltp_keys = ("ltp", "last_traded_price", "LTP", "last_price", "tradedPrice")
                ltp_val = None
                for k in ltp_keys:
                    v = data.get(k)
                    if v is not None:
                        ltp_val = v
                        break

                # TEMPORARY diagnostics: log raw keys once (thread-safe). Remove after verification.
                should_log = False
                with self._lock:
                    if not self._raw_data_keys_logged:
                        self._raw_data_keys_logged = True
                        should_log = True
                if should_log:
                    self._logger.warning(
                        "LTP extraction diagnostics – raw data keys: %s, sample: %s",
                        list(data.keys())[:15],
                        str(data)[:200],
                    )

                ltp = _safe_float(ltp_val, default=None)

                o = _safe_float(data.get("open"), default=None)
                h = _safe_float(data.get("high"), default=None)
                l = _safe_float(data.get("low"), default=None)
                c = _safe_float(data.get("close"), default=None)

                if ltp is None or not (math.isfinite(ltp) and ltp > 0):
                    last_err = "ltp_missing_or_zero"
                    raise RuntimeError("ltp missing/zero/unusable")

                return {
                    "token": str(token),
                    "symbol": str(symbol),
                    "ltp": float(ltp),
                    "open": float(o or 0.0),
                    "high": float(h or 0.0),
                    "low": float(l or 0.0),
                    "close": float(c or 0.0),
                    "api_latency_ms": round(dt_ms, 2),
                    "data_ok": True,
                }

            except Exception as e:
                last_err = last_err or str(e)[:120]
                if attempt < self._api_retries:
                    backoff = self._api_backoff_base * (2 ** attempt)
                    time.sleep(min(2.0, backoff))
                    continue

                self._logger.debug(
                    "LTP fetch failed after retries",
                    strike=strike,
                    opt_type=opt_type,
                    symbol=symbol,
                    error=str(e)[:120],
                )
                return {
                    "token": str(token),
                    "symbol": str(symbol),
                    "ltp": None,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "api_latency_ms": None,
                    "data_ok": False,
                    "error": last_err,
                }

        return None

    # ------------------------------------------------------------------
    # Build option dict with intelligence filters
    # ------------------------------------------------------------------
    def _build_option_data(
        self,
        strike: int,
        expiry_str: str,
        opt_type: str,
        spot_price: float,
        spot_change: float,
        T: float,
        quote: Dict[str, Any],
    ) -> Dict[str, Any]:
        token = str(quote.get("token", "") or "")
        symbol = quote.get("symbol", "")
        ltp = quote.get("ltp", None)

        oi, vol, bid, ask, oi_age_sec, anomaly_flags = self._get_cached_oi_with_anomaly_check(token, strike, opt_type)

        prev_opt = self._get_prev_opt(strike=strike, opt_type=opt_type)
        prev_ltp = prev_opt.get("ltp")
        prev_oi = prev_opt.get("oi")

        option_price_change = (
            float(ltp - prev_ltp)
            if (isinstance(ltp, (int, float)) and isinstance(prev_ltp, (int, float)) and prev_ltp > 0)
            else 0.0
        )
        oi_change = None
        if isinstance(oi, int) and isinstance(prev_oi, int):
            oi_change = int(oi - prev_oi)

        classification = self._classify_oi(
            spot_change=spot_change,
            option_price_change=option_price_change,
            oi_change=oi_change,
        )

        iv, greeks, iv_valid, iv_liquidity_ok, spread_ratio = self._compute_iv_greeks_if_liquid(
            ltp=ltp,
            spot_price=spot_price,
            strike=strike,
            T=T,
            opt_type=opt_type,
            oi=oi,
            volume=vol,
            bid=bid,
            ask=ask,
        )

        data_available = bool(quote.get("data_ok")) and (ltp is not None)

        opt_data = {
            "ltp": ltp,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "bid": bid,
            "ask": ask,
            "spread_ratio": spread_ratio,
            "volume": vol,
            "oi": oi,
            "oi_change": oi_change,
            "iv": (round(iv, 4) if iv is not None else None),
            "iv_pct": (round(iv * 100, 2) if iv is not None else None),
            "delta": (round(greeks["delta"], 4) if greeks.get("delta") is not None else None),
            "gamma": (round(greeks["gamma"], 6) if greeks.get("gamma") is not None else None),
            "theta": (round(greeks["theta"], 2) if greeks.get("theta") is not None else None),
            "vega": (round(greeks["vega"], 2) if greeks.get("vega") is not None else None),
            "classification": classification,
            "token": token,
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry_str,
            "option_type": opt_type,
            "data_available": data_available,
            "oi_available": oi is not None,
            "oi_age_sec": (round(oi_age_sec, 2) if oi_age_sec is not None else None),
            "oi_anomaly": bool(anomaly_flags.get("oi_anomaly", False)),
            "volume_anomaly": bool(anomaly_flags.get("volume_anomaly", False)),
            "iv_valid": iv_valid,
            "iv_liquidity_ok": iv_liquidity_ok,
            "api_latency_ms": quote.get("api_latency_ms"),
        }

        return opt_data

    def _compute_iv_greeks_if_liquid(
        self,
        ltp: Any,
        spot_price: float,
        strike: int,
        T: float,
        opt_type: str,
        oi: Optional[int],
        volume: Optional[int],
        bid: Optional[float],
        ask: Optional[float],
    ) -> Tuple[Optional[float], Dict[str, Optional[float]], bool, bool, Optional[float]]:
        greeks = {"delta": None, "gamma": None, "theta": None, "vega": None}

        ltp_f = _safe_float(ltp, default=None)
        if ltp_f is None or ltp_f <= 0 or spot_price <= 0:
            return None, greeks, False, False, None

        has_interest = (isinstance(volume, int) and volume > 0) or (isinstance(oi, int) and oi > 0)
        ltp_ok = ltp_f >= float(self._iv_min_ltp)

        spread_ratio = None
        spread_ok = True
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            spread_ratio = (ask - bid) / max(ltp_f, 0.0001)
            spread_ok = spread_ratio < float(self._iv_max_spread_ratio)

        liquidity_ok = bool(has_interest and ltp_ok and spread_ok)
        if not liquidity_ok:
            return None, greeks, False, False, spread_ratio

        try:
            iv_raw = implied_volatility(
                float(ltp_f),
                float(spot_price),
                float(strike),
                float(T),
                float(self._risk_free_rate),
                opt_type,
            )
            if iv_raw is None:
                return None, greeks, False, True, spread_ratio

            iv_val = float(iv_raw)
            if not (self._iv_min <= iv_val <= self._iv_max):
                self._logger.debug("IV rejected (out of bounds)", strike=strike, opt_type=opt_type, iv=iv_val)
                return None, greeks, False, True, spread_ratio

            g = compute_all_greeks(
                float(spot_price),
                float(strike),
                float(T),
                float(self._risk_free_rate),
                float(iv_val),
                opt_type,
            )
            if isinstance(g, dict):
                greeks = {
                    "delta": _safe_float(g.get("delta"), default=None),
                    "gamma": _safe_float(g.get("gamma"), default=None),
                    "theta": _safe_float(g.get("theta"), default=None),
                    "vega": _safe_float(g.get("vega"), default=None),
                }
            return iv_val, greeks, True, True, spread_ratio
        except Exception as e:
            self._logger.debug("IV/Greeks computation failed", strike=strike, opt_type=opt_type, error=str(e)[:120])
            return None, greeks, False, True, spread_ratio

    def _empty_opt_data(self, strike: int, expiry_str: str, opt_type: str, reason: str) -> Dict[str, Any]:
        return {
            "ltp": None, "open": None, "high": None, "low": None, "close": None,
            "bid": None, "ask": None, "spread_ratio": None,
            "volume": None,
            "oi": None, "oi_change": None,
            "iv": None, "iv_pct": None,
            "delta": None, "gamma": None, "theta": None, "vega": None,
            "classification": "NO_DATA",
            "token": None, "symbol": None,
            "strike": strike, "expiry": expiry_str, "option_type": opt_type,
            "data_available": False,
            "oi_available": False,
            "oi_age_sec": None,
            "oi_anomaly": False,
            "volume_anomaly": False,
            "iv_valid": False,
            "iv_liquidity_ok": False,
            "api_latency_ms": None,
            "error": reason,
        }

    def _filter_chain_by_relevance(
        self, chain: Dict[int, Dict[str, Dict[str, Any]]], atm_strike: int
    ) -> Dict[int, Dict[str, Dict[str, Any]]]:
        thr = max(1, int(self._min_vol_oi_threshold))

        def leg_ok(leg: Dict[str, Any]) -> bool:
            oi = leg.get("oi")
            vol = leg.get("volume")
            if isinstance(vol, int) and vol >= thr:
                return True
            if isinstance(oi, int) and oi >= thr:
                return True
            return False

        filtered: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for strike, sd in chain.items():
            if strike == atm_strike:
                filtered[strike] = sd
                continue
            ce = sd.get("ce", {}) if isinstance(sd, dict) else {}
            pe = sd.get("pe", {}) if isinstance(sd, dict) else {}
            if leg_ok(ce) or leg_ok(pe):
                filtered[strike] = sd

        removed = len(chain) - len(filtered)
        if removed > 0:
            self._logger.info("Strike relevance filtering applied", removed=removed, kept=len(filtered), atm=atm_strike)
        return filtered

    def _classify_oi(self, spot_change: float, option_price_change: float, oi_change: Optional[int]) -> str:
        if oi_change is None:
            if spot_change > 0:
                return "SPOT_UP_OI_UNKNOWN"
            if spot_change < 0:
                return "SPOT_DOWN_OI_UNKNOWN"
            if option_price_change > 0:
                return "PRICE_UP_OI_UNKNOWN"
            if option_price_change < 0:
                return "PRICE_DOWN_OI_UNKNOWN"
            return "NO_OI"

        dir_up = None
        if spot_change > 0:
            dir_up = True
        elif spot_change < 0:
            dir_up = False
        else:
            if option_price_change > 0:
                dir_up = True
            elif option_price_change < 0:
                dir_up = False
            else:
                return "OI_ONLY_INCREASE" if oi_change > 0 else "OI_ONLY_DECREASE"

        if dir_up:
            if oi_change > 0:
                return "LONG_BUILDUP"
            elif oi_change < 0:
                return "SHORT_COVERING"
            else:
                return "SPOT_UP"
        else:
            if oi_change > 0:
                return "SHORT_BUILDUP"
            elif oi_change < 0:
                return "LONG_UNWINDING"
            else:
                return "SPOT_DOWN"

    def _get_cached_oi_with_anomaly_check(
        self,
        token: str,
        strike: int,
        opt_type: str,
    ) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], Dict[str, bool]]:
        flags = {"oi_anomaly": False, "volume_anomaly": False}

        oi, vol, bid, ask, age = self._get_cached_oi(token)

        prev_opt = self._get_prev_opt(strike=strike, opt_type=opt_type)
        prev_oi = prev_opt.get("oi")
        prev_vol = prev_opt.get("volume")

        if isinstance(prev_oi, int) and prev_oi > 1000 and isinstance(oi, int):
            if oi > 10 * prev_oi:
                flags["oi_anomaly"] = True
                self._logger.warning(
                    "OI anomaly detected; nulling OI for this poll",
                    token=token, strike=strike, opt_type=opt_type,
                    prev_oi=prev_oi, oi=oi
                )
                oi = None

        if isinstance(prev_vol, int) and prev_vol > 1000 and isinstance(vol, int):
            if vol > 10 * prev_vol:
                flags["volume_anomaly"] = True
                self._logger.warning(
                    "Volume anomaly detected; nulling volume for this poll",
                    token=token, strike=strike, opt_type=opt_type,
                    prev_vol=prev_vol, vol=vol
                )
                vol = None

        return oi, vol, bid, ask, age, flags

    def _get_cached_oi(self, token: str) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], Optional[float]]:
        if not token:
            return None, None, None, None, None

        with self._lock:
            entry = self._oi_cache.get(token)

        if entry is None:
            if not self._warned_no_oi_cache:
                self._logger.warning("OI cache is empty/unavailable. OI features will be degraded until cache is wired.")
                self._warned_no_oi_cache = True
            return None, None, None, None, None

        now = datetime.now(IST)
        age = (now - entry.updated_at).total_seconds()
        if age > self._oi_stale_sec:
            return None, None, None, None, age

        return entry.oi, entry.volume, entry.bid, entry.ask, age

    def _apply_atm_iv_consistency_check(self, chain: Dict[int, Dict[str, Dict[str, Any]]], atm_strike: int):
        try:
            sd = chain.get(atm_strike)
            if not isinstance(sd, dict):
                return
            ce = sd.get("ce", {})
            pe = sd.get("pe", {})
            ce_iv = ce.get("iv")
            pe_iv = pe.get("iv")
            if isinstance(ce_iv, (int, float)) and isinstance(pe_iv, (int, float)):
                if abs(float(ce_iv) - float(pe_iv)) > 0.10:
                    self._logger.warning(
                        "ATM CE/PE IV inconsistency detected; invalidating both",
                        atm=atm_strike,
                        ce_iv=float(ce_iv),
                        pe_iv=float(pe_iv),
                    )
                    for leg in (ce, pe):
                        leg["iv_valid"] = False
                        leg["iv_consistency_suspect"] = True
                        leg["iv"] = None
                        leg["iv_pct"] = None
                        leg["delta"] = None
                        leg["gamma"] = None
                        leg["theta"] = None
                        leg["vega"] = None
        except Exception as e:
            self._logger.debug("ATM IV consistency check failed", error=str(e)[:120])

    def _get_prev_opt(self, strike: int, opt_type: str) -> Dict[str, Any]:
        with self._lock:
            prev_strike = self.previous_chain.get(strike, {})
            prev_opt = prev_strike.get(opt_type.lower(), {}) if isinstance(prev_strike, dict) else {}
            return prev_opt if isinstance(prev_opt, dict) else {}

    def _compute_T_years(self, expiry_date: Optional[date]) -> float:
        dte = None
        try:
            dte = days_to_expiry()
        except Exception:
            dte = None

        if dte is None and expiry_date is not None:
            try:
                today = datetime.now(IST).date()
                dte = max(0, (expiry_date - today).days)
            except Exception:
                dte = 0

        if dte is None:
            dte = 0

        return float(max(0.5, float(dte)) / 365.0)

    # ------------------------------------------------------------------
    # Change 1 — Diagnostic logging around token/symbol resolution
    # ------------------------------------------------------------------
    def _resolve_token_symbol(
        self,
        strike: int,
        expiry_date: Optional[date],
        expiry_str: str,
        opt_type: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve token and symbol for a NIFTY option contract.

        Returns (token, symbol) or (None, None) if unresolvable.
        LOGS WARNING when resolution fails so the problem is visible.
        """
        token = None
        symbol = None

        # ------ token ------
        if hasattr(self._mapper, "get_option_token"):
            try:
                if expiry_date is not None:
                    token = self._mapper.get_option_token(strike, expiry_date, opt_type)
                else:
                    token = None
                if token is None:
                    token = self._mapper.get_option_token(strike, expiry_str, opt_type)
            except Exception as e:
                self._logger.error(
                    "get_option_token raised exception",
                    strike=strike,
                    expiry=expiry_str,
                    opt_type=opt_type,
                    error=str(e)[:120],
                )
                token = None
        else:
            self._logger.error(
                "InstrumentMapper missing get_option_token; cannot resolve token",
                strike=strike,
                expiry=expiry_str,
                opt_type=opt_type,
            )
            return None, None

        # ------ symbol ------
        if hasattr(self._mapper, "get_option_symbol"):
            try:
                if expiry_date is not None:
                    symbol = self._mapper.get_option_symbol(strike, expiry_date, opt_type)
                else:
                    symbol = None
                if symbol is None:
                    symbol = self._mapper.get_option_symbol(strike, expiry_str, opt_type)
            except Exception as e:
                self._logger.error(
                    "get_option_symbol raised exception",
                    strike=strike,
                    expiry=expiry_str,
                    opt_type=opt_type,
                    error=str(e)[:120],
                )
                symbol = None
        else:
            self._logger.error(
                "InstrumentMapper missing get_option_symbol; cannot resolve symbol",
                strike=strike,
                expiry=expiry_str,
                opt_type=opt_type,
            )
            return None, None

        if token is None or symbol is None:
            self._logger.warning(
                "Failed to resolve token/symbol; skipping option quote",
                strike=strike,
                expiry=expiry_str,
                opt_type=opt_type,
                token_resolved=bool(token),
                symbol_resolved=bool(symbol),
            )
            return None, None

        return str(token), str(symbol)

    def _configure_smartapi_timeouts(self, smart_api: Any):
        if smart_api is None:
            return
        try:
            for attr in ("timeout", "_timeout", "req_timeout", "request_timeout"):
                if hasattr(smart_api, attr):
                    setattr(smart_api, attr, self._api_timeout_sec)

            if not self._warned_timeout_config:
                self._logger.info("Configured SmartAPI timeout (best-effort)", timeout_sec=self._api_timeout_sec)
                self._warned_timeout_config = True
        except Exception as e:
            self._logger.debug("SmartAPI timeout configuration failed", error=str(e)[:120])

    def _mark_poll_failure(self, reason: str):
        with self._lock:
            self._health.consecutive_failures += 1
            self._health.last_error = reason
            self._health.last_poll_time = datetime.now(IST)
            self.last_poll_time = self._health.last_poll_time

    def _maybe_trip_circuit_breaker(self):
        with self._lock:
            fails = self._health.consecutive_failures
            cd_until = self._health.cooldown_until

        if cd_until is not None and datetime.now(IST) < cd_until:
            return

        if fails >= self._cb_fail_threshold:
            until = datetime.now(IST) + timedelta(seconds=self._cb_cooldown_sec)
            with self._lock:
                self._health.cooldown_until = until
            self._logger.critical(
                "OptionChainPoller circuit breaker TRIPPED",
                consecutive_failures=fails,
                cooldown_sec=self._cb_cooldown_sec,
                cooldown_until=until.isoformat(),
            )

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "poll_count": self._health.poll_count,
                "last_poll_time": self._health.last_poll_time.isoformat() if self._health.last_poll_time else None,
                "last_success_time": self._health.last_success_time.isoformat() if self._health.last_success_time else None,
                "last_duration_ms": (round(self._health.last_duration_ms, 2) if self._health.last_duration_ms is not None else None),
                "avg_duration_ms": (round(self._health.avg_duration_ms, 2) if self._health.avg_duration_ms is not None else None),
                "consecutive_failures": self._health.consecutive_failures,
                "cooldown_until": self._health.cooldown_until.isoformat() if self._health.cooldown_until else None,
                "last_error": self._health.last_error,
                "last_success_ratio": self._health.last_success_ratio,
                "strikes_in_chain": len(self.current_chain),
                "has_data": len(self.current_chain) > 0,
                "parallel_enabled": self._parallel_enabled,
                "max_workers": self._max_workers,
                "api_timeout_sec": self._api_timeout_sec,
                "api_retries": self._api_retries,
                "oi_cache_size": len(self._oi_cache),
                "oi_stale_sec": self._oi_stale_sec,
                "max_requests_per_second": self._max_rps,
                "filter_illiquid_strikes": self._filter_illiquid_strikes,
            }


if __name__ == "__main__":
    print("=" * 60)
    print("  JUNIOR ALADDIN — Option Chain Poller Test (Hardened v2)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    class MockAuth:
        def get_smart_api(self):
            return None

    class MockMapper:
        def get_current_expiry(self):
            return None

    poller = OptionChainPoller(MockAuth(), MockMapper())

    print("  [Test 1] OI classification uses spot_change...")
    r = poller._classify_oi(spot_change=10.0, option_price_change=-5.0, oi_change=100)
    if r == "LONG_BUILDUP":
        print("    ✅ spot_change UP drives classification")
        passed += 1
    else:
        print("    ❌ classification wrong:", r)
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)