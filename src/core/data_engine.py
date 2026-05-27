# src/core/data_engine.py
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List

import pandas as pd

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.utils.helpers import ist_now, is_market_hours, IST  # type: ignore

from src.core.auth_manager import AuthManager
from src.core.instrument_mapper import InstrumentMapper
from src.core.websocket_manager import WebSocketManager
from src.core.tick_validator import TickValidator
from src.core.candle_builder import CandleBuilder
from src.core.market_state import MarketState
from data_center.connectors.backend_connector import backend_connector

# Prefer updated module names; keep fallback for repo compatibility
try:
    from src.core.option_chain_poller import OptionChainPoller as _OptionChainComponent  # type: ignore
except Exception:  # pragma: no cover
    from src.core.option_chain import OptionChainManager as _OptionChainComponent  # type: ignore

try:
    from src.core.feed_health_monitor import FeedHealthMonitor  # type: ignore
except Exception:  # pragma: no cover
    from src.core.feed_health import FeedHealthMonitor  # type: ignore


CANDLES_DIR = os.path.join("data", "historical", "candles")


class _InstrumentMapperAdapter:
    """
    Adapter around InstrumentMapper that provides:
      - get_instrument_spec(token)
      - is_token_known(token)
      - register_subscription_tokens(subscription_list)

    This is required because the institutional TickValidator validates tokens against
    instrument specs. Spot/VIX tokens may not be present in the options-only token map.
    """

    def __init__(self, base: InstrumentMapper, nifty_token: str, vix_token: str):
        self._base = base
        self._nifty_token = str(nifty_token)
        self._vix_token = str(vix_token)

        # reverse lookup: token_str -> info dict
        self._token_reverse: Dict[str, Dict[str, Any]] = {}

        # Provide minimal specs for spot/VIX
        self._static_specs: Dict[str, Dict[str, Any]] = {
            self._nifty_token: {
                "token": self._nifty_token,
                "symbol": "NIFTY",
                "name": "NIFTY",
                "exch_seg": "NSE",
                "exchange": "nse_cm",
                "instrumenttype": "INDEX",
                # normalized extras
                "instrument_class": "INDEX",
                "tick_size": 0.05,
                "min_price": 10000.0,
                "max_price": 50000.0,
            },
            self._vix_token: {
                "token": self._vix_token,
                "symbol": "INDIAVIX",
                "name": "INDIAVIX",
                "exch_seg": "NSE",
                "exchange": "nse_cm",
                "instrumenttype": "INDEX",
                # normalized extras
                "instrument_class": "INDEX",
                "tick_size": 0.01,  # Change 1 — INDIAVIX tick size fix
                "min_price": 1.0,
                "max_price": 200.0,
            },
        }

    def _rebuild_reverse_index(self) -> None:
        rev: Dict[str, Dict[str, Any]] = {}
        try:
            token_map = getattr(self._base, "_token_map", None)
            if isinstance(token_map, dict):
                for _k, info in token_map.items():
                    if isinstance(info, dict):
                        tok = info.get("token")
                        if tok:
                            rev[str(tok)] = dict(info)
        except Exception:
            rev = {}
        self._token_reverse = rev

    def register_subscription_tokens(self, subscription_list: List[Dict[str, Any]]) -> None:
        """Pre-populate reverse index with tokens from the initial subscription list.
        This ensures TickValidator knows all subscribed tokens immediately at startup."""
        if not isinstance(subscription_list, list):
            return

        for item in subscription_list:
            if not isinstance(item, dict):
                # tolerate token-only subscription lists (list[str]/list[int]) if any
                tok = str(item).strip()
                if not tok:
                    continue
                if tok in self._static_specs or tok in self._token_reverse:
                    continue
                self._token_reverse[tok] = {
                    "token": tok,
                    "symbol": tok,
                    "exch_seg": "NFO",
                    "instrument_class": "OPTION",
                    "tick_size": 0.05,
                    "min_price": 0.0,
                    "max_price": 1_000_000.0,
                }
                continue

            tok = str(item.get("token", "")).strip()
            if not tok:
                continue
            if tok in self._static_specs or tok in self._token_reverse:
                continue

            # Build minimal info from subscription dict so validator can recognize the token
            self._token_reverse[tok] = {
                "token": tok,
                "symbol": item.get("symbol", "") or item.get("tradingsymbol", "") or tok,
                "exch_seg": item.get("exchange", "") or item.get("exch_seg", "") or "NFO",
                "instrument_class": "OPTION",
                "tick_size": 0.05,
                "min_price": 0.0,
                "max_price": 1_000_000.0,
            }

    def get_instrument_spec(self, token: Any) -> Optional[Dict[str, Any]]:
        tok = str(token) if token is not None else ""
        if not tok:
            return None

        if tok in self._static_specs:
            return dict(self._static_specs[tok])

        info = self._token_reverse.get(tok)
        if info:
            spec = dict(info)
            spec.setdefault("token", tok)
            spec.setdefault("exch_seg", "NFO")
            spec.setdefault("exchange", "nfo_cm")
            spec.setdefault("instrumenttype", "OPTIDX")

            # normalized extras (do not change original fields)
            spec.setdefault("instrument_class", "OPTION")
            spec.setdefault("tick_size", 0.05)
            spec.setdefault("min_price", 0.0)
            spec.setdefault("max_price", 1_000_000.0)
            return spec

        return None

    def is_token_known(self, token: Any) -> bool:
        tok = str(token) if token is not None else ""
        return bool(tok) and (tok in self._static_specs or tok in self._token_reverse)

    def __getattr__(self, item: str):
        return getattr(self._base, item)

    def build_map(self, smart_api=None, spot_price: float = 24500.0) -> Dict:
        out = self._base.build_map(smart_api=smart_api, spot_price=spot_price)
        self._rebuild_reverse_index()
        return out


class DataEngine:
    """
    Institutional-grade master data engine.

    Public API signatures unchanged:
        - start() -> bool
        - stop() -> None
        - get_status() -> Dict
    """

    def __init__(self, market_state: MarketState):
        self._logger = setup_logger("data_engine")
        self._state = market_state

        # Core dependencies
        self._auth: Optional[AuthManager] = None
        self._mapper: Optional[_InstrumentMapperAdapter] = None
        self._ws: Optional[WebSocketManager] = None
        self._validator: Optional[TickValidator] = None
        self._candle_builder: Optional[CandleBuilder] = None
        self._option_component: Optional[Any] = None
        self._feed_health: Optional[FeedHealthMonitor] = None

        # Control
        self._stop_event = threading.Event()

        # Threads
        self._option_poll_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None

        # State
        self.is_running: bool = False

        # Tokens
        self._nifty_token = str(Config.get("market", "nifty_spot_token", default="99926000"))
        self._vix_token = str(Config.get("market", "india_vix_token", default="26017"))

        # Timings
        self._option_poll_interval = float(Config.get("data", "option_chain_poll_interval_sec", default=30))
        self._health_check_interval = float(Config.get("data", "feed_health_check_interval_sec", default=0.5))
        self._quality_alert_threshold = float(Config.get("data", "quality_alert_threshold", default=40.0))
        self._ws_connect_timeout_sec = float(Config.get("data", "ws_connect_timeout_sec", default=30.0))
        self._stop_step_timeout_sec = float(Config.get("data", "engine_stop_step_timeout_sec", default=8.0))

        # Subscription update after first tick (non-blocking only)
        self._live_subscription_applied = threading.Event()
        self._live_subscription_skipped = False

        # Optional downstream callback hook (kept)
        self._on_candle_close_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Health transition callback
        self._health_transition_callback: Optional[Callable[..., None]] = None
        self._last_feed_health: Optional[str] = None
        self._last_quality_band: Optional[str] = None

        # Validator degraded mode
        self._validator_degraded: bool = False
        self._validator_degraded_since: Optional[datetime] = None

        # Runtime flow counters (phase-1 wiring validation)
        self._tick_seen_count: int = 0
        self._tick_validated_count: int = 0
        self._tick_rejected_count: int = 0
        self._spot_tick_validated_count: int = 0
        self._candle_close_count: int = 0
        self._option_poll_count: int = 0
        self._option_poll_success_count: int = 0
        self._option_poll_error_count: int = 0

        self._last_tick_seen_at: Optional[datetime] = None
        self._last_tick_validated_at: Optional[datetime] = None
        self._last_spot_tick_at: Optional[datetime] = None
        self._last_candle_close_at: Optional[datetime] = None
        self._last_option_poll_at: Optional[datetime] = None
        self._last_option_update_at: Optional[datetime] = None
        self._last_tick_exception: Optional[str] = None
        self._last_tick_exception_at: Optional[datetime] = None

        os.makedirs(CANDLES_DIR, exist_ok=True)

    def _reset_flow_counters(self) -> None:
        self._tick_seen_count = 0
        self._tick_validated_count = 0
        self._tick_rejected_count = 0
        self._spot_tick_validated_count = 0
        self._candle_close_count = 0
        self._option_poll_count = 0
        self._option_poll_success_count = 0
        self._option_poll_error_count = 0

        self._last_tick_seen_at = None
        self._last_tick_validated_at = None
        self._last_spot_tick_at = None
        self._last_candle_close_at = None
        self._last_option_poll_at = None
        self._last_option_update_at = None
        self._last_tick_exception = None
        self._last_tick_exception_at = None

    @staticmethod
    def _normalize_ws_tick(raw_tick: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and normalizes fields from Angel One SmartAPI WebSocket v2 ticks.

        Accepts multiple possible field name aliases. If a required field is missing,
        it is set to None rather than raising an exception.
        """
        if not isinstance(raw_tick, dict):
            return {}

        token = (
            raw_tick.get("token")
            or raw_tick.get("instrument_token")
            or raw_tick.get("symboltoken")
            or raw_tick.get("symbolToken")
            or raw_tick.get("tradingSymbolToken")
            or raw_tick.get("symbol")
        )
        token_str = str(token).strip() if token is not None else ""

        ts = (
            raw_tick.get("exchange_timestamp")
            or raw_tick.get("exchangeTime")
            or raw_tick.get("timestamp")
            or raw_tick.get("last_traded_time")
            or raw_tick.get("trade_time")
            or raw_tick.get("filled_time")
            or raw_tick.get("time")
        )

        ltp = (
            raw_tick.get("ltp")
            or raw_tick.get("LTP")
            or raw_tick.get("last_price")
            or raw_tick.get("price")
            or raw_tick.get("last_traded_price")
            or raw_tick.get("tradedPrice")
            or raw_tick.get("fill_price")
            or raw_tick.get("lastPrice")
            or raw_tick.get("ltp_price")
        )

        vol = (
            raw_tick.get("volume")
            or raw_tick.get("vol")
            or raw_tick.get("tradedVolume")
            or raw_tick.get("qty")
            or raw_tick.get("last_traded_quantity")
            or raw_tick.get("tradedQty")
            or raw_tick.get("fill_qty")
            or raw_tick.get("volume_traded")
            or raw_tick.get("traded_volume")
        )

        if vol is not None:
            try:
                vol_int = int(str(vol).split(".")[0])
            except (ValueError, TypeError):
                vol_int = -1
        else:
            vol_int = -1

        return {"token": token_str, "exchange_timestamp": ts, "ltp": ltp, "volume": vol_int}

    # ------------------------------------------------------------------
    # Data Center Bridge
    # ------------------------------------------------------------------
    def _on_data_center_tick(self, cleaned_tick: Dict[str, Any]) -> None:
        """Subscriber for Data Center processed ticks."""
        if not self.is_running:
            return
            
        token_str = str(cleaned_tick.get("token", ""))
        self._tick_seen_count += 1
        self._last_tick_seen_at = ist_now()
        
        # Feed directly to validated logic
        self._on_validated_tick(cleaned_tick, token_str=token_str)

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------
    def set_candle_close_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self._on_candle_close_callback = callback

    def set_health_transition_callback(self, callback: Optional[Callable[..., None]]) -> None:
        self._health_transition_callback = callback

    def _emit_health_event(self, event: str, **kwargs: Any) -> None:
        cb = self._health_transition_callback
        if cb is None:
            return
        try:
            cb(event, **kwargs)
        except TypeError:
            try:
                cb(**kwargs)
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------
    def _fetch_previous_close(self, api: Any) -> float:
        default_spot = float(Config.get("data", "startup_spot_default", default=24000))
        if api is None:
            self._logger.warning(
                "SmartAPI unavailable for prev_close fetch; using default",
                extra={"default": default_spot},
            )
            return default_spot

        try:
            # 100% Accuracy Fix: Use ltpData for official close if available
            resp = api.ltpData("NSE", "NIFTY", self._nifty_token)
            if resp and resp.get("status"):
                data = resp.get("data", {})
                close_price = data.get("close") or data.get("ltp")
                if close_price:
                    self._logger.info(f"Official Previous Close fetched via ltpData: {close_price}")
                    return float(close_price)

            to_dt = ist_now()
            from_dt = to_dt - timedelta(days=10)
            params = {
                "exchange": "NSE",
                "symboltoken": self._nifty_token,
                "interval": "ONE_DAY",
                "fromdate": from_dt.strftime("%Y-%m-%d 00:00"),
                "todate": to_dt.strftime("%Y-%m-%d 23:59"),
            }
            resp = api.getCandleData(params)
            if not (isinstance(resp, dict) and resp.get("status")):
                self._logger.warning(
                    "Prev_close fetch failed (bad response); using default",
                    extra={"default": default_spot, "response": str(resp)[:200]},
                )
                return default_spot

            data = resp.get("data") or []
            if not isinstance(data, list) or len(data) == 0:
                self._logger.warning(
                    "Prev_close fetch returned empty; using default",
                    extra={"default": default_spot},
                )
                return default_spot

            last = data[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 5:
                prev_close = float(last[4])
                if prev_close > 0:
                    return prev_close

            self._logger.warning(
                "Prev_close parse failed; using default",
                extra={"default": default_spot},
            )
            return default_spot

        except Exception as e:
            self._logger.warning(
                "Prev_close fetch exception; using default",
                extra={"default": default_spot, "error": str(e)},
            )
            return default_spot

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if self.is_running:
            self._logger.warning("Data Engine already running")
            return True

        self._logger.info("Starting Data Engine...")
        self._stop_event.clear()
        self._reset_flow_counters()

        # Auth
        self._auth = AuthManager()
        if hasattr(self._auth, "has_credentials") and not self._auth.has_credentials():
            self._logger.error("No credentials configured in .env")
            return False

        if hasattr(self._auth, "authenticate"):
            if not self._auth.authenticate():
                self._logger.error("Authentication failed")
                return False

        api = self._auth.get_smart_api()
        if api is None:
            self._logger.critical("SmartAPI unavailable after authentication")
            return False

        # Prev close
        prev_close = self._fetch_previous_close(api)
        self._logger.info("Prev close resolved", extra={"prev_close": prev_close})

        # Mapper build
        base_mapper = InstrumentMapper()
        self._mapper = _InstrumentMapperAdapter(base_mapper, self._nifty_token, self._vix_token)

        token_map = self._mapper.build_map(api, spot_price=prev_close)
        if not token_map or not getattr(self._mapper, "is_built", False):
            self._logger.critical("Instrument mapping failed; cannot start DataEngine safely")
            return False

        self._logger.info(
            "Instrument map built",
            extra={"mapped": getattr(self._mapper, "total_instruments_mapped", None)},
        )
        
        # Unified Bridge: Register listener
        backend_connector.register_tick_listener(self._on_data_center_tick)
        self._logger.info("Registered with BackendConnector for unified data supply")

        # IMPORTANT: subscription list must be registered BEFORE TickValidator
        initial_sub = self._mapper.get_subscription_list(prev_close)
        self._mapper.register_subscription_tokens(initial_sub)

        # TickValidator (after subscription tokens pre-registered)
        self._validator = TickValidator(instrument_mapper=self._mapper)

        # Sub-components
        self._candle_builder = CandleBuilder(default_token=self._nifty_token)
        self._feed_health = FeedHealthMonitor()
        self._option_component = _OptionChainComponent(self._auth, self._mapper)  # standardized to poll(spot_price)

        # WebSocket
        self._ws = WebSocketManager(
            auth_manager=self._auth,
            instrument_mapper=self._mapper,
            on_tick_callback=self._process_tick,
            on_connect_callback=self._on_ws_connect,
            on_disconnect_callback=self._on_ws_disconnect,
        )

        ws_started = self._ws.connect(subscription_list=initial_sub, spot_price=prev_close)
        if not ws_started:
            self._logger.error(
                "WebSocket failed to start; continuing in degraded observation mode",
                extra={"using_fallback": True},
            )

        # Start Data Center Pipelines
        try:
            from data_center.pipeline.raw_pipeline import start_raw_pipeline
            from data_center.pipeline.clean_pipeline import start_clean_pipeline
            from data_center.pipeline.minor_pipeline import start_minor_pipeline
            start_raw_pipeline()
            start_clean_pipeline()
            start_minor_pipeline(self._auth, self._mapper, self._state)
            self._logger.info("Data Center Unified Pipelines Active.")
        except Exception as e:
            self._logger.debug(f"Data Center pipeline initialization failed: {e}")

        if not self._start_background_threads():
            return False

        self.is_running = True
        self._state.update(
            system_state="OBSERVE",
            feed_health=getattr(self._feed_health, "current_health", "DOWN"),
            data_quality_score=float(getattr(self._feed_health, "data_quality_score", 0.0) or 0.0),
            using_fallback=not ws_started,
            previous_close=prev_close,
        )

        self._logger.info("Data Engine started", extra={"ws_started": ws_started})
        return True

    def stop(self):
        self._logger.info("Stopping Data Engine...")
        self.is_running = False
        self._stop_event.set()

        if self._ws:
            self._run_stop_step_with_timeout(
                "websocket_disconnect",
                self._ws.disconnect,
                timeout_sec=max(2.0, float(self._stop_step_timeout_sec)),
            )

        # Stop Data Center Pipelines
        try:
            from data_center.pipeline.raw_pipeline import stop_raw_pipeline
            from data_center.pipeline.clean_pipeline import stop_clean_pipeline
            from data_center.pipeline.minor_pipeline import stop_minor_pipeline
            stop_raw_pipeline()
            stop_clean_pipeline()
            stop_minor_pipeline()
        except: pass

        for th, timeout in (
            (self._option_poll_thread, 5),
            (self._health_check_thread, 2),
        ):
            if th and th.is_alive():
                th.join(timeout=timeout)

        # Final archival only
        self._run_stop_step_with_timeout(
            "final_archive",
            self._archive_daily_data,
            timeout_sec=max(2.0, float(self._stop_step_timeout_sec)),
        )

        if self._auth and getattr(self._auth, "is_authenticated", False):
            self._run_stop_step_with_timeout(
                "auth_logout",
                self._auth.logout,
                timeout_sec=max(2.0, float(self._stop_step_timeout_sec)),
            )

        self._state.update(system_state="SHUTDOWN")
        self._logger.info("Data Engine stopped")

    def _run_stop_step_with_timeout(self, step_name: str, fn: Callable[[], Any], timeout_sec: float) -> bool:
        done = threading.Event()
        err_holder: Dict[str, Exception] = {}

        def _worker() -> None:
            try:
                fn()
            except Exception as exc:
                err_holder["exc"] = exc
            finally:
                done.set()

        th = threading.Thread(target=_worker, daemon=True, name=f"DataEngineStop-{step_name}")
        th.start()

        wait_sec = max(1.0, float(timeout_sec or 1.0))
        if not done.wait(timeout=wait_sec):
            self._logger.error(
                "Stop step timed out; continuing shutdown",
                extra={"step": step_name, "timeout_sec": wait_sec},
            )
            return False

        if "exc" in err_holder:
            exc = err_holder["exc"]
            self._logger.warning(
                "Stop step raised exception",
                extra={"step": step_name, "error": str(exc), "type": type(exc).__name__},
            )
            return False

        return True

    # ------------------------------------------------------------------
    # WebSocket connect helpers
    # ------------------------------------------------------------------
    def _ws_connect(self, subscription_list: List[Dict[str, Any]], spot_price: Optional[float]) -> bool:
        if not self._ws:
            return False

        timeout_sec = max(1.0, float(self._ws_connect_timeout_sec or 30.0))
        result: Dict[str, Any] = {"ok": False, "error": None}
        done = threading.Event()

        def _connect_worker() -> None:
            try:
                sp = float(spot_price) if (spot_price is not None and spot_price > 0) else 24000.0
                try:
                    result["ok"] = bool(self._ws.connect(subscription_list=subscription_list, spot_price=sp))
                except TypeError:
                    self._logger.warning(
                        "WebSocketManager.connect(subscription_list=...) not supported; falling back to connect(spot_price=...)",
                        extra={"spot_price": sp},
                    )
                    result["ok"] = bool(self._ws.connect(spot_price=sp))
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=_connect_worker, daemon=True, name="DataEngine-WSConnect")
        worker.start()

        if not done.wait(timeout=timeout_sec):
            self._logger.error(
                "WebSocket connect timed out; proceeding in degraded mode",
                extra={"timeout_sec": timeout_sec},
            )
            return False

        if result["error"] is not None:
            err = result["error"]
            self._logger.error("WebSocket connect error", extra={"error": str(err), "type": type(err).__name__})
            return False

        return bool(result["ok"])

    def _ws_apply_subscription(self, subscription_list: List[Dict[str, Any]], spot_price: float) -> bool:
        if not self._ws:
            return False

        for method_name in ("update_subscriptions", "update_subscription_list", "resubscribe", "set_subscription_list", "subscribe_tokens"):
            if hasattr(self._ws, method_name):
                try:
                    method = getattr(self._ws, method_name)
                    if method_name == "update_subscriptions":
                        ok = bool(method(spot_price))
                    else:
                        ok = bool(method(subscription_list))
                    self._logger.info(
                        "Applied WebSocket subscription via method",
                        extra={"method": method_name, "tokens": len(subscription_list)},
                    )
                    if ok:
                        return True
                except Exception as e:
                    self._logger.warning(
                        "Failed applying subscription via method",
                        extra={"method": method_name, "error": str(e)},
                    )

        self._logger.warning(
            "WebSocket live subscription update skipped (no supported update method). No reconnect will be performed.",
            extra={"tokens": len(subscription_list), "spot": spot_price},
        )
        return False

    def _on_ws_connect(self, *_args: Any, **_kwargs: Any) -> None:
        self._logger.info("WebSocket connected")

    def _on_ws_disconnect(self, *_args: Any, **_kwargs: Any) -> None:
        self._logger.warning("WebSocket disconnected")

    # ------------------------------------------------------------------
    # Tick validation + degraded mode
    # ------------------------------------------------------------------
    def _validate_tick_safely(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._validator:
            self._logger.critical("TickValidator missing; cannot validate tick")
            return None

        try:
            res = self._validator.validate(tick_data)
        except Exception as e:
            self._validator_degraded = True
            if self._validator_degraded_since is None:
                self._validator_degraded_since = ist_now()
            self._logger.critical(
                "TickValidator exception; entering degraded mode (tick will pass with flag)",
                extra={"error": str(e), "type": type(e).__name__},
            )

            ts = tick_data.get("exchange_timestamp") or tick_data.get("timestamp") or ist_now()
            if isinstance(ts, datetime):
                ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=IST)
            else:
                ts_dt = ist_now()

            try:
                ltp = float(tick_data.get("ltp") or tick_data.get("price") or 0.0)
            except Exception:
                return None

            tok = str(tick_data.get("token_str") or tick_data.get("token") or "")
            vol = tick_data.get("volume", -1)
            try:
                vol_i = int(vol) if vol is not None else -1
            except Exception:
                vol_i = -1

            return {
                "timestamp": ts_dt,
                "token": tok or str(tick_data.get("token") or "UNKNOWN"),
                "symbol": tok or "UNKNOWN",
                "instrument_class": "UNKNOWN",
                "ltp": ltp,
                "volume": vol_i,
                "is_spike": False,
                "feed_gap_sec": 0.0,
                "feed_health": "DEGRADED",
                "latency_sec": 0.0,
                "is_stale": False,
                "reconnect_burst": False,
                "same_timestamp_update": False,
                "tick_size": None,
                "dynamic_spike_threshold_pct": 0.0,
                "validator_degraded": True,
            }

        if res is None:
            return None

        if isinstance(res, dict):
            return res

        if isinstance(res, tuple) and len(res) == 2:
            if not bool(res[0]):
                return None
            ts = tick_data.get("exchange_timestamp") or tick_data.get("timestamp") or ist_now()
            if isinstance(ts, datetime):
                ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=IST)
            else:
                ts_dt = ist_now()
            tok = str(tick_data.get("token_str") or tick_data.get("token") or "")
            return {
                "timestamp": ts_dt,
                "token": tok,
                "symbol": tok,
                "instrument_class": "UNKNOWN",
                "ltp": float(tick_data.get("ltp", 0.0) or 0.0),
                "volume": int(tick_data.get("volume", -1) or -1),
                "is_spike": False,
                "feed_gap_sec": 0.0,
                "feed_health": "HEALTHY",
                "latency_sec": 0.0,
                "is_stale": False,
                "reconnect_burst": False,
                "same_timestamp_update": False,
                "tick_size": None,
                "dynamic_spike_threshold_pct": 0.0,
                "validator_legacy": True,
            }

        if isinstance(res, bool):
            if not res:
                return None
            ts = tick_data.get("exchange_timestamp") or tick_data.get("timestamp") or ist_now()
            if isinstance(ts, datetime):
                ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=IST)
            else:
                ts_dt = ist_now()
            tok = str(tick_data.get("token_str") or tick_data.get("token") or "")
            return {
                "timestamp": ts_dt,
                "token": tok,
                "symbol": tok,
                "instrument_class": "UNKNOWN",
                "ltp": float(tick_data.get("ltp", 0.0) or 0.0),
                "volume": int(tick_data.get("volume", -1) or -1),
                "is_spike": False,
                "feed_gap_sec": 0.0,
                "feed_health": "HEALTHY",
                "latency_sec": 0.0,
                "is_stale": False,
                "reconnect_burst": False,
                "same_timestamp_update": False,
                "tick_size": None,
                "dynamic_spike_threshold_pct": 0.0,
                "validator_legacy": True,
            }

        self._logger.critical(
            "TickValidator returned unexpected type; rejecting tick",
            extra={"return_type": type(res).__name__},
        )
        return None

    def _on_validated_tick(self, validated_tick: Dict[str, Any], *, token_str: str) -> None:
        cb = self._candle_builder
        if cb is None:
            self._logger.critical("CandleBuilder is None during validated tick handling (wiring fault)")
            raise RuntimeError("CandleBuilder is None")

        # Only NIFTY spot drives candle stream in current design
        if token_str == self._nifty_token:
            self._spot_tick_validated_count += 1
            self._last_spot_tick_at = ist_now()

            try:
                price = float(validated_tick.get("ltp", 0.0) or 0.0)
            except Exception:
                return
            if price <= 0:
                return

            ts = validated_tick.get("timestamp")
            if not isinstance(ts, datetime):
                ts = ist_now()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)

            vol = validated_tick.get("volume", -1)
            try:
                vol_i = int(vol) if vol is not None else -1
            except Exception:
                vol_i = -1

            self._state.update(timestamp=ts, spot=price)

            if (not self._live_subscription_applied.is_set()) and self._mapper and (not self._live_subscription_skipped):
                try:
                    new_sub = self._mapper.get_subscription_list(price)
                    ok = self._ws_apply_subscription(new_sub, spot_price=price)
                    if ok:
                        self._live_subscription_applied.set()
                        self._logger.info(
                            "Live subscription applied after first valid tick",
                            extra={"spot": price, "tokens": len(new_sub)},
                        )
                    else:
                        self._live_subscription_skipped = True
                        self._live_subscription_applied.set()
                except Exception as e:
                    self._logger.warning(
                        "Live subscription update error (skipping further attempts)",
                        extra={"error": str(e)},
                    )
                    self._live_subscription_skipped = True
                    self._live_subscription_applied.set()

            result = cb.on_tick(price, vol_i, ts, token=token_str)
            if result == "candle_closed":
                self._on_candle_closed(token_str)
        
        # FIX: Populate VIX in MarketState
        elif token_str == self._vix_token:
            price = float(validated_tick.get("ltp", 0.0) or 0.0)
            if price > 0:
                self._state.update(vix=price)

    def _on_rejected_tick(self, _tick: Dict[str, Any]) -> None:
        return

    def _process_tick(self, raw_tick: Dict):
        if not raw_tick:
            return

        self._tick_seen_count += 1
        self._last_tick_seen_at = ist_now()

        if self._feed_health:
            try:
                self._feed_health.on_tick(ist_now())
            except Exception:
                pass

        try:
            std_tick = self._normalize_ws_tick(raw_tick)

            # Smart mapping to ensure TickCleaner doesn't produce 'empty output'
            ltp = (raw_tick.get("ltp") or raw_tick.get("last_traded_price") or 
                   raw_tick.get("LTP") or raw_tick.get("price") or 0.0)

            token_val = std_tick.get("token")
            token_str = str(token_val).strip() if token_val is not None else ""
            if not token_str:
                return

            received_at = ist_now()
            std_tick["received_at"] = raw_tick.get("received_at") or received_at
            std_tick["timestamp"] = std_tick.get("exchange_timestamp") or std_tick["received_at"]
            std_tick["exchange_timestamp"] = std_tick.get("exchange_timestamp") or std_tick["received_at"]
            std_tick["token_str"] = token_str

            # 100% Accuracy Fix: Sync official close from Live Meta
            if token_str == self._nifty_token and raw_tick.get("close"):
                self._state.update(previous_close=float(raw_tick["close"]))

            validated = self._validate_tick_safely(std_tick)
            if validated is None:
                self._tick_rejected_count += 1
                self._on_rejected_tick(std_tick)
            else:
                self._tick_validated_count += 1
                self._last_tick_validated_at = ist_now()
                self._on_validated_tick(validated, token_str=token_str)

            # ------------------------------------------------------------------
            # Data Center ingestion hook (Restored & Hardened)
            try:
                from data_center.queues.tick_queue import tick_queue
                tick_queue.put_nowait({
                    "token": token_str,
                    "ltp": float(ltp),
                    "volume": int(raw_tick.get("volume") or raw_tick.get("vol") or 0),
                    "timestamp": int(time.time() * 1000),
                    "open": float(raw_tick.get("open", 0.0)),
                    "high": float(raw_tick.get("high", 0.0)),
                    "low": float(raw_tick.get("low", 0.0)),
                    "close": float(raw_tick.get("close", 0.0)),
                    "direction": int(raw_tick.get("direction", 0))
                })
            except: pass

            # ------------------------------------------------------------------
            # FEED OI / BID / ASK INTO OPTION CHAIN POLLER CACHE
            # ------------------------------------------------------------------
            if self._option_component is not None and hasattr(self._option_component, "update_from_tick"):
                try:
                    self._option_component.update_from_tick(validated if validated else std_tick)
                except Exception:
                    pass  # must never crash the hot tick path

            if self._feed_health:
                health = getattr(self._feed_health, "current_health", None)
                score = getattr(self._feed_health, "data_quality_score", None)
                lag = 0.0
                tps = 0.0
                if hasattr(self._feed_health, "get_feed_lag"):
                    try:
                        lag = float(self._feed_health.get_feed_lag())
                    except Exception:
                        lag = 0.0
                if hasattr(self._feed_health, "get_tps"):
                    try:
                        tps = float(self._feed_health.get_tps())
                    except Exception:
                        tps = 0.0

                if health is not None:
                    self._state.update(
                        feed_health=str(health),
                        data_quality_score=float(score or 0.0),
                        feed_lag_ms=float(lag),
                        ticks_per_second=float(tps),
                    )

        except ValueError:
            return
        except Exception as e:
            self._last_tick_exception = str(e)
            self._last_tick_exception_at = ist_now()
            self._logger.critical(
                "Unhandled exception in _process_tick (will re-raise)",
                extra={"error": str(e), "type": type(e).__name__},
            )
            raise

    def _on_candle_closed(self, token: str):
        cb = self._candle_builder
        if cb is None:
            return

        try:
            last_1m = cb.get_last_closed("1min", token=token)
            status = {
                "token": token,
                "last_1min": last_1m,
                "counts": {
                    "1min": cb.get_candle_count("1min", token=token),
                    "3min": cb.get_candle_count("3min", token=token),
                    "5min": cb.get_candle_count("5min", token=token),
                    "15min": cb.get_candle_count("15min", token=token),
                },
            }

            self._candle_close_count += 1
            self._last_candle_close_at = ist_now()

            self._state.update(candles=cb.candles)

            if self._on_candle_close_callback:
                try:
                    self._on_candle_close_callback(status)
                except Exception as e:
                    self._logger.error("Candle close callback error", extra={"error": str(e)})

        except Exception as e:
            self._logger.error("on_candle_closed failed", extra={"error": str(e), "type": type(e).__name__})

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------
    def _start_background_threads(self) -> bool:
        self._option_poll_thread = threading.Thread(
            target=self._option_poll_loop,
            daemon=True,
            name="OptionPoll-Thread",
        )
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="HealthCheck-Thread",
        )

        self._option_poll_thread.start()
        time.sleep(0.05)
        if not self._option_poll_thread.is_alive():
            self._logger.critical("Option poll thread failed to start")
            self.stop()
            return False

        self._health_check_thread.start()
        time.sleep(0.05)
        if not self._health_check_thread.is_alive():
            self._logger.critical("Health check thread failed to start")
            self.stop()
            return False

        return True

    def _option_poll_loop(self):
        while not self._stop_event.is_set():
            try:
                snap = self._state.snapshot()
                feed_health = str(snap.get("feed_health", "DOWN"))
                spot = float(snap.get("spot", 0.0) or 0.0)

                if not is_market_hours():
                    self._stop_event.wait(self._option_poll_interval)
                    continue

                if feed_health == "DOWN" or spot <= 0:
                    self._stop_event.wait(self._option_poll_interval)
                    continue

                if self._option_component is None:
                    self._stop_event.wait(self._option_poll_interval)
                    continue

                self._option_poll_count += 1
                self._last_option_poll_at = ist_now()
                chain = self._option_component.poll(spot)
                if isinstance(chain, dict) and chain:
                    self._option_poll_success_count += 1
                    self._last_option_update_at = ist_now()
                    self._state.update(option_chain=chain)
                    if self._feed_health:
                        try:
                            self._feed_health.on_option_update()
                        except Exception:
                            pass
                        score = getattr(self._feed_health, "data_quality_score", None)
                        if score is not None:
                            self._state.update(data_quality_score=float(score))

            except Exception as e:
                self._option_poll_error_count += 1
                self._logger.error("Option poll error", extra={"error": str(e), "type": type(e).__name__})

            self._stop_event.wait(self._option_poll_interval)

    def _health_check_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._feed_health:
                    health, score = self._feed_health.check()
                    health = str(health)
                    score_f = float(score)

                    self._state.update(feed_health=health, data_quality_score=score_f)

                    if self._last_feed_health is None:
                        self._last_feed_health = health
                    else:
                        if health != self._last_feed_health:
                            old = self._last_feed_health
                            self._last_feed_health = health
                            if health in ("DOWN", "STALE"):
                                self._emit_health_event(
                                    "feed_health_transition",
                                    old=old,
                                    new=health,
                                    score=score_f,
                                )

                    band = "OK" if score_f >= self._quality_alert_threshold else "LOW"
                    if self._last_quality_band is None:
                        self._last_quality_band = band
                    else:
                        if band != self._last_quality_band:
                            oldb = self._last_quality_band
                            self._last_quality_band = band
                            if band == "LOW":
                                self._emit_health_event(
                                    "data_quality_low",
                                    old=oldb,
                                    new=band,
                                    score=score_f,
                                )

                    if self._validator_degraded:
                        self._emit_health_event(
                            "validator_degraded",
                            since=self._validator_degraded_since.isoformat() if self._validator_degraded_since else None,
                        )

            except Exception:
                pass
            self._stop_event.wait(self._health_check_interval)

    # ------------------------------------------------------------------
    # Archival
    # ------------------------------------------------------------------
    def _archive_daily_data(self):
        cb = self._candle_builder
        if cb is None:
            return

        try:
            candles_1m = list(cb.get_candles("1min", token=self._nifty_token))
        except Exception:
            candles_1m = []

        if not candles_1m:
            return

        df_new = pd.DataFrame(candles_1m)
        if "timestamp" not in df_new.columns:
            return

        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors="coerce")
        df_new = df_new.dropna(subset=["timestamp"]).copy()
        if df_new.empty:
            return

        day_str = pd.Timestamp(df_new["timestamp"].iloc[0]).date().isoformat()

        file_path = os.path.join(CANDLES_DIR, f"NIFTY_1min_{day_str}.parquet")
        tmp_path = file_path + ".tmp"
        os.makedirs(CANDLES_DIR, exist_ok=True)

        if os.path.exists(file_path):
            try:
                old = pd.read_parquet(file_path)
                if "timestamp" in old.columns:
                    old["timestamp"] = pd.to_datetime(old["timestamp"], errors="coerce")
                merged = pd.concat([old, df_new], ignore_index=True)
                merged = merged.dropna(subset=["timestamp"])
                merged = (
                    merged.drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                df_new = merged
            except Exception as e:
                self._logger.warning(
                    "Existing archive read failed; writing current snapshot",
                    extra={"error": str(e), "file": file_path},
                )

        df_new.to_parquet(tmp_path, compression="snappy", index=False)
        os.replace(tmp_path, file_path)

        self._logger.info("Data archived (merge+atomic)", extra={"file": file_path, "candles": int(len(df_new))})

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        snap = self._state.snapshot()
        spot = float(snap.get("spot", 0.0) or 0.0)

        candles_1m = self._candle_builder.get_candle_count("1min", token=self._nifty_token) if self._candle_builder else 0
        candles_3m = self._candle_builder.get_candle_count("3min", token=self._nifty_token) if self._candle_builder else 0
        candles_5m = self._candle_builder.get_candle_count("5min", token=self._nifty_token) if self._candle_builder else 0
        candles_15m = self._candle_builder.get_candle_count("15min", token=self._nifty_token) if self._candle_builder else 0

        validator_raw = self._validator.get_stats() if self._validator and hasattr(self._validator, "get_stats") else {}
        total_seen = int(validator_raw.get("total_seen", 0) or 0)
        total_valid = int(validator_raw.get("total_valid", 0) or 0)

        validator_stats = dict(validator_raw)
        validator_stats.update(
            {
                "valid": total_valid,
                "rejected": max(0, total_seen - total_valid),
                "spikes": int(validator_raw.get("flagged_spikes", 0) or 0),
            }
        )

        return {
            "is_running": self.is_running,
            "spot_price": spot,
            "feed_health": snap.get("feed_health", "DOWN"),
            "data_quality": float(snap.get("data_quality_score", 0.0) or 0.0),
            "ws_connected": getattr(self._ws, "is_connected", False) if self._ws else False,
            "ws_ticks": getattr(self._ws, "total_ticks_received", 0) if self._ws else 0,
            "candles": {
                "1min": candles_1m,
                "3min": candles_3m,
                "5min": candles_5m,
                "15min": candles_15m,
                "candle_counts": {"1min": candles_1m, "3min": candles_3m, "5min": candles_5m, "15min": candles_15m},
            },
            "validator": validator_stats,
            "option_polls": self._get_option_poll_count_safe(),
            "using_fallback": bool(snap.get("using_fallback", False)),
            "live_subscription_applied": self._live_subscription_applied.is_set(),
            "live_subscription_skipped": self._live_subscription_skipped,
            "validator_degraded": self._validator_degraded,
            "validator_degraded_since": self._validator_degraded_since.isoformat() if self._validator_degraded_since else None,
            "mapper_status": self._mapper.get_status() if self._mapper and hasattr(self._mapper, "get_status") else {},
            "unified_mode": True
        }

    def _get_option_poll_count_safe(self) -> int:
        try:
            if self._option_component and hasattr(self._option_component, "get_status"):
                st = self._option_component.get_status()
                if isinstance(st, dict):
                    if "poll_count" in st:
                        return int(st["poll_count"])
                    ps = st.get("poller_status") or {}
                    if isinstance(ps, dict) and "poll_count" in ps:
                        return int(ps["poll_count"])
            return 0
        except Exception:
            return 0

    def get_flow_status(self) -> Dict[str, Any]:
        return {
            "tick_seen": self._tick_seen_count,
            "tick_validated": self._tick_validated_count,
            "tick_rejected": self._tick_rejected_count,
            "spot_tick_validated": self._spot_tick_validated_count,
            "candle_close_count": self._candle_close_count,
            "option_poll_count": self._option_poll_count,
            "option_poll_success": self._option_poll_success_count,
            "option_poll_errors": self._option_poll_error_count,
            "last_tick_seen_at": self._last_tick_seen_at.isoformat() if self._last_tick_seen_at else None,
            "last_tick_validated_at": self._last_tick_validated_at.isoformat() if self._last_tick_validated_at else None,
            "last_spot_tick_at": self._last_spot_tick_at.isoformat() if self._last_spot_tick_at else None,
            "last_candle_close_at": self._last_candle_close_at.isoformat() if self._last_candle_close_at else None,
            "last_option_poll_at": self._last_option_poll_at.isoformat() if self._last_option_poll_at else None,
            "last_option_update_at": self._last_option_update_at.isoformat() if self._last_option_update_at else None,
            "last_tick_exception": self._last_tick_exception,
            "last_tick_exception_at": self._last_tick_exception_at.isoformat() if self._last_tick_exception_at else None,
        }


# ============================================================================
# Module self-test (offline behavioral sanity)
# ============================================================================
def _run_tests():
    print("=" * 70)
    print(" JUNIOR ALADDIN — Data Engine Test (Institutional, Offline)")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    state = MarketState()
    engine = DataEngine(state)

    print(" [Test 1] Create Data Engine...")
    if isinstance(engine, DataEngine):
        print(" ✅ Data Engine created")
        passed += 1
    else:
        print(" ❌ Data Engine creation failed")
        failed += 1

    print("\n [Test 2] Initial status...")
    st2 = engine.get_status()
    if (not st2["is_running"]) and st2["spot_price"] == 0:
        print(" ✅ Initial status OK")
        passed += 1
    else:
        print(f" ❌ Initial status wrong: {st2}")
        failed += 1

    class _DummyValidator:
        def validate(self, tick: Dict[str, Any]) -> Dict[str, Any]:
            ts = tick.get("exchange_timestamp") or datetime(2026, 4, 1, 10, 0, 0, tzinfo=IST)
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            return {
                "timestamp": ts,
                "token": str(tick.get("token_str") or tick.get("token") or "99926000"),
                "symbol": "NIFTY",
                "instrument_class": "INDEX",
                "ltp": float(tick.get("ltp")),
                "volume": int(tick.get("volume", -1)),
                "is_spike": False,
                "feed_gap_sec": 0.0,
                "feed_health": "HEALTHY",
                "latency_sec": 0.0,
                "is_stale": False,
                "reconnect_burst": False,
                "same_timestamp_update": False,
                "tick_size": 0.05,
                "dynamic_spike_threshold_pct": 2.0,
            }

        def get_stats(self) -> Dict[str, int]:
            return {"total_seen": 2, "total_valid": 2, "flagged_spikes": 0}

    print("\n [Test 3] Manual spot-tick pipeline updates MarketState.spot + closes candle...")
    engine._validator = _DummyValidator()  # type: ignore
    engine._candle_builder = CandleBuilder(default_token=engine._nifty_token)
    engine._feed_health = FeedHealthMonitor()

    t0 = datetime(2026, 4, 1, 10, 0, 10, tzinfo=IST)
    t1 = datetime(2026, 4, 1, 10, 1, 5, tzinfo=IST)

    fake_tick0 = {"token": engine._nifty_token, "ltp": 24500.0, "volume": 1000, "received_at": t0, "exchange_timestamp": t0}
    fake_tick1 = {"token": engine._nifty_token, "ltp": 24510.0, "volume": 500, "received_at": t1, "exchange_timestamp": t1}

    engine._process_tick(fake_tick0)
    engine._process_tick(fake_tick1)

    if float(state.spot) == 24510.0:
        print(" ✅ Spot updated")
        passed += 1
    else:
        print(f" ❌ Spot not updated: {state.spot}")
        failed += 1

    last_1m = engine._candle_builder.get_last_closed("1min", token=engine._nifty_token)
    if last_1m is not None:
        print(" ✅ Candle closed")
        passed += 1
    else:
        print(f" ❌ Candle not closed")
        failed += 1

    print("\n [Test 4] Status exposes validator.valid/rejected/spikes keys...")
    st4 = engine.get_status()
    v = st4.get("validator", {}).get("valid")
    r = st4.get("validator", {}).get("rejected")
    s = st4.get("validator", {}).get("spikes")
    if isinstance(v, int) and isinstance(r, int) and isinstance(s, int):
        print(" ✅ Validator keys exposed")
        passed += 1
    else:
        print(f" ❌ Validator keys missing: validator={st4.get('validator')}")
        failed += 1

    print("\n" + "=" * 70)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
