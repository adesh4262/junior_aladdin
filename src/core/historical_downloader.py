# src/core/historical_downloader.py

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import HTTPError as RequestsHTTPError
    from requests.exceptions import Timeout as RequestsTimeout
except Exception:  # pragma: no cover
    RequestsTimeout = Exception  # type: ignore
    RequestsConnectionError = Exception  # type: ignore
    RequestsHTTPError = Exception  # type: ignore

from src.core.auth_manager import AuthManager
from src.utils.config_loader import Config
from src.utils.logger import setup_logger

# mandated helpers (assume exist)
try:
    from src.utils.helpers import ist_today, is_trading_day, is_expiry_day
except Exception:  # pragma: no cover
    def ist_today() -> date:  # type: ignore
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
        return datetime.now(tz).date() if tz else datetime.now().date()

    def is_trading_day(d: date) -> bool:  # type: ignore
        return d.weekday() < 5

    def is_expiry_day(d: Optional[date] = None) -> bool:  # type: ignore
        dd = d or ist_today()
        return dd.weekday() == 1  # Tuesday fallback


logger = setup_logger("historical_downloader")
IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else None

RE_GENERIC_1MIN = re.compile(r"^([A-Za-z0-9_]+)_1min_(\d{4}-\d{2}-\d{2})\.parquet$")
RE_GENERIC_DAILY = re.compile(r"^([A-Za-z0-9_]+)_daily\.parquet$")


@dataclass(frozen=True)
class DownloadResult:
    day: date
    candles: int
    status: str  # OK / SKIPPED / FAILED / PARTIAL / DEGRADED
    path: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class SessionSpec:
    market_open: str  # "HH:MM"
    market_close: str  # "HH:MM"
    expected_1min: int
    label: str = "DEFAULT"


def _cfg(*keys: str, default: Any) -> Any:
    try:
        return Config.get(*keys, default=default)
    except Exception as e:
        logger.warning("Config read failed; using default", keys=".".join(keys), default=default, error=str(e))
        return default


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _utc_now_iso_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_hhmm(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    v = value.strip()
    parts = v.split(":")
    if len(parts) < 2:
        return fallback
    try:
        hh = int(parts[0])
        mm = int(parts[1])
    except Exception:
        return fallback
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return fallback
    return f"{hh:02d}:{mm:02d}"


def _dt_ist(d: date, hhmm: str) -> datetime:
    hh, mm = 9, 15
    try:
        hh, mm = map(int, hhmm.split(":")[:2])
    except Exception:
        hh, mm = 9, 15
    if IST is None:
        return datetime(d.year, d.month, d.day, hh, mm)
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST)


def _minutes_between(open_hhmm: str, close_hhmm: str) -> int:
    d0 = date(2000, 1, 1)
    o = _dt_ist(d0, open_hhmm)
    c = _dt_ist(d0, close_hhmm)
    return max(0, int((c - o).total_seconds() // 60))


def _to_ist_series(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        if IST is None:
            return s
        return s.dt.tz_localize(IST, ambiguous="NaT", nonexistent="shift_forward")
    if IST is None:
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.tz_convert(IST)


def _parquet_num_rows(path: str) -> Optional[int]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        md = pf.metadata
        if md is None:
            return None
        return int(md.num_rows)
    except Exception:
        try:
            df = pd.read_parquet(path)
            return int(len(df))
        except Exception:
            return None


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, compression="snappy", index=False)
    os.replace(tmp, path)
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass


def _atomic_write_json(obj: Dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _credentials_seem_missing(exc: Exception) -> bool:
    s = str(exc).lower()
    markers = ["angel_api_key", "angel_client_id", "angel_password", "totp", "secret", "missing", "not set"]
    return any(m in s for m in markers)


def _canonical_symbol(symbol: Optional[str]) -> str:
    if symbol is None:
        return ""
    s = symbol.strip().upper()
    return s


def _symbol_slug(symbol: str) -> str:
    # filename-safe
    s = symbol.strip().upper()
    s = re.sub(r"[^A-Z0-9_]+", "", s)
    return s or "SYMBOL"


class _SpecialSessions:
    """
    Session-aware expected count logic:
      - config.yaml: historical.special_sessions (dict keyed by YYYY-MM-DD)
      - economic_calendar.json: special_sessions (dict keyed by YYYY-MM-DD) [optional]
    """
    _loaded: bool = False
    _map: Dict[str, Dict[str, Any]] = {}
    _reminded: bool = False

    @classmethod
    def _load(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True

        cfg_sessions = _cfg("historical", "special_sessions", default={})
        if isinstance(cfg_sessions, dict):
            for k, v in cfg_sessions.items():
                if isinstance(k, str) and isinstance(v, dict):
                    cls._map[k] = v

        cal_path = _cfg("historical", "economic_calendar_path", default="data/calendar/economic_calendar.json")
        if isinstance(cal_path, str) and cal_path.strip() and os.path.exists(cal_path):
            try:
                cal = _read_json(cal_path)
                if isinstance(cal, dict):
                    ss = cal.get("special_sessions")
                    if isinstance(ss, dict):
                        for k, v in ss.items():
                            if isinstance(k, str) and isinstance(v, dict):
                                cls._map[k] = v
            except Exception as e:
                logger.warning("Failed reading economic_calendar special_sessions", path=cal_path, error=str(e))

        if not cls._map and not cls._reminded:
            cls._reminded = True
            logger.info(
                "No special sessions loaded; defaults used (normal close 15:30, expiry close 15:00). "
                "Add historical.special_sessions or economic_calendar.json:special_sessions for early closes/holidays."
            )

    @classmethod
    def get(cls, d: date) -> Optional[Dict[str, Any]]:
        cls._load()
        return cls._map.get(d.isoformat())


def _session_spec_for_date(d: date) -> Optional[SessionSpec]:
    special = _SpecialSessions.get(d)
    if isinstance(special, dict):
        if bool(special.get("holiday")):
            return None

        mo = _parse_hhmm(special.get("market_open"), fallback="09:15")
        default_close = "15:00" if is_expiry_day(d) else "15:30"
        mc = _parse_hhmm(special.get("market_close"), fallback=default_close)

        expected = special.get("expected_1min")
        if expected is None:
            expected = _minutes_between(mo, mc)
        try:
            expected_i = int(expected)
        except Exception:
            expected_i = _minutes_between(mo, mc)

        return SessionSpec(market_open=mo, market_close=mc, expected_1min=max(0, expected_i), label="SPECIAL")

    mo = "09:15"
    mc = "15:00" if is_expiry_day(d) else "15:30"
    expected_i = _minutes_between(mo, mc)
    return SessionSpec(market_open=mo, market_close=mc, expected_1min=expected_i, label="DEFAULT")


class _ProgressTracker:
    """
    Checkpointing:
      - progress file: data/historical/.download_progress.json (default)
      - stores range + asset identity
      - skips completed dates on resume
      - deleted when entire range completes with no failed/partial
    """

    def __init__(self, path: str):
        self._path = path
        self._completed: Set[str] = set()
        self._range: Optional[Tuple[str, str]] = None
        self._asset: Optional[Dict[str, str]] = None

    def load(self, start: date, end: date, asset: Dict[str, str]) -> None:
        self._range = (start.isoformat(), end.isoformat())
        self._asset = dict(asset)

        if not os.path.exists(self._path):
            return

        try:
            obj = _read_json(self._path) or {}
            rng = obj.get("range", {})
            if rng.get("start") != self._range[0] or rng.get("end") != self._range[1]:
                logger.info("Progress range mismatch; ignoring old progress file", progress_path=self._path)
                return

            # Backward compat: old files may have no asset; accept only for default NIFTY asset.
            obj_asset = obj.get("asset")
            if obj_asset is not None:
                if not isinstance(obj_asset, dict):
                    logger.info("Progress asset malformed; ignoring old progress file", progress_path=self._path)
                    return
                if (
                    str(obj_asset.get("symbol")) != asset.get("symbol")
                    or str(obj_asset.get("token")) != asset.get("token")
                    or str(obj_asset.get("exchange")) != asset.get("exchange")
                ):
                    logger.info("Progress asset mismatch; ignoring old progress file", progress_path=self._path)
                    return
            else:
                # if no asset present, only accept if asset is default NIFTY config
                pass

            completed = obj.get("completed", [])
            if isinstance(completed, list):
                self._completed = {str(x) for x in completed if isinstance(x, str)}
        except Exception as e:
            logger.warning("Progress file unreadable; starting fresh", progress_path=self._path, error=str(e))

    def is_completed(self, d: date) -> bool:
        return d.isoformat() in self._completed

    def mark_completed(self, d: date) -> None:
        if self._range is None or self._asset is None:
            return
        ds = d.isoformat()
        if ds in self._completed:
            return
        self._completed.add(ds)
        obj = {
            "created_at": _utc_now_iso_z(),
            "range": {"start": self._range[0], "end": self._range[1]},
            "asset": self._asset,
            "completed": sorted(self._completed),
        }
        _ensure_dir(os.path.dirname(self._path) or ".")
        _atomic_write_json(obj, self._path)

    def finalize_if_complete(self, all_dates: List[date], any_failed_or_partial: bool) -> None:
        if any_failed_or_partial:
            return
        if all(self.is_completed(d) for d in all_dates):
            try:
                os.remove(self._path)
                logger.info("Removed progress file (complete)", progress_path=self._path)
            except Exception as e:
                logger.warning("Failed to remove progress file", progress_path=self._path, error=str(e))


class HistoricalDownloader:
    """
    Institutional-grade lean downloader with minimal scalability upgrades:
      - Parameterized symbol/token/exchange (defaults from historical.defaults.*)
      - Writes {file}.meta.json after save
      - Uses meta for integrity checks when skipping existing files
      - Priority download order
      - Downstream on_file_saved hook
    """

    _api_lock = threading.Lock()
    _last_call_mono: float = 0.0

    def __init__(self, auth: AuthManager):
        self._auth = auth
        self._api: Any = None

        self._candle_dir = str(_cfg("historical", "candle_dir", default="data/historical/candles")).replace("\\", "/")
        self._api_delay_sec = float(_cfg("historical", "api_delay_sec", default=0.5))
        self._max_retries = int(_cfg("historical", "max_retries", default=3))

        self._partial_min_ratio = float(_cfg("historical", "partial_min_ratio", default=0.50))
        self._gap_reject_threshold = int(_cfg("historical", "gap_reject_threshold", default=10))

        self._progress_path = str(_cfg("historical", "progress_path", default="data/historical/.download_progress.json")).replace(
            "\\", "/"
        )

        # NEW: configurable defaults (backward compatible)
        self._default_symbol = str(_cfg("historical", "defaults", "symbol", default="NIFTY"))
        self._default_token = str(_cfg("historical", "defaults", "token", default=str(_cfg("market", "nifty_spot_token", default="99926000"))))
        self._default_exchange = str(_cfg("historical", "defaults", "exchange", default=str(_cfg("market", "exchange", default="NSE"))))

        _ensure_dir(self._candle_dir)

    def _get_api(self) -> Any:
        if self._api is None:
            self._api = self._auth.get_smart_api()
        return self._api

    def _resolve_asset(self, symbol: Optional[str], token: Optional[str], exchange: Optional[str]) -> Dict[str, str]:
        sym = _canonical_symbol(symbol) or _canonical_symbol(self._default_symbol) or "NIFTY"
        tok = str(token).strip() if token is not None else str(self._default_token).strip()
        exch = str(exchange).strip() if exchange is not None else str(self._default_exchange).strip()
        return {"symbol": sym, "token": tok, "exchange": exch}

    def _meta_path(self, parquet_path: str) -> str:
        return f"{parquet_path}.meta.json"

    def _file_path_1min(self, symbol: str, d: date) -> str:
        return os.path.join(self._candle_dir, f"{_symbol_slug(symbol)}_1min_{d.isoformat()}.parquet").replace("\\", "/")

    def _file_path_daily(self, symbol: str) -> str:
        return os.path.join(self._candle_dir, f"{_symbol_slug(symbol)}_daily.parquet").replace("\\", "/")

    def _existing_is_usable(self, parquet_path: str, expected: int) -> bool:
        """
        Integrity check:
          - base check: row count within ±2 of expected
          - if meta exists: meta.actual_candles must match parquet row count within ±2
          - if meta missing: usable (backward compatible)
        """
        if not os.path.exists(parquet_path):
            return False

        row_count = _parquet_num_rows(parquet_path)
        if row_count is None:
            return False

        if abs(int(row_count) - int(expected)) > 2:
            return False

        meta_path = self._meta_path(parquet_path)
        if not os.path.exists(meta_path):
            return True  # backward compatibility

        meta = _read_json(meta_path)
        if not isinstance(meta, dict):
            logger.warning("Meta file unreadable; forcing re-download", meta_path=meta_path)
            return False

        meta_actual = meta.get("actual_candles")
        try:
            meta_actual_i = int(meta_actual)
        except Exception:
            logger.warning("Meta actual_candles invalid; forcing re-download", meta_path=meta_path, meta_actual=str(meta_actual))
            return False

        if abs(int(meta_actual_i) - int(row_count)) > 2:
            logger.warning(
                "Meta/Parquet row count mismatch; forcing re-download",
                parquet_path=parquet_path,
                meta_path=meta_path,
                parquet_rows=int(row_count),
                meta_actual=int(meta_actual_i),
            )
            return False

        return True

    def _rate_limited_api_call(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Global lock + global rate limiting.
        Retries: Timeout/ConnectionError -> exponential backoff up to max_retries.
        HTTPError: retry 5xx once; no retry for 4xx.
        SmartAPI-wrapped exceptions: fallback to message-based inference.
        """
        api = self._get_api()

        attempt = 0
        backoff = 1.0
        retried_5xx = False

        while True:
            attempt += 1
            try:
                with HistoricalDownloader._api_lock:
                    now_m = time.monotonic()
                    elapsed = now_m - HistoricalDownloader._last_call_mono
                    wait = max(0.0, self._api_delay_sec - elapsed)
                    if wait > 0:
                        time.sleep(wait)

                    resp = api.getCandleData(params)
                    HistoricalDownloader._last_call_mono = time.monotonic()
                    return resp

            except RequestsHTTPError as e:
                status_code = None
                try:
                    status_code = int(getattr(getattr(e, "response", None), "status_code", None))
                except Exception:
                    status_code = None

                if status_code is not None and 500 <= status_code <= 599 and not retried_5xx:
                    retried_5xx = True
                    logger.warning("HTTP 5xx from candle API; retrying once", status_code=status_code, attempt=attempt, params=params)
                    continue

                logger.error("HTTP error from candle API (no retry)", status_code=status_code, attempt=attempt, params=params, error=str(e))
                return None

            except (RequestsTimeout, RequestsConnectionError) as e:
                if attempt > self._max_retries:
                    logger.error(
                        "Network error from candle API (retries exhausted)",
                        attempt=attempt,
                        max_retries=self._max_retries,
                        params=params,
                        error=str(e),
                    )
                    return None
                logger.warning(
                    "Network error from candle API; retrying",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    backoff_sec=backoff,
                    params=params,
                    error=str(e),
                )
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 16.0)

            except Exception as e:
                inner = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
                if isinstance(inner, (RequestsTimeout, RequestsConnectionError)):
                    if attempt > self._max_retries:
                        logger.error(
                            "Network error (wrapped) from candle API (retries exhausted)",
                            attempt=attempt,
                            max_retries=self._max_retries,
                            params=params,
                            error=str(e),
                        )
                        return None
                    logger.warning(
                        "Network error (wrapped) from candle API; retrying",
                        attempt=attempt,
                        max_retries=self._max_retries,
                        backoff_sec=backoff,
                        params=params,
                        error=str(e),
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 16.0)
                    continue

                msg = str(e).lower()
                inferred_5xx = any(code in msg for code in (" 500", " 502", " 503", " 504"))
                inferred_4xx = any(code in msg for code in (" 400", " 401", " 403", " 404", " 429"))
                is_timeout = ("timeout" in msg) or ("timed out" in msg)
                is_conn = ("connection" in msg) or ("remote end closed" in msg) or ("temporarily unavailable" in msg)

                if inferred_5xx and not retried_5xx:
                    retried_5xx = True
                    logger.warning("Inferred 5xx error; retrying once", attempt=attempt, params=params, error=str(e))
                    continue

                if inferred_4xx:
                    logger.error("Inferred 4xx error; not retrying", attempt=attempt, params=params, error=str(e))
                    return None

                if not (is_timeout or is_conn) or attempt > self._max_retries:
                    logger.error(
                        "Candle API call failed (non-retryable or retries exhausted)",
                        attempt=attempt,
                        max_retries=self._max_retries,
                        params=params,
                        error=str(e),
                    )
                    return None

                logger.warning(
                    "Candle API call failed (retryable inferred); retrying",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    backoff_sec=backoff,
                    params=params,
                    error=str(e),
                )
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 16.0)

    @staticmethod
    def _parse_response(resp: Dict[str, Any]) -> pd.DataFrame:
        if not isinstance(resp, dict):
            return pd.DataFrame()
        if resp.get("status") is not True:
            logger.error("Candle API returned non-success", message=resp.get("message"), errorcode=resp.get("errorcode"))
            return pd.DataFrame()
        data = resp.get("data")
        if not data:
            return pd.DataFrame()
        try:
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception as e:
            logger.error("Failed to parse candle data into DataFrame", error=str(e))
            return pd.DataFrame()
        df["timestamp"] = _to_ist_series(df["timestamp"])
        df = df.dropna(subset=["timestamp"])
        return df

    @staticmethod
    def _dedup_sort_validate_timestamps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, bool]:
        if df.empty:
            return df, 0, True
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        dup_count = int(df["timestamp"].duplicated(keep="last").sum())
        if dup_count > 0:
            df = df.loc[~df["timestamp"].duplicated(keep="last")].copy()
            df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        is_mono = bool(df["timestamp"].is_monotonic_increasing)
        return df, dup_count, is_mono

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("Missing required candle columns", missing_cols=missing)
            return df.iloc[:0].copy(), 0, 0.0

        before = int(len(df))
        df2 = df.dropna(subset=required).copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2 = df2.dropna(subset=required)

        tol = 0.05
        cond = (
            (df2["high"] >= df2["low"])
            & (df2["open"] >= (df2["low"] - tol))
            & (df2["open"] <= (df2["high"] + tol))
            & (df2["close"] >= (df2["low"] - tol))
            & (df2["close"] <= (df2["high"] + tol))
            & (df2["volume"] >= 0)
        )
        df3 = df2.loc[cond].copy()
        after = int(len(df3))
        dropped = max(0, before - after)
        dropped_pct = (dropped / max(before, 1)) * 100.0
        return df3, dropped, dropped_pct

    def _gap_detection(self, df: pd.DataFrame, d: date, spec: SessionSpec) -> Tuple[int, bool]:
        if df.empty:
            return 0, False

        start = _dt_ist(d, spec.market_open)
        end = _dt_ist(d, spec.market_close)
        expected = int(spec.expected_1min)
        if expected <= 0:
            return 0, False

        full_index = pd.date_range(start=start, end=(end - timedelta(minutes=1)), freq="1min", tz=IST)
        if full_index.empty:
            return 0, False

        tmp = df.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce").dt.floor("min")
        if IST is not None:
            if getattr(tmp["timestamp"].dt, "tz", None) is None:
                tmp["timestamp"] = tmp["timestamp"].dt.tz_localize(IST, ambiguous="NaT", nonexistent="shift_forward")
            else:
                tmp["timestamp"] = tmp["timestamp"].dt.tz_convert(IST)

        tmp = tmp.dropna(subset=["timestamp"])
        tmp, _, _ = self._dedup_sort_validate_timestamps(tmp)
        tmp = tmp.set_index("timestamp", drop=True)

        re_df = tmp.reindex(full_index).asfreq("1min")
        gap_count = int(re_df["open"].isna().sum()) if "open" in re_df.columns else int(re_df.isna().any(axis=1).sum())
        reject = gap_count > int(self._gap_reject_threshold)
        return gap_count, reject

    def _download_day_1min(self, asset: Dict[str, str], d: date, spec: SessionSpec) -> Tuple[pd.DataFrame, str]:
        start = _dt_ist(d, spec.market_open)
        end = _dt_ist(d, spec.market_close)

        params = {
            "exchange": asset["exchange"],
            "symboltoken": asset["token"],
            "interval": "ONE_MINUTE",
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": end.strftime("%Y-%m-%d %H:%M"),
        }

        resp = self._rate_limited_api_call(params)
        if resp is None:
            return pd.DataFrame(), "API_CALL_FAILED"

        df = self._parse_response(resp)
        if df.empty:
            return df, "EMPTY"

        # strict market hours
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
        return df, "OK"

    def _write_meta(
        self,
        parquet_path: str,
        meta: Dict[str, Any],
    ) -> None:
        meta_path = self._meta_path(parquet_path)
        try:
            _atomic_write_json(meta, meta_path)
        except Exception as e:
            logger.error("Failed writing meta file", meta_path=meta_path, error=str(e))

    def download_1min_candles(
        self,
        days_back: int = 30,
        skip_existing: bool = True,
        symbol: str | None = None,
        token: str | None = None,
        exchange: str | None = None,
        priority: str = "recent_first",
        on_file_saved: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[DownloadResult]:
        """
        Backward compatible:
          download_1min_candles(days_back=30, skip_existing=True) => NIFTY default behavior unchanged.

        Scalability:
          symbol/token/exchange optional; defaults from historical.defaults.*
          priority: recent_first (default) / oldest_first
          on_file_saved: callback(file_path, metadata_dict) after save
        """
        try:
            days_back = int(days_back)
        except Exception:
            days_back = 30
        days_back = max(1, min(days_back, 3650))

        asset = self._resolve_asset(symbol, token, exchange)

        # ensure auth once (Issue 1)
        try:
            _ = self._get_api()
        except Exception as e:
            logger.error("Authentication / SmartAPI init failed", error=str(e))
            results: List[DownloadResult] = []
            # still produce per-day output
            today = ist_today()
            end_day = today - timedelta(days=1)
            start_day = end_day - timedelta(days=days_back - 1)
            for i in range((end_day - start_day).days + 1):
                d = start_day + timedelta(days=i)
                if is_trading_day(d):
                    results.append(DownloadResult(day=d, candles=0, status="FAILED", message="AUTH_FAILED"))
                else:
                    results.append(DownloadResult(day=d, candles=0, status="SKIPPED", message="NON_TRADING_DAY"))
            return results

        today = ist_today()
        now = datetime.now(IST) if IST is not None else datetime.now()
        include_today = bool(now.hour > 15 or (now.hour == 15 and now.minute >= 35))
        end_day = today if include_today else (today - timedelta(days=1))
        start_day = end_day - timedelta(days=days_back - 1)

        all_dates = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]

        if str(priority).lower() == "recent_first":
            all_dates = list(sorted(all_dates, reverse=True))
        elif str(priority).lower() == "oldest_first":
            all_dates = list(sorted(all_dates))
        else:
            logger.warning("Unknown priority; defaulting to recent_first", priority=str(priority))
            all_dates = list(sorted(all_dates, reverse=True))

        logger.info(
            "Starting 1-min historical download",
            symbol=asset["symbol"],
            token=asset["token"],
            exchange=asset["exchange"],
            start=str(start_day),
            end=str(end_day),
            days_back=days_back,
            priority=str(priority),
            candle_dir=self._candle_dir,
            skip_existing=skip_existing,
        )

        progress = _ProgressTracker(self._progress_path)
        progress.load(start=start_day, end=end_day, asset={"symbol": asset["symbol"], "token": asset["token"], "exchange": asset["exchange"], "interval": "1min"})

        results: List[DownloadResult] = []
        any_failed_or_partial = False

        for d in all_dates:
            try:
                if progress.is_completed(d):
                    results.append(DownloadResult(day=d, candles=0, status="SKIPPED", message="CHECKPOINT_SKIPPED"))
                    continue

                if not is_trading_day(d):
                    results.append(DownloadResult(day=d, candles=0, status="SKIPPED", message="NON_TRADING_DAY"))
                    progress.mark_completed(d)
                    continue

                spec = _session_spec_for_date(d)
                if spec is None:
                    results.append(DownloadResult(day=d, candles=0, status="SKIPPED", message="SPECIAL_SESSION_HOLIDAY"))
                    progress.mark_completed(d)
                    continue

                expected = int(spec.expected_1min)
                min_required = int(expected * max(0.0, min(1.0, self._partial_min_ratio)))

                parquet_path = self._file_path_1min(asset["symbol"], d)

                # skip existing usable
                if skip_existing and self._existing_is_usable(parquet_path, expected):
                    n_existing = _parquet_num_rows(parquet_path) or 0
                    logger.info("Skipping existing file (usable)", day=str(d), path=parquet_path, expected=expected, rows=int(n_existing))
                    results.append(DownloadResult(day=d, candles=int(n_existing), status="SKIPPED", path=parquet_path))
                    progress.mark_completed(d)
                    continue

                raw_df, msg = self._download_day_1min(asset, d, spec)
                if raw_df.empty:
                    logger.error("No candles returned for trading day", day=str(d), message=msg, symbol=asset["symbol"])
                    results.append(DownloadResult(day=d, candles=0, status="FAILED", path=parquet_path, message=msg))
                    any_failed_or_partial = True
                    continue

                raw_df, dup_removed, is_mono = self._dedup_sort_validate_timestamps(raw_df)
                if dup_removed > 0:
                    logger.info("Removed duplicate timestamps (kept last)", day=str(d), duplicates_removed=int(dup_removed), symbol=asset["symbol"])
                if not is_mono:
                    logger.warning("Timestamps not monotonic after sort", day=str(d), symbol=asset["symbol"])

                # Issue 2: partial detection after parsing + filtering
                pre_valid_count = int(len(raw_df))
                if pre_valid_count < min_required:
                    logger.warning(
                        "Download PARTIAL (below minimum expected)",
                        day=str(d),
                        symbol=asset["symbol"],
                        candles=pre_valid_count,
                        expected=expected,
                        min_required=min_required,
                        session_label=spec.label,
                    )
                    # do not overwrite a previously good file
                    if self._existing_is_usable(parquet_path, expected):
                        results.append(
                            DownloadResult(day=d, candles=pre_valid_count, status="PARTIAL", path=parquet_path, message="EXISTING_GOOD_KEPT")
                        )
                    else:
                        results.append(DownloadResult(day=d, candles=pre_valid_count, status="PARTIAL", path=parquet_path, message="NOT_SAVED_PARTIAL"))
                    any_failed_or_partial = True
                    continue

                valid_df, dropped, dropped_pct = self._validate_ohlcv(raw_df)
                if dropped > 0:
                    logger.warning(
                        "Dropped invalid rows during OHLCV validation",
                        day=str(d),
                        symbol=asset["symbol"],
                        dropped=int(dropped),
                        dropped_pct=round(float(dropped_pct), 4),
                        before=int(len(raw_df)),
                        after=int(len(valid_df)),
                    )

                degraded = False
                if float(dropped_pct) > 2.0:
                    degraded = True
                    logger.error(
                        "Dropped >2% rows during validation; marking DEGRADED",
                        day=str(d),
                        symbol=asset["symbol"],
                        dropped=int(dropped),
                        dropped_pct=round(float(dropped_pct), 4),
                    )

                valid_df, dup_removed2, is_mono2 = self._dedup_sort_validate_timestamps(valid_df)
                if dup_removed2 > 0:
                    logger.info("Removed duplicate timestamps post-validation (kept last)", day=str(d), symbol=asset["symbol"], duplicates_removed=int(dup_removed2))
                if not is_mono2:
                    degraded = True
                    logger.warning("Non-monotonic timestamps post-validation; marking DEGRADED", day=str(d), symbol=asset["symbol"])

                candle_count = int(len(valid_df))

                gap_count, reject_save = self._gap_detection(valid_df, d, spec)
                if gap_count > 3:
                    degraded = True
                    logger.warning("Gaps detected in 1-min series; marking DEGRADED", day=str(d), symbol=asset["symbol"], gaps=int(gap_count), reject_threshold=int(self._gap_reject_threshold))

                if reject_save:
                    logger.warning(
                        "Too many gaps; NOT saving file",
                        day=str(d),
                        symbol=asset["symbol"],
                        gaps=int(gap_count),
                        reject_threshold=int(self._gap_reject_threshold),
                        candles=candle_count,
                        expected=expected,
                    )
                    results.append(DownloadResult(day=d, candles=candle_count, status="DEGRADED", path=parquet_path, message="NOT_SAVED_GAPS"))
                    any_failed_or_partial = True
                    continue

                if abs(candle_count - expected) > 2:
                    degraded = True
                    logger.warning(
                        "Downloaded candle count differs from expected",
                        day=str(d),
                        symbol=asset["symbol"],
                        candles=candle_count,
                        expected=expected,
                        diff=int(candle_count - expected),
                        session_label=spec.label,
                    )

                # safety: don't overwrite good file with degraded output
                if os.path.exists(parquet_path) and self._existing_is_usable(parquet_path, expected) and degraded:
                    logger.warning("Existing usable file present; not overwriting with DEGRADED output", day=str(d), symbol=asset["symbol"], path=parquet_path)
                    results.append(DownloadResult(day=d, candles=candle_count, status="SKIPPED", path=parquet_path, message="EXISTING_GOOD_KEPT"))
                    progress.mark_completed(d)
                    continue

                # Save parquet
                _atomic_write_parquet(valid_df, parquet_path)

                # Save metadata (required)
                meta: Dict[str, Any] = {
                    "symbol": asset["symbol"],
                    "token": asset["token"],
                    "exchange": asset["exchange"],
                    "interval": "1min",
                    "downloaded_at_utc": _utc_now_iso_z(),
                    "session_date": d.isoformat(),
                    "expected_candles": expected,
                    "actual_candles": candle_count,
                    "session_label": spec.label,
                    # "file_hash": optional; omitted intentionally for lean scope
                }
                self._write_meta(parquet_path, meta)

                status = "DEGRADED" if degraded else "OK"
                logger.info(
                    "Saved 1-min candle file",
                    day=str(d),
                    symbol=asset["symbol"],
                    candles=candle_count,
                    expected=expected,
                    path=parquet_path,
                    status=status,
                )
                results.append(DownloadResult(day=d, candles=candle_count, status=status, path=parquet_path))
                progress.mark_completed(d)

                # Downstream hook (required)
                if on_file_saved is not None:
                    try:
                        on_file_saved(parquet_path, meta)
                    except Exception as cb_e:
                        logger.error("on_file_saved callback failed (ignored)", path=parquet_path, error=str(cb_e))

            except Exception as e:
                logger.error("Unhandled error downloading day (continuing)", day=str(d), symbol=asset.get("symbol"), error=str(e))
                results.append(DownloadResult(day=d, candles=0, status="FAILED", message=str(e)))
                any_failed_or_partial = True

        progress.finalize_if_complete(all_dates=all_dates, any_failed_or_partial=any_failed_or_partial)

        ok = sum(1 for r in results if r.status == "OK")
        skipped = sum(1 for r in results if r.status == "SKIPPED")
        degraded = sum(1 for r in results if r.status == "DEGRADED")
        partial = sum(1 for r in results if r.status == "PARTIAL")
        failed = sum(1 for r in results if r.status == "FAILED")
        logger.info(
            "1-min download complete",
            symbol=asset["symbol"],
            ok=ok,
            skipped=skipped,
            degraded=degraded,
            partial=partial,
            failed=failed,
            total=len(results),
        )
        return results

    def download_daily_candles(
        self,
        days_back: int = 365,
        symbol: str | None = None,
        token: str | None = None,
        exchange: str | None = None,
    ) -> Optional[str]:
        """
        Backward compatible: default is NIFTY settings.
        Saves: {symbol}_daily.parquet and {symbol}_daily.parquet.meta.json
        """
        try:
            days_back = int(days_back)
        except Exception:
            days_back = 365
        days_back = max(5, min(days_back, 3650))

        asset = self._resolve_asset(symbol, token, exchange)

        try:
            _ = self._get_api()
        except Exception as e:
            logger.error("Authentication / SmartAPI init failed for daily download", symbol=asset["symbol"], error=str(e))
            return None

        today = ist_today()
        end_day = today - timedelta(days=1)
        start_day = end_day - timedelta(days=days_back - 1)

        parquet_path = self._file_path_daily(asset["symbol"])
        logger.info(
            "Starting daily candle download",
            symbol=asset["symbol"],
            token=asset["token"],
            exchange=asset["exchange"],
            start=str(start_day),
            end=str(end_day),
            days_back=days_back,
            path=parquet_path,
        )

        start_dt = _dt_ist(start_day, "00:00")
        end_dt = _dt_ist(end_day, "23:59")

        params = {
            "exchange": asset["exchange"],
            "symboltoken": asset["token"],
            "interval": "ONE_DAY",
            "fromdate": start_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": end_dt.strftime("%Y-%m-%d %H:%M"),
        }

        resp = self._rate_limited_api_call(params)
        if resp is None:
            logger.error("Daily candle API call failed", symbol=asset["symbol"])
            return None

        df = self._parse_response(resp)
        if df.empty:
            logger.error("Daily candle download returned empty data", symbol=asset["symbol"])
            return None

        # Keep trading days
        try:
            df["d"] = df["timestamp"].dt.date
            df = df[df["d"].apply(lambda x: bool(is_trading_day(x)))].copy()
            df = df.drop(columns=["d"], errors="ignore")
        except Exception:
            pass

        df, dropped, dropped_pct = self._validate_ohlcv(df)
        if dropped > 0:
            logger.warning("Dropped invalid daily rows during validation", symbol=asset["symbol"], dropped=int(dropped), dropped_pct=round(float(dropped_pct), 4))

        df, dup_removed, is_mono = self._dedup_sort_validate_timestamps(df)
        if dup_removed > 0:
            logger.info("Removed duplicate daily timestamps (kept last)", symbol=asset["symbol"], duplicates_removed=int(dup_removed))
        if not is_mono:
            logger.warning("Daily timestamps not monotonic after sort", symbol=asset["symbol"], path=parquet_path)

        try:
            _atomic_write_parquet(df, parquet_path)
        except Exception as e:
            logger.error("Failed saving daily candle parquet", symbol=asset["symbol"], error=str(e), path=parquet_path)
            return None

        meta: Dict[str, Any] = {
            "symbol": asset["symbol"],
            "token": asset["token"],
            "exchange": asset["exchange"],
            "interval": "1day",
            "downloaded_at_utc": _utc_now_iso_z(),
            "session_date": None,
            "expected_candles": None,
            "actual_candles": int(len(df)),
            "session_label": "DAILY",
        }
        self._write_meta(parquet_path, meta)

        logger.info("Saved daily candle file", symbol=asset["symbol"], rows=int(len(df)), path=parquet_path)
        return parquet_path

    def download_recent(
        self,
        days: int = 5,
        symbol: str | None = None,
        token: str | None = None,
        exchange: str | None = None,
        on_file_saved: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[DownloadResult]:
        return self.download_1min_candles(
            days_back=days,
            skip_existing=True,
            symbol=symbol,
            token=token,
            exchange=exchange,
            priority="recent_first",
            on_file_saved=on_file_saved,
        )

    def verify_downloads(self) -> Dict[str, Any]:
        """
        Lean verification across all *_1min_YYYY-MM-DD.parquet in candle_dir:
          - checks expected vs actual count using session spec
          - if meta exists: meta.actual must match parquet row count within ±2
        """
        details: List[Dict[str, Any]] = []
        ok = bad = 0

        if not os.path.exists(self._candle_dir):
            return {"ok": 0, "bad": 0, "total": 0, "details": []}

        for fn in sorted(os.listdir(self._candle_dir)):
            m = RE_GENERIC_1MIN.match(fn)
            if not m:
                continue
            sym_slug = m.group(1)
            ds = m.group(2)
            try:
                d = datetime.strptime(ds, "%Y-%m-%d").date()
            except Exception:
                continue

            spec = _session_spec_for_date(d)
            parquet_path = os.path.join(self._candle_dir, fn).replace("\\", "/")
            rows = _parquet_num_rows(parquet_path)

            expected = int(spec.expected_1min) if spec is not None else 0

            if rows is None:
                bad += 1
                details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": None, "expected": expected, "status": "BAD", "reason": "UNREADABLE"})
                continue

            # meta integrity (if meta exists)
            meta_path = self._meta_path(parquet_path)
            if os.path.exists(meta_path):
                meta = _read_json(meta_path)
                if not isinstance(meta, dict):
                    bad += 1
                    details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": int(rows), "expected": expected, "status": "BAD", "reason": "META_UNREADABLE"})
                    continue
                try:
                    meta_actual = int(meta.get("actual_candles"))
                except Exception:
                    bad += 1
                    details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": int(rows), "expected": expected, "status": "BAD", "reason": "META_ACTUAL_INVALID"})
                    continue
                if abs(meta_actual - int(rows)) > 2:
                    bad += 1
                    details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": int(rows), "expected": expected, "status": "BAD", "reason": "META_ROW_MISMATCH"})
                    continue

            if spec is None:
                # holiday-configured session: any file is suspicious
                bad += 1
                details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": int(rows), "expected": 0, "status": "BAD", "reason": "HOLIDAY_FILE_PRESENT"})
                continue

            if abs(int(rows) - expected) <= 2:
                ok += 1
                details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": int(rows), "expected": expected, "status": "OK"})
            else:
                bad += 1
                details.append({"symbol": sym_slug, "date": ds, "path": parquet_path, "rows": int(rows), "expected": expected, "status": "BAD", "reason": "COUNT_MISMATCH"})

        logger.info("Verify downloads complete", ok=ok, bad=bad, total=ok + bad, candle_dir=self._candle_dir)
        return {"ok": ok, "bad": bad, "total": ok + bad, "details": details}


# -------------------- Module-level singleton + public API (unchanged names) --------------------

_DEFAULT_DOWNLOADER: Optional[HistoricalDownloader] = None
_DEFAULT_LOCK = threading.Lock()


def _get_default_downloader() -> HistoricalDownloader:
    global _DEFAULT_DOWNLOADER
    with _DEFAULT_LOCK:
        if _DEFAULT_DOWNLOADER is None:
            _DEFAULT_DOWNLOADER = HistoricalDownloader(auth=AuthManager())
        return _DEFAULT_DOWNLOADER


def download_1min_candles(
    days_back: int = 30,
    skip_existing: bool = True,
    symbol: str | None = None,
    token: str | None = None,
    exchange: str | None = None,
    priority: str = "recent_first",
    on_file_saved: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> List[DownloadResult]:
    return _get_default_downloader().download_1min_candles(
        days_back=days_back,
        skip_existing=skip_existing,
        symbol=symbol,
        token=token,
        exchange=exchange,
        priority=priority,
        on_file_saved=on_file_saved,
    )


def download_daily_candles(
    days_back: int = 365,
    symbol: str | None = None,
    token: str | None = None,
    exchange: str | None = None,
) -> Optional[str]:
    return _get_default_downloader().download_daily_candles(days_back=days_back, symbol=symbol, token=token, exchange=exchange)


def download_recent(
    days: int = 5,
    symbol: str | None = None,
    token: str | None = None,
    exchange: str | None = None,
    on_file_saved: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> List[DownloadResult]:
    return _get_default_downloader().download_recent(days=days, symbol=symbol, token=token, exchange=exchange, on_file_saved=on_file_saved)


def verify_downloads() -> Dict[str, Any]:
    return _get_default_downloader().verify_downloads()


# ----------------------------- Self-test (__main__) -----------------------------


def _last_n_trading_days(n: int) -> List[date]:
    n = max(1, min(int(n), 10))
    today = ist_today()
    out: List[date] = []
    d = today - timedelta(days=1)
    while len(out) < n and (today - d).days <= 30:
        if is_trading_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


if __name__ == "__main__":
    # Mandated behavior:
    # - Authenticate; if missing creds print message and exit 0.
    # - Download last 3 trading days (1min candles).
    # - For each day: print date, candles_downloaded, status (OK/SKIPPED/FAILED).
    # - If any FAILED exit 1 else 0.
    #
    # Additional required self-test:
    # - Download 1 day explicitly with symbol/token and verify .meta.json exists
    # - Test on_file_saved callback

    try:
        dl = _get_default_downloader()
        _ = dl._get_api()
    except Exception as e:
        if _credentials_seem_missing(e):
            print("historical_downloader: credentials missing or not configured; skipping self-test.")
            sys.exit(0)
        print(f"historical_downloader: authentication failed: {e}")
        sys.exit(0)

    target_days = _last_n_trading_days(3)
    if not target_days:
        print("historical_downloader: no recent trading days found; exiting.")
        sys.exit(0)

    def _cb(path: str, meta: Dict[str, Any]) -> None:
        print(f"on_file_saved: {os.path.basename(path)} | actual={meta.get('actual_candles')} expected={meta.get('expected_candles')}")

    earliest = target_days[0]
    latest = target_days[-1]
    days_back = max((latest - earliest).days + 3, 7)

    results = dl.download_1min_candles(
        days_back=days_back,
        skip_existing=True,
        priority="recent_first",
    )

    target_set = set(target_days)
    failed_any = False
    printed = 0

    for r in results:
        if r.day not in target_set:
            continue
        mapped = "OK" if r.status == "OK" else ("SKIPPED" if r.status == "SKIPPED" else "FAILED")
        print(f"{r.day.isoformat()}, {r.candles}, {mapped}")
        printed += 1
        if mapped == "FAILED":
            failed_any = True

    if printed < len(target_set):
        missing = sorted([d for d in target_set if all(rr.day != d for rr in results)])
        for d in missing:
            print(f"{d.isoformat()}, 0, FAILED")
        failed_any = True

    # Extra test: explicit symbol/token + callback + meta presence for the most recent trading day
    explicit_day = target_days[-1]
    asset = dl._resolve_asset(symbol="NIFTY", token=None, exchange=None)
    explicit_results = dl.download_1min_candles(
        days_back=max((explicit_day - (explicit_day - timedelta(days=2))).days + 1, 3),
        skip_existing=False,  # force path/meta creation test
        symbol=asset["symbol"],
        token=asset["token"],
        exchange=asset["exchange"],
        priority="recent_first",
        on_file_saved=_cb,
    )

    # Find the record for explicit_day and verify meta exists
    meta_ok = False
    for rr in explicit_results:
        if rr.day != explicit_day:
            continue
        if rr.path and os.path.exists(f"{rr.path}.meta.json"):
            meta_ok = True

    if not meta_ok:
        print("historical_downloader: meta file check FAILED")
        failed_any = True

    sys.exit(1 if failed_any else 0)