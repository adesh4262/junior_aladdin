"""
Junior Aladdin - NSE Data Fetcher (True Institutional Grade)
===========================================================

PURPOSE:
    Download FII/DII data from NSE and global market data from Yahoo Finance.
    Used by Narrative Engine for morning briefing and intraday updates.

MANDATES SATISFIED (Institutional Hardening + System-Level Intelligence):
    - Browser simulation retained (warmup URLs + realistic headers)
    - Persistent disk cache for FII/DII and global data
    - NO fabricated fallback estimates
    - Data freshness validation + quantified staleness: stale_age_days (always present)
    - Memory cache re-validation (global critical_ok enforced even on cached path)
    - Persistent cache integrity re-validation on load
    - Global tickers classified into CRITICAL vs OPTIONAL; overall success requires all CRITICAL
    - Global price sanity checks (price>0, abs(change_pct)<=20%)
    - Retry differentiation: retry only Timeout/ConnectionError (network); do not retry invalid/empty data
    - Lightweight NSE rate-limit throttle (min interval across calls/instances)
    - Market session awareness via helpers: pre-market does not increment NSE circuit breaker failures
    - External data health aggregation:
         get_external_data_quality() -> {nse_available, global_critical_ok, stale_age_days_max, overall_quality_score}
    - Narrative confidence multiplier:
         get_confidence_multiplier() -> float in [0.3, 1.0] decays with staleness

SCOPE LIMITS (Lean):
    - No quality scoring system (beyond lightweight health aggregation)
    - No multi-source validation
    - No corporate action awareness
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from src.utils.logger import setup_logger

# Helpers (assume exist)
try:
    from src.utils.helpers import ist_today, is_trading_day, ist_now, is_market_hours, is_pre_market
except Exception:  # pragma: no cover
    IST_TZ = timezone(timedelta(hours=5, minutes=30))

    def ist_today() -> date:  # type: ignore
        return datetime.now(IST_TZ).date()

    def ist_now() -> datetime:  # type: ignore
        return datetime.now(IST_TZ)

    def is_trading_day(d: date) -> bool:  # type: ignore
        return d.weekday() < 5

    def is_market_hours() -> bool:  # type: ignore
        now = datetime.now(IST_TZ).time()
        return (now >= datetime(2000, 1, 1, 9, 15).time()) and (now <= datetime(2000, 1, 1, 15, 30).time())

    def is_pre_market() -> bool:  # type: ignore
        now = datetime.now(IST_TZ).time()
        return (now >= datetime(2000, 1, 1, 8, 0).time()) and (now < datetime(2000, 1, 1, 9, 15).time())


# ============================================
# Constants
# ============================================
IST = timezone(timedelta(hours=5, minutes=30))

NSE_BASE_URL = "https://www.nseindia.com"
NSE_FII_DII_URL = f"{NSE_BASE_URL}/api/fiidiiTradeReact"

NSE_WARMUP_URLS = [
    NSE_BASE_URL,
    f"{NSE_BASE_URL}/market-data/live-equity-market",
    f"{NSE_BASE_URL}/report-detail/fii_dii_trading_activity",
]

CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

API_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": f"{NSE_BASE_URL}/report-detail/fii_dii_trading_activity",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

# Global tickers classification
CRITICAL_TICKERS = {"sp500", "usd_inr", "crude_oil"}
GLOBAL_TICKERS_CRITICAL = {
    "sp500": "^GSPC",
    "usd_inr": "USDINR=X",
    "crude_oil": "CL=F",
}
GLOBAL_TICKERS_OPTIONAL = {
    "sp500_futures": "ES=F",
    "nikkei": "^N225",
    "hang_seng": "^HSI",
    "gold": "GC=F",
    "us_10y_yield": "^TNX",
    "india_vix_yf": "^INDIAVIX",
}


# ============================================
# NSE Throttle (global min interval)
# ============================================
_NSE_THROTTLE_LOCK = threading.Lock()
_NSE_LAST_CALL_MONO: float = 0.0
_NSE_MIN_INTERVAL_SEC: float = 2.0


def _nse_throttle(min_interval_sec: float = _NSE_MIN_INTERVAL_SEC) -> None:
    global _NSE_LAST_CALL_MONO
    with _NSE_THROTTLE_LOCK:
        now_m = time.monotonic()
        elapsed = now_m - _NSE_LAST_CALL_MONO
        wait_sec = max(0.0, float(min_interval_sec) - elapsed)
        if wait_sec > 0:
            time.sleep(wait_sec)
        _NSE_LAST_CALL_MONO = time.monotonic()


# ============================================
# Utility helpers
# ============================================
@dataclass(frozen=True)
class MarketStatus:
    code: str  # MARKET_CLOSED / MARKET_OPEN
    should_retry: bool
    reason: str


def _market_status_now() -> MarketStatus:
    today = ist_today()
    if not is_trading_day(today):
        return MarketStatus(code="MARKET_CLOSED", should_retry=False, reason="Non-trading day (weekend/holiday)")
    return MarketStatus(code="MARKET_OPEN", should_retry=True, reason="Trading day")


def _utc_now_iso_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_date_maybe(value: Any) -> Optional[date]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    fmts = [
        "%d-%b-%Y",
        "%d-%b-%y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d-%m-%Y",
        "%d-%m-%y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue

    try:  # pragma: no cover
        from dateutil import parser as dateparser  # type: ignore

        return dateparser.parse(s, dayfirst=True).date()
    except Exception:
        return None


def _safe_float_crore(value: Any) -> float:
    try:
        if isinstance(value, str):
            v = value.replace(",", "").replace(" ", "").strip()
            if v in ("", "-"):
                return 0.0
            return float(v)
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _is_retryable_network_exc(exc: BaseException) -> bool:
    # Retry only network issues
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    inner = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if isinstance(inner, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    msg = str(exc).lower()
    return ("timeout" in msg) or ("timed out" in msg) or ("connection" in msg)


def _compute_stale_age_days(as_of: Optional[date], today: Optional[date] = None) -> float:
    """
    Quantified staleness:
      - always returns a float
      - if unknown, returns 999.0
    """
    if today is None:
        today = ist_today()
    if as_of is None:
        return 999.0
    try:
        return float((today - as_of).days)
    except Exception:
        return 999.0


def _validate_fii_cache_payload(obj: Dict[str, Any]) -> bool:
    # Required keys for FII cache integrity
    if not isinstance(obj, dict):
        return False
    if "fii_net" not in obj or "date" not in obj:
        return False
    if not str(obj.get("date", "")).strip():
        return False
    try:
        float(obj.get("fii_net", 0.0))
    except Exception:
        return False
    return True


def _validate_global_cache_payload(obj: Dict[str, Any]) -> bool:
    # Required keys for global cache integrity
    if not isinstance(obj, dict):
        return False
    for k in ("sp500", "usd_inr"):
        if k not in obj or not isinstance(obj.get(k), dict):
            return False
        if "price" not in obj[k]:
            return False
        try:
            price = float(obj[k]["price"])
        except Exception:
            return False
        if price <= 0:
            return False
    return True


def _recompute_critical_status(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for k in CRITICAL_TICKERS:
        item = payload.get(k)
        if not isinstance(item, dict) or item.get("price") is None:
            missing.append(k)
    return (len(missing) == 0), missing


def _apply_global_critical_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    critical_ok, critical_missing = _recompute_critical_status(payload)
    payload2 = dict(payload)
    payload2["critical_ok"] = bool(critical_ok)
    payload2["critical_missing"] = critical_missing
    return payload2


def _normalize_scalar(x: Any) -> float:
    """
    Safe scalar extraction avoiding FutureWarning and handling pandas Series.
    """
    # pandas/numpy scalar
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    # pandas Series (single element)
    if isinstance(x, pd.Series):
        if len(x) == 1:
            return float(x.iloc[0])
        raise ValueError("Non-scalar Series encountered")
    return float(x)


# ============================================
# NSEFetcher
# ============================================
class NSEFetcher:
    """
    Fetches external data for the Narrative Engine.

    Adds system-level intelligence:
      - stale_age_days in all outputs
      - memory cache re-validation for global critical tickers
      - persistent cache integrity checks
      - external data quality aggregation
      - narrative confidence multiplier
      - market hours awareness (pre-market does not increment circuit-breaker failures)
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        fii_cache_filename: str = "fii_dii.json",
        global_cache_filename: str = "global_cache.json",
        nse_max_attempts: int = 3,
        nse_timeout_sec: int = 15,
        nse_circuit_breaker_failures: int = 10,
        nse_circuit_breaker_cooldown_sec: int = 2 * 60 * 60,
        global_ticker_timeout_sec: int = 5,
        global_overall_timeout_sec: int = 25,
        global_min_success_tickers: int = 3,
    ):
        self._logger = setup_logger("nse_fetcher")

        self._cache_dir = cache_dir.replace("\\", "/")
        _ensure_dir(self._cache_dir)
        self._fii_cache_path = os.path.join(self._cache_dir, fii_cache_filename).replace("\\", "/")
        self._global_cache_path = os.path.join(self._cache_dir, global_cache_filename).replace("\\", "/")

        # In-memory caches
        self._fii_dii_cache: Optional[Dict[str, Any]] = None
        self._fii_dii_cache_time: Optional[datetime] = None

        self._global_cache: Optional[Dict[str, Any]] = None
        self._global_cache_day: Optional[str] = None
        self._global_cache_time: Optional[datetime] = None

        # NSE retry + circuit breaker
        self._nse_max_attempts = max(1, int(nse_max_attempts))
        self._nse_timeout_sec = max(5, int(nse_timeout_sec))
        self._nse_cb_failures = max(3, int(nse_circuit_breaker_failures))
        self._nse_cb_cooldown_sec = max(60, int(nse_circuit_breaker_cooldown_sec))

        self._consecutive_nse_failures = 0
        self._nse_skip_until: Optional[datetime] = None

        # Global fetch constraints
        self._global_ticker_timeout_sec = max(2, int(global_ticker_timeout_sec))
        self._global_overall_timeout_sec = max(5, int(global_overall_timeout_sec))
        self._global_min_success = max(1, int(global_min_success_tickers))

        # NSE session reuse
        self._session: Optional[requests.Session] = None
        self._session_ready: bool = False

        # Track last known outputs for health aggregation
        self._last_fii: Optional[Dict[str, Any]] = None
        self._last_global: Optional[Dict[str, Any]] = None

    # ============================================
    # NSE Session — Browser Simulation (retained)
    # ============================================
    def _create_or_reuse_session(self, force_new: bool = False) -> requests.Session:
        if self._session is not None and self._session_ready and not force_new:
            return self._session

        session = requests.Session()
        session.headers.update(CHROME_HEADERS)

        for i, url in enumerate(NSE_WARMUP_URLS):
            try:
                _nse_throttle()
                resp = session.get(url, timeout=self._nse_timeout_sec, allow_redirects=True)
                self._logger.info(
                    "NSE warmup page",
                    extra={
                        "step": f"{i+1}/{len(NSE_WARMUP_URLS)}",
                        "url": url.split("/")[-1] or "homepage",
                        "status": resp.status_code,
                        "cookies": len(session.cookies),
                    },
                )
            except requests.exceptions.RequestException as e:
                self._logger.debug(
                    "Warmup page failed (continuing)",
                    extra={"step": f"{i+1}/{len(NSE_WARMUP_URLS)}", "error": str(e)[:160]},
                )
            time.sleep(random.uniform(0.8, 2.2))

        session.headers.update(API_HEADERS)
        self._logger.info("NSE session ready", extra={"cookies_collected": len(session.cookies)})

        self._session = session
        self._session_ready = True
        return session

    # ============================================
    # Persistent Cache
    # ============================================
    def _save_persistent_cache(self, path: str, payload: Dict[str, Any]) -> None:
        try:
            payload2 = dict(payload)
            payload2["_cached_at_utc"] = _utc_now_iso_z()
            _atomic_write_json(path, payload2)
        except Exception as e:
            self._logger.warning("Failed to write persistent cache", extra={"path": path, "error": str(e)[:160]})

    def _load_persistent_cache(self, path: str) -> Optional[Dict[str, Any]]:
        obj = _read_json(path)
        return obj if isinstance(obj, dict) else None

    # ============================================
    # FII/DII Data from NSE
    # ============================================
    def fetch_fii_dii(self, use_cache_minutes: int = 60) -> Dict[str, Any]:
        ms = _market_status_now()
        now_ist = ist_now()

        # Memory cache (time-based) with staleness preserved; bypassable with <=0
        if use_cache_minutes is not None and int(use_cache_minutes) > 0:
            if self._fii_dii_cache is not None and self._fii_dii_cache_time is not None:
                age_sec = (now_ist - self._fii_dii_cache_time).total_seconds()
                if age_sec < int(use_cache_minutes) * 60:
                    self._logger.info("Using cached FII/DII data (memory)", extra={"cache_age_min": round(age_sec / 60, 2)})
                    self._last_fii = self._fii_dii_cache
                    return self._fii_dii_cache

        # Circuit breaker skip
        if self._nse_skip_until is not None and now_ist < self._nse_skip_until:
            self._logger.critical(
                "NSE circuit breaker active; skipping NSE fetch",
                extra={"skip_until": self._nse_skip_until.isoformat(), "consecutive_failures": self._consecutive_nse_failures},
            )
            cached = self._load_persistent_cache(self._fii_cache_path)
            if cached and _validate_fii_cache_payload(cached):
                cached2 = dict(cached)
                cached2["source"] = "persistent_cache"
                cached2["success"] = True
                cached2["stale"] = True
                cached2["error"] = ""
                cached2["error_kind"] = "NSE_CIRCUIT_BREAKER"
                cached2["market_status"] = ms.code
                cached2["should_retry"] = False

                cache_date = _parse_date_maybe(cached2.get("date"))
                cached2["stale_age_days"] = _compute_stale_age_days(cache_date, ist_today())
                self._fii_dii_cache = cached2
                self._fii_dii_cache_time = now_ist
                self._last_fii = cached2
                self._logger.warning("Using persistent cache due to NSE circuit breaker", extra={"cache_path": self._fii_cache_path})
                return cached2

            failure = {
                "fii_buy": 0.0,
                "fii_sell": 0.0,
                "fii_net": 0.0,
                "dii_buy": 0.0,
                "dii_sell": 0.0,
                "dii_net": 0.0,
                "date": "",
                "date_parsed": None,
                "source": "none",
                "success": False,
                "stale": True,
                "stale_age_days": 999.0,
                "market_status": ms.code,
                "should_retry": False,
                "error_kind": "NSE_CIRCUIT_BREAKER_NO_CACHE",
                "error": "NSE circuit breaker active and no valid persistent cache available",
            }
            self._last_fii = failure
            return failure

        # Market closed differentiation
        if ms.code == "MARKET_CLOSED":
            cached = self._load_persistent_cache(self._fii_cache_path)
            if cached and _validate_fii_cache_payload(cached):
                cached2 = dict(cached)
                cached2["source"] = "persistent_cache"
                cached2["success"] = True
                cached2["stale"] = True
                cached2["market_status"] = ms.code
                cached2["should_retry"] = False
                cached2["error_kind"] = "MARKET_CLOSED_USING_CACHE"
                cached2["error"] = ""

                cache_date = _parse_date_maybe(cached2.get("date"))
                cached2["stale_age_days"] = _compute_stale_age_days(cache_date, ist_today())

                self._fii_dii_cache = cached2
                self._fii_dii_cache_time = now_ist
                self._last_fii = cached2
                self._logger.warning("Market closed; using persistent cache for FII/DII", extra={"cache_path": self._fii_cache_path})
                return cached2

            failure = {
                "fii_buy": 0.0,
                "fii_sell": 0.0,
                "fii_net": 0.0,
                "dii_buy": 0.0,
                "dii_sell": 0.0,
                "dii_net": 0.0,
                "date": "",
                "date_parsed": None,
                "source": "none",
                "success": False,
                "stale": True,
                "stale_age_days": 999.0,
                "market_status": ms.code,
                "should_retry": False,
                "error_kind": "MARKET_CLOSED_NO_CACHE",
                "error": "Market is closed and no valid persistent FII/DII cache available",
            }
            self._last_fii = failure
            return failure

        # NSE fetch attempts
        self._logger.info("Fetching FII/DII data from NSE...", extra={"max_attempts": self._nse_max_attempts})

        last_error: str = ""
        error_kind: str = "NSE_FAILED"

        session = self._create_or_reuse_session(force_new=False)

        pre_market = bool(is_pre_market())

        for attempt in range(1, self._nse_max_attempts + 1):
            try:
                time.sleep(random.uniform(0.3, 1.0))

                _nse_throttle()
                resp = session.get(NSE_FII_DII_URL, timeout=self._nse_timeout_sec)

                ct = resp.headers.get("Content-Type", "unknown")
                self._logger.info(
                    "NSE API response",
                    extra={"status": resp.status_code, "content_type": ct, "content_length": len(resp.content), "attempt": attempt},
                )

                # Non-200 or non-JSON is treated as non-retryable (avoid bans)
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}"
                    error_kind = "HTTP_ERROR"
                    break

                if "json" not in ct and "javascript" not in ct:
                    last_error = f"Non-JSON response: {ct}"
                    error_kind = "NON_JSON_BLOCK"
                    self._logger.warning(
                        "NSE returned non-JSON content",
                        extra={"attempt": attempt, "content_type": ct, "body_preview": resp.text[:160]},
                    )
                    break

                try:
                    data = resp.json()
                except Exception as je:
                    last_error = f"JSON parse failed: {je}"
                    error_kind = "JSON_PARSE"
                    self._logger.warning("NSE JSON parse failed", extra={"attempt": attempt, "error": str(je)[:160]})
                    break

                parsed = self._parse_fii_dii_response(data)
                if not parsed.get("success"):
                    last_error = parsed.get("error", "Parse failed")
                    error_kind = "PARSE_FAILED"
                    break

                # Success
                self._consecutive_nse_failures = 0
                self._nse_skip_until = None

                today = ist_today()
                response_date = _parse_date_maybe(parsed.get("date"))
                parsed["date_parsed"] = response_date.isoformat() if response_date else None

                stale = False
                if response_date is not None and response_date < today:
                    stale = True
                    self._logger.warning(
                        "FII/DII response date is older than today (stale flag set)",
                        extra={"response_date": response_date.isoformat(), "today": today.isoformat()},
                    )

                parsed["stale"] = stale
                parsed["stale_age_days"] = _compute_stale_age_days(response_date, today)
                parsed["market_status"] = ms.code
                parsed["should_retry"] = True
                parsed["error_kind"] = ""
                parsed["error"] = ""
                parsed["source"] = "nse_api"

                self._fii_dii_cache = parsed
                self._fii_dii_cache_time = now_ist
                self._save_persistent_cache(self._fii_cache_path, parsed)

                self._last_fii = parsed
                self._logger.info(
                    "FII/DII fetched from NSE",
                    extra={"fii_net": parsed.get("fii_net"), "dii_net": parsed.get("dii_net"), "date": parsed.get("date"), "stale_age_days": parsed.get("stale_age_days")},
                )
                return parsed

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as ne:
                # Retryable only
                last_error = f"{type(ne).__name__}: {ne}"
                error_kind = "NETWORK_ERROR"
                self._logger.warning("NSE network error (retryable)", extra={"attempt": attempt, "error": str(ne)[:160]})

                # recreate session only on connection errors
                if isinstance(ne, requests.exceptions.ConnectionError):
                    session = self._create_or_reuse_session(force_new=True)

                if attempt < self._nse_max_attempts:
                    time.sleep(random.uniform(2.5, 5.5))
                    continue
                break

            except Exception as e:
                # Non retryable
                last_error = f"{type(e).__name__}: {e}"
                error_kind = "NON_RETRYABLE_EXCEPTION"
                self._logger.error("NSE non-retryable exception", extra={"attempt": attempt, "error": str(e)[:200]})
                break

        # NSE failed: circuit breaker increments ONLY if not pre-market (mandate)
        if not pre_market:
            self._consecutive_nse_failures += 1
        else:
            self._logger.info("Pre-market NSE failure does not increment circuit breaker", extra={"attempts": self._nse_max_attempts})

        self._logger.warning(
            "NSE fetch failed",
            extra={
                "consecutive_failures": self._consecutive_nse_failures,
                "threshold": self._nse_cb_failures,
                "last_error": last_error,
                "error_kind": error_kind,
                "pre_market": pre_market,
            },
        )

        if (not pre_market) and self._consecutive_nse_failures >= self._nse_cb_failures:
            self._nse_skip_until = now_ist + timedelta(seconds=self._nse_cb_cooldown_sec)
            self._logger.critical(
                "NSE circuit breaker triggered",
                extra={
                    "consecutive_failures": self._consecutive_nse_failures,
                    "skip_until": self._nse_skip_until.isoformat(),
                    "cooldown_sec": self._nse_cb_cooldown_sec,
                },
            )

        # Persistent fallback with integrity check
        cached = self._load_persistent_cache(self._fii_cache_path)
        if cached and _validate_fii_cache_payload(cached):
            cached2 = dict(cached)
            cached2["source"] = "persistent_cache"
            cached2["success"] = True
            cached2["stale"] = True
            cached2["market_status"] = ms.code
            cached2["should_retry"] = ms.should_retry
            cached2["error_kind"] = "NSE_FAILED_USING_CACHE"
            cached2["error"] = ""

            cache_date = _parse_date_maybe(cached2.get("date"))
            cached2["stale_age_days"] = _compute_stale_age_days(cache_date, ist_today())

            self._fii_dii_cache = cached2
            self._fii_dii_cache_time = now_ist
            self._last_fii = cached2
            self._logger.warning("Using persistent FII/DII cache due to NSE failure", extra={"cache_path": self._fii_cache_path, "stale_age_days": cached2["stale_age_days"]})
            return cached2

        failure = {
            "fii_buy": 0.0,
            "fii_sell": 0.0,
            "fii_net": 0.0,
            "dii_buy": 0.0,
            "dii_sell": 0.0,
            "dii_net": 0.0,
            "date": "",
            "date_parsed": None,
            "source": "none",
            "success": False,
            "stale": True,
            "stale_age_days": 999.0,
            "market_status": ms.code,
            "should_retry": ms.should_retry,
            "error_kind": error_kind,
            "error": f"NSE failed: {last_error}",
        }
        self._fii_dii_cache = failure
        self._fii_dii_cache_time = now_ist
        self._last_fii = failure
        return failure

    def _parse_fii_dii_response(self, data: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "fii_buy": 0.0,
            "fii_sell": 0.0,
            "fii_net": 0.0,
            "dii_buy": 0.0,
            "dii_sell": 0.0,
            "dii_net": 0.0,
            "date": "",
            "source": "nse_api",
            "success": False,
            "error": "",
        }

        try:
            records: List[Dict[str, Any]] = []
            if isinstance(data, list):
                records = [r for r in data if isinstance(r, dict)]
            elif isinstance(data, dict):
                recs = data.get("data", data.get("records", []))
                if isinstance(recs, list):
                    records = [r for r in recs if isinstance(r, dict)]
                else:
                    result["error"] = "Unexpected dict response format"
                    return result
            else:
                result["error"] = "Unexpected response type"
                return result

            found_fii = False
            found_dii = False

            for record in records:
                category = str(record.get("category", "")).upper()
                dt = record.get("date", "")
                if dt and not result["date"]:
                    result["date"] = str(dt)

                if "FII" in category or "FPI" in category:
                    result["fii_buy"] = _safe_float_crore(record.get("buyValue", 0))
                    result["fii_sell"] = _safe_float_crore(record.get("sellValue", 0))
                    result["fii_net"] = _safe_float_crore(record.get("netValue", 0))
                    found_fii = True
                elif "DII" in category:
                    result["dii_buy"] = _safe_float_crore(record.get("buyValue", 0))
                    result["dii_sell"] = _safe_float_crore(record.get("sellValue", 0))
                    result["dii_net"] = _safe_float_crore(record.get("netValue", 0))
                    found_dii = True

            if found_fii or found_dii:
                result["success"] = True
            else:
                result["error"] = "FII/DII categories not found in response"

        except Exception as e:
            result["error"] = str(e)

        return result

    # ============================================
    # Global Market Data from Yahoo Finance
    # ============================================
    def fetch_global_data(self, use_cache_minutes: int = 30) -> Dict[str, Any]:
        now_ist = ist_now()
        today_key = ist_today().isoformat()

        # Memory cache path MUST be re-validated for critical_ok
        if use_cache_minutes is not None and int(use_cache_minutes) > 0:
            if self._global_cache is not None and self._global_cache_time is not None and self._global_cache_day == today_key:
                age_sec = (now_ist - self._global_cache_time).total_seconds()
                if age_sec < int(use_cache_minutes) * 60:
                    cached = dict(self._global_cache)
                    cached = _apply_global_critical_fields(cached)  # enforce critical fields on cached payload
                    if bool(cached.get("critical_ok")) and _validate_global_cache_payload(cached):
                        self._logger.info("Using cached global data (memory; validated)", extra={"cache_age_min": round(age_sec / 60, 2)})
                        self._global_cache = cached
                        self._last_global = cached
                        return cached

                    self._logger.warning("Memory cached global data failed critical validation; forcing fresh fetch")
                    # fall through to fresh fetch

        self._logger.info("Fetching global market data from Yahoo Finance...")

        tickers: Dict[str, str] = {}
        tickers.update(GLOBAL_TICKERS_CRITICAL)
        tickers.update(GLOBAL_TICKERS_OPTIONAL)

        result: Dict[str, Any] = {
            "success": False,
            "source": "yahoo",
            "fetched_at": now_ist.isoformat(),
            "as_of_ist_date": today_key,
            "errors": [],
            "fetched_count": 0,
            "total_tickers": len(tickers),
            "critical_ok": False,
            "critical_missing": [],
            "stale": False,
            "stale_age_days": 0.0,
        }

        def _fetch_one(name: str, symbol: str) -> Tuple[str, Dict[str, Any]]:
            """
            - Retry only on network timeouts/connection errors (max 1 retry)
            - Reject invalid price <=0
            - Reject abs(change_pct) > 20%
            """
            last_exc: Optional[BaseException] = None
            for attempt in range(2):
                try:
                    df = yf.download(symbol, period="2d", interval="1d", progress=False, threads=False, auto_adjust=False)
                    if df is None or len(df) == 0:
                        raise ValueError("Empty history")

                    # yfinance sometimes returns DataFrame with multiindex columns; handle Close robustly
                    if "Close" not in df.columns:
                        raise ValueError("Missing Close column")

                    close_raw = df["Close"].iloc[-1]
                    close = _normalize_scalar(close_raw)

                    prev = None
                    if len(df) >= 2:
                        prev_raw = df["Close"].iloc[-2]
                        prev = _normalize_scalar(prev_raw)

                    if close <= 0:
                        raise ValueError("Invalid price <= 0")

                    change_pct = None
                    if prev is not None and prev > 0:
                        change_pct = (close - prev) / prev

                    if change_pct is not None and abs(change_pct) > 0.20:
                        raise ValueError(f"Suspicious change_pct {change_pct}")

                    payload = {
                        "price": round(close, 4) if name in ("usd_inr", "us_10y_yield") else round(close, 2),
                        "previous_close": (
                            round(prev, 4)
                            if (prev is not None and name in ("usd_inr", "us_10y_yield"))
                            else (round(prev, 2) if prev is not None else None)
                        ),
                        "change_pct": round(float(change_pct), 6) if change_pct is not None else None,
                    }
                    return name, payload

                except ValueError:
                    # non-retryable invalid/empty data
                    raise
                except Exception as e:
                    last_exc = e
                    if _is_retryable_network_exc(e) and attempt == 0:
                        time.sleep(0.5)
                        continue
                    raise RuntimeError(f"{name} failed: {str(last_exc)[:160]}")
            raise RuntimeError(f"{name} failed: {str(last_exc)[:160] if last_exc else 'unknown'}")

        futures = {}
        start_times = {}
        overall_start = time.monotonic()

        ordered_items: List[Tuple[str, str]] = list(GLOBAL_TICKERS_CRITICAL.items()) + list(GLOBAL_TICKERS_OPTIONAL.items())
        max_workers = min(8, len(ordered_items))

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="global_yf") as ex:
            for name, sym in ordered_items:
                fut = ex.submit(_fetch_one, name, sym)
                futures[fut] = name
                start_times[fut] = time.monotonic()

            pending = set(futures.keys())
            while pending and (time.monotonic() - overall_start) < self._global_overall_timeout_sec:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)

                for fut in done:
                    name = futures.get(fut, "unknown")
                    try:
                        k, payload = fut.result(timeout=0)
                        result[k] = payload
                        result["fetched_count"] += 1
                    except Exception as e:
                        result[name] = {"price": None, "previous_close": None, "change_pct": None}
                        result["errors"].append(f"{name}: {str(e)[:160]}")

                for fut in list(pending):
                    elapsed = time.monotonic() - start_times.get(fut, overall_start)
                    if elapsed > self._global_ticker_timeout_sec:
                        name = futures.get(fut, "unknown")
                        fut.cancel()
                        pending.discard(fut)
                        result[name] = {"price": None, "previous_close": None, "change_pct": None}
                        result["errors"].append(f"{name}: TIMEOUT>{self._global_ticker_timeout_sec}s")

            for fut in list(pending):
                name = futures.get(fut, "unknown")
                fut.cancel()
                result[name] = {"price": None, "previous_close": None, "change_pct": None}
                result["errors"].append(f"{name}: OVERALL_TIMEOUT>{self._global_overall_timeout_sec}s")

        # Critical enforcement
        critical_ok, critical_missing = _recompute_critical_status(result)
        result["critical_ok"] = bool(critical_ok)
        result["critical_missing"] = critical_missing

        min_count = max(self._global_min_success, len(CRITICAL_TICKERS))
        result["success"] = bool(critical_ok and int(result["fetched_count"]) >= min_count)

        if result["success"]:
            self._save_persistent_cache(self._global_cache_path, result)
            self._global_cache = result
            self._global_cache_time = now_ist
            self._global_cache_day = today_key
            self._last_global = result

            self._logger.info(
                "Global data fetched",
                extra={
                    "fetched": result["fetched_count"],
                    "total": len(tickers),
                    "errors": len(result["errors"]),
                    "critical_ok": result["critical_ok"],
                },
            )
            return result

        # Persistent cache fallback: must be re-validated AND critical recomputed
        cached = self._load_persistent_cache(self._global_cache_path)
        if cached and _validate_global_cache_payload(cached):
            cached2 = _apply_global_critical_fields(cached)
            if bool(cached2.get("critical_ok")):
                # staleness quantified from as_of_ist_date if present else attempt parse fetched_at date
                as_of = _parse_date_maybe(cached2.get("as_of_ist_date")) or _parse_date_maybe(str(cached2.get("fetched_at", ""))[:10])
                stale_age = _compute_stale_age_days(as_of, ist_today())

                cached2["source"] = "persistent_cache"
                cached2["success"] = True
                cached2["stale"] = True
                cached2["stale_age_days"] = stale_age
                cached2["error"] = ""
                cached2["error_kind"] = "YAHOO_FAILED_USING_CACHE"

                self._global_cache = cached2
                self._global_cache_time = now_ist
                self._global_cache_day = today_key
                self._last_global = cached2

                self._logger.warning(
                    "Yahoo global fetch failed; using persistent cache",
                    extra={"cache_path": self._global_cache_path, "critical_missing": cached2.get("critical_missing"), "stale_age_days": stale_age},
                )
                return cached2

            self._logger.warning("Persistent global cache exists but is missing CRITICAL tickers; discarding", extra={"cache_path": self._global_cache_path})

        # Total failure
        result["source"] = "yahoo"
        result["stale"] = True
        result["stale_age_days"] = 999.0

        self._global_cache = result
        self._global_cache_time = now_ist
        self._global_cache_day = today_key
        self._last_global = result

        self._logger.warning(
            "Global data fetch failed; no valid persistent cache available",
            extra={"errors": len(result["errors"]), "fetched": result["fetched_count"], "critical_missing": result.get("critical_missing")},
        )
        return result

    # ============================================
    # System-level intelligence exports
    # ============================================
    def get_external_data_quality(self) -> Dict[str, Any]:
        """
        Feed-health integration:
          Returns structured dict with required keys:
            - nse_available
            - global_critical_ok
            - stale_age_days_max
            - overall_quality_score (0-100)

        Scoring (per mandate idea):
          - NSE availability: up to 40
          - Global critical_ok: up to 40
          - Staleness component: up to 20 (decays to 0 at 7+ days)
        """
        fii = self._last_fii or self._fii_dii_cache or {}
        glob = self._last_global or self._global_cache or {}

        fii_success = bool(fii.get("success", False))
        fii_source = str(fii.get("source", "none"))
        fii_age = float(fii.get("stale_age_days", 999.0)) if fii.get("stale_age_days") is not None else 999.0

        global_success = bool(glob.get("success", False))
        global_source = str(glob.get("source", "none"))
        global_critical_ok = bool(glob.get("critical_ok", False))
        global_age = float(glob.get("stale_age_days", 999.0)) if glob.get("stale_age_days") is not None else 999.0

        stale_age_days_max = max(fii_age, global_age)

        # NSE component: prefer real NSE API; cache counts but lower credit
        if fii_success and fii_source == "nse_api":
            nse_component = 40
        elif fii_success and fii_source == "persistent_cache":
            nse_component = 20
        else:
            nse_component = 0

        # Global component: prefer yahoo; cache lower credit; must be critical_ok
        if global_success and global_critical_ok and global_source == "yahoo":
            global_component = 40
        elif global_success and global_critical_ok and global_source == "persistent_cache":
            global_component = 20
        else:
            global_component = 0

        # Staleness component: 20 at 0 days, linearly decays to 0 at 7+ days
        s = min(max(stale_age_days_max, 0.0), 7.0)
        stale_component = int(round(20.0 * (1.0 - (s / 7.0)), 0))

        overall = int(max(0, min(100, nse_component + global_component + stale_component)))

        return {
            "nse_available": bool(fii_success and fii_source == "nse_api" and fii_age <= 1.0),
            "global_critical_ok": bool(global_success and global_critical_ok),
            "stale_age_days_max": float(stale_age_days_max),
            "overall_quality_score": overall,
            # Extra diagnostics (safe)
            "fii_source": fii_source,
            "global_source": global_source,
            "fii_stale_age_days": fii_age,
            "global_stale_age_days": global_age,
        }

    def get_confidence_multiplier(self) -> float:
        """
        Narrative confidence multiplier based on external data freshness.
        - 1.0 for fresh (0 days)
        - linearly decays to 0.3 at 7+ days stale
        """
        q = self.get_external_data_quality()
        stale = float(q.get("stale_age_days_max", 999.0))
        if stale <= 0.0:
            return 1.0
        if stale >= 7.0:
            return 0.3
        # linear from 1.0 -> 0.3 across 0..7
        return float(1.0 - (stale / 7.0) * (1.0 - 0.3))

    # Backward-compatible aliases for older naming expectations (if any)
    def get_health_status(self) -> Dict[str, Any]:
        return self.get_external_data_quality()

    # ============================================
    # Combined Fetch / Status
    # ============================================
    def fetch_all(self) -> Dict[str, Any]:
        fii = self.fetch_fii_dii()
        glob = self.fetch_global_data()
        return {"fii_dii": fii, "global_markets": glob}

    def get_status(self) -> Dict[str, Any]:
        now_ist = ist_now()
        fii_age_min = None
        if self._fii_dii_cache_time:
            fii_age_min = round((now_ist - self._fii_dii_cache_time).total_seconds() / 60, 2)
        global_age_min = None
        if self._global_cache_time:
            global_age_min = round((now_ist - self._global_cache_time).total_seconds() / 60, 2)

        cb_active = self._nse_skip_until is not None and now_ist < self._nse_skip_until

        return {
            "fii_dii_cached": self._fii_dii_cache is not None,
            "fii_dii_source": (self._fii_dii_cache.get("source", "none") if self._fii_dii_cache else "none"),
            "fii_dii_success": (self._fii_dii_cache.get("success", False) if self._fii_dii_cache else False),
            "fii_dii_cache_age_min": fii_age_min,
            "fii_dii_persistent_cache_path": self._fii_cache_path,
            "global_cached": self._global_cache is not None,
            "global_source": (self._global_cache.get("source", "none") if self._global_cache else "none"),
            "global_success": (self._global_cache.get("success", False) if self._global_cache else False),
            "global_cache_age_min": global_age_min,
            "global_persistent_cache_path": self._global_cache_path,
            "nse_circuit_breaker_active": cb_active,
            "nse_consecutive_failures": self._consecutive_nse_failures,
            "nse_skip_until": (self._nse_skip_until.isoformat() if self._nse_skip_until else None),
            "external_quality": self.get_external_data_quality(),
            "confidence_multiplier": self.get_confidence_multiplier(),
        }


# ============================================
# Module Self-Test
# ============================================
if __name__ == "__main__":
    print("=" * 76)
    print("  JUNIOR ALADDIN — NSE Fetcher (True Institutional Grade) Self-Test")
    print("=" * 76)
    print()

    nse = NSEFetcher()

    print("[Test 1] fetch_fii_dii()")
    fii = nse.fetch_fii_dii(use_cache_minutes=0)
    print("  success:", fii.get("success"))
    print("  source:", fii.get("source"))
    print("  stale:", fii.get("stale"))
    print("  stale_age_days:", fii.get("stale_age_days"))
    print("  date:", fii.get("date"))

    print("\n[Test 2] fetch_global_data()")
    glob = nse.fetch_global_data(use_cache_minutes=0)
    print("  success:", glob.get("success"))
    print("  source:", glob.get("source"))
    print("  critical_ok:", glob.get("critical_ok"))
    print("  critical_missing:", glob.get("critical_missing"))
    print("  stale:", glob.get("stale"))
    print("  stale_age_days:", glob.get("stale_age_days"))
    print("  fetched_count:", glob.get("fetched_count"), "/", glob.get("total_tickers"))

    print("\n[Test 3] external health + multiplier")
    health = nse.get_external_data_quality()
    print("  health:", health)
    print("  confidence_multiplier:", nse.get_confidence_multiplier())

    print("\nDone.")