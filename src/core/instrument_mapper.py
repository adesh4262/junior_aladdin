# src/core/instrument_mapper.py

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from src.utils.config_loader import Config
from src.utils.helpers import round_to_strike
from src.utils.logger import setup_logger

IST = timezone(timedelta(hours=5, minutes=30))

INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

INSTRUMENT_CACHE_PATH = os.path.join("data", "live", "instrument_master.json")
INSTRUMENT_CACHE_META_PATH = os.path.join("data", "live", "instrument_master_meta.json")
INSTRUMENT_BACKUP_PATH = os.path.join("data", "live", "instrument_master_backup.json")

# Mandated canonical key:
# (strike_suffix, expiry_date, opt_type, series_identifier)
OptionKey = Tuple[int, date, str, str]


@dataclass(frozen=True)
class _RegexSpec:
    pattern: re.Pattern
    uses_named_groups: bool
    raw_pattern: str


@dataclass(frozen=True)
class ParsedSymbol:
    index: str
    expiry_date: date
    opt_type: str
    series_identifier: str
    strike_suffix_str: str  # always 3 digits
    strike_suffix: int
    full_strike: int


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return int(x)
        return int(float(str(x).strip()))
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(x)
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip())
    except Exception:
        return None


class InstrumentMapper:
    """
    COMPLETE REWRITE (CAPITAL-SAFE, FAIL-CLOSED)

    CRITICAL MANDATES:
      1) UNIQUE KEY: (strike_suffix, expiry_date, opt_type, series_identifier)
         - series_identifier extracted from symbol via regex capturing group.
         - strike_suffix extracted via regex capturing group (fixed 3 digits).
         - If series_identifier cannot be extracted -> skip instrument.
      2) NO SILENT DUPLICATE RESOLUTION:
         - If duplicate canonical key with different token => reject BOTH entries; log CRITICAL.
      3) FAIL CLOSED ON LOW QUALITY:
         - parse_failure_rate = parsing_errors / total_nfo_instruments_processed
         - if parse_failure_rate > 0.20 OR mapped_count < expected_min_mapped:
             is_built=False, return empty dict, log CRITICAL refusal.
      4) REMOVE DEGRADED MODES: no mapping_quality / is_degraded flags.
      5) STRICT PARSING:
         - regex validated at init on known-good sample; if fails, raise exception.
         - log parse failures for first 100, then periodically.
      6) PRESERVE PUBLIC API:
         - signatures unchanged
         - get_subscription_list returns only spot/VIX if is_built=False
    """

    def __init__(self) -> None:
        self._logger = setup_logger("instrument_mapper")
        self._lock = threading.RLock()

        # Supported indices
        supported = Config.get("market", "supported_indices", default=["NIFTY"])
        if isinstance(supported, str):
            supported = [supported]
        if not isinstance(supported, list) or not supported:
            supported = ["NIFTY"]
        self._supported_indices: List[str] = [str(x).strip().upper() for x in supported if str(x).strip()]
        if not self._supported_indices:
            self._supported_indices = ["NIFTY"]
        self._default_index: str = self._supported_indices[0]

        # Always-include subscription tokens
        self._spot_token = str(Config.get("market", "nifty_spot_token", default="99926000"))
        self._vix_token = str(Config.get("market", "india_vix_token", default="26017"))

        # Subscription controls
        self._strike_interval = int(Config.get("market", "strike_interval", default=50))
        self._option_strikes_range = int(Config.get("data", "option_strikes_range", default=5))
        self._min_atm_range = int(Config.get("instrument_mapper", "min_atm_range", default=2))

        # Backup max age
        self._fallback_max_age_days = int(Config.get("instrument_mapper", "fallback_max_age_days", default=7))

        # Instrument validation
        allowed_lots = Config.get("instrument_mapper", "allowed_lot_sizes", default=[25, 50, 65, 75])
        if not isinstance(allowed_lots, list) or not allowed_lots:
            allowed_lots = [25, 50, 65, 75]
        self._allowed_lot_sizes: Set[int] = {int(x) for x in allowed_lots if _safe_int(x) is not None}
        if not self._allowed_lot_sizes:
            self._allowed_lot_sizes = {25, 50, 65, 75}

        self._tick_size_min = float(Config.get("instrument_mapper", "tick_size_min", default=0.01))
        self._tick_size_max = float(Config.get("instrument_mapper", "tick_size_max", default=0.10))

        # Fail-closed thresholds
        self._expected_min_mapped = int(Config.get("instrument_mapper", "expected_min_mapped", default=200))
        self._parse_failure_threshold = 0.20  # mandated constant

        # Regex patterns per index (must capture series_identifier)
        custom_patterns = Config.get("instrument_mapper", "symbol_regex_by_index", default={})
        if not isinstance(custom_patterns, dict):
            custom_patterns = {}

        custom_samples = Config.get("instrument_mapper", "regex_samples_by_index", default={})
        if not isinstance(custom_samples, dict):
            custom_samples = {}
        self._regex_samples_by_index: Dict[str, str] = {str(k).upper(): str(v) for k, v in custom_samples.items() if v}

        self._regex_by_index: Dict[str, _RegexSpec] = {}
        for idx in self._supported_indices:
            raw_pat = custom_patterns.get(idx)
            if not isinstance(raw_pat, str) or not raw_pat.strip():
                raw_pat = (
                    rf"^{re.escape(idx)}"
                    rf"(?P<expiry>\d{{2}}[A-Z]{{3}}(?:\d{{2}}|20\d{{2}}))"
                    rf"(?P<series>\d+)"
                    rf"(?P<strike>\d{{3}})"
                    rf"(?P<opt_type>CE|PE)$"
                )

            try:
                compiled = re.compile(raw_pat)
            except Exception as e:
                raise ValueError(f"InstrumentMapper: invalid regex for index={idx}. pattern={raw_pat}. error={e}") from e

            uses_named = all(name in compiled.groupindex for name in ("expiry", "series", "strike", "opt_type"))
            if not uses_named:
                if compiled.groups != 4:
                    raise ValueError(
                        f"InstrumentMapper: regex for {idx} must have named groups "
                        f"(expiry,series,strike,opt_type) OR exactly 4 positional groups. "
                        f"Got groups={compiled.groups} pattern={raw_pat}"
                    )

            self._regex_by_index[idx] = _RegexSpec(pattern=compiled, uses_named_groups=uses_named, raw_pattern=raw_pat)

        # Validate regex at init against known-good sample (MANDATE)
        self._validate_regexes_on_startup()

        # ===== State =====
        self.is_built: bool = False
        self.total_instruments_mapped: int = 0

        self.cache_source: str = "none"  # cache/download/backup/none
        self.cache_age_days: Optional[int] = None
        self.cache_date: Optional[date] = None
        self.is_stale_cache: bool = False

        # Build stats (status only)
        self.last_parse_failure_rate: Optional[float] = None
        self.last_processed_nfo_count: int = 0
        self.last_parsing_errors: int = 0
        self.last_validation_failures: int = 0
        self.last_duplicate_rejections: int = 0

        # Maps
        self._opt_map_by_index: Dict[str, Dict[OptionKey, Dict[str, Any]]] = {}
        self._legacy_map_by_index: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {}
        self._legacy_map: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

        self._expiry_dates_by_index: Dict[str, List[date]] = {}
        self._current_expiry_by_index: Dict[str, Optional[date]] = {}
        self._next_expiry_by_index: Dict[str, Optional[date]] = {}

        # Reverse lookup token -> details (always keep spot/vix)
        self._token_details: Dict[str, Dict[str, Any]] = {
            self._spot_token: {
                "token": self._spot_token,
                "symbol": self._default_index,
                "index": self._default_index,
                "expiry": None,
                "expiry_date": None,
                "opt_type": None,
                "series_identifier": None,
                "strike_suffix": None,
                "strike": None,
                "lot_size": int(Config.get("market", "lot_size", default=65) or 65),
                "tick_size": 0.05,
            },
            self._vix_token: {
                "token": self._vix_token,
                "symbol": "INDIAVIX",
                "index": "INDIAVIX",
                "expiry": None,
                "expiry_date": None,
                "opt_type": None,
                "series_identifier": None,
                "strike_suffix": None,
                "strike": None,
                "lot_size": int(Config.get("market", "lot_size", default=65) or 65),
                "tick_size": 0.05,
            },
        }

    # ==========================
    # Regex startup validation
    # ==========================
    def _default_sample_for_index(self, idx: str) -> str:
        if idx.upper() == "NIFTY":
            return "NIFTY28APR2621400PE"
        if idx.upper() == "BANKNIFTY":
            return "BANKNIFTY28APR2648500CE"
        return f"{idx.upper()}28APR2621400PE"

    def _expected_for_builtin_sample(self, idx: str, sample: str) -> Optional[Dict[str, Any]]:
        if idx.upper() == "NIFTY" and sample == "NIFTY28APR2621400PE":
            return {
                "series_identifier": "21",
                "strike_suffix": 400,
                "full_strike": 21400,
                "opt_type": "PE",
                "expiry_date": date(2026, 4, 28),
            }
        if idx.upper() == "BANKNIFTY" and sample == "BANKNIFTY28APR2648500CE":
            return {
                "series_identifier": "48",
                "strike_suffix": 500,
                "full_strike": 48500,
                "opt_type": "CE",
                "expiry_date": date(2026, 4, 28),
            }
        return None

    def _validate_regexes_on_startup(self) -> None:
        for idx, spec in self._regex_by_index.items():
            sample = self._regex_samples_by_index.get(idx.upper(), self._default_sample_for_index(idx))
            parsed = self._parse_symbol(idx, sample)
            if parsed is None:
                raise ValueError(
                    f"InstrumentMapper: regex validation FAILED for index={idx}. "
                    f"pattern={spec.raw_pattern} sample={sample}"
                )
            if not parsed.series_identifier.isdigit():
                raise ValueError(
                    f"InstrumentMapper: series_identifier not numeric for index={idx}. "
                    f"pattern={spec.raw_pattern} sample={sample} series={parsed.series_identifier}"
                )

            expected = self._expected_for_builtin_sample(idx, sample)
            if expected is not None:
                if parsed.series_identifier != expected["series_identifier"]:
                    raise ValueError(
                        f"InstrumentMapper: wrong series_identifier for index={idx}. "
                        f"expected={expected['series_identifier']} got={parsed.series_identifier} "
                        f"pattern={spec.raw_pattern} sample={sample}"
                    )
                if parsed.strike_suffix != expected["strike_suffix"]:
                    raise ValueError(
                        f"InstrumentMapper: wrong strike_suffix for index={idx}. "
                        f"expected={expected['strike_suffix']} got={parsed.strike_suffix} "
                        f"pattern={spec.raw_pattern} sample={sample}"
                    )
                if parsed.full_strike != expected["full_strike"]:
                    raise ValueError(
                        f"InstrumentMapper: wrong full_strike for index={idx}. "
                        f"expected={expected['full_strike']} got={parsed.full_strike} "
                        f"pattern={spec.raw_pattern} sample={sample}"
                    )
                if parsed.opt_type != expected["opt_type"]:
                    raise ValueError(
                        f"InstrumentMapper: wrong opt_type for index={idx}. "
                        f"expected={expected['opt_type']} got={parsed.opt_type} "
                        f"pattern={spec.raw_pattern} sample={sample}"
                    )
                if parsed.expiry_date != expected["expiry_date"]:
                    raise ValueError(
                        f"InstrumentMapper: wrong expiry_date for index={idx}. "
                        f"expected={expected['expiry_date']} got={parsed.expiry_date} "
                        f"pattern={spec.raw_pattern} sample={sample}"
                    )

    # ==========================
    # Master data management
    # ==========================
    def _ist_today(self) -> date:
        return datetime.now(IST).date()

    def _is_trading_day_simple(self, d: date) -> bool:
        return d.weekday() < 5

    def _atomic_write_json(self, path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def _file_age_days(self, path: str) -> Optional[int]:
        try:
            if not os.path.exists(path):
                return None
            mtime = float(os.path.getmtime(path))
            return int((time.time() - mtime) // 86400)
        except Exception:
            return None

    def _validate_master_integrity(self, data: Any) -> bool:
        if not isinstance(data, list) or len(data) < 1000:
            return False
        required = {"token", "symbol", "exch_seg"}
        ok = 0
        for inst in data[:100]:
            if isinstance(inst, dict) and required.issubset(inst.keys()):
                ok += 1
        return ok >= 10

    def _load_json_list(self, path: str) -> Optional[List[Dict[str, Any]]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not self._validate_master_integrity(data):
                self._logger.error("Master integrity invalid; refusing file", path=path)
                return None
            return [x for x in data if isinstance(x, dict)]
        except Exception as e:
            self._logger.error("Failed to load master file", path=path, error=str(e))
            return None

    def _read_cache_meta(self) -> Dict[str, Any]:
        if not os.path.isfile(INSTRUMENT_CACHE_META_PATH):
            return {}
        try:
            with open(INSTRUMENT_CACHE_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta if isinstance(meta, dict) else {}
        except Exception as e:
            self._logger.warning("Failed to read cache meta", error=str(e))
            return {}

    def _write_cache(self, instruments: List[Dict[str, Any]], cache_for_date: date) -> None:
        self._atomic_write_json(INSTRUMENT_CACHE_PATH, instruments)
        meta = {
            "cache_date": cache_for_date.strftime("%Y-%m-%d"),
            "downloaded_at_ist": datetime.now(IST).isoformat(),
            "source_url": INSTRUMENT_MASTER_URL,
            "instrument_count": len(instruments),
        }
        self._atomic_write_json(INSTRUMENT_CACHE_META_PATH, meta)

    def _download_master(self) -> Optional[List[Dict[str, Any]]]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        for attempt in range(1, 4):
            try:
                resp = requests.get(INSTRUMENT_MASTER_URL, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                if not self._validate_master_integrity(data):
                    raise ValueError("Downloaded master failed integrity check")
                instruments = [x for x in data if isinstance(x, dict)]
                self._logger.info(
                    "Instrument master downloaded",
                    attempt=attempt,
                    instrument_count=len(instruments),
                    size_mb=round(len(resp.content) / (1024 * 1024), 2),
                )
                return instruments
            except Exception as e:
                self._logger.warning(
                    "Instrument master download failed",
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if attempt < 3:
                    time.sleep(2 * attempt)
        return None

    def _parse_expiry_field(self, expiry_str: str) -> Optional[date]:
        if not expiry_str:
            return None
        clean = str(expiry_str).strip().upper().split("T")[0]
        for fmt in ("%d%b%Y", "%d%b%y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(clean, fmt).date()
            except Exception:
                continue
        return None

    def _filter_expired_instruments_on_load(self, instruments: List[Dict[str, Any]], today: date) -> List[Dict[str, Any]]:
        """
        Mandatory: filter expired instruments on every load source.
        Strict:
          - For NFO derivatives: expiry must be parseable from field OR from symbol if supported index match.
          - If unparseable -> reject
          - If expiry < today -> reject
        """
        kept: List[Dict[str, Any]] = []
        removed = 0
        for inst in instruments:
            try:
                exch = str(inst.get("exch_seg", "") or "").upper().strip()
                if exch != "NFO":
                    kept.append(inst)
                    continue

                itype = str(inst.get("instrumenttype", "") or "").upper().strip()
                if itype not in ("OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"):
                    kept.append(inst)
                    continue

                exp_field = inst.get("expiry") or inst.get("expiry_date")
                exp_dt = self._parse_expiry_field(str(exp_field)) if exp_field else None
                if exp_dt is None:
                    idx = self._detect_index_from_symbol(str(inst.get("symbol", "") or ""))
                    if idx is not None:
                        ps = self._parse_symbol(idx, str(inst.get("symbol", "") or ""))
                        exp_dt = ps.expiry_date if ps else None

                if exp_dt is None:
                    removed += 1
                    continue
                if exp_dt < today:
                    removed += 1
                    continue

                kept.append(inst)
            except Exception:
                removed += 1

        if removed:
            self._logger.warning("Expired/unparseable derivatives removed on load", removed=removed, kept=len(kept))
        return kept

    def _load_or_download_master(self) -> Optional[List[Dict[str, Any]]]:
        today = self._ist_today()
        is_trading_day = self._is_trading_day_simple(today)

        meta = self._read_cache_meta()
        cache_date: Optional[date] = None
        if isinstance(meta.get("cache_date"), str) and meta.get("cache_date"):
            try:
                cache_date = datetime.strptime(str(meta["cache_date"]), "%Y-%m-%d").date()
            except Exception:
                cache_date = None

        cache_ok = False
        if cache_date is not None and os.path.isfile(INSTRUMENT_CACHE_PATH):
            if (not is_trading_day) or (cache_date == today):
                cache_ok = True

        if cache_ok:
            cached = self._load_json_list(INSTRUMENT_CACHE_PATH)
            if cached is not None:
                filtered = self._filter_expired_instruments_on_load(cached, today=today)
                with self._lock:
                    self.cache_source = "cache"
                    self.cache_date = cache_date
                    self.cache_age_days = 0 if cache_date == today else (today - cache_date).days
                    self.is_stale_cache = bool(self.cache_age_days and self.cache_age_days > 0)
                self._logger.info(
                    "Using cached instrument master",
                    cache_date=str(cache_date),
                    cache_age_days=self.cache_age_days,
                    trading_day=is_trading_day,
                    is_stale_cache=bool(self.is_stale_cache),
                )
                return filtered

        downloaded = self._download_master()
        if downloaded is not None:
            filtered = self._filter_expired_instruments_on_load(downloaded, today=today)
            try:
                self._write_cache(filtered, cache_for_date=today)
            except Exception as e:
                self._logger.warning("Failed to write cache (continuing)", error=str(e))
            try:
                self._atomic_write_json(INSTRUMENT_BACKUP_PATH, filtered)
            except Exception as e:
                self._logger.warning("Failed to write backup (continuing)", error=str(e))

            with self._lock:
                self.cache_source = "download"
                self.cache_date = today
                self.cache_age_days = 0
                self.is_stale_cache = False
            return filtered

        backup_age = self._file_age_days(INSTRUMENT_BACKUP_PATH)
        if backup_age is not None and backup_age <= self._fallback_max_age_days:
            backup = self._load_json_list(INSTRUMENT_BACKUP_PATH)
            if backup is not None:
                filtered = self._filter_expired_instruments_on_load(backup, today=today)
                with self._lock:
                    self.cache_source = "backup"
                    self.cache_date = None
                    self.cache_age_days = int(backup_age)
                    self.is_stale_cache = True
                self._logger.critical(
                    "Using BACKUP instrument master (download failed)",
                    backup_path=INSTRUMENT_BACKUP_PATH,
                    backup_age_days=int(backup_age),
                    max_age_days=int(self._fallback_max_age_days),
                )
                return filtered

        if backup_age is None:
            self._logger.critical("Download failed and backup missing; refusing to proceed", backup_path=INSTRUMENT_BACKUP_PATH)
        else:
            self._logger.critical(
                "Download failed and backup too old; refusing to proceed",
                backup_path=INSTRUMENT_BACKUP_PATH,
                backup_age_days=int(backup_age),
                max_age_days=int(self._fallback_max_age_days),
            )
        return None

    # ==========================
    # Strict index detection
    # ==========================
    def _normalize_symbol(self, symbol: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(symbol).strip().upper())

    def _detect_index_from_symbol(self, symbol: str) -> Optional[str]:
        sym = self._normalize_symbol(symbol)
        for idx in self._supported_indices:
            idx_n = self._normalize_symbol(idx)
            if sym.startswith(idx_n) and len(sym) > len(idx_n) and sym[len(idx_n)].isdigit():
                return idx
        return None

    # ==========================
    # Strict symbol parsing
    # ==========================
    def _parse_symbol(self, idx: str, symbol: str) -> Optional[ParsedSymbol]:
        spec = self._regex_by_index.get(idx)
        if spec is None:
            return None
        sym = self._normalize_symbol(symbol)
        m = spec.pattern.match(sym)
        if not m:
            return None

        try:
            if spec.uses_named_groups:
                expiry_grp = m.group("expiry")
                series_grp = m.group("series")
                strike_grp = m.group("strike")
                opt_type = m.group("opt_type")
            else:
                expiry_grp = m.group(1)
                series_grp = m.group(2)
                strike_grp = m.group(3)
                opt_type = m.group(4)
        except Exception:
            return None

        series_identifier = str(series_grp)
        if not series_identifier.isdigit():
            return None

        strike_suffix_str = str(strike_grp)
        if not strike_suffix_str.isdigit() or len(strike_suffix_str) != 3:
            return None
        strike_suffix_int = _safe_int(strike_suffix_str)
        if strike_suffix_int is None:
            return None

        opt_type_u = str(opt_type).upper()
        if opt_type_u not in ("CE", "PE"):
            return None

        exp_s = str(expiry_grp).strip().upper()
        exp_dt: Optional[date] = None
        for fmt in ("%d%b%y", "%d%b%Y"):
            try:
                exp_dt = datetime.strptime(exp_s, fmt).date()
                break
            except Exception:
                continue
        if exp_dt is None:
            return None

        full_strike_int = _safe_int(f"{series_identifier}{strike_suffix_str}")
        if full_strike_int is None:
            return None

        return ParsedSymbol(
            index=idx,
            expiry_date=exp_dt,
            opt_type=opt_type_u,
            series_identifier=series_identifier,
            strike_suffix_str=strike_suffix_str,
            strike_suffix=int(strike_suffix_int),
            full_strike=int(full_strike_int),
        )

    def _normalize_tick_size(self, tick: float) -> float:
        # paise normalization: 1..100 => /100
        if 1.0 <= tick <= 100.0:
            return tick / 100.0
        return tick

    def _validate_lot_tick(self, lot_raw: Any, tick_raw: Any) -> Optional[Tuple[int, float]]:
        lot = _safe_int(lot_raw)
        tick = _safe_float(tick_raw)
        if lot is None or tick is None:
            return None
        tick = self._normalize_tick_size(float(tick))
        if int(lot) not in self._allowed_lot_sizes:
            return None
        if not (self._tick_size_min <= float(tick) <= self._tick_size_max):
            return None
        return int(lot), float(tick)

    # ==========================
    # Fail-closed build
    # ==========================
    def build_map(self, smart_api=None, spot_price: float = 24500.0) -> Dict:
        _ = smart_api
        _ = spot_price

        self._logger.info("Building instrument map (fail-closed)", supported_indices=self._supported_indices, default_index=self._default_index)

        today = self._ist_today()
        instruments = self._load_or_download_master()
        if instruments is None:
            self._fail_closed(reason="no_master_available")
            return {}

        local_opt_map: Dict[str, Dict[OptionKey, Dict[str, Any]]] = {idx: {} for idx in self._supported_indices}
        local_legacy_map: Dict[str, Dict[Tuple[int, str, str], Dict[str, Any]]] = {idx: {} for idx in self._supported_indices}
        local_token_details: Dict[str, Dict[str, Any]] = dict(self._token_details)
        expiry_sets: Dict[str, Set[date]] = {idx: set() for idx in self._supported_indices}

        total_processed = 0
        parsing_errors = 0
        validation_failures = 0
        duplicate_rejections = 0

        rejected_keys: Dict[str, Set[OptionKey]] = {idx: set() for idx in self._supported_indices}

        for inst in instruments:
            exch = str(inst.get("exch_seg", "") or "").upper().strip()
            if exch != "NFO":
                continue
            itype = str(inst.get("instrumenttype", "") or "").upper().strip()
            if itype not in ("OPTIDX", "OPTSTK"):
                continue

            symbol = str(inst.get("symbol", "") or "")
            idx = self._detect_index_from_symbol(symbol)
            if idx is None:
                continue  # STRICT: do not count as processed, do not count as parse failure

            total_processed += 1

            token = inst.get("token")
            if not token:
                parsing_errors += 1
                if parsing_errors <= 100 or parsing_errors % 500 == 0:
                    self._logger.error("Missing token", index=idx, symbol=symbol[:120])
                continue

            parsed = self._parse_symbol(idx, symbol)
            if parsed is None:
                parsing_errors += 1
                if parsing_errors <= 100 or parsing_errors % 500 == 0:
                    self._logger.error("Symbol parse failed", index=idx, symbol=symbol[:120])
                continue

            if not parsed.series_identifier.isdigit():
                parsing_errors += 1
                if parsing_errors <= 100 or parsing_errors % 500 == 0:
                    self._logger.error("Series identifier invalid", index=idx, symbol=symbol[:120], extracted_series=str(parsed.series_identifier))
                continue

            exp_field = inst.get("expiry") or inst.get("expiry_date")
            exp_field_dt = self._parse_expiry_field(str(exp_field)) if exp_field else None
            if exp_field_dt is not None and exp_field_dt != parsed.expiry_date:
                validation_failures += 1
                if validation_failures <= 100 or validation_failures % 500 == 0:
                    self._logger.warning(
                        "Expiry mismatch (field vs symbol); skipping",
                        index=idx,
                        symbol=symbol[:120],
                        expiry_field=str(exp_field),
                        expiry_field_dt=str(exp_field_dt),
                        expiry_symbol_dt=str(parsed.expiry_date),
                    )
                continue

            if parsed.expiry_date < today:
                validation_failures += 1
                continue

            lot_raw = inst.get("lotsize", inst.get("lot_size"))
            tick_raw = inst.get("tick_size", inst.get("ticksize", inst.get("tick")))
            lt = self._validate_lot_tick(lot_raw, tick_raw)
            if lt is None:
                validation_failures += 1
                if validation_failures <= 100 or validation_failures % 500 == 0:
                    self._logger.warning(
                        "Invalid lot/tick; skipping",
                        index=idx,
                        symbol=symbol[:120],
                        token=str(token),
                        lot_size=str(lot_raw),
                        tick_size=str(tick_raw),
                    )
                continue
            lot_size, tick_size = lt

            key: OptionKey = (
                int(parsed.strike_suffix),
                parsed.expiry_date,
                parsed.opt_type,
                str(parsed.series_identifier),
            )
            if key in rejected_keys[idx]:
                continue

            entry = {
                "token": str(token),
                "symbol": symbol,
                "index": idx,
                "expiry_date": parsed.expiry_date,
                "expiry": parsed.expiry_date.strftime("%Y-%m-%d"),
                "opt_type": parsed.opt_type,
                "series_identifier": str(parsed.series_identifier),
                "strike_suffix": int(parsed.strike_suffix),
                "strike": int(parsed.full_strike),
                "lot_size": int(lot_size),
                "tick_size": float(tick_size),
            }

            existing = local_opt_map[idx].get(key)
            if existing is not None:
                if str(existing.get("token")) != str(entry.get("token")):
                    duplicate_rejections += 1
                    self._logger.critical(
                        "Duplicate canonical key detected — rejecting BOTH entries (master corruption)",
                        index=idx,
                        canonical_key={
                            "strike_suffix": key[0],
                            "expiry_date": str(key[1]),
                            "opt_type": key[2],
                            "series_identifier": key[3],
                        },
                        existing_instrument={"token": str(existing.get("token")), "symbol": str(existing.get("symbol"))},
                        new_instrument={"token": str(entry.get("token")), "symbol": str(entry.get("symbol"))},
                    )
                    try:
                        del local_opt_map[idx][key]
                    except Exception:
                        pass
                    try:
                        legacy_key_old = (int(existing["strike"]), str(existing["expiry"]), str(existing["opt_type"]))
                        if legacy_key_old in local_legacy_map[idx]:
                            del local_legacy_map[idx][legacy_key_old]
                    except Exception:
                        pass
                    try:
                        tok_old = str(existing.get("token"))
                        if tok_old in local_token_details:
                            del local_token_details[tok_old]
                    except Exception:
                        pass

                    rejected_keys[idx].add(key)
                    continue

                self._logger.warning(
                    "Duplicate canonical key with identical token; skipping duplicate row",
                    index=idx,
                    token=str(entry.get("token")),
                    symbol=str(entry.get("symbol"))[:120],
                )
                continue

            local_opt_map[idx][key] = entry

            legacy_key = (int(entry["strike"]), str(entry["expiry"]), str(entry["opt_type"]))
            existing_legacy = local_legacy_map[idx].get(legacy_key)
            if existing_legacy is not None and str(existing_legacy.get("token")) != str(entry.get("token")):
                duplicate_rejections += 1
                self._logger.critical(
                    "Legacy key collision detected — rejecting BOTH (public API ambiguity)",
                    index=idx,
                    legacy_key={"strike": legacy_key[0], "expiry": legacy_key[1], "opt_type": legacy_key[2]},
                    existing_instrument={"token": str(existing_legacy.get("token")), "symbol": str(existing_legacy.get("symbol"))},
                    new_instrument={"token": str(entry.get("token")), "symbol": str(entry.get("symbol"))},
                )

                try:
                    del local_legacy_map[idx][legacy_key]
                except Exception:
                    pass
                try:
                    del local_opt_map[idx][key]
                except Exception:
                    pass
                try:
                    tok_new = str(entry.get("token"))
                    if tok_new in local_token_details:
                        del local_token_details[tok_new]
                except Exception:
                    pass
                try:
                    tok_old = str(existing_legacy.get("token"))
                    if tok_old in local_token_details:
                        del local_token_details[tok_old]
                except Exception:
                    pass

                rejected_keys[idx].add(key)
                continue

            local_legacy_map[idx][legacy_key] = entry

            local_token_details[str(entry["token"])] = {
                "token": str(entry["token"]),
                "symbol": str(entry["symbol"]),
                "index": idx,
                "expiry": str(entry["expiry"]),
                "expiry_date": str(entry["expiry_date"]),
                "opt_type": str(entry["opt_type"]),
                "series_identifier": str(entry["series_identifier"]),
                "strike_suffix": int(entry["strike_suffix"]),
                "strike": int(entry["strike"]),
                "lot_size": int(entry["lot_size"]),
                "tick_size": float(entry["tick_size"]),
            }

            expiry_sets[idx].add(parsed.expiry_date)

        mapped_count = sum(len(local_opt_map[idx]) for idx in self._supported_indices)
        parse_failure_rate = (parsing_errors / total_processed) if total_processed > 0 else 1.0

        expiry_dates_by_index: Dict[str, List[date]] = {}
        current_by_index: Dict[str, Optional[date]] = {}
        next_by_index: Dict[str, Optional[date]] = {}
        for idx in self._supported_indices:
            exps = sorted([d for d in expiry_sets.get(idx, set()) if d >= today])
            expiry_dates_by_index[idx] = exps
            current_by_index[idx] = exps[0] if len(exps) >= 1 else None
            next_by_index[idx] = exps[1] if len(exps) >= 2 else None

        if parse_failure_rate > self._parse_failure_threshold or mapped_count < self._expected_min_mapped:
            self._logger.critical(
                "Mapper refusing to build due to critical data quality failure.",
                parse_failure_rate=float(parse_failure_rate),
                parse_failure_threshold=float(self._parse_failure_threshold),
                mapped_instrument_count=int(mapped_count),
                expected_min_mapped=int(self._expected_min_mapped),
                processed_nfo_instruments=int(total_processed),
                parsing_errors=int(parsing_errors),
                validation_failures=int(validation_failures),
                duplicate_rejections=int(duplicate_rejections),
                cache_source=str(self.cache_source),
                cache_age_days=self.cache_age_days,
                is_stale_cache=bool(self.is_stale_cache),
            )
            self._fail_closed(
                reason="critical_data_quality_failure",
                processed=total_processed,
                parsing_errors=parsing_errors,
                validation_failures=validation_failures,
                duplicate_rejections=duplicate_rejections,
                parse_failure_rate=parse_failure_rate,
            )
            return {}

        with self._lock:
            self._opt_map_by_index = local_opt_map
            self._legacy_map_by_index = local_legacy_map
            self._legacy_map = dict(local_legacy_map.get(self._default_index, {}))
            self._token_details = local_token_details

            self._expiry_dates_by_index = expiry_dates_by_index
            self._current_expiry_by_index = current_by_index
            self._next_expiry_by_index = next_by_index

            self.is_built = True
            self.total_instruments_mapped = int(mapped_count)

            self.last_parse_failure_rate = float(parse_failure_rate)
            self.last_processed_nfo_count = int(total_processed)
            self.last_parsing_errors = int(parsing_errors)
            self.last_validation_failures = int(validation_failures)
            self.last_duplicate_rejections = int(duplicate_rejections)

        self._logger.info(
            "Instrument mapper built successfully",
            total_mapped=int(mapped_count),
            parse_failure_rate=float(parse_failure_rate),
            processed_nfo_instruments=int(total_processed),
            parsing_errors=int(parsing_errors),
            validation_failures=int(validation_failures),
            duplicate_rejections=int(duplicate_rejections),
            default_index=self._default_index,
            current_expiry=str(self._current_expiry_by_index.get(self._default_index)),
            next_expiry=str(self._next_expiry_by_index.get(self._default_index)),
            cache_source=str(self.cache_source),
            cache_age_days=self.cache_age_days,
            is_stale_cache=bool(self.is_stale_cache),
        )

        return dict(self._legacy_map)

    def _fail_closed(
        self,
        *,
        reason: str,
        processed: int = 0,
        parsing_errors: int = 0,
        validation_failures: int = 0,
        duplicate_rejections: int = 0,
        parse_failure_rate: Optional[float] = None,
    ) -> None:
        with self._lock:
            self.is_built = False
            self.total_instruments_mapped = 0

            self._opt_map_by_index = {}
            self._legacy_map_by_index = {}
            self._legacy_map = {}
            self._expiry_dates_by_index = {}
            self._current_expiry_by_index = {}
            self._next_expiry_by_index = {}

            self._token_details = {
                self._spot_token: self._token_details.get(self._spot_token, {}),
                self._vix_token: self._token_details.get(self._vix_token, {}),
            }

            self.last_parse_failure_rate = float(parse_failure_rate) if parse_failure_rate is not None else None
            self.last_processed_nfo_count = int(processed)
            self.last_parsing_errors = int(parsing_errors)
            self.last_validation_failures = int(validation_failures)
            self.last_duplicate_rejections = int(duplicate_rejections)

        self._logger.critical(
            "InstrumentMapper state set to UNBUILT (fail-closed)",
            reason=str(reason),
            processed_nfo_instruments=int(processed),
            parsing_errors=int(parsing_errors),
            validation_failures=int(validation_failures),
            duplicate_rejections=int(duplicate_rejections),
            parse_failure_rate=float(parse_failure_rate) if parse_failure_rate is not None else None,
            cache_source=str(self.cache_source),
            cache_age_days=self.cache_age_days,
            is_stale_cache=bool(self.is_stale_cache),
        )

    # ==========================
    # Public API (unchanged signatures)
    # ==========================

    @staticmethod
    def _normalize_expiry_to_date(expiry: Any) -> Optional[date]:
        """Convert 'expiry' arg (date or str like '2026-05-05' or '05MAY2026') to a date object."""
        if isinstance(expiry, date):
            return expiry
        if isinstance(expiry, str):
            for fmt in ("%Y-%m-%d", "%d%b%Y", "%d%b%y", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    return datetime.strptime(expiry.strip(), fmt).date()
                except Exception:
                    continue
        return None

    def get_option_symbol(self, strike: int, expiry: Any, opt_type: str) -> Optional[str]:
        """
        Return the trading symbol for a NIFTY option contract.
        Looks up via the mapper's built legacy map (O(1) path), with a scan fallback.
        """
        target_date = self._normalize_expiry_to_date(expiry)
        if target_date is None:
            return None

        opt_type_u = str(opt_type).upper().strip()
        expiry_key = target_date.strftime("%Y-%m-%d")

        with self._lock:
            if not self.is_built:
                return None

            # Fast path: exact legacy key lookup
            entry = self._legacy_map.get((int(strike), expiry_key, opt_type_u))
            if entry is not None:
                sym = entry.get("symbol") or entry.get("tradingsymbol") or entry.get("name")
                return str(sym) if sym else None

            # Fallback scan (should be rare)
            for (_k_strike, _k_expiry, _k_type), info in self._legacy_map.items():
                if _k_strike != int(strike) or str(_k_type).upper().strip() != opt_type_u:
                    continue
                k_dt = self._parse_expiry_field(str(_k_expiry)) if _k_expiry else None
                if k_dt != target_date:
                    continue
                sym = info.get("symbol") or info.get("tradingsymbol") or info.get("name")
                if sym:
                    return str(sym)

        return None

    def get_option_token(self, strike: int, expiry: Any, opt_type: str) -> Optional[str]:
        """
        Return the instrument token for a NIFTY option contract.
        Looks up via the mapper's built legacy map (O(1) path), with a scan fallback.
        """
        target_date = self._normalize_expiry_to_date(expiry)
        if target_date is None:
            return None

        opt_type_u = str(opt_type).upper().strip()
        expiry_key = target_date.strftime("%Y-%m-%d")

        with self._lock:
            if not self.is_built:
                return None

            entry = self._legacy_map.get((int(strike), expiry_key, opt_type_u))
            if entry is not None and entry.get("token") is not None:
                return str(entry["token"])

            for (_k_strike, _k_expiry, _k_type), info in self._legacy_map.items():
                if _k_strike != int(strike) or str(_k_type).upper().strip() != opt_type_u:
                    continue
                k_dt = self._parse_expiry_field(str(_k_expiry)) if _k_expiry else None
                if k_dt != target_date:
                    continue
                tok = info.get("token")
                if tok:
                    return str(tok)

        return None

    def get_current_expiry(self) -> Optional[date]:
        with self._lock:
            return self._current_expiry_by_index.get(self._default_index)

    def get_next_expiry(self) -> Optional[date]:
        with self._lock:
            return self._next_expiry_by_index.get(self._default_index)

    def get_instrument_details(self, token: str) -> Optional[Dict]:
        tok = str(token).strip()
        if not tok:
            return None
        with self._lock:
            info = self._token_details.get(tok)
            return dict(info) if isinstance(info, dict) else None

    def get_subscription_list(self, spot_price: float) -> List[Dict]:
        """
        MUST return only spot/VIX if is_built=False.
        Otherwise: ATM ± range, respect token limit, never drop ATM; drop far OTM first.
        If ATM token missing -> spot/VIX only.
        """
        base = [
            {"exchange": "nse_cm", "token": self._spot_token},
            {"exchange": "nse_cm", "token": self._vix_token},
        ]

        with self._lock:
            if not self.is_built:
                return list(base)
            cur = self._current_expiry_by_index.get(self._default_index)
            nxt = self._next_expiry_by_index.get(self._default_index)

        try:
            sp = float(spot_price)
        except Exception:
            sp = 0.0

        if cur is None or sp <= 0.0:
            self._logger.error(
                "Subscription list invalid inputs; returning spot/VIX only",
                is_built=bool(self.is_built),
                current_expiry=str(cur),
                spot_price=spot_price,
            )
            return list(base)

        max_tokens = Config.get("data", "max_instruments_websocket", default=100)
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 100
        if max_tokens > 100:
            max_tokens = 100
        if max_tokens < 4:
            self._logger.critical("Token limit too low; returning spot/VIX only", max_tokens=max_tokens)
            return list(base)

        expiry_list: List[str] = [cur.strftime("%Y-%m-%d")]
        if nxt is not None:
            expiry_list.append(nxt.strftime("%Y-%m-%d"))

        atm = round_to_strike(sp, step=self._strike_interval)
        r_cfg = max(0, int(self._option_strikes_range))
        r_min = max(0, int(self._min_atm_range))

        def token_need(strike_count: int, expiry_count: int) -> int:
            return 2 + (strike_count * 2 * expiry_count)

        def make_strikes(r: int) -> List[int]:
            strikes = [atm + i * self._strike_interval for i in range(-r, r + 1)]
            if atm not in strikes:
                strikes.append(atm)
            return sorted(set(strikes), key=lambda s: (abs(s - atm), s))

        strikes = make_strikes(r_cfg)

        if token_need(len(strikes), len(expiry_list)) > max_tokens and len(expiry_list) > 1:
            dropped = expiry_list.pop()
            self._logger.warning("Dropping next expiry due to token limit", dropped_expiry=dropped, max_tokens=max_tokens)

        min_strikes = max(1, 2 * r_min + 1)
        while token_need(len(strikes), len(expiry_list)) > max_tokens and len(strikes) > min_strikes:
            farthest = None
            far_dist = -1
            for s in strikes:
                if s == atm:
                    continue
                d = abs(s - atm)
                if d > far_dist:
                    far_dist = d
                    farthest = s
            if farthest is None:
                break
            strikes.remove(farthest)

        if token_need(len(strikes), len(expiry_list)) > max_tokens and len(strikes) > 1:
            self._logger.critical(
                "Token limit forces violating min_atm_range; reducing further (ATM preserved)",
                max_tokens=max_tokens,
                min_atm_range=r_min,
                strikes_count=len(strikes),
                expiry_count=len(expiry_list),
            )
            while token_need(len(strikes), len(expiry_list)) > max_tokens and len(strikes) > 1:
                farthest = None
                far_dist = -1
                for s in strikes:
                    if s == atm:
                        continue
                    d = abs(s - atm)
                    if d > far_dist:
                        far_dist = d
                        farthest = s
                if farthest is None:
                    break
                strikes.remove(farthest)

        if token_need(len(strikes), len(expiry_list)) > max_tokens:
            self._logger.critical("Token limit too restrictive; returning spot/VIX only", max_tokens=max_tokens)
            return list(base)

        atm_exp = expiry_list[0]
        atm_ce = self.get_option_token(atm, atm_exp, "CE")
        atm_pe = self.get_option_token(atm, atm_exp, "PE")
        if not atm_ce or not atm_pe:
            self._logger.critical(
                "ATM CE/PE missing in map; returning spot/VIX only",
                atm=atm,
                expiry=atm_exp,
                atm_ce=atm_ce,
                atm_pe=atm_pe,
            )
            return list(base)

        sub: List[Dict[str, Any]] = list(base)
        missing = 0
        for exp in expiry_list:
            for strike in strikes:
                for ot in ("CE", "PE"):
                    tok = self.get_option_token(strike, exp, ot)
                    if tok:
                        sub.append({"exchange": "nfo_cm", "token": tok, "strike": strike, "expiry": exp, "type": ot})
                    else:
                        missing += 1

        seen: Set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for item in sub:
            tok = str(item.get("token"))
            if tok in seen:
                continue
            seen.add(tok)
            deduped.append(item)

        if len(deduped) > max_tokens:
            self._logger.warning("Subscription list exceeds max_tokens; truncating", len_before=len(deduped), max_tokens=max_tokens)
            deduped = deduped[:max_tokens]

        tokens_final = {str(x.get("token")) for x in deduped}
        if self._spot_token not in tokens_final or self._vix_token not in tokens_final or atm_ce not in tokens_final or atm_pe not in tokens_final:
            self._logger.critical("Safety check failed (base/ATM tokens missing); returning spot/VIX only")
            return list(base)

        if missing:
            self._logger.warning("Some requested option tokens missing from map", missing_count=int(missing))
        return deduped

    def get_status(self) -> Dict:
        with self._lock:
            cur = self._current_expiry_by_index.get(self._default_index)
            nxt = self._next_expiry_by_index.get(self._default_index)
            return {
                "is_built": bool(self.is_built),
                "total_mapped": int(self.total_instruments_mapped),
                "supported_indices": list(self._supported_indices),
                "default_index": self._default_index,
                "current_expiry": str(cur) if cur else None,
                "next_expiry": str(nxt) if nxt else None,
                "cache_source": str(self.cache_source),
                "cache_age_days": int(self.cache_age_days) if self.cache_age_days is not None else None,
                "cache_date": str(self.cache_date) if self.cache_date else None,
                "is_stale_cache": bool(self.is_stale_cache),
                "last_parse_failure_rate": float(self.last_parse_failure_rate) if self.last_parse_failure_rate is not None else None,
                "last_processed_nfo_count": int(self.last_processed_nfo_count),
                "last_parsing_errors": int(self.last_parsing_errors),
                "last_validation_failures": int(self.last_validation_failures),
                "last_duplicate_rejections": int(self.last_duplicate_rejections),
                "expected_min_mapped": int(self._expected_min_mapped),
                "parse_failure_threshold": float(self._parse_failure_threshold),
                "cache_path": INSTRUMENT_CACHE_PATH,
                "backup_path": INSTRUMENT_BACKUP_PATH,
            }


# ==========================
# Offline self-test (no credentials)
# ==========================
if __name__ == "__main__":
    print("=" * 78)
    print("JUNIOR ALADDIN — InstrumentMapper Offline Self-Test (Fail-Closed)")
    print("=" * 78)

    mapper = InstrumentMapper()

    idx = mapper._default_index
    sample = mapper._default_sample_for_index(idx)
    ps = mapper._parse_symbol(idx, sample)
    assert ps is not None
    if idx.upper() == "NIFTY" and sample == "NIFTY28APR2621400PE":
        assert ps.series_identifier == "21"
        assert ps.strike_suffix == 400
        assert ps.full_strike == 21400
        assert ps.opt_type == "PE"
        assert ps.expiry_date == date(2026, 4, 28)
    print(f"[OK] Regex parse: index={idx} sample={sample} -> series={ps.series_identifier} strike_suffix={ps.strike_suffix} full_strike={ps.full_strike}")

    assert mapper._detect_index_from_symbol("NIFTYNXT5030JUN2672900CE") is None
    print("[OK] Strict index detection rejects NIFTYNXT50* symbols")

    subs = mapper.get_subscription_list(spot_price=24500.0)
    assert isinstance(subs, list) and len(subs) == 2
    print("[OK] Subscription list returns spot/VIX only when is_built=False")

    print("[STATUS]", mapper.get_status())
    print("[OK] Offline self-test passed")