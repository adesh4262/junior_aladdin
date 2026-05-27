"""
Junior Aladdin — Options Cleaner
================================
Normalizes, validates, deduplicates, and orders raw option records.

Phase 4 scope:
  - normalize option structures
  - validate strikes
  - validate OI/IV

The cleaner prepares raw option ticks for the transformer layer while
preserving key market fields and adding moneyness metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from loguru import logger

from data_center.schemas.options_schema import RAW_OPTIONS_SCHEMA, VALID_OPTION_TYPES, MAX_IV, MIN_OI
from data_center.validators.duplicate_validator import DuplicateValidator
from data_center.validators.timestamp_validator import TimestampValidator
from data_center.utils.timestamps import is_valid_timestamp_ms


@dataclass(slots=True)
class CleanIssue:
    code: str
    message: str
    field: Optional[str] = None
    value: Any = None


@dataclass(slots=True)
class CleanOptionResult:
    is_clean: bool
    record: Optional[dict[str, Any]] = None
    issues: list[CleanIssue] = field(default_factory=list)
    duplicate: bool = False
    timestamp_issue: bool = False
    strike_issue: bool = False
    oi_issue: bool = False
    iv_issue: bool = False
    ordering_fixed: bool = False
    index: Optional[int] = None


class OptionsCleaner:
    """Clean and order raw option records before structured processing."""

    def __init__(
        self,
        duplicate_validator: DuplicateValidator | None = None,
        timestamp_validator: TimestampValidator | None = None,
    ):
        self.duplicate_validator = duplicate_validator or DuplicateValidator()
        self.timestamp_validator = timestamp_validator or TimestampValidator()
        self._stats = {
            "total_seen": 0,
            "total_cleaned": 0,
            "total_removed": 0,
            "duplicates_removed": 0,
            "invalid_timestamp_removed": 0,
            "invalid_strike_removed": 0,
            "invalid_oi_removed": 0,
            "invalid_iv_removed": 0,
            "ordering_fixed": 0,
        }

    def clean_record(self, record: dict[str, Any]) -> CleanOptionResult:
        """Clean a single option record."""
        self._stats["total_seen"] += 1

        normalized = self._normalize(record)
        if normalized is None:
            self._stats["total_removed"] += 1
            return CleanOptionResult(
                is_clean=False,
                issues=[CleanIssue(code="invalid_packet", message="Unable to normalize option record")],
            )

        duplicate_result = self.duplicate_validator.validate_record(normalized, record_type="option")
        timestamp_result = self.timestamp_validator.validate_record(normalized, record_type="option")

        issues: list[CleanIssue] = []
        if duplicate_result.duplicate:
            issues.append(CleanIssue(code="duplicate", message="Duplicate option removed", field="fingerprint"))
        if timestamp_result.invalid_timestamp:
            issues.append(CleanIssue(code="invalid_timestamp", message="Invalid timestamp", field="timestamp"))

        strike_issue = not self._is_valid_strike(normalized.get("strike"))
        oi_issue = not self._is_valid_oi(normalized.get("oi"))
        iv_issue = not self._is_valid_iv(normalized.get("iv"))

        if strike_issue:
            issues.append(CleanIssue(code="invalid_strike", message="Invalid strike", field="strike", value=normalized.get("strike")))
        if oi_issue:
            issues.append(CleanIssue(code="invalid_oi", message="Invalid OI", field="oi", value=normalized.get("oi")))
        if iv_issue:
            issues.append(CleanIssue(code="invalid_iv", message="Invalid IV", field="iv", value=normalized.get("iv")))

        if duplicate_result.duplicate:
            self._stats["duplicates_removed"] += 1
        if timestamp_result.invalid_timestamp:
            self._stats["invalid_timestamp_removed"] += 1
        if strike_issue:
            self._stats["invalid_strike_removed"] += 1
        if oi_issue:
            self._stats["invalid_oi_removed"] += 1
        if iv_issue:
            self._stats["invalid_iv_removed"] += 1

        is_clean = not (duplicate_result.duplicate or timestamp_result.invalid_timestamp or strike_issue or oi_issue or iv_issue)
        if is_clean:
            self._stats["total_cleaned"] += 1
        else:
            self._stats["total_removed"] += 1

        return CleanOptionResult(
            is_clean=is_clean,
            record=normalized if is_clean else None,
            issues=issues,
            duplicate=duplicate_result.duplicate,
            timestamp_issue=timestamp_result.invalid_timestamp,
            strike_issue=strike_issue,
            oi_issue=oi_issue,
            iv_issue=iv_issue,
            ordering_fixed=timestamp_result.ordering_issue,
        )

    def clean_batch(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean, de-duplicate, and order a batch of option ticks."""
        cleaned_results = [self.clean_record(record) for record in records]
        cleaned_records = [result.record for result in cleaned_results if result.record is not None]
        original_order = list(cleaned_records)
        cleaned_records.sort(key=lambda item: (item.get("timestamp", 0), str(item.get("token", "")), float(item.get("strike", 0.0))))

        if cleaned_records != original_order:
            self._stats["ordering_fixed"] += 1
            logger.info(f"Option batch ordering fixed for {len(cleaned_records)} records")

        return cleaned_records

    def reset(self) -> None:
        self._stats = {
            "total_seen": 0,
            "total_cleaned": 0,
            "total_removed": 0,
            "duplicates_removed": 0,
            "invalid_timestamp_removed": 0,
            "invalid_strike_removed": 0,
            "invalid_oi_removed": 0,
            "invalid_iv_removed": 0,
            "ordering_fixed": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _normalize(self, record: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not isinstance(record, dict):
            return None

        token = record.get("token", record.get("tok"))
        ltp = record.get("ltp", record.get("price"))
        if token is None or ltp is None:
            return None

        try:
            timestamp = int(record.get("timestamp", record.get("ts")))
            if not is_valid_timestamp_ms(timestamp):
                return None

            option_type = str(record.get("option_type", record.get("type", ""))).upper().strip()
            if option_type not in VALID_OPTION_TYPES:
                return None

            expiry = str(record.get("expiry", record.get("expiry_date", ""))).strip()
            strike = float(record.get("strike", record.get("strike_price", 0.0)))
            spot_ltp = record.get("spot_ltp")
            normalized = {
                "token": str(token).strip(),
                "ltp": float(ltp),
                "volume": int(record.get("volume", record.get("vol", 0))),
                "open": float(record.get("open", 0.0)),
                "high": float(record.get("high", 0.0)),
                "low": float(record.get("low", 0.0)),
                "close": float(record.get("close", 0.0)),
                "timestamp": timestamp,
                "oi": int(record.get("oi", 0)),
                "oi_change": int(record.get("oi_change", record.get("oichg", 0))),
                "iv": float(record.get("iv", record.get("implied_volatility", 0.0))),
                "strike": strike,
                "option_type": option_type,
                "expiry": expiry,
                "spot_ltp": float(spot_ltp) if spot_ltp is not None else None,
                "moneyness": self._classify_moneyness(spot_ltp=spot_ltp, strike=strike, option_type=option_type),
            }
        except (TypeError, ValueError):
            return None

        for field in RAW_OPTIONS_SCHEMA:
            if field not in normalized:
                return None

        for optional_field in ("sequence", "exchange", "symbol"):
            if optional_field in record and record[optional_field] is not None:
                normalized[optional_field] = record[optional_field]

        return normalized

    def _is_valid_strike(self, strike: Any) -> bool:
        try:
            strike_value = float(strike)
        except (TypeError, ValueError):
            return False
        return strike_value > 0.0

    def _is_valid_oi(self, oi: Any) -> bool:
        try:
            oi_value = int(oi)
        except (TypeError, ValueError):
            return False
        return oi_value >= MIN_OI

    def _is_valid_iv(self, iv: Any) -> bool:
        try:
            iv_value = float(iv)
        except (TypeError, ValueError):
            return False
        return 0.0 <= iv_value <= MAX_IV

    def _classify_moneyness(self, spot_ltp: Optional[float], strike: float, option_type: str) -> str:
        if spot_ltp is None:
            return "ATM"

        try:
            spot = float(spot_ltp)
        except (TypeError, ValueError):
            return "ATM"

        if strike <= 0:
            return "ATM"

        diff = abs(spot - strike)
        if diff <= max(1.0, spot * 0.0025):
            return "ATM"

        if option_type == "CE":
            return "ITM" if spot > strike else "OTM"
        if option_type == "PE":
            return "ITM" if spot < strike else "OTM"
        return "ATM"


options_cleaner = OptionsCleaner()
