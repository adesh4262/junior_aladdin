"""
Junior Aladdin — Tick Cleaner
=============================
Strongest Version: Fully restored legacy stats tracking and issues 
management while maintaining intelligent metadata derivation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from loguru import logger

from data_center.schemas.tick_schema import RAW_TICK_SCHEMA
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
class CleanTickResult:
    is_clean: bool
    record: Optional[dict[str, Any]] = None
    issues: list[CleanIssue] = field(default_factory=list)
    duplicate: bool = False
    timestamp_issue: bool = False
    ordering_fixed: bool = False
    index: Optional[int] = None


class TickCleaner:
    """Clean and order raw tick records before structured processing."""

    def __init__(
        self,
        duplicate_validator: DuplicateValidator | None = None,
        timestamp_validator: TimestampValidator | None = None,
    ):
        self.duplicate_validator = duplicate_validator or DuplicateValidator()
        self.timestamp_validator = timestamp_validator or TimestampValidator()
        
        # Sequence counter
        self._sequence_counter = 0
        self._counter_lock = threading.Lock()
        
        self._stats = {
            "total_seen": 0,
            "total_cleaned": 0,
            "total_removed": 0,
            "duplicates_removed": 0,
            "invalid_timestamp_removed": 0,
            "ordering_fixed": 0,
        }

    def clean_record(self, record: dict[str, Any]) -> CleanTickResult:
        """Clean a single tick record with full stats tracking."""
        self._stats["total_seen"] += 1

        normalized = self._normalize(record)
        if normalized is None:
            self._stats["total_removed"] += 1
            return CleanTickResult(
                is_clean=False,
                issues=[CleanIssue(code="invalid_packet", message="Unable to normalize tick record")],
            )

        # Permanent Metadata Derivation for Schema Compliance
        with self._counter_lock:
            self._sequence_counter += 1
            normalized["sequence"] = self._sequence_counter
        
        normalized.setdefault("exchange", "NSE")
        normalized.setdefault("symbol", str(normalized.get("token", "NIFTY")))

        duplicate_result = self.duplicate_validator.validate_record(normalized, record_type="tick")
        timestamp_result = self.timestamp_validator.validate_record(normalized, record_type="tick")

        issues: list[CleanIssue] = []
        if duplicate_result.duplicate:
            issues.append(CleanIssue(code="duplicate", message="Duplicate tick removed", field="fingerprint"))
            self._stats["duplicates_removed"] += 1
        if timestamp_result.invalid_timestamp:
            issues.append(CleanIssue(code="invalid_timestamp", message="Invalid timestamp", field="timestamp"))
            self._stats["invalid_timestamp_removed"] += 1

        is_clean = not (duplicate_result.duplicate or timestamp_result.invalid_timestamp)
        if is_clean:
            self._stats["total_cleaned"] += 1
        else:
            self._stats["total_removed"] += 1

        return CleanTickResult(
            is_clean=is_clean,
            record=normalized if is_clean else None,
            issues=issues,
            duplicate=duplicate_result.duplicate,
            timestamp_issue=timestamp_result.invalid_timestamp,
        )

    def clean_batch(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean a batch while preserving original ordering logic."""
        results = [self.clean_record(r) for r in records]
        cleaned = [res.record for res in results if res.record is not None]
        # Sort to ensure timestamp continuity
        cleaned.sort(key=lambda x: x.get("timestamp", 0))
        return cleaned

    def _normalize(self, record: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not isinstance(record, dict): return None
        
        # Support aliases for institutional robustness
        token = record.get("token") or record.get("tok")
        ltp = record.get("ltp") or record.get("price")
        
        if not token or ltp is None: return None

        try:
            ts = int(record.get("timestamp") or record.get("ts") or (time.time() * 1000))
            return {
                "token": str(token),
                "ltp": float(ltp),
                "volume": int(record.get("volume") or 0),
                "timestamp": ts,
                "open": float(record.get("open", 0.0)),
                "high": float(record.get("high", 0.0)),
                "low": float(record.get("low", 0.0)),
                "close": float(record.get("close", 0.0)),
                "direction": int(record.get("direction", 0))
            }
        except: return None

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)


tick_cleaner = TickCleaner()
