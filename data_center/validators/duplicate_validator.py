"""
Junior Aladdin — Duplicate Validator
====================================
Detects duplicate or repeated records before cleaner/structured stages.

Phase 1 scope:
  - repeated ticks
  - sequence mismatch
  - timestamp duplicates

The validator is designed to be reusable across tick and option streams.
It supports both single-record and batch validation, and keeps a small
in-memory history for streaming duplicate detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

from loguru import logger

from data_center.schemas.options_schema import RAW_OPTIONS_SCHEMA
from data_center.schemas.tick_schema import RAW_TICK_SCHEMA
from data_center.utils.timestamps import is_valid_timestamp_ms, timestamp_continuity_check


RecordType = Literal["tick", "option"]


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    field: Optional[str] = None
    value: Any = None


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    record_type: RecordType
    issues: list[ValidationIssue] = field(default_factory=list)
    duplicate: bool = False
    sequence_mismatch: bool = False
    timestamp_duplicate: bool = False
    fingerprint: Optional[str] = None
    index: Optional[int] = None


class DuplicateValidator:
    """
    Stateful duplicate detector.

    The validator keeps track of the last fingerprint per record type,
    the last seen sequence per token, and a bounded set of recent fingerprints
    to flag repeated packets even when sequences are missing.
    """

    def __init__(self, history_limit: int = 50_000):
        self.history_limit = history_limit
        self._recent_fingerprints: dict[RecordType, list[str]] = {"tick": [], "option": []}
        self._recent_fingerprint_sets: dict[RecordType, set[str]] = {"tick": set(), "option": set()}
        self._last_sequence_by_token: dict[str, int] = {}
        self._last_timestamp_by_token: dict[str, int] = {}
        self._last_fingerprint_by_token: dict[str, str] = {}
        self._stats = {
            "total_seen": 0,
            "duplicates": 0,
            "sequence_mismatches": 0,
            "timestamp_duplicates": 0,
            "invalid_records": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_record(self, record: dict[str, Any], record_type: RecordType = "tick") -> ValidationResult:
        """Validate a single record and update duplicate history."""
        result = self._validate_record(record, record_type)
        self._stats["total_seen"] += 1
        if not result.is_valid:
            self._stats["invalid_records"] += 1
        if result.duplicate:
            self._stats["duplicates"] += 1
        if result.sequence_mismatch:
            self._stats["sequence_mismatches"] += 1
        if result.timestamp_duplicate:
            self._stats["timestamp_duplicates"] += 1
        return result

    def validate_batch(
        self,
        records: Iterable[dict[str, Any]],
        record_type: RecordType = "tick",
    ) -> list[ValidationResult]:
        """Validate a batch and check timestamp continuity inside the batch."""
        batch = list(records)
        results = [self._validate_record(record, record_type, index=index) for index, record in enumerate(batch)]

        # Apply continuity checks for records that carry timestamps.
        timestamps = [self._safe_int(record.get("timestamp") or record.get("ts")) for record in batch]
        timestamps = [ts for ts in timestamps if ts is not None]
        gap_indices = set(timestamp_continuity_check(timestamps, max_gap_ms=5_000)) if len(timestamps) > 1 else set()
        if gap_indices:
            logger.warning(f"Timestamp continuity gaps detected in {record_type} batch: {sorted(gap_indices)}")

        for result in results:
            self._stats["total_seen"] += 1
            if not result.is_valid:
                self._stats["invalid_records"] += 1
            if result.duplicate:
                self._stats["duplicates"] += 1
            if result.sequence_mismatch:
                self._stats["sequence_mismatches"] += 1
            if result.timestamp_duplicate:
                self._stats["timestamp_duplicates"] += 1

        return results

    def reset(self) -> None:
        """Clear in-memory duplicate tracking state."""
        self._recent_fingerprints = {"tick": [], "option": []}
        self._recent_fingerprint_sets = {"tick": set(), "option": set()}
        self._last_sequence_by_token.clear()
        self._last_timestamp_by_token.clear()
        self._last_fingerprint_by_token.clear()
        self._stats = {
            "total_seen": 0,
            "duplicates": 0,
            "sequence_mismatches": 0,
            "timestamp_duplicates": 0,
            "invalid_records": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_record(
        self,
        record: dict[str, Any],
        record_type: RecordType,
        index: Optional[int] = None,
    ) -> ValidationResult:
        if not isinstance(record, dict):
            return ValidationResult(
                is_valid=False,
                record_type=record_type,
                issues=[ValidationIssue(code="type_error", message="Record must be a dict")],
                index=index,
            )

        schema = RAW_TICK_SCHEMA if record_type == "tick" else RAW_OPTIONS_SCHEMA
        required = [field for field in schema.keys() if field in {"token", "ltp", "volume", "timestamp"}]
        missing = [field for field in required if field not in record or record[field] is None]

        issues: list[ValidationIssue] = []
        if missing:
            issues.append(
                ValidationIssue(
                    code="missing_required_fields",
                    message=f"Missing required fields: {missing}",
                    field=missing[0],
                    value=missing,
                )
            )

        timestamp = self._safe_int(record.get("timestamp", record.get("ts")))
        if timestamp is None or not is_valid_timestamp_ms(timestamp):
            issues.append(
                ValidationIssue(
                    code="invalid_timestamp",
                    message="Timestamp is missing or outside valid range",
                    field="timestamp",
                    value=record.get("timestamp", record.get("ts")),
                )
            )

        token = str(record.get("token") or record.get("tok") or "").strip()
        fingerprint = self._fingerprint(record, record_type)
        duplicate = False
        sequence_mismatch = False
        timestamp_duplicate = False

        if token:
            duplicate = self._is_duplicate(record_type, token, fingerprint)
            timestamp_duplicate = self._is_timestamp_duplicate(token, timestamp)
            sequence_mismatch = self._sequence_mismatch(token, record)

        if duplicate:
            issues.append(
                ValidationIssue(
                    code="duplicate_record",
                    message="Repeated record fingerprint detected",
                    field="fingerprint",
                    value=fingerprint,
                )
            )

        if timestamp_duplicate:
            issues.append(
                ValidationIssue(
                    code="duplicate_timestamp",
                    message="Timestamp repeated for token",
                    field="timestamp",
                    value=timestamp,
                )
            )

        if sequence_mismatch:
            issues.append(
                ValidationIssue(
                    code="sequence_mismatch",
                    message="Sequence gap or mismatch detected",
                    field="sequence",
                    value=record.get("sequence"),
                )
            )

        is_valid = len([issue for issue in issues if issue.code in {"type_error", "missing_required_fields", "invalid_timestamp"}]) == 0

        result = ValidationResult(
            is_valid=is_valid,
            record_type=record_type,
            issues=issues,
            duplicate=duplicate,
            sequence_mismatch=sequence_mismatch,
            timestamp_duplicate=timestamp_duplicate,
            fingerprint=fingerprint,
            index=index,
        )

        # Update history only after evaluation so the current record is tested against prior state.
        if token and fingerprint:
            self._remember(record_type, token, fingerprint, timestamp, record.get("sequence"))

        return result

    def _fingerprint(self, record: dict[str, Any], record_type: RecordType) -> str:
        token = str(record.get("token") or record.get("tok") or "").strip()
        timestamp = self._safe_int(record.get("timestamp", record.get("ts")))
        if record_type == "option":
            strike = record.get("strike", record.get("strike_price", ""))
            option_type = str(record.get("option_type", record.get("type", ""))).upper().strip()
            return f"option|{token}|{timestamp}|{strike}|{option_type}|{record.get('expiry', '')}"

        return (
            f"tick|{token}|{timestamp}|{record.get('ltp')}|{record.get('volume')}|"
            f"{record.get('direction', 0)}"
        )

    def _is_duplicate(self, record_type: RecordType, token: str, fingerprint: str) -> bool:
        if fingerprint in self._recent_fingerprint_sets[record_type]:
            return True
        return self._last_fingerprint_by_token.get(token) == fingerprint

    def _is_timestamp_duplicate(self, token: str, timestamp: Optional[int]) -> bool:
        if timestamp is None:
            return False
        return self._last_timestamp_by_token.get(token) == timestamp

    def _sequence_mismatch(self, token: str, record: dict[str, Any]) -> bool:
        sequence = record.get("sequence")
        if sequence is None:
            return False
        try:
            sequence_int = int(sequence)
        except (TypeError, ValueError):
            return True

        last_sequence = self._last_sequence_by_token.get(token)
        if last_sequence is None:
            return False
        return sequence_int != last_sequence + 1

    def _remember(
        self,
        record_type: RecordType,
        token: str,
        fingerprint: str,
        timestamp: Optional[int],
        sequence: Any,
    ) -> None:
        self._last_fingerprint_by_token[token] = fingerprint
        if timestamp is not None:
            self._last_timestamp_by_token[token] = timestamp

        sequence_int = self._safe_int(sequence)
        if sequence_int is not None:
            self._last_sequence_by_token[token] = sequence_int

        # Keep bounded history for duplicate lookups.
        fingerprints = self._recent_fingerprints[record_type]
        fingerprint_set = self._recent_fingerprint_sets[record_type]
        if fingerprint not in fingerprint_set:
            fingerprints.append(fingerprint)
            fingerprint_set.add(fingerprint)
        if len(fingerprints) > self.history_limit:
            removed = fingerprints.pop(0)
            fingerprint_set.discard(removed)

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


duplicate_validator = DuplicateValidator()