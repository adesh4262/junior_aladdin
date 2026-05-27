"""
Junior Aladdin — Timestamp Validator
=====================================
Validates timestamp health before cleaner/structured stages.

Phase 2 scope:
  - invalid timestamps
  - missing continuity
  - ordering issues

The validator is stateful so it can detect gaps and out-of-order
timestamps across streaming records as well as within batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

from loguru import logger

from data_center.utils.timestamps import is_valid_timestamp_ms, timestamp_continuity_check


RecordType = Literal["tick", "option"]


@dataclass(slots=True)
class TimestampIssue:
    code: str
    message: str
    field: Optional[str] = None
    value: Any = None


@dataclass(slots=True)
class TimestampValidationResult:
    is_valid: bool
    record_type: RecordType
    issues: list[TimestampIssue] = field(default_factory=list)
    invalid_timestamp: bool = False
    continuity_gap: bool = False
    ordering_issue: bool = False
    timestamp: Optional[int] = None
    token: Optional[str] = None
    index: Optional[int] = None


class TimestampValidator:
    """
    Stateful timestamp validator.

    It tracks the last timestamp per token so streaming records can be
    checked for monotonic ordering and missing continuity.
    """

    def __init__(self, history_limit: int = 50_000, max_gap_ms: int = 5_000):
        self.history_limit = history_limit
        self.max_gap_ms = max_gap_ms
        self._last_timestamp_by_token: dict[str, int] = {}
        self._recent_timestamps: dict[RecordType, list[int]] = {"tick": [], "option": []}
        self._stats = {
            "total_seen": 0,
            "invalid_timestamps": 0,
            "continuity_gaps": 0,
            "ordering_issues": 0,
            "invalid_records": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_record(self, record: dict[str, Any], record_type: RecordType = "tick") -> TimestampValidationResult:
        """Validate one record and update streaming timestamp history."""
        result = self._validate_record(record, record_type)
        self._stats["total_seen"] += 1
        self._apply_stats(result)
        return result

    def validate_batch(
        self,
        records: Iterable[dict[str, Any]],
        record_type: RecordType = "tick",
    ) -> list[TimestampValidationResult]:
        """Validate a batch and check continuity/order in arrival order."""
        batch = list(records)
        results = [self._validate_record(record, record_type, index=index) for index, record in enumerate(batch)]

        batch_timestamps: list[int] = []
        last_timestamp_by_token: dict[str, int] = {}

        for index, (record, result) in enumerate(zip(batch, results)):
            timestamp = result.timestamp
            token = result.token

            if timestamp is None or token is None:
                continue

            batch_timestamps.append(timestamp)

            previous_timestamp = last_timestamp_by_token.get(token)
            if previous_timestamp is not None:
                if timestamp < previous_timestamp:
                    self._add_issue(
                        result,
                        code="ordering_issue",
                        message="Timestamp moved backwards for token",
                        field="timestamp",
                        value=timestamp,
                    )
                elif timestamp - previous_timestamp > self.max_gap_ms:
                    self._add_issue(
                        result,
                        code="continuity_gap",
                        message="Timestamp gap exceeds allowed continuity window",
                        field="timestamp",
                        value=timestamp,
                    )
            last_timestamp_by_token[token] = timestamp

        if len(batch_timestamps) > 1:
            gap_indices = set(timestamp_continuity_check(batch_timestamps, max_gap_ms=self.max_gap_ms))
            if gap_indices:
                logger.warning(
                    f"Timestamp continuity gaps detected in {record_type} batch: {sorted(gap_indices)}"
                )
                for index in gap_indices:
                    if 0 <= index < len(results):
                        self._add_issue(
                            results[index],
                            code="continuity_gap",
                            message="Timestamp continuity gap detected in batch",
                            field="timestamp",
                            value=results[index].timestamp,
                        )

        for result in results:
            self._stats["total_seen"] += 1
            self._apply_stats(result)

        return results

    def reset(self) -> None:
        """Clear timestamp tracking state."""
        self._last_timestamp_by_token.clear()
        self._recent_timestamps = {"tick": [], "option": []}
        self._stats = {
            "total_seen": 0,
            "invalid_timestamps": 0,
            "continuity_gaps": 0,
            "ordering_issues": 0,
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
    ) -> TimestampValidationResult:
        if not isinstance(record, dict):
            return TimestampValidationResult(
                is_valid=False,
                record_type=record_type,
                issues=[TimestampIssue(code="type_error", message="Record must be a dict")],
                index=index,
            )

        timestamp = self._safe_int(record.get("timestamp", record.get("ts")))
        token = str(record.get("token") or record.get("tok") or "").strip() or None

        issues: list[TimestampIssue] = []
        invalid_timestamp = False
        continuity_gap = False
        ordering_issue = False

        if timestamp is None or not is_valid_timestamp_ms(timestamp):
            invalid_timestamp = True
            issues.append(
                TimestampIssue(
                    code="invalid_timestamp",
                    message="Timestamp is missing or outside valid range",
                    field="timestamp",
                    value=record.get("timestamp", record.get("ts")),
                )
            )

        if token and timestamp is not None:
            last_timestamp = self._last_timestamp_by_token.get(token)
            if last_timestamp is not None:
                if timestamp < last_timestamp:
                    ordering_issue = True
                    issues.append(
                        TimestampIssue(
                            code="ordering_issue",
                            message="Timestamp moved backwards for token",
                            field="timestamp",
                            value=timestamp,
                        )
                    )
                elif timestamp == last_timestamp:
                    ordering_issue = True
                    issues.append(
                        TimestampIssue(
                            code="ordering_issue",
                            message="Timestamp repeated for token",
                            field="timestamp",
                            value=timestamp,
                        )
                    )
                elif timestamp - last_timestamp > self.max_gap_ms:
                    continuity_gap = True
                    issues.append(
                        TimestampIssue(
                            code="continuity_gap",
                            message="Timestamp gap exceeds allowed continuity window",
                            field="timestamp",
                            value=timestamp,
                        )
                    )

            if not invalid_timestamp:
                self._remember(token, timestamp, record_type)

        is_valid = not (invalid_timestamp or continuity_gap or ordering_issue)

        return TimestampValidationResult(
            is_valid=is_valid,
            record_type=record_type,
            issues=issues,
            invalid_timestamp=invalid_timestamp,
            continuity_gap=continuity_gap,
            ordering_issue=ordering_issue,
            timestamp=timestamp,
            token=token,
            index=index,
        )

    def _remember(self, token: str, timestamp: Optional[int], record_type: RecordType) -> None:
        if timestamp is None:
            return

        self._last_timestamp_by_token[token] = timestamp

        recent = self._recent_timestamps[record_type]
        recent.append(timestamp)
        if len(recent) > self.history_limit:
            recent.pop(0)

    def _apply_stats(self, result: TimestampValidationResult) -> None:
        if not result.is_valid:
            self._stats["invalid_records"] += 1
        if result.invalid_timestamp:
            self._stats["invalid_timestamps"] += 1
        if result.continuity_gap:
            self._stats["continuity_gaps"] += 1
        if result.ordering_issue:
            self._stats["ordering_issues"] += 1

    def _add_issue(
        self,
        result: TimestampValidationResult,
        code: str,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
    ) -> None:
        result.issues.append(TimestampIssue(code=code, message=message, field=field, value=value))
        if code == "invalid_timestamp":
            result.invalid_timestamp = True
        elif code == "continuity_gap":
            result.continuity_gap = True
        elif code == "ordering_issue":
            result.ordering_issue = True
        result.is_valid = False

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


timestamp_validator = TimestampValidator()
