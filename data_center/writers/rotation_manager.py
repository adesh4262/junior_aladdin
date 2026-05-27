"""
Junior Aladdin — Rotation Manager
=================================
Adaptive raw-file rotation controller for the data center pipeline.

Responsibilities:
  - Adaptive rotation
  - 1h / 2h / 4h duration logic
  - Rollover management

This module is intentionally small and stateful so it can be wired into
the raw writer without changing the broader pipeline architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from configs.storage_config import MAJOR_RAW, ROTATION_DEFAULT, ROTATION_HIGH, ROTATION_LOW, ROTATION_MODERATE
from data_center.utils.timestamps import format_date_partition, format_time_partition


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@dataclass(slots=True)
class RotationDecision:
    rotation_seconds: int
    reason: str


class RotationManager:
    """
    Tracks file rotation state for raw parquet output.

    The decision model is intentionally simple:
    - high load -> 1 hour
    - moderate load -> 2 hours
    - low load -> 4 hours

    The manager keeps the current file slot and decides when rollover is due.
    """

    def __init__(self, default_rotation_seconds: int = ROTATION_DEFAULT):
        self.default_rotation_seconds = default_rotation_seconds
        self._current_slot_start_ms: Optional[int] = None
        self._current_file_path: Optional[Path] = None
        self._current_symbol: Optional[str] = None
        self._current_date: Optional[str] = None
        self._current_rotation_seconds: int = default_rotation_seconds

    def decide_rotation_seconds(self, load_level: str = "moderate") -> RotationDecision:
        """Return the adaptive rotation interval in seconds."""
        normalized = str(load_level).strip().lower()
        if normalized in {"high", "busy", "hot"}:
            return RotationDecision(rotation_seconds=ROTATION_HIGH, reason="high_load")
        if normalized in {"low", "idle", "quiet"}:
            return RotationDecision(rotation_seconds=ROTATION_LOW, reason="low_load")
        if normalized in {"moderate", "default", "normal"}:
            return RotationDecision(rotation_seconds=ROTATION_MODERATE, reason="moderate_load")
        return RotationDecision(rotation_seconds=self.default_rotation_seconds, reason="fallback_default")

    def _slot_start_ms(self, timestamp_ms: int, rotation_seconds: int) -> int:
        slot_size_ms = max(rotation_seconds, 1) * 1000
        return (timestamp_ms // slot_size_ms) * slot_size_ms

    def _build_file_path(self, timestamp_ms: int, symbol: str, slot_start_ms: int) -> Path:
        date_str = format_date_partition(timestamp_ms)
        time_str = format_time_partition(slot_start_ms)
        return MAJOR_RAW / date_str / symbol / f"{time_str}.parquet"

    def current_file_path(self) -> Optional[Path]:
        return self._current_file_path

    def should_rollover(self, timestamp_ms: Optional[int] = None) -> bool:
        """Check whether the current file slot has expired."""
        if self._current_slot_start_ms is None:
            return True
        ts = timestamp_ms or _utc_now_ms()
        return ts >= (self._current_slot_start_ms + self._current_rotation_seconds * 1000)

    def update(self, timestamp_ms: int, symbol: str, load_level: str = "moderate") -> Path:
        """
        Update rotation state and return the active file path.

        If the current slot has expired or the symbol/date changed, a new path is created.
        """
        decision = self.decide_rotation_seconds(load_level)
        slot_start_ms = self._slot_start_ms(timestamp_ms, decision.rotation_seconds)
        date_str = format_date_partition(timestamp_ms)

        needs_new_file = (
            self._current_slot_start_ms is None
            or self._current_file_path is None
            or self._current_symbol != symbol
            or self._current_date != date_str
            or self._current_slot_start_ms != slot_start_ms
        )

        if needs_new_file:
            self._current_slot_start_ms = slot_start_ms
            self._current_rotation_seconds = decision.rotation_seconds
            self._current_symbol = symbol
            self._current_date = date_str
            self._current_file_path = self._build_file_path(timestamp_ms, symbol, slot_start_ms)
            self._current_file_path.parent.mkdir(parents=True, exist_ok=True)

        return self._current_file_path

    def rollover_state(self) -> dict[str, Any]:
        """Expose the current rollover state for monitoring."""
        return {
            "current_file_path": str(self._current_file_path) if self._current_file_path else None,
            "current_slot_start_ms": self._current_slot_start_ms,
            "current_symbol": self._current_symbol,
            "current_date": self._current_date,
            "current_rotation_seconds": self._current_rotation_seconds,
        }


rotation_manager = RotationManager()