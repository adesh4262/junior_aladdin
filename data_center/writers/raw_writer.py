"""
Junior Aladdin — Raw Writer
===========================
Strongest Version: Optimized for Hourly Data Partitioning to prevent
the Small Files Problem in the Raw layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import polars as pl
from loguru import logger

from configs.storage_config import MAJOR_RAW
from data_center.utils.parquet_utils import append_parquet
from data_center.utils.timestamps import epoch_ms_to_datetime, format_date_partition, format_time_partition


@dataclass(slots=True)
class RawWriterStats:
    total_batches: int = 0
    total_records: int = 0
    total_files_written: int = 0
    total_write_errors: int = 0


class RawWriter:
    """Batch writer for raw data using hourly partitioning."""

    def __init__(self, base_path: Path = MAJOR_RAW):
        self.base_path = base_path
        self._stats = RawWriterStats()

    def _resolve_symbol(self, record: dict[str, Any]) -> str:
        symbol = str(record.get("symbol") or record.get("underlying_symbol") or "NIFTY").strip()
        return symbol or "NIFTY"

    def _resolve_timestamp(self, record: dict[str, Any]) -> int:
        timestamp = record.get("timestamp", record.get("ts"))
        if timestamp is None:
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            return int(timestamp)
        except (TypeError, ValueError):
            return int(datetime.now(timezone.utc).timestamp() * 1000)

    def build_partition_path(self, record: dict[str, Any]) -> Path:
        """Build raw parquet path. Now Uses format_time_partition (Hourly)."""
        timestamp = self._resolve_timestamp(record)
        symbol = self._resolve_symbol(record)
        date_str = format_date_partition(timestamp)
        time_str = format_time_partition(timestamp) # returns only HH
        path = self.base_path / date_str / symbol / f"{time_str}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_batch(self, records: Iterable[dict[str, Any]]) -> Optional[Path]:
        batch = list(records)
        if not batch: return None
        try:
            df = pl.DataFrame(batch)
            first_path = self.build_partition_path(batch[0])
            append_parquet(df, first_path)
            self._stats.total_batches += 1
            self._stats.total_records += len(batch)
            return first_path
        except Exception as exc:
            logger.error(f"Raw batch write failed: {exc}")
            return None

    def write_batches_by_partition(self, records: Iterable[dict[str, Any]]) -> list[Path]:
        grouped: dict[Path, list[dict[str, Any]]] = {}
        for record in records:
            path = self.build_partition_path(record)
            grouped.setdefault(path, []).append(record)
        written_paths: list[Path] = []
        for path, group in grouped.items():
            try:
                append_parquet(pl.DataFrame(group), path)
                written_paths.append(path)
            except Exception as exc:
                logger.error(f"Partition batch write failed for {path}: {exc}")
        return written_paths

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_batches": self._stats.total_batches,
            "total_records": self._stats.total_records,
        }

raw_writer = RawWriter()
