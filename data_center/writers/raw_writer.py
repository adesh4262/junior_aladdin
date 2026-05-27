"""
Junior Aladdin — Raw Writer
===========================
Writes raw market data batches into partitioned parquet files.

Responsibilities:
  - Batch write raw tick/option records
  - Parquet write using shared storage utilities
  - Rolling file creation by time slot
  - Partition handling by date and symbol

Data Center Architecture compliant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import polars as pl
from loguru import logger

from configs.storage_config import MAJOR_RAW, PARTITION_TIME_FORMAT
from data_center.utils.parquet_utils import append_parquet
from data_center.utils.timestamps import epoch_ms_to_datetime, format_date_partition, format_time_partition


def _utc_datetime_from_timestamp(timestamp_ms: int) -> datetime:
    return epoch_ms_to_datetime(timestamp_ms)


@dataclass(slots=True)
class RawWriterStats:
    total_batches: int = 0
    total_records: int = 0
    total_files_written: int = 0
    total_write_errors: int = 0


class RawWriter:
    """
    Batch writer for raw data.

    Default storage layout:
        major/raw/YYYY-MM-DD/SYMBOL/HH_MM.parquet
    """

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
        """Build raw parquet path for a single record."""
        timestamp = self._resolve_timestamp(record)
        symbol = self._resolve_symbol(record)
        date_str = format_date_partition(timestamp)
        time_str = format_time_partition(timestamp)
        path = self.base_path / date_str / symbol / f"{time_str}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_batch(self, records: Iterable[dict[str, Any]]) -> Optional[Path]:
        """Write a batch of raw records to parquet. Returns the output path when written."""
        batch = list(records)
        if not batch:
            return None

        try:
            df = pl.DataFrame(batch)
            first_path = self.build_partition_path(batch[0])
            append_parquet(df, first_path)

            self._stats.total_batches += 1
            self._stats.total_records += len(batch)
            self._stats.total_files_written += 1

            logger.info(f"Wrote raw batch: {len(batch)} records -> {first_path}")
            return first_path
        except Exception as exc:
            self._stats.total_write_errors += 1
            logger.error(f"Raw batch write failed: {exc}")
            return None

    def write_record(self, record: dict[str, Any]) -> Optional[Path]:
        """Write a single raw record using the same batch pipeline."""
        return self.write_batch([record])

    def write_batches_by_partition(self, records: Iterable[dict[str, Any]]) -> list[Path]:
        """
        Group records by partition path and write each group separately.
        Useful when a batch spans multiple symbols or time slots.
        """
        grouped: dict[Path, list[dict[str, Any]]] = {}
        for record in records:
            path = self.build_partition_path(record)
            grouped.setdefault(path, []).append(record)

        written_paths: list[Path] = []
        for path, group in grouped.items():
            try:
                df = pl.DataFrame(group)
                append_parquet(df, path)
                self._stats.total_batches += 1
                self._stats.total_records += len(group)
                self._stats.total_files_written += 1
                written_paths.append(path)
                logger.info(f"Wrote partition batch: {len(group)} records -> {path}")
            except Exception as exc:
                self._stats.total_write_errors += 1
                logger.error(f"Partition batch write failed for {path}: {exc}")

        return written_paths

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_batches": self._stats.total_batches,
            "total_records": self._stats.total_records,
            "total_files_written": self._stats.total_files_written,
            "total_write_errors": self._stats.total_write_errors,
        }


raw_writer = RawWriter()