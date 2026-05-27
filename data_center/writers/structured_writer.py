"""
Junior Aladdin — Structured Writer
==================================
Writes structured tick and option records into partitioned parquet files.

Strongest Version: Improved validation logging and resilient record 
normalization to prevent silent data loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import polars as pl
from loguru import logger

from configs.storage_config import MAJOR_STRUCTURED
from data_center.schemas.options_schema import STRUCTURED_OPTIONS_SCHEMA
from data_center.schemas.tick_schema import STRUCTURED_TICK_SCHEMA
from data_center.utils.structured_structure import major_structured_file_path
from data_center.utils.parquet_utils import append_parquet
from data_center.utils.timestamps import format_date_partition, format_time_partition


STRUCTURED_TICK_FIELDS = list(STRUCTURED_TICK_SCHEMA.keys())
STRUCTURED_OPTIONS_FIELDS = list(STRUCTURED_OPTIONS_SCHEMA.keys())


def _utc_timestamp_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@dataclass(slots=True)
class StructuredWriterStats:
    total_batches: int = 0
    total_records: int = 0
    total_files_written: int = 0
    total_write_errors: int = 0
    invalid_records: int = 0


class StructuredWriter:
    """Write structured records into the major/structured hierarchy."""

    def __init__(self, base_path: Path = MAJOR_STRUCTURED):
        self.base_path = base_path
        self._stats = StructuredWriterStats()

    def _resolve_record_type(self, record: dict[str, Any]) -> str:
        # SMC/Option specific fields detection
        if any(f in record for f in ("moneyness", "spot_ltp", "option_type", "strike", "iv")):
            return "option"
        return "tick"

    def _resolve_symbol(self, record: dict[str, Any]) -> str:
        symbol = str(record.get("symbol") or record.get("underlying_symbol") or record.get("token") or "NIFTY").strip()
        return symbol or "NIFTY"

    def _resolve_expiry(self, record: dict[str, Any], record_type: str) -> str:
        if record_type == "option":
            expiry = str(record.get("expiry") or "unknown_expiry").strip()
            return expiry or "unknown_expiry"
        return "expiry"

    def _resolve_timestamp(self, record: dict[str, Any]) -> int:
        timestamp = record.get("timestamp", record.get("ts"))
        if timestamp is None:
            return _utc_timestamp_ms()
        try:
            return int(timestamp)
        except (TypeError, ValueError):
            return _utc_timestamp_ms()

    def build_partition_path(self, record: dict[str, Any]) -> Path:
        """Build structured parquet path for a single record."""
        timestamp = self._resolve_timestamp(record)
        record_type = self._resolve_record_type(record)
        symbol = self._resolve_symbol(record)
        expiry = self._resolve_expiry(record, record_type)
        date_str = format_date_partition(timestamp)
        time_str = format_time_partition(timestamp)
        path = major_structured_file_path(date_str, symbol, time_str, expiry, self.base_path)
        return path

    def _validate_record(self, record: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False

        record_type = self._resolve_record_type(record)
        required_fields = STRUCTURED_OPTIONS_FIELDS if record_type == "option" else STRUCTURED_TICK_FIELDS

        missing = [f for f in required_fields if f not in record]
        if missing:
            # Strong Logging: Identity exactly what's missing
            token = record.get('token', 'unknown')
            logger.warning(f"Record validation failed for {token} ({record_type}). Missing fields: {missing}")
            return False

        return True

    def _normalize_record(self, record: dict[str, Any]) -> Optional[dict[str, Any]]:
        # Pre-normalization to satisfy validation
        rec = dict(record)
        
        # Ensure mandatory metadata exists before validation check
        if "sequence" not in rec: rec["sequence"] = 0
        if "exchange" not in rec: rec["exchange"] = "NSE"
        if "symbol" not in rec: rec["symbol"] = str(rec.get("token", "NIFTY"))

        if not self._validate_record(rec):
            return None

        record_type = self._resolve_record_type(rec)
        try:
            normalized = dict(rec)
            normalized["token"] = str(normalized["token"]).strip()
            normalized["ltp"] = float(normalized["ltp"])
            normalized["volume"] = int(normalized["volume"])
            normalized["open"] = float(normalized.get("open", 0.0))
            normalized["high"] = float(normalized.get("high", 0.0))
            normalized["low"] = float(normalized.get("low", 0.0))
            normalized["close"] = float(normalized.get("close", 0.0))
            normalized["timestamp"] = int(normalized.get("timestamp", _utc_timestamp_ms()))
            normalized["sequence"] = int(normalized.get("sequence", 0))
            normalized["exchange"] = str(normalized.get("exchange") or "NSE").strip() or "NSE"
            normalized["symbol"] = str(normalized.get("symbol") or normalized.get("token") or "NIFTY").strip() or "NIFTY"

            if record_type == "option":
                normalized["oi"] = int(normalized.get("oi", 0))
                normalized["oi_change"] = int(normalized.get("oi_change", 0))
                normalized["iv"] = float(normalized.get("iv", 0.0))
                normalized["strike"] = float(normalized.get("strike", 0.0))
                normalized["option_type"] = str(normalized.get("option_type", "")).strip().upper()
                normalized["expiry"] = str(normalized.get("expiry", "")).strip()
                normalized["spot_ltp"] = float(normalized["spot_ltp"]) if normalized.get("spot_ltp") is not None else None
                normalized["moneyness"] = str(normalized.get("moneyness", "ATM")).strip().upper() or "ATM"
                
                # Double check option specific mandatory fields
                for f in ("oi", "strike", "option_type", "expiry"):
                    if normalized.get(f) is None:
                        logger.error(f"Option record missing critical field {f} after normalization")
                        return None
                        
        except (TypeError, ValueError, KeyError) as e:
            logger.error(f"Normalization error: {e}")
            return None

        return normalized

    def write_batch(self, records: Iterable[dict[str, Any]]) -> Optional[Path]:
        """Write a batch of structured records to parquet. Returns the output path when written."""
        batch = list(records)
        if not batch:
            return None

        normalized_batch = [record for record in (self._normalize_record(item) for item in batch) if record is not None]
        if not normalized_batch:
            self._stats.invalid_records += len(batch)
            return None

        try:
            first_path = self.build_partition_path(normalized_batch[0])
            frame = pl.DataFrame(normalized_batch)
            append_parquet(frame, first_path)

            self._stats.total_batches += 1
            self._stats.total_records += len(normalized_batch)
            self._stats.total_files_written += 1

            logger.info(f"Wrote structured batch: {len(normalized_batch)} records -> {first_path}")
            return first_path
        except Exception as exc:
            self._stats.total_write_errors += 1
            logger.error(f"Structured batch write failed: {exc}")
            return None

    def write_record(self, record: dict[str, Any]) -> Optional[Path]:
        return self.write_batch([record])

    def write_batches_by_partition(self, records: Iterable[dict[str, Any]]) -> list[Path]:
        grouped: dict[Path, list[dict[str, Any]]] = {}
        for record in records:
            normalized = self._normalize_record(record)
            if normalized is None:
                self._stats.invalid_records += 1
                continue
            path = self.build_partition_path(normalized)
            grouped.setdefault(path, []).append(normalized)

        written_paths: list[Path] = []
        for path, group in grouped.items():
            try:
                append_parquet(pl.DataFrame(group), path)
                self._stats.total_batches += 1
                self._stats.total_records += len(group)
                self._stats.total_files_written += 1
                written_paths.append(path)
                logger.info(f"Wrote structured partition batch: {len(group)} records -> {path}")
            except Exception as exc:
                self._stats.total_write_errors += 1
                logger.error(f"Structured partition batch write failed for {path}: {exc}")

        return written_paths

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_batches": self._stats.total_batches,
            "total_records": self._stats.total_records,
            "total_files_written": self._stats.total_files_written,
            "total_write_errors": self._stats.total_write_errors,
            "invalid_records": self._stats.invalid_records,
        }


structured_writer = StructuredWriter()
