"""
Junior Aladdin — Structured Structure Utilities
===============================================
Helpers for the fixed major/structured partition layout used by the
cleaned and transformed pipeline.

Phase 7 contract:
    major/structured/YYYY-MM-DD/SYMBOL/expiry/HH_MM.parquet
"""

from __future__ import annotations

from pathlib import Path

from configs.storage_config import MAJOR_STRUCTURED


def major_structured_date_dir(date: str, base_path: Path = MAJOR_STRUCTURED) -> Path:
    """Return the date directory under major/structured and ensure it exists."""
    path = base_path / date
    path.mkdir(parents=True, exist_ok=True)
    return path


def major_structured_symbol_dir(date: str, symbol: str, base_path: Path = MAJOR_STRUCTURED) -> Path:
    """Return the symbol directory under major/structured/date and ensure it exists."""
    path = major_structured_date_dir(date, base_path) / symbol
    path.mkdir(parents=True, exist_ok=True)
    return path


def major_structured_partition_dir(
    date: str,
    symbol: str,
    partition: str = "expiry",
    base_path: Path = MAJOR_STRUCTURED,
) -> Path:
    """Return the partition directory under major/structured/date/symbol and ensure it exists."""
    path = major_structured_symbol_dir(date, symbol, base_path) / partition
    path.mkdir(parents=True, exist_ok=True)
    return path


def major_structured_file_path(
    date: str,
    symbol: str,
    time_slot: str,
    partition: str = "expiry",
    base_path: Path = MAJOR_STRUCTURED,
) -> Path:
    """Return the structured parquet file path for a date/symbol/partition/time partition."""
    directory = major_structured_partition_dir(date, symbol, partition, base_path)
    return directory / f"{time_slot}.parquet"


def ensure_major_structured_structure(
    date: str,
    symbol: str,
    time_slot: str | None = None,
    partition: str = "expiry",
    base_path: Path = MAJOR_STRUCTURED,
) -> Path:
    """
    Ensure the requested major/structured hierarchy exists.

    If time_slot is provided, returns the parquet file path; otherwise returns the partition directory.
    """
    if time_slot:
        return major_structured_file_path(date, symbol, time_slot, partition, base_path)
    return major_structured_partition_dir(date, symbol, partition, base_path)


def verify_major_structured_structure(
    date: str,
    symbol: str,
    time_slot: str | None = None,
    partition: str = "expiry",
    base_path: Path = MAJOR_STRUCTURED,
) -> bool:
    """Verify the expected structured storage path exists."""
    if time_slot:
        return major_structured_file_path(date, symbol, time_slot, partition, base_path).parent.exists()
    return major_structured_partition_dir(date, symbol, partition, base_path).exists()
