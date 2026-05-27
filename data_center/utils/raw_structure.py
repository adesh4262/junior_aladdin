"""
Junior Aladdin — Raw Structure Utilities
=========================================
Helpers for the fixed major/raw partition layout used by the live ingestion pipeline.

Phase 7 contract:
    major/raw/YYYY-MM-DD/SYMBOL/HH_MM.parquet
"""

from __future__ import annotations

from pathlib import Path

from configs.storage_config import MAJOR_RAW


def major_raw_date_dir(date: str) -> Path:
    """Return the date directory under major/raw and ensure it exists."""
    path = MAJOR_RAW / date
    path.mkdir(parents=True, exist_ok=True)
    return path


def major_raw_symbol_dir(date: str, symbol: str) -> Path:
    """Return the symbol directory under major/raw/date and ensure it exists."""
    path = major_raw_date_dir(date) / symbol
    path.mkdir(parents=True, exist_ok=True)
    return path


def major_raw_file_path(date: str, symbol: str, time_slot: str) -> Path:
    """Return the raw parquet file path for a date/symbol/time partition."""
    directory = major_raw_symbol_dir(date, symbol)
    return directory / f"{time_slot}.parquet"


def ensure_major_raw_structure(date: str, symbol: str, time_slot: str | None = None) -> Path:
    """
    Ensure the requested major/raw structure exists.

    If time_slot is provided, returns the parquet file path; otherwise returns the symbol directory.
    """
    if time_slot:
        return major_raw_file_path(date, symbol, time_slot)
    return major_raw_symbol_dir(date, symbol)


def verify_major_raw_structure(date: str, symbol: str, time_slot: str | None = None) -> bool:
    """Verify the expected raw structure exists."""
    if time_slot:
        return major_raw_file_path(date, symbol, time_slot).parent.exists()
    return major_raw_symbol_dir(date, symbol).exists()