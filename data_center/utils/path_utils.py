"""
Junior Aladdin — Path Utilities
=================================
Functions for building and resolving data center file paths.
"""

from pathlib import Path
from typing import Optional

from configs.storage_config import (
    MAJOR_RAW, MAJOR_CLEANED, MAJOR_STRUCTURED, MAJOR_REVIEW,
    MINOR_RAW, MINOR_CLEANED, MINOR_STRUCTURED, MINOR_REVIEW,
    COMPUTED_VOLATILITY, COMPUTED_STRUCTURE, COMPUTED_LIQUIDITY,
    COMPUTED_ORDERFLOW, COMPUTED_TREND, COMPUTED_BIAS,
)


def major_raw_path(date: str, symbol: str, time_slot: str) -> Path:
    """Build path: major/raw/YYYY-MM-DD/SYMBOL/HH_MM.parquet"""
    return MAJOR_RAW / date / symbol / f"{time_slot}.parquet"


def major_cleaned_path(date: str, symbol: str, time_slot: str) -> Path:
    """Build path: major/cleaned/YYYY-MM-DD/SYMBOL/HH_MM.parquet"""
    return MAJOR_CLEANED / date / symbol / f"{time_slot}.parquet"


def major_structured_path(date: str, symbol: str, expiry: str, time_slot: str) -> Path:
    """Build path: major/structured/YYYY-MM-DD/SYMBOL/expiry/HH_MM.parquet"""
    return MAJOR_STRUCTURED / date / symbol / expiry / f"{time_slot}.parquet"


def major_review_path(date: str, symbol: str) -> Path:
    """Build path: major/review/YYYY-MM-DD/SYMBOL_verification.json"""
    return MAJOR_REVIEW / date / f"{symbol}_verification.json"


def minor_raw_path(date: str, symbol: str) -> Path:
    """Build path: minor/raw/YYYY-MM-DD/SYMBOL/"""
    path = MINOR_RAW / date / symbol
    path.mkdir(parents=True, exist_ok=True)
    return path


def minor_structured_path(date: str, symbol: str) -> Path:
    """Build path: minor/structured/YYYY-MM-DD/SYMBOL/"""
    path = MINOR_STRUCTURED / date / symbol
    path.mkdir(parents=True, exist_ok=True)
    return path


def computed_path(computed_type: str, date: str) -> Path:
    """Build path for computed data by type and date."""
    base_map = {
        "volatility": COMPUTED_VOLATILITY,
        "structure": COMPUTED_STRUCTURE,
        "liquidity": COMPUTED_LIQUIDITY,
        "orderflow": COMPUTED_ORDERFLOW,
        "trend": COMPUTED_TREND,
        "bias": COMPUTED_BIAS,
    }
    base = base_map.get(computed_type)
    if base is None:
        raise ValueError(f"Unknown computed type: {computed_type}")
    path = base / date
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_date_dir(base_path: Path, date: str, symbol: str) -> Path:
    """Ensure and return a date/symbol directory."""
    path = base_path / date / symbol
    path.mkdir(parents=True, exist_ok=True)
    return path