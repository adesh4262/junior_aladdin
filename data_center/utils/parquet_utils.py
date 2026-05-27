"""
Junior Aladdin — Parquet Utilities
====================================
Functions for parquet read/write, schema conversion, and partition handling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from configs.storage_config import PARQUET_COMPRESSION, PARQUET_ROW_GROUP_SIZE


def write_parquet(
    df: pl.DataFrame,
    filepath: Path,
    compression: str = PARQUET_COMPRESSION,
    row_group_size: int = PARQUET_ROW_GROUP_SIZE,
) -> None:
    """
    Write a polars DataFrame to parquet file.
    
    Args:
        df: DataFrame to write
        filepath: Output file path
        compression: Compression codec (zstd, snappy, lz4)
        row_group_size: Number of rows per row group
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        str(filepath),
        compression=compression,
        row_group_size=row_group_size,
    )


def read_parquet(filepath: Path) -> Optional[pl.DataFrame]:
    """Read a parquet file, returning None if file doesn't exist."""
    if not filepath.exists():
        return None
    return pl.read_parquet(str(filepath))


def append_parquet(
    df: pl.DataFrame,
    filepath: Path,
    compression: str = PARQUET_COMPRESSION,
) -> None:
    """
    Append data to an existing parquet file.
    If file doesn't exist, creates a new one.
    """
    existing = read_parquet(filepath)
    if existing is not None:
        combined = pl.concat([existing, df], how="vertical")
        write_parquet(combined, filepath, compression=compression)
    else:
        write_parquet(df, filepath, compression=compression)


def list_parquet_files(directory: Path) -> List[Path]:
    """List all parquet files in a directory (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def get_partition_path(
    base_path: Path,
    date_str: str,
    symbol: str,
    expiry: Optional[str] = None,
    time_str: Optional[str] = None,
) -> Path:
    """
    Build a partitioned file path.
    
    Example:
        base_path / 2026-05-25 / NIFTY / expiry / 10_11.parquet
    """
    path = base_path / date_str / symbol
    if expiry:
        path = path / expiry
    if time_str:
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{time_str}.parquet"
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_parquet_integrity(filepath: Path) -> bool:
    """
    Check if a parquet file is valid and readable.
    Returns True if valid, False otherwise.
    """
    try:
        df = read_parquet(filepath)
        return df is not None and len(df) > 0
    except Exception:
        return False