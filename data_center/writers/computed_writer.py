"""
Junior Aladdin — Computed Data Writer
=====================================
Writes intelligence metrics into categorized parquet files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
from loguru import logger

from configs.storage_config import DATA_CENTER_ROOT
from data_center.utils.parquet_utils import append_parquet
from data_center.utils.timestamps import format_date_partition

class ComputedWriter:
    """Write computed intelligence into the data_center/computed/ hierarchy."""

    def __init__(self, base_path: Path = DATA_CENTER_ROOT / "computed"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def write_metric(self, category: str, data: Dict[str, Any]) -> Optional[Path]:
        """Write a computed metric (trend, volatility, etc.) to its folder."""
        try:
            timestamp = data.get("timestamp")
            symbol = data.get("symbol", "NIFTY")
            date_str = format_date_partition(timestamp)
            
            # Structure: computed/{category}/{YYYY-MM-DD}/{SYMBOL}/metrics.parquet
            folder_path = self.base_path / category / date_str / symbol
            os.makedirs(folder_path, exist_ok=True)
            file_path = folder_path / "metrics.parquet"
            
            frame = pl.DataFrame([data])
            append_parquet(frame, file_path)
            
            return file_path
        except Exception as e:
            logger.error(f"Computed metric write failed ({category}): {e}")
            return None

computed_writer = ComputedWriter()
