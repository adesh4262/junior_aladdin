"""
Junior Aladdin — Minor Data Writer
==================================
Writes structured contextual snapshots into partitioned parquet files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
from loguru import logger

from configs.storage_config import MINOR_STRUCTURED
from data_center.schemas.minor_schema import MINOR_SNAPSHOT_SCHEMA
from data_center.utils.parquet_utils import append_parquet
from data_center.utils.timestamps import format_date_partition

class MinorWriter:
    """Write structured minor records using strict schema enforcement."""

    def __init__(self, base_path: Path = MINOR_STRUCTURED):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def write_snapshot(self, snapshot: Dict[str, Any]) -> Optional[Path]:
        """Write a single minor snapshot to parquet."""
        try:
            timestamp = snapshot.get("timestamp")
            if timestamp is None:
                return None
                
            date_str = format_date_partition(timestamp)
            symbol = snapshot.get("symbol", "NIFTY")
            
            folder_path = self.base_path / date_str / symbol
            os.makedirs(folder_path, exist_ok=True)
            file_path = folder_path / "snapshots.parquet"
            
            # Use the strict schema from minor_schema.py
            frame = pl.DataFrame([snapshot], schema=MINOR_SNAPSHOT_SCHEMA)
            append_parquet(frame, file_path)
            
            logger.info(f"Minor snapshot saved: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Minor snapshot write failed: {e}")
            return None

minor_writer = MinorWriter()
