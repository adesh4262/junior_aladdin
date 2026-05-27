"""
Junior Aladdin — Minor Data Writer
==================================
Writes structured contextual snapshots into partitioned parquet files.
Strongest Version: Now uses hourly descriptive naming to prevent file bloat.
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
from data_center.utils.timestamps import format_date_partition, format_time_partition

class MinorWriter:
    """Write structured minor records using strict schema and hourly ranges."""

    def __init__(self, base_path: Path = MINOR_STRUCTURED):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def write_snapshot(self, snapshot: Dict[str, Any]) -> Optional[Path]:
        """Write a single minor snapshot to an hourly parquet file."""
        try:
            timestamp = snapshot.get("timestamp")
            if timestamp is None: return None
                
            date_str = format_date_partition(timestamp)
            time_range = format_time_partition(timestamp) # returns "09_10", "10_11", etc.
            symbol = snapshot.get("symbol", "NIFTY")
            
            # Subfolder structure preserved, but filename improved
            folder_path = self.base_path / date_str / symbol
            os.makedirs(folder_path, exist_ok=True)
            file_path = folder_path / f"snapshots_{time_range}.parquet"
            
            frame = pl.DataFrame([snapshot], schema=MINOR_SNAPSHOT_SCHEMA)
            append_parquet(frame, file_path)
            
            logger.info(f"Minor snapshot stored in range {time_range}: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Minor snapshot write failed: {e}")
            return None

minor_writer = MinorWriter()
