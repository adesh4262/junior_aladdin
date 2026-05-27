"""
Junior Aladdin — Minor Snapshot Collector
=========================================
Collects contextual snapshots (PCR, Max Pain, etc.) periodically.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from loguru import logger

from src.core.option_chain_poller import OptionChainPoller
from data_center.transformers.minor_transformer import minor_transformer
from data_center.writers.minor_writer import minor_writer

class SnapshotCollector:
    """Collects and processes minor contextual data."""

    def __init__(self, auth_manager, instrument_mapper):
        self.poller = OptionChainPoller(auth_manager, instrument_mapper)
        self._stats = {"snapshots_collected": 0}

    def collect_once(self, spot_price: float, vix: float = 0.0, symbol: str = "NIFTY") -> Optional[Dict[str, Any]]:
        """Perform a single collection cycle."""
        try:
            # 1) Poll the option chain
            chain = self.poller.poll(spot_price)
            if not chain:
                logger.warning("Empty chain during snapshot collection")
                return None
                
            # 2) Transform into minor metrics
            timestamp = int(time.time() * 1000)
            snapshot = minor_transformer.transform_snapshot(
                chain_data=chain,
                spot_price=spot_price,
                vix=vix,
                timestamp=timestamp,
                symbol=symbol
            )
            
            # 3) Write to structured storage
            minor_writer.write_snapshot(snapshot)
            
            self._stats["snapshots_collected"] += 1
            return snapshot
        except Exception as e:
            logger.error(f"Snapshot collection failed: {e}")
            return None

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
