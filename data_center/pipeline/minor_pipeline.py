"""
Junior Aladdin — Minor Data Pipeline
====================================
Background loop for periodic minor snapshot collection.

Strongest Version: Optimized startup checks to ensure instant 
collection as soon as first tick arrives.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Any
from loguru import logger

from data_center.collectors.snapshot_collector import SnapshotCollector

# Control
_thread: Optional[threading.Thread] = None
_stop_event: threading.Event = threading.Event()

def _collection_loop(collector: SnapshotCollector, interval: int, market_state: Any):
    logger.info(f"Minor pipeline loop starting (Responsive Startup Mode)")
    
    first_snapshot_taken = False
    
    while not _stop_event.is_set():
        try:
            # Get current spot and vix from state
            snap = market_state.snapshot() if hasattr(market_state, "snapshot") else {}
            spot = float(snap.get("spot", 0.0) or 0.0)
            vix = float(snap.get("vix", 0.0) or 0.0)
            
            # Start only if we have a valid spot price
            if spot > 0:
                logger.info(f"Attempting Minor Snapshot collection (Spot: {spot}, VIX: {vix})...")
                res = collector.collect_once(spot_price=spot, vix=vix)
                if res:
                    logger.success(f"Minor Snapshot stored: PCR={res.get('pcr')}")
                    first_snapshot_taken = True
            
            # Responsive sleep logic:
            # If we haven't taken the first snapshot yet, check every 1s.
            # Once started, wait for the full interval.
            wait_time = interval if first_snapshot_taken else 1.0
            
            for _ in range(int(wait_time)):
                if _stop_event.is_set(): break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Minor pipeline loop error: {e}")
            time.sleep(5)

def start_minor_pipeline(auth_manager, instrument_mapper, market_state, interval: int = 60):
    global _thread, _stop_event
    if _thread and _thread.is_alive(): return
    collector = SnapshotCollector(auth_manager, instrument_mapper)
    _stop_event.clear()
    _thread = threading.Thread(target=_collection_loop, args=(collector, interval, market_state), daemon=True, name="MinorPipeline")
    _thread.start()
    logger.info("Minor pipeline thread spawned (Strong Mode)")

def stop_minor_pipeline():
    global _thread, _stop_event
    if _thread:
        _stop_event.set()
        _thread.join(timeout=2.0)
        _thread = None
