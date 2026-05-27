"""
Data Center — Cleaner -> Structured -> Review Pipeline
=====================================================
Strongest Version: Fully restored legacy heartbeats and review engine 
triggers with added Intelligence Enrichment for the Backend stream.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional, List, Dict, Any

from loguru import logger

from data_center.queues.cleaner_queue import cleaner_queue
from data_center.cleaners.tick_cleaner import tick_cleaner
from data_center.writers.structured_writer import structured_writer
from data_center.validators.review_engine import review_engine
from data_center.connectors.backend_connector import backend_connector
from data_center.pipeline.computed_pipeline import computed_pipeline


# Pipeline control
_thread: Optional[threading.Thread] = None
_stop_event: threading.Event = threading.Event()

# ──────────────────────────────────────────────
# OBSERVABILITY: legacy heartbeat counters
# ──────────────────────────────────────────────
_heartbeat_interval: float = 60.0
_last_heartbeat_mono: float = 0.0

_total_batches_consumed: int = 0
_total_cleaned_records: int = 0
_total_review_triggers: int = 0

# Optimization Settings
BATCH_WRITE_THRESHOLD = 50
BUFFER_TIMEOUT_SEC = 2.0


async def _consume_loop(batch_size: int = 200, idle_sleep: float = 0.1, review_every: int = 10) -> None:
    logger.info("Clean pipeline (Intelligent & Optimized) starting...")
    
    global _last_heartbeat_mono, _total_batches_consumed, _total_cleaned_records, _total_review_triggers
    _last_heartbeat_mono = time.monotonic()
    
    buffer: List[Dict[str, Any]] = []
    last_write_time = time.monotonic()
    written_batches = 0

    try:
        while not _stop_event.is_set():
            try:
                batch = await cleaner_queue.get_batch(max_batch=batch_size, timeout=0.5)
            except Exception:
                await asyncio.sleep(idle_sleep)
                continue

            if batch:
                _total_batches_consumed += 1
                cleaned_batch = tick_cleaner.clean_batch(batch)
                
                for tick in cleaned_batch:
                    # 1) Derive Intelligence (Trend, Volatility, etc.)
                    # This now returns enrichment data
                    intelligence = computed_pipeline.process_tick(tick)
                    
                    # 2) Enrich the tick with intelligence before broadcasting
                    enriched_tick = {**tick, **intelligence}
                    
                    # 3) Broadcast the INTELLIGENT tick to Backend instantly
                    backend_connector.broadcast_tick(enriched_tick)
                    
                    # Disk Buffering (storing the enriched version for full audit trail)
                    buffer.append(enriched_tick)
                    _total_cleaned_records += 1

            # Bulk Write & Heartbeat logic
            now = time.monotonic()
            
            # Legacy Heartbeat Log
            if (now - _last_heartbeat_mono) >= _heartbeat_interval:
                logger.info(f"Clean Pipeline Heartbeat | Cleaned: {_total_cleaned_records} | Batches: {_total_batches_consumed}")
                _last_heartbeat_mono = now

            if len(buffer) >= BATCH_WRITE_THRESHOLD or (buffer and (now - last_write_time) >= BUFFER_TIMEOUT_SEC):
                try:
                    structured_paths = structured_writer.write_batches_by_partition(buffer)
                    buffer.clear()
                    last_write_time = now
                    written_batches += 1
                    
                    # Legacy Review Engine Trigger
                    if written_batches % review_every == 0:
                        _total_review_triggers += 1
                        threading.Thread(target=review_engine.review, daemon=True).start()
                except Exception as e:
                    logger.error(f"Clean pipeline write failed: {e}")

            if not batch:
                await asyncio.sleep(idle_sleep)

    except Exception as e:
        logger.exception(f"Clean pipeline fatal error: {e}")

def _thread_main(batch_size: int = 200) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_consume_loop(batch_size=batch_size))
    finally:
        loop.close()

def start_clean_pipeline(batch_size: int = 200):
    global _thread, _stop_event
    if _thread and _thread.is_alive(): return
    _stop_event.clear()
    _thread = threading.Thread(target=_thread_main, args=(batch_size,), daemon=True, name="CleanPipeline")
    _thread.start()
    logger.info("Clean pipeline started (Intelligence Hub Active)")

def stop_clean_pipeline():
    global _thread, _stop_event
    if _thread:
        _stop_event.set()
        _thread.join(timeout=2.0)
        _thread = None
