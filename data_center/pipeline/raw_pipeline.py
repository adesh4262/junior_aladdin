"""
Data Center — Raw Writer Pipeline
=================================
Background consumer that drains `data_center.queues.tick_queue` and writes
to `data_center.writers.raw_writer` in batches. Designed to be non-blocking
and failure-tolerant; errors are logged and do not affect the main backend.

This module exposes `start_raw_pipeline()` and `stop_raw_pipeline()` helpers.

OBSERVABILITY:
- Heartbeat log every 60 seconds (alive + queue stats)
- Batch write confirmation logs (path, record count)
- Cleaner queue forward count
- Consumer alive/info logs on start/stop
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from loguru import logger

from data_center.queues.tick_queue import tick_queue
from data_center.writers.raw_writer import raw_writer
from data_center.queues.cleaner_queue import cleaner_queue


# Pipeline control
_thread: Optional[threading.Thread] = None
_stop_event: threading.Event = threading.Event()


# ──────────────────────────────────────────────
# OBSERVABILITY: lightweight heartbeat counters
# ──────────────────────────────────────────────
_heartbeat_interval: float = 60.0          # log every 60s
_last_heartbeat_mono: float = 0.0

_total_batches_read: int = 0
_total_records_read: int = 0
_total_records_forwarded: int = 0
_total_records_dropped_cleaner: int = 0
_total_write_errors: int = 0
_total_forward_errors: int = 0


async def _consume_loop(batch_size: int = 200, idle_sleep: float = 0.1) -> None:
    """Async consumer that pulls batches and writes them."""
    global _last_heartbeat_mono, _total_batches_read, _total_records_read
    global _total_records_forwarded, _total_records_dropped_cleaner
    global _total_write_errors, _total_forward_errors

    logger.info("Raw pipeline consumer loop starting", batch_size=batch_size)

    _last_heartbeat_mono = time.monotonic()

    try:
        while not _stop_event.is_set():
            try:
                batch = await tick_queue.get_batch(max_batch=batch_size, timeout=0.5)
            except Exception as e:
                logger.debug("tick_queue.get_batch error (will retry)", error=str(e))
                await asyncio.sleep(idle_sleep)
                continue

            if not batch:
                # ── OBSERVABILITY: periodic heartbeat ──
                now = time.monotonic()
                if (now - _last_heartbeat_mono) >= _heartbeat_interval:
                    _last_heartbeat_mono = now
                    qsize = tick_queue.qsize
                    fullness = tick_queue.fullness_pct
                    logger.info(
                        "Raw pipeline heartbeat",
                        queue_size=qsize,
                        queue_fullness_pct=fullness,
                        total_batches_read=_total_batches_read,
                        total_records_read=_total_records_read,
                        total_records_forwarded=_total_records_forwarded,
                        total_dropped_cleaner=_total_records_dropped_cleaner,
                        total_write_errors=_total_write_errors,
                        total_forward_errors=_total_forward_errors,
                    )
                await asyncio.sleep(idle_sleep)
                continue

            _total_batches_read += 1
            _total_records_read += len(batch)

            try:
                # Prefer batched write by partition for efficiency
                raw_paths = raw_writer.write_batches_by_partition(batch)

                # ── OBSERVABILITY: batch write confirmation ──
                if raw_paths:
                    if isinstance(raw_paths, (list, tuple)):
                        for p in raw_paths:
                            logger.info(
                                "Raw batch written",
                                path=str(p),
                                records=len(batch),
                                batch_number=_total_batches_read,
                            )
                    else:
                        logger.info(
                            "Raw batch written",
                            path=str(raw_paths),
                            records=len(batch),
                            batch_number=_total_batches_read,
                        )
                else:
                    logger.info(
                        "Raw batch processed (no new files)",
                        records=len(batch),
                        batch_number=_total_batches_read,
                    )

                # Forward raw records to cleaner queue for downstream processing.
                # Use non-blocking put_nowait to avoid impacting the raw writer thread.
                fwd_count = 0
                drop_count = 0
                try:
                    for record in batch:
                        try:
                            cleaner_queue.put_nowait(record)
                            fwd_count += 1
                        except Exception:
                            drop_count += 1
                            logger.debug("Cleaner queue put failed (dropping)")
                except Exception as _e:
                    _total_forward_errors += 1
                    logger.debug("Failed to forward to cleaner_queue", error=str(_e))

                _total_records_forwarded += fwd_count
                if drop_count > 0:
                    _total_records_dropped_cleaner += drop_count
                    logger.warning(
                        "Cleaner queue forward partial drop",
                        forwarded=fwd_count,
                        dropped=drop_count,
                    )

            except Exception as e:
                _total_write_errors += 1
                logger.error(
                    "Raw writer batch write failed (dropping batch)",
                    error=str(e),
                    batch_number=_total_batches_read,
                    records=len(batch),
                )

    except asyncio.CancelledError:
        logger.info("Raw pipeline consumer cancelled")
    except Exception as e:
        logger.exception("Raw pipeline consumer unexpected error: %s", e)
    finally:
        logger.info(
            "Raw pipeline consumer exiting",
            total_batches_read=_total_batches_read,
            total_records_read=_total_records_read,
        )


def _thread_main(batch_size: int = 200) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_consume_loop(batch_size=batch_size))
    finally:
        try:
            pending = asyncio.all_tasks(loop=loop)
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()


def start_raw_pipeline(batch_size: int = 200) -> None:
    global _thread, _stop_event
    if _thread and _thread.is_alive():
        logger.debug("Raw pipeline already running")
        return

    _stop_event.clear()
    _thread = threading.Thread(target=_thread_main, args=(batch_size,), daemon=True, name="RawPipeline")
    _thread.start()
    # Give thread a moment to spin up
    time.sleep(0.05)
    logger.info("Raw pipeline started", batch_size=batch_size, thread_alive=_thread.is_alive())


def stop_raw_pipeline(timeout: float = 2.0) -> None:
    global _thread, _stop_event
    if not _thread:
        return
    _stop_event.set()
    logger.info(
        "Raw pipeline stopping",
        total_batches_read=_total_batches_read,
        total_records_read=_total_records_read,
    )
    try:
        _thread.join(timeout=timeout)
    except Exception:
        pass
    _thread = None
    logger.info("Raw pipeline stopped")