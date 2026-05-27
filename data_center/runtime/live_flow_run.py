"""
Run a 2-minute live-flow data ingestion test.

This script:
 - Starts raw and clean pipelines
 - Emits synthetic ticks at configurable rate for a configured duration (default 120s)
 - Waits for processing, stops pipelines
 - Runs review and prints summary of stored files

Run as module:
    .venv\Scripts\python.exe -m data_center.runtime.live_flow_run

Be cautious: this will run for ~duration seconds.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
import random

from loguru import logger

from data_center.queues.tick_queue import tick_queue
from data_center.validators.review_engine import review_engine
from configs.storage_config import MAJOR_RAW, MAJOR_STRUCTURED

try:
    from data_center.pipeline.raw_pipeline import start_raw_pipeline, stop_raw_pipeline
    from data_center.pipeline.clean_pipeline import start_clean_pipeline, stop_clean_pipeline
except Exception:
    start_raw_pipeline = stop_raw_pipeline = None
    start_clean_pipeline = stop_clean_pipeline = None


def make_tick(token: str, ts_ms: int) -> dict:
    return {
        "token": token,
        "ltp": round(1000.0 + random.random() * 10.0, 2),
        "volume": random.randint(1, 1000),
        "open": 1000.0,
        "high": 1015.0,
        "low": 995.0,
        "close": 1005.0,
        "timestamp": ts_ms,
        "direction": 0,
    }


def find_parquet(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return list(root.rglob("*.parquet"))


def producer(stop_event: threading.Event, duration: int, rate: float, tokens: list[str]):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval = 1.0 / float(max(0.0001, rate))
    sent = 0
    start = time.time()
    while not stop_event.is_set() and (time.time() - start) < duration:
        tok = random.choice(tokens)
        tick = make_tick(tok, now_ms + sent)
        try:
            tick_queue.put_nowait(tick)
            sent += 1
        except Exception:
            logger.debug("tick_queue put_nowait failed; dropping tick")
        time.sleep(interval)
    logger.info("Producer exiting; sent_count=%d", sent)


def run(duration: int = 120, rate: float = 5.0, tokens: list[str] | None = None):
    tokens = tokens or ["99926000", "51234", "12345678"]

    # Start pipelines
    if start_raw_pipeline:
        try:
            start_raw_pipeline()
        except Exception as e:
            logger.warning("start_raw_pipeline failed", error=str(e))

    if start_clean_pipeline:
        try:
            start_clean_pipeline()
        except Exception as e:
            logger.warning("start_clean_pipeline failed", error=str(e))

    stop_event = threading.Event()
    prod = threading.Thread(target=producer, args=(stop_event, duration, rate, tokens), daemon=True)
    prod.start()

    try:
        prod.join(timeout=duration + 10)
    except KeyboardInterrupt:
        logger.info("Interrupted; stopping producer")
        stop_event.set()
        prod.join(timeout=5)

    # Wait briefly for pipelines to drain
    time.sleep(2.0)

    raw_files = find_parquet(MAJOR_RAW)
    structured_files = find_parquet(MAJOR_STRUCTURED)

    # Trigger review
    try:
        report = review_engine.review()
    except Exception as e:
        logger.warning("Review run failed", error=str(e))
        report = None

    # Stop pipelines
    if stop_clean_pipeline:
        try:
            stop_clean_pipeline()
        except Exception:
            pass
    if stop_raw_pipeline:
        try:
            stop_raw_pipeline()
        except Exception:
            pass

    return {
        "raw_count": len(raw_files),
        "structured_count": len(structured_files),
        "raw_files_sample": [str(p) for p in raw_files[-5:]],
        "structured_files_sample": [str(p) for p in structured_files[-5:]],
        "review_report": report,
    }


if __name__ == "__main__":
    # Default: 120s, 5 ticks/sec => 600 ticks
    DURATION = 120
    RATE = 5.0
    logger.info("Starting 2-minute data flow run: duration=%s rate=%s tps", DURATION, RATE)
    res = run(duration=DURATION, rate=RATE)
    logger.info("Run complete. raw_files=%d structured_files=%d", res["raw_count"], res["structured_count"])
    print("Raw sample files:")
    for p in res.get("raw_files_sample", []):
        print(" -", p)
    print("Structured sample files:")
    for p in res.get("structured_files_sample", []):
        print(" -", p)
    if res.get("review_report"):
        print("Review verified:", res["review_report"].get("verified"))
    else:
        print("Review not available")
