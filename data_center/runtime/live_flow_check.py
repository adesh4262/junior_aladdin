"""
Live Flow Smoke Check
======================
Lightweight script to validate end-to-end data flow inside Data Center.

Behavior:
- Start raw and clean pipelines (best-effort)
- Push synthetic normalized ticks into `data_center.queues.tick_queue`
- Wait for pipelines to process
- Check for generated parquet files under major/raw and major/structured
- Trigger a review run and print summary
- Stop pipelines

Run:
    .venv\Scripts\python.exe data_center\runtime\live_flow_check.py

This is non-invasive and best-effort; failures are logged but do not affect
other runtime components.
"""

from __future__ import annotations

import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import random

from loguru import logger

from data_center.queues.tick_queue import tick_queue

try:
    from data_center.pipeline.raw_pipeline import start_raw_pipeline, stop_raw_pipeline
    from data_center.pipeline.clean_pipeline import start_clean_pipeline, stop_clean_pipeline
except Exception:
    start_raw_pipeline = stop_raw_pipeline = None
    start_clean_pipeline = stop_clean_pipeline = None

from data_center.validators.review_engine import review_engine
from configs.storage_config import MAJOR_RAW, MAJOR_STRUCTURED


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


def run_smoke(num_ticks: int = 50, tokens: list[str] | None = None, wait_sec: float = 2.0) -> dict:
    tokens = tokens or ["99926000", "51234", "12345678"]

    # Start pipelines if available
    if start_raw_pipeline:
        try:
            start_raw_pipeline()
        except Exception as e:
            logger.debug("start_raw_pipeline failed", error=str(e))

    if start_clean_pipeline:
        try:
            start_clean_pipeline()
        except Exception as e:
            logger.debug("start_clean_pipeline failed", error=str(e))

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for i in range(num_ticks):
        tok = random.choice(tokens)
        tick = make_tick(tok, now_ms + i)
        try:
            tick_queue.put_nowait(tick)
        except Exception:
            logger.warning("tick_queue put_nowait failed; queue may be full")

    # Wait for pipelines to pick up
    time.sleep(wait_sec)

    raw_files = find_parquet(MAJOR_RAW)
    structured_files = find_parquet(MAJOR_STRUCTURED)

    # Trigger a review run (best-effort)
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
        "raw_files_sample": [str(p) for p in raw_files[:5]],
        "structured_files_sample": [str(p) for p in structured_files[:5]],
        "review_report": report,
    }


if __name__ == "__main__":
    logger.info("Starting Data Center live-flow smoke check")
    out = run_smoke()
    logger.info("Smoke result", **{k: (v if not isinstance(v, list) else len(v)) for k, v in out.items() if k.endswith('_count')})
    print("Raw files sample:")
    for p in out.get("raw_files_sample", []):
        print(" -", p)
    print("Structured files sample:")
    for p in out.get("structured_files_sample", []):
        print(" -", p)
    if out.get("review_report"):
        print("Review verified:", out["review_report"].get("verified"))
    else:
        print("No review report")
