"""
Junior Aladdin — Cleaner Queue
==============================
Async queue for cleaned data between cleaner and structured writer pipeline.

Design:
  - asyncio.Queue based — non-blocking, bounded
  - Configurable maxsize from queue_config
  - Tracks queue health (size, fullness %)
  - Mirrors the tick queue contract for pipeline consistency

Data Center Architecture compliant.
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from configs.queue_config import (
    CLEANER_QUEUE_MAXSIZE,
    CLEANER_QUEUE_TIMEOUT,
    QUEUE_HIGH_WATERMARK,
    QUEUE_LOW_WATERMARK,
)


class CleanerQueue:
    """
    Bounded async queue for cleaned data.

    Producer: Cleaner / transformer stage → put()
    Consumer: StructuredWriter / downstream batch consumer → get()
    """

    def __init__(self, maxsize: int = CLEANER_QUEUE_MAXSIZE):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize

        # Stats
        self._total_put: int = 0
        self._total_get: int = 0
        self._total_overflow: int = 0

    async def put(self, item: Any, timeout: Optional[float] = CLEANER_QUEUE_TIMEOUT) -> bool:
        """Put an item into the queue with timeout-based backpressure."""
        try:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
            self._total_put += 1
            self._check_high_watermark()
            return True
        except asyncio.TimeoutError:
            self._total_overflow += 1
            return False

    def put_nowait(self, item: Any) -> bool:
        """Put an item without blocking."""
        try:
            self._queue.put_nowait(item)
            self._total_put += 1
            self._check_high_watermark()
            return True
        except asyncio.QueueFull:
            self._total_overflow += 1
            return False

    async def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Get an item from the queue; returns None on timeout."""
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._total_get += 1
            self._queue.task_done()
            self._check_low_watermark()
            return item
        except asyncio.TimeoutError:
            return None

    async def get_batch(self, max_batch: int = 500, timeout: float = 0.5) -> list[Any]:
        """Get a batch of items from the queue."""
        items: list[Any] = []

        first = await self.get(timeout=timeout)
        if first is None:
            return items

        items.append(first)

        while len(items) < max_batch:
            try:
                item = self._queue.get_nowait()
                items.append(item)
                self._total_get += 1
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        return items

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def fullness_pct(self) -> float:
        if self._maxsize <= 0:
            return 0.0
        return round((self._queue.qsize() / self._maxsize) * 100, 2)

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    def _check_high_watermark(self) -> None:
        if self.fullness_pct >= QUEUE_HIGH_WATERMARK:
            logger.warning(
                f"Cleaner queue at {self.fullness_pct}% capacity "
                f"({self.qsize}/{self._maxsize})"
            )

    def _check_low_watermark(self) -> None:
        if self.fullness_pct <= QUEUE_LOW_WATERMARK and self.fullness_pct > 0:
            logger.info(
                f"Cleaner queue drained to {self.fullness_pct}% "
                f"({self.qsize}/{self._maxsize})"
            )

    @property
    def stats(self) -> dict:
        return {
            "qsize": self.qsize,
            "maxsize": self._maxsize,
            "fullness_pct": self.fullness_pct,
            "total_put": self._total_put,
            "total_get": self._total_get,
            "total_overflow": self._total_overflow,
        }


cleaner_queue = CleanerQueue()