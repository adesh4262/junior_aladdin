"""
Junior Aladdin — Tick Queue
============================
Async queue for raw tick data between collector and writer pipeline.

The tick queue is the central buffer between:
  - TickReceiver (producer)
  - RawWriter (consumer)

Design:
  - asyncio.Queue based — non-blocking, bounded
  - Configurable maxsize from queue_config
  - Tracks queue health (size, fullness %)
  - Thread-safe for async producers/consumers

Data Center Architecture compliant.
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from configs.queue_config import (
    TICK_QUEUE_MAXSIZE,
    TICK_QUEUE_TIMEOUT,
    QUEUE_HIGH_WATERMARK,
    QUEUE_LOW_WATERMARK,
)


class TickQueue:
    """
    Bounded async queue for tick data.

    Producer: TickReceiver.on_tick() → put()
    Consumer: RawWriter / batch consumer → get()
    """

    def __init__(self, maxsize: int = TICK_QUEUE_MAXSIZE):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize

        # Stats
        self._total_put: int = 0
        self._total_get: int = 0
        self._total_overflow: int = 0

    # ──────────────────────────────────────────
    # PRODUCER API
    # ──────────────────────────────────────────

    async def put(self, item: Any, timeout: Optional[float] = TICK_QUEUE_TIMEOUT) -> bool:
        """
        Put an item into the queue. Non-blocking with timeout.

        Args:
            item: Tick dict to enqueue.
            timeout: Max seconds to wait if queue is full.

        Returns:
            True if enqueued, False if timed out (queue full).
        """
        try:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
            self._total_put += 1
            self._check_high_watermark()
            return True
        except asyncio.TimeoutError:
            self._total_overflow += 1
            return False

    def put_nowait(self, item: Any) -> bool:
        """
        Put an item without blocking.

        Returns:
            True if enqueued, False if queue is full.
        """
        try:
            self._queue.put_nowait(item)
            self._total_put += 1
            self._check_high_watermark()
            return True
        except asyncio.QueueFull:
            self._total_overflow += 1
            return False

    # ──────────────────────────────────────────
    # CONSUMER API
    # ──────────────────────────────────────────

    async def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get an item from the queue. Blocks until available.

        Args:
            timeout: Max seconds to wait. None = wait forever.

        Returns:
            Item if available, None if timeout.
        """
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._total_get += 1
            self._queue.task_done()
            self._check_low_watermark()
            return item
        except asyncio.TimeoutError:
            return None

    # ──────────────────────────────────────────
    # BATCH CONSUMER
    # ──────────────────────────────────────────

    async def get_batch(
        self, max_batch: int = 1000, timeout: float = 0.5
    ) -> list[Any]:
        """
        Get a batch of items from the queue.

        Waits up to `timeout` seconds for first item,
        then collects up to `max_batch` items.

        Returns:
            List of items (may be empty).
        """
        items: list[Any] = []

        # Wait for first item
        first = await self.get(timeout=timeout)
        if first is None:
            return items

        items.append(first)

        # Collect remaining batch (non-blocking)
        while len(items) < max_batch:
            try:
                item = self._queue.get_nowait()
                items.append(item)
                self._total_get += 1
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        return items

    # ──────────────────────────────────────────
    # QUEUE STATE
    # ──────────────────────────────────────────

    @property
    def qsize(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def fullness_pct(self) -> float:
        """Queue fullness as percentage (0-100)."""
        if self._maxsize <= 0:
            return 0.0
        return round((self._queue.qsize() / self._maxsize) * 100, 2)

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    # ──────────────────────────────────────────
    # HEALTH
    # ──────────────────────────────────────────

    def _check_high_watermark(self) -> None:
        """Log warning if queue exceeds high watermark."""
        if self.fullness_pct >= QUEUE_HIGH_WATERMARK:
            logger.warning(
                f"Tick queue at {self.fullness_pct}% capacity "
                f"({self.qsize}/{self._maxsize})"
            )

    def _check_low_watermark(self) -> None:
        """Log info when queue drops below low watermark."""
        if self.fullness_pct <= QUEUE_LOW_WATERMARK and self.fullness_pct > 0:
            logger.info(
                f"Tick queue drained to {self.fullness_pct}% "
                f"({self.qsize}/{self._maxsize})"
            )

    @property
    def stats(self) -> dict:
        """Return queue statistics for monitoring."""
        return {
            "qsize": self.qsize,
            "maxsize": self._maxsize,
            "fullness_pct": self.fullness_pct,
            "total_put": self._total_put,
            "total_get": self._total_get,
            "total_overflow": self._total_overflow,
        }


# ──────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ──────────────────────────────────────────
# Single tick queue instance shared across the system.
tick_queue = TickQueue()