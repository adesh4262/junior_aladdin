"""
Junior Aladdin — Queue Monitor
===============================
Week 4 Phase 6: monitor the health of queue-based ingestion buffers.

This module reads queue stats without mutating queue contents. It works with
the built-in tick and cleaner queues as well as compatible custom queue
implementations that expose stats/qsize/maxsize/fullness_pct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from loguru import logger


@dataclass(slots=True)
class QueueSnapshot:
    name: str
    qsize: int = 0
    maxsize: int = 0
    fullness_pct: float = 0.0
    total_put: int = 0
    total_get: int = 0
    total_overflow: int = 0
    health: str = "UNKNOWN"
    alert_level: str = "INFO"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QueueMonitor:
    """Create queue health snapshots and aggregate alerts."""

    def __init__(self, *queues: Any, queue_names: Iterable[str] | None = None):
        self._queues = list(queues)
        names = list(queue_names or [])
        self._names = names + [f"queue_{index + 1}" for index in range(max(0, len(self._queues) - len(names)))]

    def snapshot(self) -> list[dict[str, Any]]:
        snapshots = [self._snapshot_queue(name, queue) for name, queue in zip(self._names, self._queues)]
        logger.info("Queue monitor snapshot created", queue_count=len(snapshots))
        return [self._serialize_snapshot(item) for item in snapshots]

    def overall_health(self) -> str:
        snapshots = [self._snapshot_queue(name, queue) for name, queue in zip(self._names, self._queues)]
        if not snapshots:
            return "UNKNOWN"
        if any(item.health == "CRITICAL" for item in snapshots):
            return "CRITICAL"
        if any(item.health == "WARNING" for item in snapshots):
            return "WARNING"
        if all(item.health == "HEALTHY" for item in snapshots):
            return "HEALTHY"
        return "DEGRADED"

    def _snapshot_queue(self, name: str, queue: Any) -> QueueSnapshot:
        stats = self._safe_stats(queue)
        qsize = int(stats.get("qsize", getattr(queue, "qsize", 0) if not callable(getattr(queue, "qsize", None)) else queue.qsize()))
        maxsize = int(stats.get("maxsize", getattr(queue, "maxsize", 0) if not callable(getattr(queue, "maxsize", None)) else queue.maxsize()))
        fullness_pct = float(stats.get("fullness_pct", getattr(queue, "fullness_pct", 0.0) if not callable(getattr(queue, "fullness_pct", None)) else queue.fullness_pct()))
        total_put = int(stats.get("total_put", 0))
        total_get = int(stats.get("total_get", 0))
        total_overflow = int(stats.get("total_overflow", 0))

        health, alert = self._classify(fullness_pct, total_overflow)
        return QueueSnapshot(
            name=name,
            qsize=qsize,
            maxsize=maxsize,
            fullness_pct=round(fullness_pct, 2),
            total_put=total_put,
            total_get=total_get,
            total_overflow=total_overflow,
            health=health,
            alert_level=alert,
        )

    @staticmethod
    def _safe_stats(queue: Any) -> dict[str, Any]:
        stats = getattr(queue, "stats", None)
        if callable(stats):
            try:
                result = stats()
                return result if isinstance(result, dict) else {}
            except Exception:
                return {}
        if isinstance(stats, dict):
            return stats
        return {}

    @staticmethod
    def _classify(fullness_pct: float, total_overflow: int) -> tuple[str, str]:
        if total_overflow > 0 or fullness_pct >= 95.0:
            return "CRITICAL", "ERROR"
        if fullness_pct >= 80.0:
            return "WARNING", "WARN"
        if fullness_pct >= 40.0:
            return "DEGRADED", "INFO"
        return "HEALTHY", "INFO"

    @staticmethod
    def _serialize_snapshot(snapshot: QueueSnapshot) -> dict[str, Any]:
        return {
            "name": snapshot.name,
            "qsize": snapshot.qsize,
            "maxsize": snapshot.maxsize,
            "fullness_pct": snapshot.fullness_pct,
            "total_put": snapshot.total_put,
            "total_get": snapshot.total_get,
            "total_overflow": snapshot.total_overflow,
            "health": snapshot.health,
            "alert_level": snapshot.alert_level,
            "timestamp": snapshot.timestamp,
        }


queue_monitor = QueueMonitor()
