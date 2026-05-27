"""
Junior Aladdin — WebSocket Monitor
==================================
Week 4 Phase 6: monitor websocket health, feed freshness, and connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


@dataclass(slots=True)
class WebSocketSnapshot:
    connected: bool = False
    feed_health: str = "UNKNOWN"
    tick_gap_sec: Optional[float] = None
    total_ticks_received: int = 0
    dropped_ticks: int = 0
    status: str = "UNKNOWN"
    alert_level: str = "INFO"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WebSocketMonitor:
    """Create snapshots from websocket clients or websocket managers."""

    def __init__(self, websocket: Any):
        self.websocket = websocket

    def snapshot(self) -> dict[str, Any]:
        status = self._collect_status()
        logger.info("WebSocket monitor snapshot created", status=status.status, connected=status.connected)
        return self._serialize_snapshot(status)

    def health(self) -> str:
        status = self._collect_status()
        return status.status

    def _collect_status(self) -> WebSocketSnapshot:
        ws = self.websocket
        raw = self._safe_status(ws)

        connected = bool(raw.get("is_connected", getattr(ws, "is_connected", False)))
        feed_health = str(raw.get("feed_health", getattr(ws, "feed_health", "UNKNOWN")))
        tick_gap_sec = raw.get("tick_gap_sec", getattr(ws, "tick_gap_sec", None))
        total_ticks_received = int(raw.get("total_ticks_received", getattr(ws, "total_ticks_received", 0) or 0))
        dropped_ticks = int(raw.get("dropped_ticks", getattr(ws, "dropped_ticks", 0) or 0))

        status, alert = self._classify(connected, feed_health, tick_gap_sec, dropped_ticks)
        return WebSocketSnapshot(
            connected=connected,
            feed_health=feed_health,
            tick_gap_sec=float(tick_gap_sec) if tick_gap_sec is not None else None,
            total_ticks_received=total_ticks_received,
            dropped_ticks=dropped_ticks,
            status=status,
            alert_level=alert,
        )

    @staticmethod
    def _safe_status(websocket: Any) -> dict[str, Any]:
        getter = getattr(websocket, "get_status", None)
        if callable(getter):
            try:
                result = getter()
                return result if isinstance(result, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _classify(connected: bool, feed_health: str, tick_gap_sec: Any, dropped_ticks: int) -> tuple[str, str]:
        health = str(feed_health).upper()
        gap = None
        try:
            gap = float(tick_gap_sec) if tick_gap_sec is not None else None
        except (TypeError, ValueError):
            gap = None

        if not connected:
            return "DOWN", "ERROR"
        if health in {"DOWN", "STALE"}:
            return health, "ERROR"
        if gap is not None and gap > 5.0:
            return "STALE", "WARN"
        if dropped_ticks > 0:
            return "DEGRADED", "WARN"
        if health in {"DELAYED"} or (gap is not None and gap > 1.0):
            return "DELAYED", "INFO"
        return "HEALTHY", "INFO"

    @staticmethod
    def _serialize_snapshot(snapshot: WebSocketSnapshot) -> dict[str, Any]:
        return {
            "connected": snapshot.connected,
            "feed_health": snapshot.feed_health,
            "tick_gap_sec": snapshot.tick_gap_sec,
            "total_ticks_received": snapshot.total_ticks_received,
            "dropped_ticks": snapshot.dropped_ticks,
            "status": snapshot.status,
            "alert_level": snapshot.alert_level,
            "timestamp": snapshot.timestamp,
        }


websocket_monitor = WebSocketMonitor
