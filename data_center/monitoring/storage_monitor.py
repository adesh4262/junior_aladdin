"""
Junior Aladdin — Storage Monitor
================================
Week 4 Phase 6: monitor file counts, byte usage, and storage freshness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from data_center.utils.file_utils import get_disk_usage


@dataclass(slots=True)
class StorageSnapshot:
    name: str
    path: str
    exists: bool = False
    total_bytes: int = 0
    file_count: int = 0
    dir_count: int = 0
    newest_mtime: Optional[float] = None
    lag_seconds: Optional[float] = None
    status: str = "UNKNOWN"
    alert_level: str = "INFO"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StorageMonitor:
    """Inspect storage roots for size, freshness, and basic health."""

    def __init__(self, *paths: Path | str, names: Iterable[str] | None = None):
        self._paths = [Path(path) for path in paths]
        provided_names = list(names or [])
        self._names = provided_names + [f"storage_{index + 1}" for index in range(max(0, len(self._paths) - len(provided_names)))]

    def snapshot(self) -> list[dict[str, Any]]:
        snapshots = [self._snapshot_path(name, path) for name, path in zip(self._names, self._paths)]
        logger.info("Storage monitor snapshot created", storage_count=len(snapshots))
        return [self._serialize_snapshot(item) for item in snapshots]

    def overall_health(self) -> str:
        snapshots = [self._snapshot_path(name, path) for name, path in zip(self._names, self._paths)]
        if not snapshots:
            return "UNKNOWN"
        if any(item.status == "DOWN" for item in snapshots):
            return "DOWN"
        if any(item.status == "STALE" for item in snapshots):
            return "STALE"
        if any(item.status == "DEGRADED" for item in snapshots):
            return "DEGRADED"
        return "HEALTHY"

    def _snapshot_path(self, name: str, path: Path) -> StorageSnapshot:
        usage = get_disk_usage(path)
        newest_mtime = self._newest_mtime(path)
        lag_seconds = None
        if newest_mtime is not None:
            lag_seconds = max(0.0, datetime.now().timestamp() - newest_mtime)

        status, alert = self._classify(usage.get("exists", False), usage.get("file_count", 0), lag_seconds)
        return StorageSnapshot(
            name=name,
            path=str(path),
            exists=bool(usage.get("exists", False)),
            total_bytes=int(usage.get("total_bytes", 0) or 0),
            file_count=int(usage.get("file_count", 0) or 0),
            dir_count=int(usage.get("dir_count", 0) or 0),
            newest_mtime=newest_mtime,
            lag_seconds=lag_seconds,
            status=status,
            alert_level=alert,
        )

    @staticmethod
    def _newest_mtime(path: Path) -> Optional[float]:
        if not path.exists():
            return None
        mtimes: list[float] = []
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    mtimes.append(item.stat().st_mtime)
                except OSError:
                    continue
        return max(mtimes) if mtimes else None

    @staticmethod
    def _classify(exists: bool, file_count: int, lag_seconds: Optional[float]) -> tuple[str, str]:
        if not exists:
            return "DOWN", "ERROR"
        if file_count == 0:
            return "STALE", "WARN"
        if lag_seconds is not None and lag_seconds > 3600:
            return "STALE", "WARN"
        if lag_seconds is not None and lag_seconds > 300:
            return "DEGRADED", "INFO"
        return "HEALTHY", "INFO"

    @staticmethod
    def _serialize_snapshot(snapshot: StorageSnapshot) -> dict[str, Any]:
        return {
            "name": snapshot.name,
            "path": snapshot.path,
            "exists": snapshot.exists,
            "total_bytes": snapshot.total_bytes,
            "file_count": snapshot.file_count,
            "dir_count": snapshot.dir_count,
            "newest_mtime": snapshot.newest_mtime,
            "lag_seconds": snapshot.lag_seconds,
            "status": snapshot.status,
            "alert_level": snapshot.alert_level,
            "timestamp": snapshot.timestamp,
        }


storage_monitor = StorageMonitor
