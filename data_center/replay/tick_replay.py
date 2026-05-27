"""
Junior Aladdin — Tick Replay Engine
===================================
Week 4 Phase 4: replay raw or structured tick parquet files in timestamp order.

The engine is intentionally simple and deterministic. It can:
  - load parquet files from a file or directory tree
  - sort records by timestamp
  - emit records to a callback
  - optionally sleep to approximate real-time playback
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import polars as pl
from loguru import logger

from data_center.utils.parquet_utils import read_parquet


@dataclass(slots=True)
class TickReplayEvent:
    timestamp: int
    record: dict[str, Any]


@dataclass(slots=True)
class TickReplayReport:
    source: str
    files_loaded: int = 0
    rows_loaded: int = 0
    rows_emitted: int = 0
    duration_seconds: float = 0.0
    replayed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class TickReplayEngine:
    """Replay tick parquet data in chronological order."""

    def __init__(self, source: Path | str, *, realtime: bool = False, speed: float = 1.0):
        self.source = Path(source)
        self.realtime = bool(realtime)
        self.speed = max(float(speed), 0.01)

    def load_events(self) -> list[TickReplayEvent]:
        frames = self._load_frames(self.source)
        events: list[TickReplayEvent] = []

        for frame in frames:
            if "timestamp" not in frame.columns:
                continue
            for row in frame.to_dicts():
                try:
                    ts = int(row.get("timestamp"))
                except (TypeError, ValueError):
                    continue
                events.append(TickReplayEvent(timestamp=ts, record=dict(row)))

        events.sort(key=lambda item: (item.timestamp, str(item.record.get("token", ""))))
        return events

    def replay(
        self,
        on_tick: Callable[[dict[str, Any]], None],
        *,
        realtime: Optional[bool] = None,
        speed: Optional[float] = None,
    ) -> dict[str, Any]:
        events = self.load_events()
        effective_realtime = self.realtime if realtime is None else bool(realtime)
        effective_speed = self.speed if speed is None else max(float(speed), 0.01)

        report = TickReplayReport(source=str(self.source), files_loaded=self._count_files(self.source), rows_loaded=len(events))
        start = time.perf_counter()

        previous_timestamp: Optional[int] = None
        for event in events:
            if effective_realtime and previous_timestamp is not None:
                delta_ms = max(0, event.timestamp - previous_timestamp)
                sleep_seconds = (delta_ms / 1000.0) / effective_speed
                if sleep_seconds > 0:
                    time.sleep(min(sleep_seconds, 0.5))

            on_tick(dict(event.record))
            report.rows_emitted += 1
            previous_timestamp = event.timestamp

        report.duration_seconds = round(time.perf_counter() - start, 6)
        logger.info(
            "Tick replay completed",
            source=str(self.source),
            files_loaded=report.files_loaded,
            rows_emitted=report.rows_emitted,
            realtime=effective_realtime,
            speed=effective_speed,
        )
        return self._serialize_report(report)

    def _load_frames(self, source: Path) -> list[pl.DataFrame]:
        if source.is_file():
            frame = read_parquet(source)
            return [frame] if frame is not None else []

        if not source.exists():
            return []

        frames: list[pl.DataFrame] = []
        for path in sorted(source.rglob("*.parquet")):
            frame = read_parquet(path)
            if frame is not None:
                frames.append(frame)
        return frames

    @staticmethod
    def _count_files(source: Path) -> int:
        if source.is_file():
            return 1
        if not source.exists():
            return 0
        return sum(1 for path in source.rglob("*.parquet") if path.is_file())

    @staticmethod
    def _serialize_report(report: TickReplayReport) -> dict[str, Any]:
        return {
            "source": report.source,
            "files_loaded": report.files_loaded,
            "rows_loaded": report.rows_loaded,
            "rows_emitted": report.rows_emitted,
            "duration_seconds": report.duration_seconds,
            "replayed_at": report.replayed_at,
        }


tick_replay_engine = TickReplayEngine


def _run_tests() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        source = tmp_root / "data_center" / "major" / "raw"
        source.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(
            [
                {"token": "99926000", "ltp": 18500.0, "volume": 100, "open": 18400.0, "high": 18520.0, "low": 18380.0, "close": 18450.0, "timestamp": 1716600002000, "direction": 1},
                {"token": "99926000", "ltp": 18501.0, "volume": 101, "open": 18400.0, "high": 18520.0, "low": 18380.0, "close": 18450.0, "timestamp": 1716600001000, "direction": 1},
            ]
        )
        parquet_path = source / "2026-05-25" / "NIFTY" / "10_11.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(parquet_path)

        emitted: list[dict[str, Any]] = []
        engine = TickReplayEngine(source)
        report = engine.replay(emitted.append, realtime=False)

        assert report["rows_loaded"] == 2
        assert report["rows_emitted"] == 2
        assert [item["timestamp"] for item in emitted] == [1716600001000, 1716600002000]


if __name__ == "__main__":
    _run_tests()