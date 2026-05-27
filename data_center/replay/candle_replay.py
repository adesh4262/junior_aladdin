"""
Junior Aladdin — Candle Replay Engine
=====================================
Week 4 Phase 4: replay candle parquet files in timestamp order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
from loguru import logger

from data_center.utils.parquet_utils import read_parquet


@dataclass(slots=True)
class CandleReplayReport:
    source: str
    files_loaded: int = 0
    rows_loaded: int = 0
    rows_emitted: int = 0
    duration_seconds: float = 0.0
    replayed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class CandleReplayEngine:
    """Replay candle parquet data in chronological order."""

    def __init__(self, source: Path | str, *, realtime: bool = False, speed: float = 1.0):
        self.source = Path(source)
        self.realtime = bool(realtime)
        self.speed = max(float(speed), 0.01)

    def replay(
        self,
        on_candle: Callable[[dict[str, Any]], None],
        *,
        realtime: Optional[bool] = None,
        speed: Optional[float] = None,
    ) -> dict[str, Any]:
        frames = self._load_frames(self.source)
        rows: list[dict[str, Any]] = []
        for frame in frames:
            if "timestamp" not in frame.columns:
                continue
            rows.extend(frame.to_dicts())

        rows.sort(key=lambda item: (int(item.get("timestamp", 0) or 0), str(item.get("token", ""))))

        effective_realtime = self.realtime if realtime is None else bool(realtime)
        effective_speed = self.speed if speed is None else max(float(speed), 0.01)

        report = CandleReplayReport(source=str(self.source), files_loaded=self._count_files(self.source), rows_loaded=len(rows))
        start = time.perf_counter()

        previous_timestamp: Optional[int] = None
        for row in rows:
            try:
                ts = int(row.get("timestamp"))
            except (TypeError, ValueError):
                continue

            if effective_realtime and previous_timestamp is not None:
                delta_ms = max(0, ts - previous_timestamp)
                sleep_seconds = (delta_ms / 1000.0) / effective_speed
                if sleep_seconds > 0:
                    time.sleep(min(sleep_seconds, 0.5))

            on_candle(dict(row))
            report.rows_emitted += 1
            previous_timestamp = ts

        report.duration_seconds = round(time.perf_counter() - start, 6)
        logger.info(
            "Candle replay completed",
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
    def _serialize_report(report: CandleReplayReport) -> dict[str, Any]:
        return {
            "source": report.source,
            "files_loaded": report.files_loaded,
            "rows_loaded": report.rows_loaded,
            "rows_emitted": report.rows_emitted,
            "duration_seconds": report.duration_seconds,
            "replayed_at": report.replayed_at,
        }


candle_replay_engine = CandleReplayEngine


def _run_tests() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        source = tmp_root / "data_center" / "historical" / "candles"
        source.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(
            [
                {"token": "NIFTY", "timestamp": 1716600002000, "open": 18400.0, "high": 18520.0, "low": 18380.0, "close": 18450.0, "volume": 1000},
                {"token": "NIFTY", "timestamp": 1716600001000, "open": 18390.0, "high": 18510.0, "low": 18370.0, "close": 18440.0, "volume": 900},
            ]
        )
        parquet_path = source / "NIFTY_1m_2026-05-25.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(parquet_path)

        emitted: list[dict[str, Any]] = []
        engine = CandleReplayEngine(source)
        report = engine.replay(emitted.append, realtime=False)

        assert report["rows_loaded"] == 2
        assert report["rows_emitted"] == 2
        assert [item["timestamp"] for item in emitted] == [1716600001000, 1716600002000]


if __name__ == "__main__":
    _run_tests()