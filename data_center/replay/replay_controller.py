"""
Junior Aladdin — Replay Controller
==================================
Week 4 Phase 5: orchestrate tick and candle replay with playback speed,
synchronization, and pause/resume control.

The controller merges replay events from tick and candle parquet sources in
timestamp order. It does not mutate the underlying replay engines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from data_center.replay.candle_replay import CandleReplayEngine
from data_center.replay.tick_replay import TickReplayEngine


@dataclass(slots=True)
class ReplayControllerReport:
    source: str
    merged_events: int = 0
    tick_events: int = 0
    candle_events: int = 0
    pauses_observed: int = 0
    duration_seconds: float = 0.0
    replayed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ReplayController:
    """Coordinate tick/candle replay and expose pause/resume/speed controls."""

    def __init__(
        self,
        tick_source: Path | str | None = None,
        candle_source: Path | str | None = None,
        *,
        speed: float = 1.0,
    ):
        self.tick_source = Path(tick_source) if tick_source is not None else None
        self.candle_source = Path(candle_source) if candle_source is not None else None
        self.speed = max(float(speed), 0.01)

        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pause_count = 0

    def pause(self) -> None:
        with self._lock:
            if self._pause_event.is_set():
                self._pause_count += 1
            self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()

    def set_speed(self, speed: float) -> None:
        self.speed = max(float(speed), 0.01)

    def replay(
        self,
        *,
        on_tick: Optional[Callable[[dict[str, Any]], None]] = None,
        on_candle: Optional[Callable[[dict[str, Any]], None]] = None,
        realtime: bool = False,
        speed: Optional[float] = None,
    ) -> dict[str, Any]:
        if on_tick is None and on_candle is None:
            raise ValueError("At least one callback must be provided")

        effective_speed = self.speed if speed is None else max(float(speed), 0.01)
        events = self._build_merged_events()

        report = ReplayControllerReport(
            source=self._source_label(),
            merged_events=len(events),
        )

        start = time.perf_counter()
        previous_timestamp: Optional[int] = None

        for event in events:
            if self._stop_event.is_set():
                break

            self._wait_if_paused()
            if self._stop_event.is_set():
                break

            if realtime and previous_timestamp is not None:
                delta_ms = max(0, event["timestamp"] - previous_timestamp)
                sleep_seconds = (delta_ms / 1000.0) / effective_speed
                self._sleep_with_control(sleep_seconds)
                if self._stop_event.is_set():
                    break

            if event["kind"] == "tick" and on_tick is not None:
                on_tick(dict(event["record"]))
                report.tick_events += 1
            elif event["kind"] == "candle" and on_candle is not None:
                on_candle(dict(event["record"]))
                report.candle_events += 1

            previous_timestamp = event["timestamp"]

        report.pauses_observed = self._pause_count
        report.duration_seconds = round(time.perf_counter() - start, 6)
        logger.info(
            "Replay controller completed",
            source=report.source,
            merged_events=report.merged_events,
            tick_events=report.tick_events,
            candle_events=report.candle_events,
            pauses_observed=report.pauses_observed,
            speed=effective_speed,
            realtime=realtime,
        )
        return self._serialize_report(report)

    def _build_merged_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if self.tick_source is not None:
            for event in TickReplayEngine(self.tick_source).load_events():
                events.append({"kind": "tick", "timestamp": event.timestamp, "record": event.record})

        if self.candle_source is not None:
            for row in CandleReplayEngine(self.candle_source)._load_frames(self.candle_source):
                if "timestamp" not in row.columns:
                    continue
                for item in row.to_dicts():
                    try:
                        ts = int(item.get("timestamp"))
                    except (TypeError, ValueError):
                        continue
                    events.append({"kind": "candle", "timestamp": ts, "record": dict(item)})

        events.sort(key=lambda item: (item["timestamp"], 0 if item["kind"] == "tick" else 1, str(item["record"].get("token", ""))))
        return events

    def _wait_if_paused(self) -> None:
        while not self._pause_event.is_set():
            if self._stop_event.wait(timeout=0.05):
                return

    def _sleep_with_control(self, seconds: float) -> None:
        remaining = max(0.0, float(seconds))
        while remaining > 0:
            if self._stop_event.is_set():
                return
            if not self._pause_event.is_set():
                self._wait_if_paused()
                continue
            slice_seconds = min(remaining, 0.05)
            if self._stop_event.wait(timeout=slice_seconds):
                return
            remaining -= slice_seconds

    def _source_label(self) -> str:
        parts = []
        if self.tick_source is not None:
            parts.append(f"tick={self.tick_source}")
        if self.candle_source is not None:
            parts.append(f"candle={self.candle_source}")
        return ";".join(parts) if parts else "memory"

    @staticmethod
    def _serialize_report(report: ReplayControllerReport) -> dict[str, Any]:
        return {
            "source": report.source,
            "merged_events": report.merged_events,
            "tick_events": report.tick_events,
            "candle_events": report.candle_events,
            "pauses_observed": report.pauses_observed,
            "duration_seconds": report.duration_seconds,
            "replayed_at": report.replayed_at,
        }


replay_controller = ReplayController


def _run_tests() -> None:
    import tempfile

    import polars as pl

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        tick_source = tmp_root / "data_center" / "major" / "raw"
        candle_source = tmp_root / "data_center" / "historical" / "candles"
        tick_source.mkdir(parents=True, exist_ok=True)
        candle_source.mkdir(parents=True, exist_ok=True)

        tick_frame = pl.DataFrame(
            [
                {"token": "99926000", "ltp": 18500.0, "volume": 100, "open": 18400.0, "high": 18520.0, "low": 18380.0, "close": 18450.0, "timestamp": 1716600002000, "direction": 1},
                {"token": "99926000", "ltp": 18501.0, "volume": 101, "open": 18400.0, "high": 18520.0, "low": 18380.0, "close": 18450.0, "timestamp": 1716600001000, "direction": 1},
            ]
        )
        candle_frame = pl.DataFrame(
            [
                {"token": "NIFTY", "timestamp": 1716600001500, "open": 18400.0, "high": 18520.0, "low": 18380.0, "close": 18450.0, "volume": 1000},
            ]
        )

        tick_path = tick_source / "2026-05-25" / "NIFTY" / "10_11.parquet"
        candle_path = candle_source / "NIFTY_1m_2026-05-25.parquet"
        tick_path.parent.mkdir(parents=True, exist_ok=True)
        candle_path.parent.mkdir(parents=True, exist_ok=True)
        tick_frame.write_parquet(tick_path)
        candle_frame.write_parquet(candle_path)

        seen: list[tuple[str, int]] = []
        controller = ReplayController(tick_source=tick_source, candle_source=candle_source, speed=1.0)

        def on_tick(record: dict[str, Any]) -> None:
            seen.append(("tick", int(record["timestamp"])))
            if len(seen) == 1:
                controller.pause()

        def on_candle(record: dict[str, Any]) -> None:
            seen.append(("candle", int(record["timestamp"])))

        def _resume_later() -> None:
            time.sleep(0.1)
            controller.resume()

        thread = threading.Thread(target=_resume_later, daemon=True)
        thread.start()

        report = controller.replay(on_tick=on_tick, on_candle=on_candle, realtime=False)
        assert report["merged_events"] == 3
        assert report["tick_events"] == 2
        assert report["candle_events"] == 1
        assert report["pauses_observed"] >= 1
        assert seen[0][1] == 1716600001000
        assert seen[1][0] == "candle"


if __name__ == "__main__":
    _run_tests()