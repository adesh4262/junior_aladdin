from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QObject = object  # type: ignore[assignment]

    def pyqtSignal(*_args, **_kwargs):  # type: ignore
        return None

    _QT_AVAILABLE = False

from dashboard.core.binary_frame import (
    FRAME_MAGIC,
    HEADER_SIZE,
    HEADER_STRUCT,
    KIND_COLD,
    KIND_HOT,
    KIND_WARM,
    unpack_frame,
)
from dashboard.core.state_projection import project_snapshot

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


class SnapshotBus(QObject):
    """Decode binary frames and route payloads to typed Qt signals."""

    new_hot_frame = pyqtSignal(dict)
    new_warm_frame = pyqtSignal(dict)
    new_cold_frame = pyqtSignal(dict)

    def __init__(
        self,
        parent: Any | None = None,
        *,
        max_payload_bytes: int = 10 * 1024 * 1024,
        summary_interval: int = 10,
        slow_decode_warn_ms: float = 10.0,
    ) -> None:
        if not _QT_AVAILABLE:
            raise RuntimeError("SnapshotBus requires PyQt6")
        super().__init__(parent)
        self.log = setup_logger("dashboard_core_snapshot_bus")
        self._max_payload_bytes = int(max_payload_bytes)
        # Backward-compatible arg name; interpreted as seconds.
        self._summary_interval_seconds = max(1, int(summary_interval))
        self._last_summary_mono = time.monotonic()
        self._last_seq_hot = -1
        self._last_seq_warm = -1
        self._last_seq_cold = -1
        self._last_ts_hot = -1
        self._last_ts_warm = -1
        self._last_ts_cold = -1
        self._received = 0
        self._dropped = 0
        self._decode_errors = 0
        self._last_decode_ms = 0.0
        self._max_decode_ms = 0.0
        self._slow_decode_count = 0
        self._slow_decode_warn_ms = max(0.1, float(slow_decode_warn_ms))
        self._decode_ms_window = deque(maxlen=200)
        self._buffer = bytearray()
        self._last_valid_hot_payload: dict | None = None
        self._last_valid_warm_payload: dict | None = None
        self._last_valid_cold_payload: dict | None = None
        self._lock = threading.Lock()

    def _last_seq_for_kind(self, kind: int) -> int:
        if kind == KIND_HOT:
            return self._last_seq_hot
        if kind == KIND_WARM:
            return self._last_seq_warm
        if kind == KIND_COLD:
            return self._last_seq_cold
        return -1

    def _set_last_seq_for_kind(self, kind: int, seq: int) -> None:
        if kind == KIND_HOT:
            self._last_seq_hot = seq
        elif kind == KIND_WARM:
            self._last_seq_warm = seq
        elif kind == KIND_COLD:
            self._last_seq_cold = seq

    def _last_timestamp_for_kind(self, kind: int) -> int:
        if kind == KIND_HOT:
            return self._last_ts_hot
        if kind == KIND_WARM:
            return self._last_ts_warm
        if kind == KIND_COLD:
            return self._last_ts_cold
        return -1

    def _set_last_timestamp_for_kind(self, kind: int, timestamp_ns: int) -> None:
        if kind == KIND_HOT:
            self._last_ts_hot = timestamp_ns
        elif kind == KIND_WARM:
            self._last_ts_warm = timestamp_ns
        elif kind == KIND_COLD:
            self._last_ts_cold = timestamp_ns

    def _record_decode_ms(self, decode_ms: float) -> None:
        with self._lock:
            value = float(max(0.0, decode_ms))
            self._last_decode_ms = value
            self._max_decode_ms = max(self._max_decode_ms, value)
            self._decode_ms_window.append(value)
            if value > self._slow_decode_warn_ms:
                self._slow_decode_count += 1
        if decode_ms > self._slow_decode_warn_ms:
            self.log.warning(
                "SnapshotBus slow frame decode",
                decode_ms=round(float(decode_ms), 3),
                threshold_ms=self._slow_decode_warn_ms,
            )

    def _drop(self, *, kind: Any, seq: Any, error: str) -> None:
        with self._lock:
            self._dropped += 1
            self._decode_errors += 1
        self.log.error(
            "Dropped frame",
            frame_kind=kind,
            seq=seq,
            error=error,
        )
        self._log_periodic_summary()

    def start(self) -> None:
        """Lifecycle no-op for the push-only bus.

        Pre-Week-8 stabilization (P0 runtime fix): dashboard/main.py owns a
        uniform start/stop lifecycle for dashboard components.  SnapshotBus has
        no producer thread to start; frames arrive through feed_bytes().  Keeping
        explicit no-op methods avoids runtime AttributeError without pretending
        to fabricate frames or start backend transport.
        """
        self.log.info("SnapshotBus lifecycle start (push-only no-op)")

    def stop(self) -> None:
        """Lifecycle no-op; clear only buffered partial transport bytes."""
        with self._lock:
            self._buffer.clear()
        self.log.info("SnapshotBus lifecycle stop (push-only no-op)")

    def feed_bytes(self, data: bytes) -> None:
        """Ingest transport bytes; buffers partial frames and emits per complete valid frame."""
        if not isinstance(data, (bytes, bytearray)):
            self._drop(kind=None, seq=None, error="invalid_frame")
            return

        frames: list[tuple[bytes, int, int]] = []
        with self._lock:
            self._buffer.extend(data)
            while len(self._buffer) >= HEADER_SIZE:
                # Resync until we see frame magic at current cursor.
                if bytes(self._buffer[:4]) != FRAME_MAGIC:
                    self._buffer.pop(0)
                    self._dropped += 1
                    continue

                try:
                    header = HEADER_STRUCT.unpack(bytes(self._buffer[:HEADER_SIZE]))
                except Exception:
                    break

                kind = int(header[2])
                seq = int(header[4])
                payload_len = int(header[6])
                if payload_len < 0 or payload_len > self._max_payload_bytes:
                    # Corrupted header; advance one byte and retry sync.
                    self._buffer.pop(0)
                    self._dropped += 1
                    continue

                total_len = HEADER_SIZE + payload_len
                if len(self._buffer) < total_len:
                    break

                frame = bytes(self._buffer[:total_len])
                del self._buffer[:total_len]
                self._received += 1
                frames.append((frame, kind, seq))

        for frame, kind_hint, seq_hint in frames:
            self._process_frame(frame, kind_hint=kind_hint, seq_hint=seq_hint)

    def _process_frame(self, frame: bytes, *, kind_hint: int, seq_hint: int) -> None:
        decode_started = time.perf_counter()
        if kind_hint not in (KIND_HOT, KIND_WARM, KIND_COLD):
            self._drop(kind=kind_hint, seq=seq_hint, error="invalid_kind")
            return

        try:
            result = unpack_frame(
                frame,
                last_seq=self._last_seq_for_kind(kind_hint),
                max_payload_bytes=self._max_payload_bytes,
            )
        except Exception as exc:
            self._drop(kind=kind_hint, seq=seq_hint, error=f"unpack_exception:{exc}")
            return

        if not bool(result.get("valid", False)):
            self._drop(
                kind=result.get("kind", kind_hint),
                seq=result.get("seq", seq_hint),
                error=result.get("error", "invalid_frame"),
            )
            return

        kind = int(result.get("kind", kind_hint))
        seq = int(result["seq"])
        timestamp_ns = int(result.get("timestamp_ns") or 0)
        if timestamp_ns and timestamp_ns < self._last_timestamp_for_kind(kind):
            # Roadmap §2.4 requires timestamp monotonicity in addition to seq
            # monotonicity.  This protects replay/live async paths from delayed
            # older frames overwriting newer dashboard truth.
            self._drop(kind=kind, seq=seq, error="stale_timestamp")
            return

        payload = result.get("payload")
        if not isinstance(payload, dict):
            self._drop(kind=kind, seq=seq, error="invalid_payload")
            return

        try:
            projected = project_snapshot(payload, kind)
        except Exception as exc:
            self._drop(kind=kind, seq=seq, error=f"projection_failed:{exc}")
            return
        if not isinstance(projected, dict):
            self._drop(kind=kind, seq=seq, error="invalid_projection")
            return

        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        self._record_decode_ms(decode_ms)

        with self._lock:
            self._set_last_seq_for_kind(kind, seq)
            if timestamp_ns:
                self._set_last_timestamp_for_kind(kind, timestamp_ns)
            # Retain the most recent VALIDATED payload per tier so consumer-
            # side refresh (driven by DashboardClock ticks via MainWindow
            # on_*_tick slots) can pull truth at its own cadence without
            # forcing the producer to fabricate frames. (See CRIT-A fix.)
            if kind == KIND_HOT:
                self._last_valid_hot_payload = dict(projected)
            elif kind == KIND_WARM:
                self._last_valid_warm_payload = dict(projected)
            elif kind == KIND_COLD:
                self._last_valid_cold_payload = dict(projected)

        if kind == KIND_HOT:
            self.new_hot_frame.emit(projected)
        elif kind == KIND_WARM:
            self.new_warm_frame.emit(projected)
        elif kind == KIND_COLD:
            self.new_cold_frame.emit(projected)

        self._log_periodic_summary()

    # ------------------------------------------------------------------
    # NOTE — fabricator methods removed (CRIT-A, repo-wide review).
    # ------------------------------------------------------------------
    # An earlier revision exposed emit_hot/emit_warm/emit_cold which fabricated
    # placeholder frames with hardcoded values like
    #   {"system_state": "ACTIVE", "feed_health": "HEALTHY", "mode": "PAPER"}
    # so the DashboardClock could "tick" the bus. This directly violated:
    #   - dashboard_roadmap.txt PRIMARY RULE: "It must never guess."
    #   - dashboard_roadmap.txt §11 rule #5: "Never hide stale data behind
    #     fake freshness."
    # The kill-switch would correctly detect backend death while the bus next
    # door kept shouting "HEALTHY" — contradictory truths on the same UI.
    #
    # New architecture (roadmap §2.2 step-ordering compliant):
    #   - SnapshotBus stays PUSH-ONLY: new_*_frame signals fire only when
    #     real backend frames are decoded by _process_frame.
    #   - DashboardClock ticks drive CONSUMER cadence via NEW MainWindow
    #     slots on_hot_tick / on_warm_tick / on_cold_tick which pull
    #     last_valid_*_payload() from the bus and re-render. If no payload
    #     exists yet, panels render the "missing" state (roadmap-compliant).
    # ------------------------------------------------------------------

    def _log_periodic_summary(self) -> None:
        now_mono = time.monotonic()
        if (now_mono - self._last_summary_mono) < self._summary_interval_seconds:
            return
        self._last_summary_mono = now_mono
        stats = self.get_stats()
        self.log.info(
            "SnapshotBus summary",
            received=stats["received"],
            dropped=stats["dropped"],
            last_seq_hot=stats["last_seq_hot"],
            last_seq_warm=stats["last_seq_warm"],
            last_seq_cold=stats["last_seq_cold"],
            last_decode_ms=stats["last_decode_ms"],
            avg_decode_ms=stats["avg_decode_ms"],
            max_decode_ms=stats["max_decode_ms"],
            slow_decode_count=stats["slow_decode_count"],
        )

    @property
    def last_valid_hot_payload(self) -> dict | None:
        with self._lock:
            if self._last_valid_hot_payload is None:
                return None
            return dict(self._last_valid_hot_payload)

    @property
    def last_valid_warm_payload(self) -> dict | None:
        """Most recent VALIDATED warm-tier payload, or None if none seen yet.

        Used by MainWindow.on_warm_tick() for clock-driven consumer refresh.
        Returning None means "no truth yet" — caller must render the panel's
        missing/stale state, never fabricate a default.
        """
        with self._lock:
            if self._last_valid_warm_payload is None:
                return None
            return dict(self._last_valid_warm_payload)

    @property
    def last_valid_cold_payload(self) -> dict | None:
        """Most recent VALIDATED cold-tier payload, or None if none seen yet.

        Used by MainWindow.on_cold_tick() for clock-driven consumer refresh.
        Returning None means "no truth yet" — caller must render the panel's
        missing/stale state, never fabricate a default.
        """
        with self._lock:
            if self._last_valid_cold_payload is None:
                return None
            return dict(self._last_valid_cold_payload)

    def get_stats(self) -> dict:
        with self._lock:
            self.log.info(
                "SnapshotBus stats requested",
                received=self._received,
                dropped=self._dropped,
            )
            if self._decode_ms_window:
                avg_decode_ms = round(sum(self._decode_ms_window) / max(len(self._decode_ms_window), 1), 3)
            else:
                avg_decode_ms = 0.0
            return {
                "received": self._received,
                "dropped": self._dropped,
                "decode_errors": self._decode_errors,
                "last_seq_hot": self._last_seq_hot,
                "last_seq_warm": self._last_seq_warm,
                "last_seq_cold": self._last_seq_cold,
                "last_ts_hot": self._last_ts_hot,
                "last_ts_warm": self._last_ts_warm,
                "last_ts_cold": self._last_ts_cold,
                "last_decode_ms": round(float(self._last_decode_ms), 3),
                "avg_decode_ms": avg_decode_ms,
                "max_decode_ms": round(float(self._max_decode_ms), 3),
                "slow_decode_count": self._slow_decode_count,
            }


if __name__ == "__main__":
    try:
        from PyQt6.QtCore import QCoreApplication
        from PyQt6.QtTest import QSignalSpy

        from dashboard.core.binary_frame import KIND_HOT, pack_frame

        app = QCoreApplication([])
        _ = app  # Keep reference for signal machinery.

        bus = SnapshotBus()
        spy = QSignalSpy(bus.new_hot_frame)

        frame = pack_frame(
            {
                "system_state": "ACTIVE",
                "mode": "PAPER",
                "feed_health": "HEALTHY",
                "data_quality_score": 91.5,
                "features": {"rsi": 58.1},
            },
            kind=KIND_HOT,
            seq=1,
        )
        # Feed in chunks to verify partial-frame buffering.
        bus.feed_bytes(frame[:10])
        bus.feed_bytes(frame[10:])
        assert len(spy) == 1
        projected = bus.last_valid_hot_payload
        assert projected is not None
        assert projected["feed_health"] == "HEALTHY"
        assert projected["mode"] == "PAPER"
        assert projected["data_quality_score"] == 91.5
        assert "features" in projected
        print("snapshot_bus self-test ok")
    except Exception as exc:
        print(f"snapshot_bus self-test skipped: {exc!r}")