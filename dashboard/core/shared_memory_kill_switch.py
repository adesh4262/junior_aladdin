from __future__ import annotations

import struct
import time
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from multiprocessing import shared_memory

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


# Module-level constants — published for external consumers (tests, monitors, etc.).
# The kill-switch block uses struct format "!7I2Q64s" (128 bytes).
# Offsets are derived from that layout:
#   !7I  -> 7 * 4 = 28 bytes (meta section)
#   !2Q  -> 2 * 8 = 16 bytes (tail section)
#   64s  -> 1 * 64 = 64 bytes (reason string)
EMERGENCY_STOP_FLAG_OFFSET = 16      # 5th uint32 in !7I (index 4)
HEARTBEAT_NS_OFFSET = 28             # after 7 uint32s (28 bytes) → first uint64
SEQUENCE_OFFSET = 36                 # after 7 uint32s + 1 uint64 (28 + 8) → second uint64
SHM_SIZE = 128                       # total block size (matches _MIN_BLOCK_BYTES)
HEARTBEAT_TIMEOUT_NS = 5 * 1_000_000_000  # 5 seconds stale threshold in ns


class KillSwitchReader(QObject):
    """Read-only monitor for backend kill-switch shared memory."""

    emergency_activated = pyqtSignal(str)

    _REOPEN_INTERVAL_SEC = 5.0
    _MIN_BLOCK_BYTES = 128

    # version, backend_pid, dashboard_pid, mode, emergency, live_approval, compliance
    _META_FMT = "!7I"
    _META_SIZE = struct.calcsize(_META_FMT)

    # heartbeat_timestamp_ns, last_acknowledged_seq
    _TAIL_FMT = "!2Q"
    _TAIL_SIZE = struct.calcsize(_TAIL_FMT)

    # safe_lock_reason[64]
    _REASON_SIZE = 64

    def __init__(
        self,
        name: str,
        parent: QObject | None = None,
        stale_threshold_sec: float = 5.0,
        poll_interval_ms: int = 200,
    ) -> None:
        super().__init__(parent)
        self.log = setup_logger("dashboard_core_kill_switch")

        self.name = str(name)
        self.stale_threshold_sec = float(stale_threshold_sec)
        self.poll_interval_ms = int(poll_interval_ms)

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._available = False
        self._emergency_active = False
        self._last_heartbeat_ns = 0
        self._stale_consecutive = 0
        self._last_open_attempt_ns = 0

        self._try_open(initial=True)

    def _try_open(self, initial: bool = False) -> None:
        now_ns = time.time_ns()
        if not initial and (now_ns - self._last_open_attempt_ns) < int(self._REOPEN_INTERVAL_SEC * 1e9):
            return
        self._last_open_attempt_ns = now_ns

        try:
            self._close_shm()
            self._shm = shared_memory.SharedMemory(name=self.name, create=False)
            if not self._available:
                self.log.info(
                    "Kill-switch shared memory available",
                    dashboard_component="kill_switch",
                    shm_name=self.name,
                )
            self._available = True
        except FileNotFoundError:
            if initial:
                self.log.critical(
                    "Kill-switch shared memory not found",
                    dashboard_component="kill_switch",
                    shm_name=self.name,
                )
            self._available = False
            self._shm = None
        except Exception as exc:
            self.log.error(
                "Kill-switch shared memory open failed",
                dashboard_component="kill_switch",
                shm_name=self.name,
                error=str(exc),
            )
            self._available = False
            self._shm = None

    def _close_shm(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    def stop(self) -> None:
        self._close_shm()

    def _parse_block(self, raw: bytes) -> Dict[str, Any]:
        if len(raw) < self._MIN_BLOCK_BYTES:
            raise ValueError("shared_memory_block_too_small")

        version, backend_pid, dashboard_pid, mode, emergency_flag, live_approval_flag, compliance_state = struct.unpack_from(
            self._META_FMT, raw, 0
        )

        heartbeat_ns, last_ack_seq = struct.unpack_from(self._TAIL_FMT, raw, self._META_SIZE)
        reason_offset = self._META_SIZE + self._TAIL_SIZE
        reason_raw = struct.unpack_from(f"!{self._REASON_SIZE}s", raw, reason_offset)[0]
        safe_lock_reason = reason_raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

        return {
            "version": version,
            "backend_pid": backend_pid,
            "dashboard_pid": dashboard_pid,
            "mode": mode,
            "emergency_stop_flag": emergency_flag,
            "live_approval_flag": live_approval_flag,
            "compliance_state": compliance_state,
            "heartbeat_timestamp_ns": heartbeat_ns,
            "last_acknowledged_seq": last_ack_seq,
            "safe_lock_reason": safe_lock_reason,
        }

    def check(self) -> None:
        if not self._available or self._shm is None:
            self._try_open()
            if not self._available or self._shm is None:
                return

        try:
            raw = bytes(self._shm.buf[: self._MIN_BLOCK_BYTES])
            info = self._parse_block(raw)
        except FileNotFoundError:
            if self._available:
                self.log.error(
                    "Kill-switch shared memory unavailable",
                    dashboard_component="kill_switch",
                    shm_name=self.name,
                )
            self._available = False
            self._close_shm()
            return
        except Exception as exc:
            self.log.error(
                "Kill-switch read/parse failed",
                dashboard_component="kill_switch",
                error=str(exc),
            )
            self._stale_consecutive += 1
            if self._stale_consecutive >= 2 and not self._emergency_active:
                self._emergency_active = True
                self.emergency_activated.emit("backend_heartbeat_stale")
            return

        heartbeat_ns = int(info.get("heartbeat_timestamp_ns", 0))
        now_ns = time.time_ns()
        stale_limit_ns = int(self.stale_threshold_sec * 1e9)
        is_stale = heartbeat_ns <= 0 or (now_ns - heartbeat_ns) > stale_limit_ns

        if is_stale:
            self._stale_consecutive += 1
        else:
            if self._stale_consecutive >= 2 and self._emergency_active:
                self.log.info(
                    "Kill-switch heartbeat restored",
                    dashboard_component="kill_switch",
                    heartbeat_timestamp_ns=heartbeat_ns,
                )
            self._stale_consecutive = 0
            self._last_heartbeat_ns = heartbeat_ns

        emergency_flag = int(info.get("emergency_stop_flag", 0))
        if emergency_flag == 1 and not self._emergency_active:
            self._emergency_active = True
            reason = info.get("safe_lock_reason") or "emergency_stop_flag"
            self.log.error(
                "Kill-switch emergency activated",
                dashboard_component="kill_switch",
                reason=reason,
            )
            self.emergency_activated.emit(str(reason))
            return

        if self._stale_consecutive >= 2 and not self._emergency_active:
            self._emergency_active = True
            self.log.error(
                "Kill-switch heartbeat stale",
                dashboard_component="kill_switch",
                stale_checks=self._stale_consecutive,
                stale_threshold_sec=self.stale_threshold_sec,
            )
            self.emergency_activated.emit("backend_heartbeat_stale")


if __name__ == "__main__":
    test_name = "ja_kill_switch_selftest"
    shm = shared_memory.SharedMemory(name=test_name, create=True, size=128)
    emitted: list[str] = []

    def _write_block(emergency_flag: int, heartbeat_ns: int, reason: str) -> None:
        reason_bytes = reason.encode("utf-8")[:64]
        reason_bytes = reason_bytes + (b"\x00" * (64 - len(reason_bytes)))
        packed = struct.pack(
            "!7I2Q64s",
            1,  # version
            1234,  # backend_pid
            0,  # dashboard_pid
            1,  # mode
            emergency_flag,
            0,  # live_approval_flag
            1,  # compliance_state
            heartbeat_ns,
            0,  # last_acknowledged_seq
            reason_bytes,
        )
        shm.buf[: len(packed)] = packed

    try:
        reader = KillSwitchReader(test_name, stale_threshold_sec=0.01)
        reader.emergency_activated.connect(lambda r: emitted.append(r))

        _write_block(0, time.time_ns(), "")
        reader.check()
        assert not emitted

        _write_block(1, time.time_ns(), "manual_test")
        reader.check()
        assert emitted and emitted[-1] == "manual_test"

        print("shared_memory_kill_switch self-test passed")
    finally:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass