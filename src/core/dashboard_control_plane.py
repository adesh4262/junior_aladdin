"""Backend-side dashboard control-plane shared-memory publisher.

Why this module exists
----------------------
The dashboard already had a read-side kill-switch monitor
(`dashboard/core/shared_memory_kill_switch.py`) and a startup heartbeat presence
check (`dashboard/main.py::_check_backend_heartbeat`).  What was missing was the
backend-side publisher/writer that creates and updates those shared-memory
resources, plus a safe dashboard-side fallback that can trip the kill-switch
without depending only on an unconsumed command queue.

This module completes that control-plane half while staying narrowly scoped to
Part 1 of the pre-Week-9 stabilization work:

1. backend heartbeat publisher
2. backend kill-switch state publisher
3. backend emergency polling helper
4. dashboard emergency shared-memory fallback helper

Design rules
------------
- No dashboard/PyQt imports; safe for backend runtime.
- Shared memory names remain config-driven.
- Control block layout intentionally matches the existing dashboard reader:
      !7I2Q64s
- Best-effort, survivability-first behavior: writer methods never try to create
  extra business logic; they only maintain control-plane truth.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Frozen shared-memory contracts
# ---------------------------------------------------------------------------
HEARTBEAT_MAGIC = b"JAHB"
HEARTBEAT_VERSION = 1
HEARTBEAT_STRUCT = struct.Struct("!4sIIQ")
HEARTBEAT_SHM_SIZE = HEARTBEAT_STRUCT.size

KILL_SWITCH_STRUCT = struct.Struct("!7I2Q64s")
KILL_SWITCH_SHM_SIZE = KILL_SWITCH_STRUCT.size
KILL_SWITCH_VERSION = 1
REASON_MAX_BYTES = 64

MODE_FLAGS: Dict[str, int] = {
    "OBSERVE": 0,
    "PAPER": 1,
    "LIVE": 2,
    "ALERT": 3,
}


def _reason_bytes(reason: str) -> bytes:
    raw = str(reason or "").encode("utf-8", errors="ignore")[:REASON_MAX_BYTES]
    return raw + (b"\x00" * (REASON_MAX_BYTES - len(raw)))


def _decode_reason(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def _mode_flag(mode: str) -> int:
    return MODE_FLAGS.get(str(mode or "OBSERVE").upper().strip(), MODE_FLAGS["OBSERVE"])


def _open_or_create_shm(name: str, size: int) -> Tuple[shared_memory.SharedMemory, bool]:
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        return shm, True
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name, create=False)
        return shm, False


def unpack_kill_switch_block(raw: bytes) -> Dict[str, Any]:
    if len(raw) < KILL_SWITCH_SHM_SIZE:
        raise ValueError("kill_switch_block_too_small")
    version, backend_pid, dashboard_pid, mode, emergency_flag, live_approval_flag, compliance_state, heartbeat_ns, last_ack_seq, reason_raw = KILL_SWITCH_STRUCT.unpack(
        raw[:KILL_SWITCH_SHM_SIZE]
    )
    return {
        "version": int(version),
        "backend_pid": int(backend_pid),
        "dashboard_pid": int(dashboard_pid),
        "mode": int(mode),
        "emergency_stop_flag": int(emergency_flag),
        "live_approval_flag": int(live_approval_flag),
        "compliance_state": int(compliance_state),
        "heartbeat_timestamp_ns": int(heartbeat_ns),
        "last_acknowledged_seq": int(last_ack_seq),
        "safe_lock_reason": _decode_reason(reason_raw),
    }


def pack_kill_switch_block(
    *,
    backend_pid: int,
    dashboard_pid: int,
    mode: int,
    emergency_stop_flag: int,
    live_approval_flag: int,
    compliance_state: int,
    heartbeat_timestamp_ns: int,
    last_acknowledged_seq: int,
    safe_lock_reason: str,
) -> bytes:
    return KILL_SWITCH_STRUCT.pack(
        KILL_SWITCH_VERSION,
        int(max(0, backend_pid)),
        int(max(0, dashboard_pid)),
        int(max(0, mode)),
        int(1 if emergency_stop_flag else 0),
        int(1 if live_approval_flag else 0),
        int(max(0, compliance_state)),
        int(max(0, heartbeat_timestamp_ns)),
        int(max(0, last_acknowledged_seq)),
        _reason_bytes(safe_lock_reason),
    )


def trigger_emergency_kill_switch(
    kill_switch_name: str,
    *,
    reason: str,
    dashboard_pid: Optional[int] = None,
) -> bool:
    """Trip the existing kill-switch block from the dashboard side.

    This is the backend-independent emergency fallback used by the dashboard
    dialog. It does NOT create the shared memory block if the backend has not
    started; failing closed here is intentional because the backend publisher is
    the owner of the control-plane resources.
    """
    log = setup_logger("dashboard_control_plane")
    try:
        shm = shared_memory.SharedMemory(name=str(kill_switch_name), create=False)
    except FileNotFoundError:
        log.critical(
            "Emergency fallback failed; kill-switch shared memory missing",
            kill_switch_name=kill_switch_name,
        )
        return False
    except Exception as exc:
        log.critical(
            "Emergency fallback failed; cannot open kill-switch shared memory",
            kill_switch_name=kill_switch_name,
            error=str(exc),
        )
        return False

    try:
        current = unpack_kill_switch_block(bytes(shm.buf[:KILL_SWITCH_SHM_SIZE]))
        payload = pack_kill_switch_block(
            backend_pid=current.get("backend_pid", 0),
            dashboard_pid=int(max(0, dashboard_pid if dashboard_pid is not None else os.getpid())),
            mode=current.get("mode", MODE_FLAGS["OBSERVE"]),
            emergency_stop_flag=1,
            live_approval_flag=current.get("live_approval_flag", 0),
            compliance_state=current.get("compliance_state", 0),
            heartbeat_timestamp_ns=time.time_ns(),
            last_acknowledged_seq=current.get("last_acknowledged_seq", 0) + 1,
            safe_lock_reason=str(reason),
        )
        shm.buf[: len(payload)] = payload
        log.error(
            "Emergency fallback wrote kill-switch shared memory",
            kill_switch_name=kill_switch_name,
            reason=reason,
        )
        return True
    except Exception as exc:
        log.critical(
            "Emergency fallback failed while writing kill-switch shared memory",
            kill_switch_name=kill_switch_name,
            error=str(exc),
        )
        return False
    finally:
        try:
            shm.close()
        except Exception:
            pass


@dataclass
class ControlPlaneStatus:
    heartbeat_name: str
    kill_switch_name: str
    backend_pid: int
    mode: str
    live_approval_flag: int
    compliance_state: int
    heartbeat_timestamp_ns: int
    last_acknowledged_seq: int
    emergency_stop_flag: int
    safe_lock_reason: str


class DashboardControlPlanePublisher:
    """Owns backend-side heartbeat + kill-switch shared memory publishing."""

    def __init__(
        self,
        *,
        heartbeat_name: str,
        kill_switch_name: str,
        mode: str,
        backend_pid: Optional[int] = None,
        compliance_state: int = 1,
        live_approval_flag: Optional[int] = None,
    ) -> None:
        self._log = setup_logger("dashboard_control_plane")
        self.heartbeat_name = str(heartbeat_name)
        self.kill_switch_name = str(kill_switch_name)
        self.mode = str(mode or "OBSERVE").upper().strip()
        self.backend_pid = int(max(0, backend_pid if backend_pid is not None else os.getpid()))
        self.compliance_state = int(max(0, compliance_state))
        self.live_approval_flag = int(1 if (live_approval_flag if live_approval_flag is not None else self.mode == "LIVE") else 0)

        self._heartbeat_shm: Optional[shared_memory.SharedMemory] = None
        self._kill_switch_shm: Optional[shared_memory.SharedMemory] = None
        self._owns_heartbeat = False
        self._owns_kill_switch = False
        self._last_heartbeat_ns = 0
        self._last_seq = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        try:
            self._heartbeat_shm, self._owns_heartbeat = _open_or_create_shm(self.heartbeat_name, HEARTBEAT_SHM_SIZE)
            self._kill_switch_shm, self._owns_kill_switch = _open_or_create_shm(self.kill_switch_name, KILL_SWITCH_SHM_SIZE)

            # What changed:
            # Backend now becomes the owner/publisher of both control-plane shared
            # memory blocks so the dashboard's kill-switch reader and heartbeat
            # startup guard finally have real runtime state to observe.
            self.publish_state(reason="backend_starting")
            self.publish_heartbeat()
            self._log.info(
                "Dashboard control plane started",
                heartbeat_name=self.heartbeat_name,
                kill_switch_name=self.kill_switch_name,
                mode=self.mode,
                backend_pid=self.backend_pid,
                created_heartbeat=self._owns_heartbeat,
                created_kill_switch=self._owns_kill_switch,
            )
            return True
        except Exception as exc:
            self._log.error(
                "Dashboard control plane start failed",
                error=str(exc),
                heartbeat_name=self.heartbeat_name,
                kill_switch_name=self.kill_switch_name,
            )
            self.stop(reason="control_plane_start_failed")
            return False

    def stop(self, reason: str = "backend_shutdown") -> None:
        try:
            if self._kill_switch_shm is not None:
                # Why this write exists:
                # Multiple agents/tools may inspect the shared-memory block after
                # shutdown. Writing the last known reason before unlink keeps the
                # final state explicit instead of disappearing silently.
                self.publish_state(reason=reason, emergency_stop_flag=0)
        except Exception:
            pass

        for shm, owns, label in (
            (self._heartbeat_shm, self._owns_heartbeat, "heartbeat"),
            (self._kill_switch_shm, self._owns_kill_switch, "kill_switch"),
        ):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
            if owns:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    self._log.warning(
                        "Control-plane shared memory unlink failed",
                        shm_type=label,
                        error=str(exc),
                    )

        self._heartbeat_shm = None
        self._kill_switch_shm = None
        self._owns_heartbeat = False
        self._owns_kill_switch = False

    # ------------------------------------------------------------------
    # Publish / read
    # ------------------------------------------------------------------
    def publish_heartbeat(self) -> int:
        if self._heartbeat_shm is None:
            raise RuntimeError("control_plane_not_started")
        heartbeat_ns = time.time_ns()
        payload = HEARTBEAT_STRUCT.pack(
            HEARTBEAT_MAGIC,
            HEARTBEAT_VERSION,
            self.backend_pid,
            heartbeat_ns,
        )
        self._heartbeat_shm.buf[: len(payload)] = payload
        self._last_heartbeat_ns = heartbeat_ns

        # Keep the kill-switch block heartbeat fresh too. The dashboard reader
        # uses this timestamp for stale-feed / emergency detection.
        if self._kill_switch_shm is not None:
            self.publish_state(reason=self.current_status().safe_lock_reason, heartbeat_timestamp_ns=heartbeat_ns)
        return heartbeat_ns

    def publish_state(
        self,
        *,
        reason: str,
        emergency_stop_flag: Optional[int] = None,
        heartbeat_timestamp_ns: Optional[int] = None,
        dashboard_pid: Optional[int] = None,
        live_approval_flag: Optional[int] = None,
        compliance_state: Optional[int] = None,
    ) -> None:
        if self._kill_switch_shm is None:
            raise RuntimeError("control_plane_not_started")

        current = self.read_kill_switch_state(default_empty=True)
        self._last_seq = int(current.get("last_acknowledged_seq", 0)) + 1
        hb_ns = int(max(0, heartbeat_timestamp_ns if heartbeat_timestamp_ns is not None else time.time_ns()))
        payload = pack_kill_switch_block(
            backend_pid=self.backend_pid,
            dashboard_pid=int(max(0, dashboard_pid if dashboard_pid is not None else current.get("dashboard_pid", 0))),
            mode=_mode_flag(self.mode),
            emergency_stop_flag=int(current.get("emergency_stop_flag", 0) if emergency_stop_flag is None else emergency_stop_flag),
            live_approval_flag=int(self.live_approval_flag if live_approval_flag is None else live_approval_flag),
            compliance_state=int(self.compliance_state if compliance_state is None else compliance_state),
            heartbeat_timestamp_ns=hb_ns,
            last_acknowledged_seq=self._last_seq,
            safe_lock_reason=str(reason),
        )
        self._kill_switch_shm.buf[: len(payload)] = payload
        self._last_heartbeat_ns = hb_ns

    def read_kill_switch_state(self, *, default_empty: bool = False) -> Dict[str, Any]:
        if self._kill_switch_shm is None:
            if default_empty:
                return {
                    "version": KILL_SWITCH_VERSION,
                    "backend_pid": self.backend_pid,
                    "dashboard_pid": 0,
                    "mode": _mode_flag(self.mode),
                    "emergency_stop_flag": 0,
                    "live_approval_flag": self.live_approval_flag,
                    "compliance_state": self.compliance_state,
                    "heartbeat_timestamp_ns": self._last_heartbeat_ns,
                    "last_acknowledged_seq": self._last_seq,
                    "safe_lock_reason": "",
                }
            raise RuntimeError("control_plane_not_started")
        return unpack_kill_switch_block(bytes(self._kill_switch_shm.buf[:KILL_SWITCH_SHM_SIZE]))

    def poll_emergency_request(self) -> Optional[str]:
        state = self.read_kill_switch_state(default_empty=True)
        if int(state.get("emergency_stop_flag", 0)) == 1:
            reason = str(state.get("safe_lock_reason") or "emergency_stop_flag")
            return reason
        return None

    def current_status(self) -> ControlPlaneStatus:
        state = self.read_kill_switch_state(default_empty=True)
        return ControlPlaneStatus(
            heartbeat_name=self.heartbeat_name,
            kill_switch_name=self.kill_switch_name,
            backend_pid=self.backend_pid,
            mode=self.mode,
            live_approval_flag=int(state.get("live_approval_flag", self.live_approval_flag)),
            compliance_state=int(state.get("compliance_state", self.compliance_state)),
            heartbeat_timestamp_ns=int(state.get("heartbeat_timestamp_ns", self._last_heartbeat_ns)),
            last_acknowledged_seq=int(state.get("last_acknowledged_seq", self._last_seq)),
            emergency_stop_flag=int(state.get("emergency_stop_flag", 0)),
            safe_lock_reason=str(state.get("safe_lock_reason", "")),
        )


if __name__ == "__main__":
    hb_name = f"ja_backend_heartbeat_selftest_{os.getpid()}"
    ks_name = f"junior_aladdin_kill_switch_selftest_{os.getpid()}"
    cp = DashboardControlPlanePublisher(
        heartbeat_name=hb_name,
        kill_switch_name=ks_name,
        mode="OBSERVE",
    )
    assert cp.start() is True
    try:
        status = cp.current_status()
        assert status.backend_pid == os.getpid()
        assert status.emergency_stop_flag == 0
        assert status.heartbeat_timestamp_ns > 0
        assert trigger_emergency_kill_switch(ks_name, reason="selftest", dashboard_pid=999) is True
        assert cp.poll_emergency_request() == "selftest"
        print("dashboard_control_plane self-test passed")
    finally:
        cp.stop()
