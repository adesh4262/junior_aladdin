"""Backend-side dashboard IPC transport.

Scope
-----
This module implements only Part 2A of the Week-8 closeout work:

1. real snapshot transport wiring
2. real backend command channel wiring

It intentionally does NOT change trading logic, chart logic, or future-week UI
surfaces.  It only gives the backend a real IPC surface that the dashboard can
connect to.

Transport design
----------------
- localhost TCP only (safe for Windows/Linux, simpler than Unix sockets)
- length-prefixed frames/messages for deterministic stream parsing
- snapshot channel sends packed binary frames produced by
  ``dashboard.core.binary_frame.pack_frame``
- command channel receives msgpack command dictionaries and queues them for the
  backend orchestrator to poll

Why TCP and not multiprocessing.Queue?
--------------------------------------
The dashboard and backend are separate processes started independently. A local
``multiprocessing.Queue`` created inside one process is not automatically shared
with another independently-started process. A localhost TCP transport gives us a
real cross-process channel without touching trading engines.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from enum import Enum
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import msgpack

from dashboard.core.binary_frame import KIND_COLD, KIND_HOT, KIND_WARM, pack_frame

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


_LENGTH_STRUCT = struct.Struct("!I")
_DEFAULT_BACKLOG = 8
_DEFAULT_SOCKET_TIMEOUT_SEC = 0.5
_DEFAULT_CONNECT_TIMEOUT_SEC = 1.0
_DEFAULT_COMMAND_QUEUE_MAX = 1000


def _send_packet(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_LENGTH_STRUCT.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            return None
        if not chunk:
            raise ConnectionError("socket_closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_packet(sock: socket.socket) -> Optional[bytes]:
    header = _recv_exact(sock, _LENGTH_STRUCT.size)
    if header is None:
        return None
    (size,) = _LENGTH_STRUCT.unpack(header)
    if size <= 0:
        raise ValueError(f"invalid_packet_size:{size}")
    payload = _recv_exact(sock, size)
    if payload is None:
        return None
    return payload


def _snapshot_sanitize(value: Any, *, _depth: int = 0, _max_depth: int = 8) -> Any:
    """Convert backend snapshot objects into msgpack-friendly structures.

    This is a transport serializer only. It does not compute new trading truth.
    It preserves existing values as closely as possible while removing Python-
    specific container/types that msgpack cannot encode directly.
    """
    if _depth > _max_depth:
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return getattr(value, "value", str(value))

    if isinstance(value, Mapping):
        return {
            str(k): _snapshot_sanitize(v, _depth=_depth + 1, _max_depth=_max_depth)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set, deque)):
        return [_snapshot_sanitize(v, _depth=_depth + 1, _max_depth=_max_depth) for v in value]

    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _snapshot_sanitize(value.tolist(), _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            pass

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _snapshot_sanitize(value.item(), _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _snapshot_sanitize(dict(value.__dict__), _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            pass

    return str(value)


class DashboardSnapshotServer:
    """Broadcasts packed snapshot frames to connected dashboard clients."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        backlog: int = _DEFAULT_BACKLOG,
        socket_timeout_sec: float = _DEFAULT_SOCKET_TIMEOUT_SEC,
    ) -> None:
        self._log = setup_logger("dashboard_snapshot_server")
        self.host = str(host)
        self.port = int(port)
        self._backlog = int(backlog)
        self._socket_timeout_sec = float(socket_timeout_sec)

        self._server_socket: Optional[socket.socket] = None
        self._clients: Dict[int, socket.socket] = {}
        self._clients_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(self._backlog)
            srv.settimeout(self._socket_timeout_sec)
            self._server_socket = srv
            self.port = int(srv.getsockname()[1])
            self._stop_event.clear()
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="DashboardSnapshotAccept")
            self._accept_thread.start()
            self._log.info(
                "Dashboard snapshot server started",
                host=self.host,
                port=self.port,
            )
            return True
        except Exception as exc:
            self._log.error(
                "Dashboard snapshot server start failed",
                host=self.host,
                port=self.port,
                error=str(exc),
            )
            self.stop()
            return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)
        with self._clients_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for sock in clients:
            try:
                sock.close()
            except Exception:
                pass

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def publish_frame(self, frame: bytes) -> int:
        if not frame:
            return 0
        dead: List[int] = []
        sent = 0
        with self._clients_lock:
            items = list(self._clients.items())
        for fileno, sock in items:
            try:
                _send_packet(sock, frame)
                sent += 1
            except Exception:
                dead.append(fileno)
        if dead:
            self._drop_clients(dead)
        return sent

    def _accept_loop(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                client, addr = self._server_socket.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(self._socket_timeout_sec)
                with self._clients_lock:
                    self._clients[client.fileno()] = client
                self._log.info(
                    "Dashboard snapshot client connected",
                    client=str(addr),
                    clients=self.client_count(),
                )
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._log.warning("Snapshot accept loop error", error=str(exc))

    def _drop_clients(self, dead_filenos: Sequence[int]) -> None:
        with self._clients_lock:
            for fileno in dead_filenos:
                sock = self._clients.pop(fileno, None)
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


class DashboardCommandServer:
    """Receives length-prefixed msgpack command dictionaries from dashboard."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        backlog: int = _DEFAULT_BACKLOG,
        socket_timeout_sec: float = _DEFAULT_SOCKET_TIMEOUT_SEC,
        queue_max: int = _DEFAULT_COMMAND_QUEUE_MAX,
    ) -> None:
        self._log = setup_logger("dashboard_command_server")
        self.host = str(host)
        self.port = int(port)
        self._backlog = int(backlog)
        self._socket_timeout_sec = float(socket_timeout_sec)
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max(1, int(queue_max)))

        self._server_socket: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._clients_lock = threading.Lock()
        self._clients: Dict[int, socket.socket] = {}
        self._client_threads: Dict[int, threading.Thread] = {}

    def start(self) -> bool:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(self._backlog)
            srv.settimeout(self._socket_timeout_sec)
            self._server_socket = srv
            self.port = int(srv.getsockname()[1])
            self._stop_event.clear()
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="DashboardCommandAccept")
            self._accept_thread.start()
            self._log.info(
                "Dashboard command server started",
                host=self.host,
                port=self.port,
            )
            return True
        except Exception as exc:
            self._log.error(
                "Dashboard command server start failed",
                host=self.host,
                port=self.port,
                error=str(exc),
            )
            self.stop()
            return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)
        with self._clients_lock:
            clients = list(self._clients.values())
            self._clients.clear()
            threads = list(self._client_threads.values())
            self._client_threads.clear()
        for sock in clients:
            try:
                sock.close()
            except Exception:
                pass
        for th in threads:
            if th.is_alive():
                th.join(timeout=1.0)

    def poll_commands(self, limit: int = 64) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _ in range(max(1, int(limit))):
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def _accept_loop(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                client, addr = self._server_socket.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(self._socket_timeout_sec)
                with self._clients_lock:
                    self._clients[client.fileno()] = client
                th = threading.Thread(
                    target=self._client_loop,
                    args=(client, addr),
                    daemon=True,
                    name=f"DashboardCommandClient-{client.fileno()}",
                )
                with self._clients_lock:
                    self._client_threads[client.fileno()] = th
                th.start()
                self._log.info("Dashboard command client connected", client=str(addr))
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._log.warning("Command accept loop error", error=str(exc))

    def _client_loop(self, client: socket.socket, addr: Tuple[str, int]) -> None:
        fileno = client.fileno()
        try:
            while not self._stop_event.is_set():
                try:
                    packet = _recv_packet(client)
                except ConnectionError:
                    break
                if packet is None:
                    continue
                try:
                    decoded = msgpack.unpackb(packet, raw=False, strict_map_key=False)
                except Exception as exc:
                    self._log.warning("Command decode failed", client=str(addr), error=str(exc))
                    continue
                if not isinstance(decoded, Mapping):
                    self._log.warning("Command payload is not mapping; ignored", client=str(addr), payload_type=type(decoded).__name__)
                    continue
                try:
                    self._queue.put_nowait(dict(decoded))
                except queue.Full:
                    self._log.warning("Dashboard command queue full; dropping oldest command")
                    try:
                        _ = self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(dict(decoded))
                    except queue.Full:
                        pass
        except Exception as exc:
            if not self._stop_event.is_set():
                self._log.warning("Command client loop error", client=str(addr), error=str(exc))
        finally:
            with self._clients_lock:
                self._clients.pop(fileno, None)
                self._client_threads.pop(fileno, None)
            try:
                client.close()
            except Exception:
                pass


class DashboardIpcBridge:
    """Combines snapshot broadcast + command receive for backend runtime."""

    def __init__(
        self,
        *,
        host: str,
        snapshot_port: int,
        command_port: int,
        hot_interval_ms: int,
        warm_interval_ms: int,
        cold_interval_ms: int,
        max_payload_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self._log = setup_logger("dashboard_ipc_bridge")
        self.host = str(host)
        self.snapshot_port = int(snapshot_port)
        self.command_port = int(command_port)
        self._hot_interval_ms = max(50, int(hot_interval_ms))
        self._warm_interval_ms = max(100, int(warm_interval_ms))
        self._cold_interval_ms = max(500, int(cold_interval_ms))
        self._max_payload_bytes = int(max_payload_bytes)

        self._snapshot_server = DashboardSnapshotServer(host=self.host, port=self.snapshot_port)
        self._command_server = DashboardCommandServer(host=self.host, port=self.command_port)

        self._last_hot_mono = 0.0
        self._last_warm_mono = 0.0
        self._last_cold_mono = 0.0
        self._seq_hot = 0
        self._seq_warm = 0
        self._seq_cold = 0

    def start(self) -> bool:
        ok_snap = self._snapshot_server.start()
        ok_cmd = self._command_server.start()
        self.snapshot_port = self._snapshot_server.port
        self.command_port = self._command_server.port
        if ok_snap and ok_cmd:
            self._log.info(
                "Dashboard IPC bridge started",
                host=self.host,
                snapshot_port=self.snapshot_port,
                command_port=self.command_port,
            )
            return True
        self.stop()
        return False

    def stop(self) -> None:
        self._snapshot_server.stop()
        self._command_server.stop()

    def publish_due_frames(self, snapshot: Mapping[str, Any], now_mono: Optional[float] = None) -> None:
        now_v = float(time.monotonic() if now_mono is None else now_mono)
        snap = _snapshot_sanitize(dict(snapshot) if isinstance(snapshot, Mapping) else {})

        if self._last_hot_mono == 0.0 or ((now_v - self._last_hot_mono) * 1000.0) >= self._hot_interval_ms:
            self._seq_hot += 1
            self._publish_kind(KIND_HOT, self._seq_hot, snap)
            self._last_hot_mono = now_v

        if self._last_warm_mono == 0.0 or ((now_v - self._last_warm_mono) * 1000.0) >= self._warm_interval_ms:
            self._seq_warm += 1
            self._publish_kind(KIND_WARM, self._seq_warm, snap)
            self._last_warm_mono = now_v

        if self._last_cold_mono == 0.0 or ((now_v - self._last_cold_mono) * 1000.0) >= self._cold_interval_ms:
            self._seq_cold += 1
            self._publish_kind(KIND_COLD, self._seq_cold, snap)
            self._last_cold_mono = now_v

    def poll_commands(self, limit: int = 64) -> List[Dict[str, Any]]:
        return self._command_server.poll_commands(limit=limit)

    def get_status(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "snapshot_port": self.snapshot_port,
            "command_port": self.command_port,
            "snapshot_clients": self._snapshot_server.client_count(),
            "seq_hot": self._seq_hot,
            "seq_warm": self._seq_warm,
            "seq_cold": self._seq_cold,
        }

    def _publish_kind(self, kind: int, seq: int, snapshot: Mapping[str, Any]) -> None:
        try:
            frame = pack_frame(dict(snapshot), kind=kind, seq=seq, max_payload_bytes=self._max_payload_bytes)
            self._snapshot_server.publish_frame(frame)
        except Exception as exc:
            self._log.warning(
                "Dashboard IPC frame publish failed",
                kind=kind,
                seq=seq,
                error=str(exc),
            )


if __name__ == "__main__":
    bridge = DashboardIpcBridge(
        host="127.0.0.1",
        snapshot_port=0,
        command_port=0,
        hot_interval_ms=200,
        warm_interval_ms=1000,
        cold_interval_ms=5000,
    )
    assert bridge.start() is True
    try:
        bridge.publish_due_frames({"system_state": "ACTIVE", "feed_health": "HEALTHY"})
        print("dashboard_ipc self-test started", bridge.get_status())
    finally:
        bridge.stop()
