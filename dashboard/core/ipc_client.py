"""Dashboard-side IPC clients for Part 2A.

This module intentionally handles only:
- snapshot frame ingestion from backend
- command-channel socket wrapper for CommandRouter

No Qt dependency is required here; the dashboard main process can own these
objects and wire them into SnapshotBus / CommandRouter safely.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, Callable, Optional

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


_LENGTH_STRUCT = struct.Struct("!I")
_DEFAULT_TIMEOUT_SEC = 0.5
_DEFAULT_CONNECT_TIMEOUT_SEC = 1.0
_DEFAULT_RECONNECT_DELAY_SEC = 1.0


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


class SnapshotStreamClient:
    """Reconnectable background client that feeds raw frames into SnapshotBus."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        frame_handler: Callable[[bytes], None],
        reconnect_delay_sec: float = _DEFAULT_RECONNECT_DELAY_SEC,
        socket_timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        connect_timeout_sec: float = _DEFAULT_CONNECT_TIMEOUT_SEC,
    ) -> None:
        self._log = setup_logger("dashboard_snapshot_client")
        self.host = str(host)
        self.port = int(port)
        self._frame_handler = frame_handler
        self._reconnect_delay_sec = float(reconnect_delay_sec)
        self._socket_timeout_sec = float(socket_timeout_sec)
        self._connect_timeout_sec = float(connect_timeout_sec)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="DashboardSnapshotClient")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_socket()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connect()
                self._read_loop()
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._log.warning(
                        "Snapshot client disconnected / connect failed",
                        host=self.host,
                        port=self.port,
                        error=str(exc),
                    )
            finally:
                self._close_socket()
            if not self._stop_event.is_set():
                time.sleep(self._reconnect_delay_sec)

    def _connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self._connect_timeout_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self._socket_timeout_sec)
        self._sock = sock
        self._log.info("Snapshot client connected", host=self.host, port=self.port)

    def _read_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            packet = _recv_packet(self._sock)
            if packet is None:
                continue
            self._frame_handler(packet)

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class LengthPrefixedSocketCommandChannel:
    """Socket-like wrapper used by CommandRouter for real backend commands.

    CommandRouter serializes dicts to msgpack bytes; this wrapper adds a 4-byte
    network-order length prefix so the backend command server can safely parse
    a persistent stream.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        connect_timeout_sec: float = _DEFAULT_CONNECT_TIMEOUT_SEC,
        socket_timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._log = setup_logger("dashboard_command_channel")
        self.host = str(host)
        self.port = int(port)
        self._connect_timeout_sec = float(connect_timeout_sec)
        self._socket_timeout_sec = float(socket_timeout_sec)
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None

    def sendall(self, payload: bytes) -> None:
        with self._lock:
            sock = self._ensure_connected()
            try:
                sock.sendall(_LENGTH_STRUCT.pack(len(payload)) + payload)
            except Exception:
                self._close_locked()
                raise

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _ensure_connected(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.create_connection((self.host, self.port), timeout=self._connect_timeout_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self._socket_timeout_sec)
        self._sock = sock
        self._log.info("Command channel connected", host=self.host, port=self.port)
        return sock

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
