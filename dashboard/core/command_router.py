"""
dashboard/core/command_router.py

Central, non-blocking command sender from the dashboard UI to the backend.

Design goals:
- Transport-agnostic via duck-typed command channel (Queue / socket-like / custom).
- Synchronous, non-blocking sends (no threads, no async/await).
- Defensive: never raise to callers; drop commands on failure with strong logging.
- Replay-safe: can be disabled via set_enabled(False); disabled router ignores commands.

Command format:
- Msgpack-encoded dict (must be msgpack-serializable).
- Recommended: include at least {"type": "<command_type>", ...}
"""

from __future__ import annotations

from typing import Any, Mapping, Optional
import logging
import time
import queue as _queue

import msgpack

from src.utils.logger import setup_logger


class CommandRouter:
    """
    Fire-and-forget command router.

    The router serializes command dictionaries with msgpack and writes the packed
    bytes to an injected command channel.

    Supported channel shapes (duck-typed):
    - Queue-like: put_nowait(data) OR put(data, block=False)
    - Socket-like: send(data) OR sendall(data)

    Notes:
    - This router does not start background threads.
    - This router does not implement request-response (future extension).
    """

    def __init__(self, command_channel: Any, enabled: bool = True) -> None:
        self._log = self._get_logger()
        self._channel = command_channel
        self._enabled = bool(enabled)

        if self._channel is None:
            # Treat missing channel as a critical miswire; keep process alive.
            self._enabled = False
            self._log.critical(
                "CommandRouter initialized with None channel; router disabled.",
                extra={"dashboard_component": "command_router"},
            )

    @staticmethod
    def _get_logger() -> Any:
        """
        Obtain project logger defensively.

        IMPORTANT INTEGRATION FIX:
        ``src.utils.logger.setup_logger`` returns the project BoundLogger wrapper,
        not a raw ``logging.Logger``.  The previous isinstance(logging.Logger)
        check rejected the real project logger and silently fell back to an
        unconfigured stdlib logger, which removed the command audit trail from
        the normal Junior Aladdin log files.

        We accept any logger-like object exposing the methods we need.  This
        preserves the institutional audit trail while still degrading safely if
        logger construction fails.
        """
        name = "dashboard_core_command_router"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "warning", "error", "critical")):
                return log
        except Exception:
            # Fall back to stdlib logging; do not crash.
            pass
        return logging.getLogger(name)

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the router.

        In replay mode, disable to ensure commands are ignored.
        """
        self._enabled = bool(enabled)
        self._log.info(
            "CommandRouter enabled=%s",
            self._enabled,
            extra={"dashboard_component": "command_router"},
        )

    def send(self, command: Mapping[str, Any]) -> bool:
        """Compatibility wrapper for UI/dialog callers.

        Integration note:
        Early dashboard wiring and the local main.py fallback stub exposed a
        ``send(...)`` method, while the production router implemented
        ``send_command(...)``.  EmergencyStopDialog was wired to ``send`` and
        therefore could not actually dispatch commands through this router.

        Keep this wrapper instead of forcing every caller to know the canonical
        method name.  It is intentionally thin and preserves the no-raise
        contract of ``send_command``.
        """
        return self.send_command(command)

    def send_command(self, command: Mapping[str, Any]) -> bool:
        """
        Serialize and send a command (non-blocking best-effort).

        Returns:
            True if the command was successfully written to the channel.
            False if disabled, misconfigured, serialization fails, or write fails.

        Contract:
            - Never raises exceptions to the caller.
            - Drops commands on failure (logs at WARNING/ERROR/CRITICAL).
        """
        # Keep overhead minimal; use perf_counter only if needed.
        _t0 = time.perf_counter()

        if not self._enabled:
            self._log.info(
                "Replay mode / router disabled — command ignored. type=%s",
                self._command_type(command),
                extra={"dashboard_component": "command_router"},
            )
            return False

        if self._channel is None:
            self._log.error(
                "Command channel is not initialized (None) — command dropped. type=%s",
                self._command_type(command),
                extra={"dashboard_component": "command_router"},
            )
            return False

        if not isinstance(command, Mapping):
            self._log.error(
                "Invalid command payload type=%s — expected Mapping/dict; dropped.",
                type(command).__name__,
                extra={"dashboard_component": "command_router"},
            )
            return False

        try:
            packed = msgpack.packb(dict(command), use_bin_type=True)
        except Exception as e:
            self._log.error(
                "Command serialization failed — dropped. type=%s err=%s",
                self._command_type(command),
                repr(e),
                extra={"dashboard_component": "command_router"},
            )
            return False

        ok = self._write_nonblocking(packed, command_type=self._command_type(command))

        # Info-level event trail for observability (as required).
        dt_ms = (time.perf_counter() - _t0) * 1000.0
        if ok:
            self._log.info(
                "Command sent. type=%s bytes=%d latency_ms=%.3f",
                self._command_type(command),
                len(packed),
                dt_ms,
                extra={"dashboard_component": "command_router"},
            )
        else:
            self._log.info(
                "Command dropped. type=%s bytes=%d latency_ms=%.3f",
                self._command_type(command),
                len(packed),
                dt_ms,
                extra={"dashboard_component": "command_router"},
            )
        return ok

    @staticmethod
    def _command_type(command: Any) -> str:
        try:
            if isinstance(command, Mapping):
                val = command.get("type")
                return str(val) if val is not None else "<missing>"
        except Exception:
            pass
        return "<unknown>"

    def _write_nonblocking(self, packed: bytes, command_type: str) -> bool:
        """
        Write packed bytes to the underlying channel without blocking.

        Priority:
          1) put_nowait
          2) put(..., block=False)
          3) send
          4) sendall

        Returns:
            True if a write method succeeded, else False.
        """
        try:
            ch = self._channel

            # Queue-like APIs
            put_nowait = getattr(ch, "put_nowait", None)
            if callable(put_nowait):
                put_nowait(packed)
                return True

            put = getattr(ch, "put", None)
            if callable(put):
                # Many Queue implementations accept block=False.
                try:
                    put(packed, block=False)
                    return True
                except TypeError:
                    # If signature differs, do not risk blocking.
                    self._log.error(
                        "Channel put() does not support block=False — dropped. type=%s channel=%s",
                        command_type,
                        type(ch).__name__,
                        extra={"dashboard_component": "command_router"},
                    )
                    return False

            # Socket-like APIs
            send = getattr(ch, "send", None)
            if callable(send):
                # Do not attempt to set timeouts here (side effects); caller should configure.
                send(packed)
                return True

            sendall = getattr(ch, "sendall", None)
            if callable(sendall):
                sendall(packed)
                return True

            self._log.error(
                "Unsupported command channel — missing put_nowait/put/send/sendall. "
                "type=%s channel=%s",
                command_type,
                type(ch).__name__,
                extra={"dashboard_component": "command_router"},
            )
            return False

        except _queue.Full as e:
            self._log.warning(
                "Command channel full — dropped. type=%s err=%s",
                command_type,
                repr(e),
                extra={"dashboard_component": "command_router"},
            )
            return False
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError, ValueError) as e:
            # Covers common IPC/socket failure modes and closed queue/pipe scenarios.
            self._log.error(
                "Command channel write failed — dropped. type=%s err=%s channel=%s",
                command_type,
                repr(e),
                type(self._channel).__name__ if self._channel is not None else "None",
                extra={"dashboard_component": "command_router"},
            )
            return False
        except Exception as e:
            # Never allow unexpected exceptions to crash the UI.
            self._log.error(
                "Unexpected command send error — dropped. type=%s err=%s",
                command_type,
                repr(e),
                extra={"dashboard_component": "command_router"},
            )
            return False


if __name__ == "__main__":
    from multiprocessing import Queue

    q: Queue = Queue()
    router = CommandRouter(q)

    # Test successful send
    ok = router.send_command({"test": "hello"})
    assert ok is True
    received = q.get(timeout=1)
    assert msgpack.unpackb(received, raw=False) == {"test": "hello"}

    # Test disabled router
    router.set_enabled(False)
    ok2 = router.send_command({"ignored": True})
    assert ok2 is False

    # Test with invalid channel (should not crash)
    bad_router = CommandRouter(None)
    ok3 = bad_router.send_command({"x": 1})
    assert ok3 is False

    print("command_router self-test passed")