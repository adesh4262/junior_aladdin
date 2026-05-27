"""
Junior Aladdin - Logger Utility (Institutional-Grade)

Backward-compatible API:
- setup_logger(name: str, level: str|int = None, log_dir: str = "logs",
              max_bytes: int = 10*1024*1024, backup_count: int = 5,
              file_name: str = None) -> logger-like
- set_logger_level(name: str, level: str) -> bool

Institutional Guarantees:
- Thread-safe logger cache (RLock)
- Disk-safe: graceful degradation if file handler cannot be created (console-only)
- Rotating file logs by default (RotatingFileHandler)
  - If max_bytes == 0 => plain FileHandler (backward compatibility mode)
- No structlog dependency
- Supports structlog-like call patterns: logger.info("msg", key=value)
  by capturing arbitrary kwargs and storing them as record extras.

Log Formats:
- Console: human readable with key=value pairs
- File: JSON lines containing timestamp, level, logger, message, and extras
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

_logger_cache: Dict[str, "BoundLogger"] = {}
_cache_lock = threading.RLock()


# ------------------------- Formatters -------------------------

_STANDARD_LOG_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process",
    "message",  # populated by Formatter
}


class JsonLineFormatter(logging.Formatter):
    """JSON lines formatter with extras included. Never raises."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            # ensure record.message computed
            record.message = record.getMessage()
            ts = datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds")

            payload: Dict[str, Any] = {
                "timestamp": ts,
                "level": record.levelname,
                "logger": record.name,
                "message": record.message,
            }

            # Attach extras
            extras = {}
            for k, v in record.__dict__.items():
                if k in _STANDARD_LOG_RECORD_ATTRS:
                    continue
                # avoid non-serializable values
                try:
                    json.dumps(v)
                    extras[k] = v
                except Exception:
                    extras[k] = repr(v)
            if extras:
                payload["extra"] = extras

            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)

            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            # Last resort: don't crash logging
            try:
                return json.dumps(
                    {
                        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                        "level": "ERROR",
                        "logger": getattr(record, "name", "unknown"),
                        "message": "JsonLineFormatter failure",
                    },
                    ensure_ascii=False,
                )
            except Exception:
                return '{"level":"ERROR","message":"JsonLineFormatter failure"}'


class ConsoleFormatter(logging.Formatter):
    """Human console formatter with extras appended as key=value. Never raises."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            record.message = record.getMessage()
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            base = f"{ts} | {record.levelname:<8} | {record.name} | {record.message}"

            extras = []
            for k, v in record.__dict__.items():
                if k in _STANDARD_LOG_RECORD_ATTRS:
                    continue
                try:
                    extras.append(f"{k}={v!r}")
                except Exception:
                    extras.append(f"{k}=<unrepr>")

            if extras:
                base += " | " + ", ".join(extras)

            if record.exc_info:
                base += "\n" + self.formatException(record.exc_info)

            return base
        except Exception:
            return "LOGGER_FORMAT_ERROR"


# ------------------------- Logger wrapper -------------------------

class BoundLogger:
    """
    Logger wrapper that:
    - accepts arbitrary **fields in .info/.warning/etc
    - forwards unknown attributes to underlying logging.Logger
    - stores fields into LogRecord extras so formatters can render them
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __getattr__(self, item: str) -> Any:
        return getattr(self._logger, item)

    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Accepts structlog-like kwargs and maps them into LogRecord.extra.
        Reserved logging kwargs: exc_info, stack_info, stacklevel, extra
        """
        try:
            reserved = {"exc_info", "stack_info", "stacklevel", "extra"}
            fields = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in reserved}

            extra = kwargs.get("extra")
            if extra is None or not isinstance(extra, dict):
                extra = {}
            else:
                extra = dict(extra)

            # Merge fields into extra, without overwriting existing keys unless necessary
            for k, v in fields.items():
                extra[k] = v
            kwargs["extra"] = extra

            self._logger.log(level, msg, *args, **kwargs)
        except Exception:
            # Logging must never crash trading system
            try:
                self._logger.log(level, msg)
            except Exception:
                return

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        # Ensure exc_info is present
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)


# ------------------------- Public API -------------------------

def setup_logger(
    name: str,
    level: Optional[Union[str, int]] = None,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    file_name: Optional[str] = None,
) -> BoundLogger:
    """
    Get or create a cached logger for a given engine/module name.

    Args:
        name: logger name
        level: log level (e.g., "INFO") or int (e.g., logging.INFO). If None, defaults to INFO.
        log_dir: directory for log files
        max_bytes: RotatingFileHandler maxBytes. If 0 -> use plain FileHandler.
        backup_count: number of rotated backups to keep.

    Returns:
        BoundLogger wrapper (supports logger.info("msg", key=value) safely).
    """
    if not isinstance(name, str) or not name.strip():
        name = "junior_aladdin"
    name = name.strip()

    with _cache_lock:
        cached = _logger_cache.get(name)
        if cached is not None:
            # allow level override on retrieval
            if level is not None:
                try:
                    _apply_level(cached._logger, level)
                except Exception:
                    pass
            return cached

    # Create new logger (outside cache lock, but safe; we'll cache atomically)
    logger = logging.getLogger(f"junior_aladdin.{name}" if not name.startswith("junior_aladdin.") else name)
    logger.propagate = False

    # Determine base level
    _apply_level(logger, level if level is not None else "INFO")

    # Prevent duplicate handler setup if logger already has handlers (possible if created elsewhere)
    # Still wrap it.
    if not getattr(logger, "_junior_aladdin_configured", False):
        _configure_handlers(
            logger,
            name=name,
            log_dir=log_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
            file_name=file_name,
        )
        setattr(logger, "_junior_aladdin_configured", True)

    bound = BoundLogger(logger)

    with _cache_lock:
        # If another thread created in the meantime, prefer existing and drop this one
        existing = _logger_cache.get(name)
        if existing is not None:
            return existing
        _logger_cache[name] = bound
        return bound


def set_logger_level(name: str, level: str) -> bool:
    """
    Update level of an existing cached logger and its handlers.

    Returns:
        True if logger existed and was updated, else False.
    """
    if not isinstance(name, str) or not name.strip():
        return False
    name = name.strip()

    with _cache_lock:
        bound = _logger_cache.get(name)
    if bound is None:
        return False

    try:
        _apply_level(bound._logger, level)
        for h in list(bound._logger.handlers):
            try:
                h.setLevel(bound._logger.level)
            except Exception:
                continue
        bound.info("Logger level updated at runtime", new_level=str(level))
        return True
    except Exception:
        return False


# ------------------------- Internal helpers -------------------------

def _apply_level(logger: logging.Logger, level: Union[str, int]) -> None:
    if isinstance(level, int):
        logger.setLevel(level)
        return
    if isinstance(level, str):
        lvl = level.strip().upper()
        logger.setLevel(getattr(logging, lvl, logging.INFO))
        return
    logger.setLevel(logging.INFO)


def _configure_handlers(
    logger: logging.Logger,
    name: str,
    log_dir: str,
    max_bytes: int,
    backup_count: int,
    file_name: Optional[str] = None,
) -> None:
    # Always add console handler (safe)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # File handler is best-effort
    try:
        if file_name is not None and str(file_name).strip():
            file_path = Path(log_dir) / str(file_name).strip()
        else:
            file_path = Path(log_dir) / f"{name}.log"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if max_bytes and max_bytes > 0:
            file_handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=int(max_bytes),
                backupCount=int(max(0, backup_count)),
                encoding="utf-8",
                delay=True,
            )
        else:
            file_handler = logging.FileHandler(
                filename=str(file_path),
                encoding="utf-8",
                delay=True,
            )

        file_handler.setLevel(logger.level)
        file_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)

        # Emit one-time config message
        logger.info(
            "File logging enabled",
            extra={
                "file_path": str(file_path),
                "max_bytes": int(max_bytes),
                "backup_count": int(backup_count),
            },
        )
    except (OSError, PermissionError, IOError) as e:
        # Graceful degradation: console-only
        try:
            logger.warning(
                "File logging disabled due to file handler error; running console-only",
                extra={"error": repr(e), "log_dir": str(log_dir), "logger_name": name},
            )
        except Exception:
            # even if logger fails, do not crash
            pass
    except Exception as e:
        try:
            logger.warning(
                "File logging disabled due to unexpected error; running console-only",
                extra={"error": repr(e), "log_dir": str(log_dir), "logger_name": name},
            )
        except Exception:
            pass


# ------------------------- Self-test -------------------------

def _self_test_rotation() -> None:
    test_name = "logger_rotation_test"
    log_dir = "logs"
    # Use tiny rotation threshold to force rotate quickly
    log = setup_logger(test_name, level="INFO", log_dir=log_dir, max_bytes=1024, backup_count=2)

    # Write enough to rotate
    for i in range(200):
        log.info("X" * 200, i=i)

    base = Path(log_dir) / f"{test_name}.log"
    rotated1 = Path(log_dir) / f"{test_name}.log.1"
    rotated2 = Path(log_dir) / f"{test_name}.log.2"

    assert base.exists(), f"Base log file not found: {base}"
    # rotation should have created at least one rotated file
    assert rotated1.exists() or rotated2.exists(), "Rotation did not create backup files as expected"


def _self_test_thread_safety() -> None:
    import threading as _th

    results: Dict[int, int] = {}
    errors: List[str] = []
    name = "logger_thread_test"

    def worker(idx: int) -> None:
        try:
            lg = setup_logger(name, level="DEBUG")
            lg.debug("thread logger acquired", idx=idx)
            results[idx] = id(lg)
        except Exception as e:
            errors.append(repr(e))

    threads = [_th.Thread(target=worker, args=(i,), daemon=True) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"Errors during threaded setup_logger: {errors}"
    # all ids should match due to cache
    ids = set(results.values())
    assert len(ids) == 1, f"Expected single cached logger instance, got {len(ids)} instances"


if __name__ == "__main__":
    print("Running logger self-test...")

    # Basic smoke
    log = setup_logger("self_test", level="INFO", max_bytes=2048, backup_count=1)
    log.info("Logger self-test start", component="logger")

    # Rotation
    _self_test_rotation()
    log.info("Rotation test passed")

    # Thread safety
    _self_test_thread_safety()
    log.info("Thread safety test passed")

    # Runtime level change
    ok = set_logger_level("self_test", "DEBUG")
    assert ok is True, "set_logger_level failed to update existing logger"
    log.debug("Runtime level change verified", new_level="DEBUG")

    print("Logger self-test PASSED.")