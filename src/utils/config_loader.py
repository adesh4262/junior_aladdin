"""
Junior Aladdin - Config Loader (Institutional-Grade)

Backward-compatible singleton-style config loader for config.yaml with:

1) Thread safety:
   - Class-level RLock
   - Every access to _config and _config_path is guarded by the lock

2) Deep copy:
   - get_all() returns copy.deepcopy(_config)

3) Environment variable overrides (optional, enabled by default):
   - For Config.get("market","lot_size") checks:
       JUNIOR_ALADDIN_MARKET_LOT_SIZE
   - For nested: Config.get("brains","structural","max_trades_per_day") checks:
       JUNIOR_ALADDIN_BRAINS_STRUCTURAL_MAX_TRADES_PER_DAY
   - Type coercion: bool -> int -> float -> string
   - Controlled via class flag: _env_override_enabled

4) Config validation:
   - validate() checks required top-level sections
   - Called automatically at end of load() and reload()
   - Missing optional sections only emit a WARNING

5) Reload logging:
   - INFO log with config version if present

6) get_section():
   - Returns None if section does not exist (not empty dict)

API remains backward-compatible:
- load(path="config.yaml")
- reload()
- get(*keys, default=None)
- get_section(section)
- get_all()
- is_loaded()
"""

from __future__ import annotations

import copy
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from src.utils.logger import setup_logger

_LOG = setup_logger("config_loader")


def _emit_log(level: str, msg: str, **fields: Any) -> None:
    """Safe logger shim supporting structlog-style and stdlib logging.Logger. Never raises."""
    try:
        fn = getattr(_LOG, level, None)
        if fn is None:
            return
        if not fields:
            fn(msg)
            return
        try:
            fn(msg, **fields)
            return
        except TypeError:
            parts = []
            for k in sorted(fields.keys()):
                try:
                    parts.append(f"{k}={fields[k]!r}")
                except Exception:
                    parts.append(f"{k}=<unrepr>")
            fn(f"{msg} | " + ", ".join(parts))
    except Exception:
        return


# --- Tiny assertion helpers for __main__ self-test (must be defined before use) ---

def _assert_true(x: Any) -> None:
    assert bool(x) is True, f"Expected True, got {x!r}"


def _assert_eq(a: Any, b: Any) -> None:
    assert a == b, f"Expected {b!r}, got {a!r}"


def _assert_is_none(x: Any) -> None:
    assert x is None, f"Expected None, got {x!r}"


class Config:
    """
    Config singleton with backward-compatible API.

    Public methods:
    - load(path="config.yaml") -> dict
    - reload() -> dict
    - get(*keys, default=None) -> Any
    - get_section(section: str) -> Optional[dict]
    - get_all() -> dict
    - is_loaded() -> bool
    - validate(required_sections: Optional[Tuple[str,...]] = None) -> None
    """

    _lock = threading.RLock()
    _config: Optional[Dict[str, Any]] = None
    _config_path: Optional[str] = None
    _env_override_enabled: bool = True

    _required_sections_default: Tuple[str, ...] = (
        "system",
        "market",
        "data",
        "features",
        "risk",
        "compliance",
    )

    _optional_sections_default: Tuple[str, ...] = (
        "sessions",
        "narrative",
        "regime",
        "brains",
        "scoring",
        "trap",
        "ml",
        "behavioral",
        "costs",
        "position_management",
        "learning",
        "captain",
        "dashboard",
    )

    @classmethod
    def is_loaded(cls) -> bool:
        with cls._lock:
            return cls._config is not None

    @classmethod
    def load(cls, path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
        """
        Load YAML config from disk if not loaded. If already loaded from same path,
        returns cached config.

        Raises:
            FileNotFoundError: if file missing
            ValueError: if YAML invalid or required sections missing
        """
        p = str(path)

        with cls._lock:
            if cls._config is not None and cls._config_path == p:
                return cls._config

        config = cls._read_yaml(p)

        with cls._lock:
            cls._config = config
            cls._config_path = p

        cls.validate()

        version = cls.get("system", "version", default=None)
        if version is None:
            version = cls.get("version", default=None)

        _emit_log(
            "info",
            "Config loaded successfully",
            path=p,
            sections=list(config.keys())[:100],
            version=version,
        )
        return config

    @classmethod
    def reload(cls) -> Dict[str, Any]:
        """
        Force reload from last loaded path (or default config.yaml if never loaded).
        Raises the same exceptions as load().
        """
        with cls._lock:
            p = cls._config_path or "config.yaml"

        config = cls._read_yaml(p)

        with cls._lock:
            cls._config = config
            cls._config_path = p

        cls.validate()

        version = cls.get("system", "version", default=None)
        if version is None:
            version = cls.get("version", default=None)

        _emit_log("info", "Config reloaded successfully", path=p, version=version)
        return config

    @classmethod
    def get(cls, *keys: str, default: Any = None) -> Any:
        """
        Navigate nested config keys safely.

        Examples:
            Config.get("market", "lot_size") -> 65
            Config.get("brains", "structural", "max_trades_per_day") -> 3
            Config.get("fake", "key", default=0) -> 0

        Environment overrides (if enabled):
            JUNIOR_ALADDIN_{KEYPATH...} (uppercase, dots -> underscores)
        """
        with cls._lock:
            loaded = cls._config is not None
            p = cls._config_path
            env_enabled = cls._env_override_enabled

        if not loaded:
            cls.load(p or "config.yaml")

        if env_enabled and keys:
            env_name = cls._env_key_for_path(keys)
            env_val = os.getenv(env_name)
            if env_val is not None:
                return cls._coerce_env_value(env_val)

        with cls._lock:
            cfg = cls._config or {}
            node: Any = cfg

            if not keys:
                return copy.deepcopy(node)

            for k in keys:
                if not isinstance(node, dict):
                    return default
                if k not in node:
                    return default
                node = node.get(k)

            if isinstance(node, (dict, list)):
                return copy.deepcopy(node)
            return node

    @classmethod
    def get_section(cls, section: str) -> Optional[Dict[str, Any]]:
        """
        Return a deep-copied dict for a top-level section, or None if section is missing.
        """
        if not isinstance(section, str) or not section.strip():
            return None
        sec = section.strip()

        with cls._lock:
            loaded = cls._config is not None
            p = cls._config_path
        if not loaded:
            cls.load(p or "config.yaml")

        with cls._lock:
            if cls._config is None:
                return None
            node = cls._config.get(sec)
            if node is None or not isinstance(node, dict):
                return None
            return copy.deepcopy(node)

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Return a deep copy of the full config dict."""
        with cls._lock:
            loaded = cls._config is not None
            p = cls._config_path
        if not loaded:
            cls.load(p or "config.yaml")

        with cls._lock:
            return copy.deepcopy(cls._config or {})

    @classmethod
    def validate(cls, required_sections: Optional[Tuple[str, ...]] = None) -> None:
        """
        Validate required top-level sections. Raises ValueError if missing.
        Optional sections emit warning logs only.
        """
        with cls._lock:
            cfg = cls._config

        if cfg is None:
            raise ValueError("Config not loaded; cannot validate.")
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping/dict.")

        required = required_sections or cls._required_sections_default
        missing_required = [s for s in required if s not in cfg]
        if missing_required:
            raise ValueError(f"Config missing required top-level sections: {missing_required}")

        missing_optional = [s for s in cls._optional_sections_default if s not in cfg]
        if missing_optional:
            _emit_log("warning", "Config missing optional sections", missing=missing_optional)

    # ------------------------- Internal helpers -------------------------

    @classmethod
    def _read_yaml(cls, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read config file: {path}. Error: {e}") from e

        try:
            parsed = yaml.safe_load(raw)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config: {path}. Error: {e}") from e

        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError(f"Config YAML root must be a mapping/dict: {path}")
        return parsed

    @classmethod
    def _env_key_for_path(cls, keys: Tuple[str, ...]) -> str:
        parts = ["JUNIOR_ALADDIN"]
        for k in keys:
            kk = str(k).strip().upper().replace(".", "_")
            kk = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in kk)
            parts.append(kk)
        return "_".join(parts)

    @classmethod
    def _coerce_env_value(cls, s: str) -> Any:
        val = s.strip()
        if val == "":
            return ""

        low = val.lower()
        if low in {"true", "yes", "on"}:
            return True
        if low in {"false", "no", "off"}:
            return False

        # int
        try:
            if low.startswith(("+", "-")):
                if low[1:].isdigit():
                    return int(low)
            elif low.isdigit():
                return int(low)
        except Exception:
            pass

        # float
        try:
            f = float(val)
            if math.isfinite(f):
                return f
        except Exception:
            pass

        return val


if __name__ == "__main__":
    # Self-test: 8 checks (no pytest dependency)
    import tempfile
    import textwrap

    def _run_test(name: str, fn) -> None:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            raise
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            raise

    good_yaml = textwrap.dedent(
        """
        system:
          name: "Junior Aladdin"
          version: "12.0"
        market:
          lot_size: 65
          market_open: "09:15"
        data:
          feed_delay_threshold_ms: 1000
        features:
          rsi_period: 14
        risk:
          max_daily_loss_pct: 0.02
        compliance:
          allowed_order_types: ["LIMIT", "STOPLOSS_LIMIT"]
        """
    ).strip()

    bad_yaml_missing_required = textwrap.dedent(
        """
        system:
          version: "12.0"
        market:
          lot_size: 65
        """
    ).strip()

    def deep_copy_test() -> None:
        cfg1 = Config.get_all()
        cfg1["market"]["lot_size"] = 999
        _assert_eq(Config.get("market", "lot_size"), 65)

    def env_override_test() -> None:
        os.environ["JUNIOR_ALADDIN_MARKET_LOT_SIZE"] = "75"
        try:
            v = Config.get("market", "lot_size")
            _assert_eq(v, 75)
        finally:
            os.environ.pop("JUNIOR_ALADDIN_MARKET_LOT_SIZE", None)

    def reload_test(path: Path) -> None:
        original = path.read_text(encoding="utf-8")
        updated = original.replace("lot_size: 65", "lot_size: 66")
        path.write_text(updated, encoding="utf-8")
        Config.reload()
        _assert_eq(Config.get("market", "lot_size"), 66)
        path.write_text(original, encoding="utf-8")
        Config.reload()
        _assert_eq(Config.get("market", "lot_size"), 65)

    def validate_fail_test(bad_path: Path) -> None:
        with Config._lock:
            Config._config = None
            Config._config_path = None
        try:
            Config.load(bad_path)
            raise AssertionError("Expected ValueError for missing required sections")
        except ValueError:
            pass

    with tempfile.TemporaryDirectory() as td:
        good_path = Path(td) / "config.yaml"
        good_path.write_text(good_yaml, encoding="utf-8")

        bad_path = Path(td) / "bad.yaml"
        bad_path.write_text(bad_yaml_missing_required, encoding="utf-8")

        # reset state
        with Config._lock:
            Config._config = None
            Config._config_path = None
            Config._env_override_enabled = True

        _run_test("1) load() loads config", lambda: (Config.load(good_path), True))
        _run_test("2) is_loaded() returns True", lambda: _assert_true(Config.is_loaded()))
        _run_test("3) get() reads nested values", lambda: _assert_eq(Config.get("market", "lot_size"), 65))
        _run_test("4) get_section() returns None for missing section", lambda: _assert_is_none(Config.get_section("does_not_exist")))
        _run_test("5) get_all() returns deep copy", deep_copy_test)
        _run_test("6) env override works with type coercion", env_override_test)
        _run_test("7) reload() logs and preserves API", lambda: reload_test(good_path))
        _run_test("8) validate() fails on missing required sections", lambda: validate_fail_test(bad_path))

    print("\nAll 8 tests PASSED.")