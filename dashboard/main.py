from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config_loader import Config
from src.utils.logger import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Junior Aladdin Dashboard")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--replay", type=str, default=None)
    parser.add_argument("--request-live", action="store_true")
    parser.add_argument("--out-file", type=str, default=None)
    return parser.parse_args()


def _try_import_qt() -> Dict[str, Any]:
    try:
        from PyQt6.QtCore import QObject, QCoreApplication, QTimer, pyqtSignal
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QApplication, QMainWindow

        return {
            "available": True,
            "QObject": QObject,
            "QCoreApplication": QCoreApplication,
            "QTimer": QTimer,
            "pyqtSignal": pyqtSignal,
            "QColor": QColor,
            "QPalette": QPalette,
            "QApplication": QApplication,
            "QMainWindow": QMainWindow,
        }
    except ImportError:
        return {"available": False}


ROADMAP_THEME_DEFAULTS: Dict[str, str] = {
    "window": "#0B0F19",
    "window_text": "#D8DEE9",
    "base": "#111827",
    "alternate_base": "#1F2937",
    "text": "#E5E7EB",
    "button": "#111827",
    "button_text": "#E5E7EB",
    "highlight": "#3B82F6",
    "highlighted_text": "#FFFFFF",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _paths() -> Dict[str, Path]:
    root = _project_root()
    return {
        "root": root,
        "pid": root / "data" / "dashboard.pid",
        "theme": root / "dashboard" / "assets" / "theme.json",
    }


def _headless_render(args: argparse.Namespace, log: Any) -> int:
    # Delegate headless rendering to the DashboardApp facade so headless mode
    # uses the PanelRegistry and the canonical report format.
    try:
        from dashboard.app import build_dashboard_app, format_dashboard_report

        app = build_dashboard_app(
            config_path=args.config,
            headless=True,
            replay_date=args.replay,
        )
        report = app.render_once()

        if args.out_file:
            out_path = Path(args.out_file).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        else:
            print(format_dashboard_report(report))
        return 0
    except Exception as exc:
        log.critical(
            "Headless render delegation failed",
            dashboard_component="main",
            error=str(exc),
        )
        return 1


class PidLock:
    def __init__(self, pid_file: Path):
        self.pid_file = pid_file
        self.handle: Optional[Any] = None
        self.method: Optional[str] = None

    def _load_portalocker(self) -> Any | None:
        try:
            return importlib.import_module("portalocker")
        except Exception:
            return None

    def acquire(self) -> bool:
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.pid_file.open("a+", encoding="utf-8")

        portalocker = self._load_portalocker()
        try:
            if portalocker is None:
                raise ImportError("portalocker unavailable")
            portalocker.lock(self.handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
            self.method = "portalocker"
        except Exception:
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                    self.method = "msvcrt"
                except OSError:
                    return False
            else:
                import fcntl

                try:
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.method = "fcntl"
                except OSError:
                    return False

        self.handle.seek(0)
        self.handle.truncate(0)
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            if self.method == "portalocker":
                portalocker = self._load_portalocker()
                if portalocker is None:
                    raise ImportError("portalocker unavailable")
                portalocker.unlock(self.handle)
            elif self.method == "msvcrt" and os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif self.method == "fcntl" and os.name != "nt":
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self.handle.close()
        except Exception:
            pass
        self.handle = None
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass


def _load_theme(log: Any, theme_path: Path) -> Dict[str, Any]:
    theme: Dict[str, Any] = {}
    if theme_path.exists():
        try:
            parsed = json.loads(theme_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                theme = parsed
        except Exception as exc:
            log.warning(
                "Theme file unreadable; using fallback",
                dashboard_component="main",
                error=str(exc),
            )

    palette = theme.get("palette", {}) if isinstance(theme.get("palette", {}), dict) else {}
    missing = [k for k in ROADMAP_THEME_DEFAULTS if k not in palette]
    if missing:
        log.warning(
            "Theme missing keys; applying fallback defaults",
            dashboard_component="main",
            missing_keys=missing,
        )

    merged_palette = {**ROADMAP_THEME_DEFAULTS, **palette}
    theme["palette"] = merged_palette
    if "stylesheet" not in theme or not isinstance(theme.get("stylesheet"), str):
        theme["stylesheet"] = ""
    return theme


def _apply_theme(app: Any, qt: Dict[str, Any], theme: Dict[str, Any]) -> None:
    QPalette = qt["QPalette"]
    QColor = qt["QColor"]

    palette = QPalette()
    role_map = {
        "window": QPalette.ColorRole.Window,
        "window_text": QPalette.ColorRole.WindowText,
        "base": QPalette.ColorRole.Base,
        "alternate_base": QPalette.ColorRole.AlternateBase,
        "text": QPalette.ColorRole.Text,
        "button": QPalette.ColorRole.Button,
        "button_text": QPalette.ColorRole.ButtonText,
        "highlight": QPalette.ColorRole.Highlight,
        "highlighted_text": QPalette.ColorRole.HighlightedText,
    }
    for key, role in role_map.items():
        palette.setColor(role, QColor(theme["palette"][key]))
    app.setPalette(palette)
    app.setStyleSheet(theme.get("stylesheet", ""))


def _check_backend_heartbeat(name: str, retries: int = 5, delay_s: float = 2.0) -> bool:
    from multiprocessing import shared_memory
    from threading import Event

    waiter = Event()
    for idx in range(retries):
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            return True
        except FileNotFoundError:
            if idx < retries - 1:
                waiter.wait(delay_s)
        except Exception:
            if idx < retries - 1:
                waiter.wait(delay_s)
        finally:
            if shm is not None:
                shm.close()
    return False


def _deny_live_request(log: Any) -> int:
    """Fail closed until the real Week-15 LIVE handshake exists.

    Pre-Week-8 stabilization (P0 safety fix): a previous local placeholder implied that a compliance handshake was
    available.  That is unacceptable for
    a trading control plane because it can make LIVE enablement appear partially
    available.  Until the roadmap's Week-15 live_mode_handshake.py is built,
    every dashboard-side live request is explicitly denied and logged.
    """
    log.critical(
        "LIVE mode request denied; live handshake is not implemented yet",
        dashboard_component="main",
        roadmap_week="15",
        action="fail_closed",
    )
    return 2

def _setup_signal_handlers(cleanup) -> None:
    def _handler(_signum, _frame) -> None:
        cleanup()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> int:
    args = _parse_args()
    Config.load(args.config)
    log = setup_logger("dashboard_main")
    log.info(
        "Dashboard start",
        dashboard_component="main",
        config=args.config,
        headless=args.headless,
        replay=args.replay,
        request_live=args.request_live,
        out_file=args.out_file,
    )

    paths = _paths()
    pid_lock = PidLock(paths["pid"])
    if not pid_lock.acquire():
        log.critical("PID lock acquisition failed", dashboard_component="main", pid_file=str(paths["pid"]))
        return 1

    if args.request_live:
        # Fail closed before headless rendering or Qt startup so no code path can
        # imply LIVE enablement before the roadmap's Week-15 handshake exists.
        code = _deny_live_request(log)
        pid_lock.release()
        return code

    qt = _try_import_qt()
    if not qt["available"] and not args.headless:
        log.critical("PyQt6 unavailable for non-headless mode", dashboard_component="main")
        pid_lock.release()
        return 1
    if not qt["available"] and args.headless:
        try:
            return _headless_render(args, log)
        finally:
            pid_lock.release()

    heartbeat_name = Config.get("dashboard", "backend_heartbeat_name", default="ja_backend_heartbeat")
    backend_ok = _check_backend_heartbeat(str(heartbeat_name), retries=5, delay_s=2.0)
    if not backend_ok:
        log.warning("Backend heartbeat missing; running in degraded mode", dashboard_component="main", heartbeat_name=heartbeat_name)

    if args.headless:
        try:
            return _headless_render(args, log)
        finally:
            pid_lock.release()

    QApplication = qt["QApplication"]
    QTimer = qt["QTimer"]

    # CRITICAL FIX: QtWebEngine requires AA_ShareOpenGLContexts BEFORE QApplication is created.
    # Without this, MtfChart/QWebEngineView import fails with:
    # "QtWebEngineWidgets must be imported or Qt.AA_ShareOpenGLContexts must be set
    #  before a QCoreApplication instance is created"
    try:
        from PyQt6.QtCore import Qt
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    except Exception as exc:
        log.warning("Could not set AA_ShareOpenGLContexts", dashboard_component="main", error=str(exc))

    app = QApplication(sys.argv)
    app.setApplicationName("JuniorAladdinDashboard")
    app.setOrganizationName("JuniorAladdin")

    theme = _load_theme(log, paths["theme"])
    _apply_theme(app, qt, theme)

    # Import real UI classes from their modules.  Safety-critical dashboard
    # dependencies are hard imports: import failure must fail fast instead of
    # silently degrading to local stubs.
    from dashboard.ui.main_window import MainWindow
    from dashboard.ui.status_strip import StatusStrip

    # Hard import: PyQt6 has already been verified available above (non-headless
    # path returns early when qt["available"] is False). Any failure here is a
    # genuine integration bug and must surface immediately rather than degrade
    # to a divergent stub implementation. (Addresses HIGH-2.)
    from dashboard.core.dashboard_clock import DashboardClock

    # Keep import explicit to ensure status strip module resolves at runtime.
    _ = StatusStrip

    # Hard import: SnapshotBus is the decoded truth channel.  A local fallback
    # stub would violate the roadmap's "never guess" rule and can hide backend
    # integration failures, so import errors must surface immediately.
    from dashboard.core.snapshot_bus import SnapshotBus

    # Hard import: PyQt6 has already been verified above. A divergent stub here
    # would let the dashboard run while silently disabling the kill-switch — a
    # safety-critical regression. Fail fast instead. (Addresses HIGH-B.)
    from dashboard.core.shared_memory_kill_switch import KillSwitchReader
    from dashboard.core.command_router import CommandRouter
    from dashboard.core.ipc_client import LengthPrefixedSocketCommandChannel, SnapshotStreamClient

    kill_switch_name = Config.get("dashboard", "kill_switch_name", default="junior_aladdin_kill_switch")
    kill_switch_reader = KillSwitchReader(str(kill_switch_name))

    ipc_host = str(Config.get("dashboard", "ipc_host", default="127.0.0.1"))
    snapshot_port = int(Config.get("dashboard", "snapshot_port", default=18765))
    command_port = int(Config.get("dashboard", "command_port", default=18766))

    # Part 2A real command channel:
    # The dashboard now uses a length-prefixed TCP socket wrapper instead of a
    # local Queue so commands can actually reach the backend process.
    command_channel = LengthPrefixedSocketCommandChannel(
        host=ipc_host,
        port=command_port,
    )
    command_router = CommandRouter(
        command_channel=command_channel,
        enabled=not bool(args.replay),
    )
    if args.replay:
        # Replay safety: operator commands must never leak into backend control
        # paths during historical/replay runs.  The router remains injected so
        # UI code follows the same path, but CommandRouter will log-and-ignore
        # sends while disabled.
        log.info("CommandRouter disabled for replay mode", dashboard_component="main")
    snapshot_bus = SnapshotBus()
    if args.replay:
        setattr(snapshot_bus, "replay_date", args.replay)

    snapshot_client = None
    if not args.replay:
        snapshot_client = SnapshotStreamClient(
            host=ipc_host,
            port=snapshot_port,
            frame_handler=snapshot_bus.feed_bytes,
        )

    dashboard_clock = DashboardClock(
        hot_interval_ms=int(Config.get("dashboard", "hot_interval_ms", default=200)),
        warm_interval_ms=int(Config.get("dashboard", "warm_interval_ms", default=1000)),
        cold_interval_ms=int(Config.get("dashboard", "cold_interval_ms", default=5000)),
    )
    main_window = MainWindow(
        snapshot_bus=snapshot_bus,
        kill_switch_reader=kill_switch_reader,
        dashboard_clock=dashboard_clock,
        mode="ALERT",
        panel_registry=None,
        command_router=command_router,
    )

    snapshot_bus.new_hot_frame.connect(main_window.update_hot)
    snapshot_bus.new_warm_frame.connect(main_window.update_warm)
    snapshot_bus.new_cold_frame.connect(main_window.update_cold)
    kill_switch_reader.emergency_activated.connect(main_window.on_emergency)

    # Tier ticks drive CONSUMER-side refresh, not producer-side emission.
    # MainWindow.on_*_tick pulls SnapshotBus.last_valid_*_payload() and updates
    # panels. SnapshotBus stays push-only — its new_*_frame signals fire only
    # when real backend frames are decoded. This preserves the roadmap PRIMARY
    # RULE: dashboard never fabricates state. (CRIT-A fix; replaces the
    # earlier wiring that called fabricator emit_hot/emit_warm/emit_cold.)
    dashboard_clock.hot_tick.connect(main_window.on_hot_tick)
    dashboard_clock.warm_tick.connect(main_window.on_warm_tick)
    dashboard_clock.cold_tick.connect(main_window.on_cold_tick)

    kill_timer = QTimer()
    kill_timer.setInterval(200)
    kill_timer.timeout.connect(kill_switch_reader.check)
    kill_timer.start()

    snapshot_bus.start()
    if snapshot_client is not None:
        snapshot_client.start()
    dashboard_clock.start()
    main_window.show()

    cleaned = {"done": False}

    def _cleanup() -> None:
        if cleaned["done"]:
            return
        cleaned["done"] = True
        try:
            kill_timer.stop()
        except Exception:
            pass
        for obj, name in ((snapshot_client, "snapshot_client"), (snapshot_bus, "snapshot_bus"), (kill_switch_reader, "kill_switch_reader"), (dashboard_clock, "dashboard_clock"), (command_channel, "command_channel")):
            if obj is None:
                continue
            try:
                obj.stop() if hasattr(obj, "stop") else obj.close()
            except Exception as exc:
                log.warning("Guarded stop failed", dashboard_component="main", object_name=name, error=str(exc))
        pid_lock.release()
        try:
            app.quit()
        except Exception:
            pass

    _setup_signal_handlers(_cleanup)

    exit_code = 1
    try:
        if hasattr(app, "exec_"):
            exit_code = app.exec_()  # pragma: no cover
        else:
            exit_code = app.exec()
    finally:
        _cleanup()
        log.info("Dashboard shutdown", dashboard_component="main", exit_code=exit_code)

    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())