from __future__ import annotations

"""Dashboard runtime facade.

This module restores the package surface expected by ``dashboard.__init__``:

- ``DashboardApp``: lightweight runtime wrapper around the dashboard registry
- ``build_dashboard_app``: convenience constructor for callers and tests
- ``format_dashboard_report``: human-readable snapshot summary for headless use

The implementation stays read-only and lazy-loads Qt integration so the package
remains importable in environments where PyQt6 is unavailable.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

try:
    from src.utils.config_loader import Config
except Exception:  # pragma: no cover
    Config = None  # type: ignore[assignment]

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)

from dashboard.panels import PanelRegistry, build_default_registry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass(slots=True)
class DashboardAppConfig:
    mode: str = "PAPER"
    replay_date: Optional[str] = None
    request_live: bool = False
    headless: bool = False
    panel_registry: PanelRegistry = field(default_factory=build_default_registry)
    config_path: str = "config.yaml"


def _build_performance_summary(panel_results: list[Any], registry_metrics: Dict[str, Any], render_ms: float) -> Dict[str, Any]:
    """Build a read-only performance profile for headless/dashboard reports.

    Week 08 Fri contract: expose render/chart pressure without doing backend
    work or inventing market data.  Values are derived from PanelResult timings,
    PanelRegistry rolling metrics, and panel-provided UI payload profiles.
    """
    slowest_panel_id: str | None = None
    slowest_panel_ms = 0.0
    total_panel_render_ms = 0.0
    chart_profiles: Dict[str, Any] = {}

    for result in panel_results:
        panel_id = str(getattr(result, "panel_id", "unknown"))
        try:
            render_value = float(getattr(result, "render_ms", 0.0) or 0.0)
        except Exception:
            render_value = 0.0
        total_panel_render_ms += render_value
        if render_value >= slowest_panel_ms:
            slowest_panel_ms = render_value
            slowest_panel_id = panel_id

        payload = getattr(result, "payload", {}) or {}
        if isinstance(payload, dict) and isinstance(payload.get("performance_profile"), dict):
            chart_profiles[panel_id] = dict(payload["performance_profile"])

    summary = registry_metrics.get("summary", {}) if isinstance(registry_metrics, dict) else {}
    return {
        "report_render_ms": round(float(max(0.0, render_ms)), 3),
        "panel_render": {
            "total_render_ms": round(total_panel_render_ms, 3),
            "avg_render_ms": round(total_panel_render_ms / max(len(panel_results), 1), 3),
            "slowest_panel_id": slowest_panel_id,
            "slowest_panel_ms": round(slowest_panel_ms, 3),
            "registry_summary": summary if isinstance(summary, dict) else {},
        },
        "chart_profiles": chart_profiles,
    }


class DashboardApp:
    """Read-only dashboard runtime wrapper.

    The class keeps the runtime contract small and testable. It can render a
    registry-backed summary without Qt, or create a Qt main window lazily when
    PyQt6 is available.
    """

    def __init__(
        self,
        config: DashboardAppConfig,
        *,
        snapshot_bus: Any | None = None,
        kill_switch_reader: Any | None = None,
        dashboard_clock: Any | None = None,
        qt_app: Any | None = None,
    ) -> None:
        self.config = config
        self.snapshot_bus = snapshot_bus
        self.kill_switch_reader = kill_switch_reader
        self.dashboard_clock = dashboard_clock
        self.qt_app = qt_app
        self.log = setup_logger("dashboard_app")

    @property
    def panel_registry(self) -> PanelRegistry:
        return self.config.panel_registry

    def render_once(self, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        snapshot = snapshot or {}
        now = _utc_now()
        started = time.perf_counter()
        panel_results = self.panel_registry.render_all(snapshot, now)
        render_ms = (time.perf_counter() - started) * 1000.0
        try:
            registry_metrics = self.panel_registry.get_metrics()
        except Exception:
            registry_metrics = {}
        performance = _build_performance_summary(panel_results, registry_metrics, render_ms)

        return {
            "timestamp": _iso(now),
            "mode": self.config.mode,
            "replay_date": self.config.replay_date,
            "request_live": self.config.request_live,
            "panel_count": len(panel_results),
            "performance": performance,
            "panels": [asdict(result) if hasattr(result, "__dataclass_fields__") else result for result in panel_results],
        }

    def build_window(self):
        try:
            from PyQt6.QtWidgets import QApplication
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyQt6 is required to build the dashboard window") from exc

        from dashboard.ui.main_window import MainWindow

        if self.qt_app is None:
            self.qt_app = QApplication.instance() or QApplication([])

        return MainWindow(
            snapshot_bus=self.snapshot_bus,
            kill_switch_reader=self.kill_switch_reader,
            dashboard_clock=self.dashboard_clock,
            mode=self.config.mode,
            panel_registry=self.panel_registry,
        )

    def run(self) -> int:
        if self.config.headless:
            report = self.render_once()
            print(format_dashboard_report(report))
            return 0

        window = self.build_window()
        window.show()
        if self.qt_app is None:
            return 0
        return int(self.qt_app.exec())


def build_dashboard_app(
    *,
    config_path: str = "config.yaml",
    mode: str = "PAPER",
    replay_date: Optional[str] = None,
    request_live: bool = False,
    headless: bool = False,
    panel_registry: Optional[PanelRegistry] = None,
    snapshot_bus: Any | None = None,
    kill_switch_reader: Any | None = None,
    dashboard_clock: Any | None = None,
    qt_app: Any | None = None,
) -> DashboardApp:
    if Config is not None:
        try:
            Config.load(config_path)
            loaded_mode = Config.get("dashboard", "mode", default=mode)
            mode = str(loaded_mode)
        except Exception:
            pass

    config = DashboardAppConfig(
        mode=mode,
        replay_date=replay_date,
        request_live=request_live,
        headless=headless,
        panel_registry=panel_registry or build_default_registry(),
        config_path=config_path,
    )
    return DashboardApp(
        config,
        snapshot_bus=snapshot_bus,
        kill_switch_reader=kill_switch_reader,
        dashboard_clock=dashboard_clock,
        qt_app=qt_app,
    )


def format_dashboard_report(report: Dict[str, Any]) -> str:
    timestamp = report.get("timestamp", "-")
    mode = report.get("mode", "-")
    replay_date = report.get("replay_date") or "LIVE"
    panel_count = report.get("panel_count", 0)

    lines = [
        "Junior Aladdin Dashboard Report",
        f"Timestamp: {timestamp}",
        f"Mode: {mode}",
        f"Replay: {replay_date}",
        f"Panels: {panel_count}",
    ]

    performance = report.get("performance")
    if isinstance(performance, dict):
        panel_perf = performance.get("panel_render", {}) if isinstance(performance.get("panel_render"), dict) else {}
        lines.append(
            "Performance: "
            f"report={float(performance.get('report_render_ms', 0.0) or 0.0):.3f}ms "
            f"avg_panel={float(panel_perf.get('avg_render_ms', 0.0) or 0.0):.3f}ms "
            f"slowest={panel_perf.get('slowest_panel_id', '-')}:"
            f"{float(panel_perf.get('slowest_panel_ms', 0.0) or 0.0):.3f}ms"
        )
        chart_profiles = performance.get("chart_profiles", {})
        if isinstance(chart_profiles, dict) and chart_profiles:
            for panel_id, profile in chart_profiles.items():
                if isinstance(profile, dict):
                    lines.append(
                        "ChartProfile: "
                        f"{panel_id} active={profile.get('active_candle_points', '-')} "
                        f"total={profile.get('total_candle_points', '-')} "
                        f"maxpoints={profile.get('maxpoints_limit', '-')} "
                        f"bounded={profile.get('dom_window_bounded', '-')}"
                    )

    for panel in report.get("panels", []):
        if not isinstance(panel, dict):
            continue
        panel_id = panel.get("panel_id", "unknown")
        title = panel.get("title", panel_id)
        status = panel.get("status", "UNKNOWN")
        lines.append(f"- {title} [{panel_id}] -> {status}")

    return "\n".join(lines)


__all__ = ["DashboardApp", "DashboardAppConfig", "build_dashboard_app", "format_dashboard_report"]