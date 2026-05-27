from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from dashboard.dialogs import EmergencyStopDialog
from dashboard.panels import build_default_registry
from dashboard.ui.status_strip import StatusStrip

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


class MainWindow(QMainWindow):
    def __init__(
        self,
        snapshot_bus: Any,
        kill_switch_reader: Any,
        dashboard_clock: Any,
        mode: str,
        panel_registry: Optional[Any] = None,
        command_router: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.log = setup_logger("dashboard_ui_mainwindow")

        self.snapshot_bus = snapshot_bus
        self.kill_switch_reader = kill_switch_reader
        self.dashboard_clock = dashboard_clock
        self.command_router = command_router
        self.mode = mode
        self.panel_registry = panel_registry or build_default_registry()
        self._panel_labels: Dict[str, QLabel] = {}
        self._panel_widgets: Dict[str, QWidget] = {}
        self._last_panel_summary: Dict[str, Any] = {}

        self.setWindowTitle(self._build_title())
        self.resize(1400, 900)

        self._build_toolbar()
        self._build_central_tabs()
        self._build_status_strip()

        # Week 04 visible-artifact closeout:
        # Render the registered panels once with an empty snapshot so developer
        # preview mode immediately shows DEGRADED/missing-state information
        # instead of static placeholders. This does not fabricate backend data;
        # it exposes that backend data is missing, which matches the dashboard
        # roadmap primary rule.
        initial_summary = self._render_registered_panels({})
        self._update_status_strip(
            feed_health="STALE",
            system_state="WAITING_FOR_BACKEND",
            mode=self.mode,
            backend_state="DISCONNECTED",
            panel_status_summary=initial_summary,
        )

        self.log.info("Main window created", dashboard_component="ui_mainwindow", mode=mode)

    def _build_title(self) -> str:
        suffix = " (REPLAY)" if getattr(self.snapshot_bus, "replay_date", None) else ""
        return f"Junior Aladdin Dashboard{suffix}"

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Cockpit Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        emergency_action = QAction("Emergency Stop", self)
        emergency_action.triggered.connect(self._trigger_emergency)
        toolbar.addAction(emergency_action)

        toolbar.addSeparator()

        self.live_request_action = QAction("Request LIVE", self)
        # Pre-Week-9 safety clarity: the real LIVE handshake is roadmap Week 15.
        # Until then this control is explicitly disabled instead of clickable
        # theater.  CLI --request-live already fail-closes in dashboard/main.py.
        self.live_request_action.setEnabled(False)
        self.live_request_action.setToolTip("LIVE handshake is disabled until roadmap Week 15 compliance wiring.")
        toolbar.addAction(self.live_request_action)

        toolbar.addSeparator()

        self.mode_label = QLabel(f"Mode: {self.mode}")
        toolbar.addWidget(self.mode_label)

    def _build_central_tabs(self) -> None:
        """Build grouped cockpit tab layout.

        Sections:
          COCKPIT — STATUS, BRIEFING, GLOBAL VITALS, SYSTEM HEALTH
          MARKETS — MTF CHART, VOLUME PROFILE, OPTION CHAIN, MARKET OVERVIEW
          SYSTEMS — FEATURE SNAPSHOT, OPPORTUNITY PIPELINE, BRAIN STATUS, ENGINE HEALTH
          RISK    — POSITIONS, SESSION STATE

        Duplicate panels removed in Week 9 cockpit cleanup:
        - SystemStatusPanel (duplicated real STATUS widget)
        - NarrativePanel/Morning Briefing (duplicated real BRIEFING widget)
        """
        cockpit_panels = {
            "status": "COCKPIT",
            "briefing": "COCKPIT",
            "global_vitals": "COCKPIT",
            "system_health": "COCKPIT",
            "mtf_chart": "MARKETS",
            "volume_profile": "MARKETS",
            "option_chain": "MARKETS",
            "market_overview": "MARKETS",
            "feature_snapshot": "SYSTEMS",
            "opportunity_pipeline": "SYSTEMS",
            "brain_status": "SYSTEMS",
            "engine_health": "SYSTEMS",
            "positions": "RISK",
            "session_state": "RISK",
        }

        section_order = ["COCKPIT", "MARKETS", "SYSTEMS", "RISK"]

        self._panel_labels = {}
        self._panel_widgets = {}

        panels = []
        try:
            panels = list(self.panel_registry.list_panels())
        except Exception as exc:
            self.log.error(
                "Panel registry list failed",
                dashboard_component="ui_mainwindow",
                error=str(exc),
            )
            panels = []

        if not panels:
            self.setCentralWidget(QTabWidget(self))
            return

        # Build section map: section_name -> list of panels
        sections: Dict[str, QTabWidget] = {}
        section_panels: Dict[str, list] = {s: [] for s in section_order}

        for panel in panels:
            panel_id = str(getattr(panel, "panel_id", "unknown"))
            section_name = cockpit_panels.get(panel_id, "SYSTEMS")
            section_panels.setdefault(section_name, []).append(panel)

        # Build top-level tab widget
        self._main_tabs = QTabWidget(self)
        self._main_tabs.setDocumentMode(True)

        for section_name in section_order:
            section_panels_list = section_panels.get(section_name, [])
            if not section_panels_list:
                continue

            # Create a sub-tab widget for this section
            sub_tabs = QTabWidget()
            sub_tabs.setDocumentMode(True)
            sections[section_name] = sub_tabs

            for panel in section_panels_list:
                panel_title = str(getattr(panel, "title", panel_id))
                panel_id = str(getattr(panel, "panel_id", "unknown"))

                widget = self._create_real_panel_widget(panel_id)
                if widget is not None:
                    self._panel_widgets[panel_id] = widget
                    sub_tabs.addTab(widget, str(panel_title))
                    continue

                # Fallback: text label for headless adapters
                page = QWidget()
                layout = QVBoxLayout(page)
                layout.setContentsMargins(20, 20, 20, 20)
                panel_label = QLabel(
                    f"{panel_title}\n\n"
                    f"Panel ID: {panel_id}\n"
                    "Status: waiting for backend snapshot",
                    page,
                )
                panel_label.setWordWrap(True)
                panel_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                panel_label.setMinimumSize(900, 420)
                layout.addWidget(panel_label)
                layout.addStretch(1)
                self._panel_labels[panel_id] = panel_label
                sub_tabs.addTab(page, str(panel_title))

            # Add section tab
            self._main_tabs.addTab(sub_tabs, section_name)

        self.setCentralWidget(self._main_tabs)

    def _create_real_panel_widget(self, panel_id: str) -> Optional[QWidget]:
        """Create a real visual widget for panels already implemented.

        This method intentionally uses local imports so headless/package import
        paths do not load every PyQt/WebEngine panel eagerly.  It is the A1
        bridge from registry shell to visible Obsidian Night widgets.
        """
        try:
            if panel_id == "status":
                from dashboard.panels.status_panel import StatusPanel

                return StatusPanel(self)
            if panel_id == "briefing":
                from dashboard.panels.briefing_panel import BriefingPanel

                return BriefingPanel(self)
            if panel_id == "global_vitals":
                from dashboard.panels.global_vitals_panel import GlobalVitalsPanel

                return GlobalVitalsPanel(self)
            if panel_id == "mtf_chart":
                from dashboard.panels.mtf_chart_panel import MtfChartPanel

                return MtfChartPanel(self)
            if panel_id == "system_health":
                from dashboard.panels.system_health_panel import SystemHealthPanel

                return SystemHealthPanel(self)
            if panel_id == "volume_profile":
                from dashboard.panels.volume_profile_panel import VolumeProfilePanel

                return VolumeProfilePanel(self)
        except Exception as exc:
            self.log.warning(
                "Real panel widget unavailable; falling back to registry preview",
                dashboard_component="ui_mainwindow",
                panel_id=panel_id,
                error=str(exc),
            )
        return None

    def _build_status_strip(self) -> None:
        self.status_strip = StatusStrip(self)
        self.setStatusBar(self.status_strip)

    def _current_status_value(self, label_attr: str, prefix: str, fallback: str = "-") -> str:
        """Read the current StatusStrip value without inventing backend truth."""
        try:
            label = getattr(getattr(self, "status_strip", None), label_attr, None)
            text = label.text() if label is not None and hasattr(label, "text") else ""
            if isinstance(text, str) and text.startswith(prefix):
                value = text[len(prefix) :].strip()
                return value if value and value != "-" else fallback
        except Exception:
            pass
        return fallback

    def _extract_frame_meta(self, frame: Any) -> tuple[Dict[str, Any], str, str, Optional[str], float, str]:
        """Return display metadata for a backend-provided frame.

        Pre-Week-8 stabilization (P0 truth fix): clock ticks may fire before
        SnapshotBus has decoded any real backend payload.  Earlier routing used
        HOT/WARM/COLD defaults such as HEALTHY/current mode/current timestamp;
        that made a missing backend look fresh.  This helper is deliberately
        fail-closed: empty or invalid frames render as STALE +
        WAITING_FOR_BACKEND and never create a fake last-update timestamp.
        Returns: (payload, feed_health, system_state, last_update, dq_value, backend_state)
        """
        if not isinstance(frame, dict) or not frame:
            return {}, "STALE", "WAITING_FOR_BACKEND", None, 0.0, "DISCONNECTED"

        payload = frame
        feed_health = str(
            payload.get("feed_health")
            or self._current_status_value("feed_health_label", "Feed: ", "UNKNOWN")
        )
        system_state = str(
            payload.get("system_state")
            or self._current_status_value("system_state_label", "State: ", "UNKNOWN")
        )
        last_update = payload.get("timestamp", payload.get("last_update_timestamp"))
        if not isinstance(last_update, str) or not last_update:
            last_update = None

        dq = payload.get("data_quality_score", payload.get("data_quality", 0.0))
        try:
            dq_value = float(dq)
        except Exception:
            dq_value = 0.0
        return payload, feed_health, system_state, last_update, dq_value, "CONNECTED"

    def _update_status_strip(
        self,
        *,
        feed_health: str = "-",
        system_state: str = "-",
        mode: str = "-",
        emergency: bool = False,
        last_update_timestamp: str | None = None,
        data_quality_score: float | None = None,
        backend_state: str = "DISCONNECTED",
        panel_status_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        if hasattr(self, "status_strip"):
            self.status_strip.update_status(
                {
                    "feed_health": feed_health,
                    "system_state": system_state,
                    "mode": mode,
                    "last_update_timestamp": last_update_timestamp,
                    "data_quality_score": data_quality_score if data_quality_score is not None else 0.0,
                    "backend_state": backend_state,
                    "panel_status_summary": panel_status_summary or self._last_panel_summary,
                    "emergency": emergency,
                }
            )

    def _render_registered_panels(self, snapshot: Dict[str, Any], *, refresh_class: Optional[str] = None) -> Dict[str, Any]:
        """Render registry-backed panel placeholders and return summary counts.

        Week 04 integration note:
        This bridges the panel registry contract into visible UI state without
        building future rich widgets early.  Individual panel rendering remains
        isolated in PanelRegistry; MainWindow only displays status, warnings,
        errors, and a compact payload preview.

        Pre-Week-9 tier-purity fix: HOT/WARM ticks render only matching
        registry adapters; the full registry refresh is reserved for COLD/headless
        paths so Week 9+ panels cannot overload the 200ms HOT loop.
        """
        try:
            panel_results = self.panel_registry.render_all(
                snapshot if isinstance(snapshot, dict) else {},
                refresh_class=refresh_class,
            )
        except Exception as exc:
            self.log.error(
                "Panel registry render failed",
                dashboard_component="ui_mainwindow",
                error=str(exc),
            )
            self._last_panel_summary = {
                "total_panels": len(self._panel_labels),
                "ok_count": 0,
                "degraded_count": 0,
                "stale_count": 0,
                "error_count": len(self._panel_labels),
                "warnings_count": 0,
            }
            return self._last_panel_summary

        warnings_count = 0
        for result in panel_results:
            warnings_count += len(getattr(result, "warnings", []) or [])
            label = self._panel_labels.get(str(getattr(result, "panel_id", "")))
            if label is None:
                continue
            label.setText(self._format_panel_result(result))

        metrics = {}
        try:
            metrics = self.panel_registry.get_metrics()
        except Exception:
            metrics = {}
        summary = dict(metrics.get("summary", {}) if isinstance(metrics, dict) else {})
        summary["warnings_count"] = warnings_count
        summary["last_rendered_refresh_class"] = refresh_class or "all"
        summary["last_rendered_panel_count"] = len(panel_results)
        self._last_panel_summary = summary
        return summary

    def _format_panel_result(self, result: Any) -> str:
        payload = getattr(result, "payload", {}) or {}
        try:
            payload_preview = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            payload_preview = str(payload)
        if len(payload_preview) > 1200:
            payload_preview = payload_preview[:1200] + "..."

        warnings = getattr(result, "warnings", []) or []
        errors = getattr(result, "errors", []) or []
        status = str(getattr(result, "status", "UNKNOWN"))

        return (
            f"{getattr(result, 'title', 'Panel')}\n\n"
            f"Panel ID: {getattr(result, 'panel_id', 'unknown')}\n"
            f"Status: {status}\n"
            f"Warnings: {len(warnings)}" + (f" — {warnings[:2]}" if warnings else "") + "\n"
            f"Errors: {len(errors)}" + (f" — {errors[:2]}" if errors else "") + "\n"
            f"Render: {float(getattr(result, 'render_ms', 0.0) or 0.0):.2f} ms\n\n"
            f"Payload preview:\n{payload_preview}"
        )


    def _call_panel_method(self, panel_id: str, method_name: str, payload: Dict[str, Any]) -> None:
        """Safely route a projected frame to a real embedded panel widget.

        Pre-Week-8 A2 integration note:
        The visible widgets are consumers of backend/projected truth only.
        MainWindow does not transform or invent panel data here; it only forwards
        the latest validated HOT/WARM/COLD payload to widgets that already expose
        the matching update method.  Any widget-local failure is isolated so one
        panel cannot break the refresh loop.
        """
        widget = self._panel_widgets.get(panel_id)
        if widget is None:
            return
        method = getattr(widget, method_name, None)
        if not callable(method):
            return
        try:
            method(payload)
        except Exception as exc:
            self.log.error(
                "Embedded panel update failed",
                dashboard_component="ui_mainwindow",
                panel_id=panel_id,
                method=method_name,
                error=str(exc),
            )

    def _route_hot_widgets(self, frame: Dict[str, Any]) -> None:
        self._call_panel_method("status", "update_hot", frame)
        self._call_panel_method("global_vitals", "update_hot", frame)

        guard_payload = None
        if isinstance(frame, dict):
            guard_payload = frame.get("component_guard_heavyweights", frame.get("heavyweights"))
        if guard_payload is not None:
            # Component Guard is a separate hot feed on the visual widget.  We
            # forward only a backend-provided list; if absent, the widget keeps
            # its explicit unavailable/degraded visual state.
            widget = self._panel_widgets.get("global_vitals")
            method = getattr(widget, "update_component_guard", None) if widget is not None else None
            if callable(method):
                try:
                    method(guard_payload)
                except Exception as exc:
                    self.log.error(
                        "GlobalVitals component guard update failed",
                        dashboard_component="ui_mainwindow",
                        error=str(exc),
                    )

    def _route_warm_widgets(self, frame: Dict[str, Any]) -> None:
        self._call_panel_method("briefing", "update_warm", frame)
        self._call_panel_method("mtf_chart", "update_warm", frame)
        self._call_panel_method("volume_profile", "update_cold", frame)

    def _route_cold_widgets(self, frame: Dict[str, Any]) -> None:
        self._call_panel_method("system_health", "update_cold", frame)

    def _trigger_emergency(self) -> None:
        """Show the EmergencyStopDialog instead of bypassing to kill-switch directly.
        Roadmap §5.3: emergency stop must be one click away and preserve state for audit.
        """
        dialog = EmergencyStopDialog(
            parent=self,
            command_router=getattr(self, "command_router", None),
        )
        dialog.stop_activated.connect(lambda: self.on_emergency("Manual emergency stop"))
        dialog.exec()

    def update_hot(self, frame: dict) -> None:
        frame_payload, feed_health, system_state, last_update, dq_value, backend_state = self._extract_frame_meta(frame)
        self._route_hot_widgets(frame_payload)
        panel_summary = self._render_registered_panels(frame_payload, refresh_class="hot")
        self._update_status_strip(
            feed_health=feed_health,
            system_state=system_state,
            mode=self.mode,
            last_update_timestamp=last_update,
            data_quality_score=dq_value,
            backend_state=backend_state,
            panel_status_summary=panel_summary,
        )
        self.log.debug(
            "Hot frame received",
            dashboard_component="ui_mainwindow",
            has_frame=bool(frame),
        )

    def update_warm(self, frame: dict) -> None:
        frame_payload, feed_health, system_state, last_update, dq_value, backend_state = self._extract_frame_meta(frame)
        self._route_warm_widgets(frame_payload)
        panel_summary = self._render_registered_panels(frame_payload, refresh_class="warm")
        self._update_status_strip(
            feed_health=feed_health,
            system_state=system_state,
            mode=self.mode,
            last_update_timestamp=last_update,
            data_quality_score=dq_value,
            backend_state=backend_state,
            panel_status_summary=panel_summary,
        )
        self.log.info(
            "Warm frame received",
            dashboard_component="ui_mainwindow",
            has_frame=bool(frame),
        )

    def update_cold(self, frame: dict) -> None:
        frame_payload, feed_health, system_state, last_update, dq_value, backend_state = self._extract_frame_meta(frame)
        self._route_cold_widgets(frame_payload)
        panel_summary = self._render_registered_panels(frame_payload)
        self._update_status_strip(
            feed_health=feed_health,
            system_state=system_state,
            mode=self.mode,
            last_update_timestamp=last_update,
            data_quality_score=dq_value,
            backend_state=backend_state,
            panel_status_summary=panel_summary,
        )
        self.log.info(
            "Cold frame received",
            dashboard_component="ui_mainwindow",
            has_frame=bool(frame),
        )

    # ------------------------------------------------------------------
    # Clock-driven consumer-side refresh slots (CRIT-A wiring).
    # ------------------------------------------------------------------
    # DashboardClock emits hot_tick / warm_tick / cold_tick on its tiered
    # cadence. Those signals connect to these slots (not to SnapshotBus,
    # which is push-only). Each slot pulls the most recent VALIDATED payload
    # for its tier from the bus and re-renders.
    #
    # If the bus has not yet seen a real frame, payload is None and update_*
    # renders an explicit STALE / WAITING_FOR_BACKEND state.  We never invent
    # HEALTHY status or a fake last-update timestamp — that would violate
    # dashboard_roadmap.txt PRIMARY RULE.
    # ------------------------------------------------------------------

    def on_hot_tick(self) -> None:
        bus = self.snapshot_bus
        payload = None
        try:
            # last_valid_hot_payload is a @property, not a function
            payload = getattr(bus, "last_valid_hot_payload", None)
        except Exception as exc:
            self.log.error(
                "Hot tick payload pull failed",
                dashboard_component="ui_mainwindow",
                error=str(exc),
            )
            payload = None
        self.update_hot(payload if isinstance(payload, dict) else {})

    def on_warm_tick(self) -> None:
        bus = self.snapshot_bus
        payload = None
        try:
            # last_valid_warm_payload is a @property, not a function
            payload = getattr(bus, "last_valid_warm_payload", None)
        except Exception as exc:
            self.log.error(
                "Warm tick payload pull failed",
                dashboard_component="ui_mainwindow",
                error=str(exc),
            )
            payload = None
        self.update_warm(payload if isinstance(payload, dict) else {})

    def on_cold_tick(self) -> None:
        bus = self.snapshot_bus
        payload = None
        try:
            # last_valid_cold_payload is a @property, not a function
            payload = getattr(bus, "last_valid_cold_payload", None)
        except Exception as exc:
            self.log.error(
                "Cold tick payload pull failed",
                dashboard_component="ui_mainwindow",
                error=str(exc),
            )
            payload = None
        self.update_cold(payload if isinstance(payload, dict) else {})

    def on_emergency(self, reason: str) -> None:
        self.setWindowTitle("EMERGENCY - Junior Aladdin")
        self.live_request_action.setEnabled(False)
        self._update_status_strip(feed_health="DOWN", system_state=reason, mode=self.mode, emergency=True)
        self.log.error(
            "Emergency activated",
            dashboard_component="ui_mainwindow",
            reason=reason,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.log.info("Main window closed", dashboard_component="ui_mainwindow")
        super().closeEvent(event)


if __name__ == "__main__":
    class DummyBus:
        replay_date = None

    class DummyKillSwitch(QObject):
        emergency_activated = pyqtSignal(str)

    class DummyClock:
        pass

    app = QApplication(sys.argv)
    window = MainWindow(
        snapshot_bus=DummyBus(),
        kill_switch_reader=DummyKillSwitch(),
        dashboard_clock=DummyClock(),
        mode="ALERT",
    )
    window.show()
    raise SystemExit(app.exec())