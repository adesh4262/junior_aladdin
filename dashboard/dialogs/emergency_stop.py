"""Emergency stop dialog with one-click kill action."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QShowEvent
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout

try:
    from src.utils.config_loader import Config
except Exception:  # pragma: no cover
    Config = None  # type: ignore

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)

from src.core.dashboard_control_plane import trigger_emergency_kill_switch


class EmergencyStopDialog(QDialog):
    """Fast one-click emergency stop dialog."""

    stop_activated = pyqtSignal()

    def __init__(self, parent: Any = None, command_router: Any = None) -> None:
        super().__init__(parent)
        self.log = setup_logger("dashboard_dialogs_emergency")
        self.command_router = command_router

        self.setWindowTitle("EMERGENCY STOP")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setModal(True)
        self.setStyleSheet("QDialog { background-color: #7F1D1D; color: #FFFFFF; }")

        layout = QVBoxLayout(self)

        icon_label = QLabel("⚠")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_font = QFont()
        icon_font.setPointSize(40)
        icon_font.setBold(True)
        icon_label.setFont(icon_font)
        layout.addWidget(icon_label)

        title_label = QLabel("EMERGENCY STOP - Click to kill all trading")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        self.stop_button = QPushButton("STOP NOW")
        self.stop_button.setMinimumSize(200, 80)
        btn_font = QFont()
        btn_font.setPointSize(16)
        btn_font.setBold(True)
        self.stop_button.setFont(btn_font)
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #991B1B; color: #FFFFFF; border: 2px solid #FCA5A5; }"
            "QPushButton:pressed { background-color: #7F1D1D; }"
        )
        self.stop_button.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.resize(460, 260)

    def _on_stop_clicked(self) -> None:
        # Safety-critical integration fix:
        # CommandRouter's canonical API is send_command(...), while older stubs
        # exposed send(...).  Resolve both explicitly so the emergency action
        # cannot be silently lost because of a method-name mismatch.
        #
        # The payload uses "type" (not only "command") because
        # dashboard.core.command_router documents "type" as the backend routing
        # discriminator.  The duplicate "command" field is kept for transitional
        # backend consumers during the multi-AI wiring phase.
        payload = {
            "type": "emergency_stop",
            "command": "emergency_stop",
            "source": "dashboard",
            "timestamp_ns": time.time_ns(),
        }

        sent = False
        if self.command_router is not None:
            try:
                sender = getattr(self.command_router, "send_command", None)
                if not callable(sender):
                    sender = getattr(self.command_router, "send", None)
                if callable(sender):
                    sent = bool(sender(payload))
                else:
                    self.log.critical(
                        "Emergency stop router has no send method",
                        dashboard_component="emergency_stop_dialog",
                        router_type=type(self.command_router).__name__,
                    )
            except Exception as exc:
                # Dialog must never crash during emergency operation.  We still
                # put the dashboard UI into EMERGENCY state below so the
                # operator sees containment, but the failed command is logged as
                # critical because backend containment may not have happened.
                self.log.critical(
                    "Emergency stop command send failed",
                    dashboard_component="emergency_stop_dialog",
                    error=str(exc),
                )
        else:
            self.log.critical(
                "Emergency stop clicked without command router",
                dashboard_component="emergency_stop_dialog",
            )

        kill_switch_name = "junior_aladdin_kill_switch"
        if Config is not None:
            try:
                kill_switch_name = str(Config.get("dashboard", "kill_switch_name", default=kill_switch_name))
            except Exception:
                pass

        # Why this fallback exists:
        # The backend command channel is still being stabilized.  Writing the
        # emergency flag directly into the shared-memory kill-switch block gives
        # the dashboard a backend-independent containment path for Part 1.
        fallback_reason = f"dashboard_emergency_stop:{payload['timestamp_ns']}"
        fallback_sent = trigger_emergency_kill_switch(
            kill_switch_name,
            reason=fallback_reason,
            dashboard_pid=os.getpid(),
        )

        if not sent and not fallback_sent:
            self.log.critical(
                "Emergency stop command and shared-memory fallback both failed; dashboard entering local emergency state",
                dashboard_component="emergency_stop_dialog",
                kill_switch_name=kill_switch_name,
            )
        elif not sent and fallback_sent:
            self.log.critical(
                "Emergency stop command path failed but shared-memory fallback succeeded",
                dashboard_component="emergency_stop_dialog",
                kill_switch_name=kill_switch_name,
            )

        self.log.error(
            "Emergency stop activated",
            command_sent=sent,
            kill_switch_fallback_sent=fallback_sent,
            kill_switch_name=kill_switch_name,
        )
        self.stop_activated.emit()
        self.accept()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.raise_()
        self.activateWindow()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = EmergencyStopDialog()
    dialog.stop_activated.connect(lambda: print("EMERGENCY SIGNAL RECEIVED"))
    dialog.show()
    raise SystemExit(app.exec())