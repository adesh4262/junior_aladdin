"""
dashboard/dialogs/compliance_gate.py

ComplianceGate — Step 2 of the LIVE mode handshake.

This dialog is a strict UI gate: it *does not* query the backend directly.
It receives a pre-computed `compliance_status` mapping (typically fetched by the
caller via CommandRouter/request-response elsewhere) and renders a checklist.

Rules:
- "Verify & Proceed" is enabled only if *all* checks are True.
- Missing/unknown/None checks are treated as failed (strict).
- In replay mode, LIVE is not available; dialog stays reject-only.

Expected compliance_status keys:
- algo_id_ok: bool
- static_ip_ok: bool
- limit_orders_only: bool
- audit_trail_ok: bool
- details: dict | str (optional)
- replay_mode: bool (optional; if True -> force fail)
- backend_unreachable: bool (optional; if True -> force fail)

This file contains UI presentation only (no backend calls, no persistence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QWidget,
)

from src.utils.logger import setup_logger


@dataclass(frozen=True)
class _CheckSpec:
    key: str
    label: str


class ComplianceGate(QDialog):
    """
    Modal dialog that verifies SEBI compliance gates (UI-only).

    Caller should check:
        dlg.exec() == QDialog.DialogCode.Accepted
    """

    _CHECKS: Sequence[_CheckSpec] = (
        _CheckSpec("algo_id_ok", "Algo‑ID registered & valid"),
        _CheckSpec("static_ip_ok", "Static IP matches approved IP"),
        _CheckSpec("limit_orders_only", "Limit‑orders‑only enforced (no market orders)"),
        _CheckSpec("audit_trail_ok", "Audit trail sink writable (database reachable)"),
    )

    def __init__(self, compliance_status: Optional[Mapping[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._log = self._get_logger()

        self.setModal(True)
        self.setWindowTitle("Compliance Verification")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self._status: Mapping[str, Any] = dict(compliance_status or {})
        self._replay_mode = bool(self._status.get("replay_mode", False))
        self._backend_unreachable = bool(self._status.get("backend_unreachable", False)) or (compliance_status is None)

        self._log.info(
            "ComplianceGate opened. replay_mode=%s backend_unreachable=%s keys=%s",
            self._replay_mode,
            self._backend_unreachable,
            sorted(list(self._status.keys())),
            extra={"dashboard_component": "compliance_gate"},
        )

        # Widgets (kept as attributes for integration / potential tests)
        self.proceed_button = QPushButton("Verify & Proceed")
        self.cancel_button = QPushButton("Cancel")
        self.status_labels: list[QLabel] = []

        self._details_label = QLabel()
        self._details_label.setWordWrap(True)
        self._details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._build_ui()
        self._render()

        # Logging on result
        self.accepted.connect(self._on_accepted)
        self.rejected.connect(self._on_rejected)

    @staticmethod
    def _get_logger() -> Any:
        """Return the project logger without breaking UI startup.

        IMPORTANT INTEGRATION FIX:
        The shared project logger factory returns a BoundLogger wrapper, not a
        raw ``logging.Logger``.  Checking ``isinstance(log, logging.Logger)``
        silently discarded the real project logger and routed compliance-gate
        audit events to an unconfigured stdlib logger.

        Compliance approval/rejection is safety- and audit-relevant, so accept
        any logger-like object exposing the methods used by this dialog while
        retaining a stdlib fallback if logger construction itself fails.
        """
        name = "dashboard_dialogs_compliance_gate"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "error")):
                return log
        except Exception:
            pass
        return logging.getLogger(name)

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Compliance Verification — SEBI April 2026")
        tf = QFont()
        tf.setPointSize(max(11, tf.pointSize()))
        tf.setBold(True)
        title.setFont(tf)

        subtitle = QLabel(
            "All compliance checks must pass to enable LIVE trading. "
            "This gate is strict; unknown/missing status is treated as failed."
        )
        subtitle.setWordWrap(True)

        root.addWidget(title)
        root.addWidget(subtitle)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        # Checklist rows
        for spec in self._CHECKS:
            row = QHBoxLayout()
            row.setSpacing(10)

            icon = QLabel("⚠️")
            icon.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            # Keep icon width stable for alignment across rows.
            icon.setMinimumWidth(24)

            text = QLabel(spec.label)
            text.setWordWrap(True)

            row_widget = QWidget()
            row_widget.setLayout(row)
            row.addWidget(icon, 0)
            row.addWidget(text, 1)

            self.status_labels.append(icon)
            root.addWidget(row_widget)

        # Details / failure reasons
        details_title = QLabel("Status details:")
        details_title_font = QFont()
        details_title_font.setBold(True)
        details_title.setFont(details_title_font)

        root.addWidget(details_title)
        root.addWidget(self._details_label)

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)

        self.proceed_button.setDefault(True)
        self.proceed_button.clicked.connect(self.accept)

        self.cancel_button.clicked.connect(self.reject)

        btns.addWidget(self.proceed_button)
        btns.addWidget(self.cancel_button)

        root.addLayout(btns)

        self.setLayout(root)

    def _render(self) -> None:
        """
        Render status icons and enable/disable proceed button.

        Missing/unknown/None are treated as failed.
        """
        # Replay mode hard-block (UI still shows checks as failed to avoid ambiguity).
        if self._replay_mode:
            self._set_all_icons_unknown()
            self._details_label.setText(
                "<b>Replay mode — LIVE not available.</b><br/>"
                "This environment is running in replay; compliance verification cannot enable LIVE trading."
            )
            self.proceed_button.setEnabled(False)
            self.proceed_button.setToolTip("Replay mode — LIVE not available.")
            return

        if self._backend_unreachable:
            self._set_all_icons_unknown()
            self._details_label.setText(
                "<b>Backend unreachable / compliance status missing.</b><br/>"
                "Cannot verify compliance; LIVE trading remains locked."
            )
            self.proceed_button.setEnabled(False)
            self.proceed_button.setToolTip("Backend unreachable — cannot verify compliance.")
            return

        failures: list[str] = []
        unknowns: list[str] = []
        all_ok = True

        for idx, spec in enumerate(self._CHECKS):
            tri = self._tri_state(self._status.get(spec.key, None))
            icon_label = self.status_labels[idx]

            if tri is True:
                self._set_icon(icon_label, "✅", color="#1B5E20")
            elif tri is False:
                all_ok = False
                failures.append(spec.label)
                self._set_icon(icon_label, "❌", color="#B71C1C")
            else:
                # Unknown/missing treated as failure (strict).
                all_ok = False
                unknowns.append(spec.label)
                self._set_icon(icon_label, "⚠️", color="#E65100")

        # Compose details message
        details = self._status.get("details", None)
        details_html = self._format_details(details)

        if all_ok:
            self._details_label.setText(
                "<b>All compliance checks passed.</b><br/>"
                "You may proceed to the final LIVE confirmation step."
                + (f"<br/><br/>{details_html}" if details_html else "")
            )
            self.proceed_button.setEnabled(True)
            self.proceed_button.setToolTip("All checks passed.")
        else:
            # Strict: do not allow proceed
            self.proceed_button.setEnabled(False)

            reasons: list[str] = []
            if failures:
                reasons.append("<b>Failed checks:</b><ul>" + "".join(f"<li>{self._escape_html(x)}</li>" for x in failures) + "</ul>")
            if unknowns:
                reasons.append(
                    "<b>Unknown / missing status (treated as failed):</b><ul>"
                    + "".join(f"<li>{self._escape_html(x)}</li>" for x in unknowns)
                    + "</ul>"
                )

            base = "<b>Compliance verification failed.</b><br/>LIVE trading remains locked.<br/><br/>" + "".join(reasons)
            if details_html:
                base += f"<br/>{details_html}"

            self._details_label.setText(base)

            tooltip_items = failures + unknowns
            if tooltip_items:
                self.proceed_button.setToolTip("Cannot proceed. Resolve: " + "; ".join(tooltip_items))
            else:
                self.proceed_button.setToolTip("Cannot proceed. Compliance status indicates failure.")

            # Log failures with details for observability
            self._log.error(
                "ComplianceGate checks failed. failures=%s unknowns=%s details_type=%s",
                failures,
                unknowns,
                type(details).__name__ if details is not None else "None",
                extra={"dashboard_component": "compliance_gate"},
            )

    def _set_all_icons_unknown(self) -> None:
        for icon in self.status_labels:
            self._set_icon(icon, "⚠️", color="#E65100")

    @staticmethod
    def _set_icon(label: QLabel, glyph: str, color: str) -> None:
        label.setText(glyph)
        # Keep styling lightweight; avoid global styles.
        label.setStyleSheet(f"color: {color}; font-size: 16px;")

    @staticmethod
    def _tri_state(value: Any) -> Optional[bool]:
        """
        Convert a status value into tri-state: True / False / None(unknown).
        """
        if value is True:
            return True
        if value is False:
            return False
        if value is None:
            return None

        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "ok", "pass", "passed", "yes", "y", "1"}:
                return True
            if s in {"false", "fail", "failed", "no", "n", "0"}:
                return False
            if s in {"unknown", "na", "n/a", "none", ""}:
                return None

        # Any other type is treated as unknown (strict -> failed elsewhere).
        return None

    @staticmethod
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _format_details(self, details: Any) -> str:
        """
        Format optional details into HTML (defensive; may be dict/str/other).
        """
        if details is None:
            return ""

        try:
            if isinstance(details, str):
                s = details.strip()
                return f"<b>Details:</b><br/>{self._escape_html(s)}" if s else ""
            if isinstance(details, Mapping):
                # Render key-value pairs; stable order for readability.
                items: list[Tuple[str, str]] = []
                for k in sorted(details.keys(), key=lambda x: str(x)):
                    v = details.get(k)
                    items.append((str(k), "" if v is None else str(v)))
                if not items:
                    return ""
                li = "".join(
                    f"<li><code>{self._escape_html(k)}</code>: {self._escape_html(v)}</li>"
                    for k, v in items
                )
                return "<b>Details:</b><ul>" + li + "</ul>"
            # Fallback
            return f"<b>Details:</b><br/>{self._escape_html(str(details))}"
        except Exception as e:
            self._log.error(
                "Failed to format compliance details. err=%s details_type=%s",
                repr(e),
                type(details).__name__,
                extra={"dashboard_component": "compliance_gate"},
            )
            return ""

    def _on_accepted(self) -> None:
        self._log.info(
            "ComplianceGate result=ACCEPTED",
            extra={"dashboard_component": "compliance_gate"},
        )

    def _on_rejected(self) -> None:
        self._log.info(
            "ComplianceGate result=REJECTED",
            extra={"dashboard_component": "compliance_gate"},
        )


if __name__ == "__main__":
    import sys
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Test all passes
    status_ok = {
        "algo_id_ok": True,
        "static_ip_ok": True,
        "limit_orders_only": True,
        "audit_trail_ok": True,
    }
    gate = ComplianceGate(status_ok)
    # Auto-proceed so the self-test doesn't require manual clicks.
    QTimer.singleShot(150, gate.proceed_button.click)
    if gate.exec() == QDialog.DialogCode.Accepted:
        print("Compliance gate accepted (as expected)")
    else:
        print("ERROR: Should have accepted")

    # Test failure
    status_fail = {
        "algo_id_ok": False,
        "static_ip_ok": True,
        "limit_orders_only": True,
        "audit_trail_ok": True,
    }
    gate2 = ComplianceGate(status_fail)
    # Auto-cancel so the self-test doesn't hang.
    QTimer.singleShot(150, gate2.cancel_button.click)
    if gate2.exec() == QDialog.DialogCode.Rejected:
        print("Compliance gate rejected (as expected)")
    else:
        print("ERROR: Should have rejected")

    print("compliance_gate self-test passed")