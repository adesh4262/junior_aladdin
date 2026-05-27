"""Dashboard command and control dialogs.

Emergency Stop — one-click kill all trading (roadmap §5.3).
Future dialogs:
- compliance_gate.py  (roadmap §5.1 Step 2)
- live_mode_handshake.py  (roadmap §5.1 Step 1 & 3)
- operator_notes.py
"""

from __future__ import annotations

try:
    from .emergency_stop import EmergencyStopDialog
except ImportError:  # pragma: no cover
    EmergencyStopDialog = None  # type: ignore[assignment]

try:
    from .compliance_gate import ComplianceGate
except ImportError:  # pragma: no cover
    ComplianceGate = None  # type: ignore[assignment]

__all__ = ["EmergencyStopDialog", "ComplianceGate"]
