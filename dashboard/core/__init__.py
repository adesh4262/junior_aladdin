"""Dashboard core infrastructure."""

from .binary_frame import KIND_COLD, KIND_HOT, KIND_WARM, pack_frame, unpack_frame
from .state_projection import (
    project_cold_snapshot,
    project_hot_snapshot,
    project_snapshot,
    project_warm_snapshot,
)

# ---------------------------------------------------------------------------
# PyQt6-dependent classes
# ---------------------------------------------------------------------------
# Both DashboardClock and KillSwitchReader require PyQt6 (QObject + signals /
# QTimer). We bind a None placeholder first so the symbol is always defined,
# then attempt import. Only swallow PyQt6/Qt-related import failures — any
# other ImportError (e.g. an internal helper refactor breakage) must propagate
# so the whole package does not silently degrade.
#
# Why explicit export here:
#   Keeping all public dashboard.core symbols importable as
#       from dashboard.core import DashboardClock, KillSwitchReader
#   gives consumers a single, stable surface. Without this, callers fall back
#   to deep imports (dashboard.core.shared_memory_kill_switch.KillSwitchReader),
#   which couples them to file layout and breaks on future reorganization.
#   (Addresses HIGH-A from repo-wide review of commit d33cb41.)
# ---------------------------------------------------------------------------

DashboardClock = None  # type: ignore[assignment]
try:
    from .dashboard_clock import DashboardClock as _DashboardClock

    DashboardClock = _DashboardClock  # type: ignore[assignment]
except ImportError as exc:
    msg = str(exc)
    if not ("PyQt6" in msg or "Qt" in msg):
        raise

KillSwitchReader = None  # type: ignore[assignment]
try:
    from .shared_memory_kill_switch import KillSwitchReader as _KillSwitchReader

    KillSwitchReader = _KillSwitchReader  # type: ignore[assignment]
except ImportError as exc:
    msg = str(exc)
    if not ("PyQt6" in msg or "Qt" in msg):
        raise

# CommandRouter — no PyQt6 dependency, always importable.
from .command_router import CommandRouter

__all__ = [
    "pack_frame",
    "unpack_frame",
    "KIND_HOT",
    "KIND_WARM",
    "KIND_COLD",
    "project_hot_snapshot",
    "project_warm_snapshot",
    "project_cold_snapshot",
    "project_snapshot",
    "DashboardClock",
    "KillSwitchReader",
    "CommandRouter",
]
