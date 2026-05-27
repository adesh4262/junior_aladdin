"""
Junior Aladdin dashboard package.

This package exposes a lightweight, read-only dashboard runtime that renders
MarketState snapshots through a deterministic panel registry.
"""

from .panels import PanelBase, PanelRegistry, PanelResult, PanelStatus, build_default_registry

try:
	from .app import DashboardApp, build_dashboard_app, format_dashboard_report
except ImportError as exc:
	# Only swallow import failures caused by missing Qt bindings; re-raise
	# for real errors in dashboard.app so bugs are not silently masked.
	if "PyQt6" in str(exc) or "Qt" in str(exc):
		DashboardApp = None
		build_dashboard_app = None
		format_dashboard_report = None
	else:
		raise

__all__ = [
	"PanelBase",
	"PanelRegistry",
	"PanelResult",
	"PanelStatus",
	"build_default_registry",
]

if DashboardApp is not None:
	__all__.extend([
		"DashboardApp",
		"build_dashboard_app",
		"format_dashboard_report",
	])
