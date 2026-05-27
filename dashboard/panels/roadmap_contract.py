"""Dashboard roadmap panel truth table.

Pre-Week-9 stabilization / roadmap clarity:
The registry contains a mix of real PyQt widgets, headless adapters, and legacy
summary placeholders.  This module prevents false claims that all 16 roadmap
panels are implemented while still allowing placeholders to support tests and
operator visibility during earlier weeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RoadmapPanelSpec:
    week: int
    panel_id: str
    title: str
    status: str
    notes: str


ROADMAP_PANEL_SPECS: Tuple[RoadmapPanelSpec, ...] = (
    RoadmapPanelSpec(5, "status", "STATUS", "implemented_widget", "Week 5 real PyQt widget + adapter."),
    RoadmapPanelSpec(5, "briefing", "BRIEFING", "implemented_widget", "Week 5 real PyQt widget + adapter."),
    RoadmapPanelSpec(7, "mtf_chart", "MTF CHART", "implemented_widget", "Week 7 real PyQt/WebEngine widget + adapter."),
    RoadmapPanelSpec(9, "volume_profile", "VOLUME PROFILE", "future_week", "Do not build before Week 9."),
    RoadmapPanelSpec(10, "option_chain", "OPTION CHAIN", "future_week", "Headless placeholder exists; full panel is Week 10."),
    RoadmapPanelSpec(11, "narrative", "NARRATIVE", "future_week", "Headless placeholder exists; full panel is Week 11."),
    RoadmapPanelSpec(11, "regime", "REGIME", "future_week", "Do not build before Week 11."),
    RoadmapPanelSpec(13, "risk", "RISK", "future_week", "Do not build before Week 13."),
    RoadmapPanelSpec(13, "positions", "POSITIONS", "future_week", "Headless placeholder exists; full panel is Week 13."),
    RoadmapPanelSpec(14, "history", "HISTORY", "future_week", "Do not build before Week 14."),
    RoadmapPanelSpec(14, "behavioral", "BEHAVIORAL", "future_week", "Do not build before Week 14."),
    RoadmapPanelSpec(14, "ml_insights", "ML INSIGHTS", "future_week", "Do not build before Week 14."),
    RoadmapPanelSpec(12, "smc_visuals", "SMC VISUALS", "future_week", "Do not build before Week 12."),
    RoadmapPanelSpec(5, "system_health", "SYSTEM HEALTH", "implemented_widget", "Week 5 real PyQt widget + adapter."),
    RoadmapPanelSpec(15, "captains_log", "CAPTAIN'S LOG", "future_week", "Do not build before Week 15."),
    RoadmapPanelSpec(6, "global_vitals", "GLOBAL VITALS", "implemented_widget", "Week 6 real PyQt widget + adapter."),
)

ROADMAP_PANEL_IDS: Tuple[str, ...] = tuple(spec.panel_id for spec in ROADMAP_PANEL_SPECS)
IMPLEMENTED_WIDGET_PANEL_IDS: Tuple[str, ...] = tuple(
    spec.panel_id for spec in ROADMAP_PANEL_SPECS if spec.status == "implemented_widget"
)
FUTURE_PANEL_IDS: Tuple[str, ...] = tuple(spec.panel_id for spec in ROADMAP_PANEL_SPECS if spec.status == "future_week")

# Current registry-only placeholders that are intentionally NOT equivalent to
# final roadmap PyQt panels.  They support headless visibility but must not be
# counted as finished roadmap surfaces.
REGISTRY_PLACEHOLDER_PANEL_IDS: Tuple[str, ...] = (
    "system_status",
    "market_overview",
    "session_state",
    "feature_snapshot",
    "opportunity_pipeline",
    "brain_status",
    "engine_health",
)


def roadmap_panel_status_map() -> Dict[str, str]:
    return {spec.panel_id: spec.status for spec in ROADMAP_PANEL_SPECS}
