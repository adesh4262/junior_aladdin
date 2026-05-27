"""Part 2B dashboard dry run + sample payload capture.

What this tool validates
------------------------
- real HOT/WARM/COLD frame transport through the actual IPC bridge
- command-channel round-trip using the actual socket command path
- sample payload capture for future panel work (Week 9+)

This tool is intentionally headless and transport-only.  It does not launch the
PyQt dashboard UI.  That keeps it runnable in CI/sandbox environments while
still exercising the real backend/dashboard IPC path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Mapping

from dashboard.core.binary_frame import KIND_COLD, KIND_HOT, KIND_WARM, unpack_frame
from dashboard.core.ipc_client import LengthPrefixedSocketCommandChannel, SnapshotStreamClient
from dashboard.core.state_projection import project_snapshot
from dashboard.core.command_router import CommandRouter
from src.core.dashboard_ipc import DashboardIpcBridge

try:
    from src.utils.config_loader import Config
except Exception:  # pragma: no cover
    Config = None  # type: ignore

KIND_LABELS = {
    KIND_HOT: "hot",
    KIND_WARM: "warm",
    KIND_COLD: "cold",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _sample_snapshot() -> Dict[str, Any]:
    return {
        "timestamp": "2026-05-22T10:15:00+05:30",
        "last_update_timestamp": "2026-05-22T10:15:00+05:30",
        "system_state": "ACTIVE",
        "mode": "PAPER",
        "feed_health": "HEALTHY",
        "data_quality_score": 92.0,
        "ticks_per_second": 14.5,
        "feed_lag_ms": 34.0,
        "using_fallback": False,
        "spot": 24580.25,
        "previous_close": 24520.10,
        "capital": 50000.0,
        "daily_pnl": 1245.0,
        "drawdown_pct": 0.018,
        "trades_today": 2,
        "consecutive_losses": 0,
        "tilt_score": 14.0,
        "risk_state": "NORMAL",
        "session_phase": "GOLDEN_AM",
        "day_type": "TREND_DAY",
        "regime": "TRENDING",
        "narrative_label": "MILD_BULLISH",
        "regime_transition_prob": 0.12,
        "active_brains": ["structural", "institutional"],
        "narrative_score": 68.0,
        "regime_confidence": 0.84,
        "narrative_fit_factors": {"long_fit": 1.0, "short_fit": 0.4},
        "features": {"1min": {"rsi": 58.0, "vwap": 24562.0}, "5min": {"rsi": 61.0}},
        "features_1m": {"rsi": 58.0, "vwap": 24562.0},
        "options_features": {"pcr_oi": 1.08, "atm_iv": 0.14},
        "smart_money_5m": {"sm_direction_score": 18.0, "total_fvgs": 2, "bullish_fvgs": 2, "bearish_fvgs": 0},
        "smart_money": {"5min": {"fvg_zones": 2}},
        "mtf_candles": {
            "1m": [
                {"timestamp": "2026-05-22T10:10:00+05:30", "open": 24540, "high": 24565, "low": 24535, "close": 24560, "volume": 1100},
                {"timestamp": "2026-05-22T10:11:00+05:30", "open": 24560, "high": 24585, "low": 24558, "close": 24580, "volume": 980},
            ],
            "5m": [
                {"timestamp": "2026-05-22T10:10:00+05:30", "open": 24520, "high": 24585, "low": 24510, "close": 24580, "volume": 4200},
            ],
        },
        "vwap_bands": {"5m": {"vwap": [24540.0], "upper1": [24570.0], "lower1": [24510.0], "upper2": [24600.0], "lower2": [24480.0]}},
        "or_levels": {"high": 24590.0, "low": 24505.0},
        "ib_levels": {"high": 24605.0, "low": 24495.0},
        "volume_profile": {"poc": 24550, "vah": 24590, "val": 24515},
        "session_volume_profile": {"poc": 24550},
        "microstructure": {"spread": 0.8, "imbalance": 0.61},
        "order_flow": {"delta": 125, "aggression": "buy"},
        "cvd": 640,
        "imbalance": {"buy": 0.61, "sell": 0.39},
        "absorption_alerts": [{"id": 1, "side": "buy"}],
        "exhaustion_alerts": [{"id": 1, "side": "sell"}],
        "session_profile_anchors": {"open": 24520, "mid": 24555},
        "poc": 24550,
        "vah": 24590,
        "val": 24515,
        "raw_opportunities": [{"id": 1, "strategy": "VWAP Pullback"}],
        "approved_opportunities": [{"id": 1, "strategy": "VWAP Pullback"}],
        "open_positions": [],
        "engine_health": {"data_engine": {"alive": True, "last_heartbeat": "2026-05-22T10:15:00+05:30"}},
    }


def run_dry_run(out_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    host = "127.0.0.1"
    if Config is not None:
        try:
            host = str(Config.get("dashboard", "ipc_host", default=host))
        except Exception:
            pass

    bridge = DashboardIpcBridge(
        host=host,
        snapshot_port=0,
        command_port=0,
        hot_interval_ms=200,
        warm_interval_ms=1000,
        cold_interval_ms=5000,
    )
    if not bridge.start():
        raise RuntimeError("dashboard_ipc_bridge_start_failed")

    received: "queue.Queue[bytes]" = queue.Queue()
    snapshot_client = SnapshotStreamClient(
        host=host,
        port=bridge.snapshot_port,
        frame_handler=lambda frame: received.put(frame),
    )
    snapshot_client.start()

    command_channel = LengthPrefixedSocketCommandChannel(host=host, port=bridge.command_port)
    command_router = CommandRouter(command_channel=command_channel)

    summary: Dict[str, Any] = {
        "snapshot_port": bridge.snapshot_port,
        "command_port": bridge.command_port,
        "frames": {},
        "command_round_trip": False,
        "success": False,
    }

    try:
        snapshot = _sample_snapshot()
        # Wait briefly for the snapshot client connection so the first publish
        # can deliver HOT/WARM/COLD immediately, including the 5s cold tier.
        connect_deadline = time.time() + 2.0
        while time.time() < connect_deadline:
            if bridge.get_status().get("snapshot_clients", 0) > 0:
                break
            time.sleep(0.05)

        deadline = time.time() + 3.0
        seen_kinds = set()

        while time.time() < deadline and seen_kinds != {KIND_HOT, KIND_WARM, KIND_COLD}:
            bridge.publish_due_frames(snapshot)
            try:
                frame = received.get(timeout=0.25)
            except queue.Empty:
                continue
            decoded = unpack_frame(frame)
            if not decoded.get("valid"):
                continue
            kind = int(decoded["kind"])
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            label = KIND_LABELS[kind]
            raw_payload = decoded["payload"]
            projected = project_snapshot(raw_payload, kind)
            (output_dir / f"sample_{label}_raw.json").write_text(
                json.dumps(_jsonable(raw_payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / f"sample_{label}_projected.json").write_text(
                json.dumps(_jsonable(projected), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary["frames"][label] = {
                "seq": decoded["seq"],
                "timestamp_ns": decoded["timestamp_ns"],
                "payload_keys": sorted(list(raw_payload.keys())),
            }

        if seen_kinds != {KIND_HOT, KIND_WARM, KIND_COLD}:
            raise RuntimeError(f"missing_frame_kinds:{sorted(seen_kinds)}")

        assert command_router.send_command({"type": "emergency_stop", "source": "dashboard_part2b_dry_run"}) is True
        command_deadline = time.time() + 2.0
        commands = []
        while time.time() < command_deadline:
            commands = bridge.poll_commands(limit=8)
            if commands:
                break
            time.sleep(0.05)
        if not commands:
            raise RuntimeError("missing_command_round_trip")

        command = commands[0]
        (output_dir / "sample_command_emergency_stop.json").write_text(
            json.dumps(_jsonable(command), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["command_round_trip"] = True
        summary["command_type"] = command.get("type")
        summary["success"] = True
        return summary
    finally:
        command_channel.close()
        snapshot_client.stop()
        bridge.stop()
        (output_dir / "dry_run_summary.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dashboard Part 2B dry run and capture sample payloads")
    default_dir = Path("artifacts") / "dashboard_ipc_samples"
    parser.add_argument("--out-dir", default=str(default_dir))
    args = parser.parse_args()

    summary = run_dry_run(args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
