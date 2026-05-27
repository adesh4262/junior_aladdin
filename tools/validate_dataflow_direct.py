"""
TEMPORARY — DIRECT DATAFLOW PIPELINE VALIDATION
================================================

This tests the REAL pipeline:
    FakePublisher → TCP → SnapshotStreamClient → SnapshotBus
    → binary_frame.decode → state_projection → PanelRegistry.render_all

It does NOT use dashboard.main --headless (which is static).
Instead it:
    1. Starts the fake TCP publisher
    2. Creates SnapshotBus + SnapshotStreamClient + PanelRegistry
    3. Waits for frames to arrive over TCP
    4. Reads projected payloads from SnapshotBus
    5. Renders panels with real data
    6. Validates output

USAGE:
    python -m tools.validate_dataflow_direct

CLEANUP:
    Delete this file when real backend IPC exists.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 70)
    print("  DATAFLOW PIPELINE VALIDATION (DIRECT)")
    print("=" * 70)
    print()

    # ─── Step 1: Start fake publisher ──────────────────────────────────
    print("[1/5] Starting fake snapshot publisher...")
    publisher = subprocess.Popen(
        [sys.executable, "-m", "tools.fake_snapshot_publisher"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(1.0)
    if publisher.poll() is not None:
        print("[FAIL] Fake publisher died on startup")
        return 1
    print("[OK] Publisher running")
    print()

    try:
        # ─── Step 2: Create SnapshotBus + StreamClient ─────────────────
        print("[2/5] Creating SnapshotBus and connecting to publisher...")
        from PyQt6.QtCore import QCoreApplication
        
        app = QCoreApplication([])

        from dashboard.core.snapshot_bus import SnapshotBus
        from dashboard.core.ipc_client import SnapshotStreamClient
        from dashboard.panels import build_default_registry

        bus = SnapshotBus()
        registry = build_default_registry()

        received_frames: list[Dict[str, Any]] = []

        def on_hot(frame: dict) -> None:
            received_frames.append(("hot", frame))
            print(f"  [HOT] received: feed_health={frame.get('feed_health')}, "
                  f"spot={frame.get('spot')}, state={frame.get('system_state')}")

        def on_warm(frame: dict) -> None:
            received_frames.append(("warm", frame))
            print(f"  [WARM] received: narrative={frame.get('narrative_label')}, "
                  f"regime={frame.get('regime')}")

        def on_cold(frame: dict) -> None:
            received_frames.append(("cold", frame))
            print(f"  [COLD] received: opportunities={len(frame.get('raw_opportunities', []))}, "
                  f"positions={len(frame.get('open_positions', []))}")

        bus.new_hot_frame.connect(on_hot)
        bus.new_warm_frame.connect(on_warm)
        bus.new_cold_frame.connect(on_cold)

        client = SnapshotStreamClient(
            host="127.0.0.1",
            port=18765,
            frame_handler=bus.feed_bytes,
        )

        bus.start()
        client.start()
        print("[OK] SnapshotBus and client created, waiting for frames...")
        print()

        # ─── Step 3: Wait for frames to arrive ─────────────────────────
        print("[3/5] Waiting for data frames (5 seconds)...")
        # Process Qt events so signal callbacks fire
        for _ in range(50):
            app.processEvents()
            time.sleep(0.1)
        print()

        # ─── Step 4: Validate received frames ──────────────────────────
        print("[4/5] Validating received frames...")
        hot_count = sum(1 for kind, _ in received_frames if kind == "hot")
        warm_count = sum(1 for kind, _ in received_frames if kind == "warm")
        cold_count = sum(1 for kind, _ in received_frames if kind == "cold")

        print(f"  HOT frames:  {hot_count} (expected ~25 at 200ms)")
        print(f"  WARM frames: {warm_count} (expected ~5 at 1s)")
        print(f"  COLD frames: {cold_count} (expected ~1 at 5s)")

        checks = 0
        check_passed = 0

        # Check 1: At least some HOT frames received
        checks += 1
        if hot_count >= 5:
            check_passed += 1
            print("  ✅ HOT frame pipeline works (>=5 frames in 5s)")
        else:
            print("  ❌ Too few HOT frames — dataflow issue")

        # Check 2: At least 1 WARM frame
        checks += 1
        if warm_count >= 1:
            check_passed += 1
            print("  ✅ WARM frame pipeline works")
        else:
            print("  ❌ No WARM frames received")

        # Check 3: At least 1 COLD frame
        checks += 1
        if cold_count >= 1:
            check_passed += 1
            print("  ✅ COLD frame pipeline works")
        else:
            print("  ❌ No COLD frames received")

        # Check 4: Last HOT payload has live values
        last_hot = bus.last_valid_hot_payload
        checks += 1
        if last_hot and last_hot.get("feed_health") in ("HEALTHY", "DELAYED"):
            check_passed += 1
            print(f"  ✅ HOT payload valid: feed={last_hot.get('feed_health')}, "
                  f"spot={last_hot.get('spot')}, mode={last_hot.get('mode')}")
        else:
            print(f"  ❌ HOT payload invalid: {last_hot}")

        # Check 5: Last WARM payload has narrative data
        last_warm = bus.last_valid_warm_payload
        checks += 1
        if last_warm and last_warm.get("narrative_label"):
            check_passed += 1
            print(f"  ✅ WARM payload valid: narrative={last_warm.get('narrative_label')}, "
                  f"regime={last_warm.get('regime')}, session={last_warm.get('session_phase')}")
        else:
            print(f"  ❌ WARM payload invalid: {last_warm}")

        # Check 6: Last COLD payload has opportunity data
        last_cold = bus.last_valid_cold_payload
        checks += 1
        if last_cold and isinstance(last_cold.get("raw_opportunities"), list):
            check_passed += 1
            print(f"  ✅ COLD payload valid: raw_opps={len(last_cold['raw_opportunities'])}, "
                  f"approved={len(last_cold.get('approved_opportunities', []))}")
        else:
            print(f"  ❌ COLD payload invalid: {last_cold}")

        # ─── Step 5: Render panels with projected data ─────────────────
        print()
        print("[5/5] Rendering panels with projected data...")

        # Render with HOT data
        if last_hot:
            hot_results = registry.render_all(last_hot, refresh_class="hot")
            ok_hot = sum(1 for r in hot_results if r.status == "OK")
            deg_hot = sum(1 for r in hot_results if r.status == "DEGRADED")
            print(f"  HOT panels ({len(hot_results)}): OK={ok_hot}, DEGRADED={deg_hot}")

        # Render with WARM data
        if last_warm:
            warm_results = registry.render_all(last_warm, refresh_class="warm")
            ok_warm = sum(1 for r in warm_results if r.status == "OK")
            deg_warm = sum(1 for r in warm_results if r.status == "DEGRADED")
            print(f"  WARM panels ({len(warm_results)}): OK={ok_warm}, DEGRADED={deg_warm}")

        # Render ALL with COLD data
        if last_cold:
            all_results = registry.render_all(last_cold)
            ok_all = sum(1 for r in all_results if r.status == "OK")
            deg_all = sum(1 for r in all_results if r.status == "DEGRADED")
            err_all = sum(1 for r in all_results if r.status == "ERROR")
            checks += 1
            if ok_all > 0:
                check_passed += 1
                print(f"  ALL panels ({len(all_results)}): OK={ok_all}, DEGRADED={deg_all}, ERROR={err_all}")
                print("  ✅ Panels successfully rendered with projected data")
            else:
                print(f"  ALL panels ({len(all_results)}): OK={ok_all}, DEGRADED={deg_all}")
                print("  ⚠️  No OK panels — data may not satisfy all required keys")

        # Also render ALL with HOT data to check
        print()
        full_results = registry.render_all(last_hot or {})
        ok_full = sum(1 for r in full_results if r.status == "OK")
        deg_full = sum(1 for r in full_results if r.status == "DEGRADED")
        err_full = sum(1 for r in full_results if r.status == "ERROR")
        print(f"  Full render with last available data:")
        print(f"    Total:  {len(full_results)}")
        print(f"    OK:     {ok_full}")
        print(f"    Degraded: {deg_full}")
        print(f"    Error:  {err_full}")

        # Show individual panel results
        for r in full_results:
            marker = "✅" if r.status == "OK" else "⚠️" if r.status == "DEGRADED" else "❌"
            payload = r.payload or {}
            sample = ""
            if r.panel_id == "status" and payload.get("feed_health"):
                sample = f"feed={payload['feed_health']}"
            elif r.panel_id == "global_vitals" and payload.get("feed_health"):
                sample = f"feed={payload['feed_health']}, spot={payload.get('spot')}"
            elif r.panel_id == "briefing" and payload.get("narrative_label"):
                sample = f"narrative={payload['narrative_label']}"
            elif r.panel_id == "volume_profile" and payload.get("has_profile"):
                sample = f"has_profile={payload.get('has_profile')}"
            elif r.panel_id == "mtf_chart":
                counts = payload.get("timeframes", {})
                total = sum(counts.values()) if isinstance(counts, dict) else 0
                sample = f"candles={total}"
            elif r.panel_id == "system_health" and payload.get("feed_health"):
                sample = f"feed={payload['feed_health']}"
            elif r.panel_id == "market_overview" and payload.get("spot"):
                sample = f"spot={payload.get('spot')}"
            elif r.panel_id == "opportunity_pipeline" and payload.get("raw_count", 0) > 0:
                sample = f"raw={payload.get('raw_count')}, approved={payload.get('approved_count')}"
            elif r.panel_id == "positions" and payload.get("open_count", 0) > 0:
                sample = f"open={payload.get('open_count')}"
            elif r.panel_id == "brain_status" and payload.get("active_brains"):
                sample = f"brains={payload.get('active_brains')}"
            elif r.panel_id == "engine_health" and payload.get("engine_health"):
                sample = "engine_data_present"
            elif r.panel_id == "option_chain" and payload.get("spot"):
                sample = f"spot={payload.get('spot')}"
            elif r.panel_id == "feature_snapshot" and payload.get("timeframes"):
                sample = "features_present"
            
            print(f"    {marker} {r.panel_id:<25s} {r.status:<12s} {sample}")

        # ─── Summary ───────────────────────────────────────────────────
        print()
        print("=" * 70)
        total_checks = checks
        passed = check_passed
        if passed == total_checks:
            print(f"  RESULT: ✅ ALL {total_checks} VALIDATION CHECKS PASSED")
            print()
            print("  Dashboard dataflow pipeline is FULLY OPERATIONAL:")
            print("  1. TCP server → binary_frame.pack_frame() ✅")
            print("  2. SnapshotStreamClient reads frames ✅")
            print("  3. SnapshotBus.feed_bytes() decodes frames ✅")
            print("  4. state_projection.project_snapshot() maps fields ✅")
            print("  5. PanelRegistry.render_all() displays data ✅")
            print("  6. HOT/WARM/COLD tier separation works ✅")
            exit_code = 0
        else:
            print(f"  RESULT: ⚠️  {passed}/{total_checks} checks passed")
            exit_code = 1

        print()
        print("  TEMPORARY VALIDATION INFRASTRUCTURE")
        print("  When real backend IPC is ready:")
        print("    1. Stop the fake publisher (Ctrl+C)")
        print("    2. Delete tools/fake_snapshot_publisher.py")
        print("    3. Delete tools/validate_dataflow_direct.py")
        print("    4. Delete tools/run_fake_validation.py")
        print("=" * 70)

    finally:
        # Cleanup
        try:
            client.stop()
        except Exception:
            pass
        try:
            bus.stop()
        except Exception:
            pass
        publisher.terminate()
        try:
            publisher.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            publisher.kill()
            publisher.wait()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())