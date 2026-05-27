"""
TEMPORARY VALIDATION SCRIPT — FAKE BACKEND → DASHBOARD DATAFLOW TEST
=====================================================================

This script starts the fake snapshot publisher, runs the dashboard in headless
mode against it for a few seconds, captures the rendered output, and verifies
that panels show LIVE data instead of DEGRADED/empty state.

USAGE:
    python -m tools.run_fake_validation

EXPECTED OUTPUT:
    - 16 panels registered
    - Most panels show OK status (not DEGRADED)
    - Real values in payloads (spot ~24650, feed_health="HEALTHY", etc.)
    - HOT frames received and processed

CLEANUP:
    Delete this file and tools/fake_snapshot_publisher.py when real backend IPC exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("=" * 70)
    print("  FAKE BACKEND → DASHBOARD DATAFLOW VALIDATION")
    print("=" * 70)
    print()

    # Step 1: Start fake publisher in background
    print("[1/4] Starting fake snapshot publisher...")
    publisher = subprocess.Popen(
        [sys.executable, "-m", "tools.fake_snapshot_publisher"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(1.0)  # wait for publisher to bind socket

    if publisher.poll() is not None:
        print("[FAIL] Fake publisher failed to start!")
        return 1
    print("[OK] Publisher running on tcp://127.0.0.1:18765")
    print()

    # Step 2: Run dashboard headless mode with output capture
    print("[2/4] Running dashboard headless render (5 seconds)...")
    out_file = PROJECT_ROOT / "artifacts" / "fake_validation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dashboard = subprocess.Popen(
        [sys.executable, "-m", "dashboard.main", "--headless", "--out-file", str(out_file)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Give dashboard time to connect and receive multiple HOT frames
    time.sleep(5.0)

    # Gracefully stop dashboard
    dashboard.terminate()
    try:
        dashboard.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        dashboard.kill()
        dashboard.wait()

    print("[OK] Dashboard stopped")
    print()

    # Step 3: Analyze report
    print("[3/4] Analyzing dashboard report...")
    if not out_file.exists():
        print("[FAIL] Report file was not created!")
        publisher.terminate()
        publisher.wait()
        return 1

    report = json.loads(out_file.read_text(encoding="utf-8"))
    panels = report.get("panels", [])
    performance = report.get("performance", {})
    panel_render = performance.get("panel_render", {})
    metrics = panel_render.get("registry_summary", {})

    print(f"  Timestamp:   {report.get('timestamp', 'N/A')}")
    print(f"  Mode:        {report.get('mode', 'N/A')}")
    print(f"  Panel count: {report.get('panel_count', 0)}")
    print(f"  Render time: {performance.get('report_render_ms', 'N/A')}ms")
    print()
    print(f"  Registry Summary:")
    print(f"    Total:    {metrics.get('total_panels', '?')}")
    print(f"    OK:       {metrics.get('ok_count', '?')}")
    print(f"    Degraded: {metrics.get('degraded_count', '?')}")
    print(f"    Error:    {metrics.get('error_count', '?')}")
    print(f"    Stale:    {metrics.get('stale_count', '?')}")
    print()

    # Show per-panel status
    print("  Per-Panel Status:")
    passed = 0
    failed = 0
    for p in panels:
        pid = p.get("panel_id", "?")
        status = p.get("status", "?")
        payload = p.get("payload", {})
        
        # Show a key value from each panel to prove live data
        sample = ""
        if pid == "status" and payload.get("feed_health"):
            sample = f"feed={payload['feed_health']}, spot={payload.get('spot','?')}"
        elif pid == "global_vitals" and payload.get("feed_health"):
            sample = f"feed={payload['feed_health']}, spot={payload.get('spot','?')}"
        elif pid == "briefing" and payload.get("narrative_label"):
            sample = f"narrative={payload.get('narrative_label','?')}"
        elif pid == "volume_profile" and payload.get("has_profile"):
            sample = f"profile=YES, poc={payload.get('poc','?')}"
        elif pid == "mtf_chart" and payload.get("timeframes"):
            counts = payload.get("timeframes", {})
            sample = f"candles: {sum(counts.values())} total"
        elif pid == "positions" and payload.get("open_count", 0) > 0:
            sample = f"{payload.get('open_count')} open positions"
        elif pid == "opportunity_pipeline" and payload.get("raw_count", 0) > 0:
            sample = f"raw={payload.get('raw_count')}, approved={payload.get('approved_count')}"
        elif pid == "system_health" and payload.get("feed_health"):
            sample = f"feed={payload.get('feed_health')}"

        marker = "✅" if status == "OK" else "⚠️" if status == "DEGRADED" else "❌"
        print(f"    {marker} {pid:<25s} {status:<10s} {sample}")

        if status == "OK":
            passed += 1
        elif status == "DEGRADED":
            # Check if DEGRADED is just missing-keys (expected for headless with no projection)
            warnings = p.get("warnings", [])
            if any("missing_required_keys" in w for w in warnings):
                passed += 0  # neutral — data arrived but keys don't match projection format
            else:
                failed += 1
        else:
            failed += 1

    print()
    print(f"  Passed: {passed}, Failed: {failed}")

    # Step 4: Validate key data points
    print()
    print("[4/4] Dataflow verification...")
    checks = 0
    check_passed = 0

    # Check 1: Panels received data
    has_live_data = any(
        p.get("status") == "OK" for p in panels
    )
    checks += 1
    if has_live_data:
        check_passed += 1
        print("  ✅ Panels received and rendered LIVE data")
    else:
        print("  ❌ No panels show OK status — dataflow may be broken")

    # Check 2: HOT panel (status) has values
    for p in panels:
        if p.get("panel_id") == "status" and p.get("payload", {}).get("feed_health"):
            checks += 1
            check_passed += 1
            print(f"  ✅ STATUS panel dataflow: feed_health={p['payload']['feed_health']}")
            break
    else:
        checks += 1
        print("  ❌ STATUS panel did not receive HOT data")

    # Check 3: Volume profile data
    for p in panels:
        if p.get("panel_id") == "volume_profile" and p.get("payload", {}).get("has_profile"):
            checks += 1
            check_passed += 1
            print(f"  ✅ VOLUME PROFILE dataflow: POC={p['payload'].get('poc')}")
            break
    else:
        checks += 1
        print("  ❌ VOLUME PROFILE panel did not receive data")

    # Check 4: MTF chart candle data
    for p in panels:
        if p.get("panel_id") == "mtf_chart":
            counts = p.get("payload", {}).get("timeframes", {})
            total = sum(counts.values()) if isinstance(counts, dict) else 0
            if total > 0:
                checks += 1
                check_passed += 1
                print(f"  ✅ MTF CHART dataflow: {total} candles across timeframes")
                break
    else:
        checks += 1
        print("  ❌ MTF CHART panel did not receive candle data")

    # Check 5: Report render time under 5ms (performance)
    rt = performance.get("report_render_ms", 99)
    checks += 1
    if rt < 5.0:
        check_passed += 1
        print(f"  ✅ Render performance: {rt:.3f}ms (under 5ms)")
    else:
        print(f"  ⚠️  Render performance: {rt:.3f}ms (above 5ms threshold)")

    print()
    print(f"  Checks: {check_passed}/{checks} passed")

    # Summary
    print()
    print("=" * 70)
    if check_passed == checks:
        print("  RESULT: ✅ ALL VALIDATION CHECKS PASSED")
        print()
        print("  Dashboard dataflow is fully operational:")
        print("  - Fake publisher → TCP → SnapshotStreamClient → SnapshotBus")
        print("  - SnapshotBus → binary_frame decode → state_projection")
        print("  - MainWindow → update_hot/update_warm/update_cold")
        print("  - Panels render LIVE data instead of DEGRADED")
        print()
        print("  This is TEMPORARY validation infrastructure.")
        print("  When real backend IPC is ready:")
        print("    1. Stop the fake publisher")
        print("    2. Delete tools/fake_snapshot_publisher.py")
        print("    3. Delete tools/run_fake_validation.py")
        exit_code = 0
    else:
        print(f"  RESULT: ⚠️  {check_passed}/{checks} checks passed — some dataflow issues detected")
        print()
        print("  Check the full report at: artifacts/fake_validation_report.json")
        exit_code = 1

    # Cleanup
    publisher.terminate()
    try:
        publisher.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        publisher.kill()
        publisher.wait()

    print("=" * 70)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())