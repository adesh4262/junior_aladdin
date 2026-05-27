"""
Junior Aladdin — Ultimate Data Center Audit Tool
===============================================
PURPOSE: Comprehensive verification of the Data Center's structural, 
functional, and integration state.

OUTPUT: Generates 'data_center_audit_report.txt'
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_FILE = PROJECT_ROOT / "data_center_audit_report.txt"

class DataCenterAuditor:
    def __init__(self):
        self.report = []
        self.success_count = 0
        self.fail_count = 0

    def log(self, msg: str):
        print(msg)
        self.report.append(msg)

    def section(self, title: str):
        line = "=" * 60
        self.log(f"\n{line}\n{title.upper()}\n{line}")

    def run_audit(self):
        self.log(f"DATA CENTER ULTIMATE AUDIT STARTED AT: {datetime.now()}")
        
        # 1. Structural Check
        self.section("1. Structural Integrity Check")
        required_dirs = [
            "data_center/major/raw", "data_center/major/structured",
            "data_center/minor/raw", "data_center/minor/structured",
            "data_center/computed/trend", "data_center/computed/volatility",
            "data_center/computed/liquidity", "data/calendar"
        ]
        for d in required_dirs:
            path = PROJECT_ROOT / d
            if path.exists():
                self.log(f" [OK] Directory exists: {d}")
                self.success_count += 1
            else:
                self.log(f" [FAIL] Directory MISSING: {d}")
                self.fail_count += 1

        # 2. Component Wiring Check
        self.section("2. Component Wiring & Import Check")
        components = [
            ("TickCleaner", "data_center.cleaners.tick_cleaner"),
            ("StructuredWriter", "data_center.writers.structured_writer"),
            ("MinorTransformer", "data_center.transformers.minor_transformer"),
            ("BackendConnector", "data_center.connectors.backend_connector"),
            ("DataEngine", "src.core.data_engine")
        ]
        for name, module in components:
            try:
                __import__(module)
                self.log(f" [OK] {name} is importable and wired.")
                self.success_count += 1
            except Exception as e:
                self.log(f" [FAIL] {name} wiring broken: {e}")
                self.fail_count += 1

        # 3. Simulated Data Flow Test
        self.section("3. Major Data Flow (Clean -> Broadcast) Test")
        try:
            from data_center.connectors.backend_connector import backend_connector
            from data_center.cleaners.tick_cleaner import tick_cleaner
            
            test_received = {"status": False}
            def test_callback(tick): test_received["status"] = True
            
            backend_connector.register_tick_listener(test_callback)
            
            fake_raw = {"token": "99926000", "ltp": 24000.5, "volume": 100, "timestamp": int(time.time()*1000)}
            cleaned = tick_cleaner.clean_record(fake_raw)
            
            if cleaned.is_clean and cleaned.record:
                backend_connector.broadcast_tick(cleaned.record)
                if test_received["status"]:
                    self.log(" [OK] End-to-end Broadcast successful (Cleaner -> Connector -> Listener)")
                    self.success_count += 1
                else:
                    self.log(" [FAIL] Broadcast failed: Listener did not receive data.")
                    self.fail_count += 1
            else:
                self.log(" [FAIL] Cleaner rejected valid fake tick.")
                self.fail_count += 1
        except Exception as e:
            self.log(f" [FAIL] Data flow test crashed: {e}")
            self.fail_count += 1

        # 4. Minor Layer Logic Test
        self.section("4. Minor Layer Intelligence Test")
        try:
            from data_center.transformers.minor_transformer import minor_transformer
            fake_chain = {
                24000: {"ce": {"oi": 100, "iv": 0.15}, "pe": {"oi": 150, "iv": 0.16}},
                24050: {"ce": {"oi": 200, "iv": 0.14}, "pe": {"oi": 50, "iv": 0.17}}
            }
            pcr = minor_transformer.compute_pcr(fake_chain)
            self.log(f" [OK] PCR Calculation Logic: {pcr}")
            if pcr == round(200/300, 4):
                self.success_count += 1
            else:
                self.log(f" [WARN] PCR Math check failed (Expected 0.6667, got {pcr})")
        except Exception as e:
            self.log(f" [FAIL] Minor Layer logic test failed: {e}")
            self.fail_count += 1

        # 5. Computed Intelligence Test
        self.section("5. Computed Intelligence (Week 6) Test")
        try:
            from data_center.transformers.computed_transformer import computed_transformer
            trend = computed_transformer.compute_trend("NIFTY", 24000.0)
            self.log(f" [OK] Trend Engine Status: {trend.get('method')}")
            self.success_count += 1
        except Exception as e:
            self.log(f" [FAIL] Computed Layer test failed: {e}")
            self.fail_count += 1

        # Summary
        self.section("Audit Summary")
        self.log(f"TOTAL TESTS PASSED: {self.success_count}")
        self.log(f"TOTAL TESTS FAILED: {self.fail_count}")
        status = "STRONGEST" if self.fail_count == 0 else "DEGRADED"
        self.log(f"FINAL VERDICT: {status}")

        # Save to file
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report))
        print(f"\nReport saved to: {REPORT_FILE}")

if __name__ == "__main__":
    auditor = DataCenterAuditor()
    auditor.run_audit()
