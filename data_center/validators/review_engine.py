"""
Junior Aladdin — Review Engine
===============================
Week 4 Phase 1: compare raw vs structured storage and produce a verification
report for the major data center layer.

The engine is intentionally read-only with respect to market data. It only
scans parquet files, validates schema shape, checks timestamp continuity, and
records corruption/mismatch signals.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl
from loguru import logger


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.storage_config import MAJOR_RAW, MAJOR_STRUCTURED, MAJOR_REVIEW
from data_center.schemas.options_schema import RAW_OPTIONS_SCHEMA, STRUCTURED_OPTIONS_SCHEMA
from data_center.schemas.tick_schema import RAW_TICK_SCHEMA, STRUCTURED_TICK_SCHEMA
from data_center.utils.file_utils import safe_write_json
from data_center.utils.parquet_utils import read_parquet


_RAW_SCHEMA_SETS = {
    frozenset(RAW_TICK_SCHEMA.keys()),
    frozenset(RAW_OPTIONS_SCHEMA.keys()),
}

_STRUCTURED_SCHEMA_SETS = {
    frozenset(STRUCTURED_TICK_SCHEMA.keys()),
    frozenset(STRUCTURED_OPTIONS_SCHEMA.keys()),
}


@dataclass(slots=True)
class ReviewFileResult:
    path: str
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    schema_ok: bool = False
    continuity_ok: bool = True
    corruption: bool = False
    timestamp_duplicates: int = 0
    timestamp_out_of_order: bool = False
    error: Optional[str] = None


@dataclass(slots=True)
class ReviewSummary:
    generated_at: str
    raw_root: str
    structured_root: str
    raw_files: int = 0
    structured_files: int = 0
    raw_rows: int = 0
    structured_rows: int = 0
    dropped_rows: int = 0
    corruption_count: int = 0
    schema_mismatch_count: int = 0
    continuity_issue_count: int = 0
    verified: bool = False
    raw_file_results: list[ReviewFileResult] = field(default_factory=list)
    structured_file_results: list[ReviewFileResult] = field(default_factory=list)


class ReviewEngine:
    """Compare raw and structured parquet trees and write a verification report."""

    def __init__(
        self,
        raw_root: Path = MAJOR_RAW,
        structured_root: Path = MAJOR_STRUCTURED,
        review_root: Path = MAJOR_REVIEW,
        report_name: str = "verification.json",
    ):
        self.raw_root = raw_root
        self.structured_root = structured_root
        self.review_root = review_root
        self.report_path = review_root / report_name
        self.archive_dir = review_root / "archives"

    def review(self) -> dict[str, Any]:
        """Run the week 4 phase-1 review and persist a JSON report."""
        raw_results = self._scan_tree(self.raw_root, expected_layer="raw")
        structured_results = self._scan_tree(self.structured_root, expected_layer="structured")

        raw_rows = sum(item.rows for item in raw_results)
        structured_rows = sum(item.rows for item in structured_results)
        dropped_rows = max(raw_rows - structured_rows, 0)

        corruption_count = sum(1 for item in raw_results + structured_results if item.corruption)
        schema_mismatch_count = sum(1 for item in raw_results + structured_results if not item.schema_ok)
        continuity_issue_count = sum(
            1
            for item in raw_results + structured_results
            if (not item.continuity_ok) or item.timestamp_out_of_order or item.timestamp_duplicates > 0
        )

        summary = ReviewSummary(
            generated_at=datetime.now(timezone.utc).isoformat(),
            raw_root=str(self.raw_root),
            structured_root=str(self.structured_root),
            raw_files=len(raw_results),
            structured_files=len(structured_results),
            raw_rows=raw_rows,
            structured_rows=structured_rows,
            dropped_rows=dropped_rows,
            corruption_count=corruption_count,
            schema_mismatch_count=schema_mismatch_count,
            continuity_issue_count=continuity_issue_count,
            verified=(
                raw_rows > 0
                and structured_rows > 0
                and dropped_rows == 0
                and corruption_count == 0
                and schema_mismatch_count == 0
                and continuity_issue_count == 0
            ),
            raw_file_results=raw_results,
            structured_file_results=structured_results,
        )

        report = self._serialize_summary(summary)
        safe_write_json(self.report_path, report)
        self._archive_report(report, summary.generated_at)

        logger.info(
            "Review verification report written",
            raw_rows=raw_rows,
            structured_rows=structured_rows,
            dropped_rows=dropped_rows,
            verified=summary.verified,
            report_path=str(self.report_path),
            archive_dir=str(self.archive_dir),
        )
        return report

    def _scan_tree(self, root: Path, expected_layer: str) -> list[ReviewFileResult]:
        if not root.exists():
            return []

        parquet_files = sorted(root.rglob("*.parquet"))
        results: list[ReviewFileResult] = []
        for path in parquet_files:
            results.append(self._scan_file(path, expected_layer=expected_layer))
        return results

    def _scan_file(self, path: Path, expected_layer: str) -> ReviewFileResult:
        result = ReviewFileResult(path=str(path))

        try:
            frame = read_parquet(path)
            if frame is None:
                result.corruption = True
                result.error = "missing_or_unreadable"
                return result

            result.rows = len(frame)
            result.columns = list(frame.columns)
            result.schema_ok = self._schema_matches(frame)
            result.continuity_ok, result.timestamp_duplicates, result.timestamp_out_of_order = self._check_continuity(frame)

            if not result.schema_ok:
                result.error = f"schema_mismatch:{expected_layer}"

        except Exception as exc:
            result.corruption = True
            result.error = f"corrupt:{type(exc).__name__}:{exc}"

        return result

    def _schema_matches(self, frame: pl.DataFrame) -> bool:
        columns = frozenset(frame.columns)
        return columns in _RAW_SCHEMA_SETS or columns in _STRUCTURED_SCHEMA_SETS

    def _check_continuity(self, frame: pl.DataFrame) -> tuple[bool, int, bool]:
        if "timestamp" not in frame.columns:
            return False, 0, False

        timestamps: list[int] = []
        for value in frame.get_column("timestamp").to_list():
            if value is None:
                continue
            try:
                timestamps.append(int(value))
            except (TypeError, ValueError):
                return False, 0, False

        if not timestamps:
            return False, 0, False

        duplicates = len(timestamps) - len(set(timestamps))
        out_of_order = timestamps != sorted(timestamps)
        continuity_ok = duplicates == 0 and not out_of_order
        return continuity_ok, max(0, duplicates), out_of_order

    def _serialize_summary(self, summary: ReviewSummary) -> dict[str, Any]:
        return {
            "generated_at": summary.generated_at,
            "raw_root": summary.raw_root,
            "structured_root": summary.structured_root,
            "raw_files": summary.raw_files,
            "structured_files": summary.structured_files,
            "raw_rows": summary.raw_rows,
            "structured_rows": summary.structured_rows,
            "dropped_rows": summary.dropped_rows,
            "corruption_count": summary.corruption_count,
            "schema_mismatch_count": summary.schema_mismatch_count,
            "continuity_issue_count": summary.continuity_issue_count,
            "verified": summary.verified,
            "raw_file_results": [self._serialize_file_result(item) for item in summary.raw_file_results],
            "structured_file_results": [self._serialize_file_result(item) for item in summary.structured_file_results],
        }

    def _archive_report(self, report: dict[str, Any], generated_at: str) -> Optional[Path]:
        try:
            stamp = generated_at.replace(":", "-").replace(".", "-")
            archive_path = self.archive_dir / f"verification_{stamp}.json"
            safe_write_json(archive_path, report)
            latest_link = self.archive_dir / "latest_verification.json"
            safe_write_json(latest_link, report)

            # Keep a simple top-level copy for operators who check the root review dir.
            # This mirrors the roadmap's review/verification.json contract while also
            # providing a timestamped historical archive.
            review_copy = self.review_root / "verification.json"
            safe_write_json(review_copy, report)

            return archive_path
        except Exception as exc:
            logger.warning("Failed to archive verification report", error=str(exc))
            return None

    @staticmethod
    def _serialize_file_result(result: ReviewFileResult) -> dict[str, Any]:
        return {
            "path": result.path,
            "rows": result.rows,
            "columns": result.columns,
            "schema_ok": result.schema_ok,
            "continuity_ok": result.continuity_ok,
            "corruption": result.corruption,
            "timestamp_duplicates": result.timestamp_duplicates,
            "timestamp_out_of_order": result.timestamp_out_of_order,
            "error": result.error,
        }


review_engine = ReviewEngine()


def _run_tests() -> None:
    from pathlib import Path
    import tempfile

    import polars as pl

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        raw_root = tmp_root / "data_center" / "major" / "raw"
        structured_root = tmp_root / "data_center" / "major" / "structured"
        review_root = tmp_root / "data_center" / "major" / "review"

        raw_path = raw_root / "2026-05-25" / "NIFTY" / "10_11.parquet"
        structured_path = structured_root / "2026-05-25" / "NIFTY" / "expiry" / "10_11.parquet"

        raw_frame = pl.DataFrame(
            [
                {
                    "token": "99926000",
                    "ltp": 18500.5,
                    "volume": 123,
                    "open": 18400.0,
                    "high": 18520.0,
                    "low": 18380.0,
                    "close": 18450.0,
                    "timestamp": 1716600000000,
                    "direction": 1,
                }
            ]
        )
        structured_frame = pl.DataFrame(
            [
                {
                    "token": "99926000",
                    "ltp": 18500.5,
                    "volume": 123,
                    "open": 18400.0,
                    "high": 18520.0,
                    "low": 18380.0,
                    "close": 18450.0,
                    "timestamp": 1716600000000,
                    "direction": 1,
                    "sequence": 1,
                    "exchange": "NSE",
                    "symbol": "NIFTY",
                }
            ]
        )

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        raw_frame.write_parquet(raw_path)
        structured_frame.write_parquet(structured_path)

        engine = ReviewEngine(raw_root=raw_root, structured_root=structured_root, review_root=review_root)
        report = engine.review()

        assert report["raw_rows"] == 1
        assert report["structured_rows"] == 1
        assert report["dropped_rows"] == 0
        assert report["corruption_count"] == 0
        assert report["schema_mismatch_count"] == 0
        assert report["continuity_issue_count"] == 0
        assert report["verified"] is True
        assert engine.report_path.exists()


if __name__ == "__main__":
    _run_tests()