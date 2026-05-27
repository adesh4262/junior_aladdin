"""
Junior Aladdin — Retention Engine
=================================
Week 4 Phase 3: enforce raw and review retention policies.

The engine only removes storage that the roadmap marks as disposable:
raw data older than RAW_RETENTION_DAYS and review verification reports older
than REVIEW_RETENTION_DAYS. Structured storage is permanent and is never
deleted by this engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from configs.retention_config import RAW_RETENTION_DAYS, REVIEW_RETENTION_DAYS
from configs.storage_config import MAJOR_RAW, MAJOR_REVIEW, MAJOR_STRUCTURED
from data_center.utils.file_utils import safe_delete, safe_remove_directory


@dataclass(slots=True)
class RetentionAction:
    path: str
    reason: str
    deleted: bool = False


@dataclass(slots=True)
class RetentionReport:
    generated_at: str
    raw_root: str
    review_root: str
    structured_root: str
    raw_scanned: int = 0
    raw_deleted: int = 0
    raw_removed_dirs: int = 0
    review_scanned: int = 0
    review_deleted: int = 0
    review_removed_dirs: int = 0
    structured_scanned: int = 0
    structured_deleted: int = 0
    verified: bool = True
    actions: list[RetentionAction] = field(default_factory=list)


class RetentionEngine:
    """Apply age-based cleanup rules to data-center storage."""

    def __init__(
        self,
        raw_root: Path = MAJOR_RAW,
        review_root: Path = MAJOR_REVIEW,
        structured_root: Path = MAJOR_STRUCTURED,
        raw_retention_days: int = RAW_RETENTION_DAYS,
        review_retention_days: int = REVIEW_RETENTION_DAYS,
    ):
        self.raw_root = raw_root
        self.review_root = review_root
        self.structured_root = structured_root
        self.raw_retention_days = int(raw_retention_days)
        self.review_retention_days = int(review_retention_days)

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Run raw and review cleanup. Structured storage is never touched."""
        raw_report = self._cleanup_date_partition_root(self.raw_root, self.raw_retention_days, dry_run=dry_run, label="raw")
        review_report = self._cleanup_review_root(self.review_root, self.review_retention_days, dry_run=dry_run)

        report = RetentionReport(
            generated_at=datetime.utcnow().isoformat() + "Z",
            raw_root=str(self.raw_root),
            review_root=str(self.review_root),
            structured_root=str(self.structured_root),
            raw_scanned=raw_report["scanned"],
            raw_deleted=raw_report["deleted"],
            raw_removed_dirs=raw_report["removed_dirs"],
            review_scanned=review_report["scanned"],
            review_deleted=review_report["deleted"],
            review_removed_dirs=review_report["removed_dirs"],
            structured_scanned=self._count_parquet_files(self.structured_root),
            structured_deleted=0,
            verified=True,
            actions=raw_report["actions"] + review_report["actions"],
        )

        logger.info(
            "Retention run complete",
            dry_run=dry_run,
            raw_deleted=report.raw_deleted,
            review_deleted=report.review_deleted,
            structured_deleted=report.structured_deleted,
        )
        return self._serialize_report(report)

    def _cleanup_date_partition_root(self, root: Path, retention_days: int, *, dry_run: bool, label: str) -> dict[str, Any]:
        cutoff_date = date.today() - timedelta(days=max(0, int(retention_days)))
        scanned = 0
        deleted = 0
        removed_dirs = 0
        actions: list[RetentionAction] = []

        if not root.exists():
            return {"scanned": 0, "deleted": 0, "removed_dirs": 0, "actions": actions}

        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir():
                continue
            scanned += 1
            partition_date = self._parse_date_dir(date_dir.name)
            if partition_date is None:
                continue
            if partition_date >= cutoff_date:
                continue

            files = [path for path in date_dir.rglob("*.parquet") if path.is_file()]
            for file_path in files:
                if not dry_run and safe_delete(file_path):
                    deleted += 1
                    actions.append(RetentionAction(path=str(file_path), reason=f"expired_{label}_file", deleted=True))
                elif dry_run:
                    deleted += 1
                    actions.append(RetentionAction(path=str(file_path), reason=f"dry_run_expired_{label}_file", deleted=False))

            # Remove empty symbol/date directories after file deletion.
            if not dry_run:
                if self._prune_empty_dirs(date_dir):
                    removed_dirs += 1
            else:
                removed_dirs += 1

        return {"scanned": scanned, "deleted": deleted, "removed_dirs": removed_dirs, "actions": actions}

    def _cleanup_review_root(self, root: Path, retention_days: int, *, dry_run: bool) -> dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=max(0, int(retention_days)))
        scanned = 0
        deleted = 0
        removed_dirs = 0
        actions: list[RetentionAction] = []

        if not root.exists():
            return {"scanned": 0, "deleted": 0, "removed_dirs": 0, "actions": actions}

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            scanned += 1
            if path.suffix.lower() != ".json":
                continue

            report_dt = self._parse_review_report_datetime(path)
            if report_dt is not None:
                if report_dt >= cutoff:
                    continue
            else:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime >= cutoff:
                    continue

            if dry_run:
                deleted += 1
                actions.append(RetentionAction(path=str(path), reason="dry_run_expired_review_report", deleted=False))
            elif safe_delete(path):
                deleted += 1
                actions.append(RetentionAction(path=str(path), reason="expired_review_report", deleted=True))

        if not dry_run:
            removed_dirs = self._prune_empty_dirs(root)
        else:
            removed_dirs = 1 if root.exists() else 0

        return {"scanned": scanned, "deleted": deleted, "removed_dirs": removed_dirs, "actions": actions}

    def _prune_empty_dirs(self, root: Path) -> int:
        removed = 0
        if not root.exists():
            return 0

        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    next(path.iterdir())
                except StopIteration:
                    if safe_remove_directory(path):
                        removed += 1
        return removed

    @staticmethod
    def _parse_date_dir(name: str) -> Optional[date]:
        try:
            return datetime.strptime(name, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _parse_review_report_datetime(path: Path) -> Optional[datetime]:
        """Return a report timestamp from archive filenames when present."""
        if path.name in {"verification.json", "latest_verification.json"}:
            return None

        match = re.match(r"^verification_(\d{4}-\d{2}-\d{2})", path.stem)
        if not match:
            return None

        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d")
        except ValueError:
            return None

    @staticmethod
    def _count_parquet_files(root: Path) -> int:
        if not root.exists():
            return 0
        return sum(1 for path in root.rglob("*.parquet") if path.is_file())

    def _serialize_report(self, report: RetentionReport) -> dict[str, Any]:
        return {
            "generated_at": report.generated_at,
            "raw_root": report.raw_root,
            "review_root": report.review_root,
            "structured_root": report.structured_root,
            "raw_scanned": report.raw_scanned,
            "raw_deleted": report.raw_deleted,
            "raw_removed_dirs": report.raw_removed_dirs,
            "review_scanned": report.review_scanned,
            "review_deleted": report.review_deleted,
            "review_removed_dirs": report.review_removed_dirs,
            "structured_scanned": report.structured_scanned,
            "structured_deleted": report.structured_deleted,
            "verified": report.verified,
            "actions": [
                {"path": action.path, "reason": action.reason, "deleted": action.deleted}
                for action in report.actions
            ],
        }


retention_engine = RetentionEngine()


def _run_tests() -> None:
    import tempfile

    import polars as pl

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        raw_root = tmp_root / "data_center" / "major" / "raw"
        review_root = tmp_root / "data_center" / "major" / "review"
        structured_root = tmp_root / "data_center" / "major" / "structured"

        old_raw_path = raw_root / "2026-05-20" / "NIFTY" / "10_11.parquet"
        fresh_raw_path = raw_root / "2026-05-25" / "NIFTY" / "10_11.parquet"
        old_review_path = review_root / "archives" / "verification_2026-05-01T10-00-00Z.json"
        fresh_review_path = review_root / "verification.json"
        structured_path = structured_root / "2026-05-20" / "NIFTY" / "expiry" / "10_11.parquet"

        raw_frame = pl.DataFrame([
            {
                "token": "99926000",
                "ltp": 18500.5,
                "volume": 123,
                "open": 18400.0,
                "high": 18520.0,
                "low": 18380.0,
                "close": 18450.0,
                "timestamp": 1716177600000,
                "direction": 1,
            }
        ])
        structured_frame = pl.DataFrame([
            {
                "token": "99926000",
                "ltp": 18500.5,
                "volume": 123,
                "open": 18400.0,
                "high": 18520.0,
                "low": 18380.0,
                "close": 18450.0,
                "timestamp": 1716177600000,
                "direction": 1,
                "sequence": 1,
                "exchange": "NSE",
                "symbol": "NIFTY",
            }
        ])

        old_raw_path.parent.mkdir(parents=True, exist_ok=True)
        fresh_raw_path.parent.mkdir(parents=True, exist_ok=True)
        old_review_path.parent.mkdir(parents=True, exist_ok=True)
        fresh_review_path.parent.mkdir(parents=True, exist_ok=True)
        structured_path.parent.mkdir(parents=True, exist_ok=True)

        raw_frame.write_parquet(old_raw_path)
        raw_frame.write_parquet(fresh_raw_path)
        structured_frame.write_parquet(structured_path)
        safe_write_json(old_review_path, {"verified": True})
        safe_write_json(fresh_review_path, {"verified": True})

        engine = RetentionEngine(raw_root=raw_root, review_root=review_root, structured_root=structured_root, raw_retention_days=3, review_retention_days=30)
        report = engine.run(dry_run=False)

        assert report["raw_deleted"] >= 1
        assert report["review_deleted"] >= 1
        assert report["structured_deleted"] == 0
        assert structured_path.exists()


if __name__ == "__main__":
    _run_tests()