"""
Junior Aladdin — File Utilities
=================================
Functions for file operations, safety checks, and batch management.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def safe_write_text(filepath: Path, content: str, encoding: str = "utf-8") -> None:
    """Safely write text content to a file, creating directories as needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding=encoding)


def safe_read_text(filepath: Path, encoding: str = "utf-8") -> Optional[str]:
    """Safely read text content from a file, returning None if missing."""
    if not filepath.exists():
        return None
    return filepath.read_text(encoding=encoding)


def safe_write_json(filepath: Path, data: Any, indent: int = 2) -> None:
    """Safely write JSON data to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")


def safe_read_json(filepath: Path) -> Optional[Any]:
    """Safely read JSON data from a file, returning None if missing."""
    if not filepath.exists():
        return None
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def safe_delete(filepath: Path) -> bool:
    """
    Safely delete a file. Returns True if deleted, False if not found.
    Does NOT raise exceptions.
    """
    try:
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    except Exception:
        return False


def safe_remove_directory(dirpath: Path) -> bool:
    """Safely remove a directory and all its contents."""
    try:
        if dirpath.exists():
            shutil.rmtree(dirpath)
            return True
        return False
    except Exception:
        return False


def count_files(directory: Path, pattern: str = "*.parquet") -> int:
    """Count files matching a pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def total_bytes(directory: Path, pattern: str = "*") -> int:
    """Calculate total byte size of files matching a pattern."""
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.glob(pattern) if f.is_file())


def batch_delete_files(filepaths: List[Path]) -> Dict[str, int]:
    """
    Delete a batch of files. Returns stats.
    
    Returns:
        {"deleted": N, "failed": M}
    """
    result = {"deleted": 0, "failed": 0}
    for fp in filepaths:
        if safe_delete(fp):
            result["deleted"] += 1
        else:
            result["failed"] += 1
    return result


def get_disk_usage(path: Path) -> Dict[str, Any]:
    """Get disk usage info for a path."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    
    files = list(path.rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    file_count = sum(1 for f in files if f.is_file())
    dir_count = sum(1 for f in files if f.is_dir())
    
    return {
        "path": str(path),
        "exists": True,
        "total_bytes": total,
        "file_count": file_count,
        "dir_count": dir_count,
    }