"""
Junior Aladdin — Storage Configuration
=======================================
Defines storage paths, formats, partitioning, and file rotation settings.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# PROJECT ROOT
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CENTER_ROOT = PROJECT_ROOT / "data_center"

# ──────────────────────────────────────────────
# MAJOR DATA PATHS
# ──────────────────────────────────────────────
MAJOR_RAW = DATA_CENTER_ROOT / "major" / "raw"
MAJOR_CLEANED = DATA_CENTER_ROOT / "major" / "cleaned"
MAJOR_STRUCTURED = DATA_CENTER_ROOT / "major" / "structured"
MAJOR_REVIEW = DATA_CENTER_ROOT / "major" / "review"

# ──────────────────────────────────────────────
# MINOR DATA PATHS
# ──────────────────────────────────────────────
MINOR_RAW = DATA_CENTER_ROOT / "minor" / "raw"
MINOR_CLEANED = DATA_CENTER_ROOT / "minor" / "cleaned"
MINOR_STRUCTURED = DATA_CENTER_ROOT / "minor" / "structured"
MINOR_REVIEW = DATA_CENTER_ROOT / "minor" / "review"

# ──────────────────────────────────────────────
# COMPUTED DATA PATHS
# ──────────────────────────────────────────────
COMPUTED_ROOT = DATA_CENTER_ROOT / "computed"
COMPUTED_VOLATILITY = COMPUTED_ROOT / "volatility"
COMPUTED_STRUCTURE = COMPUTED_ROOT / "structure"
COMPUTED_LIQUIDITY = COMPUTED_ROOT / "liquidity"
COMPUTED_ORDERFLOW = COMPUTED_ROOT / "orderflow"
COMPUTED_TREND = COMPUTED_ROOT / "trend"
COMPUTED_BIAS = COMPUTED_ROOT / "bias"

# ──────────────────────────────────────────────
# FILE ROTATION (adaptive — seconds per file)
# ──────────────────────────────────────────────
# High tick load  -> 1 hour
# Moderate load   -> 2 hours
# Low load        -> 4 hours
ROTATION_HIGH = 3600       # 1 hour
ROTATION_MODERATE = 7200   # 2 hours
ROTATION_LOW = 14400       # 4 hours

# Default rotation strategy
ROTATION_DEFAULT = ROTATION_MODERATE

# ──────────────────────────────────────────────
# PARTITION FORMAT
# ──────────────────────────────────────────────
# Example: major/structured/2026-05-25/NIFTY/expiry/10_11.parquet
PARTITION_DATE_FORMAT = "%Y-%m-%d"
PARTITION_TIME_FORMAT = "%H_%M"

# ──────────────────────────────────────────────
# STORAGE FORMAT
# ──────────────────────────────────────────────
STORAGE_FORMAT = "parquet"
PARQUET_COMPRESSION = "zstd"
PARQUET_ROW_GROUP_SIZE = 65536

# ──────────────────────────────────────────────
# DIRECTORY CREATION UTILITY
# ──────────────────────────────────────────────
ALL_STORAGE_PATHS = [
    MAJOR_RAW, MAJOR_CLEANED, MAJOR_STRUCTURED, MAJOR_REVIEW,
    MINOR_RAW, MINOR_CLEANED, MINOR_STRUCTURED, MINOR_REVIEW,
    COMPUTED_VOLATILITY, COMPUTED_STRUCTURE, COMPUTED_LIQUIDITY,
    COMPUTED_ORDERFLOW, COMPUTED_TREND, COMPUTED_BIAS,
]


def ensure_storage_dirs():
    """Create all storage directories if they do not exist."""
    for path in ALL_STORAGE_PATHS:
        path.mkdir(parents=True, exist_ok=True)