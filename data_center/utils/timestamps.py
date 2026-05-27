"""
Junior Aladdin — Timestamp Utilities
======================================
Strongest Version: Optimized for Hourly Data Partitioning with descriptive 
range naming (e.g., 09_10) as per the architectural blueprint.
"""

from datetime import datetime, timezone


def epoch_ms_now() -> int:
    """Return current time in epoch milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def epoch_ms_to_datetime(ms: int) -> datetime:
    """Convert epoch milliseconds to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_epoch_ms(dt: datetime) -> int:
    """Convert datetime to epoch milliseconds."""
    return int(dt.timestamp() * 1000)


def format_date_partition(ms: int) -> str:
    """Convert epoch ms to date partition key: YYYY-MM-DD."""
    return epoch_ms_to_datetime(ms).strftime("%Y-%m-%d")


def format_time_partition(ms: int) -> str:
    """
    Convert epoch ms to descriptive hourly range string.
    Example: 10:15 AM -> "10_11" (represents 10:00 to 11:00 slot)
    """
    dt = epoch_ms_to_datetime(ms)
    start_hour = dt.hour
    end_hour = start_hour + 1
    return f"{start_hour:02d}_{end_hour:02d}"


def is_valid_timestamp_ms(ms: int, min_ms: int = 946684800000, max_ms: int = 4102444800000) -> bool:
    """Check if timestamp is within valid range."""
    return min_ms <= ms <= max_ms


def timestamp_continuity_check(
    timestamps: list[int],
    max_gap_ms: int = 5000
) -> list[int]:
    """Find gaps in timestamp sequence."""
    gaps = []
    for i in range(1, len(timestamps)):
        diff = timestamps[i] - timestamps[i - 1]
        if diff > max_gap_ms:
            gaps.append(i)
    return gaps
