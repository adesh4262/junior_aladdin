"""
Junior Aladdin — Tick Schema
=============================
Defines the strict schema for raw and structured tick data.
"""

from typing import Dict, Any

# ──────────────────────────────────────────────
# RAW TICK SCHEMA
# ──────────────────────────────────────────────
# This is the schema for raw incoming ticks — stored as-is.
RAW_TICK_SCHEMA: Dict[str, str] = {
    "token": "str",         # Instrument token
    "ltp": "float64",       # Last traded price
    "volume": "int64",      # Traded volume
    "open": "float64",      # Open price (day)
    "high": "float64",      # High price (day)
    "low": "float64",       # Low price (day)
    "close": "float64",     # Close price (previous day)
    "timestamp": "int64",   # Exchange timestamp (epoch ms)
    "direction": "int8",    # Tick direction: 1=up, -1=down, 0=unchanged
}

# ──────────────────────────────────────────────
# STRUCTURED TICK SCHEMA
# ──────────────────────────────────────────────
# Cleaned and validated tick data.
STRUCTURED_TICK_SCHEMA: Dict[str, str] = {
    "token": "str",
    "ltp": "float64",
    "volume": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "timestamp": "int64",
    "direction": "int8",
    "sequence": "int64",         # Global sequence number
    "exchange": "str",           # Exchange identifier (e.g., "NSE")
    "symbol": "str",             # Symbol name (e.g., "NIFTY")
}

# ──────────────────────────────────────────────
# REQUIRED FIELDS (non-nullable)
# ──────────────────────────────────────────────
REQUIRED_TICK_FIELDS: list[str] = [
    "token", "ltp", "volume", "timestamp"
]

# ──────────────────────────────────────────────
# VALIDATION RANGES
# ──────────────────────────────────────────────
# Price must be positive
MIN_PRICE: float = 0.0
MAX_PRICE: float = 1_000_000.0

# Volume must be non-negative
MIN_VOLUME: int = 0

# Timestamp must be within reasonable range (epoch ms)
MIN_TIMESTAMP_MS: int = 946684800000    # 2000-01-01
MAX_TIMESTAMP_MS: int = 4102444800000   # 2100-01-01

# Direction must be -1, 0, or 1
VALID_DIRECTIONS: set[int] = {-1, 0, 1}