"""
Junior Aladdin — Options Schema
=================================
Defines the strict schema for raw and structured options tick data.
"""

from typing import Dict

# ──────────────────────────────────────────────
# RAW OPTIONS TICK SCHEMA
# ──────────────────────────────────────────────
RAW_OPTIONS_SCHEMA: Dict[str, str] = {
    "token": "str",
    "ltp": "float64",
    "volume": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "timestamp": "int64",
    "oi": "int64",               # Open interest
    "oi_change": "int64",        # Change in open interest
    "iv": "float64",             # Implied volatility
    "strike": "float64",         # Strike price
    "option_type": "str",        # "CE" or "PE"
    "expiry": "str",             # Expiry date (YYYY-MM-DD)
}

# ──────────────────────────────────────────────
# STRUCTURED OPTIONS SCHEMA
# ──────────────────────────────────────────────
STRUCTURED_OPTIONS_SCHEMA: Dict[str, str] = {
    "token": "str",
    "ltp": "float64",
    "volume": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "timestamp": "int64",
    "oi": "int64",
    "oi_change": "int64",
    "iv": "float64",
    "strike": "float64",
    "option_type": "str",
    "expiry": "str",
    "sequence": "int64",         # Global sequence number
    "exchange": "str",           # Exchange identifier
    "symbol": "str",             # Underlying symbol
    "spot_ltp": "float64",       # Spot LTP at tick time
    "moneyness": "str",          # "ITM", "ATM", "OTM"
}

# ──────────────────────────────────────────────
# REQUIRED FIELDS (non-nullable)
# ──────────────────────────────────────────────
REQUIRED_OPTIONS_FIELDS: list[str] = [
    "token", "ltp", "volume", "timestamp", "oi", "iv",
    "strike", "option_type", "expiry"
]

# ──────────────────────────────────────────────
# OPTIONS CONSTRAINTS
# ──────────────────────────────────────────────
VALID_OPTION_TYPES: set[str] = {"CE", "PE"}

# IV must be positive
MIN_IV: float = 0.0
MAX_IV: float = 200.0  # 200% IV ceiling

# OI must be non-negative
MIN_OI: int = 0