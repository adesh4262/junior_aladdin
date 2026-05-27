"""
Junior Aladdin — Computed Data Schema
=====================================
Defines the schema for generated intelligence (Trend, Volatility, etc.).
"""

from typing import Dict

# ──────────────────────────────────────────────
# COMPUTED DATA SCHEMAS
# ──────────────────────────────────────────────

# Trend Strength & Direction
TREND_SCHEMA: Dict[str, str] = {
    "timestamp": "int64",
    "symbol": "str",
    "direction": "int8",        # 1=Up, -1=Down, 0=Neutral
    "strength": "float64",     # 0 to 100
    "method": "str",           # e.g., "EMA_CROSS"
}

# Volatility Regime
VOLATILITY_SCHEMA: Dict[str, str] = {
    "timestamp": "int64",
    "symbol": "str",
    "regime": "str",           # "CALM", "NORMAL", "VOLATILE", "PANIC"
    "value": "float64",        # Current ATR or StdDev
    "percentile": "float64",   # vs history
}

# Liquidity & Orderflow Pressure
LIQUIDITY_SCHEMA: Dict[str, str] = {
    "timestamp": "int64",
    "symbol": "str",
    "imbalance": "float64",    # Bid vs Ask ratio
    "pressure": "int8",        # 1=Buying, -1=Selling, 0=Balanced
    "intensity": "float64",    # Ticks per second
}

# Market Structure
STRUCTURE_SCHEMA: Dict[str, str] = {
    "timestamp": "int64",
    "symbol": "str",
    "state": "str",            # "TRENDING", "RANGE", "CHOP"
    "high_boundary": "float64",
    "low_boundary": "float64",
}
