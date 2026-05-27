"""
Junior Aladdin — Minor Data Schema
==================================
Defines the strict Polars schema for contextual snapshots (PCR, Max Pain, etc.).
"""

import polars as pl
from typing import Dict

# ──────────────────────────────────────────────
# MINOR DATA (SNAPSHOT) SCHEMA
# ──────────────────────────────────────────────
# Strong Version: Using direct Polars types to prevent writer errors.
MINOR_SNAPSHOT_SCHEMA: Dict[str, pl.DataType] = {
    "timestamp": pl.Int64,        # Collection time
    "symbol": pl.String,         # Underlying symbol
    "pcr": pl.Float64,           # Put-Call Ratio
    "max_pain": pl.Float64,      # Max Pain Strike
    "total_ce_oi": pl.Int64,     # Total CE Open Interest
    "total_pe_oi": pl.Int64,     # Total PE Open Interest
    "atm_iv": pl.Float64,        # ATM Implied Volatility
    "vix": pl.Float64,           # India VIX level
}
