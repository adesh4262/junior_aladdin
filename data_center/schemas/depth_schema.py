"""
Junior Aladdin — Market Depth Schema
======================================
Defines the strict schema for raw and structured market depth data.
"""

from typing import Dict

# ──────────────────────────────────────────────
# RAW DEPTH SCHEMA
# ──────────────────────────────────────────────
RAW_DEPTH_SCHEMA: Dict[str, str] = {
    "token": "str",
    "timestamp": "int64",
    "bid_price_1": "float64", "bid_qty_1": "int64",
    "bid_price_2": "float64", "bid_qty_2": "int64",
    "bid_price_3": "float64", "bid_qty_3": "int64",
    "bid_price_4": "float64", "bid_qty_4": "int64",
    "bid_price_5": "float64", "bid_qty_5": "int64",
    "ask_price_1": "float64", "ask_qty_1": "int64",
    "ask_price_2": "float64", "ask_qty_2": "int64",
    "ask_price_3": "float64", "ask_qty_3": "int64",
    "ask_price_4": "float64", "ask_qty_4": "int64",
    "ask_price_5": "float64", "ask_qty_5": "int64",
}

# ──────────────────────────────────────────────
# STRUCTURED DEPTH SCHEMA
# ──────────────────────────────────────────────
STRUCTURED_DEPTH_SCHEMA: Dict[str, str] = {
    "token": "str",
    "timestamp": "int64",
    "bid_price_1": "float64", "bid_qty_1": "int64",
    "bid_price_2": "float64", "bid_qty_2": "int64",
    "bid_price_3": "float64", "bid_qty_3": "int64",
    "bid_price_4": "float64", "bid_qty_4": "int64",
    "bid_price_5": "float64", "bid_qty_5": "int64",
    "ask_price_1": "float64", "ask_qty_1": "int64",
    "ask_price_2": "float64", "ask_qty_2": "int64",
    "ask_price_3": "float64", "ask_qty_3": "int64",
    "ask_price_4": "float64", "ask_qty_4": "int64",
    "ask_price_5": "float64", "ask_qty_5": "int64",
    "spread": "float64",           # Best ask - best bid
    "mid_price": "float64",        # (bid1 + ask1) / 2
    "total_bid_qty": "int64",      # Sum of all bid quantities
    "total_ask_qty": "int64",      # Sum of all ask quantities
    "imbalance": "float64",        # (bid_qty - ask_qty) / (bid_qty + ask_qty)
    "sequence": "int64",
    "exchange": "str",
    "symbol": "str",
}

# ──────────────────────────────────────────────
# REQUIRED FIELDS (non-nullable)
# ──────────────────────────────────────────────
REQUIRED_DEPTH_FIELDS: list[str] = [
    "token", "timestamp",
    "bid_price_1", "bid_qty_1",
    "ask_price_1", "ask_qty_1",
]

# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────
# Prices must be positive
MIN_DEPTH_PRICE: float = 0.0

# Quantities must be non-negative
MIN_DEPTH_QTY: int = 0

# Spread and mid price must be non-negative
MIN_SPREAD: float = 0.0
MIN_MID_PRICE: float = 0.0