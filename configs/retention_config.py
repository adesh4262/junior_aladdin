"""
Junior Aladdin — Retention Configuration
==========================================
Defines data retention policies for raw, structured, and computed storage.
"""

# ──────────────────────────────────────────────
# RAW DATA RETENTION
# ──────────────────────────────────────────────
# Raw tick data is kept for debugging and recovery.
# Auto-deleted after retention period.
RAW_RETENTION_DAYS = 3

# ──────────────────────────────────────────────
# STRUCTURED DATA RETENTION
# ──────────────────────────────────────────────
# Structured data is permanent — used for replay, research, AI, backtesting.
STRUCTURED_RETENTION_DAYS = None  # None = permanent

# ──────────────────────────────────────────────
# COMPUTED DATA RETENTION (adaptive per type)
# ──────────────────────────────────────────────
COMPUTED_RETENTION = {
    "trend": 30,          # 30 days
    "volatility": 90,     # 90 days
    "liquidity": 15,      # 15 days
    "structure": 30,      # 30 days
    "orderflow": 15,      # 15 days
    "bias": 30,           # 30 days
}

# ──────────────────────────────────────────────
# CLEANUP SCHEDULE
# ──────────────────────────────────────────────
# How often retention cleanup runs (in seconds)
CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour

# ──────────────────────────────────────────────
# REVIEW DATA RETENTION
# ──────────────────────────────────────────────
# Review verification reports — kept for audit trail
REVIEW_RETENTION_DAYS = 30