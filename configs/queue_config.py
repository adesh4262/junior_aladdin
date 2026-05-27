"""
Junior Aladdin — Queue Configuration
=====================================
Defines queue settings for tick processing and cleaner pipeline.
"""

from typing import Optional

# ──────────────────────────────────────────────
# TICK QUEUE SETTINGS
# ──────────────────────────────────────────────
TICK_QUEUE_MAXSIZE: int = 100_000       # Max items in tick queue
TICK_QUEUE_TIMEOUT: float = 0.1         # Seconds before retry on full queue
TICK_BATCH_SIZE: int = 1000             # Items per batch write

# ──────────────────────────────────────────────
# CLEANER QUEUE SETTINGS
# ──────────────────────────────────────────────
CLEANER_QUEUE_MAXSIZE: int = 50_000     # Max items in cleaner queue
CLEANER_QUEUE_TIMEOUT: float = 0.1      # Seconds before retry
CLEANER_BATCH_SIZE: int = 500           # Items per batch clean
CLEANER_WORKER_COUNT: int = 2           # Number of parallel cleaner workers

# ──────────────────────────────────────────────
# QUEUE HEALTH
# ──────────────────────────────────────────────
QUEUE_HIGH_WATERMARK: int = 80          # Percentage — alert when queue is 80% full
QUEUE_LOW_WATERMARK: int = 20           # Percentage — ok when queue drops to 20%

# ──────────────────────────────────────────────
# RETRY SETTINGS
# ──────────────────────────────────────────────
QUEUE_PUT_RETRIES: int = 3
QUEUE_PUT_RETRY_DELAY: float = 0.5      # Seconds between retries

# ──────────────────────────────────────────────
# QUEUE TYPES
# ──────────────────────────────────────────────
TICK_QUEUE_NAME: str = "tick_queue"
CLEANER_QUEUE_NAME: str = "cleaner_queue"