"""
Junior Aladdin — WebSocket Configuration
==========================================
Defines WebSocket connection, reconnection, heartbeat, and subscription settings.
"""

# ──────────────────────────────────────────────
# CONNECTION SETTINGS
# ──────────────────────────────────────────────
WS_URL: str = "wss://ws.angelone.in/feed"
WS_CONNECT_TIMEOUT: int = 10            # Seconds
WS_RECONNECT_DELAY: int = 5             # Seconds before first reconnect attempt
WS_MAX_RECONNECT_DELAY: int = 60        # Max delay between reconnects (exponential backoff)
WS_MAX_RECONNECT_ATTEMPTS: int = 10     # Max reconnect attempts before giving up

# ──────────────────────────────────────────────
# HEARTBEAT
# ──────────────────────────────────────────────
WS_HEARTBEAT_INTERVAL: int = 30         # Seconds between heartbeat pings
WS_HEARTBEAT_TIMEOUT: int = 10          # Seconds to wait for heartbeat response

# ──────────────────────────────────────────────
# SUBSCRIPTION
# ──────────────────────────────────────────────
# NIFTY spot tokens
NIFTY_SPOT_TOKEN: str = "99926000"

# Options tokens (ATM, 2 ITM, 2 OTM for CE and PE)
# These should be fetched dynamically via API — placeholders for now
OPTIONS_TOKENS: list[str] = []

# Market depth tokens
DEPTH_TOKENS: list[str] = []

# ──────────────────────────────────────────────
# SUBSCRIPTION MODE
# ──────────────────────────────────────────────
# "full" — all tokens
# "spot_only" — only NIFTY spot
# "options_only" — only options
SUBSCRIPTION_MODE: str = "full"

# ──────────────────────────────────────────────
# PING/PONG
# ──────────────────────────────────────────────
WS_PING_INTERVAL: int = 25              # Seconds
WS_PING_TIMEOUT: int = 10               # Seconds

# ──────────────────────────────────────────────
# BUFFER
# ──────────────────────────────────────────────
WS_RECV_BUFFER_SIZE: int = 65536        # 64KB receive buffer
WS_SEND_BUFFER_SIZE: int = 65536        # 64KB send buffer