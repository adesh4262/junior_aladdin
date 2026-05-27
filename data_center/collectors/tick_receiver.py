"""
Junior Aladdin — Tick Receiver
================================
Receives raw ticks from AngelOne WebSocket, normalizes packet structure,
and pushes into the tick queue for downstream processing.

Responsibilities:
  - Receive raw tick dict from AngelWSClient callback
  - Normalize AngelOne packet format → standardized RAW_TICK_SCHEMA
  - Compute tick direction (up/down/unchanged)
  - Push normalized tick to tick_queue
  - Handle high-frequency flow without blocking

Data Center Architecture compliant.
"""

import asyncio
from typing import Optional

from loguru import logger

from data_center.schemas.tick_schema import RAW_TICK_SCHEMA
from data_center.queues.tick_queue import tick_queue
from data_center.utils.timestamps import epoch_ms_now, is_valid_timestamp_ms


class TickReceiver:
    """
    Receives raw AngelOne ticks, normalizes them, and enqueues
    for raw storage and downstream processing.

    This is the bridge between WebSocket client and queue system.
    """

    def __init__(self):
        # Track last LTP for tick direction computation
        self._last_ltp: Optional[float] = None

        # Stats
        self._total_received: int = 0
        self._total_enqueued: int = 0
        self._total_dropped: int = 0

    # ──────────────────────────────────────────
    # PUBLIC — Callback for AngelWSClient
    # ──────────────────────────────────────────

    async def on_tick(self, raw_packet: dict) -> None:
        """
        Callback invoked by AngelWSClient for each incoming tick.

        Args:
            raw_packet: Raw tick dict from AngelOne WebSocket (msgpack-decoded).
        """
        self._total_received += 1

        # ── Step 1: Normalize packet to RAW_TICK_SCHEMA ──
        normalized = self._normalize(raw_packet)
        if normalized is None:
            self._total_dropped += 1
            return

        # ── Step 2: Push to tick queue (non-blocking) ──
        try:
            await asyncio.wait_for(
                tick_queue.put(normalized),
                timeout=0.1,
            )
            self._total_enqueued += 1
        except asyncio.TimeoutError:
            self._total_dropped += 1
            logger.warning("Tick queue full — dropping tick")
        except Exception as e:
            self._total_dropped += 1
            logger.error(f"Failed to enqueue tick: {e}")

    # ──────────────────────────────────────────
    # NORMALIZATION
    # ──────────────────────────────────────────

    def _normalize(self, packet: dict) -> Optional[dict]:
        """
        Normalize AngelOne tick packet to standard RAW_TICK_SCHEMA format.

        AngelOne SmartAPI sends packets like (varies by mode):
            { "tok": "99926000", "ltp": 18500.0, "vol": 12345,
              "open": 18400.0, "high": 18520.0, "low": 18380.0,
              "close": 18450.0, "ts": 1716600000000 }

        We map to RAW_TICK_SCHEMA:
            { "token": str, "ltp": float, "volume": int,
              "open": float, "high": float, "low": float,
              "close": float, "timestamp": int, "direction": int }

        Returns None if packet is invalid and should be dropped.
        """
        if not isinstance(packet, dict):
            return None

        # ── Extract fields with AngelOne key mapping ──
        token = packet.get("tok")
        if token is None:
            # Try alternate key
            token = packet.get("token")
        if token is None:
            return None

        ltp = packet.get("ltp") or packet.get("price")
        if ltp is None or ltp == 0:
            return None

        # ── Map fields ──
        try:
            timestamp = int(packet.get("ts", packet.get("timestamp", epoch_ms_now())))
            if not is_valid_timestamp_ms(timestamp):
                logger.warning(f"Invalid tick timestamp: {timestamp}")
                return None

            normalized = {
                "token": str(token).strip(),
                "ltp": float(ltp),
                "volume": int(packet.get("vol", packet.get("volume", 0))),
                "open": float(packet.get("open", 0.0)),
                "high": float(packet.get("high", 0.0)),
                "low": float(packet.get("low", 0.0)),
                "close": float(packet.get("close", 0.0)),
                "timestamp": timestamp,
                "direction": self._compute_direction(float(ltp)),
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Normalization error: {e}")
            return None

        missing_fields = [field for field in RAW_TICK_SCHEMA if field not in normalized]
        if missing_fields:
            logger.warning(f"Normalized tick missing fields: {missing_fields}")
            return None

        # Update last LTP for next direction computation
        self._last_ltp = float(ltp)

        return normalized

    def _compute_direction(self, current_ltp: float) -> int:
        """
        Compute tick direction:
            1  → price up
            -1 → price down
            0  → unchanged
        """
        if self._last_ltp is None:
            return 0
        if current_ltp > self._last_ltp:
            return 1
        elif current_ltp < self._last_ltp:
            return -1
        return 0

    # ──────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return receiver statistics for monitoring."""
        return {
            "total_received": self._total_received,
            "total_enqueued": self._total_enqueued,
            "total_dropped": self._total_dropped,
            "enqueue_rate": (
                round(self._total_enqueued / max(self._total_received, 1) * 100, 2)
            ),
        }


# ──────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ──────────────────────────────────────────
# A single receiver instance used across the system.
tick_receiver = TickReceiver()