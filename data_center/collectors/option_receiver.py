"""
Junior Aladdin — Option Receiver
=================================
Receives raw options ticks from AngelOne WebSocket, normalizes packet structure,
classifies moneyness, and pushes into the tick queue for downstream processing.

Responsibilities:
  - Receive raw options tick dict from AngelWSClient callback
  - Normalize AngelOne packet format -> standardized RAW_OPTIONS_SCHEMA
  - Handle ATM / ITM / OTM classification when spot/strike data is available
  - Push normalized option tick to tick_queue

Data Center Architecture compliant.
"""

import asyncio
from typing import Optional

from loguru import logger

from data_center.queues.tick_queue import tick_queue
from data_center.schemas.options_schema import (
    RAW_OPTIONS_SCHEMA,
    VALID_OPTION_TYPES,
)
from data_center.utils.timestamps import epoch_ms_now, is_valid_timestamp_ms


class OptionReceiver:
    """
    Receives raw AngelOne options ticks, normalizes them, and enqueues
    for raw storage and downstream processing.
    """

    def __init__(self):
        self._total_received: int = 0
        self._total_enqueued: int = 0
        self._total_dropped: int = 0

    async def on_option(self, raw_packet: dict) -> None:
        """Callback invoked by AngelWSClient for each incoming option tick."""
        self._total_received += 1

        normalized = self._normalize(raw_packet)
        if normalized is None:
            self._total_dropped += 1
            return

        try:
            await asyncio.wait_for(tick_queue.put(normalized), timeout=0.1)
            self._total_enqueued += 1
        except asyncio.TimeoutError:
            self._total_dropped += 1
            logger.warning("Tick queue full — dropping option tick")
        except Exception as exc:
            self._total_dropped += 1
            logger.error(f"Failed to enqueue option tick: {exc}")

    def _normalize(self, packet: dict) -> Optional[dict]:
        """Normalize AngelOne option packet to standard RAW_OPTIONS_SCHEMA format."""
        if not isinstance(packet, dict):
            return None

        token = packet.get("tok") or packet.get("token")
        ltp = packet.get("ltp") or packet.get("price")
        if token is None or ltp is None or ltp == 0:
            return None

        option_type = str(packet.get("option_type", packet.get("type", ""))).upper().strip()
        if option_type and option_type not in VALID_OPTION_TYPES:
            logger.warning(f"Invalid option type: {option_type}")
            return None

        try:
            timestamp = int(packet.get("ts", packet.get("timestamp", epoch_ms_now())))
            if not is_valid_timestamp_ms(timestamp):
                logger.warning(f"Invalid option timestamp: {timestamp}")
                return None

            strike = float(packet.get("strike", packet.get("strike_price", 0.0)))
            expiry = str(packet.get("expiry", packet.get("expiry_date", ""))).strip()

            normalized = {
                "token": str(token).strip(),
                "ltp": float(ltp),
                "volume": int(packet.get("vol", packet.get("volume", 0))),
                "open": float(packet.get("open", 0.0)),
                "high": float(packet.get("high", 0.0)),
                "low": float(packet.get("low", 0.0)),
                "close": float(packet.get("close", 0.0)),
                "timestamp": timestamp,
                "oi": int(packet.get("oi", 0)),
                "oi_change": int(packet.get("oi_change", packet.get("oichg", 0))),
                "iv": float(packet.get("iv", packet.get("implied_volatility", 0.0))),
                "strike": strike,
                "option_type": option_type,
                "expiry": expiry,
            }
        except (ValueError, TypeError) as exc:
            logger.warning(f"Option normalization error: {exc}")
            return None

        missing_fields = [field for field in RAW_OPTIONS_SCHEMA if field not in normalized]
        if missing_fields:
            logger.warning(f"Normalized option tick missing fields: {missing_fields}")
            return None

        if not normalized["option_type"]:
            logger.warning("Missing option_type on option tick")
            return None

        normalized["moneyness"] = self._classify_moneyness(
            spot_ltp=packet.get("spot_ltp"),
            strike=normalized["strike"],
            option_type=normalized["option_type"],
        )

        return normalized

    def _classify_moneyness(
        self,
        spot_ltp: Optional[float],
        strike: float,
        option_type: str,
    ) -> str:
        """Classify option as ITM / ATM / OTM when spot data is available."""
        if spot_ltp is None:
            return "ATM"

        try:
            spot = float(spot_ltp)
        except (TypeError, ValueError):
            return "ATM"

        if strike <= 0:
            return "ATM"

        diff = abs(spot - strike)
        if diff <= max(1.0, spot * 0.0025):
            return "ATM"

        if option_type == "CE":
            return "ITM" if spot > strike else "OTM"
        if option_type == "PE":
            return "ITM" if spot < strike else "OTM"
        return "ATM"

    @property
    def stats(self) -> dict:
        """Return receiver statistics for monitoring."""
        return {
            "total_received": self._total_received,
            "total_enqueued": self._total_enqueued,
            "total_dropped": self._total_dropped,
            "enqueue_rate": round(self._total_enqueued / max(self._total_received, 1) * 100, 2),
        }


option_receiver = OptionReceiver()