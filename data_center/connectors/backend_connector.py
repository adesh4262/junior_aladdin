"""
Junior Aladdin — Backend Connector
==================================
The high-performance bridge between the Data Center and the Backend (src).
It allows the Backend to subscribe to cleaned data streams without 
knowing about the internal workings of the Data Center.

Strongest Version: Implements the Observer pattern for zero-latency 
intra-process communication.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from loguru import logger


class BackendConnector:
    """The bridge that broadcasts cleaned data to registered backend listeners."""

    def __init__(self):
        # List of callbacks from the backend (e.g., from DataEngine or MarketState)
        self._tick_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._option_listeners: List[Callable[[Dict[str, Any]], None]] = []
        
        self._stats = {
            "ticks_broadcasted": 0,
            "options_broadcasted": 0,
            "listener_count": 0
        }

    def register_tick_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Backend components call this to receive real-time cleaned ticks."""
        if callback not in self._tick_listeners:
            self._tick_listeners.append(callback)
            self._stats["listener_count"] = len(self._tick_listeners) + len(self._option_listeners)
            logger.info("New tick listener registered in BackendConnector")

    def broadcast_tick(self, cleaned_tick: Dict[str, Any]) -> None:
        """Called by the Data Center pipeline to send data to the backend."""
        for listener in self._tick_listeners:
            try:
                listener(cleaned_tick)
            except Exception as e:
                logger.error(f"Error in backend tick listener: {e}")
        
        self._stats["ticks_broadcasted"] += 1

    def broadcast_option_chain(self, cleaned_option_data: Dict[str, Any]) -> None:
        """Called by the Data Center to broadcast processed option snapshots."""
        for listener in self._option_listeners:
            try:
                listener(cleaned_option_data)
            except Exception as e:
                logger.error(f"Error in backend option listener: {e}")
                
        self._stats["options_broadcasted"] += 1

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# Global singleton instance for the entire system
backend_connector = BackendConnector()
