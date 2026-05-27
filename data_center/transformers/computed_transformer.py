"""
Junior Aladdin — Computed Data Transformer
==========================================
Strongest Version: Precision intelligence derivation with return values
for real-time backend enrichment.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional
from loguru import logger

class ComputedTransformer:
    """Derives high-precision trend, volatility, and liquidity metrics."""

    def __init__(self):
        self._price_buffer: Dict[str, List[float]] = {}
        self._buffer_size = 100 # Increased buffer for better precision

    def compute_trend(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """EMA-based trend detection with strength mapping."""
        if symbol not in self._price_buffer:
            self._price_buffer[symbol] = []
        
        buf = self._price_buffer[symbol]
        buf.append(current_price)
        if len(buf) > self._buffer_size:
            buf.pop(0)
            
        if len(buf) < 21:
            return {"direction": 0, "strength": 0.0, "method": "INITIALIZING"}
            
        # Using accurate EMA calculation
        ema_9 = self._calculate_ema(buf, 9)
        ema_21 = self._calculate_ema(buf, 21)
        
        direction = 1 if ema_9 > ema_21 else -1
        # Strength: percentage difference normalized to 0-100 scale
        diff = abs(ema_9 - ema_21) / ema_21
        strength = min(100.0, diff * 5000.0) 
        
        return {
            "direction": int(direction),
            "strength": float(round(strength, 2)),
            "method": "EMA_9_21_PRECISION"
        }

    def compute_volatility(self, symbol: str) -> Dict[str, Any]:
        """Calculates volatility regime based on standard deviation of recent ticks."""
        buf = self._price_buffer.get(symbol, [])
        if len(buf) < 30:
            return {"regime": "UNKNOWN", "value": 0.0}
            
        std_dev = np.std(buf)
        
        # Standardized Regime Classification
        if std_dev < 1.5: regime = "CALM"
        elif std_dev < 4.0: regime = "NORMAL"
        elif std_dev < 8.0: regime = "VOLATILE"
        else: regime = "PANIC"
        
        return {
            "regime": regime,
            "value": float(round(std_dev, 2))
        }

    def compute_liquidity(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes orderflow pressure from tick direction."""
        direction = tick_data.get("direction", 0)
        # 1 = Buying Pressure, -1 = Selling Pressure
        pressure = 1 if direction > 0 else (-1 if direction < 0 else 0)
        
        return {
            "pressure": int(pressure),
            "intensity": 1.0 # Base intensity
        }

    def _calculate_ema(self, data: List[float], window: int) -> float:
        """Helper for accurate EMA calculation."""
        if len(data) < window: return data[-1]
        alpha = 2 / (window + 1)
        ema = data[0]
        for price in data[1:]:
            ema = price * alpha + ema * (1 - alpha)
        return ema

computed_transformer = ComputedTransformer()
