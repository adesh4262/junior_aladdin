"""
Junior Aladdin — Computed Data Pipeline
=======================================
Strongest Version: Processes ticks into intelligence AND returns them 
for real-time enrichment of the backend data stream.
"""

from __future__ import annotations
import time
from typing import Any, Dict
from loguru import logger

from data_center.transformers.computed_transformer import computed_transformer
from data_center.writers.computed_writer import computed_writer

class ComputedPipeline:
    """Consumes cleaned data and generates real-time intelligence."""

    def process_tick(self, cleaned_tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive intelligence from a single cleaned tick.
        Returns the enrichment data to be merged into the main tick.
        """
        enrichment = {}
        try:
            symbol = cleaned_tick.get("symbol", "NIFTY")
            price = cleaned_tick.get("ltp", 0.0)
            timestamp = cleaned_tick.get("timestamp", int(time.time() * 1000))

            # 1) Trend Intelligence
            trend = computed_transformer.compute_trend(symbol, price)
            trend.update({"timestamp": timestamp, "symbol": symbol})
            computed_writer.write_metric("trend", trend)
            
            enrichment["trend_direction"] = trend["direction"]
            enrichment["trend_strength"] = trend["strength"]

            # 2) Volatility Intelligence
            vol = computed_transformer.compute_volatility(symbol)
            vol.update({"timestamp": timestamp, "symbol": symbol})
            computed_writer.write_metric("volatility", vol)
            
            enrichment["volatility_regime"] = vol["regime"]
            enrichment["volatility_value"] = vol["value"]

            # 3) Liquidity Intelligence
            liq = computed_transformer.compute_liquidity(cleaned_tick)
            liq.update({"timestamp": timestamp, "symbol": symbol})
            computed_writer.write_metric("liquidity", liq)
            
            enrichment["liquidity_pressure"] = liq["pressure"]

        except Exception as e:
            logger.error(f"Computed pipeline processing failed: {e}")
            
        return enrichment

computed_pipeline = ComputedPipeline()
