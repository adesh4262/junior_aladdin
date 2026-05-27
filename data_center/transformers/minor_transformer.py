"""
Junior Aladdin — Minor Data Transformer
=======================================
Transforms raw option chain data into minor metrics like PCR and Max Pain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

class MinorTransformer:
    """Transform raw chain data into structured minor metrics."""

    def compute_pcr(self, chain_data: Dict[int, Dict[str, Any]]) -> float:
        """Calculate Put-Call Ratio from Open Interest."""
        total_ce_oi = 0
        total_pe_oi = 0
        
        for strike_data in chain_data.values():
            total_ce_oi += strike_data.get("ce", {}).get("oi") or 0
            total_pe_oi += strike_data.get("pe", {}).get("oi") or 0
            
        if total_ce_oi == 0:
            return 0.0
        return round(total_pe_oi / total_ce_oi, 4)

    def compute_max_pain(self, chain_data: Dict[int, Dict[str, Any]]) -> float:
        """Calculate the Max Pain strike point."""
        strikes = sorted(chain_data.keys())
        if not strikes:
            return 0.0
            
        min_loss = float('inf')
        max_pain_strike = strikes[0]
        
        for p_strike in strikes:
            total_loss = 0.0
            for strike in strikes:
                ce_oi = chain_data[strike].get("ce", {}).get("oi") or 0
                pe_oi = chain_data[strike].get("pe", {}).get("oi") or 0
                
                # Loss to CE writers if price expires at p_strike
                if p_strike > strike:
                    total_loss += (p_strike - strike) * ce_oi
                
                # Loss to PE writers if price expires at p_strike
                if p_strike < strike:
                    total_loss += (strike - p_strike) * pe_oi
                    
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = p_strike
                
        return float(max_pain_strike)

    def get_atm_iv(self, chain_data: Dict[int, Dict[str, Any]], spot_price: float) -> float:
        """Get ATM Implied Volatility (average of CE and PE)."""
        if not chain_data:
            return 0.0
            
        # Find closest strike
        atm_strike = min(chain_data.keys(), key=lambda x: abs(x - spot_price))
        ce_iv = chain_data[atm_strike].get("ce", {}).get("iv") or 0.0
        pe_iv = chain_data[atm_strike].get("pe", {}).get("iv") or 0.0
        
        if ce_iv > 0 and pe_iv > 0:
            return round((ce_iv + pe_iv) / 2.0, 4)
        return ce_iv or pe_iv or 0.0

    def transform_snapshot(self, chain_data: Dict[int, Dict[str, Any]], spot_price: float, vix: float, timestamp: int, symbol: str) -> Dict[str, Any]:
        """Combine all metrics into a single snapshot record."""
        total_ce_oi = sum(s.get("ce", {}).get("oi", 0) or 0 for s in chain_data.values())
        total_pe_oi = sum(s.get("pe", {}).get("oi", 0) or 0 for s in chain_data.values())
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "pcr": self.compute_pcr(chain_data),
            "max_pain": self.compute_max_pain(chain_data),
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "atm_iv": self.get_atm_iv(chain_data, spot_price),
            "vix": vix,
        }

minor_transformer = MinorTransformer()
