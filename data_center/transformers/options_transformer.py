"""
Junior Aladdin — Options Transformer
====================================
Converts cleaned options records into structured parquet-ready records.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import polars as pl

from data_center.schemas.options_schema import STRUCTURED_OPTIONS_SCHEMA


STRUCTURED_OPTIONS_FIELDS = list(STRUCTURED_OPTIONS_SCHEMA.keys())


class OptionsTransformer:
    """Transform cleaned option records into the structured schema."""

    def __init__(self, default_exchange: str = "NSE"):
        self.default_exchange = default_exchange

    def transform_record(
        self,
        record: dict[str, Any],
        *,
        sequence: Optional[int] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(record, dict):
            return None

        token = record.get("token")
        timestamp = record.get("timestamp")
        ltp = record.get("ltp")
        volume = record.get("volume")
        strike = record.get("strike")
        option_type = record.get("option_type")
        expiry = record.get("expiry")

        if None in {token, timestamp, ltp, volume, strike, option_type, expiry}:
            return None

        try:
            spot_ltp = record.get("spot_ltp")
            structured = {
                "token": str(token).strip(),
                "ltp": float(ltp),
                "volume": int(volume),
                "open": float(record.get("open", 0.0)),
                "high": float(record.get("high", 0.0)),
                "low": float(record.get("low", 0.0)),
                "close": float(record.get("close", 0.0)),
                "timestamp": int(timestamp),
                "oi": int(record.get("oi", 0)),
                "oi_change": int(record.get("oi_change", 0)),
                "iv": float(record.get("iv", 0.0)),
                "strike": float(strike),
                "option_type": str(option_type).strip().upper(),
                "expiry": str(expiry).strip(),
                "sequence": int(sequence if sequence is not None else record.get("sequence", 0)),
                "exchange": str(exchange or record.get("exchange") or self.default_exchange).strip() or self.default_exchange,
                "symbol": str(symbol or record.get("symbol") or record.get("underlying_symbol") or record.get("token") or "").strip(),
                "spot_ltp": float(spot_ltp) if spot_ltp is not None else None,
                "moneyness": str(record.get("moneyness", "ATM")).strip().upper() or "ATM",
            }
        except (TypeError, ValueError):
            return None

        return structured if all(field in structured for field in STRUCTURED_OPTIONS_FIELDS) else None

    def transform_batch(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [record for record in (self.transform_record(item) for item in records) if record is not None]

    def to_frame(self, records: Iterable[dict[str, Any]]) -> pl.DataFrame:
        structured_records = self.transform_batch(records)
        if not structured_records:
            return pl.DataFrame()
        return pl.DataFrame(structured_records).select(STRUCTURED_OPTIONS_FIELDS)


options_transformer = OptionsTransformer()
