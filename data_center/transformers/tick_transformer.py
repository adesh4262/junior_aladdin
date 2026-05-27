"""
Junior Aladdin — Tick Transformer
=================================
Converts cleaned tick records into structured parquet-ready records.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import polars as pl

from data_center.schemas.tick_schema import STRUCTURED_TICK_SCHEMA


STRUCTURED_TICK_FIELDS = list(STRUCTURED_TICK_SCHEMA.keys())


class TickTransformer:
    """Transform cleaned tick records into the structured schema."""

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

        if token is None or timestamp is None or ltp is None or volume is None:
            return None

        try:
            structured = {
                "token": str(token).strip(),
                "ltp": float(ltp),
                "volume": int(volume),
                "open": float(record.get("open", 0.0)),
                "high": float(record.get("high", 0.0)),
                "low": float(record.get("low", 0.0)),
                "close": float(record.get("close", 0.0)),
                "timestamp": int(timestamp),
                "direction": int(record.get("direction", 0)),
                "sequence": int(sequence if sequence is not None else record.get("sequence", 0)),
                "exchange": str(exchange or record.get("exchange") or self.default_exchange).strip() or self.default_exchange,
                "symbol": str(symbol or record.get("symbol") or record.get("token") or "").strip(),
            }
        except (TypeError, ValueError):
            return None

        return structured if all(field in structured for field in STRUCTURED_TICK_FIELDS) else None

    def transform_batch(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [record for record in (self.transform_record(item) for item in records) if record is not None]

    def to_frame(self, records: Iterable[dict[str, Any]]) -> pl.DataFrame:
        structured_records = self.transform_batch(records)
        if not structured_records:
            return pl.DataFrame()
        return pl.DataFrame(structured_records).select(STRUCTURED_TICK_FIELDS)


tick_transformer = TickTransformer()
