"""Spreadsheet parsing response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SpreadsheetRowSchema(BaseModel):
    row_index: int
    transaction: Optional[str] = None
    date: Optional[str] = None
    invoice_number: Optional[str] = None
    amount: Optional[float] = None
    supplier: Optional[str] = None
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class SpreadsheetResponse(BaseModel):
    filename: str
    row_count: int
    columns: List[str]
    total_amount: float
    rows: List[SpreadsheetRowSchema]
    processing_time_ms: float

