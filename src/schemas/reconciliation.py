"""Cross-document reconciliation schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class DiscrepancyItem(BaseModel):
    id: str
    transaction: str
    ledger_amount: Optional[float] = None
    bank_amount: Optional[float] = None
    ledger_date: Optional[str] = None
    bank_date: Optional[str] = None
    flag_type: Literal["UNMATCHED", "DUPLICATE", "TIMING DIFF", "MISSING RECEIPT"]
    difference: float
    notes: str


class ReconciliationSummary(BaseModel):
    total_sources: int
    total_transactions: int
    matched_count: int
    flagged_count: int
    unexplained_difference: float
    period: str


class ReconciliationResponse(BaseModel):
    summary: ReconciliationSummary
    sources: List[Dict[str, Any]]
    discrepancies: List[DiscrepancyItem]

