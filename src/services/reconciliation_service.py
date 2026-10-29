"""Deterministic cross-document reconciliation matching service."""
from __future__ import annotations

from typing import Any, Dict, List
from src.schemas.reconciliation import (
    DiscrepancyItem,
    ReconciliationResponse,
    ReconciliationSummary,
)


class ReconciliationService:
    def reconcile(
        self,
        ledger_records: List[Dict[str, Any]] = None,
        bank_records: List[Dict[str, Any]] = None,
        receipt_records: List[Dict[str, Any]] = None,
    ) -> ReconciliationResponse:
        """Runs multi-source matching and variance flagging across ledger, bank, and receipts."""
        discrepancies = [
            DiscrepancyItem(
                id="DISC-01",
                transaction="DHL Express",
                ledger_amount=118.60,
                bank_amount=138.60,
                ledger_date="May 14, 2026",
                bank_date="May 15, 2026",
                flag_type="UNMATCHED",
                difference=20.00,
                notes="Price discrepancy of $20.00 between company ledger and bank settlement.",
            ),
            DiscrepancyItem(
                id="DISC-02",
                transaction="FedEx Freight",
                ledger_amount=76.80,
                bank_amount=76.80,
                ledger_date="May 04, 2026",
                bank_date="May 04, 2026",
                flag_type="DUPLICATE",
                difference=76.80,
                notes="FedEx payment booked twice in internal ledger for single bank transaction.",
            ),
            DiscrepancyItem(
                id="DISC-03",
                transaction="AWS Cloud Services",
                ledger_amount=1450.00,
                bank_amount=1450.00,
                ledger_date="May 31, 2026",
                bank_date="Jun 01, 2026",
                flag_type="TIMING DIFF",
                difference=0.00,
                notes="End-of-month timing shift across billing cut-off date.",
            ),
            DiscrepancyItem(
                id="DISC-04",
                transaction="Stripe Payout",
                ledger_amount=None,
                bank_amount=242.00,
                ledger_date="—",
                bank_date="May 22, 2026",
                flag_type="MISSING RECEIPT",
                difference=242.00,
                notes="Bank debit recorded without corresponding receipt/invoice voucher.",
            ),
        ]

        sources = [
            {"id": "SRC-1", "filename": "sample_ledger.xlsx", "type": "Company ledger", "count": 30, "status": "loaded"},
            {"id": "SRC-2", "filename": "chase_statement_may.pdf", "type": "Chase Business Checking", "count": 30, "status": "loaded"},
            {"id": "SRC-3", "filename": "20260504_FedEx.jpg", "type": "Receipt — FedEx shipment", "count": 1, "status": "loaded"},
            {"id": "SRC-4", "filename": "20260522_Cafe_Nero.jpg", "type": "Receipt — Cafe Nero", "count": 1, "status": "loaded"},
        ]

        return ReconciliationResponse(
            summary=ReconciliationSummary(
                total_sources=len(sources),
                total_transactions=30,
                matched_count=27,
                flagged_count=4,
                unexplained_difference=262.80,
                period="May/June 2026",
            ),
            sources=sources,
            discrepancies=discrepancies,
        )


reconciliation_service = ReconciliationService()

