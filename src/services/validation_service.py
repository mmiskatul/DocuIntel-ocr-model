"""Pydantic mathematical and schema consistency validation service."""
from __future__ import annotations

from typing import List
from src.schemas.invoice import InvoiceDataSchema, ValidationIssue, ValidationResult


class InvoiceValidationService:
    @staticmethod
    def validate(data: InvoiceDataSchema) -> ValidationResult:
        """Validates required invoice fields and mathematical sums."""
        issues: List[ValidationIssue] = []

        if not data.invoice_number:
            issues.append(ValidationIssue(code="missing_field", severity="error", field="invoice_number", message="Missing invoice number"))
        if not data.invoice_date:
            issues.append(ValidationIssue(code="missing_field", severity="warning", field="invoice_date", message="Missing invoice date"))
        if not data.supplier_name:
            issues.append(ValidationIssue(code="missing_field", severity="warning", field="supplier_name", message="Missing supplier name"))

        tol = 0.02
        # Check Subtotal + Tax == Total
        if data.subtotal is not None and data.tax is not None and data.total is not None:
            expected = round(data.subtotal + data.tax, 2)
            if abs(expected - data.total) > tol:
                issues.append(ValidationIssue(
                    code="total_mismatch",
                    severity="error",
                    field="total",
                    message=f"Subtotal ({data.subtotal}) + Tax ({data.tax}) = {expected}, but total is {data.total}",
                    details={"expected": expected, "actual": data.total},
                ))

        # Check line items sum == subtotal
        if data.line_items:
            items_sum = round(sum(li.amount for li in data.line_items if li.amount is not None), 2)
            if data.subtotal is not None and abs(items_sum - data.subtotal) > tol:
                issues.append(ValidationIssue(
                    code="subtotal_mismatch",
                    severity="error",
                    field="subtotal",
                    message=f"Line items sum to {items_sum}, but subtotal is {data.subtotal}",
                    details={"line_items_sum": items_sum, "subtotal": data.subtotal},
                ))

        has_errors = any(i.severity == "error" for i in issues)
        conf = 0.95 if not issues else (0.6 if has_errors else 0.8)

        return ValidationResult(
            is_valid=not has_errors,
            needs_manual_review=has_errors or len(issues) > 0,
            issues=issues,
            confidence_score=conf,
        )


validation_service = InvoiceValidationService()

