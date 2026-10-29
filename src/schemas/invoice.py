"""Invoice extraction and validation Pydantic schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class LineItemSchema(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class InvoiceDataSchema(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    supplier_name: Optional[str] = None
    customer_name: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    line_items: List[LineItemSchema] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    field: Optional[str] = None
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    is_valid: bool
    needs_manual_review: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    confidence_score: float = 1.0


class ExtractResponse(BaseModel):
    provider_used: str
    data: InvoiceDataSchema
    validation: ValidationResult
    confidence: Dict[str, float] = Field(default_factory=dict)
    uncertain_fields: List[str] = Field(default_factory=list)
    ocr_confidence: float = 0.0
    processing_time_ms: float

