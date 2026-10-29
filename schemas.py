"""Pydantic v2 schemas for DocuIntel OCR & Multi-LLM intelligence microservice.
Defines models for OCR bounding boxes, invoice extraction, spreadsheet tabular data,
validation issues, background jobs, and cross-document reconciliation.
"""
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


class OCRBoxSchema(BaseModel):
    box: List[List[float]]
    text: str
    confidence: float


class OCRResponse(BaseModel):
    text: str
    confidence: float
    boxes: List[OCRBoxSchema]
    processing_time_ms: float


class EmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    model: Literal["bge-small", "minilm", "openai"] = "bge-small"


class EmbeddingResponse(BaseModel):
    model: str
    dimension: int
    embeddings: List[List[float]]
    processing_time_ms: float


class ExtractResponse(BaseModel):
    provider_used: str
    data: InvoiceDataSchema
    validation: ValidationResult
    confidence: Dict[str, float] = Field(default_factory=dict)
    uncertain_fields: List[str] = Field(default_factory=list)
    ocr_confidence: float = 0.0
    processing_time_ms: float


class ProcessResponse(BaseModel):
    text: str
    confidence: float
    boxes: List[OCRBoxSchema]
    embedding: List[float]
    embedding_model: str
    dimension: int
    processing_time_ms: float


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


class JobEnqueueResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    created_at: float


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: float
    retries: int
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: float
    updated_at: float


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
