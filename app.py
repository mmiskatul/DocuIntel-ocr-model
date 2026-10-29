"""DocuIntel AI & OCR Microservice Entrypoint.

Delegates to modular architecture in src/:
- src/config.py
- src/schemas/
- src/services/
- src/queue/
- src/api/
"""
from __future__ import annotations

import sys
import os

# Ensure package import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import app, job_queue
from src.schemas.invoice import InvoiceDataSchema, LineItemSchema, ValidationResult
from src.schemas.ocr import OCRBoxSchema
from src.services.ocr_engine import ocr_service
from src.services.pdf_parser import pdf_parser_service
from src.services.spreadsheet_parser import spreadsheet_parser_service
from src.services.validation_service import validation_service


# Backward-compatibility functional wrappers
def _run_ocr(img):
    return ocr_service.run_ocr(img)


def _parse_pdf_with_pymupdf(content: bytes):
    return pdf_parser_service.parse_pdf(content)


def _parse_spreadsheet_with_pandas(content: bytes, filename: str):
    return spreadsheet_parser_service.parse(content, filename)


def validate_invoice_pydantic(data: InvoiceDataSchema) -> ValidationResult:
    return validation_service.validate(data)


if __name__ == "__main__":
    import uvicorn
    from src.config import HOST, PORT
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
