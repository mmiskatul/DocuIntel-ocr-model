import io
import pytest
import pymupdf
import pandas as pd
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import app
from src.schemas.invoice import InvoiceDataSchema, LineItemSchema
from src.services.pdf_parser import pdf_parser_service
from src.services.spreadsheet_parser import spreadsheet_parser_service
from src.services.validation_service import validation_service


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    """Test FastAPI health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client):
    """Test API catalog discovery."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "FastAPI" in data["stack"]
    assert "online" in data["status"]


def test_pydantic_validation_clean_invoice():
    """Test Pydantic validation on a mathematically consistent invoice."""
    data = InvoiceDataSchema(
        invoice_number="INV-2026-TEST",
        invoice_date="2026-05-28",
        supplier_name="Test Supplier LLC",
        subtotal=100.0,
        tax=10.0,
        total=110.0,
        currency="USD",
        line_items=[
            LineItemSchema(description="Item 1", quantity=1, unit_price=50.0, amount=50.0),
            LineItemSchema(description="Item 2", quantity=1, unit_price=50.0, amount=50.0),
        ],
    )
    res = validation_service.validate(data)
    assert res.is_valid is True
    assert res.needs_manual_review is False
    assert len(res.issues) == 0


def test_pydantic_validation_math_mismatch():
    """Test Pydantic validation catches arithmetic discrepancy (Subtotal + Tax != Total)."""
    data = InvoiceDataSchema(
        invoice_number="INV-2026-FAIL",
        invoice_date="2026-05-28",
        supplier_name="Discrepancy Corp",
        subtotal=100.0,
        tax=10.0,
        total=125.0,  # Expected 110, difference 15
        currency="USD",
    )
    res = validation_service.validate(data)
    assert res.is_valid is False
    assert res.needs_manual_review is True
    assert any(i.code == "total_mismatch" for i in res.issues)


def test_pymupdf_pdf_parser():
    """Test digital PDF creation, text stream extraction, and page rasterization via PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "INVOICE #INV-PYMUPDF-2026\nTotal: $1,250.00\nSupplier: PyMuPDF Test")
    pdf_bytes = doc.tobytes()

    text, conf, boxes, page_count = pdf_parser_service.parse_pdf(pdf_bytes)
    assert page_count == 1
    assert "INV-PYMUPDF-2026" in text
    assert "$1,250.00" in text
    assert conf >= 80.0


def test_pandas_openpyxl_spreadsheet_parser():
    """Test Excel parsing via pandas and openpyxl."""
    df = pd.DataFrame([
        {"transaction": "DHL Express", "date": "2026-05-14", "amount": 118.60, "supplier": "DHL"},
        {"transaction": "FedEx Freight", "date": "2026-05-04", "amount": 76.80, "supplier": "FedEx"},
    ])
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    result = spreadsheet_parser_service.parse(buf.getvalue(), "ledger.xlsx")
    assert result["row_count"] == 2
    assert result["total_amount"] == 195.40
    assert result["rows"][0].transaction == "DHL Express"
    assert result["rows"][0].amount == 118.60


def test_redis_job_queue(client):
    """Test background job enqueue and retrieval from Redis / queue manager."""
    # Enqueue job
    enq_resp = client.post("/jobs/enqueue", data={"filename": "test_invoice.pdf"})
    assert enq_resp.status_code == 200
    job_id = enq_resp.json()["job_id"]
    assert job_id is not None

    # Query status
    status_resp = client.get(f"/jobs/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"
    assert data["stage"] == "Queued"
    assert data["progress"] == 0.1


def test_reconcile_endpoint(client):
    """Test deterministic multi-document reconciliation engine endpoint."""
    resp = client.post("/reconcile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["matched_count"] == 27
    assert data["summary"]["flagged_count"] == 4
    assert len(data["discrepancies"]) == 4
    assert any(d["flag_type"] == "UNMATCHED" for d in data["discrepancies"])
    assert any(d["flag_type"] == "DUPLICATE" for d in data["discrepancies"])
    assert any(d["flag_type"] == "TIMING DIFF" for d in data["discrepancies"])
    assert any(d["flag_type"] == "MISSING RECEIPT" for d in data["discrepancies"])


def test_embed_endpoint(client):
    """Test 384-dimensional vector embedding generation endpoint."""
    resp = client.post("/embed", json={"text": "Invoice from Tech Supplies Inc for $450", "model": "openai"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dimension"] == 384
    assert len(data["embeddings"]) == 1
    assert len(data["embeddings"][0]) == 384

