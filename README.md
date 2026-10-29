# DocuIntel OCR, Vector & Multi-LLM Intelligence Service

Enterprise-grade microservice delivering deep-learning OCR, PyMuPDF digital document parsing, pandas/openpyxl tabular data parsing, Redis background queues, multi-model vector embeddings, deterministic cross-document reconciliation, and multi-LLM invoice extraction.

---

## Clean Project Folder Structure

```text
ocr-model/
├── src/                               # Core Application Package
│   ├── __init__.py
│   ├── config.py                      # Environment configuration & model settings
│   ├── main.py                        # FastAPI application factory & router mounting
│   ├── api/                           # HTTP API Layer
│   │   ├── __init__.py
│   │   ├── dependencies.py            # API key authentication & security
│   │   └── routes.py                  # All REST endpoints (OCR, PDF, Excel, LLM, Redis, Reconcile)
│   ├── queue/                         # Asynchronous Background Processing
│   │   ├── __init__.py
│   │   ├── manager.py                 # Redis Job Queue Manager
│   │   └── worker.py                  # Redis worker loop with retries & progress
│   ├── schemas/                       # Pydantic v2 Domain Models
│   │   ├── __init__.py
│   │   ├── invoice.py                 # Invoice & LineItem schemas
│   │   ├── ocr.py                     # OCR Box & Response schemas
│   │   ├── spreadsheet.py             # Tabular spreadsheet row & response schemas
│   │   ├── jobs.py                    # Redis background queue schemas
│   │   ├── reconciliation.py          # Discrepancy & Reconciliation summary schemas
│   │   └── embeddings.py              # Dense vector request & response schemas
│   └── services/                      # Business Logic & Model Inference Engines
│       ├── __init__.py
│       ├── ocr_engine.py              # RapidOCR (PaddleOCR ONNX) engine with reading order
│       ├── pdf_parser.py              # PyMuPDF digital parser with 150 DPI raster fallback
│       ├── spreadsheet_parser.py      # pandas + openpyxl Excel & CSV parser
│       ├── llm_extractor.py           # Multi-LLM extraction (Claude, GPT-4o, DeepSeek, Grok)
│       ├── vector_service.py          # SentenceTransformers & OpenAI vector generation
│       ├── validation_service.py      # Pydantic deterministic arithmetic & consistency rules
│       └── reconciliation_service.py  # Multi-way reconciliation across ledger, bank, and receipts
├── tests/                             # Automated Test Suite
│   ├── __init__.py
│   └── test_api.py                    # Pytest test suite (8/8 test cases)
├── scripts/                           # Utility & Verification Scripts
│   └── test_client.py                 # CLI integration test script
├── app.py                             # Root FastAPI entrypoint (`uvicorn app:app`)
├── worker.py                          # Root Background Worker entrypoint (`python worker.py`)
├── Dockerfile                         # Production multi-stage Docker container
├── requirements.txt                   # Microservice dependencies
├── README.md                          # Architecture & API documentation
├── .env.example                       # Environment template
├── .dockerignore                      # Docker ignore rules
└── .gitignore                         # Git ignore rules
```

---

## Architectural Stack Mapping

| Layer / Requirement | Technology Choice & Implementation |
| :--- | :--- |
| **API Framework** | FastAPI 0.115+, Pydantic v2 schemas |
| **Digital PDF Parsing** | PyMuPDF (`pymupdf` / `fitz`) with 150 DPI raster fallback |
| **OCR Engine** | `RapidOCR-ONNX` (PP-OCRv4 CPU-optimized with angle classification) |
| **Excel / CSV** | `pandas`, `openpyxl` with column normalization & row arithmetic |
| **AI Extraction** | Multi-LLM API with schema-constrained JSON (Claude, GPT-4o, DeepSeek, Grok) |
| **Background Jobs** | Redis queue (`jobs:queue`), retry handling, progress tracking, fallback in-memory manager |
| **Deterministic Reconciliation** | Fuzzy matching & mathematical variance detection across general ledger, bank statements, and receipts |
| **Dense Vector Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim), `all-MiniLM-L6-v2`, `text-embedding-3-small` |
| **Testing** | `pytest` test suite with 8/8 comprehensive test cases |
| **Deployment** | Multi-stage Docker container (`python:3.11-slim`) with model pre-caching |

---

## Quick Start (Local Development)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment (optional API keys)
cp .env.example .env

# 3. Start the FastAPI microservice
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# 4. In a separate terminal, launch the Redis background worker
python worker.py
```

---

## Running the Automated Test Suite

```bash
# Run the 8 comprehensive pytest tests
pytest tests/test_api.py -v
```

---

## Containerized Deployment (Docker)

```bash
# Build the Docker image
docker build -t docuintel-ocr .

# Run container with environment configuration
docker run -d -p 7860:7860 --name docuintel-ocr --env-file .env docuintel-ocr
```

---

## API Catalog & Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Microservice metadata, engine discovery, and loaded LLM providers |
| `GET` | `/health` | Kubernetes / container healthcheck probe |
| `POST` | `/ocr` | Image upload $\rightarrow$ Text lines, reading order, bounding boxes, confidence |
| `POST` | `/parse-pdf` | PyMuPDF digital text stream extraction with OCR box fallback |
| `POST` | `/parse-spreadsheet` | Excel (.xlsx, .xls) and CSV parsing via pandas & openpyxl |
| `POST` | `/jobs/enqueue` | Asynchronous Redis background job submission (`jobs:queue`) |
| `GET` | `/jobs/{job_id}` | Redis background task progress, retry count, and status tracking |
| `POST` | `/reconcile` | Cross-document reconciliation matching ledger vs bank vs receipts |
| `POST` | `/embed` | Dense vector embedding generation (bge-small, minilm, openai) |
| `POST` | `/validate` | Pydantic arithmetic validation (subtotal, tax, total, line items) |
| `POST` | `/extract` | Multi-LLM structured invoice extraction (Claude, GPT-4o, DeepSeek, Grok) |
| `POST` | `/process` | Combined OCR and dense vectorization pipeline |
