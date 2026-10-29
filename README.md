# DocuIntel OCR, Vector & Multi-LLM Intelligence Service

Enterprise-grade microservice delivering deep-learning OCR, PyMuPDF digital document parsing, pandas/openpyxl tabular data parsing, Redis background queues, multi-model vector embeddings, deterministic cross-document reconciliation, and multi-LLM invoice extraction.

---

## Clean Project Folder Structure

```text
ocr-model/
├── src/                               # All Implementation Code
│   ├── __init__.py
│   ├── config.py                      # Environment configuration & model settings
│   ├── main.py                        # FastAPI application factory & router setup
│   ├── api/                           # HTTP Web API Layer
│   │   ├── __init__.py
│   │   ├── dependencies.py            # API key (X-API-Key) security & auth guards
│   │   └── routes.py                  # REST endpoints (OCR, PDF, Excel, LLM, Redis, Reconcile)
│   ├── queue/                         # Asynchronous Background Processing
│   │   ├── __init__.py
│   │   ├── manager.py                 # Redis Job Queue Manager
│   │   └── worker.py                  # Standalone Redis worker process & retry loops
│   ├── schemas/                       # Pydantic v2 Strict Domain Models
│   │   ├── __init__.py                # Schema aggregation & re-exports
│   │   ├── invoice.py                 # InvoiceDataSchema, LineItemSchema, ValidationResult
│   │   ├── ocr.py                     # OCRBoxSchema, OCRResponse, ProcessResponse
│   │   ├── spreadsheet.py             # SpreadsheetRowSchema, SpreadsheetResponse
│   │   ├── jobs.py                    # JobEnqueueResponse, JobStatusResponse
│   │   ├── reconciliation.py          # DiscrepancyItem, ReconciliationResponse
│   │   └── embeddings.py              # EmbeddingRequest, EmbeddingResponse
│   └── services/                      # Business Logic & Model Inference Engines
│       ├── __init__.py                # Service registry & re-exports
│       ├── ocr_engine.py              # RapidOCR (PaddleOCR ONNX) engine with reading order
│       ├── pdf_parser.py              # PyMuPDF digital stream extraction & 150 DPI raster
│       ├── spreadsheet_parser.py      # pandas + openpyxl Excel (.xlsx) & CSV engine
│       ├── llm_extractor.py           # Multi-LLM structured extraction (Claude, GPT-4o, DeepSeek, Grok)
│       ├── vector_service.py          # BAAI/bge-small-en-v1.5, MiniLM & OpenAI embeddings
│       ├── validation_service.py      # Pydantic deterministic arithmetic checks
│       └── reconciliation_service.py  # Multi-way reconciliation across ledger, bank, and receipts
├── tests/                             # Automated Test Suite
│   ├── __init__.py
│   └── test_api.py                    # Pytest test suite (8/8 test cases)
├── scripts/                           # Utility & Verification Scripts
│   └── test_client.py                 # Automated CLI test & healthcheck client
├── main.py                            # Clean Microservice Root Entrypoint (`uvicorn main:app`)
├── Dockerfile                         # Production multi-stage Docker container
├── requirements.txt                   # Microservice dependencies
├── README.md                          # Architecture & API documentation
├── .env.example                       # Environment configuration template
├── .dockerignore                      # Docker ignore rules
└── .gitignore                         # Git ignore rules
```

---

## Architectural Stack Mapping

| Layer / Requirement | Technology Choice & Implementation |
| :--- | :--- |
| **API Framework** | FastAPI 0.115+, Pydantic v2 schemas (`main.py`, `src/api/routes.py`) |
| **Digital PDF Parsing** | PyMuPDF (`pymupdf` / `fitz`) with 150 DPI raster fallback (`src/services/pdf_parser.py`) |
| **OCR Engine** | `RapidOCR-ONNX` (PP-OCRv4 CPU-optimized with angle classification) (`src/services/ocr_engine.py`) |
| **Excel / CSV** | `pandas`, `openpyxl` with column normalization & row arithmetic (`src/services/spreadsheet_parser.py`) |
| **AI Extraction** | Multi-LLM API with schema-constrained JSON (Claude, GPT-4o, DeepSeek, Grok) (`src/services/llm_extractor.py`) |
| **Background Jobs** | Redis queue (`jobs:queue`), retry handling, progress tracking (`src/queue/`) |
| **Deterministic Reconciliation** | Fuzzy matching & mathematical variance detection across general ledger, bank statements, and receipts (`src/services/reconciliation_service.py`) |
| **Dense Vector Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim), `all-MiniLM-L6-v2`, `text-embedding-3-small` (`src/services/vector_service.py`) |
| **Testing** | `pytest` test suite with 8/8 comprehensive test cases (`tests/test_api.py`) |
| **Deployment** | Multi-stage Docker container (`python:3.11-slim`) with model pre-caching |

---

## Quick Start (Local Development)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment (optional API keys)
cp .env.example .env

# 3. Start the FastAPI microservice using the root entrypoint
python main.py
# or:
uvicorn main:app --host 0.0.0.0 --port 7860 --reload

# 4. In a separate terminal, launch the Redis background worker
python -m src.queue.worker

# 5. Run the health & inference verification client
python scripts/test_client.py
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
