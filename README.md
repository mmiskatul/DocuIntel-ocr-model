# DocuIntel OCR, Vector & Multi-LLM Intelligence Service

Standalone FastAPI + Pydantic v2 microservice providing deep-learning OCR, PyMuPDF digital document parsing, pandas/openpyxl tabular data parsing, Redis background queues, multi-model vector embeddings, deterministic cross-document reconciliation, and multi-LLM invoice extraction. Runs locally or containerized via Docker.

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
