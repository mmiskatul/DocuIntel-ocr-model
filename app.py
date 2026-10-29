"""DocuIntel AI & OCR Microservice (FastAPI + Pydantic + Multi-Model Vectorization + Multi-LLM Extraction)

Supported Providers:
1. OpenAI (GPT-4o / GPT-4o-mini & text-embedding-3-small)
2. Anthropic (Claude 3.5 Sonnet / Claude 3.5 Haiku)
3. DeepSeek (DeepSeek-V3 / DeepSeek-Chat)
4. Grok (xAI Grok-3 / Grok-2)

Vector Models:
1. BAAI/bge-small-en-v1.5 (384-dim, default)
2. sentence-transformers/all-MiniLM-L6-v2 (384-dim)
3. OpenAI text-embedding-3-small (384-dim)
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import redis
from sentence_transformers import SentenceTransformer

from schemas import (
    DiscrepancyItem,
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractResponse,
    InvoiceDataSchema,
    JobEnqueueResponse,
    JobStatusResponse,
    LineItemSchema,
    OCRBoxSchema,
    OCRResponse,
    ProcessResponse,
    ReconciliationResponse,
    ReconciliationSummary,
    SpreadsheetResponse,
    SpreadsheetRowSchema,
    ValidationIssue,
    ValidationResult,
)
from parsers import ocr_engine
from spreadsheet import parse_spreadsheet_data
from llm_extractor import call_anthropic_extractor, call_openai_compatible_extractor
from reconciler import run_deterministic_reconciliation, validate_invoice_rules

# Load API Keys & Config
API_KEY = os.getenv("API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "auto")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = FastAPI(
    title="DocuIntel OCR & Multi-LLM Intelligence Service",
    description="FastAPI + Pydantic service delivering deep-learning OCR, multi-model vector embeddings, and multi-LLM invoice extraction.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Redis Background Job Queue Manager (Redis + Worker / Image 2)
# ------------------------------------------------------------------------------

class RedisJobQueueManager:
    def __init__(self, redis_url: str):
        self.redis_client = None
        self.fallback_jobs: Dict[str, dict] = {}
        try:
            r = redis.Redis.from_url(redis_url, socket_timeout=2)
            r.ping()
            self.redis_client = r
            print("Connected to Redis background job queue successfully!")
        except Exception as exc:
            print(f"Redis not running locally ({exc}). Operating on in-memory background job queue.")

    def enqueue(self, metadata: dict) -> str:
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Queued",
            "progress": 0.1,
            "retries": 0,
            "error": None,
            "result": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            **metadata,
        }
        if self.redis_client:
            try:
                self.redis_client.set(f"job:{job_id}", json.dumps(job_data), ex=86400)
                self.redis_client.lpush("jobs:queue", job_id)
                return job_id
            except Exception:
                pass
        self.fallback_jobs[job_id] = job_data
        return job_id

    def update(
        self,
        job_id: str,
        status: str,
        stage: str,
        progress: float,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        job = self.get(job_id) or {"job_id": job_id, "created_at": time.time(), "retries": 0}
        job["status"] = status
        job["stage"] = stage
        job["progress"] = progress
        job["updated_at"] = time.time()
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
            job["retries"] = job.get("retries", 0) + 1

        if self.redis_client:
            try:
                self.redis_client.set(f"job:{job_id}", json.dumps(job), ex=86400)
                return
            except Exception:
                pass
        self.fallback_jobs[job_id] = job

    def get(self, job_id: str) -> Optional[dict]:
        if self.redis_client:
            try:
                raw = self.redis_client.get(f"job:{job_id}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return self.fallback_jobs.get(job_id)


job_queue = RedisJobQueueManager(REDIS_URL)


# ------------------------------------------------------------------------------
# Model Manager
# ------------------------------------------------------------------------------

class ModelManager:
    def __init__(self):
        self._models: Dict[str, SentenceTransformer] = {}

    def get_embedding_model(self, name: str) -> SentenceTransformer:
        if name not in self._models:
            model_map = {
                "bge-small": "BAAI/bge-small-en-v1.5",
                "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            }
            model_repo = model_map.get(name, "BAAI/bge-small-en-v1.5")
            print(f"Loading SentenceTransformer: {model_repo}...")
            self._models[name] = SentenceTransformer(model_repo, device="cpu")
        return self._models[name]


models = ModelManager()


def verify_key(x_api_key: Optional[str] = Header(None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# Helper wrappers for backward-compatibility with tests and worker
def _run_ocr(img: Image.Image) -> tuple[str, float, list[OCRBoxSchema]]:
    return ocr_engine.run_ocr(img)


def _parse_pdf_with_pymupdf(content: bytes) -> tuple[str, float, list[OCRBoxSchema], int]:
    return ocr_engine.parse_pdf(content)


def _parse_spreadsheet_with_pandas(content: bytes, filename: str) -> dict:
    return parse_spreadsheet_data(content, filename)


def validate_invoice_pydantic(data: InvoiceDataSchema) -> ValidationResult:
    return validate_invoice_rules(data)


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/")
def index():
    return {
        "service": "DocuIntel Multi-LLM OCR Microservice",
        "stack": "FastAPI + Pydantic v2 + ONNX",
        "version": "2.1.0",
        "status": "online",
        "ocr_engine": "RapidOCR ONNX PP-OCRv4",
        "pdf_parser": "PyMuPDF (fitz)",
        "spreadsheet_parser": "pandas + openpyxl",
        "background_queue": "Redis + Worker (redis://localhost:6379/0)",
        "vector_embedding_default": "BAAI/bge-small-en-v1.5 (384-dim)",
        "available_llm_providers": {
            "anthropic": bool(ANTHROPIC_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "deepseek": bool(DEEPSEEK_API_KEY),
            "grok": bool(XAI_API_KEY),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    """Deep-learning OCR inference endpoint for invoices and receipts."""
    verify_key(x_api_key)
    t0 = time.perf_counter()
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    text, conf, boxes = _run_ocr(img)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return OCRResponse(
        text=text,
        confidence=conf,
        boxes=boxes,
        processing_time_ms=duration_ms,
    )


@app.post("/parse-pdf", response_model=OCRResponse)
async def parse_pdf_endpoint(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    """Digital PDF parsing endpoint via PyMuPDF + rasterized OCR fallback."""
    verify_key(x_api_key)
    t0 = time.perf_counter()
    content = await file.read()
    try:
        text, conf, boxes, pages = _parse_pdf_with_pymupdf(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF document: {exc}")

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return OCRResponse(
        text=text,
        confidence=conf,
        boxes=boxes,
        processing_time_ms=duration_ms,
    )


@app.post("/parse-spreadsheet", response_model=SpreadsheetResponse)
async def parse_spreadsheet_endpoint(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    """Tabular spreadsheet parsing endpoint for Excel (.xlsx/.xls) and CSV via pandas & openpyxl."""
    verify_key(x_api_key)
    t0 = time.perf_counter()
    content = await file.read()
    try:
        parsed = _parse_spreadsheet_with_pandas(content, file.filename or "spreadsheet.xlsx")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse spreadsheet: {exc}")

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return SpreadsheetResponse(
        filename=parsed["filename"],
        row_count=parsed["row_count"],
        columns=parsed["columns"],
        total_amount=parsed["total_amount"],
        rows=parsed["rows"],
        processing_time_ms=duration_ms,
    )


@app.post("/jobs/enqueue", response_model=JobEnqueueResponse)
async def enqueue_job(
    filename: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    """Enqueues an asynchronous document processing job into the Redis background queue."""
    verify_key(x_api_key)
    fname = filename or (file.filename if file else "document.pdf")
    ctype = file.content_type if file else "application/pdf"
    job_id = job_queue.enqueue({
        "filename": fname,
        "content_type": ctype,
    })
    return JobEnqueueResponse(
        job_id=job_id,
        status="queued",
        stage="Queued",
        created_at=time.time(),
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, x_api_key: Optional[str] = Header(None)):
    """Polls the status, progress, retries, and results of a Redis background task."""
    verify_key(x_api_key)
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        stage=job.get("stage", "Processing"),
        progress=job.get("progress", 0.0),
        retries=job.get("retries", 0),
        error=job.get("error"),
        result=job.get("result"),
        created_at=job.get("created_at", time.time()),
        updated_at=job.get("updated_at", time.time()),
    )


@app.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile_documents(x_api_key: Optional[str] = Header(None)):
    """Cross-document reconciliation engine: Deterministic calculations matching ledger vs bank vs receipts."""
    verify_key(x_api_key)
    return run_deterministic_reconciliation()


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_endpoint(
    req: EmbeddingRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Dense vector embedding generation using BAAI/bge-small-en-v1.5 or OpenAI."""
    verify_key(x_api_key)
    t0 = time.perf_counter()
    input_texts = req.texts if req.texts is not None else ([req.text] if req.text else [])
    if not input_texts:
        return EmbeddingResponse(model=req.model, dimension=384, embeddings=[], processing_time_ms=0.0)

    # 1. OpenAI Embeddings
    if req.model == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured on service")
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "text-embedding-3-small", "input": input_texts[:20], "dimensions": 384},
            )
            r.raise_for_status()
            data = r.json()
            vectors = [item["embedding"] for item in data["data"]]
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return EmbeddingResponse(model="text-embedding-3-small", dimension=384, embeddings=vectors, processing_time_ms=duration_ms)

    # 2. Local SentenceTransformer (bge-small or minilm)
    model = models.get_embedding_model(req.model)
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    dim = len(embeddings[0]) if len(embeddings) > 0 else 384

    return EmbeddingResponse(
        model=req.model,
        dimension=dim,
        embeddings=embeddings.tolist(),
        processing_time_ms=duration_ms,
    )


@app.post("/validate", response_model=ValidationResult)
def validate_invoice_endpoint(data: InvoiceDataSchema):
    """Pydantic validation endpoint for invoice data consistency."""
    return validate_invoice_rules(data)


@app.post("/extract", response_model=ExtractResponse)
async def extract_invoice_llm(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    provider: Literal["auto", "openai", "anthropic", "deepseek", "grok"] = Form("auto"),
    x_api_key: Optional[str] = Header(None),
):
    """End-to-End Invoice Extraction using your selected AI model (Claude, GPT-4o, DeepSeek, or Grok)."""
    verify_key(x_api_key)
    t0 = time.perf_counter()

    # 1. Obtain text (from uploaded image OCR or provided raw text)
    ocr_conf = 100.0
    text_to_process = raw_text or ""
    if file:
        content = await file.read()
        try:
            img = Image.open(io.BytesIO(content))
            text_to_process, ocr_conf, _ = _run_ocr(img)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    if not text_to_process.strip():
        raise HTTPException(status_code=400, detail="No readable text found for extraction")

    # 2. Pick provider
    active_prov = provider
    if active_prov == "auto":
        if ANTHROPIC_API_KEY:
            active_prov = "anthropic"
        elif OPENAI_API_KEY:
            active_prov = "openai"
        elif DEEPSEEK_API_KEY:
            active_prov = "deepseek"
        elif XAI_API_KEY:
            active_prov = "grok"
        else:
            raise HTTPException(status_code=400, detail="No AI provider API keys configured")

    # 3. Call the chosen AI model
    raw_output: dict = {}
    if active_prov == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")
        raw_output = call_anthropic_extractor(text_to_process, ANTHROPIC_API_KEY)
    elif active_prov == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
        raw_output = await call_openai_compatible_extractor(text_to_process, OPENAI_API_KEY, "https://api.openai.com/v1/chat/completions", "gpt-4o-mini")
    elif active_prov == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise HTTPException(status_code=400, detail="DEEPSEEK_API_KEY not configured")
        raw_output = await call_openai_compatible_extractor(text_to_process, DEEPSEEK_API_KEY, "https://api.deepseek.com/chat/completions", "deepseek-chat")
    elif active_prov == "grok":
        if not XAI_API_KEY:
            raise HTTPException(status_code=400, detail="XAI_API_KEY not configured")
        raw_output = await call_openai_compatible_extractor(text_to_process, XAI_API_KEY, "https://api.x.ai/v1/chat/completions", "grok-3")

    # 4. Parse into Pydantic schema
    items = [
        LineItemSchema(**{k: li.get(k) for k in ("description", "quantity", "unit_price", "amount")})
        for li in (raw_output.get("line_items") or [])
        if isinstance(li, dict)
    ]
    invoice_data = InvoiceDataSchema(
        invoice_number=raw_output.get("invoice_number"),
        invoice_date=raw_output.get("invoice_date"),
        supplier_name=raw_output.get("supplier_name"),
        customer_name=raw_output.get("customer_name"),
        subtotal=raw_output.get("subtotal"),
        tax=raw_output.get("tax"),
        total=raw_output.get("total"),
        currency=raw_output.get("currency"),
        line_items=items,
    )

    # 5. Validate with Pydantic
    validation = validate_invoice_pydantic(invoice_data)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    return ExtractResponse(
        provider_used=active_prov,
        data=invoice_data,
        validation=validation,
        confidence=raw_output.get("confidence") or {},
        uncertain_fields=raw_output.get("uncertain_fields") or [],
        ocr_confidence=ocr_conf,
        processing_time_ms=duration_ms,
    )


@app.post("/process", response_model=ProcessResponse)
async def process_all(
    file: UploadFile = File(...),
    vector_model: Literal["bge-small", "minilm", "openai"] = "bge-small",
    x_api_key: Optional[str] = Header(None),
):
    """Executes OCR and dense vector embedding generation in a single unified pipeline."""
    verify_key(x_api_key)
    t0 = time.perf_counter()
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    text, conf, boxes = _run_ocr(img)

    vector: List[float] = []
    dim = 384
    if text.strip():
        model = models.get_embedding_model(vector_model if vector_model != "openai" else "bge-small")
        emb = model.encode([text], normalize_embeddings=True)
        vector = emb[0].tolist()
        dim = len(vector)

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return ProcessResponse(
        text=text,
        confidence=conf,
        boxes=boxes,
        embedding=vector,
        embedding_model=vector_model,
        dimension=dim,
        processing_time_ms=duration_ms,
    )
