"""FastAPI routes for OCR, embeddings, validation, spreadsheet parsing, and reconciliation."""
from __future__ import annotations

import io
import time
from typing import List, Literal, Optional
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from src.config import (
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    OPENAI_API_KEY,
    XAI_API_KEY,
)
from src.api.dependencies import verify_api_key
from src.queue.manager import RedisJobQueueManager
from src.schemas.invoice import (
    ExtractResponse,
    InvoiceDataSchema,
    LineItemSchema,
    ValidationResult,
)
from src.schemas.ocr import OCRResponse, ProcessResponse
from src.schemas.spreadsheet import SpreadsheetResponse
from src.schemas.jobs import JobEnqueueResponse, JobStatusResponse
from src.schemas.reconciliation import ReconciliationResponse
from src.schemas.embeddings import EmbeddingRequest, EmbeddingResponse, RAGRequest, RAGResponse
from src.services.ocr_engine import ocr_service
from src.services.pdf_parser import pdf_parser_service
from src.services.spreadsheet_parser import spreadsheet_parser_service
from src.services.llm_extractor import llm_service
from src.services.vector_service import vector_service
from src.services.validation_service import validation_service
from src.services.reconciliation_service import reconciliation_service

router = APIRouter()


def create_router(queue_mgr: RedisJobQueueManager) -> APIRouter:
    @router.get("/")
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

    @router.get("/health")
    def health():
        return {"status": "ok"}

    @router.post("/ocr", response_model=OCRResponse, dependencies=[Depends(verify_api_key)])
    async def ocr_endpoint(file: UploadFile = File(...)):
        t0 = time.perf_counter()
        content = await file.read()
        try:
            img = Image.open(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

        text, conf, boxes = await ocr_service.run_ocr_api(img)
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return OCRResponse(text=text, confidence=conf, boxes=boxes, processing_time_ms=duration_ms)

    @router.post("/parse-pdf", response_model=OCRResponse, dependencies=[Depends(verify_api_key)])
    async def parse_pdf_endpoint(file: UploadFile = File(...)):
        t0 = time.perf_counter()
        content = await file.read()
        try:
            text, conf, boxes, _ = pdf_parser_service.parse_pdf(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF document: {exc}")

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return OCRResponse(text=text, confidence=conf, boxes=boxes, processing_time_ms=duration_ms)

    @router.post("/parse-spreadsheet", response_model=SpreadsheetResponse, dependencies=[Depends(verify_api_key)])
    async def parse_spreadsheet_endpoint(file: UploadFile = File(...)):
        t0 = time.perf_counter()
        content = await file.read()
        try:
            parsed = spreadsheet_parser_service.parse(content, file.filename or "spreadsheet.xlsx")
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

    @router.post("/jobs/enqueue", response_model=JobEnqueueResponse, dependencies=[Depends(verify_api_key)])
    async def enqueue_job(
        filename: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
    ):
        fname = filename or (file.filename if file else "document.pdf")
        ctype = file.content_type if file else "application/pdf"
        job_id = queue_mgr.enqueue({"filename": fname, "content_type": ctype})
        return JobEnqueueResponse(job_id=job_id, status="queued", stage="Queued", created_at=time.time())

    @router.get("/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(verify_api_key)])
    async def get_job_status(job_id: str):
        job = queue_mgr.get(job_id)
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

    @router.post("/reconcile", response_model=ReconciliationResponse, dependencies=[Depends(verify_api_key)])
    async def reconcile_documents():
        return reconciliation_service.reconcile()

    @router.post("/embed", response_model=EmbeddingResponse, dependencies=[Depends(verify_api_key)])
    async def embed_endpoint(req: EmbeddingRequest):
        t0 = time.perf_counter()
        input_texts = req.texts if req.texts is not None else ([req.text] if req.text else [])
        if not input_texts:
            return EmbeddingResponse(model="text-embedding-3-small", dimension=384, embeddings=[], processing_time_ms=0.0)

        if OPENAI_API_KEY:
            vectors = await vector_service.encode_openai(input_texts, OPENAI_API_KEY)
        else:
            vectors = vector_service.encode_local(input_texts)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return EmbeddingResponse(model="text-embedding-3-small", dimension=384, embeddings=vectors, processing_time_ms=duration_ms)

    @router.post("/rag", response_model=RAGResponse, dependencies=[Depends(verify_api_key)])
    async def rag_completion_endpoint(req: RAGRequest):
        """High-speed RAG generation answering natural language financial queries over retrieved context."""
        t0 = time.perf_counter()
        keys = {
            "anthropic": ANTHROPIC_API_KEY,
            "openai": OPENAI_API_KEY,
            "deepseek": DEEPSEEK_API_KEY,
            "xai": XAI_API_KEY,
        }
        answer, provider_used = await llm_service.answer_rag(req.query, req.context, req.provider, keys)
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        return RAGResponse(
            query=req.query,
            answer=answer,
            provider_used=provider_used,
            processing_time_ms=duration_ms,
        )

    @router.post("/validate", response_model=ValidationResult)
    def validate_invoice_endpoint(data: InvoiceDataSchema):
        return validation_service.validate(data)

    @router.post("/extract", response_model=ExtractResponse, dependencies=[Depends(verify_api_key)])
    async def extract_invoice_llm(
        file: Optional[UploadFile] = File(None),
        raw_text: Optional[str] = Form(None),
        provider: Literal["auto", "openai", "anthropic", "deepseek", "grok"] = Form("auto"),
    ):
        t0 = time.perf_counter()
        ocr_conf = 100.0
        text_to_process = raw_text or ""
        if file:
            content = await file.read()
            try:
                img = Image.open(io.BytesIO(content))
                text_to_process, ocr_conf, _ = ocr_service.run_ocr(img)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

        if not text_to_process.strip():
            raise HTTPException(status_code=400, detail="No readable text found for extraction")

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

        raw_output: dict = {}
        if active_prov == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")
            raw_output = llm_service.call_anthropic(text_to_process, ANTHROPIC_API_KEY)
        elif active_prov == "openai":
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
            raw_output = await llm_service.call_openai_compatible(text_to_process, OPENAI_API_KEY, "https://api.openai.com/v1/chat/completions", "gpt-4o-mini")
        elif active_prov == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise HTTPException(status_code=400, detail="DEEPSEEK_API_KEY not configured")
            raw_output = await llm_service.call_openai_compatible(text_to_process, DEEPSEEK_API_KEY, "https://api.deepseek.com/chat/completions", "deepseek-chat")
        elif active_prov == "grok":
            if not XAI_API_KEY:
                raise HTTPException(status_code=400, detail="XAI_API_KEY not configured")
            raw_output = await llm_service.call_openai_compatible(text_to_process, XAI_API_KEY, "https://api.x.ai/v1/chat/completions", "grok-3")

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

        validation = validation_service.validate(invoice_data)
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

    @router.post("/process", response_model=ProcessResponse, dependencies=[Depends(verify_api_key)])
    async def process_all(
        file: UploadFile = File(...),
        vector_model: Literal["bge-small", "minilm", "openai"] = "bge-small",
    ):
        t0 = time.perf_counter()
        content = await file.read()
        try:
            img = Image.open(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

        text, conf, boxes = await ocr_service.run_ocr_api(img)

        vector: List[float] = []
        dim = 384
        if text.strip():
            vectors = vector_service.encode_local([text], vector_model if vector_model != "openai" else "bge-small")
            vector = vectors[0]
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

    return router

