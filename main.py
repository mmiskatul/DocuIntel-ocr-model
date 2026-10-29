"""DocuIntel OCR & Multi-LLM Microservice Root Entrypoint.

All implementation code is cleanly decoupled within src/:
- src/config.py        : Environment settings & model registries
- src/schemas/          : Pydantic v2 domain schemas (invoice, ocr, jobs, reconciliation)
- src/services/         : OCR engine, PyMuPDF, spreadsheet parser, multi-LLM, embeddings
- src/queue/            : Redis background queue manager & worker
- src/api/              : FastAPI routes & authentication guards
"""
import uvicorn
from src.main import app
from src.config import HOST, PORT

# Expose app for ASGI servers (uvicorn main:app)
__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
