"""FastAPI application factory and bootstrap configuration."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import REDIS_URL
from src.queue.manager import RedisJobQueueManager
from src.api.routes import create_router

job_queue = RedisJobQueueManager(REDIS_URL)

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

app.include_router(create_router(job_queue))
