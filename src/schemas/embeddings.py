"""Vector embedding and RAG request and response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    model: Literal["bge-small", "minilm", "openai"] = "bge-small"


class EmbeddingResponse(BaseModel):
    model: str
    dimension: int
    embeddings: List[List[float]]
    processing_time_ms: float


class RAGRequest(BaseModel):
    query: str
    context: str
    provider: Literal["auto", "openai", "anthropic", "deepseek", "grok"] = "auto"


class RAGResponse(BaseModel):
    query: str
    answer: str
    provider_used: str
    processing_time_ms: float
