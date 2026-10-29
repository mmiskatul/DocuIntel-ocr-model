"""Vector embedding request and response schemas."""
from __future__ import annotations

from typing import List, Literal, Optional
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
