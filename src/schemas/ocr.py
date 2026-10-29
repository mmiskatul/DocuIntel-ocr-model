"""OCR bounding box and recognition response schemas."""
from __future__ import annotations

from typing import List
from pydantic import BaseModel


class OCRBoxSchema(BaseModel):
    box: List[List[float]]
    text: str
    confidence: float


class OCRResponse(BaseModel):
    text: str
    confidence: float
    boxes: List[OCRBoxSchema]
    processing_time_ms: float


class ProcessResponse(BaseModel):
    text: str
    confidence: float
    boxes: List[OCRBoxSchema]
    embedding: List[float]
    embedding_model: str
    dimension: int
    processing_time_ms: float
