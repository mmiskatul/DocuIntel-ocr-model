"""Redis background job status and submission schemas."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class JobEnqueueResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    created_at: float


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: float
    retries: int
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: float
    updated_at: float

