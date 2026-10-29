"""API authentication and security dependencies."""
from typing import Optional
from fastapi import Header, HTTPException
from src.config import API_KEY


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Verifies incoming X-API-Key header against configured API_KEY."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
