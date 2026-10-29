"""FastAPI routing and dependencies."""
from src.api.routes import create_router
from src.api.dependencies import verify_api_key

__all__ = ["create_router", "verify_api_key"]

