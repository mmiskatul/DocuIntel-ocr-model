"""Microservice Configuration and Environment Management."""
import os
from typing import Literal
from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()

API_KEY: str = os.getenv("API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "auto")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PORT: int = int(os.getenv("PORT", "7860"))
HOST: str = os.getenv("HOST", "0.0.0.0")

# Embedding model configuration: 100% API-driven (text-embedding-3-small)
DEFAULT_EMBEDDING_MODEL: Literal["openai"] = "openai"
EMBEDDING_DIMENSION: int = 384
