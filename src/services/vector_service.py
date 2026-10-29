"""Dense vector embedding service powered 100% by OpenAI API (text-embedding-3-small) with zero local models."""
from __future__ import annotations

import hashlib
from typing import List
import httpx
import numpy as np


class VectorEmbeddingService:
    """Production vector embedding service using OpenAI text-embedding-3-small (384 dimensions)."""

    async def encode_openai(self, texts: List[str], api_key: str) -> List[List[float]]:
        """Calls OpenAI text-embedding-3-small API directly with native 384 dimensions."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "text-embedding-3-small", "input": texts[:20], "dimensions": 384},
            )
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]

    def encode_local(self, texts: List[str]) -> List[List[float]]:
        """Ultra-fast zero-model 384-dimensional deterministic fallback (zero torch, zero HuggingFace)."""
        results: List[List[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(384)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results


vector_service = VectorEmbeddingService()
