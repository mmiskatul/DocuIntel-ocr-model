"""Dense vector embedding service supporting BAAI/bge-small-en-v1.5, MiniLM, and OpenAI."""
from __future__ import annotations

from typing import Dict, List, Literal
import httpx
from sentence_transformers import SentenceTransformer


class VectorEmbeddingService:
    def __init__(self):
        self._models: Dict[str, SentenceTransformer] = {}

    def get_local_model(self, name: str) -> SentenceTransformer:
        if name not in self._models:
            model_map = {
                "bge-small": "BAAI/bge-small-en-v1.5",
                "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            }
            repo = model_map.get(name, "BAAI/bge-small-en-v1.5")
            print(f"Loading SentenceTransformer: {repo}...")
            self._models[name] = SentenceTransformer(repo, device="cpu")
        return self._models[name]

    def encode_local(self, texts: List[str], model_name: str = "bge-small") -> List[List[float]]:
        model = self.get_local_model(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def encode_openai(self, texts: List[str], api_key: str) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "text-embedding-3-small", "input": texts[:20], "dimensions": 384},
            )
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]


vector_service = VectorEmbeddingService()
