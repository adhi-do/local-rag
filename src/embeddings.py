from abc import ABC, abstractmethod
import math
import hashlib
from typing import List
import requests


class BaseEmbedding(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, model: str = "qwen2.5-coder:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/api/embeddings"

    def embed_text(self, text: str) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as err:
            raise ConnectionError(
                f"Could not connect to Ollama at '{self.endpoint}'. "
                f"Is Ollama running? (Start it with `ollama serve` or use `--mock`)"
            ) from err
        data = response.json()
        return data["embedding"]


class MockEmbedding(BaseEmbedding):
    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            vector[idx] += 1.0

        norm = math.sqrt(sum(val * val for val in vector))
        if norm > 0:
            vector = [val / norm for val in vector]
        return vector
