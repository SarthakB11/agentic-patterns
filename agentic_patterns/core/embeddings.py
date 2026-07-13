"""Text embedders for the RAG and memory patterns.

`HashEmbedder` is a deterministic, stdlib-only stand-in for a real embedding
model: it preserves the geometry-of-similarity idea that overlapping token
sets produce higher cosine similarity, without needing a network call or a
trained model. Swap in `OpenAIEmbedder` for real embeddings.
"""

from __future__ import annotations

import abc
import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Any


class Embedder(abc.ABC):
    """Interface every text embedder implements."""

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text."""
        raise NotImplementedError


class HashEmbedder(Embedder):
    """A deterministic, stdlib-only embedder using feature hashing.

    This is a teaching stand-in for a real embedding model. It cannot
    capture meaning, but it preserves the property that matters for
    demonstrating retrieval: texts with overlapping vocabulary land closer
    together in vector space than texts with disjoint vocabulary. The same
    text always maps to the same vector, on any machine, since it uses only
    `hashlib.md5` rather than Python's randomized `hash()`.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm, rather than raising a
    division-by-zero error.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class OpenAIEmbedder(Embedder):
    """An `Embedder` backed by the OpenAI embeddings API."""

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                'OpenAIEmbedder requires httpx. Install with: pip install "agentic-patterns[providers]"'
            ) from exc
        self._httpx = httpx
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or "text-embedding-3-small"

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [item["embedding"] for item in data["data"]]
