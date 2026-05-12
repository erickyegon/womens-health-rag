"""
Embedder — Episode 3

Converts text chunks into vector embeddings.
Two backends are supported:

1. OPENAI  — text-embedding-3-small via the OpenAI API (1536 dims)
             Fast, high quality, costs ~$0.02 per million tokens.

2. ONNX    — a local sentence-transformers model exported to ONNX.
             Zero API cost, fully private, slightly lower quality.
             Useful when data cannot leave your environment.

The Embedder class wraps both so the rest of the codebase never needs
to know which backend is in use.

Episode 3 walkthrough:
  - Run both backends on the same query
  - Compute cosine similarity between their outputs
  - Discuss when you'd choose each
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.config.settings import get_settings

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class EmbedderBackend(str, Enum):
    OPENAI = "openai"
    ONNX   = "onnx"


class Embedder:
    """
    Unified embedding interface supporting OpenAI and local ONNX backends.

    Usage:
        embedder = Embedder(backend=EmbedderBackend.OPENAI)
        vectors  = embedder.embed_documents(docs)
        query_v  = embedder.embed_query("maternal mortality Nigeria")
    """

    def __init__(
        self,
        backend: EmbedderBackend = EmbedderBackend.OPENAI,
        model_name_or_path: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.backend    = backend
        self.settings   = get_settings()
        self.batch_size = batch_size or self.settings.embedding_batch_size
        self._client    = self._init_client(model_name_or_path)

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """
        Embed a list of Documents, returning one vector per document.
        Automatically batches to respect API rate limits.
        """
        texts   = [doc.page_content for doc in documents]
        vectors = self._embed_texts(texts)
        logger.info(
            "Embedded %d documents → %d-dim vectors [%s]",
            len(documents),
            len(vectors[0]) if vectors else 0,
            self.backend.value,
        )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        vectors = self._embed_texts([query])
        return vectors[0]

    @property
    def dimensions(self) -> int:
        """Return the embedding vector dimensions."""
        if self.backend == EmbedderBackend.OPENAI:
            return 1536  # text-embedding-3-small
        return 768  # all-MiniLM-L12-v2 ONNX default

    # ── Private ───────────────────────────────────────────────────────────────

    def _init_client(self, model_name_or_path: str | None):
        if self.backend == EmbedderBackend.OPENAI:
            return self._init_openai()
        elif self.backend == EmbedderBackend.ONNX:
            return self._init_onnx(model_name_or_path)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _init_openai(self):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key.get_secret_value(),  # type: ignore[arg-type]
        )

    def _init_onnx(self, model_name_or_path: str | None):
        """
        Initialise a local ONNX model via sentence-transformers.

        Default model: all-MiniLM-L12-v2 — small (33M params), fast, good quality.
        For production privacy use cases, prefer: bge-small-en-v1.5 (BAAI)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: uv add sentence-transformers"
            )
        model_path = model_name_or_path or "sentence-transformers/all-MiniLM-L12-v2"
        logger.info("Loading local ONNX model: %s", model_path)
        return SentenceTransformer(model_path)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, with retry on transient API errors."""
        if not texts:
            return []

        if self.backend == EmbedderBackend.OPENAI:
            return self._openai_embed_batched(texts)
        else:
            return self._onnx_embed_batched(texts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True,
    )
    def _openai_embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API in batches with exponential retry."""
        all_vectors: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            logger.debug("Embedding batch %d/%d (%d texts)", batch_num, total_batches, len(batch))

            start = time.perf_counter()
            vectors = self._client.embed_documents(batch)
            elapsed = time.perf_counter() - start

            logger.debug("Batch %d embedded in %.2fs", batch_num, elapsed)
            all_vectors.extend(vectors)

        return all_vectors

    def _onnx_embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Run local ONNX model in batches."""
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._client.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine similarity friendly
            )
            all_vectors.extend(embeddings.tolist())

        return all_vectors


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Used in Episode 3 to compare OpenAI vs ONNX embeddings.

    Returns a value in [-1, 1] where 1 = identical direction.
    """
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
