"""
Embedder — Episode 3
====================
Two backends: OpenAI cloud (1536 dims) and local ONNX (384 dims).
Teaching: what embeddings ARE, cosine similarity, when to use each.
"""
from __future__ import annotations
import logging, time
from enum import Enum
import numpy as np
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)

class EmbedderBackend(str, Enum):
    OPENAI = "openai"
    ONNX   = "onnx"

class Embedder:
    def __init__(self, backend: EmbedderBackend = EmbedderBackend.OPENAI,
                 model_name_or_path: str | None = None, batch_size: int | None = None):
        self.backend    = backend
        self.settings   = get_settings()
        self.batch_size = batch_size or self.settings.embedding_batch_size
        self._client    = self._init_client(model_name_or_path)

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        texts   = [d.page_content for d in documents]
        vectors = self._embed_texts(texts)
        logger.info("Embedded %d docs → %d-dim [%s]", len(documents),
                    len(vectors[0]) if vectors else 0, self.backend.value)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self._embed_texts([query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    @property
    def dimensions(self) -> int:
        return 1536 if self.backend == EmbedderBackend.OPENAI else 384

    def _init_client(self, model_name_or_path):
        if self.backend == EmbedderBackend.OPENAI:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                openai_api_key=self.settings.openai_api_key.get_secret_value())  # type: ignore
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: uv add sentence-transformers")
        path = model_name_or_path or "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Loading ST model: %s", path)
        return SentenceTransformer(path)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._openai_batched(texts) if self.backend == EmbedderBackend.OPENAI \
               else self._onnx_batched(texts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=30), reraise=True)
    def _openai_batched(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            out.extend(self._client.embed_documents(batch))
        return out

    def _onnx_batched(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vecs  = self._client.encode(batch, batch_size=self.batch_size,
                                        show_progress_bar=False, normalize_embeddings=True)
            out.extend(vecs.tolist())
        return out


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a, b = np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

def similarity_matrix(texts: list[str], embedder: Embedder) -> np.ndarray:
    vecs  = np.array(embedder.embed_texts(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    vecs  = vecs / norms
    return vecs @ vecs.T

def top_similar(query: str, candidates: list[str], embedder: Embedder,
                top_k: int = 5) -> list[tuple[str, float]]:
    q  = np.array(embedder.embed_query(query), dtype=np.float32)
    cs = np.array(embedder.embed_texts(candidates), dtype=np.float32)
    q  = q / (np.linalg.norm(q) or 1)
    cs = cs / np.linalg.norm(cs, axis=1, keepdims=True).clip(min=1e-8)
    scores = cs @ q
    idx    = np.argsort(scores)[::-1][:top_k]
    return [(candidates[i], float(scores[i])) for i in idx]
