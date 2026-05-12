"""
Reranker — Episode 10
======================
Two-stage reranking: vector retrieves candidates, reranker picks the best.
Backends: Cohere Rerank API and local cross-encoder (sentence-transformers).

Episode 10 teaching: cross-encoder vs bi-encoder, why reranking improves
RAGAS context precision by the largest single margin of Phase 2.
"""
from __future__ import annotations
import logging
from enum import Enum
from langchain_core.documents import Document
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)


class RerankerBackend(str, Enum):
    COHERE        = "cohere"
    CROSS_ENCODER = "cross_encoder"


class Reranker:
    """
    Reranks a list of retrieved documents for a given query.

    Usage:
        reranker = Reranker(backend=RerankerBackend.COHERE)
        reranked = reranker.rerank(query, documents, top_n=5)
    """
    def __init__(self, backend: RerankerBackend = RerankerBackend.COHERE,
                 model: str | None = None):
        self.backend  = backend
        self.settings = get_settings()
        self.model    = model
        self._client  = self._init_client()

    def rerank(self, query: str, documents: list[Document],
               top_n: int | None = None) -> list[Document]:
        if not documents:
            return []
        n = top_n or self.settings.rerank_top_n
        if self.backend == RerankerBackend.COHERE:
            return self._cohere_rerank(query, documents, n)
        return self._cross_encoder_rerank(query, documents, n)

    def _init_client(self):
        if self.backend == RerankerBackend.COHERE:
            try:
                import cohere
            except ImportError:
                raise ImportError("Run: uv add cohere")
            key = self.settings.cohere_api_key
            if not key:
                raise ValueError("COHERE_API_KEY not set in .env")
            return cohere.Client(key.get_secret_value())
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Run: uv add sentence-transformers")
        model = self.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info("Loading CrossEncoder: %s", model)
        return CrossEncoder(model)

    def _cohere_rerank(self, query: str, docs: list[Document],
                       top_n: int) -> list[Document]:
        texts   = [d.page_content for d in docs]
        resp    = self._client.rerank(
            query=query, documents=texts,
            top_n=top_n,
            model=self.model or "rerank-english-v3.0",
        )
        results = []
        for r in resp.results:
            doc = docs[r.index]
            d   = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata,
                           "rerank_score": round(r.relevance_score, 4),
                           "rerank_rank":  r.index},
            )
            results.append(d)
        logger.info("Cohere reranked %d → %d docs", len(docs), len(results))
        return results

    def _cross_encoder_rerank(self, query: str, docs: list[Document],
                               top_n: int) -> list[Document]:
        pairs  = [(query, d.page_content) for d in docs]
        scores = self._client.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]
        results = []
        for rank, (doc, score) in enumerate(ranked):
            d = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata,
                           "rerank_score": round(float(score), 4),
                           "rerank_rank":  rank},
            )
            results.append(d)
        logger.info("CrossEncoder reranked %d → %d docs", len(docs), len(results))
        return results
