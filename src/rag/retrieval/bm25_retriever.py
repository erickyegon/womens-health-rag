"""
BM25 Retriever — Episode 6
============================
Keyword search using rank-bm25. Catches exact terms, country codes, years
that semantic search misses.
"""
from __future__ import annotations
import logging, re
from langchain_core.documents import Document
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    In-memory BM25 retriever built from a list of Documents.
    Episode 6: show how 'DHS 2022 Kenya' retrieves better with BM25 than vector.
    """
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: uv add rank-bm25")
        self.documents = documents
        corpus         = [_tokenize(d.page_content) for d in documents]
        self._bm25     = BM25Okapi(corpus, k1=k1, b=b)
        logger.info("BM25 index built over %d documents", len(documents))

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        settings = get_settings()
        k        = top_k or settings.retrieval_top_k
        tokens   = _tokenize(query)
        scores   = self._bm25.get_scores(tokens)
        top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results  = []
        for idx in top_idx:
            doc = self.documents[idx]
            d   = Document(page_content=doc.page_content,
                           metadata={**doc.metadata, "bm25_score": round(float(scores[idx]), 4)})
            results.append(d)
        return results

    @classmethod
    def from_vector_index(cls, index, top_k: int = 10000) -> "BM25Retriever":
        """Build from all documents in a VectorIndex (fetches without embedding)."""
        raise NotImplementedError("Use from_documents() with pre-loaded docs")

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "BM25Retriever":
        return cls(documents)
