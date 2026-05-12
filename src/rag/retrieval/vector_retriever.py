"""
Vector Retriever — Episodes 4 & 5
===================================
Cosine similarity search against pgvector.
Returns LangChain Documents for LCEL chain composition.
"""
from __future__ import annotations
import logging
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag.config.settings import get_settings
from rag.ingestion.embedder import Embedder, EmbedderBackend
from rag.ingestion.indexer import VectorIndex

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Retrieves documents using cosine similarity from pgvector.
    Supports metadata filters: country, year, report_type.
    """
    def __init__(self, embedder: Embedder | None = None, top_k: int | None = None,
                 filters: dict | None = None,
                 backend: EmbedderBackend = EmbedderBackend.OPENAI):
        self.settings  = get_settings()
        self.embedder  = embedder or Embedder(backend=backend)
        self.top_k     = top_k or self.settings.retrieval_top_k
        self.filters   = filters or {}
        self._index    = VectorIndex(embedder=self.embedder)

    def retrieve(self, query: str, top_k: int | None = None,
                 filters: dict | None = None) -> list[Document]:
        k       = top_k or self.top_k
        active  = filters if filters is not None else self.filters
        q_vec   = self.embedder.embed_query(query)
        rows    = self._index.similarity_search(q_vec, top_k=k, filters=active)
        docs    = [self._to_doc(r) for r in rows]
        logger.info("Retrieved %d docs for: %.60s", len(docs), query)
        return docs

    def as_langchain_retriever(self, top_k: int | None = None,
                                filters: dict | None = None) -> "LCRetriever":
        return LCRetriever(vector_retriever=self,
                           _top_k=top_k or self.top_k,
                           _filters=filters or self.filters)

    @staticmethod
    def _to_doc(row: dict) -> Document:
        import json
        extra = {}
        if row.get("metadata"):
            try:
                extra = json.loads(row["metadata"]) if isinstance(row["metadata"], str) \
                        else row["metadata"]
            except Exception:
                pass
        meta = {k: row.get(k) for k in ("source","file_name","page_number","country",
                                          "year","report_type","report_title",
                                          "chunk_index","chunk_id")}
        meta["similarity_score"] = round(float(row.get("score", 0)), 4)
        meta.update(extra)
        return Document(page_content=row["content"], metadata=meta)


class LCRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter for LCEL pipe composition."""
    vector_retriever: VectorRetriever
    _top_k:   int   = 20
    _filters: dict  = {}

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **_: Any) -> list[Document]:
        return self.vector_retriever.retrieve(query, top_k=self._top_k,
                                              filters=self._filters)

    async def _aget_relevant_documents(self, query: str, **_: Any) -> list[Document]:
        return self._get_relevant_documents(query)
