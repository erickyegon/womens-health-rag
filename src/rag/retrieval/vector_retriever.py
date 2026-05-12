"""
Vector Retriever — Episode 4 / 5

Runs cosine similarity search against the pgvector table.
Supports optional metadata filters for country, year, and report type
(used in Episode 7 — self-querying retrieval).

Returns LangChain Document objects so the results plug directly into chains.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import create_engine, text

from rag.config.settings import get_settings
from rag.ingestion.embedder import Embedder, EmbedderBackend

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Retrieves documents from pgvector using cosine similarity.

    Supports optional metadata filters:
        retriever = VectorRetriever(filters={"country": "Nigeria", "year": "2021"})
        docs = retriever.retrieve("maternal mortality rates")
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        backend: EmbedderBackend = EmbedderBackend.OPENAI,
    ) -> None:
        self.settings = get_settings()
        self.embedder = embedder or Embedder(backend=backend)
        self.top_k    = top_k or self.settings.retrieval_top_k
        self.filters  = filters or {}
        self.table    = self.settings.vector_table_name
        self._engine  = create_engine(
            self.settings.database_url_str,
            pool_pre_ping=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[Document]:
        """
        Embed the query and return the top-k most similar chunks.

        Args:
            query:   The user's question.
            top_k:   Number of results (overrides constructor value).
            filters: Metadata filters (overrides constructor value).

        Returns:
            List of Document objects sorted by cosine similarity (most similar first).
        """
        k       = top_k or self.top_k
        active_filters = filters if filters is not None else self.filters

        query_vector = self.embedder.embed_query(query)
        sql, params  = self._build_query(query_vector, k, active_filters)

        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()

        docs = [self._row_to_document(row) for row in rows]
        logger.info(
            "Retrieved %d documents for query: %.60s...",
            len(docs),
            query,
        )
        return docs

    def as_langchain_retriever(self) -> "LangChainRetrieverAdapter":
        """
        Return a LangChain-compatible BaseRetriever adapter.
        This is what plugs into LCEL chains with the | operator.
        """
        return LangChainRetrieverAdapter(vector_retriever=self)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_query(
        self,
        vector: list[float],
        k: int,
        filters: dict[str, str],
    ) -> tuple[str, dict]:
        """Build the parameterised SQL query with optional WHERE clauses."""
        # Allowed filter columns — prevents SQL injection via metadata keys
        allowed_filters = {"country", "year", "report_type", "report_title"}
        where_clauses   = []
        params: dict[str, Any] = {"k": k, "embedding": str(vector)}

        for key, value in filters.items():
            if key not in allowed_filters:
                logger.warning("Ignoring unknown filter key: %s", key)
                continue
            param_name = f"filter_{key}"
            where_clauses.append(f"{key} = :{param_name}")
            params[param_name] = value

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT
                content,
                source,
                file_name,
                page_number,
                country,
                year,
                report_type,
                report_title,
                chunk_index,
                metadata,
                1 - (embedding <=> :embedding::vector) AS similarity_score
            FROM {self.table}
            {where_sql}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :k
        """
        return sql, params

    @staticmethod
    def _row_to_document(row: Any) -> Document:
        """Convert a SQLAlchemy Row to a LangChain Document."""
        import json
        extra_meta = {}
        if row.metadata:
            try:
                extra_meta = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
            except (ValueError, TypeError):
                pass

        metadata = {
            "source":           row.source,
            "file_name":        row.file_name,
            "page_number":      row.page_number,
            "country":          row.country,
            "year":             row.year,
            "report_type":      row.report_type,
            "report_title":     row.report_title,
            "chunk_index":      row.chunk_index,
            "similarity_score": round(float(row.similarity_score), 4),
            **extra_meta,
        }
        return Document(page_content=row.content, metadata=metadata)


class LangChainRetrieverAdapter(BaseRetriever):
    """
    Thin adapter so VectorRetriever works as a LangChain BaseRetriever.
    Required for LCEL chain composition with the | operator.
    """

    vector_retriever: VectorRetriever

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return self.vector_retriever.retrieve(query)

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        # Sync fallback — async DB support added in Phase 4
        return self._get_relevant_documents(query)
