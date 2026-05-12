"""
Indexer — Episode 4

Creates and manages the pgvector table, upserts embedded chunks,
and provides the database connection pool.

Schema design decisions (Episode 4 walkthrough):
- id:         UUID primary key — stable across re-ingestion
- embedding:  vector(1536) with HNSW index — fast approximate search
- content:    full chunk text — returned with search results
- metadata:   JSONB — flexible, queryable with Postgres JSON operators
- Separate typed columns for key metadata fields:
    country, year, report_type, source, page_number
  These are indexed separately for efficient WHERE-clause filtering.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from rag.config.settings import get_settings
from rag.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

# ── SQL statements ────────────────────────────────────────────────────────────

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) UNIQUE NOT NULL,   -- SHA-256 of content — idempotent upserts
    content      TEXT NOT NULL,
    embedding    vector({dims}),
    -- Typed metadata columns for indexed filtering
    source       TEXT,
    file_name    TEXT,
    page_number  INTEGER,
    country      TEXT,
    year         TEXT,
    report_type  TEXT,
    report_title TEXT,
    chunk_index  INTEGER,
    -- Full JSONB metadata for anything else
    metadata     JSONB DEFAULT '{{}}'::jsonb,
    -- Housekeeping
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
ON {table}
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

CREATE_METADATA_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS {table}_country_idx      ON {table} (country);
CREATE INDEX IF NOT EXISTS {table}_year_idx         ON {table} (year);
CREATE INDEX IF NOT EXISTS {table}_report_type_idx  ON {table} (report_type);
CREATE INDEX IF NOT EXISTS {table}_source_idx       ON {table} (source);
"""

UPSERT_SQL = """
INSERT INTO {table} (
    id, content_hash, content, embedding,
    source, file_name, page_number,
    country, year, report_type, report_title,
    chunk_index, metadata
)
VALUES (
    :id, :content_hash, :content, :embedding,
    :source, :file_name, :page_number,
    :country, :year, :report_type, :report_title,
    :chunk_index, :metadata
)
ON CONFLICT (content_hash) DO UPDATE SET
    embedding    = EXCLUDED.embedding,
    metadata     = EXCLUDED.metadata,
    created_at   = NOW();
"""


class VectorIndex:
    """
    Manages the pgvector table and provides upsert / search operations.

    Used by:
      - scripts/ingest.py    → upsert all documents
      - retrieval/vector_retriever.py → similarity search
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.settings = get_settings()
        self.table    = self.settings.vector_table_name
        self.dims     = self.settings.embedding_dimensions
        self.embedder = embedder or Embedder()
        self._engine  = create_engine(
            self.settings.database_url_str,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # reconnect on stale connections
        )

    def init_schema(self) -> None:
        """
        Create the pgvector extension, table, and indexes if they don't exist.
        Safe to call multiple times — all statements use IF NOT EXISTS.
        """
        with self._engine.begin() as conn:
            conn.execute(text(CREATE_EXTENSION_SQL))
            conn.execute(text(CREATE_TABLE_SQL.format(table=self.table, dims=self.dims)))
            conn.execute(text(CREATE_HNSW_INDEX_SQL.format(table=self.table)))
            conn.execute(text(CREATE_METADATA_INDEXES_SQL.format(table=self.table)))
        logger.info("Schema initialised — table: %s, dims: %d", self.table, self.dims)

    def upsert_documents(self, documents: list[Document]) -> int:
        """
        Embed and upsert a list of Documents into pgvector.

        Uses content_hash as the conflict key so re-ingesting the same
        document is idempotent — no duplicate embeddings.

        Returns the number of rows affected.
        """
        if not documents:
            return 0

        logger.info("Upserting %d documents...", len(documents))
        vectors = self.embedder.embed_documents(documents)

        rows = [
            self._doc_to_row(doc, vector)
            for doc, vector in zip(documents, vectors, strict=True)
        ]

        with self._engine.begin() as conn:
            for row in rows:
                conn.execute(text(UPSERT_SQL.format(table=self.table)), row)

        logger.info("Upserted %d rows into %s", len(rows), self.table)
        return len(rows)

    def count(self) -> int:
        """Return the total number of chunks indexed."""
        with self._engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self.table}"))
            return result.scalar() or 0

    def drop_and_recreate(self) -> None:
        """
        Drop the table and recreate it from scratch.
        Use with caution — this deletes all indexed data.
        """
        with self._engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table}"))
        self.init_schema()
        logger.warning("Table %s dropped and recreated.", self.table)

    # ── Private ───────────────────────────────────────────────────────────────

    def _doc_to_row(self, doc: Document, vector: list[float]) -> dict[str, Any]:
        """Convert a Document + embedding vector to a DB row dict."""
        meta = doc.metadata
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()

        # Extract typed columns from metadata, leave the rest in metadata JSONB
        typed_keys = {
            "source", "file_name", "page_number", "country",
            "year", "report_type", "report_title", "chunk_index",
        }
        extra_meta = {k: v for k, v in meta.items() if k not in typed_keys}

        return {
            "id":           str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash)),
            "content_hash": content_hash,
            "content":      doc.page_content,
            "embedding":    json.dumps(vector),   # pgvector accepts JSON array string
            "source":       meta.get("source"),
            "file_name":    meta.get("file_name"),
            "page_number":  meta.get("page_number"),
            "country":      meta.get("country"),
            "year":         meta.get("year"),
            "report_type":  meta.get("report_type"),
            "report_title": meta.get("report_title"),
            "chunk_index":  meta.get("chunk_index"),
            "metadata":     json.dumps(extra_meta),
        }


def main() -> None:
    """CLI entry point: python -m rag.ingestion.indexer"""
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = Path("data/raw")
    if not data_dir.exists():
        logger.error("data/raw directory not found. Add PDFs there first.")
        sys.exit(1)

    from rag.ingestion.chunker import ChunkStrategy, chunk_pages
    from rag.ingestion.cleaner import clean_pages
    from rag.ingestion.loader import load_directory

    logger.info("Starting ingestion pipeline...")

    pages = load_directory(data_dir)
    if not pages:
        logger.warning("No pages loaded. Check that data/raw/ contains PDFs.")
        return

    cleaned = clean_pages(pages)
    docs    = chunk_pages(cleaned, strategy=ChunkStrategy.RECURSIVE)

    index = VectorIndex()
    index.init_schema()
    count = index.upsert_documents(docs)

    logger.info("Ingestion complete. %d chunks indexed. Total in DB: %d", count, index.count())


if __name__ == "__main__":
    main()
