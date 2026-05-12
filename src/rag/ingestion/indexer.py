"""
Indexer — Episode 4
====================
Creates the pgvector table and upserts embedded chunks.
Teaches: HNSW vs IVFFlat, cosine vs L2, idempotent upserts via content hash.
"""
from __future__ import annotations
import hashlib, json, logging, uuid
from typing import Any
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from rag.config.settings import get_settings
from rag.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector({dims}),
    source       TEXT, file_name TEXT, page_number INTEGER,
    country      TEXT, year TEXT, report_type TEXT, report_title TEXT,
    chunk_index  INTEGER, chunk_id VARCHAR(16),
    metadata     JSONB DEFAULT '{{}}'::jsonb,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);"""

CREATE_HNSW = """
CREATE INDEX IF NOT EXISTS {table}_hnsw
ON {table} USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS {table}_country_idx ON {table}(country);
CREATE INDEX IF NOT EXISTS {table}_year_idx    ON {table}(year);
CREATE INDEX IF NOT EXISTS {table}_type_idx    ON {table}(report_type);"""

UPSERT = """
INSERT INTO {table} (id,content_hash,content,embedding,source,file_name,
  page_number,country,year,report_type,report_title,chunk_index,chunk_id,metadata)
VALUES (:id,:content_hash,:content,:embedding,:source,:file_name,
  :page_number,:country,:year,:report_type,:report_title,:chunk_index,:chunk_id,:metadata)
ON CONFLICT (content_hash) DO UPDATE SET
  embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata, created_at=NOW();"""


class VectorIndex:
    def __init__(self, embedder: Embedder | None = None):
        self.settings = get_settings()
        self.table    = self.settings.vector_table_name
        self.dims     = self.settings.embedding_dimensions
        self.embedder = embedder or Embedder()
        self._engine  = create_engine(
            self.settings.database_url_str, poolclass=QueuePool,
            pool_size=5, max_overflow=10, pool_pre_ping=True)

    def init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(CREATE_EXTENSION))
            conn.execute(text(CREATE_TABLE.format(table=self.table, dims=self.dims)))
            conn.execute(text(CREATE_HNSW.format(table=self.table)))
            conn.execute(text(CREATE_INDEXES.format(table=self.table)))
        logger.info("Schema ready — table: %s, dims: %d", self.table, self.dims)

    def upsert_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        vectors = self.embedder.embed_documents(documents)
        rows    = [self._to_row(d, v) for d, v in zip(documents, vectors, strict=True)]
        with self._engine.begin() as conn:
            for row in rows:
                conn.execute(text(UPSERT.format(table=self.table)), row)
        logger.info("Upserted %d rows into %s", len(rows), self.table)
        return len(rows)

    def similarity_search(
        self, query_vector: list[float], top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Raw similarity search — used by VectorRetriever."""
        where, params = self._build_where(filters or {})
        params["k"]         = top_k
        params["embedding"] = str(query_vector)
        sql = f"""
            SELECT content, source, file_name, page_number, country, year,
                   report_type, report_title, chunk_index, chunk_id, metadata,
                   1 - (embedding <=> :embedding::vector) AS score
            FROM {self.table} {where}
            ORDER BY embedding <=> :embedding::vector LIMIT :k"""
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return [dict(r._mapping) for r in rows]

    def count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(text(f"SELECT COUNT(*) FROM {self.table}")).scalar() or 0

    def drop_and_recreate(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table}"))
        self.init_schema()
        logger.warning("Table %s dropped and recreated.", self.table)

    def _to_row(self, doc: Document, vector: list[float]) -> dict:
        meta = doc.metadata
        ch   = hashlib.sha256(doc.page_content.encode()).hexdigest()
        extra = {k: v for k, v in meta.items()
                 if k not in {"source","file_name","page_number","country","year",
                               "report_type","report_title","chunk_index","chunk_id"}}
        return {
            "id":           str(uuid.uuid5(uuid.NAMESPACE_DNS, ch)),
            "content_hash": ch,
            "content":      doc.page_content,
            "embedding":    json.dumps(vector),
            "source":       meta.get("source"),
            "file_name":    meta.get("file_name"),
            "page_number":  meta.get("page_number"),
            "country":      meta.get("country"),
            "year":         meta.get("year"),
            "report_type":  meta.get("report_type"),
            "report_title": meta.get("report_title"),
            "chunk_index":  meta.get("chunk_index"),
            "chunk_id":     meta.get("chunk_id"),
            "metadata":     json.dumps(extra),
        }

    def _build_where(self, filters: dict) -> tuple[str, dict]:
        allowed = {"country", "year", "report_type", "report_title"}
        clauses, params = [], {}
        for k, v in filters.items():
            if k not in allowed:
                continue
            clauses.append(f"{k} = :f_{k}")
            params[f"f_{k}"] = v
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params


def main() -> None:
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from rag.ingestion.chunker import ChunkStrategy, chunk_pages
    from rag.ingestion.cleaner import clean_pages
    from rag.ingestion.loader import load_directory
    import json as _json

    data_dir  = Path("data/raw")
    meta_file = Path("data/metadata.json")
    meta_map  = _json.loads(meta_file.read_text()) if meta_file.exists() else {}

    pages   = load_directory(data_dir, metadata_map=meta_map)
    cleaned = clean_pages(pages)
    docs    = chunk_pages(cleaned, ChunkStrategy.RECURSIVE)

    idx = VectorIndex()
    idx.init_schema()
    n   = idx.upsert_documents(docs)
    logger.info("Done. %d chunks indexed. Total: %d", n, idx.count())

if __name__ == "__main__":
    main()
