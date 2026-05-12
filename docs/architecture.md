# System Architecture

## Data flow

```
data/raw/*.pdf
    │
    ▼ ingestion/loader.py      (PyMuPDF + pdfplumber fallback)
    │
    ▼ ingestion/cleaner.py     (Unicode, hyphenation, headers, whitespace)
    │
    ▼ ingestion/chunker.py     (fixed | recursive | semantic)
    │
    ▼ ingestion/embedder.py    (OpenAI text-embedding-3-small | ONNX)
    │
    ▼ ingestion/indexer.py     (pgvector upsert with HNSW index)
    │
  pgvector (PostgreSQL 16)
    │
    ▼ retrieval/
    │   ├── vector_retriever.py   (cosine similarity, metadata filters)
    │   ├── bm25_retriever.py     (keyword search via rank-bm25)
    │   ├── hybrid_retriever.py   (RRF fusion of vector + BM25)
    │   └── reranker.py           (Cohere Rerank | cross-encoder)
    │
    ▼ chains/ (Phase 1–2 — LangChain LCEL)
    │   ├── rag_chain.py          (base retrieval-generation chain)
    │   ├── conversational_chain.py
    │   └── structured_chain.py   (Pydantic typed outputs + citations)
    │
    ▼ agent/ (Phase 3 — LangGraph)
    │   ├── graph.py              (StateGraph definition)
    │   ├── router.py             (query complexity classifier)
    │   ├── nodes.py              (retrieve, grade, rewrite, answer)
    │   └── checkpointer.py       (human-in-the-loop persistence)
    │
    ▼ api/main.py (Phase 4 — FastAPI)
    │   └── routes/query.py       (POST /query — streaming SSE)
    │
    ▼ ui/app.py (Phase 4 — Streamlit)
        ├── components/chat.py
        └── components/sources.py
```

## pgvector schema

```sql
CREATE TABLE document_chunks (
    id           UUID PRIMARY KEY,
    content_hash VARCHAR(64) UNIQUE,   -- idempotent upserts
    content      TEXT,
    embedding    vector(1536),          -- HNSW indexed
    country      TEXT,                  -- indexed for filtering
    year         TEXT,                  -- indexed for filtering
    report_type  TEXT,
    report_title TEXT,
    source       TEXT,
    page_number  INTEGER,
    chunk_index  INTEGER,
    metadata     JSONB,
    created_at   TIMESTAMPTZ
);
```

## LangGraph agent (Phase 3)

```
START
  │
  ▼ router_node          ← classifies query complexity
  │
  ├─ direct  ──────────► answer_node ──► END
  │
  ├─ vector  ──────────► retrieve_node
  │                           │
  ├─ hybrid  ──────────►      ▼
  │                      grade_node   ← grades each retrieved chunk
  └─ multihop ─────────►      │
                         ┌────┴────┐
                    relevant?     not relevant?
                         │              │
                         ▼              ▼
                    answer_node    rewrite_node ──► retrieve_node (retry)
                         │
                    hallucination_check
                         │
                    ┌────┴────┐
                grounded?   not grounded?
                    │              │
                    ▼              ▼
                   END        rewrite_node (retry, max 2x)
```
