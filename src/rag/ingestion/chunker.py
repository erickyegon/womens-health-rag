"""
Chunker — Episode 2

Converts cleaned pages into LangChain Document chunks with full metadata.
Three strategies are implemented so we can compare them in Episode 2:

1. FIXED    — split every N characters with M overlap (fastest, dumbest)
2. RECURSIVE — split on paragraph → sentence → word boundaries (best default)
3. SEMANTIC  — use embeddings to split on semantic similarity shifts (Episode 15)

The strategy is passed as an argument so callers can swap and benchmark.
RECURSIVE is the default throughout the course until Episode 15.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config.settings import get_settings
from rag.ingestion.loader import RawPage

logger = logging.getLogger(__name__)


class ChunkStrategy(str, Enum):
    FIXED     = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC  = "semantic"     # introduced in Episode 15


def chunk_pages(
    pages: list[RawPage],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Chunk a list of cleaned pages into LangChain Documents.

    Args:
        pages:         Cleaned RawPage objects from cleaner.py.
        strategy:      Chunking algorithm to use.
        chunk_size:    Override settings.chunk_size if provided.
        chunk_overlap: Override settings.chunk_overlap if provided.

    Returns:
        List of LangChain Document objects with full metadata attached.

    Episode 2 walkthrough:
        We run this with all three strategies on the same DHS report and
        print summary statistics (num_chunks, avg_chars, min_chars, max_chars).
        This reveals why FIXED chunking cuts through sentences mid-thought.
    """
    settings    = get_settings()
    chunk_size  = chunk_size  or settings.chunk_size
    chunk_over  = chunk_overlap or settings.chunk_overlap

    if strategy == ChunkStrategy.FIXED:
        splitter = _fixed_splitter(chunk_size, chunk_over)
    elif strategy == ChunkStrategy.RECURSIVE:
        splitter = _recursive_splitter(chunk_size, chunk_over)
    elif strategy == ChunkStrategy.SEMANTIC:
        # Semantic chunking requires an embedder — imported lazily
        # to avoid import errors in Episodes 1–14 where it isn't used.
        return _semantic_chunk(pages, chunk_size)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    all_docs: list[Document] = []

    for page in pages:
        # Convert RawPage to a temporary Document for the splitter
        base_doc = Document(
            page_content=page.text,
            metadata=_build_metadata(page),
        )
        chunks = splitter.split_documents([base_doc])

        # Enrich each chunk with its position within the source page
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]    = i
            chunk.metadata["chunk_count"]    = len(chunks)
            chunk.metadata["char_count"]     = len(chunk.page_content)
            chunk.metadata["chunk_strategy"] = strategy.value

        all_docs.extend(chunks)

    _log_stats(all_docs, strategy)
    return all_docs


def chunk_stats(docs: list[Document]) -> dict:
    """Return summary statistics for a list of chunks — used in Episode 2 demo."""
    if not docs:
        return {}
    char_counts = [len(d.page_content) for d in docs]
    return {
        "total_chunks": len(docs),
        "avg_chars":    round(sum(char_counts) / len(char_counts)),
        "min_chars":    min(char_counts),
        "max_chars":    max(char_counts),
        "total_chars":  sum(char_counts),
    }


# ── Splitter factories ────────────────────────────────────────────────────────

def _fixed_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Fixed character splitter — no respect for sentence or paragraph boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],   # fallback to char-level
        length_function=len,
        add_start_index=True,
    )


def _recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Recursive character splitter — the workhorse of this course.

    Tries to split on paragraph breaks first, then sentences, then words.
    This is the right default for dense prose documents like DHS reports.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",    # paragraph break — strongest signal
            "\n",      # line break
            ". ",      # sentence end
            "! ",
            "? ",
            "; ",      # semi-colon — common in statistical text
            ", ",      # comma — last resort before word split
            " ",
            "",        # character-level fallback
        ],
        length_function=len,
        add_start_index=True,
    )


def _semantic_chunk(pages: list[RawPage], chunk_size: int) -> list[Document]:
    """
    Semantic chunking — splits on embedding similarity shifts.
    Introduced in Episode 15. Requires the embedder module.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "Semantic chunking requires langchain-experimental and langchain-openai.\n"
            "Run: uv add langchain-experimental"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter   = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    all_docs: list[Document] = []
    for page in pages:
        base_doc = Document(page_content=page.text, metadata=_build_metadata(page))
        chunks   = splitter.split_documents([base_doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]    = i
            chunk.metadata["chunk_count"]    = len(chunks)
            chunk.metadata["char_count"]     = len(chunk.page_content)
            chunk.metadata["chunk_strategy"] = "semantic"
        all_docs.extend(chunks)

    _log_stats(all_docs, ChunkStrategy.SEMANTIC)
    return all_docs


# ── Metadata builder ──────────────────────────────────────────────────────────

def _build_metadata(page: RawPage) -> dict:
    """
    Build the metadata dict attached to every chunk.

    This metadata is stored alongside the embedding in pgvector and
    is what allows us to do filtered retrieval in Episode 7:
    "Show me data from Nigeria between 2018 and 2022."
    """
    return {
        # Source provenance
        "source":       page.source_file,
        "file_name":    page.file_name,
        "page_number":  page.page_number,
        "total_pages":  page.total_pages,
        # Domain metadata — the key differentiator vs generic RAG
        "country":      page.country,
        "year":         page.year,
        "report_type":  page.report_type,
        "report_title": page.report_title,
    }


# ── Logging ───────────────────────────────────────────────────────────────────

def _log_stats(docs: list[Document], strategy: ChunkStrategy) -> None:
    stats = chunk_stats(docs)
    logger.info(
        "Chunking complete [%s] — %d chunks | avg %d chars | range %d–%d chars",
        strategy.value,
        stats["total_chunks"],
        stats["avg_chars"],
        stats["min_chars"],
        stats["max_chars"],
    )
