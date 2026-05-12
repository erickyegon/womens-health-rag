"""
Chunker — Episode 2
====================
Converts cleaned RawPages into LangChain Document chunks with full metadata.

Four strategies are implemented and benchmarked in this episode:

  FIXED      Split every N characters regardless of content boundaries.
             Fastest. Worst quality. Good baseline to compare against.

  RECURSIVE  Split on paragraph → sentence → word boundaries in order.
             Best general-purpose default for dense prose documents.
             Used as the course default from Episode 2 through Episode 14.

  SENTENCE   Split strictly on sentence boundaries using regex.
             Useful when documents have very long paragraphs.
             Good for the Ethiopia mini-report (FR363) which is narrative-heavy.

  PARENT_CHILD  Index small child chunks for retrieval precision.
                Pass the larger parent chunk to the LLM for context.
                Episode 15 introduces the full semantic version; here we use
                a simple recursive implementation to preview the concept.

The SEMANTIC strategy (Episode 15) is scaffolded but gated behind a lazy import.

Episode 2 walkthrough:
  1. Run FIXED on one page — show mid-sentence splits
  2. Run RECURSIVE on same page — show clean paragraph boundaries
  3. Run SENTENCE — compare
  4. Compare all strategies with chunk_stats() on the full corpus
  5. Inspect metadata on a chunk — show every field attached
  6. Introduce PARENT_CHILD — preview of Episode 15's advanced technique
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config.settings import get_settings
from rag.ingestion.loader import RawPage

logger = logging.getLogger(__name__)


# ── Strategy enum ─────────────────────────────────────────────────────────────

class ChunkStrategy(str, Enum):
    FIXED        = "fixed"
    RECURSIVE    = "recursive"    # default throughout the course
    SENTENCE     = "sentence"
    PARENT_CHILD = "parent_child" # introduced conceptually in Episode 2
    SEMANTIC     = "semantic"     # full implementation in Episode 15


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """
    Returned by chunk_pages() — bundles the documents with their statistics.
    Makes it easy to compare strategies in the notebook without re-running.
    """
    documents: list[Document]
    strategy:  str
    stats:     dict[str, Any]

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"ChunkResult(strategy={self.strategy!r}, "
            f"chunks={s['total_chunks']}, "
            f"avg_chars={s['avg_chars']}, "
            f"range={s['min_chars']}–{s['max_chars']})"
        )


@dataclass
class ParentChildPair:
    """
    A parent chunk paired with its child sub-chunks.
    Used in Episode 2 to introduce the concept; full implementation in Episode 15.
    """
    parent:   Document
    children: list[Document]


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[RawPage],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    return_result: bool = False,
) -> list[Document] | ChunkResult:
    """
    Chunk a list of cleaned RawPages into LangChain Documents.

    Args:
        pages:         Cleaned RawPage objects from cleaner.py.
        strategy:      Chunking algorithm — see ChunkStrategy enum.
        chunk_size:    Character target per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Overlap between adjacent chunks. Defaults to settings.chunk_overlap.
        return_result: If True, return a ChunkResult with stats attached.
                       If False (default), return list[Document] for API compatibility.

    Returns:
        list[Document] or ChunkResult depending on return_result flag.

    Episode 2 notebook usage:
        # Compare strategies easily
        r1 = chunk_pages(pages, ChunkStrategy.FIXED,     return_result=True)
        r2 = chunk_pages(pages, ChunkStrategy.RECURSIVE, return_result=True)
        compare_strategies([r1, r2])
    """
    settings   = get_settings()
    chunk_size = chunk_size    or settings.chunk_size
    overlap    = chunk_overlap or settings.chunk_overlap

    if strategy == ChunkStrategy.SEMANTIC:
        docs = _semantic_chunk(pages, chunk_size)
    elif strategy == ChunkStrategy.PARENT_CHILD:
        docs = _parent_child_chunk(pages, chunk_size, overlap)
    else:
        splitter = _get_splitter(strategy, chunk_size, overlap)
        docs     = _apply_splitter(pages, splitter, strategy)

    stats = chunk_stats(docs)
    _log_stats(stats, strategy)

    if return_result:
        return ChunkResult(documents=docs, strategy=strategy.value, stats=stats)
    return docs


def chunk_stats(docs: list[Document]) -> dict[str, Any]:
    """
    Compute summary statistics for a list of chunks.

    Episode 2 usage:
        stats = chunk_stats(docs)
        print(f"Total: {stats['total_chunks']} | Avg: {stats['avg_chars']} chars")

    Returns dict with keys:
        total_chunks, avg_chars, min_chars, max_chars, total_chars,
        median_chars, std_chars, short_chunks, long_chunks
    """
    if not docs:
        return {}

    import statistics
    char_counts = [len(d.page_content) for d in docs]
    return {
        "total_chunks":  len(docs),
        "avg_chars":     round(sum(char_counts) / len(char_counts)),
        "median_chars":  round(statistics.median(char_counts)),
        "std_chars":     round(statistics.stdev(char_counts)) if len(char_counts) > 1 else 0,
        "min_chars":     min(char_counts),
        "max_chars":     max(char_counts),
        "total_chars":   sum(char_counts),
        # Quality indicators
        "short_chunks":  sum(1 for c in char_counts if c < 100),   # likely noise
        "long_chunks":   sum(1 for c in char_counts if c > 1200),  # may exceed context
    }


def compare_strategies(results: list[ChunkResult]) -> None:
    """
    Print a formatted comparison table of multiple ChunkResults.

    Episode 2 notebook cell — this is the money shot of the episode.

    Usage:
        results = [
            chunk_pages(pages, ChunkStrategy.FIXED,     return_result=True),
            chunk_pages(pages, ChunkStrategy.RECURSIVE, return_result=True),
            chunk_pages(pages, ChunkStrategy.SENTENCE,  return_result=True),
        ]
        compare_strategies(results)
    """
    header = f"{'Strategy':<14} {'Chunks':>7} {'Avg':>6} {'Median':>7} {'Std':>6} {'Min':>6} {'Max':>6} {'Short':>6} {'Long':>6}"
    print(header)
    print("─" * len(header))

    for r in results:
        s = r.stats
        print(
            f"{r.strategy:<14} "
            f"{s['total_chunks']:>7,} "
            f"{s['avg_chars']:>6} "
            f"{s['median_chars']:>7} "
            f"{s['std_chars']:>6} "
            f"{s['min_chars']:>6} "
            f"{s['max_chars']:>6} "
            f"{s['short_chunks']:>6} "
            f"{s['long_chunks']:>6}"
        )

    print()
    print("Short = chunks < 100 chars (likely noise)")
    print("Long  = chunks > 1200 chars (may truncate in LLM context)")


def build_parent_child_pairs(
    pages: list[RawPage],
    parent_size: int = 1600,
    child_size:  int = 400,
    overlap:     int = 50,
) -> list[ParentChildPair]:
    """
    Build parent-child chunk pairs for the parent-child retrieval pattern.

    Concept (Episode 2 introduction, full implementation Episode 15):
        - Child chunks are small — indexed in pgvector for precise retrieval
        - Parent chunks are large — passed to the LLM for fuller context
        - Retrieval happens at child level; generation uses the parent

    Why this matters:
        Small chunks = better retrieval precision (less noise around the answer)
        Large chunks = better generation quality (more context for the LLM)
        The parent-child pattern gets BOTH.

    Args:
        pages:       Cleaned RawPages to chunk.
        parent_size: Target chars for parent chunks.
        child_size:  Target chars for child chunks (subset of parent).
        overlap:     Overlap between child chunks within a parent.

    Returns:
        List of ParentChildPair objects.

    Episode 2 talking point:
        "Every child knows its parent ID. When the retriever finds a child chunk,
         we look up its parent and pass THAT to the LLM. More context, less noise."
    """
    parent_splitter = _get_splitter(ChunkStrategy.RECURSIVE, parent_size, 0)
    child_splitter  = _get_splitter(ChunkStrategy.RECURSIVE, child_size, overlap)

    pairs: list[ParentChildPair] = []

    for page in pages:
        base_doc = Document(
            page_content=page.text,
            metadata=_build_metadata(page),
        )
        parent_docs = parent_splitter.split_documents([base_doc])

        for p_idx, parent_doc in enumerate(parent_docs):
            parent_id = f"{page.source_file}::p{page.page_number}::chunk{p_idx}"
            parent_doc.metadata["chunk_id"]       = parent_id
            parent_doc.metadata["chunk_type"]     = "parent"
            parent_doc.metadata["chunk_index"]    = p_idx
            parent_doc.metadata["chunk_strategy"] = "parent_child"

            # Split parent into children
            child_docs = child_splitter.split_documents([parent_doc])
            for c_idx, child in enumerate(child_docs):
                child.metadata["parent_id"]       = parent_id
                child.metadata["chunk_type"]      = "child"
                child.metadata["child_index"]     = c_idx
                child.metadata["char_count"]      = len(child.page_content)
                child.metadata["chunk_strategy"]  = "parent_child"

            pairs.append(ParentChildPair(parent=parent_doc, children=child_docs))

    total_children = sum(len(p.children) for p in pairs)
    logger.info(
        "Parent-child chunking: %d parents → %d children (avg %.1f children/parent)",
        len(pairs), total_children,
        total_children / len(pairs) if pairs else 0,
    )
    return pairs


# ── Splitter factories ─────────────────────────────────────────────────────────

def _get_splitter(
    strategy: ChunkStrategy,
    chunk_size: int,
    overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Return the appropriate splitter for a strategy."""
    if strategy == ChunkStrategy.FIXED:
        return _fixed_splitter(chunk_size, overlap)
    elif strategy == ChunkStrategy.SENTENCE:
        return _sentence_splitter(chunk_size, overlap)
    else:
        return _recursive_splitter(chunk_size, overlap)


def _fixed_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Split on character count only — no respect for boundaries.

    Baseline comparison. Shows why content-aware splitting matters.

    Episode 2 on-screen demo:
        Take page 42 of PR157 (Nigeria).
        Show how FIXED cuts mid-sentence: "...maternal mortal | ity rates declined..."
        Then show RECURSIVE keeping the sentence intact.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[""],        # force character-level split only
        length_function=len,
        add_start_index=True,
    )


def _recursive_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Split on paragraph → sentence → word boundaries in priority order.

    This is the workhorse splitter for the entire course.

    Separator hierarchy for DHS reports:
        \\n\\n  paragraph break (strongest signal — kept by cleaner)
        \\n    line break
        . + space  sentence end
        ! + space  exclamation sentence
        ? + space  question sentence
        ;     semi-colon (common in statistical prose: "rate was 23%; up from 18%")
        ,     comma (last word-level resort)
        (space)  word boundary
        (empty)  character-level fallback

    Episode 2 talking point:
        "The double newline is the most important separator. Every time you see
         \\n\\n in the cleaned text, that's a paragraph boundary — the strongest
         signal we have that two ideas are different enough to split on."
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",   # paragraph — highest priority
            "\n",     # line break
            ". ",     # sentence (with trailing space to avoid decimals)
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",       # character fallback
        ],
        length_function=len,
        add_start_index=True,
    )


def _sentence_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Sentence-first splitter — prioritises sentence boundaries above all.

    Better for the Ethiopia mini-report (FR363) which is narrative-heavy
    with very long paragraphs where \\n\\n rarely appears.

    The key difference from RECURSIVE: sentence boundaries come BEFORE
    paragraph breaks in the separator hierarchy.

    Episode 2 talking point:
        "The Ethiopia mini-report reads like a journal article — long paragraphs,
         almost no double newlines. RECURSIVE gets confused; SENTENCE handles it."
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            ". ",     # sentence — now highest priority
            "! ",
            "? ",
            "\n\n",   # paragraph
            "\n",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
        add_start_index=True,
    )


# ── Core processing ───────────────────────────────────────────────────────────

def _apply_splitter(
    pages: list[RawPage],
    splitter: RecursiveCharacterTextSplitter,
    strategy: ChunkStrategy,
) -> list[Document]:
    """Apply a splitter to all pages, attaching position metadata to each chunk."""
    all_docs: list[Document] = []

    for page in pages:
        base_doc = Document(
            page_content=page.text,
            metadata=_build_metadata(page),
        )
        chunks = splitter.split_documents([base_doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]    = i
            chunk.metadata["chunk_count"]    = len(chunks)
            chunk.metadata["char_count"]     = len(chunk.page_content)
            chunk.metadata["chunk_strategy"] = strategy.value
            # Stable content-based ID — used for idempotent pgvector upserts
            import hashlib
            chunk.metadata["chunk_id"] = hashlib.sha256(
                chunk.page_content.encode()
            ).hexdigest()[:16]

        all_docs.extend(chunks)

    return all_docs


def _parent_child_chunk(
    pages: list[RawPage],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    """
    Build child chunks from parent-child pairs.
    Returns only the child Documents (what gets indexed in pgvector).
    Parent documents are accessible via the parent_id metadata field.
    """
    pairs = build_parent_child_pairs(
        pages,
        parent_size=chunk_size * 2,
        child_size=chunk_size,
        overlap=overlap,
    )
    return [child for pair in pairs for child in pair.children]


def _semantic_chunk(pages: list[RawPage], chunk_size: int) -> list[Document]:
    """
    Semantic chunking — splits on embedding similarity shifts.
    Full implementation introduced in Episode 15.
    Gated behind a lazy import to avoid breaking Episodes 1–14.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "Semantic chunking requires langchain-experimental.\n"
            "Run: uv add langchain-experimental\n"
            "This strategy is fully introduced in Episode 15."
        )

    settings   = get_settings()
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
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

    return all_docs


# ── Metadata builder ──────────────────────────────────────────────────────────

def _build_metadata(page: RawPage) -> dict:
    """
    Build the metadata dict attached to every chunk.

    Every field here has a purpose — this metadata is stored alongside the
    embedding in pgvector and enables filtered retrieval in Episode 7.

    Fields:
        source        Full path to the source PDF (for provenance)
        file_name     Basename of PDF (for display)
        page_number   Page within the PDF (for citation: "Page 42 of 350")
        total_pages   Total pages in the source PDF
        country       For filtering: WHERE country = 'Nigeria'
        year          For temporal filtering: WHERE year >= '2018'
        report_type   'dhs' | 'status_of_women' | 'other'
        report_title  Full human-readable title (shown in citations)

    Episode 2 talking point:
        "The metadata is what makes this different from a generic RAG system.
         Any LLM can retrieve text. Only a well-designed system can answer:
         'Show me Kenya data from the last 5 years' — because only a well-designed
         system attached country and year to every single chunk at index time."
    """
    return {
        "source":       page.source_file,
        "file_name":    page.file_name,
        "page_number":  page.page_number,
        "total_pages":  page.total_pages,
        "country":      page.country,
        "year":         page.year,
        "report_type":  page.report_type,
        "report_title": page.report_title,
    }


# ── Logging ───────────────────────────────────────────────────────────────────

def _log_stats(stats: dict, strategy: ChunkStrategy) -> None:
    if not stats:
        return
    logger.info(
        "[%s] %d chunks | avg %d | median %d | range %d–%d | short=%d long=%d",
        strategy.value,
        stats["total_chunks"],
        stats["avg_chars"],
        stats["median_chars"],
        stats["min_chars"],
        stats["max_chars"],
        stats["short_chunks"],
        stats["long_chunks"],
    )
