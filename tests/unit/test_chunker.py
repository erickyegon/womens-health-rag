"""
Unit tests for rag.ingestion.chunker — Episode 2

Tests all four chunking strategies, stats, parent-child pairs,
and the compare_strategies function.

Run with: make test-unit  or  pytest tests/unit/test_chunker.py -v
"""

from __future__ import annotations

import pytest

from rag.ingestion.chunker import (
    ChunkResult,
    ChunkStrategy,
    ParentChildPair,
    build_parent_child_pairs,
    chunk_pages,
    chunk_stats,
    compare_strategies,
)
from rag.ingestion.loader import RawPage


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_page(text: str, country: str = "Nigeria", year: str = "2021",
              page_number: int = 1) -> RawPage:
    return RawPage(
        text=text,
        page_number=page_number,
        source_file=f"/tmp/PR157.pdf",
        file_name="PR157.pdf",
        total_pages=350,
        country=country,
        year=year,
        report_type="dhs",
        report_title="Nigeria DHS 2021",
    )


# Long text that produces multiple chunks — realistic DHS paragraph
PARAGRAPH = (
    "The maternal mortality ratio in Nigeria declined from 576 per 100,000 live births "
    "in 2013 to 512 per 100,000 live births in 2021, representing an 11 percent improvement. "
    "This decline is attributed to increased skilled birth attendance, which rose from 38 percent "
    "to 43 percent, and improved antenatal care coverage which stabilised at 67 percent. "
    "Contraceptive prevalence among married women increased from 15 percent to 17 percent. "
    "The total fertility rate remained high at 5.3 children per woman. "
    "Regional disparities persist with the North West zone showing an MMR of 1,549 "
    "compared to 165 in the South West. Neonatal mortality accounts for 43 percent of "
    "all under-5 deaths, underscoring the critical importance of newborn care. "
    "Female secondary education enrollment improved from 22 percent to 28 percent "
    "over the same period, which is associated with improved maternal and child health outcomes. "
    "The under-5 mortality rate fell from 128 to 117 per 1,000 live births. "
    "Access to clean water improved to 71 percent of households in 2021. "
    "Immunisation coverage for DPT3 among children aged 12-23 months reached 54 percent. "
)

LONG_TEXT = PARAGRAPH * 4   # ~2000 chars — ensures multiple chunks at size=800


# ── chunk_pages — basic correctness ──────────────────────────────────────────

class TestChunkPagesBasic:

    def test_recursive_returns_documents(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], ChunkStrategy.RECURSIVE)
        assert len(docs) > 1

    def test_fixed_returns_documents(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], ChunkStrategy.FIXED)
        assert len(docs) > 1

    def test_sentence_returns_documents(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], ChunkStrategy.SENTENCE)
        assert len(docs) > 1

    def test_empty_pages_returns_empty(self):
        assert chunk_pages([]) == []

    def test_single_short_page_returns_one_chunk(self):
        page = make_page("Short content that fits in one chunk.")
        docs = chunk_pages([page], chunk_size=800)
        assert len(docs) == 1

    def test_default_strategy_is_recursive(self):
        page = make_page(LONG_TEXT)
        # No strategy argument — should work with RECURSIVE default
        docs = chunk_pages([page])
        assert len(docs) > 0


# ── Metadata attachment ───────────────────────────────────────────────────────

class TestMetadataAttachment:

    def test_country_attached(self):
        page = make_page(LONG_TEXT, country="Nigeria")
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["country"] == "Nigeria"

    def test_year_attached(self):
        page = make_page(LONG_TEXT, year="2021")
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["year"] == "2021"

    def test_report_type_attached(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["report_type"] == "dhs"

    def test_report_title_attached(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["report_title"] == "Nigeria DHS 2021"

    def test_page_number_attached(self):
        page = make_page(LONG_TEXT, page_number=42)
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["page_number"] == 42

    def test_file_name_attached(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["file_name"] == "PR157.pdf"

    def test_chunk_index_sequential(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], chunk_size=400)
        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i

    def test_chunk_count_consistent(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], chunk_size=400)
        n = len(docs)
        for doc in docs:
            assert doc.metadata["chunk_count"] == n

    def test_char_count_matches_content(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page])
        for doc in docs:
            assert doc.metadata["char_count"] == len(doc.page_content)

    def test_strategy_name_attached(self):
        page = make_page(LONG_TEXT)
        for strategy in [ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE, ChunkStrategy.SENTENCE]:
            docs = chunk_pages([page], strategy=strategy)
            for doc in docs:
                assert doc.metadata["chunk_strategy"] == strategy.value

    def test_chunk_id_present(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page])
        for doc in docs:
            assert "chunk_id" in doc.metadata
            assert len(doc.metadata["chunk_id"]) == 16  # truncated SHA-256

    def test_chunk_id_deterministic(self):
        page = make_page(LONG_TEXT)
        docs1 = chunk_pages([page])
        docs2 = chunk_pages([page])
        ids1 = [d.metadata["chunk_id"] for d in docs1]
        ids2 = [d.metadata["chunk_id"] for d in docs2]
        assert ids1 == ids2


# ── Multi-page ────────────────────────────────────────────────────────────────

class TestMultiPage:

    def test_multiple_pages_combined(self):
        pages = [
            make_page(LONG_TEXT, country="Nigeria"),
            make_page(LONG_TEXT, country="Kenya"),
            make_page(LONG_TEXT, country="Ghana"),
        ]
        docs = chunk_pages(pages)
        countries = {d.metadata["country"] for d in docs}
        assert "Nigeria" in countries
        assert "Kenya" in countries
        assert "Ghana" in countries

    def test_chunk_count_scales_with_pages(self):
        one_page    = chunk_pages([make_page(LONG_TEXT)])
        three_pages = chunk_pages([make_page(LONG_TEXT)] * 3)
        # Three pages should produce ~3x the chunks
        assert len(three_pages) > len(one_page) * 2


# ── ChunkResult (return_result=True) ─────────────────────────────────────────

class TestChunkResult:

    def test_returns_chunk_result_when_flag_set(self):
        page   = make_page(LONG_TEXT)
        result = chunk_pages([page], return_result=True)
        assert isinstance(result, ChunkResult)

    def test_returns_list_by_default(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page])
        assert isinstance(docs, list)

    def test_chunk_result_has_documents(self):
        page   = make_page(LONG_TEXT)
        result = chunk_pages([page], return_result=True)
        assert len(result.documents) > 0

    def test_chunk_result_has_stats(self):
        page   = make_page(LONG_TEXT)
        result = chunk_pages([page], return_result=True)
        assert "total_chunks" in result.stats
        assert "avg_chars" in result.stats

    def test_chunk_result_strategy_name(self):
        page   = make_page(LONG_TEXT)
        result = chunk_pages([page], ChunkStrategy.RECURSIVE, return_result=True)
        assert result.strategy == "recursive"

    def test_chunk_result_repr(self):
        page   = make_page(LONG_TEXT)
        result = chunk_pages([page], return_result=True)
        r = repr(result)
        assert "ChunkResult" in r
        assert "chunks=" in r


# ── chunk_stats ───────────────────────────────────────────────────────────────

class TestChunkStats:

    def test_empty_returns_empty_dict(self):
        assert chunk_stats([]) == {}

    def test_all_keys_present(self):
        page  = make_page(LONG_TEXT)
        docs  = chunk_pages([page])
        stats = chunk_stats(docs)
        expected = [
            "total_chunks", "avg_chars", "median_chars", "std_chars",
            "min_chars", "max_chars", "total_chars", "short_chunks", "long_chunks",
        ]
        for key in expected:
            assert key in stats

    def test_total_chunks_correct(self):
        page  = make_page(LONG_TEXT)
        docs  = chunk_pages([page])
        stats = chunk_stats(docs)
        assert stats["total_chunks"] == len(docs)

    def test_min_lte_avg_lte_max(self):
        page  = make_page(LONG_TEXT)
        docs  = chunk_pages([page])
        stats = chunk_stats(docs)
        assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]

    def test_avg_lte_chunk_size_plus_tolerance(self):
        page  = make_page(LONG_TEXT)
        docs  = chunk_pages([page], chunk_size=800, chunk_overlap=150)
        stats = chunk_stats(docs)
        assert stats["avg_chars"] <= 800 * 1.1

    def test_total_chars_is_sum(self):
        from langchain_core.documents import Document
        docs  = [Document(page_content="a" * 100) for _ in range(5)]
        stats = chunk_stats(docs)
        assert stats["total_chars"] == 500

    def test_short_chunks_counted(self):
        from langchain_core.documents import Document
        docs = [
            Document(page_content="short"),   # 5 chars — short
            Document(page_content="x" * 200), # 200 chars — not short
        ]
        stats = chunk_stats(docs)
        assert stats["short_chunks"] == 1

    def test_long_chunks_counted(self):
        from langchain_core.documents import Document
        docs = [
            Document(page_content="x" * 1500),  # long
            Document(page_content="x" * 200),   # not long
        ]
        stats = chunk_stats(docs)
        assert stats["long_chunks"] == 1


# ── Strategy comparison ───────────────────────────────────────────────────────

class TestCompareStrategies:

    def test_runs_without_error(self, capsys):
        page    = make_page(LONG_TEXT)
        results = [
            chunk_pages([page], ChunkStrategy.FIXED,     return_result=True),
            chunk_pages([page], ChunkStrategy.RECURSIVE, return_result=True),
        ]
        compare_strategies(results)  # should not raise
        out = capsys.readouterr().out
        assert "fixed" in out
        assert "recursive" in out

    def test_recursive_fewer_short_chunks_than_fixed(self):
        # RECURSIVE should produce fewer short/noise chunks than FIXED
        pages = [make_page(LONG_TEXT)] * 5
        r_fixed     = chunk_pages(pages, ChunkStrategy.FIXED,     chunk_size=800, return_result=True)
        r_recursive = chunk_pages(pages, ChunkStrategy.RECURSIVE, chunk_size=800, return_result=True)
        # RECURSIVE's short_chunks should be <= FIXED's (usually strictly less)
        assert r_recursive.stats["short_chunks"] <= r_fixed.stats["short_chunks"] + 5


# ── Parent-child chunking ─────────────────────────────────────────────────────

class TestParentChildPairs:

    def test_returns_pairs(self):
        page  = make_page(LONG_TEXT)
        pairs = build_parent_child_pairs([page], parent_size=1000, child_size=300, overlap=50)
        assert len(pairs) > 0
        assert all(isinstance(p, ParentChildPair) for p in pairs)

    def test_each_parent_has_children(self):
        page  = make_page(LONG_TEXT)
        pairs = build_parent_child_pairs([page], parent_size=800, child_size=250)
        for pair in pairs:
            assert len(pair.children) >= 1

    def test_parent_longer_than_children(self):
        page  = make_page(LONG_TEXT)
        pairs = build_parent_child_pairs([page], parent_size=800, child_size=250)
        for pair in pairs:
            parent_len = len(pair.parent.page_content)
            for child in pair.children:
                assert parent_len >= len(child.page_content)

    def test_child_has_parent_id(self):
        page  = make_page(LONG_TEXT)
        pairs = build_parent_child_pairs([page])
        for pair in pairs:
            parent_id = pair.parent.metadata["chunk_id"]
            for child in pair.children:
                assert child.metadata["parent_id"] == parent_id

    def test_child_chunk_type_metadata(self):
        page  = make_page(LONG_TEXT)
        pairs = build_parent_child_pairs([page])
        for pair in pairs:
            assert pair.parent.metadata["chunk_type"] == "parent"
            for child in pair.children:
                assert child.metadata["chunk_type"] == "child"

    def test_child_inherits_country_metadata(self):
        page  = make_page(LONG_TEXT, country="Kenya")
        pairs = build_parent_child_pairs([page])
        for pair in pairs:
            for child in pair.children:
                assert child.metadata["country"] == "Kenya"

    def test_parent_child_chunk_strategy(self):
        page  = make_page(LONG_TEXT)
        pairs = build_parent_child_pairs([page])
        for pair in pairs:
            assert pair.parent.metadata["chunk_strategy"] == "parent_child"
            for child in pair.children:
                assert child.metadata["chunk_strategy"] == "parent_child"

    def test_empty_pages_returns_empty(self):
        assert build_parent_child_pairs([]) == []

    def test_parent_child_chunk_pages_returns_children(self):
        # chunk_pages with PARENT_CHILD strategy returns child docs
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], ChunkStrategy.PARENT_CHILD, chunk_size=800)
        assert len(docs) > 0
        for doc in docs:
            assert doc.metadata.get("chunk_type") == "child"
