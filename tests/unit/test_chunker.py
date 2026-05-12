"""
Unit tests for rag.ingestion.chunker

Tests chunking strategies and metadata attachment without any LLM calls.
"""

import pytest

from rag.ingestion.chunker import ChunkStrategy, chunk_pages, chunk_stats
from rag.ingestion.loader import RawPage


def make_page(text: str, country: str = "Kenya", year: str = "2022") -> RawPage:
    return RawPage(
        text=text,
        page_number=1,
        source_file="/tmp/kenya_dhs.pdf",
        file_name="kenya_dhs.pdf",
        total_pages=5,
        country=country,
        year=year,
        report_type="dhs",
        report_title="Kenya DHS 2022",
    )


LONG_TEXT = (
    "The maternal mortality ratio (MMR) in Kenya declined significantly between 2014 and 2022. "
    "According to the 2022 DHS report, the MMR stood at 355 deaths per 100,000 live births, "
    "representing a 23% reduction from the 2014 figure of 462. This improvement is attributed "
    "to increased skilled birth attendance, expanded access to antenatal care, and targeted "
    "interventions in high-burden counties. Contraceptive prevalence also rose during this period, "
    "reaching 61% among married women of reproductive age. Female secondary education enrollment "
    "improved from 48% to 67%, which correlates strongly with improved maternal health outcomes "
    "across sub-Saharan Africa. The under-5 mortality rate fell from 52 to 41 per 1,000 live births. "
    "Regional disparities remain significant, with the North Eastern region showing the highest MMR "
    "at 724 per 100,000, compared to the national average. Neonatal mortality accounts for 43% of "
    "all under-5 deaths, highlighting the need for focused newborn care interventions."
) * 3  # Repeat to create enough text for multiple chunks


class TestChunkPages:
    def test_recursive_produces_documents(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE, chunk_size=400, chunk_overlap=50)
        assert len(docs) > 1

    def test_fixed_produces_documents(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.FIXED, chunk_size=400, chunk_overlap=50)
        assert len(docs) > 1

    def test_metadata_attached_to_all_chunks(self):
        page = make_page(LONG_TEXT, country="Kenya", year="2022")
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE)
        for doc in docs:
            assert doc.metadata["country"] == "Kenya"
            assert doc.metadata["year"] == "2022"
            assert doc.metadata["report_type"] == "dhs"
            assert doc.metadata["file_name"] == "kenya_dhs.pdf"

    def test_chunk_index_sequential(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE, chunk_size=400)
        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i

    def test_chunk_count_consistent(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE, chunk_size=400)
        expected_count = len(docs)
        for doc in docs:
            assert doc.metadata["chunk_count"] == expected_count

    def test_char_count_attached(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE)
        for doc in docs:
            assert "char_count" in doc.metadata
            assert doc.metadata["char_count"] == len(doc.page_content)

    def test_strategy_name_attached(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE)
        for doc in docs:
            assert doc.metadata["chunk_strategy"] == "recursive"

    def test_empty_pages_returns_empty(self):
        assert chunk_pages([]) == []

    def test_multiple_pages_combined(self):
        pages = [make_page(LONG_TEXT, country=f"Country{i}") for i in range(3)]
        docs = chunk_pages(pages, strategy=ChunkStrategy.RECURSIVE)
        countries = {doc.metadata["country"] for doc in docs}
        assert len(countries) == 3  # all countries represented


class TestChunkStats:
    def test_returns_expected_keys(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE)
        stats = chunk_stats(docs)
        for key in ["total_chunks", "avg_chars", "min_chars", "max_chars", "total_chars"]:
            assert key in stats

    def test_empty_returns_empty_dict(self):
        assert chunk_stats([]) == {}

    def test_avg_within_chunk_size(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE, chunk_size=400, chunk_overlap=50)
        stats = chunk_stats(docs)
        # Average should be less than or equal to chunk_size
        assert stats["avg_chars"] <= 400 * 1.1  # allow 10% tolerance

    def test_min_max_logical(self):
        page = make_page(LONG_TEXT)
        docs = chunk_pages([page], strategy=ChunkStrategy.RECURSIVE)
        stats = chunk_stats(docs)
        assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]
