"""
Shared pytest fixtures available to all test modules.

Fixtures defined here are auto-discovered by pytest.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """
    Set minimal required environment variables for all tests.
    Prevents tests from accidentally reading a real .env file.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-not-real")
    monkeypatch.setenv("DATABASE_URL",   "postgresql://rag:rag@localhost:5432/rag_test")
    monkeypatch.setenv("CHUNK_SIZE",     "800")
    monkeypatch.setenv("CHUNK_OVERLAP",  "150")
    monkeypatch.setenv("LOG_LEVEL",      "WARNING")  # suppress noise in tests


@pytest.fixture
def sample_raw_page():
    """A realistic RawPage fixture for testing."""
    from rag.ingestion.loader import RawPage
    return RawPage(
        text=(
            "The maternal mortality ratio in Nigeria declined from 576 per 100,000 live births "
            "in 2013 to 512 per 100,000 in 2021, according to the Nigeria DHS report. "
            "This represents an 11% improvement. Skilled birth attendance increased from "
            "38% to 43% over the same period. Antenatal care coverage remained at 67%."
        ),
        page_number=42,
        source_file="/tmp/nigeria_dhs_2021.pdf",
        file_name="nigeria_dhs_2021.pdf",
        total_pages=350,
        country="Nigeria",
        year="2021",
        report_type="dhs",
        report_title="Nigeria Demographic and Health Survey 2021",
    )


@pytest.fixture
def sample_documents(sample_raw_page):
    """Pre-chunked Document fixtures."""
    from rag.ingestion.chunker import ChunkStrategy, chunk_pages
    return chunk_pages([sample_raw_page], strategy=ChunkStrategy.RECURSIVE, chunk_size=200)
