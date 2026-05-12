"""
Unit tests for rag.config.settings

Verifies that settings validation catches bad configurations early.
"""

import pytest
from pydantic import ValidationError


class TestSettingsValidation:
    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        """chunk_overlap >= chunk_size should raise a validation error."""
        import os
        # Override env vars for this test
        os.environ["OPENAI_API_KEY"]  = "sk-test"
        os.environ["DATABASE_URL"]    = "postgresql://user:pass@localhost:5432/db"
        os.environ["CHUNK_SIZE"]      = "400"
        os.environ["CHUNK_OVERLAP"]   = "400"   # equal — should fail

        from rag.config.settings import Settings
        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings()  # type: ignore[call-arg]

    def test_valid_settings_instantiate(self, monkeypatch):
        """Valid configuration should instantiate without errors."""
        monkeypatch.setenv("OPENAI_API_KEY",  "sk-test-key-1234")
        monkeypatch.setenv("DATABASE_URL",    "postgresql://rag:rag@localhost:5432/rag")
        monkeypatch.setenv("CHUNK_SIZE",      "800")
        monkeypatch.setenv("CHUNK_OVERLAP",   "150")

        from rag.config.settings import Settings
        s = Settings()  # type: ignore[call-arg]
        assert s.chunk_size    == 800
        assert s.chunk_overlap == 150

    def test_database_url_str_property(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("DATABASE_URL",   "postgresql://rag:rag@localhost:5432/rag")

        from rag.config.settings import Settings
        s = Settings()  # type: ignore[call-arg]
        assert "postgresql" in s.database_url_str
