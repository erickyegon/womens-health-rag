"""
Centralised configuration using Pydantic Settings.

All values are read from environment variables (or .env file).
Never hardcode secrets or tuning parameters anywhere else in the codebase —
always import from here.

Episode 1: This file is introduced in full so viewers understand the
configuration pattern from day one.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model used for generation",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model — 1536-dim output",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection string with pgvector extension",
    )

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable LangSmith tracing",
    )
    langchain_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key (optional in dev)",
    )
    langchain_project: str = Field(
        default="womens-health-rag",
        description="LangSmith project name",
    )

    # ── Reranking (Phase 2) ───────────────────────────────────────────────────
    cohere_api_key: SecretStr | None = Field(
        default=None,
        description="Cohere API key for reranking — required from Episode 10",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_key: SecretStr = Field(
        default=SecretStr("dev-secret-change-in-prod"),
        description="Bearer token for API authentication",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ── RAG tuning ────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Target character count per chunk",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=500,
        description="Character overlap between adjacent chunks",
    )
    retrieval_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of chunks retrieved before reranking",
    )
    rerank_top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks kept after reranking",
    )
    embedding_batch_size: int = Field(
        default=50,
        ge=1,
        le=2048,
        description="Documents per embedding API call",
    )

    # ── Vector index ──────────────────────────────────────────────────────────
    vector_table_name: str = Field(
        default="document_chunks",
        description="pgvector table name",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Must match the embedding model output dimensions",
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Overlap must be smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self

    @property
    def database_url_str(self) -> str:
        """Return DATABASE_URL as a plain string for SQLAlchemy."""
        return str(self.database_url)

    @property
    def openai_api_key_str(self) -> str:
        """Unwrap SecretStr for SDK usage."""
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Use this everywhere instead of instantiating Settings() directly.
    The lru_cache means .env is only read once, regardless of how many
    modules import get_settings().

    Usage:
        from rag.config.settings import get_settings
        settings = get_settings()
    """
    return Settings()  # type: ignore[call-arg]
