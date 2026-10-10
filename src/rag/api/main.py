"""
FastAPI Backend — Episode 25
==============================
Production API with streaming SSE, auth, rate limiting, health checks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routes import health, query
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting Women's Health RAG API — model: %s", settings.openai_model)
    yield
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Women's Health RAG API",
        description="Production RAG system for global women's health data",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://ui:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        logger.info(
            "%s %s → %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed
        )
        return response

    # Routes
    app.include_router(health.router, tags=["health"])
    app.include_router(query.router, tags=["query"])

    return app


app = create_app()
