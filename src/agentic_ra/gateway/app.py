"""FastAPI application factory for the Agentic RAG gateway."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_ra.core.config import get_settings
from agentic_ra.gateway.routes.health import router as health_router
from agentic_ra.gateway.routes.ingest import router as ingest_router
from agentic_ra.gateway.routes.query import router as query_router
from agentic_ra.observability.tracing import instrument_fastapi, setup_tracing
from agentic_ra.pipelines.factory import build_pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: build pipeline on startup, tear down on shutdown."""
    settings = get_settings()

    # Setup tracing
    tracer_provider = setup_tracing(settings)

    # Build the RAG pipeline
    pipeline = build_pipeline(settings)
    app.state.pipeline = pipeline

    # Warm up embedding models (sync, runs in main thread at startup)
    logger.info("Warming up embedding models...")
    pipeline.embedding_pipeline.warm_up()
    pipeline.query_embedder.warm_up()
    logger.info("Embedding models ready")

    # Ensure collections and indices exist
    await pipeline.qdrant.ensure_collection(
        "documents",
        vector_size=settings.embedding_dimension,
    )
    await pipeline.opensearch.ensure_index("documents")
    logger.info("Qdrant collection and OpenSearch index ready")

    yield

    # Shutdown
    logger.info("Shutting down RAG pipeline...")
    await pipeline.close()
    tracer_provider.shutdown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routes mounted.
    """
    app = FastAPI(
        title="Agentic RAG",
        description="Near-perfect agentic RAG system API",
        version="0.1.0",
        lifespan=lifespan,
    )

    instrument_fastapi(app)

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingest_router)

    return app
