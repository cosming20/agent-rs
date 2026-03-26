"""FastAPI application factory for the Agentic RAG gateway."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_ra.auth.opa import OPAClient
from agentic_ra.core.config import get_settings
from agentic_ra.core.logging import setup_logging
from agentic_ra.gateway.middleware.auth import OPAAuthMiddleware
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

    setup_logging(level=settings.log_level)

    tracer_provider = setup_tracing(settings)

    pipeline = build_pipeline(settings)
    app.state.pipeline = pipeline

    opa_client = OPAClient(base_url=settings.opa_url)
    app.state.opa_client = opa_client

    logger.info("Warming up embedding models...")
    pipeline.embedding_pipeline.warm_up()
    logger.info("Embedding models ready")

    yield

    # Shutdown — close all components, tolerating individual failures
    logger.info("Shutting down RAG pipeline...")
    for close_fn in [opa_client.close, pipeline.close]:
        try:
            await close_fn()
        except Exception:
            logger.exception("Error during shutdown")
    tracer_provider.shutdown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

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

    # OPA middleware uses settings — single client, properly closed in lifespan
    app.add_middleware(
        OPAAuthMiddleware,
        opa_client=OPAClient(base_url=settings.opa_url),
        enabled=settings.opa_auth_enabled,
    )

    return app
