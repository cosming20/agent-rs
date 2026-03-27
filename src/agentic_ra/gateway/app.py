"""FastAPI application factory for the Agentic RAG gateway."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from agentic_ra.auth.opa import OPAClient
from agentic_ra.core.errors import AgenticRAError
from agentic_ra.core.config import get_settings
from agentic_ra.core.logging import setup_logging
from agentic_ra.gateway.middleware.auth import OPAAuthMiddleware
from agentic_ra.gateway.routes.health import router as health_router
from agentic_ra.gateway.routes.ingest import router as ingest_router
from agentic_ra.gateway.routes.query import router as query_router
from agentic_ra.observability.cost_tracker import setup_cost_tracking
from agentic_ra.observability.metrics import setup_metrics
from agentic_ra.observability.tracing import instrument_fastapi, setup_tracing
from agentic_ra.pipelines.factory import build_pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: build pipeline on startup, tear down on shutdown."""
    settings = get_settings()

    setup_logging(level=settings.log_level)

    tracer_provider = setup_tracing(settings)
    meter_provider = setup_metrics(settings)

    instrument_fastapi(app)

    # Initialize LLM cost tracking (Postgres + structured logs)
    setup_cost_tracking(pg_dsn=settings.postgres_dsn_async.get_secret_value())

    pipeline = build_pipeline(settings)
    app.state.pipeline = pipeline

    logger.info("Warming up embedding models...")
    pipeline.embedding_pipeline.warm_up()
    logger.info("Embedding models ready")

    yield

    # Shutdown — close all components, tolerating individual failures
    logger.info("Shutting down RAG pipeline...")
    for close_fn in [pipeline.close]:
        try:
            await close_fn()
        except Exception:
            logger.exception("Error during shutdown")

    # Close OPA httpx client
    if hasattr(app.state, "opa_client"):
        try:
            await app.state.opa_client.close()
        except Exception:
            logger.exception("Error closing OPA client")

    meter_provider.shutdown()
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

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingest_router)

    # OPA middleware — client stored on app.state for lifecycle management
    opa_client = OPAClient(base_url=settings.opa_url)
    app.state.opa_client = opa_client
    app.add_middleware(
        OPAAuthMiddleware,
        opa_client=opa_client,
        enabled=settings.opa_auth_enabled,
    )

    # C4: Global exception handler for domain exceptions
    @app.exception_handler(AgenticRAError)
    async def agentic_ra_error_handler(
        request: Request, exc: AgenticRAError
    ) -> JSONResponse:
        logger.error("Domain error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    return app
