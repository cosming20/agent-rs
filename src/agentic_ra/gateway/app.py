"""FastAPI application factory for the Agentic RAG gateway."""


import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from agentic_ra.auth.opa import OPAClient
from agentic_ra.core.errors import AgenticRAError, AuthError, OPAError, SSRFError, WebBudgetExhaustedError
from agentic_ra.core.config import get_settings
from agentic_ra.core.logging import setup_logging
from agentic_ra.gateway.middleware.auth import OPAAuthMiddleware
from agentic_ra.gateway.routes.health import router as health_router
from agentic_ra.gateway.routes.ingest import router as ingest_router
from agentic_ra.gateway.routes.query import router as query_router
from agentic_ra.gateway.routes.upload import router as upload_router
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
    try:
        await pipeline.close()
    except Exception:
        logger.exception("Error during pipeline shutdown")

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

    # Suppress noisy health check access logs
    logging.getLogger("uvicorn.access").addFilter(
        type("_HealthFilter", (), {"filter": staticmethod(lambda r: "/health" not in r.getMessage())})()
    )

    app = FastAPI(
        title="Agentic RAG",
        description="Near-perfect agentic RAG system API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(upload_router)

    # CORS middleware — needed when frontend runs on a different origin (dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OPA middleware — client stored on app.state for lifecycle management
    opa_client = OPAClient(base_url=settings.opa_url)
    app.state.opa_client = opa_client
    app.add_middleware(
        OPAAuthMiddleware,
        opa_client=opa_client,
        enabled=settings.opa_auth_enabled,
    )

    # HTTP status code map for domain errors — order matters: most specific first
    _ERROR_STATUS_MAP: dict[type, int] = {
        SSRFError: 400,
        AuthError: 403,
        OPAError: 403,
        WebBudgetExhaustedError: 429,
    }

    # Global exception handler for domain exceptions
    @app.exception_handler(AgenticRAError)
    async def agentic_ra_error_handler(
        request: Request, exc: AgenticRAError
    ) -> JSONResponse:
        logger.error("Domain error: %s", exc, exc_info=True)
        status_code = next(
            (code for err_type, code in _ERROR_STATUS_MAP.items() if isinstance(exc, err_type)),
            500,
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    # Convenience redirect: / → /ui/
    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    # Serve frontend static files at /ui/
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app
