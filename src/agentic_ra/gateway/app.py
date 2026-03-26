"""FastAPI application factory for the Agentic RAG gateway."""

from __future__ import annotations

from fastapi import FastAPI

from agentic_ra.gateway.routes.health import router as health_router
from agentic_ra.gateway.routes.query import router as query_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routes mounted.
    """
    app = FastAPI(
        title="Agentic RAG",
        description="Near-perfect agentic RAG system API",
        version="0.1.0",
    )

    app.include_router(health_router)
    app.include_router(query_router)

    return app
