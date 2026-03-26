"""Query endpoint — main RAG query API."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agentic_ra.pipelines.factory import RAGPipeline

router = APIRouter(prefix="/api/v1", tags=["query"])


class QueryRequest(BaseModel):
    """Incoming RAG query request."""

    query: str = Field(
        ..., min_length=1, max_length=10000, description="User query text"
    )
    session_id: uuid.UUID | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    user_id: str = Field(default="anonymous", description="User identifier")
    strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy hint (semantic, keyword, hybrid, graph, graph_hybrid)",
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max evidence documents")


class CitationResponse(BaseModel):
    """Citation in the response."""

    index: int
    content_snippet: str
    source: str
    score: float


class QueryResponse(BaseModel):
    """RAG query response with citations."""

    answer: str
    citations: list[CitationResponse]
    query: str
    confidence: float
    is_grounded: bool
    strategy: str
    session_id: uuid.UUID


def _get_pipeline(request: Request) -> RAGPipeline:
    """Extract the RAG pipeline from app state."""
    return request.app.state.pipeline  # type: ignore[no-any-return]


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    """Process a RAG query through the full orchestrator pipeline."""
    pipeline = _get_pipeline(request)
    session_id = body.session_id or uuid.uuid4()

    result = await pipeline.orchestrator.process_query(
        query=body.query,
        session_id=session_id,
        user_id=body.user_id,
        strategy=body.strategy,
        limit=body.limit,
    )

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                index=c.index,
                content_snippet=c.content_snippet,
                source=c.source,
                score=c.score,
            )
            for c in result.citations
        ],
        query=result.query,
        confidence=result.confidence,
        is_grounded=result.is_grounded,
        strategy=result.strategy,
        session_id=session_id,
    )
