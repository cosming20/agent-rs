"""Query endpoint — main RAG query API."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from agentic_ra.pipelines.factory import RAGPipeline
from agentic_ra.retrieval.router import RetrievalStrategy

router = APIRouter(prefix="/api/v1", tags=["query"])

VALID_ROLES = frozenset({"user", "admin"})


class QueryRequest(BaseModel):
    """Incoming RAG query request."""

    query: str = Field(
        ..., min_length=1, max_length=10000, description="User query text"
    )
    session_id: uuid.UUID | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy (semantic, keyword, hybrid, graph, graph_hybrid)",
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
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    x_user_id: str = Header(default="anonymous", alias="X-User-Id"),
    x_user_role: str = Header(default="user", alias="X-User-Role"),
) -> QueryResponse:
    """Process a RAG query through the full orchestrator pipeline."""
    if x_user_role not in VALID_ROLES:
        x_user_role = "user"  # Default to least-privileged

    pipeline = _get_pipeline(request)
    session_id = body.session_id or uuid.uuid4()

    result = await pipeline.orchestrator.process_query(
        query=body.query,
        session_id=session_id,
        user_id=x_user_id,
        role=x_user_role,
        strategy=body.strategy.value,
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
