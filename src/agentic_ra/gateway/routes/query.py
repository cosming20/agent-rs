"""Query endpoint — main RAG query API."""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["query"])


class QueryRequest(BaseModel):
    """Incoming RAG query request."""

    query: str = Field(..., min_length=1, max_length=10000, description="User query text")
    session_id: uuid.UUID | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    user_id: str = Field(default="anonymous", description="User identifier")
    strategy: str = Field(default="hybrid", description="Retrieval strategy: vector, keyword, hybrid")
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


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Process a RAG query and return an answer with citations.

    This is a placeholder — the orchestrator will be wired in Phase 8.
    """
    session_id = request.session_id or uuid.uuid4()

    # Placeholder response — will be wired to AgentOrchestrator
    return QueryResponse(
        answer=f"Placeholder response for: {request.query}",
        citations=[],
        query=request.query,
        confidence=0.0,
        is_grounded=False,
        strategy=request.strategy,
        session_id=session_id,
    )
