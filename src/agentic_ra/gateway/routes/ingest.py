"""Document ingestion endpoint — chunk, embed, and store documents."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from haystack import Document
from pydantic import BaseModel, Field
from qdrant_client import models as qdrant_models

from agentic_ra.pipelines.factory import RAGPipeline

router = APIRouter(prefix="/api/v1", tags=["ingest"])


class DocumentInput(BaseModel):
    """A document to ingest."""

    content: str = Field(..., min_length=1, description="Document text content")
    metadata: dict[str, str] = Field(default_factory=dict, description="Document metadata")


class IngestRequest(BaseModel):
    """Batch document ingestion request."""

    documents: list[DocumentInput] = Field(..., min_length=1, max_length=100)


class IngestResponse(BaseModel):
    """Ingestion result."""

    ingested_count: int
    chunk_count: int
    collection: str
    index: str


def _get_pipeline(request: Request) -> RAGPipeline:
    """Extract the RAG pipeline from app state."""
    return request.app.state.pipeline  # type: ignore[no-any-return]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store documents in Qdrant and OpenSearch.

    Documents are chunked with Docling HybridChunker (structure-aware),
    embedded via OpenAI, and stored in both Qdrant (vector) and
    OpenSearch (keyword) for hybrid retrieval.
    """
    pipeline = _get_pipeline(request)

    # Convert to Haystack Documents
    docs = [
        Document(content=d.content, meta=d.metadata)
        for d in body.documents
    ]

    # Chunk with Docling HybridChunker + embed with OpenAI (sync — run in thread)
    embedded = await asyncio.to_thread(pipeline.embedding_pipeline.run, docs)

    # Store in Qdrant (vector search)
    points = [
        qdrant_models.PointStruct(
            id=idx,
            vector=doc.embedding,
            payload={
                "content": doc.content,
                "chunk_text": doc.meta.get("chunk_text", doc.content),
                **(doc.meta or {}),
            },
        )
        for idx, doc in enumerate(embedded)
        if doc.embedding is not None
    ]

    if points:
        await pipeline.qdrant._client.upsert(
            collection_name="documents",
            points=points,
        )

    # Store in OpenSearch (keyword/BM25 search)
    for idx, doc in enumerate(embedded):
        await pipeline.opensearch._client.index(
            index="documents",
            id=str(idx),
            body={
                "content": doc.meta.get("chunk_text", doc.content),
                **(doc.meta or {}),
            },
            refresh=True,
        )

    return IngestResponse(
        ingested_count=len(body.documents),
        chunk_count=len(embedded),
        collection="documents",
        index="documents",
    )
