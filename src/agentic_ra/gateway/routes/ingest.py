"""Document ingestion endpoint — chunk, embed, store, and persist parsed docs."""

from __future__ import annotations

import asyncio
import uuid as uuid_mod

from fastapi import APIRouter, Request
from haystack import Document
from pydantic import BaseModel, Field
from qdrant_client import models as qdrant_models

from agentic_ra.core.config import get_settings
from agentic_ra.pipelines.factory import RAGPipeline

router = APIRouter(prefix="/api/v1", tags=["ingest"])


class DocumentInput(BaseModel):
    """A document to ingest."""

    content: str = Field(..., min_length=1, description="Document text content")
    metadata: dict[str, str] = Field(default_factory=dict, description="Document metadata")


class IngestRequest(BaseModel):
    """Batch document ingestion request."""

    documents: list[DocumentInput] = Field(..., min_length=1, max_length=100)
    source_path: str = Field(default="api_input", description="Source identifier for provenance")


class IngestResponse(BaseModel):
    """Ingestion result."""

    ingested_count: int
    chunk_count: int
    parsed_document_id: uuid_mod.UUID
    collection: str
    index: str


def _get_pipeline(request: Request) -> RAGPipeline:
    """Extract the RAG pipeline from app state."""
    return request.app.state.pipeline  # type: ignore[no-any-return]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store documents. Persists parsed DoclingDocument JSON."""
    pipeline = _get_pipeline(request)
    settings = get_settings()
    collection = settings.qdrant_collection
    index = settings.opensearch_index

    docs = [
        Document(content=d.content, meta=d.metadata)
        for d in body.documents
    ]

    # Chunk with Docling HybridChunker
    chunk_result = await asyncio.to_thread(
        pipeline.embedding_pipeline.chunk_text, docs
    )

    # Persist parsed DoclingDocument JSON
    docling_json = (
        chunk_result.docling_doc.export_to_dict()
        if chunk_result.docling_doc is not None
        else {}
    )
    parsed_doc = await pipeline.document_store.save(
        source_path=body.source_path,
        docling_json=docling_json,
        chunk_count=chunk_result.chunk_count,
        metadata={"ingested_doc_count": len(body.documents)},
    )

    # Inject parsed_document_id into chunk metadata
    for doc in chunk_result.documents:
        doc.meta["parsed_document_id"] = str(parsed_doc.id)

    # Embed with OpenAI
    embedded = await asyncio.to_thread(
        pipeline.embedding_pipeline.embed, chunk_result.documents
    )

    # Store in Qdrant via public API
    points = [
        qdrant_models.PointStruct(
            id=str(uuid_mod.uuid4()),
            vector=doc.embedding,
            payload={
                "content": doc.content,
                "chunk_text": doc.meta.get("chunk_text", doc.content),
                "parsed_document_id": doc.meta.get("parsed_document_id", ""),
                "headings": doc.meta.get("headings", []),
                "source_path": body.source_path,
            },
        )
        for doc in embedded
        if doc.embedding is not None
    ]

    if points:
        await pipeline.qdrant.upsert(collection=collection, points=points)

    # Store in OpenSearch via public API
    for doc in embedded:
        await pipeline.opensearch.index_document(
            index=index,
            doc_id=str(uuid_mod.uuid4()),
            body={
                "content": doc.meta.get("chunk_text", doc.content),
                "parsed_document_id": doc.meta.get("parsed_document_id", ""),
                "headings": doc.meta.get("headings", []),
                "source_path": body.source_path,
            },
        )

    return IngestResponse(
        ingested_count=len(body.documents),
        chunk_count=len(embedded),
        parsed_document_id=parsed_doc.id,
        collection=collection,
        index=index,
    )
