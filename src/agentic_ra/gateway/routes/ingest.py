"""Document ingestion endpoint — Docling parse → Cognee ingest.

Flow:
  1. Docling parses documents (structure-aware chunking, DoclingDocument JSON)
  2. Cognee add + cognify (chunking, embedding → Qdrant, entity extraction → Neo4j)
  3. Postgres persists DoclingDocument JSON for provenance
"""

from __future__ import annotations

import asyncio
import logging
import uuid as uuid_mod

from fastapi import APIRouter, Header, Request
from haystack import Document
from pydantic import BaseModel, Field

from agentic_ra.agent.tools import _sanitize_user_id
from agentic_ra.pipelines.factory import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingest"])


class DocumentInput(BaseModel):
    """A document to ingest."""

    content: str = Field(
        ..., min_length=1, max_length=500_000, description="Document text content"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Document metadata"
    )


class IngestRequest(BaseModel):
    """Batch document ingestion request."""

    documents: list[DocumentInput] = Field(..., min_length=1, max_length=100)
    source_path: str = Field(
        default="api_input", description="Source identifier for provenance"
    )


class IngestResponse(BaseModel):
    """Ingestion result."""

    ingested_count: int
    chunk_count: int
    parsed_document_id: uuid_mod.UUID
    cognee_ingested: bool


def _get_pipeline(request: Request) -> RAGPipeline:
    """Extract the RAG pipeline from app state."""
    return request.app.state.pipeline  # type: ignore[no-any-return]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: Request,
    body: IngestRequest,
    x_user_id: str = Header(default="anonymous", alias="X-User-Id"),
) -> IngestResponse:
    """Ingest documents: Docling parse → Cognee graph+vector."""
    pipeline = _get_pipeline(request)
    dataset = f"user_{_sanitize_user_id(x_user_id)}"

    docs = [Document(content=d.content, meta=d.metadata) for d in body.documents]

    # 1. Parse with Docling HybridChunker (structure-aware chunking)
    chunk_result = await asyncio.to_thread(pipeline.embedding_pipeline.chunk_text, docs)

    # 2. Persist DoclingDocument JSON in Postgres (provenance)
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
        user_id=x_user_id,
    )

    # 3. Cognee: add + cognify (chunking, embedding → Qdrant, entities → Neo4j)
    cognee_ok = False
    if pipeline.cognee_store is not None:
        try:
            full_text = "\n\n".join(d.content for d in body.documents)
            await pipeline.cognee_store.add_and_cognify(text=full_text, dataset=dataset)
            cognee_ok = True
            logger.info(
                "Cognee ingestion complete: %d docs → graph + vector",
                len(body.documents),
            )
        except Exception:
            # Intentional partial success: DoclingDocument JSON is saved in Postgres
            # for provenance. Cognee graph/vector ingestion failed but can be retried
            # later by re-running ingest. This prevents total ingestion failure when
            # Cognee/Neo4j/Qdrant is temporarily unavailable.
            logger.exception("Cognee add+cognify failed (non-fatal -- Postgres saved)")
    else:
        logger.warning("CogneeStore not configured — skipping graph+vector ingestion")

    return IngestResponse(
        ingested_count=len(body.documents),
        chunk_count=chunk_result.chunk_count,
        parsed_document_id=parsed_doc.id,
        cognee_ingested=cognee_ok,
    )
