"""gRPC server: DoclingChunker service implementation.

Implements the DoclingChunker service from chunker.proto using grpcio.aio.
Also registers:
  - Standard grpc.health.v1.Health service (grpcio-health-checking)
  - gRPC server reflection (grpcio-reflection)

Stream protocol (per chunker.proto comment):
  1. Exactly ONE DocumentHeader (first message).
  2. Zero or more Chunk messages (in order, 0-based chunk_index).
  3. Exactly ONE DocumentFooter (last message) — OR —
  4. Exactly ONE IngestError (instead of footer, on any failure).
  Never emit both footer and error.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

import grpc
import grpc.aio
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from .chunker import ChunkerPool, _sniff_mimetype
from .errors import classify_exception, format_stack
from .ids import chunk_id, document_id, section_id
from .limits import ServerLimits

# Import generated stubs — compiled into /app/src/agentic_rag/v1/ by Dockerfile.
from agentic_rag.v1 import chunker_pb2, chunker_pb2_grpc  # type: ignore[import]

logger = logging.getLogger(__name__)

# Current wire-compat schema version. Bump only on breaking proto changes.
_SCHEMA_VERSION: int = 1

# Informational version strings, injected from package at build time.
try:
    import docling
    _DOCLING_VERSION: str = docling.__version__
except Exception:
    _DOCLING_VERSION = "unknown"

try:
    import docling_core
    _DOCLING_CORE_VERSION: str = docling_core.__version__
except Exception:
    _DOCLING_CORE_VERSION = "unknown"


# ---------------------------------------------------------------------------
# Label → proto enum mapping
# ---------------------------------------------------------------------------

_LABEL_TO_PROTO: dict[str, int] = {
    "title":           chunker_pb2.DOC_ITEM_LABEL_TITLE,
    "section_header":  chunker_pb2.DOC_ITEM_LABEL_SECTION_HEADER,
    "text":            chunker_pb2.DOC_ITEM_LABEL_TEXT,
    "paragraph":       chunker_pb2.DOC_ITEM_LABEL_PARAGRAPH,
    "list_item":       chunker_pb2.DOC_ITEM_LABEL_LIST_ITEM,
    "caption":         chunker_pb2.DOC_ITEM_LABEL_CAPTION,
    "footnote":        chunker_pb2.DOC_ITEM_LABEL_FOOTNOTE,
    "page_header":     chunker_pb2.DOC_ITEM_LABEL_PAGE_HEADER,
    "page_footer":     chunker_pb2.DOC_ITEM_LABEL_PAGE_FOOTER,
    "table":           chunker_pb2.DOC_ITEM_LABEL_TABLE,
    "picture":         chunker_pb2.DOC_ITEM_LABEL_PICTURE,
    "formula":         chunker_pb2.DOC_ITEM_LABEL_FORMULA,
    "code":            chunker_pb2.DOC_ITEM_LABEL_CODE,
    "form":            chunker_pb2.DOC_ITEM_LABEL_FORM,
    "key_value":       chunker_pb2.DOC_ITEM_LABEL_KEY_VALUE,
    "document_index":  chunker_pb2.DOC_ITEM_LABEL_DOCUMENT_INDEX,
    "reference":       chunker_pb2.DOC_ITEM_LABEL_REFERENCE,
}


def _label_str_to_proto(label_str: str) -> int:
    """Convert a docling label string to its proto enum int value."""
    normalized = label_str.lower().strip()
    return _LABEL_TO_PROTO.get(normalized, chunker_pb2.DOC_ITEM_LABEL_OTHER)


# ---------------------------------------------------------------------------
# Servicer implementation
# ---------------------------------------------------------------------------


class DoclingChunkerServicer(chunker_pb2_grpc.DoclingChunkerServicer):
    """gRPC servicer for the DoclingChunker service."""

    def __init__(self, chunker_pool: ChunkerPool, server_limits: ServerLimits) -> None:
        self._pool = chunker_pool
        self._server_limits = server_limits

    async def ChunkDocument(
        self,
        request: chunker_pb2.ChunkDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[chunker_pb2.ChunkDocumentResponse]:
        """Parse + chunk a document, streaming results back.

        Stream order:
          DocumentHeader → Chunk* → DocumentFooter
          OR
          DocumentHeader → IngestError (on failure after header is sent)

        The header is always sent first so the client can correlate by
        document_id even in the error case.
        """
        start_ns = time.monotonic_ns()

        # -- 1. Resolve per-request limits --
        req_limits = request.limits
        effective = self._server_limits.merge_with_request(
            req_max_pdf_bytes=req_limits.max_pdf_bytes if req_limits else 0,
            req_max_pages=req_limits.max_pages if req_limits else 0,
            req_parse_timeout_s=req_limits.parse_timeout_s if req_limits else 0,
            req_stream_idle_timeout_s=req_limits.stream_idle_timeout_s if req_limits else 0,
            req_max_chunks_per_doc=req_limits.max_chunks_per_doc if req_limits else 0,
        )

        content: bytes = request.content
        filename: str = request.filename or "document"
        mimetype_hint: str = request.mimetype or ""
        doc_id_str: str = request.document_id or ""
        expected_sha256: str = request.sha256 or ""

        # -- 2. Size enforcement (before touching disk) --
        n_bytes = len(content)
        if n_bytes > effective.max_pdf_bytes:
            from .errors import RequestTooLargeError
            exc = RequestTooLargeError(
                f"Document size {n_bytes} bytes exceeds limit {effective.max_pdf_bytes}"
            )
            code_int, retryable = classify_exception(exc)
            yield chunker_pb2.ChunkDocumentResponse(
                error=chunker_pb2.IngestError(
                    code=code_int,
                    message=str(exc),
                    retryable=retryable,
                )
            )
            return

        # -- 3. Compute sha256 and sniff mimetype --
        actual_sha256 = hashlib.sha256(content).hexdigest()
        effective_mimetype = _sniff_mimetype(content, mimetype_hint)

        # -- 4. Emit DocumentHeader (always first, even before conversion) --
        yield chunker_pb2.ChunkDocumentResponse(
            header=chunker_pb2.DocumentHeader(
                schema_version=_SCHEMA_VERSION,
                document_id=doc_id_str,
                sha256=actual_sha256,
                filename=filename,
                mimetype=effective_mimetype,
                docling_version=_DOCLING_VERSION,
                docling_core_version=_DOCLING_CORE_VERSION,
                # n_pages and language filled in on footer after conversion
            )
        )

        # -- 5. Check client cancel after header --
        if context.cancelled():
            logger.info("Client cancelled after header for doc %s", doc_id_str)
            return

        # -- 6. Write bytes to a temp file (Docling needs a path) --
        # Determine file extension from filename for Docling's format detection.
        suffix = Path(filename).suffix or ".pdf"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(content)
        except OSError as exc:
            code_int, retryable = classify_exception(exc)
            yield chunker_pb2.ChunkDocumentResponse(
                error=chunker_pb2.IngestError(
                    code=code_int,
                    message=f"Failed to write temp file: {exc}",
                    retryable=retryable,
                    docling_stack=format_stack(exc),
                )
            )
            return

        # -- 7. Run Docling in thread pool (CPU-bound, not async) --
        parse_start_ns = time.monotonic_ns()
        try:
            conversion_result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._pool.convert_and_chunk,
                    tmp_path,
                    effective_mimetype,
                    actual_sha256,
                    expected_sha256,
                ),
                timeout=float(effective.parse_timeout_s),
            )
        except asyncio.TimeoutError as exc:
            timeout_exc = TimeoutError(
                f"Parse timed out after {effective.parse_timeout_s}s"
            )
            code_int, retryable = classify_exception(timeout_exc)
            yield chunker_pb2.ChunkDocumentResponse(
                error=chunker_pb2.IngestError(
                    code=code_int,
                    message=str(timeout_exc),
                    retryable=retryable,
                )
            )
            return
        except Exception as exc:
            code_int, retryable = classify_exception(exc)
            yield chunker_pb2.ChunkDocumentResponse(
                error=chunker_pb2.IngestError(
                    code=code_int,
                    message=str(exc),
                    retryable=retryable,
                    docling_stack=format_stack(exc),
                )
            )
            return
        finally:
            # Always clean up temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        parse_latency_ms = (time.monotonic_ns() - parse_start_ns) // 1_000_000

        # -- 8. Check page limit after conversion --
        if (
            conversion_result.n_pages > 0
            and conversion_result.n_pages > effective.max_pages
        ):
            from .errors import RequestTooLargeError
            exc = RequestTooLargeError(
                f"Document has {conversion_result.n_pages} pages, "
                f"exceeds limit {effective.max_pages}"
            )
            code_int, retryable = classify_exception(exc)
            yield chunker_pb2.ChunkDocumentResponse(
                error=chunker_pb2.IngestError(
                    code=code_int,
                    message=str(exc),
                    retryable=retryable,
                )
            )
            return

        # -- 9. Emit chunks --
        total_chunks = len(conversion_result.chunks)
        # Cap at max_chunks_per_doc if configured
        if (
            effective.max_chunks_per_doc > 0
            and total_chunks > effective.max_chunks_per_doc
        ):
            chunks_to_emit = conversion_result.chunks[: effective.max_chunks_per_doc]
            total_chunks = len(chunks_to_emit)
        else:
            chunks_to_emit = conversion_result.chunks

        # Derive the document UUID (stable from sha256)
        doc_uuid = document_id(actual_sha256) if not doc_id_str else None
        doc_uuid_str = doc_id_str if doc_id_str else str(doc_uuid)

        n_tables = 0
        n_pictures = 0
        n_formulas = 0
        n_code = 0

        for idx, chunk in enumerate(chunks_to_emit):
            if context.cancelled():
                logger.info("Client cancelled during chunk emission, doc %s", doc_id_str)
                return

            # Derive stable chunk_id
            cid = str(chunk_id(
                document_id(actual_sha256),
                idx,
                chunk.embed_text,
            ))

            # Derive stable section_id from heading path
            sec_id = ""
            if chunk.heading_path:
                sec_id = str(
                    section_id(document_id(actual_sha256), chunk.heading_path)
                )

            # Build proto DocItem list
            proto_doc_items: list[chunker_pb2.DocItem] = []
            for di in chunk.doc_items:
                label_proto = _label_str_to_proto(di["label_str"])
                bbox_proto = None
                if di["bbox"]:
                    bbox_proto = chunker_pb2.BBox(
                        l=di["bbox"]["l"],
                        t=di["bbox"]["t"],
                        r=di["bbox"]["r"],
                        b=di["bbox"]["b"],
                    )
                proto_doc_items.append(
                    chunker_pb2.DocItem(
                        self_ref=di["self_ref"],
                        label=label_proto,
                        raw_label=di["label_str"],
                        page_no=di["page_no"],
                        bbox=bbox_proto,
                        text=di["text"],
                        table_markdown=di["table_markdown"],
                        table_n_rows=di["table_n_rows"],
                        table_n_cols=di["table_n_cols"],
                        picture_caption=di["picture_caption"],
                    )
                )

            # Aggregate footer counts
            if chunk.has_table:
                n_tables += 1
            if chunk.has_picture:
                n_pictures += 1
            if chunk.has_formula:
                n_formulas += 1
            if chunk.has_code:
                n_code += 1

            proto_chunk = chunker_pb2.Chunk(
                chunk_index=idx,
                total_chunks=total_chunks,
                chunk_id=cid,
                embed_text=chunk.embed_text,
                raw_text=chunk.raw_text,
                heading_path=chunk.heading_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                doc_items=proto_doc_items,
                has_table=chunk.has_table,
                has_picture=chunk.has_picture,
                has_code=chunk.has_code,
                has_formula=chunk.has_formula,
                section_title=chunk.section_title,
                section_id=sec_id,
            )

            yield chunker_pb2.ChunkDocumentResponse(chunk=proto_chunk)

        # -- 10. Emit DocumentFooter --
        total_latency_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        yield chunker_pb2.ChunkDocumentResponse(
            footer=chunker_pb2.DocumentFooter(
                n_chunks=total_chunks,
                n_tables=n_tables,
                n_pictures=n_pictures,
                n_formulas=n_formulas,
                n_code=n_code,
                total_handle_latency_ms=total_latency_ms,
            )
        )

        logger.info(
            "ChunkDocument complete: doc=%s pages=%d chunks=%d latency_ms=%d",
            doc_id_str or doc_uuid_str,
            conversion_result.n_pages,
            total_chunks,
            total_latency_ms,
        )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


async def create_server(
    port: int,
    chunker_pool: ChunkerPool,
    server_limits: ServerLimits,
) -> grpc.aio.Server:
    """Create and configure the gRPC aio server."""
    server = grpc.aio.server()

    # Register DoclingChunker service
    servicer = DoclingChunkerServicer(chunker_pool, server_limits)
    chunker_pb2_grpc.add_DoclingChunkerServicer_to_server(servicer, server)

    # Register standard gRPC health service
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    # Mark our service as SERVING
    health_servicer.set(
        "agentic_rag.v1.DoclingChunker",
        health_pb2.HealthCheckResponse.SERVING,
    )
    # Mark overall server health
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    # Enable gRPC server reflection (for grpcurl without -proto flag)
    service_names = (
        chunker_pb2.DESCRIPTOR.services_by_name["DoclingChunker"].full_name,
        health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port(f"0.0.0.0:{port}")
    return server
