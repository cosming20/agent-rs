"""Docling DocumentConverter + HybridChunker wrapper.

Wraps the Docling pipeline behind a simple synchronous interface. The gRPC
server calls these methods in a thread pool via asyncio.to_thread() since
Docling's converter is CPU-bound and synchronous.

All Docling objects are constructed once at server startup (ChunkerPool.__init__)
so that model weights are loaded into memory before the first request arrives.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from .errors import (
    ChunkingError,
    DoclingParseError,
    InvalidMetadataError,
    UnsupportedMimetypeError,
)

logger = logging.getLogger(__name__)

# Supported MIME types that Docling can handle.
_SUPPORTED_MIMETYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/html",
        "text/markdown",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/tiff",
    }
)

# Magic bytes for mimetype sniffing when mimetype field is empty.
_MAGIC_PDF = b"%PDF"


def _sniff_mimetype(content: bytes, hint: str) -> str:
    """Return the effective mimetype from hint or magic bytes."""
    if hint:
        return hint.lower().strip()
    if content[:4] == _MAGIC_PDF:
        return "application/pdf"
    # Default fallback
    return "application/octet-stream"


@dataclass
class ChunkResult:
    """A single chunk produced by HybridChunker, with all fields needed for proto."""

    raw_text: str
    embed_text: str
    heading_path: list[str]
    page_start: int
    page_end: int
    token_count: int
    char_count: int
    doc_items: list[dict[str, Any]]
    has_table: bool
    has_picture: bool
    has_code: bool
    has_formula: bool
    section_title: str


@dataclass
class ConversionResult:
    """Output of ChunkerPool.convert_and_chunk()."""

    docling_doc: Any  # docling_core.types.doc.DoclingDocument
    chunks: list[ChunkResult]
    n_pages: int
    language: str
    title: str
    mimetype: str


class ChunkerPool:
    """Single shared DocumentConverter + HybridChunker, configured per spec.

    Thread-safe: Docling's DocumentConverter is stateless across convert() calls.
    Construction pre-loads all ONNX model weights so the first request is fast.
    """

    def __init__(self) -> None:
        logger.info("Initialising ChunkerPool — loading Docling models...")
        self._converter = self._build_converter()
        self._chunker = self._build_chunker()
        self._enc = tiktoken.encoding_for_model("text-embedding-3-large")
        logger.info("ChunkerPool ready.")

    # ------------------------------------------------------------------
    # Public interface (called synchronously, wrapped in to_thread by server)
    # ------------------------------------------------------------------

    def convert_and_chunk(
        self,
        path: Path,
        mimetype: str,
        sha256: str,
        expected_sha256: str,
    ) -> ConversionResult:
        """Convert a document and chunk it.

        Args:
            path: Filesystem path to the temporary document file.
            mimetype: Effective mimetype (already sniffed if not supplied).
            sha256: SHA-256 of the file bytes, computed by the caller.
            expected_sha256: SHA-256 from the request. Must match sha256.

        Raises:
            InvalidMetadataError: sha256 mismatch.
            UnsupportedMimetypeError: Docling cannot handle this mimetype.
            DoclingParseError: Docling raised during conversion.
            ChunkingError: HybridChunker raised.
        """
        if expected_sha256 and sha256 != expected_sha256:
            raise InvalidMetadataError(
                f"sha256 mismatch: got {sha256}, expected {expected_sha256}"
            )

        if mimetype not in _SUPPORTED_MIMETYPES:
            raise UnsupportedMimetypeError(
                f"Unsupported mimetype: {mimetype!r}. "
                f"Supported: {sorted(_SUPPORTED_MIMETYPES)}"
            )

        try:
            result = self._converter.convert(str(path))
        except Exception as exc:
            raise DoclingParseError(f"DocumentConverter failed: {exc}") from exc

        doc = result.document

        # --- page metadata ---
        n_pages = 0
        if hasattr(doc, "pages") and doc.pages:
            n_pages = len(doc.pages)

        # --- language ---
        language = ""
        if hasattr(doc, "detected_language") and doc.detected_language:
            language = str(doc.detected_language)

        # --- title ---
        title = ""
        if hasattr(doc, "name") and doc.name:
            title = str(doc.name)
        # Try to extract from metadata if present
        if not title and hasattr(doc, "metadata") and doc.metadata:
            meta = doc.metadata
            if hasattr(meta, "title") and meta.title:
                title = str(meta.title)

        # --- chunking ---
        try:
            raw_chunks = list(self._chunker.chunk(doc))
        except Exception as exc:
            raise ChunkingError(f"HybridChunker failed: {exc}") from exc

        chunks = [self._map_chunk(doc, chunk) for chunk in raw_chunks]

        return ConversionResult(
            docling_doc=doc,
            chunks=chunks,
            n_pages=n_pages,
            language=language,
            title=title,
            mimetype=mimetype,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_chunk(self, doc: Any, chunk: Any) -> ChunkResult:
        """Map a docling HybridChunker chunk to our ChunkResult."""
        raw_text: str = chunk.text if hasattr(chunk, "text") else ""

        # contextualize() prefixes the chunk with its heading hierarchy for
        # better embedding quality.
        try:
            embed_text: str = self._chunker.contextualize(chunk)
        except Exception as ctx_exc:
            # W7 fix #9: log when contextualize() fails — silent fallback to
            # raw_text means embed_text loses heading hierarchy context, which
            # degrades retrieval quality silently.
            logger.warning(
                "contextualize() failed, using raw_text: %s", ctx_exc
            )
            embed_text = raw_text

        # Heading path
        heading_path: list[str] = []
        if hasattr(chunk, "meta") and chunk.meta:
            meta = chunk.meta
            if hasattr(meta, "headings") and meta.headings:
                heading_path = list(meta.headings)

        # Doc items
        doc_items_raw: list[Any] = []
        if hasattr(chunk, "meta") and chunk.meta:
            meta = chunk.meta
            if hasattr(meta, "doc_items") and meta.doc_items:
                doc_items_raw = list(meta.doc_items)

        # Page range + flags
        page_start = 0
        page_end = 0
        has_table = False
        has_picture = False
        has_code = False
        has_formula = False
        doc_items: list[dict[str, Any]] = []

        for item in doc_items_raw:
            label_str = ""
            if hasattr(item, "label"):
                label_val = item.label
                label_str = str(label_val.value) if hasattr(label_val, "value") else str(label_val)

            # Page provenance
            item_page = 0
            item_bbox: dict[str, float] | None = None
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no:
                        p = int(prov.page_no)
                        if item_page == 0:
                            item_page = p
                        if page_start == 0 or p < page_start:
                            page_start = p
                        if p > page_end:
                            page_end = p
                    if hasattr(prov, "bbox") and prov.bbox and item_bbox is None:
                        b = prov.bbox
                        item_bbox = {
                            "l": float(getattr(b, "l", 0)),
                            "t": float(getattr(b, "t", 0)),
                            "r": float(getattr(b, "r", 0)),
                            "b": float(getattr(b, "b", 0)),
                        }

            # Flags
            label_upper = label_str.upper()
            if "TABLE" in label_upper:
                has_table = True
            if "PICTURE" in label_upper:
                has_picture = True
            if "CODE" in label_upper:
                has_code = True
            if "FORMULA" in label_upper:
                has_formula = True

            # Table markdown
            table_markdown = ""
            table_n_rows = 0
            table_n_cols = 0
            if "TABLE" in label_upper and hasattr(item, "data") and item.data:
                try:
                    table_markdown = item.export_to_markdown() if hasattr(item, "export_to_markdown") else ""
                except Exception:
                    table_markdown = ""
                if hasattr(item.data, "num_rows"):
                    table_n_rows = int(item.data.num_rows)
                if hasattr(item.data, "num_cols"):
                    table_n_cols = int(item.data.num_cols)

            # Picture caption
            picture_caption = ""
            if "PICTURE" in label_upper:
                if hasattr(item, "captions") and item.captions:
                    picture_caption = " ".join(
                        str(c.text) if hasattr(c, "text") else str(c)
                        for c in item.captions
                    )

            doc_items.append(
                {
                    "self_ref": str(item.self_ref) if hasattr(item, "self_ref") else "",
                    "label_str": label_str,
                    "page_no": item_page,
                    "bbox": item_bbox,
                    "text": str(item.text) if hasattr(item, "text") else "",
                    "table_markdown": table_markdown,
                    "table_n_rows": table_n_rows,
                    "table_n_cols": table_n_cols,
                    "picture_caption": picture_caption,
                }
            )

        token_count = len(
            self._enc.encode(embed_text, disallowed_special="all")
        )
        char_count = len(embed_text)
        section_title = heading_path[-1] if heading_path else ""

        return ChunkResult(
            raw_text=raw_text,
            embed_text=embed_text,
            heading_path=heading_path,
            page_start=page_start,
            page_end=page_end,
            token_count=token_count,
            char_count=char_count,
            doc_items=doc_items,
            has_table=has_table,
            has_picture=has_picture,
            has_code=has_code,
            has_formula=has_formula,
            section_title=section_title,
        )

    @staticmethod
    def _build_converter() -> DocumentConverter:
        """Build a DocumentConverter for production PDF ingestion.

        Table structure mode is configurable via `DOCLING_TABLEFORMER_MODE`
        env var: "accurate" (default in spec) or "fast" (3-4 GB lower peak
        memory usage — required when Docker Desktop VM is <12 GB).
        """
        import os

        from docling.datamodel.pipeline_options import TableStructureOptions

        mode_env = os.environ.get("DOCLING_TABLEFORMER_MODE", "accurate").lower()
        mode = (
            TableFormerMode.ACCURATE
            if mode_env == "accurate"
            else TableFormerMode.FAST
        )
        logger.info("TableFormer mode: %s", mode.name)

        pdf_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(mode=mode),
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )

    @staticmethod
    def _build_chunker() -> HybridChunker:
        """Build a HybridChunker with cl100k tokenizer, 1024-token chunks.

        OpenAITokenizer(tokenizer=enc, max_tokens=N) — the `tokenizer` field
        takes a tiktoken.Encoding object, not a model name string.
        """
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        enc = tiktoken.encoding_for_model("text-embedding-3-large")
        tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=1024)
        return HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
        )
