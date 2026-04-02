"""Document parsing and chunking pipeline.

Architecture:
  1. Parse source → DoclingDocument (preserves hierarchy, tables, provenance)
  2. Export to markdown (clean text, no serializer artifacts)
  3. Chunk with paragraph-aware splitter (overlap for context)
  4. Store chunk metadata alongside embeddings
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument  # type: ignore[attr-defined]
from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret

from agentic_ra.core.errors import EmbeddingError
from agentic_ra.retrieval.chunker import chunk_text
from agentic_ra.retrieval.html_preprocessor import is_legi_html, preprocess_legi_html

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkResult:
    """Result of document parsing and chunking.

    Attributes:
        documents: Haystack Documents with text ready for embedding.
        docling_doc: The parsed DoclingDocument (persist for provenance).
        chunk_count: Number of chunks produced.
    """

    documents: list[Document]
    docling_doc: DoclingDocument | None
    chunk_count: int


class EmbeddingPipeline:
    """Document parsing (Docling) + chunking (paragraph splitter) pipeline.

    Args:
        model: OpenAI embedding model name.
        api_key: OpenAI API key.
        dimension: Embedding output dimensionality.
        batch_size: Embedding batch size.
        max_tokens: Maximum tokens per chunk (~4 chars per token).
        merge_peers: Unused, kept for config compatibility.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimension: int = 1536,
        batch_size: int = 32,
        max_tokens: int = 1024,
        merge_peers: bool = True,
    ) -> None:
        self._model_name = model
        self._max_chars = max_tokens * 4  # ~4 chars per token
        self._overlap_chars = min(200, self._max_chars // 5)

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = False
        pdf_opts.force_backend_text = True
        pdf_opts.do_table_structure = True
        try:
            pdf_opts.table_structure_options = TableStructureOptions(mode="accurate")
        except Exception:
            logger.debug("TableStructureOptions 'accurate' mode not available, using default")

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            },
        )

        self._embedder = OpenAIDocumentEmbedder(
            model=model,
            api_key=Secret.from_token(api_key)
            if api_key
            else Secret.from_env_var("OPENAI_API_KEY"),
            dimensions=dimension,
            batch_size=batch_size,
        )

    def warm_up(self) -> None:
        """No-op for API-based embedders. Kept for interface compatibility."""

    def chunk_text(self, documents: list[Document]) -> ChunkResult:
        """Chunk pre-parsed text documents.

        For plain text that doesn't need Docling file parsing
        (e.g., API-submitted text).
        """
        if not documents:
            return ChunkResult(documents=[], docling_doc=None, chunk_count=0)

        try:
            full_text = "\n\n".join(
                d.content for d in documents if d.content
            )
            return self._chunk_text(full_text, extra_meta={})

        except Exception as e:
            raise EmbeddingError(
                "Document chunking failed",
                model=self._model_name,
                operation="chunk_text",
                document_count=len(documents),
                original_error=str(e),
            ) from e

    def chunk_file(self, source: str | Path) -> ChunkResult:
        """Parse a file/URL with Docling and chunk.

        Supports PDF, DOCX, PPTX, HTML, Markdown, images, and more.
        """
        try:
            actual_source = self._maybe_preprocess_html(source)
            result = self._converter.convert(actual_source)
            docling_doc: DoclingDocument = result.document

            # Export to markdown, strip code fences from mislabeled blocks
            markdown = docling_doc.export_to_markdown()
            markdown = re.sub(r"```\n?", "", markdown)

            return self._chunk_text(
                markdown,
                extra_meta={"source_path": str(source)},
                docling_doc=docling_doc,
            )

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                "File parsing/chunking failed",
                model=self._model_name,
                operation="chunk_file",
                source=str(source),
                original_error=str(e),
            ) from e

    def chunk_html(self, html: str, source_path: str = "html_input") -> ChunkResult:
        """Parse an HTML string through the pipeline."""
        if is_legi_html(html):
            logger.info("Detected legislatie.just.ro HTML — preprocessing")
            html = preprocess_legi_html(html)

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8",
        ) as f:
            f.write(html)
            tmp_path = f.name

        try:
            result = self._converter.convert(tmp_path)
            docling_doc: DoclingDocument = result.document
            markdown = docling_doc.export_to_markdown()
            markdown = re.sub(r"```\n?", "", markdown)

            return self._chunk_text(
                markdown,
                extra_meta={"source_path": source_path},
                docling_doc=docling_doc,
            )
        except Exception as e:
            raise EmbeddingError(
                "HTML parsing/chunking failed",
                model=self._model_name,
                operation="chunk_html",
                source=source_path,
                original_error=str(e),
            ) from e
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _maybe_preprocess_html(source: str | Path) -> str | Path:
        """If *source* is an HTML file with legislatie.just.ro content, preprocess it."""
        path = Path(source) if not isinstance(source, Path) else source
        if path.suffix.lower() not in (".html", ".htm"):
            return source

        raw = path.read_text(encoding="utf-8", errors="replace")
        if not is_legi_html(raw):
            return source

        logger.info("Detected legislatie.just.ro HTML in %s — preprocessing", path.name)
        cleaned = preprocess_legi_html(raw)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8",
        )
        tmp.write(cleaned)
        tmp.close()
        return Path(tmp.name)

    @staticmethod
    def build_breadcrumb_chunks(chunk_result: ChunkResult) -> list[str]:
        """Build chunk texts — just return the text content.

        Headings are already embedded in the markdown text from Docling's
        export, so no separate breadcrumb injection is needed.
        """
        return [
            doc.meta.get("chunk_text", doc.content or "")
            for doc in chunk_result.documents
            if (doc.meta.get("chunk_text", doc.content or "")).strip()
        ]

    def _chunk_text(
        self,
        text: str,
        extra_meta: dict[str, Any],
        docling_doc: DoclingDocument | None = None,
    ) -> ChunkResult:
        """Chunk plain text and convert to Haystack Documents."""
        chunks = chunk_text(
            text,
            max_words=self._max_chars // 5,  # ~5 chars per word average
            step_words=self._max_chars // 7,  # ~30% overlap
        )

        haystack_docs: list[Document] = []
        for chunk in chunks:
            meta: dict[str, Any] = {
                "chunk_text": chunk,
                "headings": [],
                "captions": [],
                **extra_meta,
            }
            haystack_docs.append(
                Document(content=chunk, meta=meta)
            )

        return ChunkResult(
            documents=haystack_docs,
            docling_doc=docling_doc,
            chunk_count=len(chunks),
        )
