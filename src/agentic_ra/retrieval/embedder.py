"""Document chunking and embedding pipeline using Docling + OpenAI.

Architecture:
  1. Parse source → DoclingDocument (preserves hierarchy, tables, provenance)
  2. Chunk with HybridChunker (structure-aware + token-budget-aware)
  3. Embed chunker.contextualize(chunk) text (headings/captions injected)
  4. Store chunk metadata and provenance alongside embeddings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker  # type: ignore[attr-defined]  # re-exported at runtime
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument  # type: ignore[attr-defined]  # re-exported at runtime
from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils import Secret

from agentic_ra.core.errors import EmbeddingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkResult:
    """Result of document parsing and chunking.

    Attributes:
        documents: Haystack Documents with contextualized text ready for embedding.
        docling_doc: The parsed DoclingDocument (persist this for context expansion).
        chunk_count: Number of chunks produced.
    """

    documents: list[Document]
    docling_doc: DoclingDocument | None
    chunk_count: int


class EmbeddingPipeline:
    """Document chunking and embedding pipeline using Docling + OpenAI.

    Uses Docling's HybridChunker for structure-aware, token-budget-aware
    chunking, then embeds the contextualized text via OpenAI.

    Args:
        model: OpenAI embedding model name.
        api_key: OpenAI API key.
        dimension: Embedding output dimensionality.
        batch_size: Embedding batch size.
        max_tokens: Maximum tokens per chunk (aligned to embedding model).
        merge_peers: Whether to merge small neighboring chunks with matching headings.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimension: int = 1536,
        batch_size: int = 32,
        max_tokens: int = 2048,
        merge_peers: bool = True,
    ) -> None:
        self._model_name = model

        # Match lawren-docling settings for better heading detection:
        # - force_backend_text=True: use PDF's native text, skip ML detection
        # - do_ocr=False: no OCR (faster, cleaner for digital PDFs)
        # - table_structure mode="accurate": better table parsing
        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = False
        pdf_opts.force_backend_text = True
        pdf_opts.do_table_structure = True
        try:
            pdf_opts.table_structure_options = TableStructureOptions(mode="accurate")
        except Exception:
            pass

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            },
        )

        # Use a tokenizer with max_tokens matching the embedding model's context window
        # (8191 for OpenAI text-embedding-3-*) to avoid false truncation warnings.
        # The chunk max_tokens is set separately and controls actual chunk size.
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name="bert-base-uncased",
            max_tokens=max_tokens,
        )
        self._chunker = HybridChunker(  # type: ignore[call-arg]  # pydantic model accepts kwargs at runtime
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
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
        """Chunk pre-parsed text documents using HybridChunker.

        For plain text that doesn't need Docling file parsing (e.g., API-submitted text),
        wraps content into a DoclingDocument then chunks.

        Args:
            documents: Haystack Documents with text content.

        Returns:
            ChunkResult with contextualized documents ready for embedding.
        """
        if not documents:
            return ChunkResult(documents=[], docling_doc=None, chunk_count=0)

        try:
            doc = DoclingDocument(name="api_input")
            for haystack_doc in documents:
                if haystack_doc.content:
                    doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=haystack_doc.content,
                    )

            return self._chunk_docling_doc(doc, extra_meta={})

        except Exception as e:
            raise EmbeddingError(
                "Document chunking failed",
                model=self._model_name,
                operation="chunk_text",
                document_count=len(documents),
                original_error=str(e),
            ) from e

    def chunk_file(self, source: str | Path) -> ChunkResult:
        """Parse a file/URL with Docling and chunk with HybridChunker.

        Supports PDF, DOCX, PPTX, HTML, Markdown, images, and more.

        Args:
            source: File path or URL to parse.

        Returns:
            ChunkResult with contextualized documents and the DoclingDocument.
        """
        try:
            result = self._converter.convert(source)
            docling_doc: DoclingDocument = result.document

            return self._chunk_docling_doc(
                docling_doc,
                extra_meta={"source_path": str(source)},
            )

        except Exception as e:
            raise EmbeddingError(
                "File parsing/chunking failed",
                model=self._model_name,
                operation="chunk_file",
                source=str(source),
                original_error=str(e),
            ) from e

    @staticmethod
    def build_breadcrumb_chunks(chunk_result: ChunkResult) -> list[str]:
        """Build chunk texts with heading breadcrumbs prepended.

        Each chunk gets its section path (e.g. "Titlul III > Capitolul II >
        Articolul 170") prepended so that heading context is embedded alongside
        the content.  This matches the lawren-docling approach and makes
        individual articles searchable by their heading text.

        Args:
            chunk_result: Result from chunk_file() or chunk_text().

        Returns:
            List of breadcrumb-prefixed chunk texts.
        """
        breadcrumb_chunks: list[str] = []
        for doc in chunk_result.documents:
            headings: list[str] = doc.meta.get("headings", [])
            chunk_text: str = doc.meta.get("chunk_text", doc.content or "")

            if headings:
                breadcrumb = " > ".join(headings)
                text = f"{breadcrumb}\n\n{chunk_text}"
            else:
                text = chunk_text

            if text.strip():
                breadcrumb_chunks.append(text)

        return breadcrumb_chunks

    def _chunk_docling_doc(
        self,
        docling_doc: DoclingDocument,
        extra_meta: dict[str, Any],
    ) -> ChunkResult:
        """Chunk a DoclingDocument and convert to Haystack Documents."""
        chunks = list(self._chunker.chunk(docling_doc))

        haystack_docs: list[Document] = []
        for chunk in chunks:
            # Use contextualize() — injects headings/captions for better embeddings
            contextualized_text: str = self._chunker.contextualize(chunk)

            chunk_meta: Any = chunk.meta  # DocMeta at runtime, untyped in stubs
            meta: dict[str, Any] = {
                "headings": list(chunk_meta.headings) if chunk_meta.headings else [],
                "captions": list(chunk_meta.captions) if chunk_meta.captions else [],
                "doc_items_count": len(chunk_meta.doc_items)
                if chunk_meta.doc_items
                else 0,
                "chunk_text": chunk.text,
                **extra_meta,
            }

            haystack_docs.append(
                Document(
                    content=contextualized_text,
                    meta=meta,
                )
            )

        return ChunkResult(
            documents=haystack_docs,
            docling_doc=docling_doc,
            chunk_count=len(chunks),
        )
