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

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker  # type: ignore[attr-defined]  # re-exported at runtime
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
        max_tokens: int = 512,
        merge_peers: bool = True,
    ) -> None:
        self._model_name = model
        self._converter = DocumentConverter()
        self._chunker = HybridChunker(  # type: ignore[call-arg]  # pydantic model accepts kwargs at runtime
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )
        self._embedder = OpenAIDocumentEmbedder(
            model=model,
            api_key=Secret.from_token(api_key) if api_key else Secret.from_env_var("OPENAI_API_KEY"),
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

    def embed(self, documents: list[Document]) -> list[Document]:
        """Generate embeddings for pre-chunked documents.

        Args:
            documents: Haystack Documents (typically from chunk_text/chunk_file).

        Returns:
            Documents with populated ``embedding`` field.
        """
        if not documents:
            return []

        try:
            result = self._embedder.run(documents=documents)
            embedded: list[Document] = result["documents"]
            return embedded
        except Exception as e:
            raise EmbeddingError(
                "Embedding generation failed",
                model=self._model_name,
                operation="embed",
                document_count=len(documents),
                original_error=str(e),
            ) from e

    def run(self, documents: list[Document]) -> list[Document]:
        """Chunk and embed text documents in one step.

        Convenience method that chains chunk_text → embed.

        Args:
            documents: Haystack Documents with text content.

        Returns:
            Embedded documents with chunk metadata.
        """
        if not documents:
            return []

        chunk_result = self.chunk_text(documents)
        return self.embed(chunk_result.documents)

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
                "doc_items_count": len(chunk_meta.doc_items) if chunk_meta.doc_items else 0,
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
