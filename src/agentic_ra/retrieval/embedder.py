"""Document chunking and embedding pipeline using Haystack components."""

from __future__ import annotations

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter

from agentic_ra.core.errors import EmbeddingError


class EmbeddingPipeline:
    """Document chunking and embedding pipeline.

    Wraps Haystack's ``DocumentSplitter`` and
    ``SentenceTransformersDocumentEmbedder`` into a single pipeline that
    chunks documents and generates embeddings.

    Args:
        model: Sentence-transformers model name or path.
        batch_size: Embedding batch size.
        chunk_size: Number of words per chunk.
        chunk_overlap: Word overlap between consecutive chunks.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ) -> None:
        self._model_name = model
        self._splitter = DocumentSplitter(
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
        )
        self._embedder = SentenceTransformersDocumentEmbedder(
            model=model,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

    def warm_up(self) -> None:
        """Load the embedding model into memory. Call once at startup."""
        try:
            self._embedder.warm_up()  # type: ignore[no-untyped-call]  # haystack warm_up() lacks annotations
        except Exception as e:
            raise EmbeddingError(
                "Failed to warm up embedding model",
                model=self._model_name,
                operation="warm_up",
                original_error=str(e),
            ) from e

    def run(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks and generate embeddings.

        Args:
            documents: Haystack Document objects with content to embed.

        Returns:
            Documents with populated ``embedding`` field.
        """
        if not documents:
            return []

        try:
            split_result = self._splitter.run(documents=documents)
            chunks: list[Document] = split_result["documents"]
        except Exception as e:
            raise EmbeddingError(
                "Document splitting failed",
                model=self._model_name,
                operation="split",
                document_count=len(documents),
                original_error=str(e),
            ) from e

        try:
            embed_result = self._embedder.run(documents=chunks)
            embedded: list[Document] = embed_result["documents"]
        except Exception as e:
            raise EmbeddingError(
                "Embedding generation failed",
                model=self._model_name,
                operation="embed",
                document_count=len(chunks),
                original_error=str(e),
            ) from e

        return embedded
