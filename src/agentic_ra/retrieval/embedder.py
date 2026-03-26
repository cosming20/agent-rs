"""Document chunking and embedding pipeline using Haystack components."""

from __future__ import annotations

from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.utils import Secret

from agentic_ra.core.errors import EmbeddingError


class EmbeddingPipeline:
    """Document chunking and embedding pipeline.

    Wraps Haystack's ``DocumentSplitter`` and ``OpenAIDocumentEmbedder``
    into a single pipeline that chunks documents and generates embeddings
    via the OpenAI API.

    Args:
        model: OpenAI embedding model name.
        api_key: OpenAI API key.
        dimension: Embedding output dimensionality.
        batch_size: Embedding batch size.
        chunk_size: Number of words per chunk.
        chunk_overlap: Word overlap between consecutive chunks.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimension: int = 1536,
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
        self._embedder = OpenAIDocumentEmbedder(
            model=model,
            api_key=Secret.from_token(api_key) if api_key else Secret.from_env_var("OPENAI_API_KEY"),
            dimensions=dimension,
            batch_size=batch_size,
        )

    def warm_up(self) -> None:
        """No-op for OpenAI embedders (API-based, no local model to load).

        Kept for interface compatibility with the pipeline factory.
        """

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
