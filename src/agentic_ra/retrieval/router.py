"""Retrieval strategy router with query embedding and parallel dispatch."""

from __future__ import annotations

import asyncio
import enum

from haystack import Document
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret

from agentic_ra.core.errors import EmbeddingError, RoutingError, StoreError
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.qdrant.client import QdrantStore


class RetrievalStrategy(enum.StrEnum):
    """Available retrieval strategies."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QueryEmbedder:
    """Query text embedder for retrieval.

    Wraps Haystack ``OpenAITextEmbedder``.
    Synchronous — callers bridge via ``asyncio.to_thread()``.

    Args:
        model: OpenAI embedding model name.
        api_key: OpenAI API key.
        dimension: Embedding output dimensionality.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimension: int = 1536,
    ) -> None:
        self._model_name = model
        self._embedder = OpenAITextEmbedder(
            model=model,
            api_key=Secret.from_token(api_key) if api_key else Secret.from_env_var("OPENAI_API_KEY"),
            dimensions=dimension,
        )

    def warm_up(self) -> None:
        """No-op for OpenAI embedders (API-based, no local model to load).

        Kept for interface compatibility with the pipeline factory.
        """

    def run(self, text: str) -> list[float]:
        """Embed query text into a vector.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector as list of floats.
        """
        try:
            result = self._embedder.run(text=text)
            embedding: list[float] = result["embedding"]
            return embedding
        except Exception as e:
            raise EmbeddingError(
                "Query embedding failed",
                model=self._model_name,
                operation="embed_query",
                original_error=str(e),
            ) from e


class RetrievalRouter:
    """Orchestrates retrieval: embeds query, selects strategy, dispatches to stores.

    Args:
        query_embedder: Warmed-up QueryEmbedder instance.
        qdrant_store: QdrantStore for vector search.
        opensearch_store: OpenSearchStore for keyword search.
        collection: Qdrant collection name.
        index: OpenSearch index name.
    """

    def __init__(
        self,
        query_embedder: QueryEmbedder,
        qdrant_store: QdrantStore,
        opensearch_store: OpenSearchStore,
        collection: str,
        index: str,
    ) -> None:
        self._embedder = query_embedder
        self._qdrant = qdrant_store
        self._opensearch = opensearch_store
        self._collection = collection
        self._index = index

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        limit: int = 10,
    ) -> list[Document]:
        """Retrieve documents using the specified strategy.

        Args:
            query: User query text.
            strategy: Retrieval strategy to use.
            limit: Maximum results per source.

        Returns:
            Retrieved documents (unranked, may contain duplicates for hybrid).
        """
        try:
            if strategy == RetrievalStrategy.KEYWORD:
                return await self._keyword_search(query, limit)

            # VECTOR and HYBRID both need query embedding
            query_vector = await asyncio.to_thread(self._embedder.run, query)

            if strategy == RetrievalStrategy.VECTOR:
                return await self._vector_search(query_vector, limit)

            # HYBRID: parallel dispatch
            vector_results, keyword_results = await asyncio.gather(
                self._vector_search(query_vector, limit),
                self._keyword_search(query, limit),
            )
            return vector_results + keyword_results

        except (EmbeddingError, RoutingError, StoreError):
            raise
        except Exception as e:
            raise RoutingError(
                "Retrieval failed",
                strategy=strategy.value,
                operation="retrieve",
                original_error=str(e),
            ) from e

    async def _vector_search(
        self, query_vector: list[float], limit: int
    ) -> list[Document]:
        """Dispatch vector similarity search to Qdrant."""
        return await self._qdrant.search(
            collection=self._collection,
            query_vector=query_vector,
            limit=limit,
        )

    async def _keyword_search(self, query: str, limit: int) -> list[Document]:
        """Dispatch BM25 keyword search to OpenSearch."""
        return await self._opensearch.search(
            index=self._index,
            query=query,
            limit=limit,
        )
