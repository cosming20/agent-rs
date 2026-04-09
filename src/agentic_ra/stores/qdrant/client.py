"""Async Qdrant client wrapper with error handling and collection management."""


import uuid

from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient, models

from agentic_ra.core.errors import QdrantError


class QdrantStore:
    """Async Qdrant client wrapper.

    Provides collection management, vector upsert, and similarity search
    for the strategy learning collection (and any other standalone Qdrant use).

    Note: Document ingestion goes through Cognee, which manages its own Qdrant
    connection. This wrapper is for the ``query_strategies`` collection only.

    Args:
        url: Qdrant HTTP/REST URL (e.g., ``http://localhost:1051``).
        api_key: Qdrant API key as ``SecretStr`` for safe handling.
    """

    def __init__(self, url: str, api_key: SecretStr) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key.get_secret_value())

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """Create a collection if it does not already exist.

        Args:
            name: Collection name.
            vector_size: Dimensionality of vectors.
            distance: Distance metric for similarity search.
        """
        try:
            if await self._client.collection_exists(name):
                return
            await self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
        except Exception as e:
            raise QdrantError(
                "Failed to ensure collection",
                collection=name,
                operation="ensure_collection",
            ) from e

    async def upsert(
        self,
        collection: str,
        vector: list[float],
        payload: dict[str, object],
        point_id: str | None = None,
    ) -> None:
        """Insert or update a single vector point with payload.

        Args:
            collection: Target collection name.
            vector: Embedding vector.
            payload: Metadata to store alongside the vector.
            point_id: UUID string for the point. Auto-generated if not provided.
        """
        pid = point_id or str(uuid.uuid4())
        try:
            await self._client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=pid,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as e:
            raise QdrantError(
                "Failed to upsert point",
                collection=collection,
                operation="upsert",
                point_id=pid,
            ) from e

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 20,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, object]]:
        """Vector similarity search returning raw payload dicts.

        Args:
            collection: Collection to search.
            query_vector: Query embedding.
            limit: Maximum results to return.
            score_threshold: Optional minimum similarity score cutoff.
            query_filter: Optional Qdrant filter for server-side payload filtering.

        Returns:
            List of payload dicts from matching points, each enriched with
            the Qdrant similarity ``score`` and ``created_at`` from payload.
        """
        try:
            kwargs: dict[str, object] = {
                "collection_name": collection,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True,
            }
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold
            if query_filter is not None:
                kwargs["query_filter"] = query_filter

            results = await self._client.search(**kwargs)  # type: ignore[arg-type, attr-defined]

            payloads: list[dict[str, object]] = []
            for hit in results:
                payload = dict(hit.payload or {})
                payload["_score"] = hit.score
                payloads.append(payload)
            return payloads

        except Exception as e:
            raise QdrantError(
                "Search failed",
                collection=collection,
                operation="search",
            ) from e

    async def close(self) -> None:
        """Close the underlying async client connection."""
        try:
            await self._client.close()
        except Exception as e:
            raise QdrantError("Failed to close client", operation="close") from e


