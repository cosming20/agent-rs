"""Async Qdrant client wrapper with error handling and collection management."""

from __future__ import annotations

from haystack import Document
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient, models

from agentic_ra.core.errors import QdrantError


class QdrantStore:
    """Async Qdrant client wrapper.

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

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> list[Document]:
        """Vector similarity search returning Haystack Documents.

        Args:
            collection: Qdrant collection name.
            query_vector: Query embedding vector.
            limit: Maximum results to return.

        Returns:
            Documents with score and source in metadata.
        """
        try:
            results = await self._client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
            )
            return [
                Document(
                    content=point.payload.get("content", "") if point.payload else "",
                    meta={
                        "score": point.score,
                        "source": "qdrant",
                        **(point.payload or {}),
                    },
                )
                for point in results.points
            ]
        except Exception as e:
            raise QdrantError(
                "Vector search failed",
                collection=collection,
                operation="search",
                original_error=str(e),
            ) from e

    async def upsert(
        self,
        collection: str,
        points: list[models.PointStruct],
    ) -> None:
        """Upsert points into a collection.

        Args:
            collection: Qdrant collection name.
            points: Points with id, vector, and payload.
        """
        try:
            await self._client.upsert(
                collection_name=collection,
                points=points,
            )
        except Exception as e:
            raise QdrantError(
                "Upsert failed",
                collection=collection,
                operation="upsert",
                point_count=len(points),
                original_error=str(e),
            ) from e

    async def close(self) -> None:
        """Close the underlying client connection."""
        try:
            await self._client.close()
        except Exception as e:
            raise QdrantError("Failed to close client", operation="close") from e
