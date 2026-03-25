"""Async Qdrant client wrapper with error handling and collection management."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from agentic_ra.core.errors import QdrantError


class QdrantStore:
    """Async Qdrant client wrapper.

    Args:
        url: Qdrant HTTP/REST URL (e.g., ``http://localhost:1051``).
        api_key: Qdrant API key for authentication.
    """

    def __init__(self, url: str, api_key: str) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)

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

    async def close(self) -> None:
        """Close the underlying client connection."""
        await self._client.close()
