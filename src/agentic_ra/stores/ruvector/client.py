"""Async HTTP client for RuVector REST API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agentic_ra.core.errors import QdrantError

logger = logging.getLogger(__name__)


class RuVectorClient:
    """Async RuVector client via REST API (port 6333).

    API-compatible with Qdrant's REST surface for basic operations.

    Args:
        base_url: RuVector HTTP endpoint (e.g. http://localhost:1064).
        api_key: Optional API key for authentication.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    async def create_collection(
        self, name: str, dimensions: int, distance: str = "cosine"
    ) -> None:
        """Create a vector collection."""
        resp = await self._client.post(
            "/collections",
            json={"name": name, "dimensions": dimensions, "distance": distance},
        )
        resp.raise_for_status()
        logger.info("Created RuVector collection: %s (dim=%d)", name, dimensions)

    async def list_collections(self) -> list[str]:
        """List all collection names."""
        resp = await self._client.get("/collections")
        resp.raise_for_status()
        data = resp.json()
        return [c["name"] for c in data.get("collections", [])]

    async def has_collection(self, name: str) -> bool:
        """Check if a collection exists."""
        collections = await self.list_collections()
        return name in collections

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert vectors with metadata."""
        resp = await self._client.put(
            f"/collections/{collection}/points",
            json={"points": points},
        )
        resp.raise_for_status()

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """k-NN vector search."""
        body: dict[str, Any] = {"vector": query_vector, "k": k}
        if filter:
            body["filter"] = filter
        resp = await self._client.post(
            f"/collections/{collection}/search",
            json=body,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def batch_search(
        self, collection: str, queries: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Batch k-NN search."""
        resp = await self._client.post(
            f"/collections/{collection}/search/batch",
            json={"queries": queries},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def delete_points(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        for point_id in ids:
            resp = await self._client.delete(
                f"/collections/{collection}/points/{point_id}"
            )
            resp.raise_for_status()

    async def graph_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher graph query."""
        resp = await self._client.post(
            "/graph/query",
            json={"query": cypher, "params": params or {}},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def train_gnn(self, training_data: dict[str, Any]) -> None:
        """Submit training data for GNN index optimization."""
        resp = await self._client.post("/gnn/train", json=training_data)
        resp.raise_for_status()
        logger.info("GNN training batch submitted")

    async def health(self) -> bool:
        """Check if RuVector is healthy."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
