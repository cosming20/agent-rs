"""Async HTTP client for RuVector REST API."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from agentic_ra.core.errors import RuVectorError

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
        # H1: Validate URL scheme to prevent SSRF
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"RuVector URL must use http/https scheme, got: {parsed.scheme}"
            )

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
        try:
            resp = await self._client.post(
                "/collections",
                json={"name": name, "dimensions": dimensions, "distance": distance},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector create_collection failed",
                collection=name,
                status_code=exc.response.status_code,
                operation="create_collection",
            ) from exc
        logger.info("Created RuVector collection: %s (dim=%d)", name, dimensions)

    async def list_collections(self) -> list[str]:
        """List all collection names."""
        try:
            resp = await self._client.get("/collections")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector list_collections failed",
                status_code=exc.response.status_code,
                operation="list_collections",
            ) from exc
        data = resp.json()
        return [c["name"] for c in data.get("collections", [])]

    async def has_collection(self, name: str) -> bool:
        """Check if a collection exists."""
        collections = await self.list_collections()
        return name in collections

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert vectors with metadata."""
        try:
            resp = await self._client.put(
                f"/collections/{collection}/points",
                json={"points": points},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector upsert failed",
                collection=collection,
                status_code=exc.response.status_code,
                operation="upsert",
            ) from exc

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
        try:
            resp = await self._client.post(
                f"/collections/{collection}/search",
                json=body,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector search failed",
                collection=collection,
                status_code=exc.response.status_code,
                operation="search",
            ) from exc
        return resp.json().get("results", [])

    async def batch_search(
        self, collection: str, queries: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Batch k-NN search."""
        try:
            resp = await self._client.post(
                f"/collections/{collection}/search/batch",
                json={"queries": queries},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector batch_search failed",
                collection=collection,
                status_code=exc.response.status_code,
                operation="batch_search",
            ) from exc
        return resp.json().get("results", [])

    async def delete_points(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        for point_id in ids:
            try:
                resp = await self._client.delete(
                    f"/collections/{collection}/points/{point_id}"
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuVectorError(
                    "RuVector delete_points failed",
                    collection=collection,
                    point_id=point_id,
                    status_code=exc.response.status_code,
                    operation="delete_points",
                ) from exc

    async def graph_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher graph query."""
        try:
            resp = await self._client.post(
                "/graph/query",
                json={"query": cypher, "params": params or {}},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector graph_query failed",
                status_code=exc.response.status_code,
                operation="graph_query",
            ) from exc
        return resp.json().get("results", [])

    async def train_gnn(self, training_data: dict[str, Any]) -> None:
        """Submit training data for GNN index optimization."""
        try:
            resp = await self._client.post("/gnn/train", json=training_data)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuVectorError(
                "RuVector train_gnn failed",
                status_code=exc.response.status_code,
                operation="train_gnn",
            ) from exc
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
