"""Cognee VectorDBInterface adapter backed by RuVector REST API."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingEngine,
)
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.engine import DataPoint

from pydantic import Field

from agentic_ra.stores.ruvector.client import RuVectorClient

logger = logging.getLogger(__name__)


class _IndexSchema(DataPoint):
    """Minimal DataPoint subclass for index_data_points text embedding."""

    text: str
    metadata: dict = Field(default_factory=lambda: {"index_fields": ["text"]})  # type: ignore[assignment]  # DataPoint.metadata is ClassVar; override as instance field for embedding config


class RuVectorVectorAdapter:
    """Cognee VectorDBInterface implementation delegating to RuVectorClient.

    Cognee's ``create_vector_engine`` instantiates registered adapters with::

        adapter(url=..., api_key=..., embedding_engine=..., database_name=...)

    This adapter accepts those kwargs and wires them into a ``RuVectorClient``.
    """

    name = "RuVector"

    def __init__(
        self,
        url: str = "",
        api_key: str = "",
        embedding_engine: EmbeddingEngine | None = None,
        *,
        database_name: str = "",
        **kwargs: Any,  # absorb extra kwargs from Cognee's factory
    ) -> None:
        if embedding_engine is None:
            from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
                get_embedding_engine,
            )

            embedding_engine = get_embedding_engine()

        self._embedding_engine: EmbeddingEngine = embedding_engine
        self._url = url
        self._api_key = api_key
        self._database_name = database_name

    def _client(self) -> RuVectorClient:
        """Create a fresh RuVectorClient (short-lived, like QDrantAdapter)."""
        return RuVectorClient(base_url=self._url, api_key=self._api_key)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """Embed text via Cognee's configured embedding engine."""
        return await self._embedding_engine.embed_text(data)

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    async def has_collection(self, collection_name: str) -> bool:
        client = self._client()
        try:
            return await client.has_collection(collection_name)
        finally:
            await client.close()

    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Any | None = None,
    ) -> None:
        client = self._client()
        try:
            if not await client.has_collection(collection_name):
                vector_size = self._embedding_engine.get_vector_size()
                await client.create_collection(
                    name=collection_name,
                    dimensions=vector_size,
                    distance="cosine",
                )
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Data points
    # ------------------------------------------------------------------

    async def create_data_points(
        self, collection_name: str, data_points: list[DataPoint]
    ) -> None:
        client = self._client()
        try:
            embeddable_texts = [
                DataPoint.get_embeddable_data(dp) for dp in data_points
            ]
            data_vectors = await self.embed_data(embeddable_texts)

            points = []
            for dp, vector in zip(data_points, data_vectors):
                points.append(
                    {
                        "id": str(dp.id),
                        "vector": vector,
                        "payload": dp.model_dump(),
                    }
                )

            await client.upsert(collection=collection_name, points=points)
        except Exception as exc:
            if "not found" in str(exc).lower():
                raise CollectionNotFoundError(
                    message=f"Collection {collection_name} not found!"
                ) from exc
            raise
        finally:
            await client.close()

    async def retrieve(
        self, collection_name: str, data_point_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Retrieve data points by ID.

        RuVector does not expose a dedicated retrieve-by-ID endpoint through
        the REST surface we wrap, so we return an empty list.  Cognee's core
        pipeline primarily relies on ``search`` and ``create_data_points``; the
        ``retrieve`` path is used only by optional introspection features.
        """
        logger.warning(
            "RuVector retrieve stub called for %d IDs in %s — no retrieve-by-ID endpoint available",
            len(data_point_ids),
            collection_name,
        )
        return []

    async def delete_data_points(
        self, collection_name: str, data_point_ids: list[UUID]
    ) -> None:
        client = self._client()
        try:
            str_ids = [str(uid) for uid in data_point_ids]
            await client.delete_points(collection=collection_name, ids=str_ids)
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        collection_name: str,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        limit: int | None = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
    ) -> list[ScoredResult]:
        if query_text is None and query_vector is None:
            from cognee.infrastructure.databases.exceptions import (
                MissingQueryParameterError,
            )

            raise MissingQueryParameterError()

        if not await self.has_collection(collection_name):
            return []

        if query_vector is None and query_text is not None:
            query_vector = (await self.embed_data([query_text]))[0]

        # Build optional filter for belongs_to_set
        ruvector_filter: dict[str, Any] | None = None
        if node_name:
            ruvector_filter = {
                "must": [
                    {
                        "key": "belongs_to_set",
                        "match": {"any": node_name},
                    }
                ]
            }

        client = self._client()
        try:
            results = await client.search(
                collection=collection_name,
                query_vector=query_vector,  # type: ignore[arg-type]
                k=limit or 15,
                filter=ruvector_filter,
            )

            scored: list[ScoredResult] = []
            for result in results:
                result_id = result.get("id", "")
                payload = result.get("payload", {})
                score = result.get("score", 0.0)
                scored.append(
                    ScoredResult(
                        id=UUID(str(result_id)) if result_id else UUID(int=0),
                        # RuVector returns cosine distance (0=identical, 1=orthogonal); invert to similarity
                        score=1.0 - score,
                        payload={**payload, "id": str(result_id)},
                    )
                )
            return scored
        except Exception as exc:
            logger.error("RuVector search error: %s", exc, exc_info=True)
            raise  # Let Cognee handle the failure — don't hide a dead backend
        finally:
            await client.close()

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int | None = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
    ) -> list[list[ScoredResult]]:
        embedded = await self.embed_data(query_texts)

        queries = [
            {"vector": vec, "k": limit or 15} for vec in embedded
        ]

        client = self._client()
        try:
            raw_results = await client.batch_search(
                collection=collection_name, queries=queries
            )

            all_scored: list[list[ScoredResult]] = []
            for batch in raw_results:
                scored: list[ScoredResult] = []
                for result in batch:
                    result_id = result.get("id", "")
                    payload = result.get("payload", {})
                    score = result.get("score", 0.0)
                    scored.append(
                        ScoredResult(
                            id=UUID(str(result_id)) if result_id else UUID(int=0),
                            # RuVector returns cosine distance (0=identical, 1=orthogonal); invert to similarity
                            score=1.0 - score,
                            payload={**payload, "id": str(result_id)},
                        )
                    )
                all_scored.append(scored)
            return all_scored
        except Exception as exc:
            logger.error("RuVector batch_search error: %s", exc, exc_info=True)
            raise  # Let Cognee handle the failure — don't hide a dead backend
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def prune(self) -> None:
        """Delete all collections managed by this adapter."""
        client = self._client()
        try:
            collections = await client.list_collections()
            for name in collections:
                # RuVector doesn't expose a delete-collection endpoint through
                # the basic REST surface; log and skip.
                logger.warning(
                    "RuVector prune: skipping collection %s (no delete-collection API)",
                    name,
                )
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Optional interface methods (pass-through defaults)
    # ------------------------------------------------------------------

    async def create_vector_index(
        self, index_name: str, index_property_name: str
    ) -> None:
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self,
        index_name: str,
        index_property_name: str,
        data_points: list[DataPoint],
    ) -> None:
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                _IndexSchema(
                    id=dp.id,
                    text=getattr(dp, dp.metadata["index_fields"][0]),
                )
                for dp in data_points
            ],
        )
