"""Async OpenSearch client wrapper with error handling and index management."""

from __future__ import annotations

from typing import Any  # opensearchpy accepts untyped dict for index body and search body

from haystack import Document
from opensearchpy import AsyncOpenSearch
from pydantic import SecretStr

from agentic_ra.core.errors import OpenSearchError


class OpenSearchStore:
    """Async OpenSearch client wrapper.

    Args:
        host: OpenSearch hostname.
        port: OpenSearch HTTP port.
        user: OpenSearch username for basic auth.
        password: OpenSearch password as ``SecretStr`` for safe handling.
        verify_certs: Whether to verify SSL certificates.
            Defaults to ``False`` for dev (self-signed certs).
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: SecretStr,
        verify_certs: bool = False,
    ) -> None:
        self._client = AsyncOpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password.get_secret_value()),
            use_ssl=True,
            verify_certs=verify_certs,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

    async def ensure_index(
        self,
        name: str,
        body: dict[str, Any] | None = None,
    ) -> None:
        """Create an index if it does not already exist.

        Args:
            name: Index name.
            body: Optional index settings and mappings.
        """
        try:
            exists: bool = await self._client.indices.exists(index=name)
            if exists:
                return
            await self._client.indices.create(
                index=name,
                body=body or {},
            )
        except Exception as e:
            raise OpenSearchError(
                "Failed to ensure index",
                index=name,
                operation="ensure_index",
                original_error=str(e),
            ) from e

    async def search(
        self,
        index: str,
        query: str,
        limit: int = 10,
    ) -> list[Document]:
        """BM25 keyword search returning Haystack Documents.

        Args:
            index: OpenSearch index name.
            query: Search query text.
            limit: Maximum results to return.

        Returns:
            Documents with score and source in metadata.
        """
        try:
            response: dict[str, Any] = await self._client.search(
                index=index,
                body={
                    "query": {"match": {"content": query}},
                    "size": limit,
                },
            )
            hits: list[dict[str, Any]] = response["hits"]["hits"]
            return [
                Document(
                    content=hit["_source"].get("content", ""),
                    meta={
                        "score": hit["_score"],
                        "source": "opensearch",
                        **{k: v for k, v in hit["_source"].items() if k != "content"},
                    },
                )
                for hit in hits
            ]
        except Exception as e:
            raise OpenSearchError(
                "Keyword search failed",
                index=index,
                operation="search",
                original_error=str(e),
            ) from e

    async def close(self) -> None:
        """Close the underlying client connection."""
        try:
            await self._client.close()
        except Exception as e:
            raise OpenSearchError(
                "Failed to close client",
                operation="close",
                original_error=str(e),
            ) from e
