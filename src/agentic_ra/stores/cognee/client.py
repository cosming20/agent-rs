"""Cognee knowledge graph store — entity extraction, graph building, and graph-augmented search."""

from __future__ import annotations

import logging
from typing import Any

import cognee  # type: ignore[import-untyped]  # cognee has no py.typed marker
from cognee.api.v1.search import SearchType  # type: ignore[import-untyped]
from haystack import Document

from agentic_ra.core.errors import CogneeError

logger = logging.getLogger(__name__)


class CogneeStore:
    """Wrapper around the Cognee SDK for knowledge graph operations.

    Provides three capabilities:
    1. **add** — ingest text into a Cognee dataset.
    2. **cognify** — extract entities/relations and build the knowledge graph.
    3. **search** — graph-augmented retrieval (GRAPH_COMPLETION, CHUNKS, TRIPLETS, etc.).

    Cognee is configured programmatically at startup via ``configure()``.

    Args:
        dataset: Default Cognee dataset name.
    """

    def __init__(self, dataset: str = "agentic_rag") -> None:
        self._dataset = dataset

    def configure(
        self,
        *,
        llm_api_key: str,
        llm_model: str,
        graph_db_provider: str = "neo4j",
        vector_db_provider: str = "qdrant",
        data_root: str | None = None,
        system_root: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str = "",
        neo4j_bolt_url: str | None = None,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        neo4j_db: str = "neo4j",
    ) -> None:
        """Configure Cognee runtime settings programmatically.

        Must be called before any add/cognify/search operations.
        By default uses our existing Qdrant for vectors and Neo4j for graph.

        Args:
            llm_api_key: OpenAI (or other provider) API key.
            llm_model: LLM model for entity extraction (e.g. ``openai/gpt-4o-mini``).
            graph_db_provider: Graph backend (neo4j, kuzu, neptune).
            vector_db_provider: Vector backend (qdrant, lancedb, chromadb, pgvector).
            data_root: Root directory for Cognee data storage.
            system_root: Root directory for Cognee system files.
            qdrant_url: Qdrant HTTP URL (reuses our existing instance).
            qdrant_api_key: Qdrant API key.
            neo4j_bolt_url: Neo4j Bolt URL (e.g. bolt://localhost:1061).
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            neo4j_db: Neo4j database name.
        """
        try:
            # Register Qdrant community adapter if using qdrant.
            # Cognee 0.5.5 passes `database_name` to adapter __init__, but the
            # community adapter v0.1.0 doesn't accept it. We wrap the adapter
            # to absorb the extra kwarg until upstream fixes the mismatch.
            if vector_db_provider == "qdrant":
                from cognee_community_vector_adapter_qdrant.register import (  # type: ignore[import-untyped]
                    QDrantAdapter as _BaseQDrantAdapter,
                    use_vector_adapter,
                )

                class _CompatQDrantAdapter(_BaseQDrantAdapter):  # type: ignore[misc]
                    """Shim: absorb extra kwargs Cognee 0.5.5 passes to adapter v0.1.0.

                    - ``__init__``: ignores ``database_name``
                    - ``search``: ignores ``include_payload``, ``node_name``
                    """

                    def __init__(  # type: ignore[no-untyped-def]
                        self, *args: Any, database_name: str = "", **kwargs: Any
                    ) -> None:
                        super().__init__(*args, **kwargs)

                    async def search(  # type: ignore[no-untyped-def]
                        self,
                        *args: Any,
                        include_payload: bool = True,
                        node_name: str | None = None,
                        **kwargs: Any,
                    ) -> list[Any]:
                        return await super().search(*args, **kwargs)

                use_vector_adapter("qdrant", _CompatQDrantAdapter)

            # LLM config
            cognee.config.set_llm_api_key(llm_api_key)
            cognee.config.set_llm_model(llm_model)

            # Vector DB config (reuse our Qdrant)
            vector_config: dict[str, object] = {
                "vector_db_provider": vector_db_provider,
            }
            if qdrant_url:
                vector_config["vector_db_url"] = qdrant_url
            if qdrant_api_key:
                vector_config["vector_db_key"] = qdrant_api_key
            cognee.config.set_vector_db_config(vector_config)

            # Graph DB config (Neo4j)
            graph_config: dict[str, object] = {
                "graph_database_provider": graph_db_provider,
            }
            if neo4j_bolt_url:
                graph_config["graph_database_url"] = neo4j_bolt_url
                graph_config["graph_database_username"] = neo4j_user
                graph_config["graph_database_password"] = neo4j_password
                graph_config["graph_database_name"] = neo4j_db
            cognee.config.set_graph_db_config(graph_config)

            # Storage directories
            if data_root:
                cognee.config.data_root_directory(data_root)
            if system_root:
                cognee.config.system_root_directory(system_root)

            logger.info(
                "Cognee configured: graph=%s, vector=%s, model=%s, qdrant=%s, neo4j=%s",
                graph_db_provider,
                vector_db_provider,
                llm_model,
                qdrant_url or "default",
                neo4j_bolt_url or "default",
            )
        except Exception as e:
            raise CogneeError(
                "Failed to configure Cognee",
                operation="configure",
                original_error=str(e),
            ) from e

    async def add_and_cognify(
        self,
        text: str,
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """Ingest text and build knowledge graph in one step.

        Args:
            text: Raw text content to ingest.
            dataset: Override dataset name (defaults to instance default).

        Returns:
            Dict with keys ``add_result`` and ``cognify_result``.
        """
        ds = dataset or self._dataset
        try:
            add_result = await cognee.add(data=text, dataset_name=ds)
            cognify_result = await cognee.cognify(datasets=[ds])

            logger.info(
                "Cognee add+cognify complete for dataset=%s",
                ds,
            )
            return {"add_result": add_result, "cognify_result": cognify_result}

        except Exception as e:
            raise CogneeError(
                "Cognee add+cognify failed",
                dataset=ds,
                operation="add_and_cognify",
                original_error=str(e),
            ) from e

    async def add_chunks(
        self,
        chunks: list[str],
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """Ingest individual chunks and build knowledge graph.

        Each chunk is added as a separate document to preserve structure
        (e.g. article boundaries in legal documents). Cognify runs once
        after all chunks are added.

        Args:
            chunks: List of chunk texts (each should have heading breadcrumbs).
            dataset: Override dataset name (defaults to instance default).

        Returns:
            Dict with keys ``add_count`` and ``cognify_result``.
        """
        ds = dataset or self._dataset
        if not chunks:
            raise CogneeError(
                "No chunks to ingest",
                dataset=ds,
                operation="add_chunks",
                original_error="Empty chunk list",
            )

        try:
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                await cognee.add(data=chunk_text, dataset_name=ds)

                if (i + 1) % 50 == 0:
                    logger.info(
                        "Cognee add progress: %d/%d chunks (dataset=%s)",
                        i + 1,
                        len(chunks),
                        ds,
                    )

            logger.info(
                "Cognee add complete: %d chunks (dataset=%s), starting cognify...",
                len(chunks),
                ds,
            )

            cognify_result = await cognee.cognify(datasets=[ds])

            logger.info("Cognee cognify complete for dataset=%s", ds)
            return {"add_count": len(chunks), "cognify_result": cognify_result}

        except CogneeError:
            raise
        except Exception as e:
            raise CogneeError(
                "Cognee chunk ingestion failed",
                dataset=ds,
                chunk_count=len(chunks),
                operation="add_chunks",
                original_error=str(e),
            ) from e

    async def add(self, text: str, dataset: str | None = None) -> object:
        """Ingest text into Cognee without building the graph.

        Useful for batching: add many documents, then cognify once.

        Args:
            text: Raw text content.
            dataset: Override dataset name.

        Returns:
            Cognee pipeline run info.
        """
        ds = dataset or self._dataset
        try:
            return await cognee.add(data=text, dataset_name=ds)
        except Exception as e:
            raise CogneeError(
                "Cognee add failed",
                dataset=ds,
                operation="add",
                original_error=str(e),
            ) from e

    async def cognify(self, dataset: str | None = None) -> object:
        """Build/update the knowledge graph for a dataset.

        Args:
            dataset: Override dataset name.

        Returns:
            Cognee cognify result.
        """
        ds = dataset or self._dataset
        try:
            return await cognee.cognify(datasets=[ds])
        except Exception as e:
            raise CogneeError(
                "Cognee cognify failed",
                dataset=ds,
                operation="cognify",
                original_error=str(e),
            ) from e

    async def prune(self) -> None:
        """Reset Cognee knowledge base.

        Removes all data from graph and vector stores. Use with caution — this
        is irreversible and affects all datasets managed by this Cognee instance.
        """
        try:
            # Cognee 0.5.5: prune() is a class, not a coroutine
            result = cognee.prune()
            if hasattr(result, "__await__"):
                await result
            logger.info("Cognee knowledge base pruned")
        except Exception as e:
            raise CogneeError(
                "Cognee prune failed",
                operation="prune",
                original_error=str(e),
            ) from e

    async def search(
        self,
        query: str,
        search_type: str = "GRAPH_COMPLETION",
        top_k: int = 10,
        dataset: str | None = None,
        session_id: str | None = None,
    ) -> list[Document]:
        """Search the knowledge graph and return Haystack Documents.

        Uses ``only_context=False`` so Cognee returns individual result
        objects (dicts with ``text``, ``id``, etc.) instead of a single
        concatenated string.  This preserves per-result granularity for
        ranking, evidence attribution, and prevents context-length overflow.

        Args:
            query: Search query text.
            search_type: Cognee SearchType value (GRAPH_COMPLETION, CHUNKS,
                TRIPLET_COMPLETION, SUMMARIES, etc.).
            top_k: Maximum results.
            dataset: Override dataset name.
            session_id: Optional session identifier for Cognee feedback tracking.
                When provided, Cognee records which graph elements were used,
                enabling downstream feedback loops to reinforce relevant nodes.

        Returns:
            List of Haystack Documents with graph search results.
        """
        ds = dataset or self._dataset
        try:
            st = SearchType(search_type)

            search_kwargs: dict[str, object] = {
                "query_text": query,
                "query_type": st,
                "datasets": [ds],
                "top_k": top_k,
                "only_context": False,
            }
            if session_id:
                search_kwargs["session_id"] = session_id

            results = await cognee.search(**search_kwargs)

            documents: list[Document] = []
            for i, result in enumerate(results or []):
                content = self._extract_content(result)
                if content:
                    documents.append(
                        Document(
                            content=content,
                            meta={
                                "source": f"cognee_{search_type.lower()}",
                                "search_type": search_type,
                                "score": 1.0 / (i + 1),
                                "dataset": ds,
                                "rank": i,
                            },
                        )
                    )

            logger.info(
                "Cognee search returned %d documents (type=%s, dataset=%s)",
                len(documents),
                search_type,
                ds,
            )
            return documents

        except Exception as e:
            raise CogneeError(
                "Cognee search failed",
                query=query,
                search_type=search_type,
                dataset=ds,
                operation="search",
                original_error=str(e),
            ) from e

    @staticmethod
    def _extract_content(result: object) -> str:
        """Extract text content from a Cognee search result.

        With ``only_context=False``, Cognee returns dicts for CHUNKS/LEXICAL
        (with ``text`` key) and structured objects for GRAPH_COMPLETION.
        This normalizes all shapes to plain text.
        """
        # Dict results (CHUNKS, CHUNKS_LEXICAL with only_context=False)
        if isinstance(result, dict):
            return str(result.get("text", result.get("content", "")))

        if isinstance(result, str):
            return result

        # SearchResult objects have a .payload or .text attribute
        if hasattr(result, "payload"):
            payload = result.payload
            if isinstance(payload, dict):
                return str(payload.get("text", payload.get("content", str(payload))))
            return str(payload)

        if hasattr(result, "text"):
            return str(result.text)

        if hasattr(result, "content"):
            return str(result.content)

        # Fallback: stringify
        return str(result)
