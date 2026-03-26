"""High-level tool functions for the Haystack Agent.

Each ``make_*`` factory returns a ``ToolConfig`` dict with ``name``,
``description``, ``function`` (sync callable), and ``parameters``
(JSON Schema).  Haystack's ``Tool`` class expects synchronous callables,
so internal async operations (Cognee, Postgres) are bridged via
``_run_async`` using a dedicated background event loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from typing import Any, TypedDict

from haystack import Document

from agentic_ra.core.errors import ToolError
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter, RetrievalStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool config type
# ---------------------------------------------------------------------------


class ToolConfig(TypedDict):
    """Descriptor returned by each ``make_*`` tool factory."""

    name: str
    description: str
    function: Any  # Callable — can't be more specific due to varying signatures
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Module-level evidence accumulator — reset per agent run
# ---------------------------------------------------------------------------

_evidence_buffer: contextvars.ContextVar[list[Document]] = contextvars.ContextVar(
    "_evidence_buffer", default=[]
)


def reset_evidence_buffer() -> None:
    """Reset evidence buffer for this request context."""
    _evidence_buffer.set([])


def get_evidence_buffer() -> list[Document]:
    """Get evidence documents accumulated in this request context."""
    return list(_evidence_buffer.get())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_EVIDENCE_CHARS = 4000
"""Soft cap on formatted evidence text sent back to the LLM."""

_TOOL_LOOP: asyncio.AbstractEventLoop | None = None
_TOOL_THREAD: threading.Thread | None = None
_TOOL_LOOP_LOCK = threading.Lock()


def _get_tool_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for tool async operations."""
    global _TOOL_LOOP, _TOOL_THREAD  # noqa: PLW0603
    with _TOOL_LOOP_LOCK:
        if (
            _TOOL_LOOP is None
            or _TOOL_LOOP.is_closed()
            or (_TOOL_THREAD is not None and not _TOOL_THREAD.is_alive())
        ):
            if _TOOL_THREAD is not None and not _TOOL_THREAD.is_alive():
                logger.warning("Tool event loop thread died — recreating")
            _TOOL_LOOP = asyncio.new_event_loop()
            _TOOL_THREAD = threading.Thread(
                target=_TOOL_LOOP.run_forever, daemon=True, name="tool-event-loop"
            )
            _TOOL_THREAD.start()
    return _TOOL_LOOP


def _run_async(coro: Any) -> Any:
    """Bridge async to sync using a dedicated background event loop.

    When called from inside an existing event loop (e.g. FastAPI/uvicorn),
    uses a persistent background thread+loop to avoid spawning a new thread
    per call. When no loop is running, falls back to ``asyncio.run``.
    """
    try:
        asyncio.get_running_loop()
        # Inside an existing loop (FastAPI) — use dedicated background loop
        loop = _get_tool_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=120)
    except RuntimeError:
        # No running loop — safe to use asyncio.run
        return asyncio.run(coro)


def _format_evidence(documents: list[Document] | tuple[Document, ...]) -> str:
    """Format a list of Documents as numbered, readable text for the LLM.

    Each entry includes the source metadata (if present) and the content,
    truncated to ``_MAX_EVIDENCE_CHARS`` total to stay within token budgets.
    """
    if not documents:
        return "No documents found."

    parts: list[str] = []
    total_chars = 0

    for idx, doc in enumerate(documents, start=1):
        source = doc.meta.get("source", "unknown")
        score = doc.meta.get("rrf_score", doc.meta.get("score", ""))
        header = f"[{idx}] (source={source}"
        if score:
            header += (
                f", score={score:.4f}"
                if isinstance(score, float)
                else f", score={score}"
            )
        header += ")"

        content = doc.content or "(empty)"
        entry = f"{header}\n{content}"

        if total_chars + len(entry) > _MAX_EVIDENCE_CHARS:
            remaining = _MAX_EVIDENCE_CHARS - total_chars
            if remaining > 80:
                parts.append(entry[:remaining] + "\n... (truncated)")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(parts)


def _search_dataset(
    cognee_store: Any,
    dataset: str,
    query: str,
    strategy: RetrievalStrategy,
    limit: int = 10,
) -> list[Document]:
    """Search a specific Cognee dataset using the given strategy.

    Maps each :class:`RetrievalStrategy` to one or more Cognee search types,
    then runs them sequentially against *dataset* and merges the results.

    Args:
        cognee_store: CogneeStore instance (typed ``Any`` to avoid circular import).
        dataset: Cognee dataset name (e.g. ``"user_42"``).
        query: The search query text.
        strategy: Which retrieval strategy to use.
        limit: Maximum results per search type.

    Returns:
        A flat list of :class:`Document` results across all search types.
    """
    search_types: dict[RetrievalStrategy, list[str]] = {
        RetrievalStrategy.SEMANTIC: ["CHUNKS"],
        RetrievalStrategy.KEYWORD: ["CHUNKS_LEXICAL"],
        RetrievalStrategy.GRAPH: ["GRAPH_COMPLETION"],
        RetrievalStrategy.HYBRID: ["CHUNKS", "CHUNKS_LEXICAL"],
        RetrievalStrategy.GRAPH_HYBRID: [
            "GRAPH_COMPLETION",
            "CHUNKS",
            "CHUNKS_LEXICAL",
        ],
    }
    types = search_types.get(strategy, ["CHUNKS", "CHUNKS_LEXICAL"])

    all_results: list[Document] = []
    for st in types:
        results: list[Document] = _run_async(
            cognee_store.search(
                query=query, search_type=st, top_k=limit, dataset=dataset
            )
        )
        all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def make_search_knowledge_tool(
    router: RetrievalRouter,
    ranker: ResultRanker,
    cognee_store: Any = None,
    document_store: Any = None,
    dataset: str = "user_anonymous",
    role: str = "user",
) -> ToolConfig:
    """Create the ``search_knowledge`` tool for the Haystack Agent.

    Searches the knowledge base using per-user dataset scoping.  Regular
    users search only their own dataset; admins fan-out across **all**
    user datasets.

    Args:
        router: Retrieval router (kept for backward compatibility).
        ranker: Result ranker for dedup and RRF fusion.
        cognee_store: CogneeStore instance for direct dataset-scoped search.
        document_store: Document store with ``list_user_datasets()`` (admin fan-out).
        dataset: Cognee dataset name for the current user (e.g. ``"user_42"``).
        role: ``"user"`` or ``"admin"``.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def search_knowledge(query: str, strategy: str = "hybrid") -> dict[str, Any]:
        """Search the knowledge base and return ranked evidence."""
        try:
            retrieval_strategy = RetrievalStrategy(strategy.lower())
        except ValueError:
            valid = ", ".join(s.value for s in RetrievalStrategy)
            raise ToolError(
                f"Invalid retrieval strategy: {strategy!r}",
                valid_strategies=valid,
                operation="search_knowledge",
            )

        try:
            if cognee_store is not None:
                if role == "admin" and document_store is not None:
                    # Admin fan-out: search all user datasets
                    user_ids: list[str] = _run_async(
                        document_store.list_user_datasets()
                    )
                    all_results: list[Document] = []
                    for uid in user_ids:
                        ds = f"user_{uid}"
                        results = _search_dataset(
                            cognee_store, ds, query, retrieval_strategy, limit=10
                        )
                        all_results.extend(results)
                    raw_docs = all_results
                else:
                    # User: search only their dataset
                    raw_docs = _search_dataset(
                        cognee_store, dataset, query, retrieval_strategy, limit=10
                    )
            else:
                # Fallback: use router (no dataset scoping)
                raw_docs = _run_async(
                    router.retrieve(query=query, strategy=retrieval_strategy)
                )

            evidence_pack = ranker.rank(
                results=raw_docs,
                query=query,
                strategy=retrieval_strategy.value,
            )
            formatted = _format_evidence(evidence_pack.documents)
            buf = _evidence_buffer.get()
            buf.extend(evidence_pack.documents)
            _evidence_buffer.set(buf)

            logger.info(
                "search_knowledge returned %d documents (strategy=%s)",
                len(evidence_pack.documents),
                strategy,
            )
            return {
                "text": formatted,
                "documents": list(evidence_pack.documents),
            }
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Knowledge search failed",
                query=query,
                strategy=strategy,
                operation="search_knowledge",
                original_error=str(e),
            ) from e

    return {
        "name": "search_knowledge",
        "description": (
            "Search the internal knowledge base using semantic, keyword, hybrid, "
            "graph, or graph_hybrid retrieval. Returns ranked evidence documents."
        ),
        "function": search_knowledge,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "strategy": {
                    "type": "string",
                    "description": (
                        "Retrieval strategy: semantic, keyword, hybrid, graph, "
                        "or graph_hybrid."
                    ),
                    "enum": [s.value for s in RetrievalStrategy],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    }


def make_search_web_tool() -> ToolConfig:
    """Create the ``search_web`` tool stub.

    Web search integration is not yet implemented. The tool returns a
    placeholder response indicating the capability is unavailable.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def search_web(query: str) -> dict[str, str]:
        """Search the web for information (not yet implemented)."""
        logger.warning("search_web called but not yet configured (query=%s)", query)
        return {"text": "Web search not yet configured."}

    return {
        "name": "search_web",
        "description": (
            "Search the web for information not available in the knowledge base. "
            "(Not yet configured -- returns a placeholder.)"
        ),
        "function": search_web,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query text.",
                },
            },
            "required": ["query"],
        },
    }


def make_ingest_tool(
    cognee_store: Any,  # CogneeStore -- typed as Any to avoid circular import
    dataset: str = "user_anonymous",
) -> ToolConfig:
    """Create the ``ingest_document`` tool for the Haystack Agent.

    Ingests text into the Cognee knowledge graph via ``add_and_cognify``,
    scoped to the given *dataset* (typically ``"user_{user_id}"``).

    Args:
        cognee_store: CogneeStore instance for knowledge graph operations.
        dataset: Cognee dataset name for the current user.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def ingest_document(text: str) -> dict[str, str]:
        """Ingest text into the knowledge graph."""
        if not text.strip():
            raise ToolError(
                "Cannot ingest empty text",
                operation="ingest_document",
            )

        try:
            _run_async(cognee_store.add_and_cognify(text=text, dataset=dataset))

            logger.info(
                "ingest_document ingested %d characters (dataset=%s)",
                len(text),
                dataset,
            )
            return {"text": f"Successfully ingested {len(text)} characters."}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Document ingestion failed",
                text_length=len(text),
                operation="ingest_document",
                original_error=str(e),
            ) from e

    return {
        "name": "ingest_document",
        "description": (
            "Ingest text into the knowledge graph. The text is chunked, "
            "entities and relations are extracted, and the graph is updated."
        ),
        "function": ingest_document,
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The raw text content to ingest.",
                },
            },
            "required": ["text"],
        },
    }


def make_remember_tool(
    memory_store: Any,  # MemoryStore -- typed as Any to avoid circular import
    user_id: str,
) -> ToolConfig:
    """Create the ``remember`` tool for the Haystack Agent.

    Provides save, get, and list operations on long-term user memory.
    The ``user_id`` is captured via closure so the LLM cannot impersonate
    other users.

    Args:
        memory_store: MemoryStore instance for persistent memory.
        user_id: User identifier bound into the closure.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def remember(action: str, key: str = "", value: str = "") -> dict[str, str]:
        """Save, retrieve, or list long-term memories."""
        action_lower = action.lower()

        try:
            if action_lower == "save":
                if not key:
                    raise ToolError(
                        "Memory 'save' requires a key",
                        operation="remember",
                    )
                _run_async(
                    memory_store.save_memory(user_id=user_id, key=key, value=value)
                )
                logger.info("remember: saved key=%s for user=%s", key, user_id)
                return {"text": f"Saved memory '{key}'."}

            if action_lower == "get":
                if not key:
                    raise ToolError(
                        "Memory 'get' requires a key",
                        operation="remember",
                    )
                entry = _run_async(memory_store.get_memory(user_id=user_id, key=key))
                if entry is None:
                    return {"text": f"No memory found for key '{key}'."}
                return {"text": f"{entry.key}: {entry.value}"}

            if action_lower == "list":
                entries = _run_async(memory_store.get_memories(user_id=user_id))
                if not entries:
                    return {"text": "No memories stored."}
                lines = [f"- {e.key}: {e.value}" for e in entries]
                return {"text": "\n".join(lines)}

            valid_actions = "save, get, list"
            raise ToolError(
                f"Invalid memory action: {action!r}",
                valid_actions=valid_actions,
                operation="remember",
            )

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Memory operation failed",
                action=action,
                key=key,
                user_id=user_id,
                operation="remember",
                original_error=str(e),
            ) from e

    return {
        "name": "remember",
        "description": (
            "Save, retrieve, or list long-term memories for the current user. "
            "Use action='save' to store a fact, 'get' to retrieve by key, "
            "or 'list' to show all stored memories."
        ),
        "function": remember,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Memory operation: save, get, or list.",
                    "enum": ["save", "get", "list"],
                },
                "key": {
                    "type": "string",
                    "description": "Memory key (required for save and get).",
                    "default": "",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (required for save).",
                    "default": "",
                },
            },
            "required": ["action"],
        },
    }
