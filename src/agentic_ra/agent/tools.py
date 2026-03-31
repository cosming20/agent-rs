"""High-level tool functions for the Haystack Agent.

Each ``make_*`` factory returns a ``ToolConfig`` dict with ``name``,
``description``, ``function`` (sync callable), and ``parameters``
(JSON Schema).  Haystack's ``Tool`` class expects synchronous callables,
so internal async operations (Cognee, Postgres) are bridged via
``_run_async`` using a dedicated background event loop.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any, TypedDict

from haystack import Document

from agentic_ra.core.errors import ToolError, WebBudgetExhaustedError
from agentic_ra.retrieval.ranker import ResultRanker
from agentic_ra.retrieval.router import RetrievalRouter, RetrievalStrategy
from agentic_ra.tools.budgets import WebToolBudget
from agentic_ra.tools.mcp_client import ToolResult
from agentic_ra.tools.mcp_http_client import MCPHttpClient
from agentic_ra.tools.url_safety import URLValidator
from agentic_ra.tools.web import fetch_url

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
# Evidence accumulator — backward-compat stubs
# ---------------------------------------------------------------------------
# The ContextVar approach was broken across thread boundaries (tools run in
# the _TOOL_LOOP thread, not the request thread).  Evidence is now passed via
# an explicit ``evidence_accumulator`` list through the tool closure.
# These functions remain as no-ops for any call sites that still reference them.


def reset_evidence_buffer() -> None:  # noqa: D401
    """No-op (kept for backward compatibility)."""


def get_evidence_buffer() -> list[Document]:  # noqa: D401
    """No-op — returns empty list (kept for backward compatibility)."""
    return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_EVIDENCE_CHARS = 4000
"""Soft cap on formatted evidence text sent back to the LLM."""

_MAX_USER_ID_LEN = 128
"""Maximum length for sanitized user IDs in dataset names."""

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

    Always uses the persistent ``_TOOL_LOOP`` thread so that all async
    operations (Cognee search, Neo4j graph queries, Qdrant vector search)
    share a single event loop.  This prevents "Future attached to a
    different loop" errors from Neo4j/asyncio drivers.
    """
    loop = _get_tool_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


def _sanitize_user_id(user_id: str) -> str:
    """Sanitize user_id for use in Cognee dataset names.

    Strips any characters that are not alphanumeric, underscore, or hyphen
    to prevent injection into dataset names.

    Args:
        user_id: Raw user identifier.

    Returns:
        Sanitized string safe for dataset name construction.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", user_id)
    sanitized = sanitized[:_MAX_USER_ID_LEN]
    return sanitized or "anonymous"


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
    session_id: str | None = None,
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
        session_id: Cognee session ID for feedback tracking.

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
                query=query, search_type=st, top_k=limit, dataset=dataset,
                session_id=session_id,
            )
        )
        all_results.extend(results)
    return all_results


def _parse_brave_results(raw_content: str, provider: str) -> list[Document]:
    """Parse Brave Search MCP text results into Documents.

    Brave MCP returns results as text content. Each result typically
    includes title, URL, and description. We parse these into Documents
    with structured metadata.

    Args:
        raw_content: Raw text content from Brave MCP tool result.
        provider: Provider tag for metadata (``"brave_web"`` or ``"brave_news"``).

    Returns:
        List of Documents with source and provider metadata.
    """
    if not raw_content.strip():
        return []

    documents: list[Document] = []
    blocks = [b.strip() for b in raw_content.split("\n\n") if b.strip()]

    for idx, block in enumerate(blocks):
        url = ""
        for line in block.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith(("http://", "https://")):
                url = line_stripped
                break

        documents.append(
            Document(
                content=block,
                meta={
                    "source": url or f"{provider}_result_{idx}",
                    "provider": provider,
                    "score": 1.0 / (idx + 1),
                },
            )
        )

    return documents


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
    evidence_accumulator: list[Document] | None = None,
    session_id: str | None = None,
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
        evidence_accumulator: Shared list to collect KB evidence documents.
        session_id: Cognee session ID for feedback tracking.

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
                    # Cap to prevent DoS on large deployments
                    MAX_FAN_OUT = 50
                    if len(user_ids) > MAX_FAN_OUT:
                        logger.warning(
                            "Admin fan-out capped at %d users (total: %d)",
                            MAX_FAN_OUT,
                            len(user_ids),
                        )
                        user_ids = user_ids[:MAX_FAN_OUT]

                    all_results: list[Document] = []
                    for uid in user_ids:
                        ds = f"user_{_sanitize_user_id(uid)}"
                        try:
                            results = _search_dataset(
                                cognee_store, ds, query, retrieval_strategy,
                                limit=10, session_id=session_id,
                            )
                            all_results.extend(results)
                        except Exception:
                            logger.warning(
                                "Admin fan-out: search failed for dataset %s", ds
                            )
                    raw_docs = all_results
                else:
                    # User: search only their dataset
                    raw_docs = _search_dataset(
                        cognee_store, dataset, query, retrieval_strategy,
                        limit=10, session_id=session_id,
                    )
            else:
                # Fallback: use router (no dataset scoping)
                raw_docs = _run_async(
                    router.retrieve(
                        query=query, strategy=retrieval_strategy,
                        session_id=session_id,
                    )
                )

            evidence_pack = ranker.rank(
                results=raw_docs,
                query=query,
                strategy=retrieval_strategy.value,
            )
            formatted = _format_evidence(evidence_pack.documents)
            if evidence_accumulator is not None:
                evidence_accumulator.extend(evidence_pack.documents)

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


def make_search_web_tool(
    brave_client: MCPHttpClient | None = None,
    evidence_accumulator: list[Document] | None = None,
    budget: WebToolBudget | None = None,
) -> ToolConfig:
    """Create the ``search_web`` tool for the Haystack Agent.

    Searches the web using Brave Search via MCP Streamable HTTP.
    If Brave is not configured (brave_client is None), returns a
    graceful degradation message.

    Args:
        brave_client: MCPHttpClient connected to Brave Search MCP server.
        evidence_accumulator: Shared list to collect web evidence documents.
        budget: Per-query web tool budget.

    Returns:
        Tool descriptor dict.
    """

    def search_web(
        query: str, country: str = "US", freshness: str = ""
    ) -> dict[str, Any]:
        """Search the web for information using Brave Search."""
        if brave_client is None:
            logger.warning("search_web called but Brave Search is not configured")
            return {"text": "Web search is not configured."}

        if budget is not None:
            try:
                budget.check_web_search()
            except WebBudgetExhaustedError as e:
                return {"text": str(e)}

        try:
            args: dict[str, Any] = {"query": query, "count": 5, "country": country}
            if freshness:
                args["freshness"] = freshness

            result: ToolResult = _run_async(
                brave_client.call_tool("brave_web_search", args)
            )

            # Record budget FIRST — the call was made regardless of result
            if budget is not None:
                budget.record_web_search()

            if result.is_error:
                raise ToolError(
                    "Brave web search returned error",
                    query=query,
                    content=result.content[:200],
                )

            documents = _parse_brave_results(result.content, provider="brave_web")
            if evidence_accumulator is not None:
                evidence_accumulator.extend(documents)

            formatted = _format_evidence(documents)
            logger.info(
                "search_web returned %d results (query=%s)", len(documents), query
            )
            return {"text": formatted, "documents": documents}

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Web search failed",
                query=query,
                operation="search_web",
                original_error=str(e),
            ) from e

    return {
        "name": "search_web",
        "description": (
            "Search the web for information not available in the knowledge base. "
            "Use for evergreen topics and general queries."
        ),
        "function": search_web,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query text.",
                },
                "country": {
                    "type": "string",
                    "description": "Country code for search results (default US).",
                    "default": "US",
                },
                "freshness": {
                    "type": "string",
                    "description": (
                        "Time filter: 'pd' (past day), 'pw' (past week), "
                        "'pm' (past month), 'py' (past year), or empty for all time."
                    ),
                    "default": "",
                },
            },
            "required": ["query"],
        },
    }


def make_search_news_tool(
    brave_client: MCPHttpClient | None = None,
    evidence_accumulator: list[Document] | None = None,
    budget: WebToolBudget | None = None,
) -> ToolConfig:
    """Create the ``search_news`` tool for the Haystack Agent.

    Searches recent news using Brave News Search via MCP.
    Better fit for recency-sensitive queries than search_web.

    Args:
        brave_client: MCPHttpClient connected to Brave Search MCP server.
        evidence_accumulator: Shared list to collect web evidence documents.
        budget: Per-query web tool budget.

    Returns:
        Tool descriptor dict.
    """

    def search_news(query: str, freshness: str = "pw") -> dict[str, Any]:
        """Search recent news articles using Brave News Search."""
        if brave_client is None:
            logger.warning("search_news called but Brave Search is not configured")
            return {"text": "News search is not configured."}

        if budget is not None:
            try:
                budget.check_news_search()
            except WebBudgetExhaustedError as e:
                return {"text": str(e)}

        try:
            args: dict[str, Any] = {"query": query, "count": 5, "freshness": freshness}

            result: ToolResult = _run_async(
                brave_client.call_tool("brave_news_search", args)
            )

            # Record budget FIRST — the call was made regardless of result
            if budget is not None:
                budget.record_news_search()

            if result.is_error:
                raise ToolError(
                    "Brave news search returned error",
                    query=query,
                    content=result.content[:200],
                )

            documents = _parse_brave_results(result.content, provider="brave_news")
            if evidence_accumulator is not None:
                evidence_accumulator.extend(documents)

            formatted = _format_evidence(documents)
            logger.info(
                "search_news returned %d results (query=%s)", len(documents), query
            )
            return {"text": formatted, "documents": documents}

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "News search failed",
                query=query,
                operation="search_news",
                original_error=str(e),
            ) from e

    return {
        "name": "search_news",
        "description": (
            "Search recent news articles. Use for queries about recent events, "
            "releases, changes, or time-sensitive topics."
        ),
        "function": search_news,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The news search query text.",
                },
                "freshness": {
                    "type": "string",
                    "description": (
                        "Time filter: 'pd' (past day), 'pw' (past week), "
                        "'pm' (past month). Default is 'pw' (past week)."
                    ),
                    "default": "pw",
                },
            },
            "required": ["query"],
        },
    }


def make_fetch_url_tool(
    url_validator: URLValidator,
    evidence_accumulator: list[Document] | None = None,
    budget: WebToolBudget | None = None,
    max_length: int = 5000,
    max_bytes: int = 2_000_000,
    timeout: int = 30,
    user_agent: str = "AgenticRA/1.0",
) -> ToolConfig:
    """Create the ``fetch_url`` tool for the Haystack Agent.

    Fetches a URL, extracts main content via trafilatura, and returns
    clean text. All URLs are SSRF-validated before any network I/O.

    Args:
        url_validator: URLValidator for SSRF protection.
        evidence_accumulator: Shared list to collect web evidence documents.
        budget: Per-query web tool budget.
        max_length: Maximum characters in returned content.
        max_bytes: Maximum bytes to download.
        timeout: HTTP request timeout in seconds.
        user_agent: User-Agent header value.

    Returns:
        Tool descriptor dict.
    """

    def fetch(url: str) -> dict[str, Any]:
        """Fetch a URL and extract its main content."""
        if budget is not None:
            try:
                budget.check_fetch()
            except WebBudgetExhaustedError as e:
                return {"text": str(e)}

        try:
            content, bytes_downloaded = _run_async(
                fetch_url(
                    url=url,
                    validator=url_validator,
                    max_length=max_length,
                    max_bytes=max_bytes,
                    timeout=timeout,
                    user_agent=user_agent,
                )
            )

            if budget is not None:
                budget.record_fetch(bytes_downloaded)

            doc = Document(
                content=content,
                meta={"source": url, "provider": "fetch"},
            )
            if evidence_accumulator is not None:
                evidence_accumulator.append(doc)

            logger.info(
                "fetch_url downloaded %d bytes, extracted %d chars (url=%s)",
                bytes_downloaded,
                len(content),
                url,
            )
            return {"text": content}

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "URL fetch failed",
                url=url,
                operation="fetch_url",
                original_error=str(e),
            ) from e

    return {
        "name": "fetch_url",
        "description": (
            "Fetch a URL and extract its main content as clean text. "
            "Use to get full page content from URLs found in search results."
        ),
        "function": fetch,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and extract content from.",
                },
            },
            "required": ["url"],
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
