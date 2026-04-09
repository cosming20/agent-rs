"""Web and news search tools — Brave Search via MCP."""


import logging
from typing import Any

from haystack import Document

from agentic_ra.core.errors import ToolError, WebBudgetExhaustedError
from agentic_ra.tools.budgets import WebToolBudget
from agentic_ra.tools.mcp_client import ToolResult
from agentic_ra.tools.mcp_http_client import MCPHttpClient

from ._async_bridge import run_async
from ._formatting import format_evidence, parse_brave_results
from ._types import ToolConfig

logger = logging.getLogger(__name__)


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

            result: ToolResult = run_async(
                brave_client.call_tool("brave_web_search", args)
            )

            # Record budget FIRST -- the call was made regardless of result
            if budget is not None:
                budget.record_web_search()

            if result.is_error:
                raise ToolError(
                    "Brave web search returned error",
                    query=query,
                    content=result.content[:200],
                )

            documents = parse_brave_results(result.content, provider="brave_web")
            if evidence_accumulator is not None:
                evidence_accumulator.extend(documents)

            formatted = format_evidence(documents)
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

            result: ToolResult = run_async(
                brave_client.call_tool("brave_news_search", args)
            )

            # Record budget FIRST -- the call was made regardless of result
            if budget is not None:
                budget.record_news_search()

            if result.is_error:
                raise ToolError(
                    "Brave news search returned error",
                    query=query,
                    content=result.content[:200],
                )

            documents = parse_brave_results(result.content, provider="brave_news")
            if evidence_accumulator is not None:
                evidence_accumulator.extend(documents)

            formatted = format_evidence(documents)
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
