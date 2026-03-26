"""Internal MCP server exposing RAG search as a tool.

Run standalone: ``python -m agentic_ra.tools.mcp_servers.search_server``
Or connect via MCPToolClient with stdio transport.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgenticRA Search", json_response=True)


@mcp.tool()
def search_documents(query: str, limit: int = 5) -> str:
    """Search the document knowledge base.

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return.

    Returns:
        Search results as formatted text.
    """
    # Placeholder — will be wired to RetrievalRouter in Phase 8
    return (
        f"[search_documents] Query: {query!r}, limit: {limit}. "
        "Results: placeholder — wire to retrieval pipeline."
    )


@mcp.tool()
def get_document_count() -> str:
    """Get the total number of indexed documents.

    Returns:
        Document count as text.
    """
    # Placeholder — will be wired to stores in Phase 8
    return "[get_document_count] Count: placeholder — wire to stores."


if __name__ == "__main__":
    mcp.run(transport="stdio")
