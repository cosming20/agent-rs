"""Internal MCP server exposing RAG search as a tool.

Run standalone: ``python -m agentic_ra.tools.mcp_servers.search_server``
Or connect via MCPToolClient with stdio transport.
"""


from mcp.server.fastmcp import FastMCP

from agentic_ra.core.config import get_settings
from agentic_ra.retrieval.router import RetrievalRouter, RetrievalStrategy
from agentic_ra.stores.cognee.client import CogneeStore

mcp = FastMCP("AgenticRA Search", json_response=True)


@mcp.tool()
async def search_documents(
    query: str, user_id: str = "anonymous", limit: int = 5
) -> str:
    """Search the document knowledge base using hybrid retrieval via Cognee.

    Args:
        query: Natural language search query.
        user_id: User identifier for dataset scoping.
        limit: Maximum number of results to return.

    Returns:
        Search results as formatted text with sources.
    """
    s = get_settings()

    if not s.cognee_enabled:
        return "Cognee is not enabled — search unavailable."

    cognee_store = CogneeStore(dataset=f"user_{user_id}")
    cognee_store.configure(
        llm_api_key=s.llm_api_key.get_secret_value(),
        llm_model=s.llm_enrichment_model,
        graph_db_provider=s.cognee_graph_db_provider,
        vector_db_provider=s.cognee_vector_db_provider,
        qdrant_url=s.qdrant_url,
        qdrant_api_key=s.qdrant_api_key.get_secret_value(),
        neo4j_bolt_url=s.neo4j_bolt_url,
        neo4j_user=s.neo4j_user,
        neo4j_password=s.neo4j_password.get_secret_value(),
        neo4j_db=s.neo4j_db,
    )
    router = RetrievalRouter(cognee_store=cognee_store)

    try:
        all_results = await router.retrieve(
            query=query, strategy=RetrievalStrategy.HYBRID, limit=limit
        )

        if not all_results:
            return "No documents found."

        seen: set[str] = set()
        lines: list[str] = []
        for doc in all_results:
            text = doc.meta.get("chunk_text", doc.content or "")[:200]
            if text not in seen:
                seen.add(text)
                source = doc.meta.get("source", "unknown")
                lines.append(f"[{source}] {text}")

        return "\n\n".join(lines[:limit])
    except Exception as e:
        return f"Search error: {e}"


@mcp.tool()
async def get_document_count() -> str:
    """Get the total number of indexed documents.

    Returns:
        Document count as text (currently not available via Cognee).
    """
    return "Document count not available via Cognee SDK."


if __name__ == "__main__":
    mcp.run(transport="stdio")
