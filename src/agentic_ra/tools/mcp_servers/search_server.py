"""Internal MCP server exposing RAG search as a tool.

Run standalone: ``python -m agentic_ra.tools.mcp_servers.search_server``
Or connect via MCPToolClient with stdio transport.
"""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from agentic_ra.core.config import get_settings
from agentic_ra.retrieval.router import QueryEmbedder
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.qdrant.client import QdrantStore

mcp = FastMCP("AgenticRA Search", json_response=True)


@mcp.tool()
async def search_documents(query: str, limit: int = 5) -> str:
    """Search the document knowledge base using hybrid retrieval.

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return.

    Returns:
        Search results as formatted text with sources.
    """
    s = get_settings()
    qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    opensearch = OpenSearchStore(
        host=s.opensearch_host, port=s.opensearch_port,
        user=s.opensearch_user, password=s.opensearch_initial_admin_password,
    )
    embedder = QueryEmbedder(
        model=s.embedding_model,
        api_key=s.openai_api_key.get_secret_value(),
        dimension=s.embedding_dimension,
    )

    try:
        query_vector = await asyncio.to_thread(embedder.run, query)
        vector_results, keyword_results = await asyncio.gather(
            qdrant.search(s.qdrant_collection, query_vector, limit=limit),
            opensearch.search(s.opensearch_index, query, limit=limit),
        )

        all_results = vector_results + keyword_results
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
    finally:
        await qdrant.close()
        await opensearch.close()


@mcp.tool()
async def get_document_count() -> str:
    """Get the total number of indexed documents in Qdrant.

    Returns:
        Document count as text.
    """
    s = get_settings()
    qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    try:
        info = await qdrant._client.get_collection(s.qdrant_collection)
        return f"Documents indexed: {info.points_count}"
    except Exception as e:
        return f"Count error: {e}"
    finally:
        await qdrant.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")
