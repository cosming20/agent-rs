"""Reset Qdrant collection and OpenSearch index."""

import asyncio

from agentic_ra.core.config import get_settings
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.qdrant.client import QdrantStore


async def main() -> None:
    s = get_settings()

    q = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    try:
        await q._client.delete_collection("documents")
    except Exception:
        pass
    await q.ensure_collection("documents", vector_size=s.embedding_dimension)
    await q.close()
    print(f"  Qdrant collection reset ({s.embedding_dimension} dims)")

    o = OpenSearchStore(
        host=s.opensearch_host,
        port=s.opensearch_port,
        user=s.opensearch_user,
        password=s.opensearch_initial_admin_password,
    )
    try:
        await o._client.indices.delete(index="documents")
    except Exception:
        pass
    await o.ensure_index("documents")
    await o.close()
    print("  OpenSearch index reset")

    print("All stores reset")


if __name__ == "__main__":
    asyncio.run(main())
