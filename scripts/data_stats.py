"""Show ingested data stats across all stores."""

import asyncio
import json

from agentic_ra.core.config import get_settings
from agentic_ra.stores.opensearch.client import OpenSearchStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.qdrant.client import QdrantStore

from sqlalchemy import func, select
from agentic_ra.stores.postgres.models import ParsedDocument, ChatMessage, MemoryEntry


async def main() -> None:
    s = get_settings()

    # Postgres
    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    async with pg.session() as sess:
        parsed_count = (await sess.execute(select(func.count(ParsedDocument.id)))).scalar() or 0
        chat_count = (await sess.execute(select(func.count(ChatMessage.id)))).scalar() or 0
        memory_count = (await sess.execute(select(func.count(MemoryEntry.id)))).scalar() or 0

    print("=== Postgres ===")
    print(f"  Parsed documents: {parsed_count}")
    print(f"  Chat messages:    {chat_count}")
    print(f"  Memory entries:   {memory_count}")
    await pg.close()

    # Qdrant
    q = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)
    try:
        info = await q._client.get_collection("documents")
        print(f"\n=== Qdrant (documents) ===")
        print(f"  Vectors:    {info.points_count}")
        print(f"  Dimensions: {info.config.params.vectors.size}")  # type: ignore[union-attr]
    except Exception:
        print("\n=== Qdrant ===\n  Collection 'documents' not found")
    await q.close()

    # OpenSearch
    o = OpenSearchStore(
        host=s.opensearch_host, port=s.opensearch_port,
        user=s.opensearch_user, password=s.opensearch_initial_admin_password,
    )
    try:
        count_resp = await o._client.count(index="documents")
        print(f"\n=== OpenSearch (documents) ===")
        print(f"  Documents: {count_resp['count']}")
    except Exception:
        print("\n=== OpenSearch ===\n  Index 'documents' not found")
    await o.close()


if __name__ == "__main__":
    asyncio.run(main())
