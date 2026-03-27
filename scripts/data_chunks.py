"""Preview chunks stored in Cognee (Qdrant via Cognee)."""

import asyncio
import sys

from agentic_ra.core.config import get_settings
from agentic_ra.stores.cognee.client import CogneeStore


async def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    s = get_settings()

    store = CogneeStore(dataset=s.cognee_dataset)
    store.configure(
        llm_api_key=s.llm_api_key.get_secret_value(),
        llm_model=s.llm_enrichment_model,
        graph_db_provider=s.cognee_graph_db_provider,
        vector_db_provider=s.cognee_vector_db_provider,
        data_root=s.cognee_data_root,
        system_root=s.cognee_system_root,
        qdrant_url=s.qdrant_url,
        qdrant_api_key=s.qdrant_api_key.get_secret_value(),
        neo4j_bolt_url=s.neo4j_bolt_url,
        neo4j_user=s.neo4j_user,
        neo4j_password=s.neo4j_password.get_secret_value(),
        neo4j_db=s.neo4j_db,
    )

    try:
        docs = await store.search(
            query="*",
            search_type="CHUNKS",
            top_k=limit,
            only_context=True,
        )

        if not docs:
            print("No chunks found in Cognee.")
        else:
            for i, doc in enumerate(docs, 1):
                content = doc.content or ""
                source = doc.meta.get("source", "unknown")
                search_type = doc.meta.get("search_type", "")
                print(f"--- Chunk {i} ---")
                print(f"  Source: {source}")
                print(f"  Search type: {search_type}")
                print(f"  Text: {content[:200]}{'...' if len(content) > 200 else ''}")
                print()

        print(f"Total chunks returned: {len(docs)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
