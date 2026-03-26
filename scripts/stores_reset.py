"""Reset Qdrant collection and Cognee knowledge graph."""

import asyncio

from agentic_ra.core.config import get_settings
from agentic_ra.stores.cognee.client import CogneeStore
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

    cognee_store = CogneeStore(dataset=s.cognee_dataset)
    cognee_store.configure(
        llm_api_key=s.openai_api_key.get_secret_value(),
        llm_model=f"openai/{s.llm_model}",
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
    await cognee_store.prune()
    print("  Cognee knowledge graph pruned")

    print("All stores reset")


if __name__ == "__main__":
    asyncio.run(main())
