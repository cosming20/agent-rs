"""Reset all Cognee data: Qdrant collections, Neo4j graph, local files."""

from __future__ import annotations

import shutil
from pathlib import Path

from agentic_ra.core.config import get_settings


def main() -> None:
    s = get_settings()

    # 1. Wipe Qdrant collections
    print("  Clearing Qdrant collections...")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=s.qdrant_url, api_key=s.qdrant_api_key.get_secret_value()
        )
        for col in client.get_collections().collections:
            client.delete_collection(col.name)
            print(f"    Deleted: {col.name}")
        client.close()
        print("  Qdrant: clean")
    except Exception as e:
        print(f"  Qdrant: failed ({e})")

    # 2. Wipe Neo4j graph
    print("  Clearing Neo4j graph...")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            s.neo4j_bolt_url,
            auth=(s.neo4j_user, s.neo4j_password.get_secret_value()),
        )
        with driver.session(database=s.neo4j_db) as session:
            result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
            count = result.single()["deleted"]
            print(f"    Deleted {count} nodes")
        driver.close()
        print("  Neo4j: clean")
    except Exception as e:
        print(f"  Neo4j: failed ({e})")

    # 3. Wipe local Cognee files
    print("  Clearing local files...")
    for d in (s.cognee_data_root, s.cognee_system_root):
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        print(f"    Wiped: {p}")
    print("  Local files: clean")

    print("Done. Ready for fresh ingest.")


if __name__ == "__main__":
    main()
