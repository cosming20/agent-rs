"""Ingest a file (PDF, DOCX, PPTX, HTML, etc.) into the RAG pipeline.

Flow: Docling parse → Postgres DoclingDocument JSON persistence
      → Cognee add+cognify (vector+graph via Qdrant + Neo4j)

Usage: uv run python scripts/ingest_file.py <file_or_url> [file2] [file3] ...
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

# Silence all noisy loggers — only errors and our progress output
logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
for _name in (
    "cognee", "aiosqlite", "neo4j", "neo4j.io", "neo4j.pool", "instructor",
    "OntologyAdapter", "cognee.shared", "httpx", "httpcore",
    "docling", "docling.backend", "docling.pipeline", "docling.models",
    "docling.document_converter", "docling.datamodel",
    "litellm", "openai", "qdrant_client",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

from agentic_ra.core.config import get_settings
from agentic_ra.observability.cost_tracker import setup_cost_tracking
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.stores.cognee.client import CogneeStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore


async def ingest_file(source: str) -> None:
    s = get_settings()
    start = time.time()

    # Initialize cost tracking so Cognee's LiteLLM calls are captured
    setup_cost_tracking(pg_dsn=s.postgres_dsn_async.get_secret_value())

    # Build components
    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    doc_store = DocumentStore(pg_client=pg)
    pipeline = EmbeddingPipeline(
        model=s.embedding_model,
        api_key=s.llm_api_key.get_secret_value(),
        dimension=s.embedding_dimension,
        max_tokens=s.chunk_max_tokens,
        merge_peers=s.chunk_merge_peers,
    )

    # Cognee store — always required (Cognee is now the sole retrieval backend)
    cognee_store = CogneeStore(dataset=s.cognee_dataset)
    cognee_store.configure(
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

    # Check for incomplete previous ingestion (resume support)
    existing = await doc_store.get_by_source(source, user_id="anonymous")
    if existing and not existing.cognify_done:
        print(f"Resuming: {source} ({existing.chunks_added}/{existing.chunk_count} chunks added)")
        parsed_doc = existing
        # Re-parse to get chunks (DoclingDocument JSON is in Postgres but we need chunk list)
        chunk_result = pipeline.chunk_file(source)
        breadcrumb_chunks = pipeline.build_breadcrumb_chunks(chunk_result)
        skip = existing.chunks_added
    elif existing and existing.cognify_done:
        print(f"Already ingested: {source} — skipping (use --force to re-ingest)")
        await pg.close()
        return
    else:
        # Fresh ingestion
        print(f"Parsing: {source}")
        chunk_result = pipeline.chunk_file(source)
        print(f"  Chunks: {chunk_result.chunk_count}")

        if chunk_result.chunk_count == 0:
            print("  No chunks produced — skipping.")
            await pg.close()
            return

        docling_json = (
            chunk_result.docling_doc.export_to_dict()
            if chunk_result.docling_doc is not None
            else {}
        )
        parsed_doc = await doc_store.save(
            source_path=source,
            docling_json=docling_json,
            chunk_count=chunk_result.chunk_count,
            metadata={"file": source},
        )
        print(f"  Persisted DoclingDocument: {parsed_doc.id}")
        breadcrumb_chunks = pipeline.build_breadcrumb_chunks(chunk_result)
        skip = 0

    # Cognee: add chunks with progress tracking
    total = len(breadcrumb_chunks)
    chunks_to_add = breadcrumb_chunks[skip:]
    print(f"  Adding chunks to Cognee: {len(chunks_to_add)}/{total} (skip={skip})")

    import cognee
    for i, chunk_text in enumerate(chunks_to_add):
        if not chunk_text.strip():
            continue
        await cognee.add(data=chunk_text, dataset_name=s.cognee_dataset)
        added = skip + i + 1
        # Progress bar
        pct = added * 100 // total
        bar = f"[{'█' * (pct // 2)}{'░' * (50 - pct // 2)}]"
        print(f"\r  Cognee add {bar} {added}/{total}", end="", flush=True)
        # Save progress every 25 chunks
        if added % 25 == 0:
            await doc_store.update_progress(parsed_doc.id, chunks_added=added)
    print()  # newline after progress bar

    await doc_store.update_progress(parsed_doc.id, chunks_added=total)

    # Cognify
    print("  Running cognify (graph + vector)...")
    await cognee_store.cognify()
    await doc_store.update_progress(parsed_doc.id, chunks_added=total, cognify_done=True)
    print("  Cognee: graph + vector ingestion complete")

    await pg.close()

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/ingest_file.py <file_or_url> [file2] ...")
        sys.exit(1)

    for source in sys.argv[1:]:
        await ingest_file(source)
        print()

    print("All files ingested.")


if __name__ == "__main__":
    asyncio.run(main())
