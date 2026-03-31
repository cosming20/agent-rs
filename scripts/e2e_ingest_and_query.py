"""E2E file ingestion + query validation.

Ingests a PDF into the RAG pipeline, then queries the live gateway API
to validate the full production path end-to-end.

Usage:
    uv run python scripts/e2e_ingest_and_query.py              # ingest test.pdf + query
    uv run python scripts/e2e_ingest_and_query.py other.pdf     # override file
    uv run python scripts/e2e_ingest_and_query.py --skip-ingest # query only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass

import httpx

from agentic_ra.core.config import get_settings
from agentic_ra.observability.cost_tracker import setup_cost_tracking
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.stores.cognee.client import CogneeStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore

# ── Constants ───────────────────────────────────────────────

DEFAULT_FILE = "test.pdf"
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:1059")
HEALTH_MAX_RETRIES = 5
HEALTH_BACKOFF_SECONDS = 2.0
QUERY_TIMEOUT_SECONDS = 120

QUERIES = [
    {
        "query": "Ce zice articolul 170?",
        "tag": "direct-lookup",
        "strategy": "graph_hybrid",
    },
    {
        "query": "Care sunt pedepsele pentru infracțiuni contra libertății sexuale?",
        "tag": "cross-reference",
        "strategy": "graph_hybrid",
    },
    {
        "query": "Ce diferențe există între tâlhărie și furt calificat?",
        "tag": "comparative",
        "strategy": "graph_hybrid",
    },
    {
        "query": "În ce condiții se aplică circumstanțele atenuante conform Codului Penal?",
        "tag": "synthesis",
        "strategy": "graph_hybrid",
    },
    {
        "query": "Care este pedeapsa pentru uciderea din culpă în cazul unui accident rutier?",
        "tag": "scenario",
        "strategy": "graph_hybrid",
    },
]


# ── Data classes ────────────────────────────────────────────


@dataclass(frozen=True)
class IngestResult:
    """Result of the ingestion phase."""

    file: str
    chunk_count: int
    parsed_document_id: str
    cognee_ingested: bool
    duration_seconds: float


@dataclass(frozen=True)
class QueryResult:
    """Result of a single query."""

    query: str
    tag: str
    answer: str
    confidence: float
    is_grounded: bool
    source_count: int
    latency_seconds: float
    error: str | None = None


# ── Phase 1: Ingestion ─────────────────────────────────────


async def run_ingestion(file_path: str) -> IngestResult:
    """Ingest a file into the RAG pipeline.

    Same flow as scripts/ingest_file.py:
    Docling parse -> Postgres persistence -> Cognee add+cognify.
    """
    s = get_settings()
    start = time.time()

    # Initialize cost tracking so Cognee's LiteLLM calls are captured
    setup_cost_tracking(pg_dsn=s.postgres_dsn_async.get_secret_value())

    pg = PostgresClient(dsn=s.postgres_dsn_async.get_secret_value())
    doc_store = DocumentStore(pg_client=pg)
    pipeline = EmbeddingPipeline(
        model=s.embedding_model,
        api_key=s.llm_api_key.get_secret_value(),
        dimension=s.embedding_dimension,
        max_tokens=s.chunk_max_tokens,
        merge_peers=s.chunk_merge_peers,
    )

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

    # 1. Parse + chunk with Docling
    print(f"  Parsing: {file_path}")
    chunk_result = pipeline.chunk_file(file_path)
    print(f"  Chunks: {chunk_result.chunk_count}")

    if chunk_result.chunk_count == 0:
        await pg.close()
        raise RuntimeError(f"No chunks produced from {file_path}")

    # 2. Persist DoclingDocument JSON in Postgres
    docling_json = (
        chunk_result.docling_doc.export_to_dict()
        if chunk_result.docling_doc is not None
        else {}
    )
    parsed_doc = await doc_store.save(
        source_path=file_path,
        docling_json=docling_json,
        chunk_count=chunk_result.chunk_count,
        metadata={"file": file_path},
    )
    print(f"  Persisted DoclingDocument: {parsed_doc.id}")

    # 3. Cognee: add individual chunks + cognify (embedding -> Qdrant, entities -> Neo4j)
    # Ingest into "user_anonymous" dataset to match the query API path.
    # The orchestrator scopes queries to "user_{user_id}" — default user is "anonymous".
    query_dataset = "user_anonymous"
    print("  Building breadcrumb chunks...")
    breadcrumb_chunks = pipeline.build_breadcrumb_chunks(chunk_result)
    print(f"  Breadcrumb chunks: {len(breadcrumb_chunks)}")

    print(f"  Cognee add (per-chunk) + cognify (dataset={query_dataset})...")
    await cognee_store.add_chunks(chunks=breadcrumb_chunks, dataset=query_dataset)
    print("  Cognee: graph + vector ingestion complete")

    await pg.close()
    elapsed = time.time() - start

    return IngestResult(
        file=file_path,
        chunk_count=chunk_result.chunk_count,
        parsed_document_id=str(parsed_doc.id),
        cognee_ingested=True,
        duration_seconds=elapsed,
    )


# ── Phase 2: Gateway health check ──────────────────────────


async def check_gateway_health(base_url: str) -> None:
    """Check gateway is reachable. Retry with backoff."""
    async with httpx.AsyncClient() as client:
        for attempt in range(1, HEALTH_MAX_RETRIES + 1):
            try:
                resp = await client.get(f"{base_url}/health", timeout=5.0)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "ok":
                    print(f"  Status: healthy (v{data.get('version', '?')})")
                    return
                raise RuntimeError(f"Unexpected health response: {data}")
            except (httpx.HTTPError, RuntimeError) as exc:
                if attempt == HEALTH_MAX_RETRIES:
                    raise RuntimeError(
                        f"Gateway at {base_url} not reachable after "
                        f"{HEALTH_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                wait = HEALTH_BACKOFF_SECONDS * attempt
                print(
                    f"  Attempt {attempt}/{HEALTH_MAX_RETRIES} failed, "
                    f"retrying in {wait:.0f}s..."
                )
                await asyncio.sleep(wait)


# ── Phase 3: Query battery ─────────────────────────────────


async def run_query(
    client: httpx.AsyncClient,
    base_url: str,
    query_def: dict[str, str],
) -> QueryResult:
    """Execute a single query against the gateway API."""
    start = time.time()
    try:
        resp = await client.post(
            f"{base_url}/api/v1/query",
            json={
                "query": query_def["query"],
                "strategy": query_def["strategy"],
            },
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start

        return QueryResult(
            query=query_def["query"],
            tag=query_def["tag"],
            answer=data.get("answer", ""),
            confidence=data.get("confidence", 0.0),
            is_grounded=data.get("is_grounded", False),
            source_count=len(data.get("citations", [])),
            latency_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.time() - start
        return QueryResult(
            query=query_def["query"],
            tag=query_def["tag"],
            answer="",
            confidence=0.0,
            is_grounded=False,
            source_count=0,
            latency_seconds=elapsed,
            error=str(exc),
        )


async def run_all_queries(base_url: str) -> list[QueryResult]:
    """Run all queries sequentially against the gateway."""
    results: list[QueryResult] = []
    async with httpx.AsyncClient() as client:
        for i, query_def in enumerate(QUERIES, 1):
            print(
                f"  [{i}/{len(QUERIES)}] {query_def['tag']}: "
                f"{query_def['query'][:50]}..."
            )
            result = await run_query(client, base_url, query_def)
            if result.error:
                print(f"    ERROR: {result.error}")
            else:
                print(
                    f"    Confidence: {result.confidence:.2f} | "
                    f"Sources: {result.source_count} | "
                    f"{result.latency_seconds:.1f}s"
                )
            results.append(result)
    return results


# ── Phase 4: Summary report ────────────────────────────────


def print_header() -> None:
    """Print the report header."""
    print()
    print("=" * 60)
    print("  E2E Ingestion + Query — Romanian Penal Code")
    print("=" * 60)
    print()


def print_ingestion_report(result: IngestResult) -> None:
    """Print ingestion phase results."""
    print("Phase 1: Ingestion")
    print(f"  File:       {result.file}")
    print(f"  Chunks:     {result.chunk_count}")
    print(f"  Doc ID:     {result.parsed_document_id}")
    print(f"  Cognee:     {'OK' if result.cognee_ingested else 'FAILED'}")
    print(f"  Duration:   {result.duration_seconds:.1f}s")
    print()


def print_query_report(results: list[QueryResult]) -> None:
    """Print query phase results as a table + full answers."""
    print("Phase 3: Queries")
    print(
        f"  {'#':<4} {'Tag':<16} {'Conf.':<7} "
        f"{'Latency':<9} {'Srcs':<5} {'Status'}"
    )
    print(f"  {'─' * 4} {'─' * 16} {'─' * 7} {'─' * 9} {'─' * 5} {'─' * 8}")

    for i, r in enumerate(results, 1):
        status = "ERROR" if r.error else ("OK" if r.is_grounded else "UNGROUNDED")
        print(
            f"  {i:<4} {r.tag:<16} {r.confidence:<7.2f} "
            f"{r.latency_seconds:<9.1f} {r.source_count:<5} {status}"
        )

    print()
    print("-" * 60)
    print("  Full Answers")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r.query}")
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            print(f"  {r.answer}")
    print()


# ── Main ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="E2E file ingestion + query validation",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=DEFAULT_FILE,
        help=f"File to ingest (default: {DEFAULT_FILE})",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion phase (assume file already ingested)",
    )
    parser.add_argument(
        "--gateway-url",
        default=GATEWAY_URL,
        help=f"Gateway base URL (default: {GATEWAY_URL})",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the full E2E validation."""
    args = parse_args()
    total_start = time.time()

    print_header()

    # Phase 1: Ingestion
    ingest_result: IngestResult | None = None
    if args.skip_ingest:
        print("Phase 1: Ingestion (SKIPPED)")
        print()
    else:
        try:
            ingest_result = await run_ingestion(args.file)
            print_ingestion_report(ingest_result)
        except Exception as exc:
            print(f"\nIngestion FAILED: {exc}")
            print("Cannot proceed without ingested data.")
            sys.exit(1)

    # Phase 2: Gateway health check
    print(f"Phase 2: Gateway ({args.gateway_url})")
    try:
        await check_gateway_health(args.gateway_url)
    except RuntimeError as exc:
        print(f"\n  Gateway check FAILED: {exc}")
        print("  Start the server with: just server")
        sys.exit(1)
    print()

    # Phase 3: Query battery
    query_results = await run_all_queries(args.gateway_url)

    # Phase 4: Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()

    if ingest_result:
        print_ingestion_report(ingest_result)

    print_query_report(query_results)

    total_elapsed = time.time() - total_start
    passed = sum(1 for r in query_results if not r.error and r.is_grounded)
    print(f"  Results: {passed}/{len(query_results)} queries grounded")
    print(f"  Total E2E duration: {total_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
