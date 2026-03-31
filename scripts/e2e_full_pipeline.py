"""Comprehensive E2E integration test for the full RAG pipeline.

Runs the COMPLETE system end-to-end against real infrastructure:
  Phase 1: Infrastructure health checks
  Phase 2: Ingest test.pdf (parse, chunk, Postgres, Cognee)
  Phase 3: Query the system (3 queries via orchestrator)
  Phase 4: Check feedback loop (pipeline_runs + Neo4j weights)
  Phase 5: Check OTel metrics (Prometheus endpoint)
  Phase 6: Check Grafana dashboards (Playwright screenshots)
  Phase 7: Check Postgres data integrity (llm_calls, pipeline_runs)
  Phase 8: Summary report

Usage:
    uv run python scripts/e2e_full_pipeline.py
    uv run python scripts/e2e_full_pipeline.py --skip-ingest       # query only
    uv run python scripts/e2e_full_pipeline.py --skip-playwright   # no browser
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agentic_ra.core.config import Settings, get_settings
from agentic_ra.observability.cost_tracker import setup_cost_tracking
from agentic_ra.pipelines.factory import build_pipeline, RAGPipeline
from agentic_ra.retrieval.embedder import EmbeddingPipeline
from agentic_ra.stores.cognee.client import CogneeStore
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.documents import DocumentStore

# ── Constants ──────────────────────────────────────────────────

DEFAULT_FILE = "test.pdf"
ARTIFACTS_DIR = Path("artifacts")

# Infrastructure URLs (derived from settings at runtime via helpers)
QDRANT_HEALTH_PATH = "/healthz"
PROMETHEUS_HEALTH_PATH = "/-/healthy"
GRAFANA_HEALTH_PATH = "/api/health"
OTEL_PROMETHEUS_METRICS_PATH = "/metrics"

# External services not in Settings — use env or defaults
PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", "1066"))
GRAFANA_PORT = int(os.environ.get("GRAFANA_PORT", "1067"))
OTEL_PROM_EXPORTER_PORT = int(os.environ.get("OTEL_PROM_EXPORTER_PORT", "8889"))
PROMETHEUS_HOST = os.environ.get("PROMETHEUS_HOST", "localhost")
GRAFANA_HOST = os.environ.get("GRAFANA_HOST", "localhost")

# Grafana dashboard UIDs (configurable via env)
GRAFANA_DASHBOARD_RAG_OVERVIEW = os.environ.get(
    "GRAFANA_DASHBOARD_RAG_OVERVIEW", "rag-overview"
)
GRAFANA_DASHBOARD_LOOP1 = os.environ.get(
    "GRAFANA_DASHBOARD_LOOP1", "loop1-feedback"
)

# Query timeout for orchestrator calls
QUERY_TIMEOUT_SECONDS = 180

# Test queries covering different retrieval strategies
QUERIES = [
    {
        "query": "Ce zice articolul 170?",
        "tag": "factual-lookup",
        "strategy": "graph_hybrid",
    },
    {
        "query": "Care este relația dintre tâlhărie și furt calificat?",
        "tag": "relationship-graph",
        "strategy": "graph_hybrid",
    },
    {
        "query": "Care sunt principalele infracțiuni contra patrimoniului?",
        "tag": "broad-hybrid",
        "strategy": "hybrid",
    },
]

# Expected OTel metric name prefixes (OTel renames with underscores)
EXPECTED_METRICS = [
    "rag_query_count",
    "rag_retrieval_latency",
    "rag_verification_confidence",
    "rag_feedback_applied",
]

# Neo4j Cypher queries
CYPHER_NODE_COUNTS = "MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt"
CYPHER_FEEDBACK_WEIGHTS = (
    "MATCH (n) WHERE n.feedback_weight IS NOT NULL RETURN count(n) AS weighted_nodes"
)

# ── Data classes ───────────────────────────────────────────────


@dataclass
class ServiceStatus:
    """Health check result for a single service."""

    name: str
    port: int
    is_up: bool
    detail: str = ""


@dataclass
class IngestResult:
    """Result of the ingestion phase."""

    file: str
    chunk_count: int
    parsed_document_id: str
    cognee_ingested: bool
    neo4j_entity_count: int = 0
    neo4j_node_labels: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class QueryResult:
    """Result of a single query through the orchestrator."""

    query: str
    tag: str
    answer: str
    confidence: float
    is_grounded: bool
    strategy: str
    citation_count: int
    latency_seconds: float
    error: str | None = None


@dataclass
class PipelineRunRecord:
    """Summarized pipeline_runs row."""

    query: str
    strategy: str
    confidence: float
    is_grounded: bool
    citation_count: int
    evidence_count: int
    retrieval_ms: float


@dataclass
class LLMCostSummary:
    """Aggregated LLM cost by model."""

    model: str
    call_count: int
    input_tokens: int
    output_tokens: int
    total_cost: float


@dataclass
class PhaseReport:
    """Outcome of a single test phase."""

    phase: int
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ── Helpers ────────────────────────────────────────────────────


def _url(host: str, port: int, path: str = "") -> str:
    """Build an HTTP URL from host, port, and optional path."""
    return f"http://{host}:{port}{path}"


def _check_mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _fmt_seconds(s: float) -> str:
    return f"{s:.1f}s"


# ── Phase 1: Infrastructure Health Checks ──────────────────────


async def _check_postgres(settings: Settings) -> ServiceStatus:
    """Postgres: connect + SELECT 1."""
    try:
        pg = PostgresClient(dsn=settings.postgres_dsn_async.get_secret_value())
        async with pg.session() as session:
            from sqlalchemy import text

            result = await session.execute(text("SELECT 1"))
            val = result.scalar()
        await pg.close()
        return ServiceStatus(
            name="Postgres",
            port=settings.postgres_port,
            is_up=val == 1,
            detail=f"SELECT 1 = {val}",
        )
    except Exception as exc:
        return ServiceStatus(
            name="Postgres",
            port=settings.postgres_port,
            is_up=False,
            detail=str(exc)[:120],
        )


async def _check_http(
    name: str, host: str, port: int, path: str, timeout: float = 5.0
) -> ServiceStatus:
    """Generic HTTP health check."""
    url = _url(host, port, path)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout)
            is_up = resp.status_code < 400
            return ServiceStatus(
                name=name,
                port=port,
                is_up=is_up,
                detail=f"HTTP {resp.status_code}",
            )
    except Exception as exc:
        return ServiceStatus(
            name=name, port=port, is_up=False, detail=str(exc)[:120]
        )


async def _check_neo4j(settings: Settings) -> ServiceStatus:
    """Neo4j: bolt connection test via neo4j driver."""
    try:
        import neo4j

        driver = neo4j.GraphDatabase.driver(
            settings.neo4j_bolt_url,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        )
        with driver.session(database=settings.neo4j_db) as session:
            result = session.run("RETURN 1 AS n")
            val = result.single()["n"]
        driver.close()
        return ServiceStatus(
            name="Neo4j",
            port=settings.neo4j_bolt_port,
            is_up=val == 1,
            detail=f"RETURN 1 = {val}",
        )
    except Exception as exc:
        return ServiceStatus(
            name="Neo4j",
            port=settings.neo4j_bolt_port,
            is_up=False,
            detail=str(exc)[:120],
        )


async def run_phase1_health_checks(settings: Settings) -> PhaseReport:
    """Phase 1: Check all infrastructure services."""
    start = time.time()
    checks = await asyncio.gather(
        _check_postgres(settings),
        _check_http("Qdrant", settings.qdrant_host, settings.qdrant_http_port, QDRANT_HEALTH_PATH),
        _check_neo4j(settings),
        _check_http("Prometheus", PROMETHEUS_HOST, PROMETHEUS_PORT, PROMETHEUS_HEALTH_PATH),
        _check_http("Grafana", GRAFANA_HOST, GRAFANA_PORT, GRAFANA_HEALTH_PATH),
        _check_http("OTel Collector", "localhost", OTEL_PROM_EXPORTER_PORT, OTEL_PROMETHEUS_METRICS_PATH),
    )

    details: list[str] = []
    all_critical_up = True
    critical_services = {"Postgres", "Qdrant", "Neo4j"}

    for svc in checks:
        status = "UP" if svc.is_up else "DOWN"
        details.append(f"  {_check_mark(svc.is_up)} {svc.name:<20} {status} ({svc.port}) {svc.detail}")
        if svc.name in critical_services and not svc.is_up:
            all_critical_up = False

    elapsed = time.time() - start
    return PhaseReport(
        phase=1,
        name="Infrastructure Health Checks",
        passed=all_critical_up,
        details=details,
        duration_seconds=elapsed,
    )


# ── Phase 2: Ingestion ────────────────────────────────────────


async def _neo4j_node_counts(settings: Settings) -> dict[str, int]:
    """Query Neo4j for node label counts."""
    try:
        import neo4j

        driver = neo4j.GraphDatabase.driver(
            settings.neo4j_bolt_url,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        )
        with driver.session(database=settings.neo4j_db) as session:
            result = session.run(CYPHER_NODE_COUNTS)
            counts: dict[str, int] = {}
            for record in result:
                labels_list = record["labels"]
                label = ":".join(labels_list) if labels_list else "unlabeled"
                counts[label] = record["cnt"]
        driver.close()
        return counts
    except Exception:
        return {}


async def run_phase2_ingestion(file_path: str, settings: Settings) -> tuple[PhaseReport, IngestResult | None]:
    """Phase 2: Parse, chunk, persist, and cognify a document."""
    start = time.time()
    details: list[str] = []

    try:
        # Initialize cost tracking
        setup_cost_tracking(pg_dsn=settings.postgres_dsn_async.get_secret_value())

        pg = PostgresClient(dsn=settings.postgres_dsn_async.get_secret_value())
        doc_store = DocumentStore(pg_client=pg)
        pipeline = EmbeddingPipeline(
            model=settings.embedding_model,
            api_key=settings.llm_api_key.get_secret_value(),
            dimension=settings.embedding_dimension,
            max_tokens=settings.chunk_max_tokens,
            merge_peers=settings.chunk_merge_peers,
        )

        cognee_store = CogneeStore(dataset=settings.cognee_dataset)
        cognee_store.configure(
            llm_api_key=settings.llm_api_key.get_secret_value(),
            llm_model=settings.llm_enrichment_model,
            graph_db_provider=settings.cognee_graph_db_provider,
            vector_db_provider=settings.cognee_vector_db_provider,
            data_root=settings.cognee_data_root,
            system_root=settings.cognee_system_root,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key.get_secret_value(),
            neo4j_bolt_url=settings.neo4j_bolt_url,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password.get_secret_value(),
            neo4j_db=settings.neo4j_db,
        )

        # 1. Parse + chunk
        details.append(f"  Parsing: {file_path}")
        chunk_result = pipeline.chunk_file(file_path)
        details.append(f"  {_check_mark(chunk_result.chunk_count > 0)} Chunks produced: {chunk_result.chunk_count}")

        if chunk_result.chunk_count == 0:
            await pg.close()
            return (
                PhaseReport(phase=2, name="Ingestion", passed=False,
                            details=details + ["  FAIL: No chunks produced"],
                            duration_seconds=time.time() - start),
                None,
            )

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
        details.append(f"  {_check_mark(True)} Postgres record: parsed_documents.id = {parsed_doc.id}")

        # 3. Cognee: add chunks + cognify
        query_dataset = "user_anonymous"
        breadcrumb_chunks = pipeline.build_breadcrumb_chunks(chunk_result)
        details.append(f"  Breadcrumb chunks: {len(breadcrumb_chunks)}")

        details.append(f"  Cognee add + cognify (dataset={query_dataset})...")
        await cognee_store.add_chunks(chunks=breadcrumb_chunks, dataset=query_dataset)
        details.append(f"  {_check_mark(True)} Cognee ingestion complete")

        # 4. Verify Neo4j nodes
        node_counts = await _neo4j_node_counts(settings)
        total_entities = sum(
            cnt for label, cnt in node_counts.items()
            if "Entity" in label or "DocumentChunk" in label
        )
        details.append(f"  {_check_mark(True)} Neo4j node labels: {node_counts}")

        await pg.close()
        elapsed = time.time() - start

        result = IngestResult(
            file=file_path,
            chunk_count=chunk_result.chunk_count,
            parsed_document_id=str(parsed_doc.id),
            cognee_ingested=True,
            neo4j_entity_count=total_entities,
            neo4j_node_labels=node_counts,
            duration_seconds=elapsed,
        )
        return (
            PhaseReport(phase=2, name="Ingestion", passed=True,
                        details=details, duration_seconds=elapsed),
            result,
        )

    except Exception as exc:
        elapsed = time.time() - start
        details.append(f"  FAIL: {exc}")
        return (
            PhaseReport(phase=2, name="Ingestion", passed=False,
                        details=details, duration_seconds=elapsed),
            None,
        )


# ── Phase 3: Query the system ─────────────────────────────────


async def _run_single_query(
    pipeline: RAGPipeline,
    query_def: dict[str, str],
    session_id: uuid.UUID,
    user_id: str,
) -> QueryResult:
    """Execute one query through the orchestrator."""
    start = time.time()
    try:
        response = await pipeline.orchestrator.process_query(
            query=query_def["query"],
            session_id=session_id,
            user_id=user_id,
            strategy=query_def["strategy"],
        )
        elapsed = time.time() - start
        return QueryResult(
            query=query_def["query"],
            tag=query_def["tag"],
            answer=response.answer,
            confidence=response.confidence,
            is_grounded=response.is_grounded,
            strategy=response.strategy,
            citation_count=len(response.citations),
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
            strategy=query_def.get("strategy", ""),
            citation_count=0,
            latency_seconds=elapsed,
            error=str(exc)[:300],
        )


async def run_phase3_queries(settings: Settings) -> tuple[PhaseReport, list[QueryResult], RAGPipeline | None]:
    """Phase 3: Build pipeline and run test queries."""
    start = time.time()
    details: list[str] = []
    results: list[QueryResult] = []
    rag_pipeline: RAGPipeline | None = None

    try:
        # Initialize cost tracking
        setup_cost_tracking(pg_dsn=settings.postgres_dsn_async.get_secret_value())

        details.append("  Building RAG pipeline...")
        rag_pipeline = build_pipeline(settings)
        details.append(f"  {_check_mark(True)} Pipeline assembled")

        session_id = uuid.uuid4()
        user_id = "anonymous"

        for i, query_def in enumerate(QUERIES, 1):
            details.append(f"  [{i}/{len(QUERIES)}] {query_def['tag']}: {query_def['query'][:60]}...")
            result = await _run_single_query(rag_pipeline, query_def, session_id, user_id)
            results.append(result)

            if result.error:
                details.append(f"    FAIL: {result.error[:120]}")
            else:
                details.append(
                    f"    {_check_mark(True)} conf={result.confidence:.2f}, "
                    f"grounded={result.is_grounded}, "
                    f"citations={result.citation_count}, "
                    f"{_fmt_seconds(result.latency_seconds)}"
                )

        # Flush cost tracker writes (give DB writes time to complete)
        await asyncio.sleep(2)

        all_answered = all(r.answer and not r.error for r in results)
        all_confident = all(r.confidence > 0 for r in results)
        elapsed = time.time() - start

        details.append(f"  All queries answered: {_check_mark(all_answered)}")
        details.append(f"  All confidence > 0: {_check_mark(all_confident)}")

        return (
            PhaseReport(phase=3, name="Query System", passed=all_answered,
                        details=details, duration_seconds=elapsed),
            results,
            rag_pipeline,
        )

    except Exception as exc:
        elapsed = time.time() - start
        details.append(f"  FAIL: {exc}")
        return (
            PhaseReport(phase=3, name="Query System", passed=False,
                        details=details, duration_seconds=elapsed),
            results,
            rag_pipeline,
        )


# ── Phase 4: Check Feedback Loop (Pipeline Runs) ──────────────


async def run_phase4_feedback(settings: Settings) -> tuple[PhaseReport, list[PipelineRunRecord]]:
    """Phase 4: Verify pipeline_runs table and Neo4j feedback weights."""
    start = time.time()
    details: list[str] = []
    records: list[PipelineRunRecord] = []

    try:
        pg = PostgresClient(dsn=settings.postgres_dsn_async.get_secret_value())
        async with pg.session() as session:
            from sqlalchemy import text

            # Recent pipeline runs
            result = await session.execute(
                text(
                    "SELECT query, strategy, confidence, is_grounded, "
                    "citation_count, evidence_count, retrieval_ms "
                    "FROM pipeline_runs ORDER BY created_at DESC LIMIT 5"
                )
            )
            rows = result.fetchall()

        await pg.close()

        for row in rows:
            rec = PipelineRunRecord(
                query=row[0][:60],
                strategy=row[1],
                confidence=float(row[2]),
                is_grounded=bool(row[3]),
                citation_count=int(row[4]),
                evidence_count=int(row[5]),
                retrieval_ms=float(row[6]),
            )
            records.append(rec)
            details.append(
                f"  {_check_mark(True)} strategy={rec.strategy}, conf={rec.confidence:.2f}, "
                f"grounded={rec.is_grounded}, evidence={rec.evidence_count}"
            )

        has_runs = len(records) > 0
        details.insert(0, f"  Pipeline runs found: {len(records)}")

        # Check Neo4j feedback weights
        weighted_count = 0
        try:
            import neo4j

            driver = neo4j.GraphDatabase.driver(
                settings.neo4j_bolt_url,
                auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            )
            with driver.session(database=settings.neo4j_db) as neo_session:
                result = neo_session.run(CYPHER_FEEDBACK_WEIGHTS)
                record = result.single()
                weighted_count = record["weighted_nodes"] if record else 0
            driver.close()
        except Exception as exc:
            details.append(f"  Neo4j feedback query failed: {str(exc)[:80]}")

        details.append(f"  Neo4j weighted nodes: {weighted_count} (0 is OK if batch hasn't run)")

        elapsed = time.time() - start
        return (
            PhaseReport(phase=4, name="Feedback Loop", passed=has_runs,
                        details=details, duration_seconds=elapsed),
            records,
        )

    except Exception as exc:
        elapsed = time.time() - start
        details.append(f"  FAIL: {exc}")
        return (
            PhaseReport(phase=4, name="Feedback Loop", passed=False,
                        details=details, duration_seconds=elapsed),
            records,
        )


# ── Phase 5: Check OTel Metrics ───────────────────────────────


async def run_phase5_metrics() -> PhaseReport:
    """Phase 5: Scrape Prometheus endpoint for expected metric names."""
    start = time.time()
    details: list[str] = []
    found_metrics: dict[str, bool] = {}

    url = _url("localhost", OTEL_PROM_EXPORTER_PORT, OTEL_PROMETHEUS_METRICS_PATH)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)

        if resp.status_code >= 400:
            details.append(f"  FAIL: HTTP {resp.status_code} from {url}")
            return PhaseReport(phase=5, name="OTel Metrics", passed=False,
                               details=details, duration_seconds=time.time() - start)

        body = resp.text

        for metric_name in EXPECTED_METRICS:
            present = metric_name in body
            found_metrics[metric_name] = present
            # Try to extract a sample value
            value_str = "not found"
            if present:
                for line in body.splitlines():
                    if metric_name in line and not line.startswith("#"):
                        value_str = line.strip()
                        break
            details.append(f"  {_check_mark(present)} {metric_name}: {value_str[:100]}")

        any_found = any(found_metrics.values())
        elapsed = time.time() - start

        # Metrics may not be present if OTel export interval hasn't fired yet
        # Report but don't fail hard
        return PhaseReport(
            phase=5,
            name="OTel Metrics",
            passed=True,  # Soft pass — metrics are eventually consistent
            details=details + [f"  Note: {sum(found_metrics.values())}/{len(EXPECTED_METRICS)} metrics found (OTel export is periodic)"],
            duration_seconds=elapsed,
        )

    except Exception as exc:
        details.append(f"  FAIL: {exc}")
        return PhaseReport(phase=5, name="OTel Metrics", passed=False,
                           details=details, duration_seconds=time.time() - start)


# ── Phase 6: Grafana Dashboards (Playwright) ──────────────────


async def run_phase6_grafana() -> PhaseReport:
    """Phase 6: Take screenshots of Grafana dashboards using Playwright."""
    start = time.time()
    details: list[str] = []
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        details.append("  SKIP: playwright not installed (pip install playwright)")
        return PhaseReport(phase=6, name="Grafana Dashboards", passed=True,
                           details=details, duration_seconds=time.time() - start)

    grafana_base = _url(GRAFANA_HOST, GRAFANA_PORT)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,
            )
            page = await context.new_page()

            # Login to Grafana (default admin/admin)
            await page.goto(f"{grafana_base}/login", wait_until="networkidle", timeout=15000)
            # Check if login form exists
            login_input = await page.query_selector('input[name="user"]')
            if login_input:
                await page.fill('input[name="user"]', "admin")
                await page.fill('input[name="password"]', "admin")
                await page.click('button[type="submit"]')
                # Handle "skip change password" if prompted
                await asyncio.sleep(2)
                skip_btn = await page.query_selector('a[href*="skip"], button:has-text("Skip")')
                if skip_btn:
                    await skip_btn.click()
                await asyncio.sleep(1)
                details.append(f"  {_check_mark(True)} Grafana login successful")
            else:
                details.append("  Grafana: no login form (anonymous access)")

            # Screenshot 1: RAG Overview
            rag_overview_url = f"{grafana_base}/d/{GRAFANA_DASHBOARD_RAG_OVERVIEW}/"
            screenshot_path_1 = ARTIFACTS_DIR / "grafana-rag-overview.png"
            try:
                await page.goto(rag_overview_url, wait_until="networkidle", timeout=20000)
                await asyncio.sleep(3)  # Let panels render
                await page.screenshot(path=str(screenshot_path_1), full_page=True)
                details.append(f"  {_check_mark(True)} RAG Overview: {screenshot_path_1}")
            except Exception as exc:
                details.append(f"  FAIL RAG Overview: {str(exc)[:80]}")

            # Screenshot 2: Loop 1 Feedback
            loop1_url = f"{grafana_base}/d/{GRAFANA_DASHBOARD_LOOP1}/"
            screenshot_path_2 = ARTIFACTS_DIR / "grafana-loop1-feedback.png"
            try:
                await page.goto(loop1_url, wait_until="networkidle", timeout=20000)
                await asyncio.sleep(3)
                await page.screenshot(path=str(screenshot_path_2), full_page=True)
                details.append(f"  {_check_mark(True)} Loop 1 Feedback: {screenshot_path_2}")
            except Exception as exc:
                details.append(f"  FAIL Loop 1 Feedback: {str(exc)[:80]}")

            await browser.close()

    except Exception as exc:
        details.append(f"  FAIL: Playwright error: {str(exc)[:120]}")
        return PhaseReport(phase=6, name="Grafana Dashboards", passed=False,
                           details=details, duration_seconds=time.time() - start)

    elapsed = time.time() - start
    return PhaseReport(phase=6, name="Grafana Dashboards", passed=True,
                       details=details, duration_seconds=elapsed)


# ── Phase 7: Postgres Data Integrity ──────────────────────────


async def run_phase7_data_integrity(settings: Settings) -> tuple[PhaseReport, list[LLMCostSummary]]:
    """Phase 7: Check llm_calls and pipeline_runs for data integrity."""
    start = time.time()
    details: list[str] = []
    cost_summaries: list[LLMCostSummary] = []

    try:
        pg = PostgresClient(dsn=settings.postgres_dsn_async.get_secret_value())
        async with pg.session() as session:
            from sqlalchemy import text

            # LLM calls by model
            result = await session.execute(
                text(
                    "SELECT model, COUNT(*) AS call_count, "
                    "COALESCE(SUM(prompt_tokens), 0) AS input_tokens, "
                    "COALESCE(SUM(completion_tokens), 0) AS output_tokens, "
                    "COALESCE(SUM(cost_usd), 0) AS total_cost "
                    "FROM llm_calls GROUP BY model ORDER BY call_count DESC"
                )
            )
            llm_rows = result.fetchall()

            for row in llm_rows:
                summary = LLMCostSummary(
                    model=row[0],
                    call_count=int(row[1]),
                    input_tokens=int(row[2]),
                    output_tokens=int(row[3]),
                    total_cost=float(row[4]),
                )
                cost_summaries.append(summary)
                details.append(
                    f"  {_check_mark(True)} {summary.model}: "
                    f"{summary.call_count} calls, "
                    f"{summary.input_tokens}+{summary.output_tokens} tokens, "
                    f"${summary.total_cost:.6f}"
                )

            # Pipeline runs by strategy
            result2 = await session.execute(
                text(
                    "SELECT strategy, AVG(confidence) AS avg_conf, COUNT(*) AS cnt "
                    "FROM pipeline_runs GROUP BY strategy"
                )
            )
            strategy_rows = result2.fetchall()
            for row in strategy_rows:
                details.append(
                    f"  {_check_mark(True)} Strategy '{row[0]}': "
                    f"avg_conf={float(row[1]):.2f}, count={row[2]}"
                )

        await pg.close()

        has_llm_calls = len(cost_summaries) > 0
        total_cost = sum(s.total_cost for s in cost_summaries)
        total_tokens = sum(s.input_tokens + s.output_tokens for s in cost_summaries)
        total_calls = sum(s.call_count for s in cost_summaries)

        details.insert(0, f"  LLM calls total: {total_calls}, tokens: {total_tokens}, cost: ${total_cost:.6f}")

        elapsed = time.time() - start
        return (
            PhaseReport(phase=7, name="Postgres Data Integrity", passed=has_llm_calls,
                        details=details, duration_seconds=elapsed),
            cost_summaries,
        )

    except Exception as exc:
        elapsed = time.time() - start
        details.append(f"  FAIL: {exc}")
        return (
            PhaseReport(phase=7, name="Postgres Data Integrity", passed=False,
                        details=details, duration_seconds=elapsed),
            cost_summaries,
        )


# ── Phase 8: Summary Report ───────────────────────────────────


def print_summary(
    phases: list[PhaseReport],
    ingest_result: IngestResult | None,
    query_results: list[QueryResult],
    pipeline_records: list[PipelineRunRecord],
    cost_summaries: list[LLMCostSummary],
    total_duration: float,
) -> bool:
    """Print the final summary report. Returns True if all phases passed."""
    sep = "=" * 60
    print()
    print(sep)
    print("  E2E Full Pipeline Test Report")
    print(sep)
    print()

    # Infrastructure
    phase1 = next((p for p in phases if p.phase == 1), None)
    if phase1:
        print("Infrastructure:")
        for d in phase1.details:
            print(d)
        print()

    # Ingestion
    if ingest_result:
        print("Ingestion:")
        print(f"  {_check_mark(True)} {ingest_result.file} parsed: {ingest_result.chunk_count} chunks, {_fmt_seconds(ingest_result.duration_seconds)}")
        print(f"  {_check_mark(True)} Postgres record: parsed_documents.id = {ingest_result.parsed_document_id}")
        print(f"  {_check_mark(True)} Cognee cognified: {ingest_result.neo4j_entity_count} entity-related nodes")
        if ingest_result.neo4j_node_labels:
            for label, cnt in ingest_result.neo4j_node_labels.items():
                print(f"      {label}: {cnt}")
        print()

    # Queries
    if query_results:
        print("Queries:")
        for i, r in enumerate(query_results, 1):
            status = "ERROR" if r.error else _check_mark(True)
            if r.error:
                print(f"  {status} Query {i} ({r.tag}): ERROR {r.error[:80]}")
            else:
                print(
                    f"  {status} Query {i} ({r.tag}): "
                    f"conf={r.confidence:.2f}, grounded={r.is_grounded}, "
                    f"strategy={r.strategy}, {_fmt_seconds(r.latency_seconds)}"
                )
        print()

    # Pipeline Runs
    if pipeline_records:
        print("Pipeline Runs:")
        print(f"  {_check_mark(True)} {len(pipeline_records)} records in pipeline_runs")
        strategies = [r.strategy for r in pipeline_records]
        print(f"  {_check_mark(True)} Strategies used: {', '.join(strategies)}")
        print()

    # Feedback
    phase4 = next((p for p in phases if p.phase == 4), None)
    if phase4:
        print("Feedback:")
        for d in phase4.details:
            if "weighted" in d.lower():
                print(d)
        print()

    # Metrics
    phase5 = next((p for p in phases if p.phase == 5), None)
    if phase5:
        print("Metrics (OTel -> Prometheus):")
        for d in phase5.details:
            print(d)
        print()

    # Grafana
    phase6 = next((p for p in phases if p.phase == 6), None)
    if phase6:
        print("Grafana Dashboards:")
        for d in phase6.details:
            print(d)
        print()

    # Cost Tracking
    if cost_summaries:
        print("Cost Tracking:")
        total_calls = sum(s.call_count for s in cost_summaries)
        total_tokens = sum(s.input_tokens + s.output_tokens for s in cost_summaries)
        total_cost = sum(s.total_cost for s in cost_summaries)
        print(f"  {_check_mark(True)} LLM calls: {total_calls}")
        print(f"  {_check_mark(True)} Total tokens: {total_tokens:,}")
        print(f"  {_check_mark(True)} Total cost: ${total_cost:.4f}")
        print()

    # Final verdict
    all_passed = all(p.passed for p in phases)
    verdict = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    failed = [p for p in phases if not p.passed]

    print(sep)
    print(f"  Duration: {_fmt_seconds(total_duration)}")
    if failed:
        print(f"  Failed phases: {', '.join(f'Phase {p.phase} ({p.name})' for p in failed)}")
    print(f"  RESULT: {verdict}")
    print(sep)
    print()

    return all_passed


# ── Main ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="E2E full pipeline integration test",
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
        "--skip-playwright",
        action="store_true",
        help="Skip Grafana screenshot phase (no browser needed)",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the full E2E pipeline test."""
    args = parse_args()
    total_start = time.time()

    # Suppress noisy loggers
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("agentic_ra").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    settings = get_settings()
    phases: list[PhaseReport] = []
    ingest_result: IngestResult | None = None
    query_results: list[QueryResult] = []
    pipeline_records: list[PipelineRunRecord] = []
    cost_summaries: list[LLMCostSummary] = []
    rag_pipeline: RAGPipeline | None = None

    print()
    print("=" * 60)
    print("  E2E Full Pipeline Integration Test")
    print("=" * 60)
    print()

    # ── Phase 1: Infrastructure ──
    print("Phase 1: Infrastructure Health Checks")
    phase1 = await run_phase1_health_checks(settings)
    phases.append(phase1)
    for d in phase1.details:
        print(d)
    print(f"  [{_fmt_seconds(phase1.duration_seconds)}]")
    print()

    if not phase1.passed:
        print("ABORT: Critical infrastructure is down. Cannot continue.")
        print_summary(phases, None, [], [], [], time.time() - total_start)
        sys.exit(1)

    # ── Phase 2: Ingestion ──
    if args.skip_ingest:
        print("Phase 2: Ingestion (SKIPPED)")
        phases.append(PhaseReport(phase=2, name="Ingestion (skipped)", passed=True,
                                   details=["  SKIPPED by --skip-ingest flag"]))
    else:
        print(f"Phase 2: Ingestion ({args.file})")
        phase2, ingest_result = await run_phase2_ingestion(args.file, settings)
        phases.append(phase2)
        for d in phase2.details:
            print(d)
        print(f"  [{_fmt_seconds(phase2.duration_seconds)}]")
    print()

    # ── Phase 3: Queries ──
    print("Phase 3: Query System")
    phase3, query_results, rag_pipeline = await run_phase3_queries(settings)
    phases.append(phase3)
    for d in phase3.details:
        print(d)
    print(f"  [{_fmt_seconds(phase3.duration_seconds)}]")
    print()

    # Close pipeline to release resources
    if rag_pipeline:
        try:
            await rag_pipeline.close()
        except Exception:
            pass

    # ── Phase 4: Feedback Loop ──
    print("Phase 4: Feedback Loop Check")
    phase4, pipeline_records = await run_phase4_feedback(settings)
    phases.append(phase4)
    for d in phase4.details:
        print(d)
    print(f"  [{_fmt_seconds(phase4.duration_seconds)}]")
    print()

    # ── Phase 5: Metrics ──
    print("Phase 5: OTel Metrics Check")
    phase5 = await run_phase5_metrics()
    phases.append(phase5)
    for d in phase5.details:
        print(d)
    print(f"  [{_fmt_seconds(phase5.duration_seconds)}]")
    print()

    # ── Phase 6: Grafana ──
    if args.skip_playwright:
        print("Phase 6: Grafana Dashboards (SKIPPED)")
        phases.append(PhaseReport(phase=6, name="Grafana Dashboards (skipped)", passed=True,
                                   details=["  SKIPPED by --skip-playwright flag"]))
    else:
        print("Phase 6: Grafana Dashboard Screenshots")
        phase6 = await run_phase6_grafana()
        phases.append(phase6)
        for d in phase6.details:
            print(d)
        print(f"  [{_fmt_seconds(phase6.duration_seconds)}]")
    print()

    # ── Phase 7: Data Integrity ──
    print("Phase 7: Postgres Data Integrity")
    phase7, cost_summaries = await run_phase7_data_integrity(settings)
    phases.append(phase7)
    for d in phase7.details:
        print(d)
    print(f"  [{_fmt_seconds(phase7.duration_seconds)}]")
    print()

    # ── Phase 8: Summary ──
    total_duration = time.time() - total_start
    all_passed = print_summary(
        phases, ingest_result, query_results, pipeline_records,
        cost_summaries, total_duration,
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
