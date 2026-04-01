# Agentic RAG — Task Runner
# Usage: just <recipe>    |    just --list

set dotenv-load := false
set shell := ["bash", "-cu"]

# Default: show available recipes
default:
    @just --list --unsorted

# ─── Setup ────────────────────────────────────────────────────

# Install all dependencies
install:
    uv sync

# Create .env from example (won't overwrite existing)
env:
    @if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — edit OPENAI_API_KEY and secrets"; else echo ".env already exists"; fi

# Full bootstrap: install + env + infra + migrate + stores
bootstrap: install env infra-up
    @echo "Waiting for services to be healthy..."
    @sleep 15
    just migrate
    just stores-reset
    @echo "Bootstrap complete. Set OPENAI_API_KEY in .env, then: just server"

# ─── Infrastructure ──────────────────────────────────────────

# Start all Docker services
infra-up:
    cd infra && docker compose up -d

# Stop all Docker services
infra-down:
    cd infra && docker compose down

# Stop and remove all volumes (full reset)
infra-reset:
    cd infra && docker compose down -v
    @echo "All volumes destroyed. Run: just infra-up && just migrate"

# Show Docker service status
infra-status:
    @docker ps --filter "name=agentic-ra" --format "table {{"{{"}}{{".Names"}}{{"}}"}}	{{"{{"}}{{".Status"}}{{"}}"}}"

# Tail Docker logs (all services or specific: just infra-logs postgres)
infra-logs *ARGS:
    cd infra && docker compose logs -f {{ ARGS }}

# ─── Database ────────────────────────────────────────────────

# Run all pending Alembic migrations
migrate:
    uv run alembic upgrade head

# Rollback last migration
migrate-rollback:
    uv run alembic downgrade -1

# Show current migration status
migrate-status:
    uv run alembic current

# Generate a new migration (auto-detect model changes)
migrate-create MESSAGE:
    uv run alembic revision --autogenerate -m "{{ MESSAGE }}"

# ─── Dev Server ──────────────────────────────────────────────

# Start the FastAPI server (foreground, with reload)
server:
    uv run uvicorn agentic_ra.gateway.app:create_app --factory --host 127.0.0.1 --port 1059 --reload

# Start the FastAPI server (background)
server-bg:
    @uv run uvicorn agentic_ra.gateway.app:create_app --factory --host 127.0.0.1 --port 1059 &
    @echo "Server starting on http://127.0.0.1:1059"

# Stop background server
server-stop:
    @pkill -f "uvicorn agentic_ra" 2>/dev/null || echo "No server running"

# Check server health
health:
    @curl -sf http://127.0.0.1:1059/health | python3 -m json.tool 2>/dev/null || echo "Server not running"

# Open API docs in browser
docs:
    @open http://127.0.0.1:1059/docs

# Open the web UI then start server in foreground (Ctrl+C to stop)
web:
    @open http://127.0.0.1:1059/ui/
    uv run uvicorn agentic_ra.gateway.app:create_app --factory --host 127.0.0.1 --port 1059 --reload

# ─── Quality ─────────────────────────────────────────────────

# Run all quality checks (lint + typecheck)
check: lint typecheck
    @echo "All checks passed"

# Run ruff linter
lint:
    uv run ruff check src/

# Run ruff formatter
fmt:
    uv run ruff format src/

# Run mypy strict type checking
typecheck:
    uv run mypy --strict src/agentic_ra/

# Fix auto-fixable lint issues
lint-fix:
    uv run ruff check --fix src/

# ─── Store Management ────────────────────────────────────────

# Verify all store connections (Postgres, Qdrant, Neo4j)
stores-check:
    uv run python scripts/stores_check.py

# Reset Qdrant collection + Cognee knowledge graph
stores-reset:
    uv run python scripts/stores_reset.py

# ─── Ingest ──────────────────────────────────────────────────

# Test Docling chunking only — outputs docling_chunks.json for inspection
# Usage: just docling test.pdf
docling FILE:
    uv run python scripts/docling_test.py {{ FILE }}

# Test Docling + chunking on a CDEP crawled page from the database
# Usage: just docling-cdep                              (random page)
#        just docling-cdep --page-type deputy_profile    (by type)
#        just docling-cdep --page-id 42                  (by ID)
#        just docling-cdep --search "comisia juridica"   (search text)
docling-cdep *ARGS:
    uv run python scripts/docling_test_cdep.py {{ ARGS }}

# Ingest sample documents for testing (server must be running)
ingest-sample:
    uv run python scripts/ingest_sample.py

# Ingest file(s) — full pipeline: parse → chunk → embed → store → persist
# Usage: just ingest-file report.pdf          (single file)
#        just ingest-file a.pdf b.docx c.html (multiple files)
#        just ingest-file https://arxiv.org/pdf/2408.09869 (URL)
ingest-file +FILES:
    uv run python scripts/ingest_file.py {{ FILES }}

# Ingest crawled CDEP pages from the crawler database
# Usage: just ingest-cdep --dry-run --limit 3         (preview)
#        just ingest-cdep --priority high              (deputies, committees)
#        just ingest-cdep --page-types deputy_profile  (specific type)
#        just ingest-cdep --depth 2                    (BFS: depth 0+1 all, depth 2 high-priority)
#        just ingest-cdep --depth 2 --dry-run --limit 5
ingest-cdep *ARGS:
    uv run python scripts/ingest_cdep.py {{ ARGS }}

# Wipe all Cognee data: Qdrant collections + Neo4j graph + local files
# Use before re-ingesting from scratch
cognee-reset:
    uv run python scripts/cognee_reset.py

# Run E2E validation: ingest test.pdf + query via gateway
# Requires: infra running (just infra-up), server running (just server)
e2e-test *ARGS:
    uv run python scripts/e2e_ingest_and_query.py {{ ARGS }}

# Run E2E queries only (skip ingestion)
e2e-query:
    uv run python scripts/e2e_ingest_and_query.py --skip-ingest

# ─── Query ───────────────────────────────────────────────────

# Quick query from CLI: just query "What is Python?"
# Strategy values: semantic, keyword, hybrid, graph, graph_hybrid
query QUERY STRATEGY="hybrid":
    @uv run python -c "import httpx,json; r=httpx.post('http://127.0.0.1:1059/api/v1/query',json={'query':'{{ QUERY }}','strategy':'{{ STRATEGY }}'},timeout=60); d=r.json(); print(json.dumps(d,indent=2))"

# ─── Cognee (Knowledge Graph) ───────────────────────────────

# Search Cognee knowledge graph directly
cognee-search QUERY SEARCH_TYPE="GRAPH_COMPLETION" TOP_K="5":
    @uv run python -c "import asyncio,json; from agentic_ra.core.config import get_settings; from agentic_ra.stores.cognee.client import CogneeStore; s=get_settings(); cs=CogneeStore(dataset=s.cognee_dataset); cs.configure(llm_api_key=s.llm_api_key.get_secret_value(), llm_model=s.llm_enrichment_model, graph_db_provider=s.cognee_graph_db_provider, vector_db_provider=s.cognee_vector_db_provider, data_root=s.cognee_data_root, system_root=s.cognee_system_root, qdrant_url=s.qdrant_url, qdrant_api_key=s.qdrant_api_key.get_secret_value(), neo4j_bolt_url=s.neo4j_bolt_url, neo4j_user=s.neo4j_user, neo4j_password=s.neo4j_password.get_secret_value(), neo4j_db=s.neo4j_db); docs=asyncio.run(cs.search('{{ QUERY }}','{{ SEARCH_TYPE }}',int('{{ TOP_K }}'))); print(json.dumps([{'content':d.content[:200],'meta':d.meta} for d in docs],indent=2,default=str))"

# Query with graph_hybrid strategy (vector + keyword + graph fusion)
query-graph QUERY:
    @uv run python -c "import httpx,json; r=httpx.post('http://127.0.0.1:1059/api/v1/query',json={'query':'{{ QUERY }}','strategy':'graph_hybrid'},timeout=120); d=r.json(); print(json.dumps(d,indent=2))"

# Reset Cognee knowledge base (prune graph + vectors)
cognee-prune:
    @uv run python -c "import asyncio; from agentic_ra.core.config import get_settings; from agentic_ra.stores.cognee.client import CogneeStore; s=get_settings(); cs=CogneeStore(dataset=s.cognee_dataset); cs.configure(llm_api_key=s.llm_api_key.get_secret_value(), llm_model=s.llm_enrichment_model, graph_db_provider=s.cognee_graph_db_provider, vector_db_provider=s.cognee_vector_db_provider, data_root=s.cognee_data_root, system_root=s.cognee_system_root, qdrant_url=s.qdrant_url, qdrant_api_key=s.qdrant_api_key.get_secret_value(), neo4j_bolt_url=s.neo4j_bolt_url, neo4j_user=s.neo4j_user, neo4j_password=s.neo4j_password.get_secret_value(), neo4j_db=s.neo4j_db); asyncio.run(cs.prune()); print('Cognee knowledge base pruned')"

# ─── MCP ─────────────────────────────────────────────────────

# Start internal MCP search server (stdio)
mcp-server:
    uv run python -m agentic_ra.tools.mcp_servers.search_server

# Test MCP client → server round trip
mcp-test:
    uv run python scripts/mcp_test.py

# ─── Observability ───────────────────────────────────────────

# Open Jaeger tracing UI in browser
jaeger:
    @open http://localhost:1058

# Check OTel collector logs
otel-status:
    @docker logs agentic-ra-otel-collector --tail 10 2>&1 || echo "OTel collector not running"

# ─── Data Inspection ─────────────────────────────────────────

# Show ingested document stats from Postgres
data-stats:
    uv run python scripts/data_stats.py

# List all parsed documents stored in Postgres
data-list:
    uv run python scripts/data_list.py

# Preview chunks stored in Qdrant
data-chunks LIMIT="5":
    uv run python scripts/data_chunks.py {{ LIMIT }}

# ─── Clean ───────────────────────────────────────────────────

# Remove Python caches and build artifacts
clean:
    @find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    @find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
    @rm -rf .pytest_cache dist build *.egg-info
    @rm -rf .cognee_data .cognee_system
    @echo "Cleaned"

# Full reset: clean + infra reset + reinstall
nuke: clean infra-reset install
    @echo "Nuked. Run: just bootstrap"
