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

# Verify all store connections (Postgres, Qdrant, OpenSearch)
stores-check:
    uv run python scripts/stores_check.py

# Reset Qdrant collection + OpenSearch index
stores-reset:
    uv run python scripts/stores_reset.py

# ─── Ingest ──────────────────────────────────────────────────

# Ingest sample documents for testing (server must be running)
ingest-sample:
    uv run python scripts/ingest_sample.py

# Ingest file(s) — full pipeline: parse → chunk → embed → store → persist
# Usage: just ingest-file report.pdf          (single file)
#        just ingest-file a.pdf b.docx c.html (multiple files)
#        just ingest-file https://arxiv.org/pdf/2408.09869 (URL)
ingest-file +FILES:
    uv run python scripts/ingest_file.py {{ FILES }}

# ─── Query ───────────────────────────────────────────────────

# Quick query from CLI: just query "What is Python?"
query QUERY STRATEGY="hybrid":
    @uv run python -c "import httpx,json; r=httpx.post('http://127.0.0.1:1059/api/v1/query',json={'query':'{{ QUERY }}','strategy':'{{ STRATEGY }}'},timeout=60); d=r.json(); print(json.dumps(d,indent=2))"

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
    @echo "Cleaned"

# Full reset: clean + infra reset + reinstall
nuke: clean infra-reset install
    @echo "Nuked. Run: just bootstrap"
