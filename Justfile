# agent-rs — task runner
# Usage:
#   just            # list recipes
#   just <recipe>   # run a recipe
#
# Recipes auto-load .env (via `set dotenv-load`). Copy .env.example to .env
# and fill in secrets before the first `just bootstrap`.

set dotenv-load := true
set shell := ["bash", "-cu"]

# ─── Defaults ─────────────────────────────────────────────────────────────

default:
    @just --list --unsorted

# ─── Setup ────────────────────────────────────────────────────────────────

# Copy .env.example to .env if it's missing. Won't overwrite existing.
env:
    @if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — edit OPENAI_API_KEY and AGENT_RS__LLM__API_KEY"; else echo ".env already exists"; fi

# Full first-time setup: copy .env, start infra, wait for health, run migrations.
bootstrap: env infra-up
    @echo "Waiting ~20s for services to be healthy..."
    @sleep 20
    just migrations
    @echo
    @echo "Bootstrap complete. Verify:"
    @echo "  docker compose ps"
    @echo "  just ingest test.pdf"

# ─── Infrastructure ───────────────────────────────────────────────────────

# Start all core services (qdrant, postgres, neo4j, docling-sidecar).
infra-up:
    docker compose up -d qdrant postgres neo4j docling-sidecar

# Stop + remove containers (volumes kept).
infra-down:
    docker compose down

# Stop + remove containers AND volumes (wipes all data).
infra-reset:
    docker compose down -v

# Tail logs for one service. Usage: `just logs postgres`
logs service:
    docker compose logs -f {{service}}

# Show the health + ports of every service.
status:
    docker compose ps

# ─── Observability stack (opt-in, not part of infra-up) ──────────────────

# Start the observability quartet: otel-collector + jaeger + prometheus + grafana.
# All four are required for traces/metrics from the Rust binaries to flow
# anywhere: OTLP → collector → jaeger (traces) + prometheus (metrics),
# grafana scrapes prometheus.
obs-up:
    docker compose up -d otel-collector jaeger prometheus grafana

# Stop observability services (volumes kept; dashboards preserved).
obs-down:
    docker compose stop otel-collector jaeger prometheus grafana
    docker compose rm -f otel-collector jaeger prometheus grafana

# Open the Jaeger trace-search UI in the default browser.
jaeger:
    @echo "Opening Jaeger UI: http://localhost:1058"
    @open http://localhost:1058 2>/dev/null || xdg-open http://localhost:1058 2>/dev/null || echo "Open manually: http://localhost:1058"

# Open the Prometheus query UI.
prom:
    @echo "Opening Prometheus UI: http://localhost:1066"
    @open http://localhost:1066 2>/dev/null || xdg-open http://localhost:1066 2>/dev/null || echo "Open manually: http://localhost:1066"

# Open Grafana (anonymous viewer enabled; admin creds from .env).
grafana:
    @echo "Opening Grafana UI: http://localhost:1067"
    @open http://localhost:1067 2>/dev/null || xdg-open http://localhost:1067 2>/dev/null || echo "Open manually: http://localhost:1067"

# Sanity-check: curl the collector's Prometheus scrape endpoint and the
# collector health-check. Useful after `just obs-up` to confirm the
# pipeline is live before running a query.
obs-smoke:
    @echo "=== collector health ==="
    @curl -s http://localhost:13133/ 2>&1 | head -3 || echo "collector NOT responding on :13133"
    @echo
    @echo "=== prometheus exporter (first 3 agentic_ra_* metrics) ==="
    @curl -s http://localhost:8889/metrics 2>&1 | grep -E '^agentic_ra_' | head -3 || echo "no agentic_ra_* metrics yet — run `just query '...'` first"
    @echo
    @echo "=== prometheus UI reachable? ==="
    @curl -sf http://localhost:1066/-/ready && echo "prometheus READY" || echo "prometheus NOT ready"
    @echo
    @echo "=== jaeger reachable? ==="
    @curl -sf http://localhost:1058/ > /dev/null && echo "jaeger UI UP" || echo "jaeger UI DOWN"

# ─── Database ─────────────────────────────────────────────────────────────

_db_url := "postgres://" + env_var_or_default("AGENT_RS__POSTGRES__USER", "agentic_ra") + \
           ":" + env_var_or_default("AGENT_RS__POSTGRES__PASSWORD", "changeme-postgres") + \
           "@" + env_var_or_default("AGENT_RS__POSTGRES__HOST", "localhost") + \
           ":" + env_var_or_default("AGENT_RS__POSTGRES__PORT", "1054") + \
           "/" + env_var_or_default("AGENT_RS__POSTGRES__DB", "agentic_ra")

# Run all pending Diesel migrations against the configured Postgres.
migrations:
    diesel migration run \
        --migration-dir crates/stores/migrations \
        --database-url '{{_db_url}}'

# List which migrations have been applied.
migrations-list:
    diesel migration list \
        --migration-dir crates/stores/migrations \
        --database-url '{{_db_url}}'

# Revert the most recent migration (destructive).
migrations-revert:
    diesel migration revert \
        --migration-dir crates/stores/migrations \
        --database-url '{{_db_url}}'

# Open an interactive psql prompt.
psql:
    docker exec -it agentic-ra-postgres psql -U "$AGENT_RS__POSTGRES__USER" -d "$AGENT_RS__POSTGRES__DB"

# ─── Docling native (bypass Docker to get MPS on Apple Silicon) ──────────

# Start the Docling sidecar natively on the host using a local Python venv.
# Apple's MPS accelerator cannot be reached from a Docker Linux VM, so to
# use Metal on macOS the sidecar must run here directly. The
# agentic-ra-docling-sidecar Docker container must be STOPPED first
# (run `docker compose stop docling-sidecar`).
#
# First time: creates .venv/docling-sidecar and installs deps. Subsequent
# runs reuse the venv.
docling-native:
    @test -d .venv/docling-sidecar || python3 -m venv .venv/docling-sidecar
    @.venv/docling-sidecar/bin/pip install --quiet --upgrade pip
    @.venv/docling-sidecar/bin/pip install --quiet -e infra/docling-sidecar
    @echo "Starting docling sidecar on host (MPS will be auto-detected if available)..."
    DOCLING_GRPC_HOST=0.0.0.0 \
    DOCLING_GRPC_PORT=1068 \
    DOCLING_TABLEFORMER_MODE=${DOCLING_TABLEFORMER_MODE:-fast} \
    DOCLING_ACCELERATOR=${DOCLING_ACCELERATOR:-auto} \
    DOCLING_MAX_PDF_BYTES=104857600 \
    DOCLING_MAX_PAGES=500 \
    DOCLING_PARSE_TIMEOUT_S=600 \
    DOCLING_STREAM_IDLE_TIMEOUT_S=600 \
    DOCLING_MAX_CHUNKS_PER_DOC=10000 \
    .venv/docling-sidecar/bin/python -m docling_sidecar

# ─── Dev loop ─────────────────────────────────────────────────────────────

# Ingest one document (PDF / txt / md). Usage: `just ingest test.pdf`
ingest path *args:
    cargo run --release --example ingest -- {{args}} {{path}}

# Run one retrieval query. Usage: `just query "ce este histograma?"`
# Extra flags go through:  `just query --strategy hybrid --limit 5 "my query"`
# Returns ranked chunks only — no LLM answer synthesis, no tools.
query *args:
    cargo run --release --example query -- {{args}}

# Ask a question through the full agent pipeline (tools + grounding +
# LLM answer synthesis with citations). Usage: `just ask "what is X?"`.
# Extra flags: --strategy <name> --limit N --user <id>.
ask *args:
    cargo run --release --example ask -- {{args}}

# A/B compare: weighted-keyword feature OFF then ON for the same query.
query-ab query:
    @echo "=== feature OFF =========================="
    EVAL_SKIP_WEIGHTED_KEYWORDS=1 cargo run --release --example query -- '{{query}}'
    @echo
    @echo "=== feature ON ==========================="
    cargo run --release --example query -- '{{query}}'

# ─── Quality gates ────────────────────────────────────────────────────────

# Fast workspace check (no codegen).
check:
    cargo check --workspace --all-targets

# Run unit + integration tests (no infra required).
test:
    cargo test --workspace --lib --tests

# Run clippy as an error-level gate.
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Apply formatting.
fmt:
    cargo fmt --all

# Check formatting without modifying.
fmt-check:
    cargo fmt --all -- --check

# Full pre-commit gate.
ci: fmt-check clippy test
    @echo "Gate passed."

# Build release binaries + examples.
build:
    cargo build --release --workspace --examples

# ─── Cleanup ──────────────────────────────────────────────────────────────

# Remove cargo target directory.
clean:
    cargo clean

# Nuke everything — containers, volumes, target. Requires confirmation.
nuke:
    @read -p "This deletes all docker volumes AND target/. Continue? [y/N] " c; \
     if [ "$c" = "y" ] || [ "$c" = "Y" ]; then \
        just infra-reset; \
        just clean; \
        echo "Nuked."; \
     else \
        echo "Aborted."; \
     fi
