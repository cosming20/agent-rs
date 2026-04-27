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

# Full first-time setup: copy .env, start infra, wait for health, run
# migrations, create MinIO bucket. Leaves every service running so the
# caller can follow up with `just serve` / `just worker` / `just web`.
bootstrap: env infra-up
    @echo "Waiting ~20s for services to be healthy..."
    @sleep 20
    just migrations
    just minio-bootstrap
    @echo
    @echo "Bootstrap complete. Next steps (in separate terminals):"
    @echo "  just serve     # agent-service gRPC on :1072"
    @echo "  just worker    # indexer-worker consumer"
    @echo "  just web       # agent-rs-web SSR on :3000 (sibling repo)"

# ─── Infrastructure ───────────────────────────────────────────────────────

# Create (or re-verify) the MinIO bucket used for document storage.
# Idempotent: mc alias set runs on every call; mc mb --ignore-existing is a
# no-op when the bucket already exists.
minio-bootstrap:
    docker compose exec -T minio mc alias set local http://localhost:9000 ${MINIO_ROOT_USER:-minioadmin} ${MINIO_ROOT_PASSWORD:-minioadmin} >/dev/null
    docker compose exec -T minio mc mb --ignore-existing local/${MINIO_BUCKET:-agent-rs-docs}
    @echo "MinIO bucket ${MINIO_BUCKET:-agent-rs-docs} ready"

# Start all core services (qdrant, postgres, neo4j, minio, rabbitmq,
# docling-sidecar). The stateless pivot adds MinIO (user document bytes)
# and RabbitMQ (indexing job queue between agent-service and
# indexer-worker) to the baseline stack.
#
# Also cascades into the sibling `../agent-rs-web` repo to bring up its
# own isolated Postgres (port 1055, deliberately adjacent to agent's
# 1054 for easy mental mapping) + Redis (port 1074). If the sibling
# repo isn't checked out, that step is a no-op with a hint.
infra-up:
    docker compose up -d qdrant postgres neo4j minio rabbitmq docling-sidecar
    @if [ -d ../agent-rs-web ]; then \
        echo "→ bringing up agent-rs-web infra (postgres on 1055, redis on 1074)..."; \
        (cd ../agent-rs-web && just infra-up); \
     else \
        echo "→ ../agent-rs-web not found; skip web infra. Clone it alongside this repo to enable."; \
     fi

# Stop + remove containers (volumes kept). Also cascades into
# ../agent-rs-web if present so a single command tears everything down.
infra-down:
    docker compose down
    @if [ -d ../agent-rs-web ]; then \
        (cd ../agent-rs-web && just infra-down); \
     fi

# Stop + remove containers AND volumes (wipes all data).
infra-reset:
    docker compose down -v

# Tail logs for one service. Usage: `just logs postgres`
logs service:
    docker compose logs -f {{service}}

# Show the health + ports of every service.
status:
    docker compose ps

# Open the RabbitMQ management UI (AMQP on :1075, mgmt on :1076).
rabbitmq-ui:
    @echo "Opening RabbitMQ management UI: http://localhost:1076"
    @open http://localhost:1076 2>/dev/null || xdg-open http://localhost:1076 2>/dev/null || echo "Open manually: http://localhost:1076"

# Open the MinIO web console (API on :1069, console on :1070).
minio-ui:
    @echo "Opening MinIO console: http://localhost:1070"
    @open http://localhost:1070 2>/dev/null || xdg-open http://localhost:1070 2>/dev/null || echo "Open manually: http://localhost:1070"

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
# pipeline is live before exercising the binaries.
obs-smoke:
    @echo "=== collector health ==="
    @curl -s http://localhost:13133/ 2>&1 | head -3 || echo "collector NOT responding on :13133"
    @echo
    @echo "=== prometheus exporter (first 3 agentic_ra_* metrics) ==="
    @curl -s http://localhost:8889/metrics 2>&1 | grep -E '^agentic_ra_' | head -3 || echo "no agentic_ra_* metrics yet"
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

_qdrant_rest_url := "http://" + env_var_or_default("AGENT_RS__QDRANT__HOST", "localhost") + \
                    ":" + env_var_or_default("AGENT_RS__QDRANT__REST_PORT", "1051")

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
    # Regenerate protobuf stubs into the installed source tree. Mirrors the
    # Dockerfile codegen step (Dockerfile:60-67) so Docker and native runs
    # stay behaviourally identical. Idempotent — grpc_tools.protoc overwrites
    # stubs each time. Required because `from agentic_rag.v1 import ...` in
    # server.py resolves against files generated here.
    @mkdir -p infra/docling-sidecar/src/agentic_rag/v1
    @touch infra/docling-sidecar/src/agentic_rag/__init__.py
    @touch infra/docling-sidecar/src/agentic_rag/v1/__init__.py
    @.venv/docling-sidecar/bin/python -m grpc_tools.protoc \
        --proto_path=proto \
        --python_out=infra/docling-sidecar/src \
        --grpc_python_out=infra/docling-sidecar/src \
        proto/agentic_rag/v1/chunker.proto
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

# Start the Marker sidecar natively on the host using a local Python venv.
# Marker is MPS-native on Apple Silicon (surya layout + texify formulas)
# and avoids Docling's CodeFormulaV2 bottleneck — formula-heavy PDFs
# ingest ~10-20× faster on a Mac. Uses port 1073 (1069 is taken by
# MinIO in the docker-compose dev stack); Docling on 1068 can stay up
# in parallel for A/B comparison.
#
# First time: creates .venv/marker-sidecar, installs deps, compiles
# protobuf stubs into the installed source tree (mirrors the Dockerfile
# codegen step). Subsequent runs reuse the venv but always regenerate
# the stubs so proto edits do not drift.
#
# Usage: `just marker-native`
#        `AGENT_RS__INGEST_BACKEND=marker just worker`    # point the worker at it
#        `MARKER_USE_LLM=true just marker-native`         # enable Marker's LLM enrichment
marker-native:
    @test -d .venv/marker-sidecar || python3 -m venv .venv/marker-sidecar
    @.venv/marker-sidecar/bin/pip install --quiet --upgrade pip
    @.venv/marker-sidecar/bin/pip install --quiet -e infra/marker-sidecar
    @mkdir -p infra/marker-sidecar/src/agentic_rag/v1
    @touch infra/marker-sidecar/src/agentic_rag/__init__.py
    @touch infra/marker-sidecar/src/agentic_rag/v1/__init__.py
    @.venv/marker-sidecar/bin/python -m grpc_tools.protoc \
        --proto_path=proto \
        --python_out=infra/marker-sidecar/src \
        --grpc_python_out=infra/marker-sidecar/src \
        proto/agentic_rag/v1/chunker.proto
    @echo "Starting Marker sidecar on host (MPS will be auto-detected if available)..."
    MARKER_GRPC_HOST=0.0.0.0 \
    MARKER_GRPC_PORT=1073 \
    MARKER_MAX_PDF_BYTES=104857600 \
    MARKER_MAX_PAGES=500 \
    MARKER_PARSE_TIMEOUT_S=600 \
    MARKER_STREAM_IDLE_TIMEOUT_S=600 \
    MARKER_MAX_CHUNKS_PER_DOC=10000 \
    .venv/marker-sidecar/bin/python -m marker_sidecar

# Bootstrap a remote Docling sidecar on a GPU host and tunnel it to
# localhost:1068. Uses scripts/docling-remote.sh which rsyncs the
# sidecar source, `docker compose up --build`s it with `--gpus all`,
# and opens a persistent SSH port-forward so the local worker is
# unchanged.
#
# Requirements on the remote host:
#   - Docker + NVIDIA runtime (`docker run --gpus all` must work)
#   - rsync + git
#   - SSH access via key (scripts use BatchMode — no password prompts)
#
# Usage:
#   just docling-remote ubuntu@runpod-12345.ssh.runpod.io
#   just docling-remote-down ubuntu@runpod-12345.ssh.runpod.io
#
# After `up` completes, run `just worker` normally — the DoclingConfig
# endpoint is already `http://localhost:1068`, which is now tunneled
# to the remote GPU box. Per-formula latency drops from ~60s on Mac
# CPU to ~2-5s on a cloud A10/A10G.
docling-remote host:
    ./scripts/docling-remote.sh {{host}} up

docling-remote-down host:
    ./scripts/docling-remote.sh {{host}} down

# ─── Runtime binaries ─────────────────────────────────────────────────────

# Run the agent gRPC service (port 1072 by default).
#
# Usage: `just serve`            # release build
#        `RELEASE=0 just serve`  # debug build
serve:
    cargo run {{ if env_var_or_default("RELEASE", "1") == "1" { "--release" } else { "" } }} -p agent-rs-agent-service --bin agent-service

# Run one indexer-worker replica. Scale horizontally by starting the
# recipe in multiple terminals — the broker fans out jobs round-robin.
#
# Usage: `just worker`
#        `AGENT_RS__INDEXER__PREFETCH=10 just worker`   # bump concurrency
#        `AGENT_RS__INDEXER__ENABLE_SUMMARIES=true just worker`   # LLM stages
worker:
    cargo run {{ if env_var_or_default("RELEASE", "1") == "1" { "--release" } else { "" } }} -p agent-rs-agent-service --bin indexer-worker

# Run the SSR web frontend from the sibling repo (expects
# ../agent-rs-web on the same branch). `cargo leptos watch` hot-reloads
# on source edits.
#
# Usage: `just web`
# Requires: `cargo install cargo-leptos` + a sibling checkout of
# github.com/cosming20/agent-rs-web.
web:
    cd ../agent-rs-web && cargo leptos watch

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

# Build release binaries.
build:
    cargo build --release --workspace

# ─── Cleanup ──────────────────────────────────────────────────────────────

# In-place wipe of all ingested data across Neo4j, Qdrant, and Postgres.
# Containers keep running, Diesel migrations + DB schema are preserved,
# `target/` is untouched — fastest path to a clean slate between ingest
# iterations. Requires confirmation. Uses `FORCE=1` to skip the prompt in
# scripts (e.g. `FORCE=1 just data-wipe`).
data-wipe:
    @if [ "${FORCE:-0}" != "1" ]; then \
        read -p "Wipe ALL data from Neo4j + Qdrant + Postgres? (schema kept, target/ kept) [y/N] " c; \
        if [ "$c" != "y" ] && [ "$c" != "Y" ]; then echo "Aborted."; exit 1; fi; \
     fi
    @echo "→ Neo4j: deleting all nodes + relationships..."
    docker exec agentic-ra-neo4j cypher-shell \
        -u "$AGENT_RS__NEO4J__USER" -p "$AGENT_RS__NEO4J__PASSWORD" \
        'MATCH (n) DETACH DELETE n;'
    @echo "→ Qdrant: listing + dropping every collection..."
    @for c in $(curl -s -H "api-key: $AGENT_RS__QDRANT__API_KEY" {{_qdrant_rest_url}}/collections | python3 -c 'import json,sys; print("\n".join(c["name"] for c in json.load(sys.stdin)["result"]["collections"]))'); do \
        echo "   drop collection: $c"; \
        curl -s -H "api-key: $AGENT_RS__QDRANT__API_KEY" -X DELETE {{_qdrant_rest_url}}/collections/$c > /dev/null; \
     done
    @echo "→ Postgres: truncating application tables (schema + migrations kept)..."
    docker exec agentic-ra-postgres psql \
        -U "$AGENT_RS__POSTGRES__USER" -d "$AGENT_RS__POSTGRES__DB" \
        -c 'TRUNCATE document_index_state, evidence_items, query_runs, memories, strategy_performance RESTART IDENTITY CASCADE;'
    @echo "→ RabbitMQ: purging the indexing queue..."
    @docker exec agentic-ra-rabbitmq rabbitmqctl purge_queue agent-rs.index 2>/dev/null || echo "   (queue not yet declared; skip)"
    @echo "→ MinIO: clearing the documents bucket (bucket itself kept)..."
    @docker compose exec -T minio mc alias set local http://localhost:9000 ${MINIO_ROOT_USER:-minioadmin} ${MINIO_ROOT_PASSWORD:-minioadmin} >/dev/null 2>&1 || true
    @docker compose exec -T minio mc rm --recursive --force --quiet local/${MINIO_BUCKET:-agent-rs-docs} 2>/dev/null || echo "   (bucket empty or missing; skip)"
    @if [ -d ../agent-rs-web ]; then \
        echo "→ web Postgres: cascading data-wipe into ../agent-rs-web..."; \
        (cd ../agent-rs-web && FORCE=1 just data-wipe); \
     else \
        echo "→ ../agent-rs-web not found; skip web data-wipe."; \
     fi
    @echo "Data wipe complete. target/ untouched; containers still running."

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
