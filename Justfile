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

# ─── Dev loop ─────────────────────────────────────────────────────────────

# Ingest one document (PDF / txt / md). Usage: `just ingest test.pdf`
ingest path *args:
    cargo run --release --example ingest -- {{args}} {{path}}

# Run one retrieval query. Usage: `just query "ce este histograma?"`
# Extra flags go through:  `just query --strategy hybrid --limit 5 "my query"`
query *args:
    cargo run --release --example query -- {{args}}

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
