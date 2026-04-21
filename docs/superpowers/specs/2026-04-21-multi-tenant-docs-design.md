# Multi-Tenant Per-User Documents + MinIO Object Storage

**Status:** Draft for review
**Date:** 2026-04-21
**Topic:** Strict per-user document library with per-conversation doc selection, MinIO object storage, and end-to-end user-scoped retrieval (Qdrant + Neo4j + Postgres + agent memory).
**Repo:** `/Users/cosmingagea/workspace/agent-rs`
**Branch target:** single PR off `main` after merge

---

## 1. Problem statement

The current agent-rs exposes a single global document corpus. Every authenticated user can retrieve every ingested chunk, every graph node, every stored memory. For a SaaS deployment this is a hard security failure: user A can issue `search_knowledge` and see user B's confidential document.

Per-query scoping is needed across four surfaces:

1. **Document library** (ingested PDFs) — owner = user, visible only to that user's sessions.
2. **Conversation activation** — a user selects which subset of their library participates in a given conversation.
3. **Agent memory** — both long-term (`memories` table, via `RememberTool`) and short-term (`chat_messages`) — scoped by user.
4. **Object storage** — PDF bytes must persist in MinIO at `users/{user_id}/docs/{document_id}.{ext}` for re-ingest and download-back.

Current state has none of this. `ingested_documents`, `chat_messages`, `memories` have no `user_id` column. `session_id` is an unscoped UUID. Qdrant chunks and Neo4j Document nodes have no owner field.

## 2. Goals

- Strict multi-tenancy: cross-user access = security violation. Enforced at every retrieval layer, not just UX.
- Persistent per-user document library (upload once, reuse across conversations).
- Per-conversation document activation — user chooses which docs participate in the active session.
- MinIO object storage for original PDF bytes — keep forever at `users/{user_id}/docs/{document_id}.{ext}`.
- CLI ergonomics: `just ask "<user_id>" "<session_id>" "<question>"` — three positional args.

## 3. Non-goals

- Authentication / authorization middleware (no JWT, no OAuth; user_id comes from CLI arg for now — future API layer will inject it).
- Document sharing between users.
- Document versioning / history.
- Quota / rate limits per user.
- Admin UX for listing all users' data.
- Migration of existing data — every ingested document must be re-ingested under a user_id. No legacy-user fallback.

## 4. Decisions (locked with user)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Reingest everything**, no legacy-data migration | Dev repo, clean slate; avoid migration debt |
| 2 | **MinIO keep-forever** at `users/{u}/docs/{d}.{ext}` | Users expect their library to persist; re-ingest without re-upload |
| 3 | **Full 3×3 security matrix** (3 stores × {read, write, delete}) + boundary tests | Strict isolation justifies it |
| 4 | **Explicit plumbing** of `user_id: UserId` through every store call | Compile-time enforcement, no ambient context; Rust's strong suit |
| 5 | **Diesel migrations** in `crates/stores/migrations/` per existing `justfile:migrations` recipe | Match the existing pattern |
| 6 | **MinIO self-hosted** on ports **1068 (API) + 1069 (Console)** | Adjacent to existing 1050-1067 infra block |

## 5. Architecture

### 5.1 New / extended stores

**MinIO** (`crates/stores/src/minio/`) — new submodule under `agent-rs-stores` (same crate; avoids adding a crate-level dependency explosion):

- `MinioStore::new(endpoint, access_key, secret_key, bucket) -> Result<Self, StoreError>`
- `MinioStore::put_object(user_id, document_id, bytes, content_type) -> Result<ObjectKey, StoreError>`
- `MinioStore::get_object(user_id, document_id) -> Result<Bytes, StoreError>`
- `MinioStore::delete_object(user_id, document_id) -> Result<(), StoreError>`
- Key layout: `users/{user_id}/docs/{document_id}.{ext}` — derived deterministically from `user_id + document_id + mimetype`
- Bucket name from config (default `agent-rs-docs`)
- Uses `aws-sdk-s3` (S3-compatible — works with MinIO, AWS, R2, GCS)

**Postgres** — new migrations in `crates/stores/migrations/2026-04-21-0001_multi_tenant/`:

```sql
-- up.sql
ALTER TABLE ingested_documents ADD COLUMN user_id TEXT NOT NULL;
ALTER TABLE ingested_documents ADD COLUMN minio_object_key TEXT;
CREATE INDEX idx_ingested_documents_user_id ON ingested_documents(user_id);

ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL;
CREATE INDEX idx_memories_user_id ON memories(user_id);

ALTER TABLE chat_messages ADD COLUMN user_id TEXT NOT NULL;
CREATE INDEX idx_chat_messages_user_id ON chat_messages(user_id);
-- session_id stays as-is on chat_messages; the (user_id, session_id) pair authorises reads.

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);

CREATE TABLE session_active_documents (
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES ingested_documents(document_id) ON DELETE CASCADE,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, document_id)
);
-- App-layer enforcement: document_id must belong to session.user_id.
-- Enforced in PgStore::activate_document_for_session via a single-query CTE.
```

Since Decision #1 is "reingest everything", `NOT NULL` without `DEFAULT` is fine — tables are wiped first.

**Qdrant** — every chunk payload gains:
- `user_id: String` — required
- `doc_id: String` — already partially present
- `section_id: String` — already present

Every filter in `retrieval::router` ANDs on `user_id == ctx.user_id`. User filter is applied **per branch** (not post-fusion), so keyword + graph branches cannot leak cross-tenant results.

**Neo4j** — every `Document` node gets a `user_id` property. Every Cypher query gains a `WHERE d.user_id = $user_id` clause at the document-match level. This propagates down the structural hierarchy (Page/Section/Chunk inherit authorisation by being descendants of a user-scoped Document).

### 5.2 Filter propagation (explicit plumbing)

The `UserId` newtype already exists in `agent_rs_core::types`. Every store method that reads or writes user data gains an explicit `user_id: &UserId` parameter.

- `PgStore::list_ingested_documents(user_id: &UserId) -> Result<Vec<IngestedDocumentRow>, _>`
- `PgStore::list_memories(user_id: &UserId) -> Result<Vec<Memory>, _>`
- `PgStore::save_memory(user_id: &UserId, content: &str) -> Result<_, _>`
- `PgStore::list_chat_messages(user_id: &UserId, session_id: &SessionId) -> Result<Vec<ChatMessageRow>, _>` — requires BOTH user_id and session_id; the query filters by AND so a session belonging to user B cannot be read by user A even if they guess the session_id.
- `PgStore::get_session(user_id: &UserId, session_id: &SessionId) -> Result<Option<Session>, _>` — returns None if the session belongs to a different user (does not leak existence).
- `Neo4jStore::list_sections_for_doc(user_id: &UserId, doc_id: &str) -> _`
- `Neo4jStore::chunks_in_sections(user_id: &UserId, section_ids: &[String]) -> _`
- `QdrantStore::retrieve_chunks_by_ids(user_id: &UserId, collection: &str, chunk_ids: &[String]) -> _`
- `QdrantStore::search(user_id: &UserId, ...)` — adds user filter at query time.
- `MinioStore::put_object(user_id: &UserId, document_id: Uuid, ...)` — key is user-scoped.

`UserId` is `#[derive(Debug, Clone, PartialEq, Eq, Hash)] pub struct UserId(String)` — newtype prevents accidentally passing `session_id` or `doc_id` where `user_id` is expected.

### 5.3 Agent pipeline changes

`QueryContext` already has `user_id: UserId` and `session_id: SessionId`. Today neither is consulted by stores. After this spec:

- `ContextBuilder` reads:
  - `sessions` row (create if missing, enforce `user_id` match)
  - `chat_messages` scoped by `(user_id, session_id)`
  - `memories` scoped by `user_id`
  - `session_active_documents` — the authorised subset of user's library to use in this conversation
- `ctx.active_document_ids: Option<Vec<Uuid>>` — loaded from `session_active_documents`
  - `None` = user has not selected any docs → fall back to ALL user's docs (reasonable default for a first-time conversation)
  - `Some(subset)` = use only these documents
- `CoveragePlanner` gets `user_id` from `ctx`; calls `PgStore::list_ingested_documents(user_id)` and filters by `ctx.active_document_ids` if set.
- `ToolFactory` constructs tools with `user_id` + `active_document_ids` baked in at request scope (re-built per query, not stored globally).
- `AgentRunner` passes `user_id` into every tool's constructor.

### 5.4 Tools

All five retrieval tools (`search_knowledge`, `outline_corpus`, `fetch_section_chunks`, `search_sections`, `document_info`) take `user_id: UserId` at construction time. Every `call()` body adds it to the filter.

`RememberTool::call` passes `user_id` to `PgStore::save_memory`.

Tool JSON schemas stay unchanged — `user_id` is not an LLM-visible parameter (the LLM never chooses it; it is pre-baked).

### 5.5 CLI / `just ask`

Signature changes from the current `--strategy X --limit N --user-id U "question"` to three positional args:

```bash
just ask "<user_id>" "<session_id>" "<question>"
```

`examples/ask.rs::parse_args` rewritten to take 3 positional args. Optional flags (`--strategy`, `--limit`) remain available after positional args. An empty `session_id` string generates a fresh UUID and creates a `sessions` row for the user.

Similarly `just ingest`:

```bash
just ingest "<user_id>" "<path/to/doc.pdf>"
```

- PDF bytes uploaded to MinIO at `users/{user_id}/docs/{document_id}.pdf` BEFORE starting the Docling pipeline.
- If ingest fails mid-way, MinIO blob stays — user can retry.
- If ingest fails terminally (Docling rejects the doc), MinIO blob stays as a record of the attempt; user can re-upload to trigger re-ingest.

### 5.6 Infrastructure

`docker-compose.yml` gains a MinIO service:

```yaml
  minio:
    image: minio/minio:RELEASE.2025-03-12T18-04-18Z
    ports:
      - "${MINIO_API_PORT:-1068}:9000"
      - "${MINIO_CONSOLE_PORT:-1069}:9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin}
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  minio_data:
```

`.env.example` entries:

```
MINIO_API_PORT=1068
MINIO_CONSOLE_PORT=1069
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET=agent-rs-docs
MINIO_ENDPOINT=http://localhost:1068
AGENT_RS__MINIO__ENDPOINT=http://localhost:1068
AGENT_RS__MINIO__ACCESS_KEY=minioadmin
AGENT_RS__MINIO__SECRET_KEY=minioadmin
AGENT_RS__MINIO__BUCKET=agent-rs-docs
```

First-boot migration: `justfile:bootstrap` creates the `agent-rs-docs` bucket via `mc mb` after MinIO health-check passes.

## 6. Security tests (full 3×3 matrix)

| Layer | READ (list / query) | WRITE (ingest / save) | DELETE |
|-------|---------------------|------------------------|--------|
| **Postgres** | `list_ingested_documents(B)` must NOT return A's docs | `save_memory(B, text)` must NOT land under A | `delete_document(B, A_doc_id)` must fail with auth error |
| **Qdrant** | `search(B, query)` must NOT return A's chunks | `upsert(B_user_id_payload, ...)` does not leak when someone searches with A | `delete_by_doc(B, A_doc_id)` must fail |
| **Neo4j** | `list_sections_for_doc(B, A_doc_id)` must return empty | `upsert Document(B, ...)` cannot link to A's chunks | `delete_document_graph(B, A_doc_id)` must fail |

Plus boundary tests:

- Empty `user_id` string → every store call rejects with `StoreError::InvalidUserId`
- SQL-injection attempts in `user_id` (`"' OR 1=1 --"`) → rejected at newtype construction (validator allows only `[A-Za-z0-9_-]{1,128}`)
- Session hijack: `ask(user=A, session=B_owned_session, …)` → rejected at ContextBuilder's `sessions` row lookup
- Concurrent writes: user A writes `memories`, user B reads `memories` in parallel → B never sees A's row (prop-test with `loom` or counter-based assertion)

Implementation location: `crates/stores/tests/multi_tenant_security.rs` (new integration test file, live Postgres + Qdrant + Neo4j via docker services).

## 7. Error model

New variants on `agent_rs_core::errors::StoreError`:

- `StoreError::InvalidUserId(String)` — newtype validator rejected input
- `StoreError::SessionNotFound { session_id, user_id }` — no row, OR row belongs to different user (same error — do not leak existence)
- `StoreError::DocumentNotOwnedByUser { document_id, user_id }` — user attempted to activate a doc that is not theirs
- `StoreError::Minio(aws_sdk_s3::Error)` — transparent wrap

## 8. Migration

Decision #1 makes this simple:

```bash
just infra-reset      # drops and recreates Postgres/Qdrant/Neo4j volumes
just migrations       # runs the new user_id migrations
just ingest "demo-user-001" "segmentare.pdf"
just ask "demo-user-001" "" "vreau cheatsheet cu TOATE metodele de segmentare"
```

A note in the README documents the reingest-required rollout step.

## 9. Testing strategy

- **Unit tests** (per crate) — newtype validator, filter construction, key derivation for MinIO.
- **Integration tests** — the 3×3 matrix + boundary tests in `crates/stores/tests/multi_tenant_security.rs`. Requires live docker services; `just test-secure` recipe runs it.
- **End-to-end** — `examples/ask.rs` run manually by the user with two user_ids, confirming A's query sees only A's docs.

The user explicitly opted into the full security matrix (Decision #3), so this is the testing baseline — not a "nice to have".

## 10. Rollout

Single PR off `main`. Additive migrations (columns + tables), but existing data is wiped per Decision #1. No feature flag.

**Order of landing (within the PR, across commits):**

1. Newtypes + errors (small, purely type-level)
2. Postgres migrations + schema.rs regeneration
3. Diesel-layer store methods (PgStore) with `user_id` threaded
4. MinIO store + docker-compose
5. Qdrant + Neo4j store methods with `user_id` filtering
6. Retrieval router changes
7. Tool-layer changes (all 5 retrieval tools + RememberTool)
8. Agent pipeline (QueryContext, ContextBuilder, CoveragePlanner, ToolFactory)
9. CLI reshape (`examples/ask.rs`, `examples/ingest.rs`, `justfile`)
10. Integration security tests (3×3 matrix + boundary)
11. README note on reingest-required rollout

## 11. Transactional safety (per user's explicit requirement)

**Invariant:** no partially committed multi-store state is ever visible to a user. Every mutation that touches more than one store must be atomic-or-clean-rollback from the user's perspective.

There is no two-phase-commit across MinIO + Qdrant + Neo4j + Postgres, so the design uses **Postgres as the source of truth** for document/session state and treats the other stores as **eventually-consistent caches** managed via the saga pattern:

### 11.1 Ingest saga

```
1. Postgres: INSERT ingested_documents row with ingest_status='pending' (INSIDE a diesel transaction — if the INSERT or the sha256 dedup lookup fails, transaction rolls back).
2. MinIO: PUT object at users/{u}/docs/{d}.{ext}. On failure: mark Postgres row ingest_status='failed', failure_reason=minio; stop. MinIO has no partial state to roll back.
3. Docling sidecar parse.
4. Qdrant: upsert chunks tagged with ingest_run_id. On failure: call rollback_ingest_run(ingest_run_id) which deletes-by-ingest_run_id across Qdrant main collection + summaries collection (already implemented, already idempotent); Postgres row marked ingest_status='partial_failure'.
5. Neo4j: upsert structural graph tagged with ingest_run_id. On failure: rollback_ingest_run deletes Neo4j nodes/edges by ingest_run_id (add idempotent Cypher DELETE path if missing); marks Postgres 'partial_failure'.
6. Postgres: UPDATE ingested_documents SET ingest_status='complete' (another diesel transaction). If this step fails (extremely rare), trigger full rollback_ingest_run across Qdrant + Neo4j + MinIO.
```

Critical property: user-visible reads filter on `ingest_status='complete'`. A document in `pending` or `partial_failure` state is invisible to the user. This makes the whole saga atomic from the user's point of view regardless of mid-flight failures.

### 11.2 Session + activation atomicity

```sql
-- Create session + first activation in one Postgres transaction.
BEGIN;
INSERT INTO sessions (session_id, user_id) VALUES ($s, $u)
    ON CONFLICT (session_id) DO NOTHING;
-- Ownership-verified activation: fails loudly if the document does not belong to the user.
INSERT INTO session_active_documents (session_id, document_id)
SELECT $s, $d
FROM ingested_documents
WHERE document_id = $d AND user_id = $u
  AND ingest_status = 'complete';
-- 0 rows inserted => caller treats as StoreError::DocumentNotOwnedByUser.
COMMIT;
```

`diesel_async`'s `.transaction()` wraps the above. `ON CONFLICT DO NOTHING` + the `SELECT ... WHERE` guard means concurrent activations from two tabs don't race into a double-insert, and a user can never activate another user's doc even if they guess the UUID.

### 11.3 Delete document saga

Deletes are the inverse of ingest:

```
1. Postgres: UPDATE ingested_documents SET ingest_status='deleting' WHERE (document_id, user_id). If 0 rows => caller gets StoreError::DocumentNotOwnedByUser.
2. Qdrant: delete-by-doc_id (idempotent).
3. Neo4j: delete-by-doc_id (idempotent).
4. MinIO: delete object (idempotent).
5. Postgres: DELETE row + cascaded session_active_documents via ON DELETE CASCADE.
```

Each step is idempotent, so a retry after a crash converges to a clean state. User-visible reads filter out `ingest_status='deleting'`, so a half-deleted doc is invisible.

### 11.4 Memory save — trivially atomic

Single `diesel::insert_into(memories).values(...).execute()` under a diesel transaction. Nothing to saga.

### 11.5 Context build — read-only, but consistent snapshot

`ContextBuilder` reads three tables (`sessions`, `chat_messages`, `memories`, `session_active_documents`). Wrap all four reads in a single diesel `.transaction()` at `REPEATABLE READ` isolation so the LLM sees a consistent snapshot of the user's state, not torn reads from a concurrent writer.

### 11.6 Testing for transactionality

Extension of the §6 security matrix. Add a **chaos layer** (the 4th row):

| Failure injection | Expected behaviour |
|-------------------|---------------------|
| MinIO unavailable mid-ingest | Postgres row ends in `failed`; user can retry |
| Qdrant crash mid-upsert | `rollback_ingest_run` cleans up; Postgres row ends in `partial_failure`; no stray chunks visible |
| Neo4j timeout mid-upsert | Same as Qdrant crash path |
| Concurrent activation of same doc in two sessions | Both succeed (distinct rows); no duplicate |
| Concurrent delete + read | Reader either sees `complete` (delete hasn't started) or nothing (`deleting` state filtered); never a partial view |

Implementation: `crates/stores/tests/chaos_transactional.rs` using `testcontainers` for transient MinIO/Qdrant restarts. Feature-gated (`--features chaos-tests`) to keep CI fast by default.

### 11.7 Rollback APIs

All rollback helpers must be:

- **Idempotent** — safe to call multiple times with the same ingest_run_id / document_id.
- **Scoped** — never touch data outside the given ingest_run_id or document_id.
- **Logged** — structured tracing with ingest_run_id / document_id / user_id for audit.

Extend existing `try_rollback_stores` in `crates/retrieval/src/ingestion/pipeline.rs` (added in Phase 2 fix commit `f93d222`) to also delete the MinIO object for the same ingest_run_id.

## 12. Open follow-ups (out of scope, explicitly deferred)

- HTTP API with real auth (JWT, OAuth) → feeds `user_id` automatically.
- Per-user rate limits / quotas.
- Doc sharing / collaboration.
- Admin endpoints for user management.
- MinIO lifecycle policies beyond keep-forever (tiering, eviction) — Decision #2 locked keep-forever for now.

## 13. Open questions — none

All design questions above were answered during brainstorming. The spec is ready for plan-writing.
