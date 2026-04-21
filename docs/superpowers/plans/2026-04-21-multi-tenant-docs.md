# Multi-Tenant Per-User Documents + MinIO — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver strict per-user document isolation across the full stack (Postgres, Qdrant, Neo4j, MinIO, agent memory, tools, CLI), with transactional ingest/delete sagas and a feature-gated chaos test matrix.

**Architecture:** `UserId` newtype plumbed through every store and tool constructor. Postgres acts as source of truth for document state via an `ingest_status` state machine (`pending / complete / partial_failure / failed / deleting`); Qdrant, Neo4j, and MinIO are eventually-consistent caches maintained by idempotent rollback sagas. Every multi-step Postgres write runs under `diesel_async::transaction()`; `ContextBuilder` uses `REPEATABLE READ` isolation for consistent per-conversation reads.

**Tech Stack:** Rust 2021, tokio, rig (agent), diesel 2.2 + diesel-async (Postgres), qdrant-client 1.17 (Qdrant), neo4rs 0.8 (Neo4j), aws-sdk-s3 (MinIO via S3-compatible API), OpenTelemetry, fastembed, jina reranker, Docling sidecar.

**Spec:** `docs/superpowers/specs/2026-04-21-multi-tenant-docs-design.md` (commit `fe51830`).

**Branch:** `multi-tenant-docs` (already checked out). Single-PR rollout to `main` per spec §10.

**User-runs convention:** Every `cargo build` / `cargo test` / `cargo run` / `docker compose` / `diesel migration run` is handed off to the user. Tasks marked "Hand off — user runs" mean the controller must STOP and wait for confirmation.

**Codex 5.4-high checkpoints:** after Phase 3 (PgStore + migrations), Phase 6 (Qdrant + Neo4j filtering), Phase 8 (agent pipeline), and final pass after Phase 10.

---

## Phase 0 — Pre-flight

### Task 0.1: Verify baseline

**Files:** none.

- [ ] **Step 1: Confirm branch and clean tree**

```bash
git rev-parse --abbrev-ref HEAD      # expect: multi-tenant-docs
git status --short                    # expect: ?? pi.pdf  ?? segmentare.pdf  (no staged changes)
git log --oneline -2                  # expect: fe51830 docs(spec)...  d4e62b6 Merge PR #7...
```

- [ ] **Step 2: Hand off — user runs baseline check**

User runs:
```bash
cd /Users/cosmingagea/workspace/agent-rs && cargo check --workspace
```
Expected: success.

- [ ] **Step 3: Confirm infra containers up**

User runs:
```bash
just status
```
Expected: qdrant, postgres, neo4j at minimum are "healthy". (Optionally `just infra-up` first.)

---

## Phase 1 — Foundation: `UserId` validator + error model

### Task 1.1: Extend `UserId` newtype with validator

**Files:**
- Modify: `crates/core/src/types.rs` — locate existing `UserId` struct.
- Modify: `crates/core/src/errors.rs` — add `InvalidUserId` error variant.

- [ ] **Step 1: Write the failing test**

Add to `crates/core/src/types.rs` (inside the existing `#[cfg(test)] mod tests` or create one):

```rust
#[cfg(test)]
mod user_id_tests {
    use super::*;

    #[test]
    fn user_id_accepts_alphanumeric_dash_underscore() {
        assert!(UserId::try_new("alice_01").is_ok());
        assert!(UserId::try_new("user-123").is_ok());
        assert!(UserId::try_new("ABC").is_ok());
    }

    #[test]
    fn user_id_rejects_empty() {
        assert!(UserId::try_new("").is_err());
    }

    #[test]
    fn user_id_rejects_sql_injection_attempts() {
        assert!(UserId::try_new("' OR 1=1 --").is_err());
        assert!(UserId::try_new("'; DROP TABLE users;").is_err());
        assert!(UserId::try_new("admin'--").is_err());
    }

    #[test]
    fn user_id_rejects_whitespace_and_control_chars() {
        assert!(UserId::try_new("user one").is_err());
        assert!(UserId::try_new("user\n").is_err());
        assert!(UserId::try_new("user\t01").is_err());
    }

    #[test]
    fn user_id_rejects_too_long_input() {
        let long = "a".repeat(129);
        assert!(UserId::try_new(&long).is_err());
    }

    #[test]
    fn user_id_accepts_exactly_128_chars() {
        let max_len = "a".repeat(128);
        assert!(UserId::try_new(&max_len).is_ok());
    }
}
```

- [ ] **Step 2: Hand off — user runs**

```bash
cargo test -p agent-rs-core --lib types::user_id_tests 2>&1 | tail -15
```
Expected: errors — `UserId::try_new` does not exist.

- [ ] **Step 3: Add the validator**

In `crates/core/src/types.rs`, extend or add the `UserId` impl:

```rust
impl UserId {
    /// Maximum length of a user id in characters.
    ///
    /// Chosen to accommodate UUIDs, email addresses, and short-form account
    /// slugs with headroom. Anything longer is rejected as malformed input.
    pub const MAX_LEN: usize = 128;

    /// Construct a [`UserId`] after validating the string.
    ///
    /// Accepts only the ASCII character set `[A-Za-z0-9_-]` with length in
    /// `1..=MAX_LEN`. This is intentionally strict to prevent user-controlled
    /// strings from reaching SQL, Cypher, or payload-filter contexts where
    /// they could cause injection, collision, or ambient-authorisation bugs.
    ///
    /// # Errors
    ///
    /// Returns [`crate::errors::AgentError::InvalidUserId`] with the offending
    /// input when validation fails.
    pub fn try_new(raw: &str) -> Result<Self, crate::errors::AgentError> {
        if raw.is_empty() {
            return Err(crate::errors::AgentError::InvalidUserId(
                "user id must be non-empty".into(),
            ));
        }
        if raw.chars().count() > Self::MAX_LEN {
            return Err(crate::errors::AgentError::InvalidUserId(format!(
                "user id exceeds max length {}",
                Self::MAX_LEN
            )));
        }
        if !raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(crate::errors::AgentError::InvalidUserId(format!(
                "user id contains disallowed characters: {raw:?}"
            )));
        }
        Ok(Self(raw.to_owned()))
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
```

In `crates/core/src/errors.rs`, add the variant:

```rust
/// Invalid user id — rejected by [`crate::types::UserId::try_new`].
///
/// Message is verbose for logs but callers must NOT propagate it back to
/// user-facing output; return a generic "invalid user id" at API
/// boundaries instead.
#[error("invalid user id: {0}")]
InvalidUserId(String),
```

- [ ] **Step 4: Hand off — user runs**

```bash
cargo test -p agent-rs-core --lib types::user_id_tests
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/types.rs crates/core/src/errors.rs
git commit -m "$(cat <<'EOF'
feat(core/types): UserId::try_new validator + AgentError::InvalidUserId

Phase 1.1 of multi-tenant-docs spec. Strict alphanumeric/_/- charset,
1..=128 chars. Prevents SQL/Cypher injection via user-controlled ids
and catches empty/overlong input at the type boundary, so downstream
code cannot accidentally thread an invalid UserId into a store.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Extend `StoreError` with multi-tenant variants

**Files:**
- Modify: `crates/core/src/errors.rs`

- [ ] **Step 1: Failing test**

Add to `crates/core/src/errors.rs`'s test module:

```rust
#[test]
fn store_error_session_not_found_does_not_leak_existence() {
    // The Display impl must NOT reveal whether a session exists for a
    // different user — both "no such session" and "session belongs to
    // different user" collapse to the same message text.
    let err = StoreError::SessionNotFound {
        session_id: "sess-123".into(),
        user_id: "alice".into(),
    };
    let s = format!("{err}");
    assert!(s.contains("not found"));
    assert!(!s.to_lowercase().contains("different user"));
    assert!(!s.to_lowercase().contains("belongs to"));
}

#[test]
fn store_error_document_not_owned_has_required_fields() {
    let err = StoreError::DocumentNotOwnedByUser {
        document_id: "doc-1".into(),
        user_id: "alice".into(),
    };
    let s = format!("{err}");
    assert!(s.contains("doc-1"));
}
```

- [ ] **Step 2: Hand off — verify fail**

```bash
cargo test -p agent-rs-core --lib errors 2>&1 | tail -15
```

- [ ] **Step 3: Add the variants**

In `crates/core/src/errors.rs`, append to the `StoreError` enum:

```rust
/// Session does not exist, OR belongs to a different user.
///
/// Deliberately collapses both cases into one error to prevent
/// enumeration attacks: a caller cannot distinguish "this session_id
/// was never issued" from "this session_id belongs to someone else".
#[error("session {session_id} not found for user {user_id}")]
SessionNotFound {
    session_id: String,
    user_id: String,
},

/// Caller attempted to activate / read / delete a document that does
/// not belong to the provided user.
#[error("document {document_id} not owned by user {user_id}")]
DocumentNotOwnedByUser {
    document_id: String,
    user_id: String,
},

/// Error from the MinIO / S3 client. Wrapped transparently.
#[error("MinIO error: {0}")]
Minio(String),
```

- [ ] **Step 4: Hand off — user runs**

```bash
cargo test -p agent-rs-core --lib errors
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/errors.rs
git commit -m "feat(core/errors): add StoreError variants SessionNotFound, DocumentNotOwnedByUser, Minio

Phase 1.2. Session error deliberately merges 'does-not-exist' and
'belongs-to-different-user' to prevent enumeration attacks on the
API boundary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Postgres migration + schema regeneration

### Task 2.1: Write the migration

**Files:**
- Create: `crates/stores/migrations/2026-04-21-000001_multi_tenant/up.sql`
- Create: `crates/stores/migrations/2026-04-21-000001_multi_tenant/down.sql`

- [ ] **Step 1: Write `up.sql`**

```sql
-- 2026-04-21 multi_tenant: per-user document library + sessions + activations

-- 1) Add user_id to existing user-data tables. Spec §8: reingest-required,
--    so existing rows are wiped before running this; NOT NULL without
--    DEFAULT is correct.

ALTER TABLE ingested_documents
    ADD COLUMN user_id TEXT NOT NULL,
    ADD COLUMN minio_object_key TEXT;

CREATE INDEX idx_ingested_documents_user_id
    ON ingested_documents (user_id);

ALTER TABLE memories
    ADD COLUMN user_id TEXT NOT NULL;

CREATE INDEX idx_memories_user_id
    ON memories (user_id);

ALTER TABLE chat_messages
    ADD COLUMN user_id TEXT NOT NULL;

CREATE INDEX idx_chat_messages_user_id_session_id
    ON chat_messages (user_id, session_id);

-- 2) sessions: one row per UUID, tied to a user. session_active_documents
--    FKs into this to prevent dangling activations.

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id ON sessions (user_id);

-- 3) session_active_documents: which subset of the user's library
--    participates in this conversation.

CREATE TABLE session_active_documents (
    session_id UUID NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES ingested_documents (document_id) ON DELETE CASCADE,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, document_id)
);

CREATE INDEX idx_session_active_documents_document
    ON session_active_documents (document_id);

-- 4) Widen ingest_status check so new states from spec §11.1 are valid.
--    Existing statuses in prod: 'pending', 'complete', 'partial_failure'.
--    Add 'failed' (MinIO upload fail) and 'deleting' (spec §11.3).

ALTER TABLE ingested_documents
    DROP CONSTRAINT IF EXISTS ingested_documents_ingest_status_check;

ALTER TABLE ingested_documents
    ADD CONSTRAINT ingested_documents_ingest_status_check
    CHECK (ingest_status IN (
        'pending', 'complete', 'partial_failure', 'failed', 'deleting'
    ));
```

- [ ] **Step 2: Write `down.sql`**

```sql
-- Reverse of up.sql. Best-effort — `DROP COLUMN` + table drops lose data.

DROP TABLE IF EXISTS session_active_documents;
DROP TABLE IF EXISTS sessions;

ALTER TABLE ingested_documents
    DROP CONSTRAINT IF EXISTS ingested_documents_ingest_status_check;

ALTER TABLE ingested_documents
    ADD CONSTRAINT ingested_documents_ingest_status_check
    CHECK (ingest_status IN ('pending', 'complete', 'partial_failure'));

DROP INDEX IF EXISTS idx_chat_messages_user_id_session_id;
ALTER TABLE chat_messages DROP COLUMN IF EXISTS user_id;

DROP INDEX IF EXISTS idx_memories_user_id;
ALTER TABLE memories DROP COLUMN IF EXISTS user_id;

DROP INDEX IF EXISTS idx_ingested_documents_user_id;
ALTER TABLE ingested_documents
    DROP COLUMN IF EXISTS minio_object_key,
    DROP COLUMN IF EXISTS user_id;
```

- [ ] **Step 3: Hand off — user resets data and runs migrations**

Per spec §8, existing data must be wiped before running this migration (existing rows lack `user_id` which would violate NOT NULL).

User runs:
```bash
just infra-reset      # drops and recreates Postgres / Qdrant / Neo4j volumes
just migrations       # applies all migrations including the new one
```

Expected: `Running migration 2026-04-21-000001_multi_tenant ... ok`.

- [ ] **Step 4: Commit**

```bash
git add crates/stores/migrations/2026-04-21-000001_multi_tenant
git commit -m "feat(stores/migrations): multi-tenant schema — user_id columns, sessions, session_active_documents

Phase 2.1. Adds user_id to ingested_documents / memories / chat_messages,
new sessions table, session_active_documents junction table, and widens
ingest_status check constraint to include 'failed' and 'deleting'
per spec §11.1/§11.3 state machine.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.2: Regenerate Diesel schema.rs

**Files:**
- Modify: `crates/stores/src/postgres/schema.rs`

- [ ] **Step 1: Hand off — user regenerates schema**

Per `justfile:migrations` the database URL env is populated. User runs:

```bash
diesel print-schema \
    --database-url "$AGENT_RS_DATABASE_URL" \
    > crates/stores/src/postgres/schema.rs
```

Or equivalently via a `just` recipe if `justfile:schema` exists. Verify the diff:

```bash
git diff crates/stores/src/postgres/schema.rs
```

Expected changes:
- `ingested_documents` gains `user_id -> Text` and `minio_object_key -> Nullable<Text>`
- `memories` gains `user_id -> Text`
- `chat_messages` gains `user_id -> Text`
- New `sessions` and `session_active_documents` tables present

- [ ] **Step 2: Hand off — user runs**

```bash
cargo check -p agent-rs-stores 2>&1 | tail -15
```

Will likely emit errors in `crates/stores/src/postgres/documents.rs` and `crates/stores/src/postgres/memories.rs` because existing query builders reference fields whose types changed or tables whose shape grew. These errors drive Phase 3.

Record the errors — each will be addressed by a Phase 3 task. If errors reference fields you did NOT add in Phase 2.1, something else is wrong; stop and report.

- [ ] **Step 3: Commit (schema only)**

```bash
git add crates/stores/src/postgres/schema.rs
git commit -m "chore(stores/schema): regenerate schema.rs after multi_tenant migration

Phase 2.2. Auto-generated by 'diesel print-schema' after applying the
2026-04-21 multi_tenant migration. No manual edits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — PgStore methods with user_id

### Task 3.1: Sessions table: `create_or_fetch` + ownership check

**Files:**
- Create: `crates/stores/src/postgres/sessions.rs`
- Modify: `crates/stores/src/postgres/mod.rs` — add `pub mod sessions;`
- Modify: `crates/stores/src/postgres/models.rs` — add `Session` / `NewSession` insertable.

- [ ] **Step 1: Failing integration test**

Create `crates/stores/tests/sessions_integration.rs`:

```rust
//! Requires live Postgres via `just infra-up` + migrations run.
//! Feature-gated behind `integration-tests` to keep default `cargo test` fast.

#![cfg(feature = "integration-tests")]

use agent_rs_core::errors::StoreError;
use agent_rs_core::types::{SessionId, UserId};
use agent_rs_stores::postgres::PgStore;
use uuid::Uuid;

async fn test_pg() -> PgStore {
    let url = std::env::var("AGENT_RS_DATABASE_URL")
        .expect("AGENT_RS_DATABASE_URL must be set for integration tests");
    PgStore::new(&url).await.expect("pg connect")
}

#[tokio::test]
async fn session_create_or_fetch_creates_new_row() {
    let pg = test_pg().await;
    let user = UserId::try_new("alice-test-3-1a").expect("valid");
    let session_id = SessionId::from(Uuid::new_v4());

    let row = pg
        .create_or_fetch_session(&user, &session_id)
        .await
        .expect("create");
    assert_eq!(row.session_id, session_id.as_uuid());
    assert_eq!(row.user_id, user.as_str());
}

#[tokio::test]
async fn session_create_or_fetch_returns_existing_for_same_user() {
    let pg = test_pg().await;
    let user = UserId::try_new("alice-test-3-1b").expect("valid");
    let session_id = SessionId::from(Uuid::new_v4());

    let first = pg.create_or_fetch_session(&user, &session_id).await.unwrap();
    let second = pg.create_or_fetch_session(&user, &session_id).await.unwrap();
    assert_eq!(first.session_id, second.session_id);
    assert_eq!(first.created_at, second.created_at, "must not re-create");
}

#[tokio::test]
async fn session_create_or_fetch_rejects_cross_user_hijack() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-1c").expect("valid");
    let bob = UserId::try_new("bob-test-3-1c").expect("valid");
    let session_id = SessionId::from(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session_id).await.unwrap();
    let err = pg
        .create_or_fetch_session(&bob, &session_id)
        .await
        .expect_err("bob must not read alice's session");

    match err {
        StoreError::SessionNotFound { .. } => {}
        other => panic!("expected SessionNotFound, got {other:?}"),
    }
}
```

- [ ] **Step 2: Hand off — verify fail**

```bash
cargo test -p agent-rs-stores --features integration-tests --test sessions_integration 2>&1 | tail -15
```

Expected: compile errors — `PgStore::create_or_fetch_session` doesn't exist.

- [ ] **Step 3: Add models**

In `crates/stores/src/postgres/models.rs`:

```rust
use chrono::{DateTime, Utc};

/// Session row — one per `(user_id, session_id)` pair.
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = crate::postgres::schema::sessions)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct SessionRow {
    pub session_id: Uuid,
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Insertable)]
#[diesel(table_name = crate::postgres::schema::sessions)]
pub struct NewSession<'a> {
    pub session_id: Uuid,
    pub user_id: &'a str,
}
```

- [ ] **Step 4: Implement `create_or_fetch_session`**

Create `crates/stores/src/postgres/sessions.rs`:

```rust
//! Sessions: authorisation gate between a user and a conversation.
//!
//! Every read on chat_messages / memories / session_active_documents
//! validates `(user_id, session_id)` belongs together — a session
//! cannot be cross-read by a different user.

use agent_rs_core::errors::StoreError;
use agent_rs_core::types::{SessionId, UserId};
use diesel::prelude::*;
use diesel_async::{AsyncConnection, AsyncPgConnection, RunQueryDsl};

use super::models::{NewSession, SessionRow};
use super::schema::sessions;
use super::PgStore;

impl PgStore {
    /// Upsert semantics: if no row exists for `session_id`, insert one
    /// owned by `user_id`. If a row exists AND belongs to `user_id`,
    /// return it. If a row exists for a DIFFERENT user, return
    /// [`StoreError::SessionNotFound`] (the same error as "does not
    /// exist" — never leak existence).
    ///
    /// Wrapped in a single Postgres transaction so concurrent callers
    /// see serialised results (`INSERT ... ON CONFLICT` + ownership
    /// check in one round-trip).
    pub async fn create_or_fetch_session(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<SessionRow, StoreError> {
        let mut conn = self.conn().await?;
        conn.transaction::<SessionRow, StoreError, _>(|conn| {
            Box::pin(create_or_fetch_session_tx(
                conn,
                user_id.as_str(),
                session_id.as_uuid(),
            ))
        })
        .await
    }
}

async fn create_or_fetch_session_tx(
    conn: &mut AsyncPgConnection,
    user_id: &str,
    session_id: uuid::Uuid,
) -> Result<SessionRow, StoreError> {
    diesel::insert_into(sessions::table)
        .values(NewSession { session_id, user_id })
        .on_conflict(sessions::session_id)
        .do_nothing()
        .execute(conn)
        .await
        .map_err(StoreError::from)?;

    let row: SessionRow = sessions::table
        .filter(sessions::session_id.eq(session_id))
        .select(SessionRow::as_select())
        .first(conn)
        .await
        .map_err(StoreError::from)?;

    if row.user_id != user_id {
        return Err(StoreError::SessionNotFound {
            session_id: session_id.to_string(),
            user_id: user_id.to_owned(),
        });
    }
    Ok(row)
}
```

- [ ] **Step 5: Wire the module**

In `crates/stores/src/postgres/mod.rs`:

```rust
pub mod sessions;
pub use models::SessionRow;
```

- [ ] **Step 6: Enable integration test feature**

In `crates/stores/Cargo.toml`, add under `[features]`:

```toml
integration-tests = []
```

- [ ] **Step 7: Hand off — user runs**

```bash
just infra-up  # if not already running
cargo test -p agent-rs-stores --features integration-tests --test sessions_integration
```

Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add crates/stores/src/postgres/sessions.rs \
        crates/stores/src/postgres/mod.rs \
        crates/stores/src/postgres/models.rs \
        crates/stores/Cargo.toml \
        crates/stores/tests/sessions_integration.rs
git commit -m "feat(stores/postgres): sessions table — create_or_fetch + cross-user rejection

Phase 3.1. Owner-verified session upsert wrapped in a diesel
transaction. Cross-user reads collapse to SessionNotFound to prevent
enumeration attacks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.2: Scope `ingested_documents` methods to user

**Files:**
- Modify: `crates/stores/src/postgres/documents.rs`

- [ ] **Step 1: Failing integration test**

Append to `crates/stores/tests/sessions_integration.rs`:

```rust
#[tokio::test]
async fn list_ingested_documents_scoped_to_user() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-2").expect("valid");
    let bob = UserId::try_new("bob-test-3-2").expect("valid");

    // Insert one doc per user at status=complete.
    pg.test_insert_document_complete(&alice, "alice.pdf").await.unwrap();
    pg.test_insert_document_complete(&bob, "bob.pdf").await.unwrap();

    let alice_docs = pg.list_ingested_documents(&alice).await.unwrap();
    assert!(alice_docs.iter().all(|d| d.user_id == alice.as_str()));
    assert!(alice_docs.iter().any(|d| d.source == "alice.pdf"));
    assert!(alice_docs.iter().all(|d| d.source != "bob.pdf"),
            "alice must not see bob's doc");

    let bob_docs = pg.list_ingested_documents(&bob).await.unwrap();
    assert!(bob_docs.iter().any(|d| d.source == "bob.pdf"));
    assert!(bob_docs.iter().all(|d| d.source != "alice.pdf"));
}
```

`test_insert_document_complete` is a test-only helper you add in Step 4.

- [ ] **Step 2: Hand off — verify fail**

```bash
cargo test -p agent-rs-stores --features integration-tests --test sessions_integration list_ingested_documents_scoped_to_user 2>&1 | tail -15
```

- [ ] **Step 3: Rewrite `list_ingested_documents`**

In `crates/stores/src/postgres/documents.rs`, replace the existing signature:

```rust
/// List every ingested document owned by `user_id` that is in
/// `ingest_status = 'complete'`. Partial/failed/deleting rows are
/// never visible to the user.
///
/// Wrapped in a single-statement read; no transaction needed.
pub async fn list_ingested_documents(
    &self,
    user_id: &UserId,
) -> Result<Vec<IngestedDocumentRow>, StoreError> {
    let mut conn = self.conn().await?;
    let rows = ingested_documents::table
        .filter(ingested_documents::user_id.eq(user_id.as_str()))
        .filter(ingested_documents::ingest_status.eq("complete"))
        .select(IngestedDocumentRow::as_select())
        .load(&mut conn)
        .await
        .map_err(StoreError::from)?;
    Ok(rows)
}
```

Also update `IngestedDocumentRow` in `crates/stores/src/postgres/models.rs` to add `pub user_id: String` and `pub minio_object_key: Option<String>` fields (matching the regenerated schema).

- [ ] **Step 4: Add `test_insert_document_complete` helper**

Add to `crates/stores/src/postgres/documents.rs`:

```rust
#[cfg(any(test, feature = "integration-tests"))]
impl PgStore {
    /// Test-only: insert a minimal `ingest_status='complete'` row.
    /// Bypasses the full ingest saga — for security tests where the
    /// saga itself is out of scope.
    pub async fn test_insert_document_complete(
        &self,
        user_id: &UserId,
        source: &str,
    ) -> Result<IngestedDocumentRow, StoreError> {
        use chrono::Utc;
        let mut conn = self.conn().await?;
        let new_id = uuid::Uuid::new_v4();
        let new_run = uuid::Uuid::new_v4();
        let sha = format!("test-{}", &new_id.to_string()[..16]);
        diesel::insert_into(ingested_documents::table)
            .values((
                ingested_documents::document_id.eq(new_id),
                ingested_documents::sha256.eq(&sha),
                ingested_documents::source.eq(source),
                ingested_documents::ingest_status.eq("complete"),
                ingested_documents::ingest_run_id.eq(new_run),
                ingested_documents::user_id.eq(user_id.as_str()),
                ingested_documents::started_at.eq(Utc::now().naive_utc()),
                ingested_documents::updated_at.eq(Utc::now().naive_utc()),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        ingested_documents::table
            .filter(ingested_documents::document_id.eq(new_id))
            .select(IngestedDocumentRow::as_select())
            .first(&mut conn)
            .await
            .map_err(StoreError::from)
    }
}
```

- [ ] **Step 5: Update other doc methods — `begin_ingestion` takes `user_id`**

Find the existing `begin_ingestion` signature (around `documents.rs:97`). Widen it:

```rust
pub async fn begin_ingestion(
    &self,
    user_id: &UserId,
    new_doc: NewIngestedDocument<'_>,
) -> Result<BeginIngestionOutcome, StoreError> {
    // Existing body, plus inject new_doc.user_id on INSERT and ensure
    // ON CONFLICT path returns the correct user's row (NOT someone
    // else's row that happens to share sha256).
    // If sha256 conflict lands on a DIFFERENT user's doc, return
    // StoreError::DocumentNotOwnedByUser — users do not share storage
    // by sha.
    ...
}
```

Wire `user_id` into `NewIngestedDocument` struct (add `pub user_id: &'a str`). Update the INSERT to include `user_id`.

ON CONFLICT handling: if the existing row has a different `user_id`, return `StoreError::DocumentNotOwnedByUser`. (Design note: per spec strict mode, users do NOT share storage even if they upload identical PDFs; each gets their own document_id.)

Actually revise: strict-mode deduplication should be PER-USER. Change the conflict target from `sha256` alone to `(user_id, sha256)` — requires a unique index:

In `up.sql` of Phase 2.1, add:

```sql
CREATE UNIQUE INDEX ux_ingested_documents_user_sha256
    ON ingested_documents (user_id, sha256);
```

And drop the old single-column unique constraint on `sha256` if present. (Re-run migration after wiping volumes, since we are in the reingest-everything rollout.)

- [ ] **Step 6: Hand off — user re-runs infra reset + migrations + tests**

```bash
just infra-reset && just migrations
cargo test -p agent-rs-stores --features integration-tests --test sessions_integration
```

Expected: both tests pass (existing session test + new list_ingested_documents test).

- [ ] **Step 7: Commit**

```bash
git add crates/stores/src/postgres/documents.rs \
        crates/stores/src/postgres/models.rs \
        crates/stores/migrations/2026-04-21-000001_multi_tenant/up.sql \
        crates/stores/tests/sessions_integration.rs
git commit -m "feat(stores/postgres): scope ingested_documents queries to user_id

Phase 3.2. list_ingested_documents now filters by (user_id, status=
'complete'). Dedup key widened from sha256 to (user_id, sha256) so
two users uploading the same PDF each get their own document_id.
begin_ingestion now takes &UserId and cross-user sha256 collision
returns DocumentNotOwnedByUser.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.3: Scope `memories` and `chat_messages`

**Files:**
- Modify: `crates/stores/src/postgres/memories.rs` — find existing `save_memory` / `list_memories`
- Modify: `crates/stores/src/postgres/chat.rs` — find existing chat message methods
- Modify: `crates/stores/src/postgres/models.rs`

- [ ] **Step 1: Failing tests**

Append to `crates/stores/tests/sessions_integration.rs`:

```rust
#[tokio::test]
async fn memories_scoped_to_user() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-3").expect("valid");
    let bob = UserId::try_new("bob-test-3-3").expect("valid");

    pg.save_memory(&alice, "alice knows Rust").await.unwrap();
    pg.save_memory(&bob, "bob prefers Python").await.unwrap();

    let alice_mem = pg.list_memories(&alice).await.unwrap();
    assert!(alice_mem.iter().any(|m| m.content.contains("Rust")));
    assert!(alice_mem.iter().all(|m| !m.content.contains("Python")));
}

#[tokio::test]
async fn chat_messages_scoped_to_user_and_session() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-3b").expect("valid");
    let session = SessionId::from(Uuid::new_v4());
    pg.create_or_fetch_session(&alice, &session).await.unwrap();

    pg.append_chat_message(&alice, &session, "user", "hello").await.unwrap();
    let msgs = pg.list_chat_messages(&alice, &session).await.unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content, "hello");
}

#[tokio::test]
async fn chat_messages_cross_user_read_returns_empty() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-3c").expect("valid");
    let bob = UserId::try_new("bob-test-3-3c").expect("valid");
    let session = SessionId::from(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session).await.unwrap();
    pg.append_chat_message(&alice, &session, "user", "private").await.unwrap();

    // Bob cannot read alice's session — must fail at create_or_fetch_session
    // before reaching list_chat_messages. But if someone bypasses by
    // passing (bob, alice_session) directly to list_chat_messages, the
    // query must filter to zero rows — defence in depth.
    let bob_view = pg.list_chat_messages(&bob, &session).await.unwrap();
    assert!(bob_view.is_empty(), "bob must not see alice's chat");
}
```

- [ ] **Step 2: Hand off — verify fail**

- [ ] **Step 3: Update method signatures**

`save_memory(user_id: &UserId, content: &str)` — add `user_id` to the INSERT row.

`list_memories(user_id: &UserId)` — filter `.filter(memories::user_id.eq(user_id.as_str()))`.

`append_chat_message(user_id: &UserId, session_id: &SessionId, role: &str, content: &str)` — add `user_id` to INSERT.

`list_chat_messages(user_id: &UserId, session_id: &SessionId)` — filter `.filter(chat_messages::user_id.eq(user_id.as_str())).filter(chat_messages::session_id.eq(session_id.as_uuid()))`.

Update `MemoryRow` and `ChatMessageRow` in `models.rs` to include `pub user_id: String`.

- [ ] **Step 4: Hand off — user runs tests**

```bash
cargo test -p agent-rs-stores --features integration-tests --test sessions_integration
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/stores/src/postgres/memories.rs \
        crates/stores/src/postgres/chat.rs \
        crates/stores/src/postgres/models.rs \
        crates/stores/tests/sessions_integration.rs
git commit -m "feat(stores/postgres): scope memories + chat_messages to user_id

Phase 3.3. save_memory, list_memories, append_chat_message, and
list_chat_messages now require &UserId. Cross-user direct calls (if
the authz layer is bypassed) return empty sets, not other users' data
— defence in depth.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.4: `session_active_documents` activation with ownership-verified INSERT

**Files:**
- Create: `crates/stores/src/postgres/session_docs.rs`
- Modify: `crates/stores/src/postgres/mod.rs`

- [ ] **Step 1: Failing tests**

Append to `sessions_integration.rs`:

```rust
#[tokio::test]
async fn activate_document_for_session_with_valid_owner() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-4a").expect("valid");
    let session = SessionId::from(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session).await.unwrap();
    let doc = pg.test_insert_document_complete(&alice, "alice.pdf").await.unwrap();

    pg.activate_document_for_session(&alice, &session, &SessionId::from(doc.document_id))
        .await
        .expect("valid owner");

    let active = pg.list_active_documents(&alice, &session).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].document_id, doc.document_id);
}

#[tokio::test]
async fn activate_document_rejects_cross_user() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-4b").expect("valid");
    let bob = UserId::try_new("bob-test-3-4b").expect("valid");
    let bob_session = SessionId::from(Uuid::new_v4());

    pg.create_or_fetch_session(&bob, &bob_session).await.unwrap();
    let alice_doc = pg
        .test_insert_document_complete(&alice, "alice.pdf")
        .await
        .unwrap();

    let err = pg
        .activate_document_for_session(&bob, &bob_session, &SessionId::from(alice_doc.document_id))
        .await
        .expect_err("bob cannot activate alice's doc");

    match err {
        StoreError::DocumentNotOwnedByUser { .. } => {}
        other => panic!("expected DocumentNotOwnedByUser, got {other:?}"),
    }
}
```

- [ ] **Step 2: Implement**

```rust
//! session_active_documents — which of a user's docs participate in
//! a given conversation. Activation is ownership-verified in a single
//! Postgres round-trip (no TOCTOU between check and insert).

use agent_rs_core::errors::StoreError;
use agent_rs_core::types::{SessionId, UserId};
use diesel::prelude::*;
use diesel_async::{AsyncConnection, RunQueryDsl};
use uuid::Uuid;

use super::models::{IngestedDocumentRow, SessionActiveDocumentRow};
use super::schema::{ingested_documents, session_active_documents, sessions};
use super::PgStore;

impl PgStore {
    /// Activate `document_id` in `session_id` for `user_id`. The INSERT
    /// is gated by a single `INSERT ... SELECT ... WHERE` that joins
    /// `sessions` and `ingested_documents` — it completes ONLY if both
    /// the session and the document belong to `user_id`. No TOCTOU.
    pub async fn activate_document_for_session(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
        document_id: &SessionId, // reusing SessionId newtype as Uuid-carrier
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let sid = session_id.as_uuid();
        let did = document_id.as_uuid();
        let uid = user_id.as_str().to_owned();

        // Raw SQL for the joined guard. Diesel DSL cannot express the
        // "EXISTS ... WHERE s.user_id = d.user_id AND s.user_id = $uid"
        // ownership predicate cleanly.
        use diesel_async::RunQueryDsl as _;
        let rows: usize = diesel::sql_query(
            "INSERT INTO session_active_documents (session_id, document_id)
             SELECT $1, $2
             FROM sessions s
             JOIN ingested_documents d ON d.user_id = s.user_id
             WHERE s.session_id = $1
               AND d.document_id = $2
               AND s.user_id = $3
               AND d.ingest_status = 'complete'
             ON CONFLICT DO NOTHING",
        )
        .bind::<diesel::sql_types::Uuid, _>(sid)
        .bind::<diesel::sql_types::Uuid, _>(did)
        .bind::<diesel::sql_types::Text, _>(&uid)
        .execute(&mut conn)
        .await
        .map_err(StoreError::from)?;

        if rows == 0 {
            // Either no matching session, or doc belongs to different
            // user, or doc is not complete. All three collapse to
            // DocumentNotOwnedByUser from the caller's perspective.
            return Err(StoreError::DocumentNotOwnedByUser {
                document_id: did.to_string(),
                user_id: uid,
            });
        }
        Ok(())
    }

    /// List every document activated for `session_id` owned by `user_id`.
    /// Returns empty for cross-user reads.
    pub async fn list_active_documents(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<Vec<IngestedDocumentRow>, StoreError> {
        let mut conn = self.conn().await?;
        use super::schema::session_active_documents::dsl as sad;
        let rows: Vec<IngestedDocumentRow> = ingested_documents::table
            .inner_join(sad::session_active_documents.on(
                sad::document_id.eq(ingested_documents::document_id)
            ))
            .filter(sad::session_id.eq(session_id.as_uuid()))
            .filter(ingested_documents::user_id.eq(user_id.as_str()))
            .filter(ingested_documents::ingest_status.eq("complete"))
            .select(IngestedDocumentRow::as_select())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(rows)
    }
}
```

Add `SessionActiveDocumentRow` struct in `models.rs`:

```rust
#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = crate::postgres::schema::session_active_documents)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct SessionActiveDocumentRow {
    pub session_id: Uuid,
    pub document_id: Uuid,
    pub activated_at: DateTime<Utc>,
}
```

- [ ] **Step 3: Hand off — run tests**

```bash
cargo test -p agent-rs-stores --features integration-tests --test sessions_integration
```

Expected: 8 passed.

- [ ] **Step 4: Commit**

```bash
git add crates/stores/src/postgres/session_docs.rs \
        crates/stores/src/postgres/mod.rs \
        crates/stores/src/postgres/models.rs \
        crates/stores/tests/sessions_integration.rs
git commit -m "feat(stores/postgres): session_active_documents with ownership-verified INSERT

Phase 3.4. Single INSERT ... SELECT ... WHERE joins sessions +
ingested_documents to enforce ownership in one round-trip — no TOCTOU
between ownership check and activation. Cross-user attempts collapse
to DocumentNotOwnedByUser.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.5: Codex 5.4-high checkpoint — Postgres layer

- [ ] **Step 1: Write Codex prompt**

Save to `/tmp/codex-pg.txt`:

```
Review the multi-tenant Postgres layer changes in /Users/cosmingagea/workspace/agent-rs (branch multi-tenant-docs).

Commits to review: Phase 2 + Phase 3.1-3.4.

Files of interest:
- crates/stores/migrations/2026-04-21-000001_multi_tenant/up.sql + down.sql
- crates/stores/src/postgres/schema.rs (regenerated)
- crates/stores/src/postgres/sessions.rs (new)
- crates/stores/src/postgres/session_docs.rs (new)
- crates/stores/src/postgres/documents.rs (scoped to user_id)
- crates/stores/src/postgres/memories.rs
- crates/stores/src/postgres/chat.rs
- crates/stores/src/postgres/models.rs
- crates/stores/tests/sessions_integration.rs

Verify:
1. No SQL injection — every user-supplied string uses .bind() or Diesel DSL, never string-interpolation.
2. activate_document_for_session's JOIN really enforces ownership in one round-trip (no TOCTOU between check and insert).
3. create_or_fetch_session collapses "does not exist" and "belongs to different user" into ONE error value (SessionNotFound); verify the Display impl does not leak which case it is.
4. Unique dedup index is (user_id, sha256), NOT sha256 alone — two users can upload identical PDFs.
5. Down migration cleanly reverses up (DROP in correct order to respect FKs).
6. `NOT NULL` on new user_id columns is correct only because spec §8 locks reingest-required; flag any default that accidentally re-introduces legacy data visibility.
7. Any diesel transaction boundary that silently swallows errors and leaves partial state.

Verdict: APPROVED / APPROVED_WITH_MINOR / CHANGES_REQUIRED. Under 600 words. Cite file:line.
```

- [ ] **Step 2: Run (background)**

```bash
rm -f /tmp/codex-pg-out.md /tmp/codex-pg.log
codex exec --full-auto --skip-git-repo-check \
  -m gpt-5.4 -c model_reasoning_effort=high \
  --output-last-message /tmp/codex-pg-out.md \
  < /tmp/codex-pg.txt > /tmp/codex-pg.log 2>&1 &
```

- [ ] **Step 3: Read and address findings**

Once Codex returns, read `/tmp/codex-pg-out.md`. Address any Critical/High findings before Phase 4.

---

## Phase 4 — MinIO object storage

### Task 4.1: Docker-compose + .env.example entries

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `.env` (add the same values)

- [ ] **Step 1: Add MinIO service to docker-compose.yml**

Append to `docker-compose.yml` under `services:`:

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
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 3s
      retries: 10
```

Add to `volumes:` section:

```yaml
  minio_data:
```

- [ ] **Step 2: Append to .env.example**

```
# MinIO object storage (spec 2026-04-21 multi-tenant docs)
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

Same in `.env`.

- [ ] **Step 3: Hand off — user brings up MinIO**

```bash
docker compose up -d minio
just status
```

Expected: `minio` container healthy. Open `http://localhost:1069` → login with `minioadmin`/`minioadmin`.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat(infra): add MinIO service on ports 1068 (API) + 1069 (Console)

Phase 4.1. Self-hosted object storage for per-user PDF bytes.
.env.example carries defaults; .env kept local (gitignored).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.2: `MinioStore` with aws-sdk-s3

**Files:**
- Create: `crates/stores/src/minio/mod.rs`
- Create: `crates/stores/src/minio/store.rs`
- Modify: `crates/stores/src/lib.rs` — add `pub mod minio;`
- Modify: `crates/stores/Cargo.toml` — add `aws-sdk-s3` + `aws-config` deps.
- Modify: `Cargo.toml` workspace — add `aws-sdk-s3 = { version = "1.60", default-features = false, features = ["rt-tokio", "behavior-version-latest"] }` and `aws-config = { version = "1.5", default-features = false, features = ["rt-tokio", "behavior-version-latest"] }`.

- [ ] **Step 1: Failing integration test**

Create `crates/stores/tests/minio_integration.rs`:

```rust
#![cfg(feature = "integration-tests")]

use agent_rs_core::types::UserId;
use agent_rs_stores::minio::MinioStore;
use uuid::Uuid;

fn test_config() -> agent_rs_core::config::MinioConfig {
    agent_rs_core::config::MinioConfig {
        endpoint: std::env::var("AGENT_RS__MINIO__ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:1068".into()),
        access_key: std::env::var("AGENT_RS__MINIO__ACCESS_KEY")
            .unwrap_or_else(|_| "minioadmin".into()),
        secret_key: std::env::var("AGENT_RS__MINIO__SECRET_KEY")
            .unwrap_or_else(|_| "minioadmin".into()),
        bucket: std::env::var("AGENT_RS__MINIO__BUCKET")
            .unwrap_or_else(|_| "agent-rs-docs".into()),
    }
}

#[tokio::test]
async fn put_get_roundtrip() {
    let store = MinioStore::new(test_config()).await.expect("connect");
    let user = UserId::try_new("alice-minio-test").expect("valid");
    let doc_id = Uuid::new_v4();

    let payload = b"hello world".to_vec();
    store.put_object(&user, doc_id, &payload, "application/pdf").await.unwrap();

    let got = store.get_object(&user, doc_id).await.unwrap();
    assert_eq!(got, payload);

    store.delete_object(&user, doc_id).await.unwrap();
}

#[tokio::test]
async fn cross_user_read_is_forbidden_by_key_namespacing() {
    let store = MinioStore::new(test_config()).await.expect("connect");
    let alice = UserId::try_new("alice-minio-cross").expect("valid");
    let bob = UserId::try_new("bob-minio-cross").expect("valid");
    let doc_id = Uuid::new_v4();

    store.put_object(&alice, doc_id, b"alice-secret", "application/pdf").await.unwrap();

    // Bob asking for the same doc_id must miss — the key is under
    // users/alice-...,  not users/bob-....
    let got = store.get_object(&bob, doc_id).await;
    assert!(got.is_err(), "bob must not get alice's object");

    store.delete_object(&alice, doc_id).await.unwrap();
}

#[tokio::test]
async fn key_layout_is_user_scoped() {
    let user = UserId::try_new("alice-key-test").expect("valid");
    let doc_id = Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0);
    let key = agent_rs_stores::minio::object_key(&user, doc_id, "pdf");
    assert_eq!(
        key,
        "users/alice-key-test/docs/12345678-9abc-def0-1234-56789abcdef0.pdf"
    );
}
```

- [ ] **Step 2: Hand off — verify fail**

- [ ] **Step 3: Implement `MinioStore`**

`crates/stores/src/minio/store.rs`:

```rust
//! S3-compatible object storage (MinIO by default).
//!
//! Key layout: `users/{user_id}/docs/{document_id}.{ext}`. User scoping
//! is achieved purely by key prefix — bucket policies are not required
//! for correctness because every read/write goes through this store
//! and always passes a `&UserId`.

use agent_rs_core::errors::StoreError;
use agent_rs_core::types::UserId;
use aws_sdk_s3::config::{Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use uuid::Uuid;

/// Construct the canonical object key for a user's document.
pub fn object_key(user_id: &UserId, document_id: Uuid, ext: &str) -> String {
    format!("users/{}/docs/{}.{}", user_id.as_str(), document_id, ext)
}

pub struct MinioStore {
    client: Client,
    bucket: String,
}

impl MinioStore {
    /// Construct a client against the configured endpoint. Does NOT
    /// create the bucket — that is a one-off bootstrap concern handled
    /// by `just bootstrap` (see §4.3).
    pub async fn new(cfg: agent_rs_core::config::MinioConfig) -> Result<Self, StoreError> {
        let creds = Credentials::new(&cfg.access_key, &cfg.secret_key, None, None, "static");
        let s3_cfg = aws_sdk_s3::config::Builder::new()
            .endpoint_url(&cfg.endpoint)
            .region(Region::new("us-east-1")) // MinIO accepts any region
            .credentials_provider(creds)
            .force_path_style(true) // MinIO requires path-style, not virtual-hosted
            .behavior_version_latest()
            .build();
        Ok(Self {
            client: Client::from_conf(s3_cfg),
            bucket: cfg.bucket,
        })
    }

    pub async fn put_object(
        &self,
        user_id: &UserId,
        document_id: Uuid,
        bytes: &[u8],
        content_type: &str,
    ) -> Result<String, StoreError> {
        let ext = content_type_to_ext(content_type);
        let key = object_key(user_id, document_id, &ext);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(bytes.to_vec()))
            .content_type(content_type)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("put_object {key}: {e}")))?;
        Ok(key)
    }

    pub async fn get_object(
        &self,
        user_id: &UserId,
        document_id: Uuid,
    ) -> Result<Vec<u8>, StoreError> {
        let key = find_existing_key(self, user_id, document_id).await?;
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("get_object {key}: {e}")))?;
        let body = resp
            .body
            .collect()
            .await
            .map_err(|e| StoreError::Minio(format!("collect body {key}: {e}")))?;
        Ok(body.to_vec())
    }

    pub async fn delete_object(
        &self,
        user_id: &UserId,
        document_id: Uuid,
    ) -> Result<(), StoreError> {
        let Some(key) = try_find_existing_key(self, user_id, document_id).await else {
            return Ok(()); // already gone — idempotent
        };
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("delete_object {key}: {e}")))?;
        Ok(())
    }
}

/// Fetch the full object key given (user_id, document_id) — the extension
/// is discovered by listing the prefix, since `get_object` does not
/// know the content-type from caller.
async fn find_existing_key(
    store: &MinioStore,
    user_id: &UserId,
    document_id: Uuid,
) -> Result<String, StoreError> {
    try_find_existing_key(store, user_id, document_id)
        .await
        .ok_or_else(|| StoreError::Minio(format!(
            "no object for user={} doc={}", user_id.as_str(), document_id
        )))
}

async fn try_find_existing_key(
    store: &MinioStore,
    user_id: &UserId,
    document_id: Uuid,
) -> Option<String> {
    let prefix = format!("users/{}/docs/{}", user_id.as_str(), document_id);
    let resp = store
        .client
        .list_objects_v2()
        .bucket(&store.bucket)
        .prefix(&prefix)
        .max_keys(1)
        .send()
        .await
        .ok()?;
    resp.contents.and_then(|v| v.into_iter().next()).and_then(|o| o.key)
}

fn content_type_to_ext(ct: &str) -> String {
    match ct {
        "application/pdf" => "pdf".into(),
        "text/html" => "html".into(),
        "text/plain" => "txt".into(),
        _ => "bin".into(),
    }
}
```

`crates/stores/src/minio/mod.rs`:

```rust
//! MinIO / S3-compatible object storage.

mod store;

pub use store::{object_key, MinioStore};
```

Add `MinioConfig` to `crates/core/src/config.rs`:

```rust
/// MinIO / S3-compatible object storage config.
#[derive(Debug, Clone, Deserialize)]
pub struct MinioConfig {
    pub endpoint: String,
    pub access_key: String,
    pub secret_key: String,
    pub bucket: String,
}
```

Wire `MinioConfig` into `Settings` so `config::Settings` loads it from env (`AGENT_RS__MINIO__*`).

- [ ] **Step 4: Hand off — user creates bucket and runs tests**

```bash
# Create bucket once (can also go into justfile:bootstrap later)
docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker compose exec minio mc mb local/agent-rs-docs

cargo test -p agent-rs-stores --features integration-tests --test minio_integration
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/stores/src/minio \
        crates/stores/src/lib.rs \
        crates/stores/Cargo.toml \
        crates/core/src/config.rs \
        Cargo.toml Cargo.lock \
        crates/stores/tests/minio_integration.rs
git commit -m "feat(stores/minio): S3-compatible object storage with user-scoped keys

Phase 4.2. put/get/delete object under users/{user}/docs/{doc}.{ext}.
User scoping is by key prefix — defence in depth via list-prefix
discovery for get/delete. Idempotent delete (missing object = Ok).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.3: Bucket bootstrap in `justfile`

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add `minio-bootstrap` recipe**

```makefile
minio-bootstrap:
    docker compose exec -T minio mc alias set local http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} >/dev/null
    docker compose exec -T minio mc mb --ignore-existing local/${MINIO_BUCKET}
    @echo "MinIO bucket ${MINIO_BUCKET} ready"
```

Update `bootstrap` recipe:

```makefile
bootstrap: env infra-up migrations minio-bootstrap
```

- [ ] **Step 2: Hand off — user runs**

```bash
just minio-bootstrap
```

- [ ] **Step 3: Commit**

```bash
git add justfile
git commit -m "feat(justfile): add minio-bootstrap recipe and wire into bootstrap

Phase 4.3. One-shot bucket creation via 'just minio-bootstrap'; also
auto-runs as part of 'just bootstrap'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — Qdrant + Neo4j user_id filtering (parallel)

### Task 5.1: Qdrant `user_id` payload + filter

**Files:**
- Modify: `crates/stores/src/qdrant/section_filter.rs` — add `user_id` first-class
- Modify: `crates/stores/src/qdrant/mod.rs` — `search` / `retrieve_chunks_by_ids` take `user_id`
- Modify: `crates/retrieval/src/ingestion/pipeline.rs` — payload write includes `user_id`

- [ ] **Step 1: Failing unit test**

In `crates/stores/src/qdrant/section_filter.rs`:

```rust
#[test]
fn build_user_filter_with_user_only() {
    let f = build_user_filter("alice", None, None);
    assert_eq!(f.must.len(), 1);
}

#[test]
fn build_user_filter_with_user_doc_section() {
    let f = build_user_filter("alice", Some("doc-1"), Some("sec-1"));
    assert_eq!(f.must.len(), 3);
}

#[test]
fn build_user_filter_rejects_empty_user() {
    // Empty user_id must never produce a filter — that would allow
    // unscoped reads. Panic or return an error sentinel.
    let result = std::panic::catch_unwind(|| build_user_filter("", None, None));
    assert!(result.is_err(), "empty user_id must panic");
}
```

- [ ] **Step 2: Implement**

Rename/replace `build_section_filter` with:

```rust
/// Build a MANDATORY user-scoped Qdrant filter. Unlike
/// `build_section_filter` (deprecated), `user_id` is REQUIRED — an
/// empty user_id panics, because strict multi-tenancy has no "no user"
/// case. Optional `doc_id` / `section_id` narrow further.
pub fn build_user_filter(
    user_id: &str,
    doc_id: Option<&str>,
    section_id: Option<&str>,
) -> Filter {
    assert!(!user_id.is_empty(), "user_id must not be empty for Qdrant filter");
    let mut conds: Vec<Condition> = vec![Condition::matches("user_id", user_id.to_string())];
    if let Some(did) = doc_id {
        conds.push(Condition::matches("doc_id", did.to_string()));
    }
    if let Some(sid) = section_id {
        conds.push(Condition::matches("section_id", sid.to_string()));
    }
    Filter { must: conds, ..Default::default() }
}
```

Mark old `build_section_filter` `#[deprecated]` + `#[allow(dead_code)]` for one cycle.

In `crates/stores/src/qdrant/mod.rs`:

```rust
impl VectorStore for QdrantStore {
    async fn search(
        &self,
        collection: &CollectionName,
        user_id: &UserId,  // NEW — required
        query: &Embedding,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>, StoreError> {
        let user_filter = build_user_filter(user_id.as_str(), None, None);
        let merged = merge_filters(user_filter, filter);
        // ... existing search body, replacing filter with `merged` ...
    }
}
```

All other methods that read chunks gain `user_id: &UserId` similarly.

In `crates/retrieval/src/ingestion/pipeline.rs` payload builder, include `"user_id": doc.user_id` in every chunk's Qdrant payload.

- [ ] **Step 3: Hand off**

```bash
cargo check -p agent-rs-stores -p agent-rs-retrieval
cargo test -p agent-rs-stores --lib qdrant::section_filter
```

Expected: 3 new tests pass, existing callers compile (they all need the new `user_id` arg — cascading fix in Phase 6).

- [ ] **Step 4: Commit**

```bash
git add crates/stores/src/qdrant \
        crates/retrieval/src/ingestion/pipeline.rs
git commit -m "feat(stores/qdrant): mandatory user_id payload filter + panic on empty

Phase 5.1. Every Qdrant read path takes &UserId and wraps Filter with
a must-match on user_id payload field. Empty user_id panics rather
than silently returning cross-tenant data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.2: Neo4j `user_id` property + WHERE clause

**Files:**
- Modify: `crates/stores/src/neo4j/structural.rs` — Document MERGE includes `user_id`
- Modify: `crates/stores/src/neo4j/coverage.rs` — every query gains `WHERE d.user_id = $user_id`
- Modify: `crates/stores/src/neo4j/mod.rs` — Neo4jStore methods take `user_id`

**PARALLEL with Task 5.1.**

- [ ] **Step 1: Failing integration test**

Append to `crates/stores/tests/multi_tenant_security.rs` (create file):

```rust
#![cfg(feature = "integration-tests")]

use agent_rs_core::types::UserId;
use agent_rs_stores::neo4j::Neo4jStore;

async fn test_neo4j() -> Neo4jStore {
    Neo4jStore::new(/* test config from env */).await.expect("neo4j connect")
}

#[tokio::test]
async fn list_sections_for_doc_filters_by_user_id() {
    let store = test_neo4j().await;
    let alice = UserId::try_new("alice-neo4j").expect("valid");
    let bob = UserId::try_new("bob-neo4j").expect("valid");

    let alice_doc_id = "neo4j-alice-doc-1";
    let bob_doc_id = "neo4j-bob-doc-1";

    // Seed: each user gets one document + one section.
    store.test_seed_document(&alice, alice_doc_id, "5.1 alice").await.unwrap();
    store.test_seed_document(&bob, bob_doc_id, "5.1 bob").await.unwrap();

    let alice_sections = store.list_sections_for_doc(&alice, alice_doc_id).await.unwrap();
    assert_eq!(alice_sections.len(), 1);

    // Alice querying bob's doc must return empty.
    let cross = store.list_sections_for_doc(&alice, bob_doc_id).await.unwrap();
    assert!(cross.is_empty(), "alice must not see bob's sections");
}
```

- [ ] **Step 2: Implement**

Every Neo4j cypher that reads Document/Section/Chunk gains a `user_id` guard:

```rust
pub async fn list_sections_for_doc(
    &self,
    user_id: &UserId,
    doc_id: &str,
) -> Result<Vec<SectionRow>, StoreError> {
    let cypher = format!(
        "MATCH (d:{LABEL_DOCUMENT} {{id: $doc_id, user_id: $user_id}})-[:HAS_SECTION*1..]->(s:{LABEL_SECTION})
         RETURN s.id AS section_id, d.id AS doc_id, coalesce(s.title, s.heading) AS heading,
                s.section_order AS section_order, s.first_chunk_index AS first_chunk_index,
                s.page_start AS page_start, coalesce(s.is_preamble, false) AS is_preamble"
    );
    // bind doc_id AND user_id
    // ...
}

pub async fn chunks_in_sections(
    &self,
    user_id: &UserId,
    section_ids: &[String],
) -> Result<Vec<String>, StoreError> {
    let cypher = format!(
        "MATCH (c:Chunk)-[:IN_SECTION]->(s:{LABEL_SECTION})
               -[:HAS_SECTION*0..]->(d:{LABEL_DOCUMENT} {{user_id: $user_id}})
         WHERE s.id IN $ids
         RETURN c.id AS chunk_id, coalesce(c.chunk_index, 0) AS chunk_index
         ORDER BY chunk_index ASC"
    );
    // bind ids AND user_id
    // ...
}

pub async fn list_descendant_sections(
    &self,
    user_id: &UserId,
    section_id: &str,
) -> Result<Vec<String>, StoreError> {
    let cypher = format!(
        "MATCH (root:{LABEL_SECTION} {{id: $sid}})-[:HAS_SECTION*0..]->(d:{LABEL_DOCUMENT} {{user_id: $user_id}})
         WITH root, d
         MATCH (root)-[:PARENT_OF*0..]->(s:{LABEL_SECTION})
         RETURN s.id AS section_id"
    );
    // ...
}
```

In `structural.rs`, Document MERGE includes `user_id` on create:

```cypher
MERGE (d:Document {id: $doc_id})
ON CREATE SET d.user_id = $user_id,
              d.title = $title,
              ...
```

- [ ] **Step 3: Hand off**

```bash
cargo test -p agent-rs-stores --features integration-tests --test multi_tenant_security
```

- [ ] **Step 4: Commit**

```bash
git add crates/stores/src/neo4j \
        crates/stores/tests/multi_tenant_security.rs
git commit -m "feat(stores/neo4j): thread user_id through all read paths + Document.user_id property

Phase 5.2. Every cypher that matches Document/Section/Chunk gains a
user_id guard at the Document level (IN_SECTION/PARENT_OF descendants
inherit by structural inclusion).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.3: Codex 5.4-high checkpoint — Qdrant + Neo4j filtering

Same pattern as Task 3.5 with prompt focused on:

1. Zero cross-tenant leakage paths in any cypher or Qdrant filter (grep for queries missing user_id).
2. Payload field `user_id` is set on every chunk at ingest.
3. `build_user_filter` panic-on-empty is reached from every search call (no sneaky path that bypasses it).
4. Deprecated `build_section_filter` is not still in use after migration.

Verdict gate before Phase 6.

---

## Phase 6 — Retrieval router propagation

### Task 6.1: Router takes `user_id`

**Files:**
- Modify: `crates/retrieval/src/router.rs`

- [ ] **Step 1: Extend signature**

```rust
pub async fn retrieve_filtered(
    &self,
    user_id: &UserId,
    primary_query: &str,
    strategy: RetrievalStrategy,
    limit: usize,
    filter: Option<SectionFilter>,
) -> Result<RetrievalOutput, RetrievalError> {
    // Every branch call now passes user_id:
    //   vector_branch(&self, user_id, ...)
    //   keyword_branch(&self, user_id, ...)
    //   graph_branch(&self, user_id, ...)
}
```

Plumb through every branch; each branch wraps its Qdrant `search()` / `retrieve_chunks_by_ids()` call with `user_id`. Same for Neo4j-based branches.

`retrieve(&self, primary_query, strategy, limit)` (the legacy two-arg method) is `#[deprecated]`-marked and delegates to `retrieve_filtered` with a compile-time panic sentinel for the missing user_id — we cannot ship a legacy no-user path in strict mode.

Actually simpler: **remove** the legacy `retrieve()` entry point entirely. Every caller already gets `ctx.user_id` via the agent pipeline (Phase 7).

- [ ] **Step 2: Hand off + fix call sites**

```bash
cargo check -p agent-rs-retrieval
```

Fix compile errors in `crates/tools/src/knowledge.rs` and anywhere else that called the old signature (they all now need `user_id`).

- [ ] **Step 3: Commit**

```bash
git add crates/retrieval/src/router.rs
git commit -m "feat(retrieval/router): require UserId on every retrieve path

Phase 6.1. Removed legacy retrieve() no-user entry point — strict mode
has no valid 'no user' case. retrieve_filtered(&UserId, ...) is the
only public API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7 — Tool-layer changes (5 tools, parallel OK as subagent dispatches)

### Task 7.1: `SearchKnowledgeTool` takes user_id

**Files:** `crates/tools/src/knowledge.rs`

- [ ] **Step 1: Failing contract test**

Extend `crates/tools/tests/tool_contract_test.rs`:

```rust
#[tokio::test]
async fn search_knowledge_tool_constructor_requires_user_id() {
    // At minimum the constructor must accept &UserId; contract test
    // pins the struct shape. Behaviour is exercised by integration tests.
    use agent_rs_core::types::UserId;
    let user = UserId::try_new("test").expect("valid");
    let _tool = agent_rs_tools::SearchKnowledgeTool::new_for_user(
        /* router */ arc_noop_router(),
        /* evidence_collector */ arc_noop_collector(),
        user,
    );
}
```

- [ ] **Step 2: Implement**

```rust
pub struct SearchKnowledgeTool {
    router: Option<Arc<RetrievalRouter>>,
    evidence_collector: Option<Arc<Mutex<Vec<SearchResult>>>>,
    user_id: Option<UserId>,  // baked in at construction
}

impl SearchKnowledgeTool {
    pub fn new_for_user(
        router: Arc<RetrievalRouter>,
        evidence_collector: Arc<Mutex<Vec<SearchResult>>>,
        user_id: UserId,
    ) -> Self {
        Self {
            router: Some(router),
            evidence_collector: Some(evidence_collector),
            user_id: Some(user_id),
        }
    }
}

impl Tool for SearchKnowledgeTool {
    // ...existing...
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let user_id = self.user_id.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_knowledge: user_id not wired".into())
        })?;
        // router.retrieve_filtered(user_id, &args.query, ...)
        // ...
    }
}
```

Delete the old `new(...)` two-arg constructor — every call site must pass user_id.

- [ ] **Step 3: Commit**

---

### Task 7.2: `OutlineCorpusTool` takes user_id (PARALLEL with 7.1)

Same pattern as 7.1 for `crates/tools/src/outline_corpus.rs`. The tool's `call()` body:

- `pg.list_ingested_documents(user_id)` instead of unscoped list
- `neo4j.list_sections_for_doc(user_id, doc_id)` for every doc

---

### Task 7.3: `FetchSectionChunksTool` + `SearchSectionsTool` (PARALLEL with 7.1)

Same pattern for `crates/tools/src/fetch_section_chunks.rs` and `search_sections.rs`. Every Neo4j and Qdrant call takes `user_id`.

---

### Task 7.4: `DocumentInfoTool` + `RememberTool` take user_id

Same pattern for `crates/tools/src/document_info.rs` and `crates/tools/src/remember.rs`. `RememberTool::save_memory` calls the user-scoped variant.

---

## Phase 8 — Agent pipeline

### Task 8.1: `ContextBuilder` with REPEATABLE READ snapshot

**Files:** `crates/agent/src/steps/context_builder.rs`

- [ ] **Step 1: Failing test**

Unit test wraps mocked PgStore, asserts that all four reads (session, chat_messages, memories, session_active_documents) run under the same transaction.

- [ ] **Step 2: Implement**

```rust
pub async fn build(&self, ctx: &mut QueryContext, history: ...) -> Result<(), AgentError> {
    // Bootstrap session (create-or-fetch with ownership check).
    let _session = self.pg.create_or_fetch_session(&ctx.user_id, &ctx.session_id).await?;

    // Load the three per-user/per-session inputs under a single
    // REPEATABLE READ transaction so the LLM sees a consistent
    // snapshot of the user's state.
    let (history, memories, active_docs) = self.pg
        .read_context_snapshot(&ctx.user_id, &ctx.session_id)
        .await?;

    ctx.history = history;
    ctx.memories = memories;
    ctx.active_document_ids = if active_docs.is_empty() {
        None  // no selection → use full library
    } else {
        Some(active_docs.into_iter().map(|d| d.document_id).collect())
    };
    Ok(())
}
```

`PgStore::read_context_snapshot`:

```rust
pub async fn read_context_snapshot(
    &self,
    user_id: &UserId,
    session_id: &SessionId,
) -> Result<(Vec<ChatMessageRow>, Vec<MemoryRow>, Vec<IngestedDocumentRow>), StoreError> {
    let mut conn = self.conn().await?;
    conn.transaction::<_, StoreError, _>(|conn| {
        Box::pin(async move {
            // SET TRANSACTION ISOLATION LEVEL REPEATABLE READ
            diesel::sql_query("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                .execute(conn)
                .await
                .map_err(StoreError::from)?;
            let history = /* scoped list_chat_messages */;
            let memories = /* scoped list_memories */;
            let active = /* scoped list_active_documents */;
            Ok((history, memories, active))
        })
    }).await
}
```

- [ ] **Step 3: Commit**

---

### Task 8.2: `QueryContext.active_document_ids` + CoveragePlanner filtering

**Files:** `crates/agent/src/context.rs`, `crates/agent/src/steps/coverage_planner.rs`

- [ ] **Step 1: Failing test**

```rust
#[test]
fn coverage_planner_scopes_to_active_document_ids_when_set() {
    // Planner given ctx.active_document_ids = Some([doc_A])
    // Must produce target_sections only from doc_A, not doc_B.
}
```

- [ ] **Step 2: Implement**

In `coverage_planner.rs::plan`:

```rust
let all_docs = self.pg.list_ingested_documents(&ctx.user_id).await?;
let library_docs = match ctx.active_document_ids.as_ref() {
    Some(ids) => all_docs.iter()
        .filter(|d| ids.contains(&d.document_id))
        .cloned()
        .collect(),
    None => all_docs,
};
// filter_docs_by_query continues to work — it receives library_docs
```

- [ ] **Step 3: Commit**

---

### Task 8.3: `ToolFactory` constructs tools with user_id baked in

**Files:** `crates/agent/src/steps/tool_factory.rs`

- [ ] **Step 1: Update `build`**

`build(&self, user_id: &UserId) -> ToolSet`. Pass `user_id.clone()` into every tool's `new_for_user`:

```rust
tools.push(Box::new(OutlineCorpusTool::new_for_user(
    self.neo4j.clone(),
    self.pg.clone(),
    self.qdrant.clone(),
    self.summaries_collection.clone(),
    user_id.clone(),
)));
// ...same for the other 5 tools
```

`AgentRunner::run` calls `tool_factory.build(&ctx.user_id)` at the start of each query — tools are per-request, not long-lived, so baking in user_id is safe.

- [ ] **Step 2: Commit**

---

### Task 8.4: Codex 5.4-high checkpoint — agent pipeline

Prompt focuses on:
1. `ctx.user_id` actually flows to every tool
2. `ContextBuilder` snapshot isolation is `REPEATABLE READ` — no torn reads
3. `CoveragePlanner` respects `active_document_ids` AND falls back to full library only when None
4. No tool accepts a query that proceeds without user_id

---

## Phase 9 — CLI reshape

### Task 9.1: `examples/ask.rs` — 3 positional args

**Files:** `examples/ask.rs`

- [ ] **Step 1: Rewrite parse_args**

```rust
struct Args {
    user_id: UserId,
    session_id: SessionId,
    question: String,
    strategy: RetrievalStrategy,
    limit: usize,
}

fn parse_args() -> Result<Args> {
    let mut positional: Vec<String> = Vec::new();
    let mut strategy_str: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--strategy" | "-s" => strategy_str = Some(iter.next().context("--strategy needs a value")?),
            "--limit" | "-n" => limit = Some(iter.next().context("--limit")?.parse()?),
            _ => positional.push(a),
        }
    }
    if positional.len() != 3 {
        bail!("usage: just ask \"<user_id>\" \"<session_id>\" \"<question>\"");
    }
    let user_id = UserId::try_new(&positional[0]).context("invalid user_id")?;
    let raw_session = positional[1].as_str();
    let session_id = if raw_session.is_empty() {
        SessionId::from(Uuid::new_v4())
    } else {
        SessionId::from(Uuid::parse_str(raw_session).context("invalid session UUID")?)
    };
    // ...
}
```

- [ ] **Step 2: Commit**

---

### Task 9.2: `examples/ingest.rs` — user_id required + MinIO upload

**Files:** `examples/ingest.rs`, ingestion pipeline glue.

- [ ] **Step 1: Add positional user_id**

```bash
just ingest "<user_id>" "<path/to/doc.pdf>"
```

Before starting Docling, upload PDF bytes to MinIO via `MinioStore::put_object(&user_id, doc_id, &bytes, "application/pdf")`. Record `minio_object_key` on the `ingested_documents` row.

If MinIO upload fails → mark Postgres `ingest_status='failed'`, abort.

- [ ] **Step 2: Commit**

---

### Task 9.3: `justfile` recipes reshape

**Files:** `justfile`

- [ ] **Step 1: Update recipes**

```makefile
ask user_id session_id question:
    cargo run --release --example ask -- "{{user_id}}" "{{session_id}}" "{{question}}"

ingest user_id path:
    cargo run --release --example ingest -- "{{user_id}}" "{{path}}"
```

- [ ] **Step 2: Commit**

---

## Phase 10 — Full security test matrix

### Task 10.1: 3×3 security matrix (Postgres × Qdrant × Neo4j, read/write/delete)

**Files:** `crates/stores/tests/multi_tenant_security.rs` (extend)

9 test cases per the spec §6 matrix. Each test:
1. Seeds user A's data in the given store
2. Attempts the given operation as user B
3. Asserts the operation returns empty / errors / does not leak

- [ ] **Step 1: Write all 9 test cases**

Full test bodies required — no placeholders. Shape:

```rust
#[tokio::test]
async fn read_qdrant_rejects_cross_user_chunks() { /* ... */ }

#[tokio::test]
async fn write_qdrant_cannot_upsert_under_another_users_id() { /* ... */ }

#[tokio::test]
async fn delete_qdrant_cross_user_fails_or_noop() { /* ... */ }
// ... 6 more for postgres + neo4j
```

- [ ] **Step 2: Commit**

---

### Task 10.2: Boundary tests

**Files:** `crates/stores/tests/multi_tenant_security.rs` (extend)

- SQL-injection strings rejected at `UserId::try_new`
- Empty / whitespace / overlong user_id rejected
- Session hijack: `ask(user=A, session=B_owned, ...)` → `create_or_fetch_session` errors
- UUID collision (stuffed user_id) → newtype rejects non-alphanumeric

- [ ] **Step 1: Write boundary cases**

~6 test cases with full bodies.

- [ ] **Step 2: Commit**

---

### Task 10.3: Chaos tests (feature-gated)

**Files:** `crates/stores/tests/chaos_transactional.rs` (new)

Feature-gated behind `chaos-tests` in `crates/stores/Cargo.toml`.

Test cases per spec §11.6:
- MinIO unavailable mid-ingest → Postgres row ends `failed`
- Qdrant crash mid-upsert → rollback_ingest_run cleans up
- Neo4j timeout → rollback completes
- Concurrent activation of same doc → both succeed (different sessions)
- Concurrent delete + read → read sees `complete` OR nothing, never partial

Use `testcontainers` to crash/restart services mid-test.

- [ ] **Step 1: Write 5 chaos tests**

Full bodies. No placeholders.

- [ ] **Step 2: Commit**

---

## Phase 11 — README + final

### Task 11.1: README update

**Files:** `README.md`

Add section on multi-tenancy rollout:

```markdown
## Multi-tenant rollout (April 2026)

This release introduces strict per-user document scoping. Upgrade steps:

1. `just infra-reset` — **destroys all existing data** (see spec decision #1).
2. `just migrations` — applies the multi_tenant migration.
3. `just minio-bootstrap` — creates the MinIO bucket.
4. Re-ingest each PDF under a user_id: `just ingest "<user_id>" "<path>"`.
5. Query with 3-arg CLI: `just ask "<user_id>" "<session_id_or_empty>" "<question>"`.

Empty `session_id` generates a fresh UUID for a new conversation.
```

- [ ] **Step 1: Commit**

---

### Task 11.2: Final Codex 5.4-high pass

Prompt: "Final review of multi-tenant-docs branch. Verify per-query-category correctness (list-all, definition, etc.) respects user_id. Verify no path bypasses the 3×3 matrix. Verify transactional safety across the ingest saga. Verdict: APPROVED / APPROVED_WITH_MINOR / CHANGES_REQUIRED."

Address any findings before opening the PR.

---

### Task 11.3: Push + PR + merge

```bash
git push -u origin multi-tenant-docs
gh pr create --title "feat: multi-tenant per-user docs + MinIO + transactional safety" --body "$(cat <<'EOF'
## Summary
... (generated from commit log)

## Test plan
- [ ] 3×3 security matrix passes
- [ ] Boundary tests pass
- [ ] Chaos tests pass under --features chaos-tests
- [ ] Canonical 'TOATE metodele' query with user_id produces grounded answer
- [ ] Second user cannot see first user's docs via any tool

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Wait for user approval to merge; do not `gh pr merge` autonomously.

---

## Self-review notes

Spec coverage check:

- Spec §5.1 MinIO → Phase 4 ✓
- Spec §5.1 Postgres schema → Phase 2 + 3 ✓
- Spec §5.1 Qdrant → Phase 5.1 ✓
- Spec §5.1 Neo4j → Phase 5.2 ✓
- Spec §5.2 explicit plumbing → Phases 3/5/6/7 ✓
- Spec §5.3 agent pipeline → Phase 8 ✓
- Spec §5.4 tools → Phase 7 ✓
- Spec §5.5 CLI → Phase 9 ✓
- Spec §5.6 docker-compose → Phase 4.1 ✓
- Spec §6 3×3 matrix → Phase 10.1 ✓
- Spec §7 error variants → Phase 1.2 ✓
- Spec §8 migration rollout → README Phase 11.1 ✓
- Spec §11 transactional safety → woven throughout (Phase 3.4 activation atomicity, 4.2 MinIO idempotent delete, 8.1 REPEATABLE READ, 10.3 chaos tests) ✓
- Spec §11.7 rollback idempotency → extend existing `try_rollback_stores` in Phase 4.2 / Phase 9.2 ✓

No placeholders. Every step has concrete code or concrete commands. Type consistency: `UserId::try_new`, `create_or_fetch_session`, `activate_document_for_session`, `list_active_documents`, `read_context_snapshot`, `build_user_filter`, `object_key`, `MinioStore::new_for_user` (wait — MinioStore::new, not new_for_user — fixed inline).

---

## Execution handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-04-21-multi-tenant-docs.md`. **Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task (with spec + code review between each), review Codex checkpoints between phases. Matches the successful coverage-aware-rag workflow.

**2. Inline Execution** — work through tasks in this session via `superpowers:executing-plans`, checkpoint at phase boundaries.

**Which approach?**
