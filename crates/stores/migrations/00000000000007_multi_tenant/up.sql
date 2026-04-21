-- 2026-04-21 multi_tenant: per-user document library + sessions + activations
--
-- Historical note: migrations 00000000000002_create_memories and
-- 00000000000003_create_query_runs ALREADY created `user_id` on their
-- respective tables (without any app-layer scoping logic, so nothing
-- enforced it). This migration completes the story for ingested_documents
-- and chat_messages, and then adds the new sessions / session_active_documents
-- tables.
--
-- Uses `IF NOT EXISTS` on column adds so a partially-applied retry is
-- safe to re-run.

-- 1) Add user_id to the tables that don't already have it.

ALTER TABLE ingested_documents
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL,
    ADD COLUMN IF NOT EXISTS minio_object_key TEXT;

CREATE INDEX IF NOT EXISTS idx_ingested_documents_user_id
    ON ingested_documents (user_id);

-- Per-user sha256 dedup (replaces the plain sha256 unique constraint so
-- two users can upload identical PDFs, each getting their own document_id).
ALTER TABLE ingested_documents
    DROP CONSTRAINT IF EXISTS ingested_documents_sha256_key;

CREATE UNIQUE INDEX IF NOT EXISTS ux_ingested_documents_user_sha256
    ON ingested_documents (user_id, sha256);

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id_session_id
    ON chat_messages (user_id, session_id);

-- NOTE: memories.user_id already exists (migration 2). Skip.
-- NOTE: query_runs.user_id already exists (migration 3). Skip.

-- 2) sessions: one row per UUID, tied to a user. session_active_documents
--    FKs into this to prevent dangling activations.

CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);

-- 3) session_active_documents: which subset of the user's library
--    participates in this conversation.

CREATE TABLE IF NOT EXISTS session_active_documents (
    session_id UUID NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES ingested_documents (document_id) ON DELETE CASCADE,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_session_active_documents_document
    ON session_active_documents (document_id);

-- 4) Widen ingest_status check so new states from spec §11.1 are valid.
--    Existing statuses: 'pending', 'complete', 'partial_failure'.
--    Add 'in_progress' (begin_ingestion transition), 'failed' (MinIO upload
--    fail), and 'deleting' (spec §11.3).

ALTER TABLE ingested_documents
    DROP CONSTRAINT IF EXISTS ingested_documents_ingest_status_check;

ALTER TABLE ingested_documents
    ADD CONSTRAINT ingested_documents_ingest_status_check
    CHECK (ingest_status IN (
        'in_progress', 'pending', 'complete', 'partial_failure', 'failed', 'deleting'
    ));
