-- Initial consolidated schema for agent-rs (stateless agent architecture).
--
-- Design notes:
--
--   * Sessions, chat_messages, session_active_documents, ingested_documents
--     used to live here, but the 2026-04 stateless pivot moved all session
--     and library state to agent-rs-web. Agent-rs now only keeps:
--
--       - memories              (user-level long-term facts, invisible to proto)
--       - query_runs            (per-query telemetry, correlated by trace_id)
--       - evidence_items        (evidence captured per run — FK cascades)
--       - strategy_performance  (adaptive strategy learning table)
--       - document_index_state  (source of truth for GetDocumentStatus RPC —
--                                maps MinIO key -> indexed state machine)
--
--   * Everything user-visible (sessions, chat history, doc library) is
--     owned by agent-rs-web's Postgres. Agent receives history + active
--     doc keys in each Ask request; no session ownership checks here.
--
--   * `trace_id` (nullable) replaces the old `session_id` for correlating
--     telemetry. It's a client-provided UUID with no authorization meaning.

-- ---------------------------------------------------------------------------
-- memories: user-level long-term memory
-- ---------------------------------------------------------------------------

CREATE TABLE memories (
    id          UUID        PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Covering index for the dominant query pattern: "last N memories for user X".
-- A plain (user_id) index would force a sort after the scan; the composite
-- with DESC lets Postgres walk the btree backwards in O(log n + N).
CREATE INDEX idx_memories_user_created ON memories (user_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- query_runs: per-Ask telemetry row
-- ---------------------------------------------------------------------------

CREATE TABLE query_runs (
    id           UUID              PRIMARY KEY,
    trace_id     UUID              NULL,
    user_id      TEXT              NOT NULL,
    query        TEXT              NOT NULL,
    answer       TEXT              NOT NULL,
    strategy     TEXT              NOT NULL,
    confidence   DOUBLE PRECISION  NOT NULL,
    is_grounded  BOOLEAN           NOT NULL,
    created_at   TIMESTAMPTZ       NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_confidence_range CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Composite descending index for "recent query runs for user X" (per-user
-- telemetry views). `(user_id, created_at DESC)` serves both "count per user"
-- and "latest N per user" without a separate single-column index.
CREATE INDEX idx_query_runs_user_created ON query_runs (user_id, created_at DESC);
-- Partial because most rows will have trace_id NULL (background jobs, CLI
-- tools); indexing only non-NULL keeps the index small.
CREATE INDEX idx_query_runs_trace_id ON query_runs (trace_id) WHERE trace_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- evidence_items: evidence surfaced per run (cascade-deleted with parent)
-- ---------------------------------------------------------------------------

CREATE TABLE evidence_items (
    id          UUID              PRIMARY KEY,
    run_id      UUID              NOT NULL REFERENCES query_runs (id) ON DELETE CASCADE,
    content     TEXT              NOT NULL,
    source      TEXT              NOT NULL,
    score       DOUBLE PRECISION  NOT NULL,
    created_at  TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_items_run_id ON evidence_items (run_id);

-- ---------------------------------------------------------------------------
-- strategy_performance: adaptive strategy selection telemetry (spec D1)
-- ---------------------------------------------------------------------------

CREATE TABLE strategy_performance (
    id                      BIGSERIAL       PRIMARY KEY,
    query_hash              CHAR(64)        NOT NULL,
    query_intent            TEXT            NULL,
    strategy                TEXT            NOT NULL,
    evidence_count          INTEGER         NOT NULL DEFAULT 0,
    avg_relevance_score     REAL            NULL,
    is_grounded             BOOLEAN         NULL,
    confidence              REAL            NULL,
    latency_ms              BIGINT          NOT NULL DEFAULT 0,
    required_retry          BOOLEAN         NOT NULL DEFAULT FALSE,
    sub_query_count         INTEGER         NOT NULL DEFAULT 1,
    used_hyde               BOOLEAN         NOT NULL DEFAULT FALSE,
    used_metadata_filter    BOOLEAN         NOT NULL DEFAULT FALSE,
    used_context_expansion  BOOLEAN         NOT NULL DEFAULT FALSE,
    linked_entities         INTEGER         NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_strategy_perf_intent_strategy ON strategy_performance (query_intent, strategy);
CREATE INDEX idx_strategy_perf_created ON strategy_performance (created_at DESC);
CREATE INDEX idx_strategy_perf_strategy ON strategy_performance (strategy);
-- Fingerprint lookup: "what strategy worked for this exact normalized query
-- before?" — the adaptive-selector's feedback loop.
CREATE INDEX idx_strategy_perf_query_hash ON strategy_performance (query_hash);

-- ---------------------------------------------------------------------------
-- document_index_state: MinIO key -> indexed state machine
--
-- The IndexerWorker consuming RabbitMQ writes to this table. The
-- GetDocumentStatus RPC reads from it. Web polls until state = 'complete'.
-- ---------------------------------------------------------------------------

CREATE TABLE document_index_state (
    minio_object_key    TEXT            PRIMARY KEY,
    user_id             TEXT            NOT NULL,
    state               TEXT            NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'indexing', 'complete', 'failed')),
    sha256              CHAR(64)        NULL
        CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    n_pages             INTEGER         NULL,
    n_chunks            INTEGER         NULL,
    error_message       TEXT            NULL,
    queued_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ     NULL,
    completed_at        TIMESTAMPTZ     NULL,
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_document_index_state_user_id ON document_index_state (user_id);
-- Partial index for the "show me recent failures" operator view. A plain
-- index on `state` has too low cardinality (4 values) to help queries with
-- realistic distributions dominated by `complete`; a partial index kept tiny
-- over just the failure rows is what the dashboard actually needs.
CREATE INDEX idx_document_index_state_failed
    ON document_index_state (updated_at DESC) WHERE state = 'failed';

-- ---------------------------------------------------------------------------
-- Shared helpers
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER document_index_state_touch
    BEFORE UPDATE ON document_index_state
    FOR EACH ROW
    EXECUTE FUNCTION trigger_set_updated_at();
