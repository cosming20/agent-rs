-- Ingested documents: tracks every PDF/document parsed via the Docling sidecar
-- (or legacy fallback). Provides sha256-keyed dedup, ingest state machine
-- tracking, and operator-visible failure observability.

CREATE TABLE ingested_documents (
    -- UUIDv5(NAMESPACE_OID, sha256). Stable across reingestion of identical bytes.
    document_id              UUID              PRIMARY KEY,

    -- Lowercase hex SHA-256 of the input bytes. Globally unique:
    -- two different document_ids cannot share a sha256 (and vice versa).
    -- UNIQUE constraint implicitly creates a unique index (idx is NOT added separately).
    sha256                   CHAR(64)          NOT NULL UNIQUE
        CONSTRAINT chk_sha256_format CHECK (sha256 ~ '^[0-9a-f]{64}$'),

    -- Originating filename or URL fragment. Free-form, not validated.
    source                   TEXT              NOT NULL,

    -- Ingest state machine. CHECK constraint instead of native ENUM to
    -- avoid pulling in diesel-derive-enum (extra dep + migration churn).
    ingest_status            TEXT              NOT NULL
        CHECK (ingest_status IN ('in_progress', 'partial_failure', 'complete')),

    -- New UUIDv4 generated each time begin_ingestion runs. Used to tag
    -- Qdrant points and Neo4j nodes for rollback. Replaced (NOT history-tracked)
    -- on retry of a partial_failure row.
    ingest_run_id            UUID              NOT NULL,

    -- Reason string when ingest_status = 'partial_failure'. NULL otherwise.
    failure_reason           TEXT              NULL,

    -- Header metadata captured at finalize time. NULL until SidecarSucceeded.
    docling_version          TEXT              NULL,
    docling_core_version     TEXT              NULL,
    schema_version           INTEGER           NULL,
    n_pages                  INTEGER           NULL,
    n_chunks                 INTEGER           NULL,
    n_tables                 INTEGER           NULL,
    n_pictures               INTEGER           NULL,
    n_formulas               INTEGER           NULL,
    language                 TEXT              NULL,
    title                    TEXT              NULL,
    mimetype                 TEXT              NULL,

    -- Latency observability.
    parse_latency_ms         BIGINT            NULL,  -- sidecar parse time
    total_ingest_latency_ms  BIGINT            NULL,  -- end-to-end (Rust caller view)

    -- Lifecycle timestamps.
    started_at               TIMESTAMP         NOT NULL DEFAULT NOW(),
    completed_at             TIMESTAMP         NULL,
    updated_at               TIMESTAMP         NOT NULL DEFAULT NOW()
);

-- Operator dashboard: list partial failures.
CREATE INDEX idx_ingested_documents_status
    ON ingested_documents (ingest_status);

-- Recent-ingests view.
CREATE INDEX idx_ingested_documents_started_desc
    ON ingested_documents (started_at DESC);

-- Auto-update updated_at on UPDATE.
CREATE OR REPLACE FUNCTION trigger_set_ingested_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_ingested_documents_updated_at
    BEFORE UPDATE ON ingested_documents
    FOR EACH ROW
    EXECUTE FUNCTION trigger_set_ingested_documents_updated_at();
