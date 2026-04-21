-- Reverse of up.sql. Best-effort — DROP COLUMN and table drops lose data.
-- Does NOT drop `memories.user_id` or `query_runs.user_id` — those are
-- owned by migrations 2 / 3 and stay intact.

DROP TABLE IF EXISTS session_active_documents;
DROP TABLE IF EXISTS sessions;

ALTER TABLE ingested_documents
    DROP CONSTRAINT IF EXISTS ingested_documents_ingest_status_check;

ALTER TABLE ingested_documents
    ADD CONSTRAINT ingested_documents_ingest_status_check
    CHECK (ingest_status IN ('pending', 'complete', 'partial_failure'));

DROP INDEX IF EXISTS idx_chat_messages_user_id_session_id;
ALTER TABLE chat_messages DROP COLUMN IF EXISTS user_id;

DROP INDEX IF EXISTS ux_ingested_documents_user_sha256;
DROP INDEX IF EXISTS idx_ingested_documents_user_id;
ALTER TABLE ingested_documents
    DROP COLUMN IF EXISTS minio_object_key,
    DROP COLUMN IF EXISTS user_id;

-- Restore the sha256 unique constraint.
ALTER TABLE ingested_documents
    ADD CONSTRAINT ingested_documents_sha256_key UNIQUE (sha256);
