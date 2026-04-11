DROP TRIGGER IF EXISTS set_ingested_documents_updated_at ON ingested_documents;
DROP FUNCTION IF EXISTS trigger_set_ingested_documents_updated_at();
DROP INDEX IF EXISTS idx_ingested_documents_started_desc;
DROP INDEX IF EXISTS idx_ingested_documents_status;
DROP TABLE IF EXISTS ingested_documents;
