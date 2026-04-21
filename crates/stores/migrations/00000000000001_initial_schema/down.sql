DROP TRIGGER IF EXISTS document_index_state_touch ON document_index_state;
DROP FUNCTION IF EXISTS trigger_set_updated_at();

DROP TABLE IF EXISTS document_index_state;
DROP TABLE IF EXISTS strategy_performance;
DROP TABLE IF EXISTS evidence_items;
DROP TABLE IF EXISTS query_runs;
DROP TABLE IF EXISTS memories;
