// POST-MIGRATION VERIFICATION — 20260418_01_unify_document_id
//
// Run after `20260418_01_unify_document_id.cypher` to confirm:
//
//   1. Zero legacy (non-UUID) Document nodes remain.
//   2. Every surviving Document carries both `id` (UUID shape) and
//      `external_doc_id` (the human-facing name).
//   3. Each Document is connected to its chunks via both
//      `HAS_CHUNK` (entity-path) and `CONTAINS_CHUNK` (structural-path)
//      with matching chunk counts on both relationships — proof the
//      transfer preserved the entity-graph view.

// 1. Any legacy Documents left?
MATCH (legacy:Document)
WHERE NOT legacy.id =~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
RETURN
    'LEGACY_DOCUMENTS_REMAINING' AS check,
    count(legacy) AS count,
    collect(legacy.id)[..10] AS sample_ids;

// 2. Documents missing `external_doc_id`.
MATCH (d:Document)
WHERE d.external_doc_id IS NULL
RETURN
    'DOCUMENTS_MISSING_EXTERNAL_DOC_ID' AS check,
    count(d) AS count,
    collect(d.id)[..10] AS sample_ids;

// 3. HAS_CHUNK vs CONTAINS_CHUNK parity per Document.
MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(hc:Chunk)
OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(cc:Chunk)
WITH
    d,
    count(DISTINCT hc) AS has_chunk_count,
    count(DISTINCT cc) AS contains_chunk_count
WHERE has_chunk_count <> contains_chunk_count
RETURN
    'HAS_CHUNK_CONTAINS_CHUNK_MISMATCH' AS check,
    d.id AS document_id,
    d.external_doc_id AS external_doc_id,
    has_chunk_count,
    contains_chunk_count
ORDER BY document_id;
