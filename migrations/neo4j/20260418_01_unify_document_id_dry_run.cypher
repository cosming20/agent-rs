// DRY-RUN — 20260418_01_unify_document_id
//
// Read-only audit of Documents that will be rewritten by
// `20260418_01_unify_document_id.cypher`. Run this BEFORE the forward
// migration; it returns one row per (legacy, canonical) pair and the
// chunk count shared between them.
//
// A "legacy" Document is any Document whose `id` is not a UUIDv4 / UUIDv5
// shape. These were created by the entity-path writer before the Q6-P2
// fix; the structural-path writer always used the sha256-derived UUID.
//
// A "canonical" Document is the sha256-keyed node created by the
// structural path. The two are joined through their shared Chunk nodes:
// legacy uses `HAS_CHUNK`, canonical uses `CONTAINS_CHUNK`.
//
// Expected output columns:
//   legacy_id          — the raw string id (e.g. "test")
//   canonical_id       — sha256-derived UUID string
//   shared_chunk_count — number of chunks linked from both Documents
//   canonical_has_external_doc_id — true when the canonical node already
//                                   carries the property (idempotent rerun
//                                   check)

MATCH (legacy:Document)-[:HAS_CHUNK]->(c:Chunk)<-[:CONTAINS_CHUNK]-(canonical:Document)
WHERE legacy.id <> canonical.id
  AND NOT legacy.id =~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
RETURN
    legacy.id    AS legacy_id,
    canonical.id AS canonical_id,
    count(DISTINCT c) AS shared_chunk_count,
    canonical.external_doc_id IS NOT NULL AS canonical_has_external_doc_id
ORDER BY legacy_id;
