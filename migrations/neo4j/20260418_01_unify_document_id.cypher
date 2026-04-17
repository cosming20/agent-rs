// FORWARD — 20260418_01_unify_document_id
//
// Consolidates the two Document nodes that pre-Q6-P2 ingests wrote per
// document:
//
//   legacy    :Document { id: <raw doc_id string> }   -[:HAS_CHUNK]->
//   canonical :Document { id: <sha256-UUIDv5 string> } -[:CONTAINS_CHUNK]->
//
// into a single canonical Document keyed by the sha256 UUID, with the
// original human-facing id preserved as `external_doc_id`.
//
// Behaviour:
//   - Detects pairs via their shared Chunk nodes.
//   - Copies `external_doc_id`, `total_chunks`, and `source` from legacy
//     to canonical only when the canonical is missing the field
//     (`coalesce` — safe to rerun).
//   - MERGEs a HAS_CHUNK edge from canonical to each shared chunk,
//     preserving the `chunk_index` property carried by the legacy edge.
//     The structural-path `CONTAINS_CHUNK` edges are left untouched.
//   - DETACH DELETEs the legacy Document, which removes its HAS_CHUNK
//     edges automatically. Chunks, Entities, MENTIONS edges, and
//     structural subgraph (Page / Section / Table / Picture) are all
//     untouched.
//
// Idempotent: once the legacy Documents are gone, the outer MATCH returns
// zero rows on subsequent runs.
//
// Run AFTER `20260418_01_unify_document_id_dry_run.cypher` confirms the
// set of pairs looks correct.
//
// Orphan legacy documents (entity-path ran, structural-path did not)
// are NOT migrated by this script — they have no canonical sibling to
// merge into. Handle those by re-ingesting the source document or by
// issuing a targeted RENAME (update `id` to the sha256-UUIDv5).

// Phase 1 — collect the pairs into a list so the write phase sees a
// stable working set even if the query is re-executed.
MATCH (legacy:Document)-[has_chunk:HAS_CHUNK]->(c:Chunk)<-[:CONTAINS_CHUNK]-(canonical:Document)
WHERE legacy.id <> canonical.id
  AND NOT legacy.id =~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
WITH legacy, canonical, c, has_chunk.chunk_index AS chunk_index

// Phase 2 — transfer the HAS_CHUNK edge to the canonical Document.
// MERGE with the existing chunk_index keeps the signal on the edge.
MERGE (canonical)-[new_rel:HAS_CHUNK]->(c)
ON CREATE SET new_rel.chunk_index = chunk_index

// Phase 3 — copy descriptive props from legacy to canonical when missing.
// Done on every row but idempotent via `coalesce`.
WITH legacy, canonical
SET
    canonical.external_doc_id = coalesce(canonical.external_doc_id, legacy.id),
    canonical.total_chunks    = coalesce(canonical.total_chunks,    legacy.total_chunks),
    canonical.source          = coalesce(canonical.source,          legacy.source)

// Phase 4 — delete the legacy Document and its edges.
WITH DISTINCT legacy
DETACH DELETE legacy;
