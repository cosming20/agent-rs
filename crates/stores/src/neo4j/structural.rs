//! Structural graph layer for Docling-sourced document metadata.
//!
//! This module creates and manages the structural skeleton of a document in
//! Neo4j: `Document`, `Page`, `Section`, `Table`, `Picture`, and `Chunk`
//! nodes connected by typed edges (`ON_PAGE`, `IN_SECTION`, `HAS_PAGE`, etc.).
//!
//! # Document identity (post 2026-04-21 stateless pivot)
//!
//! The `Document.id` property is the authoritative document handle across
//! the system and carries the **MinIO object key** (matching the
//! `AskRequest.active_document_keys` wire shape). Readers
//! (`Neo4jStore::list_sections_for_doc`, `list_descendant_sections`,
//! `chunks_in_sections`) accept this key as an opaque string; the
//! RabbitMQ `IndexerWorker` writes it into `Document.id` on every upsert.
//! `DocumentHeaderSnapshot.document_id` stays a `Uuid` because the
//! structural sub-node ids (Page, Section) are derived from it via
//! UUIDv5 for content-addressing stability — that derivation is an
//! internal detail and never leaves the graph.
//!
//! # INVARIANT — `ingest_run_id` is additive-only
//!
//! Every node created by this module carries an `ingest_run_id` property
//! set via `ON CREATE SET`, **never** via `SET`. This means:
//!
//! - `ingest_run_id` is written ONLY when the node is first created in this
//!   ingestion run.
//! - If a node already exists (e.g., the same Document was ingested before),
//!   `MERGE` finds it and the `ON CREATE SET` clause is skipped — the
//!   pre-existing node retains whatever `ingest_run_id` (or no property) it
//!   had before.
//! - `rollback_by_run_id` deletes only nodes where
//!   `ingest_run_id = $run_id`. Pre-existing `Document`, `Chunk`, and
//!   `Entity` nodes created by the LLM knowledge-graph path in
//!   `pipeline.rs` **never** carry an `ingest_run_id`, so they are
//!   **never** matched by the rollback query.
//! - Therefore a rollback cannot accidentally destroy data from a different
//!   ingestion run or from the LLM KG path.

use agent_rs_core::{StoreError, UserId};
use neo4rs::{Graph, query};
use serde_json::{Value, json};
use tracing::instrument;
use uuid::Uuid;

use super::Neo4jStore;

// ---------------------------------------------------------------------------
// Batch size
// ---------------------------------------------------------------------------

/// Maximum number of nodes or edges written in a single Neo4j transaction.
///
/// Keeping this small bounds both transaction memory and lock duration on the
/// server side. 100 is chosen to stay well within Neo4j's default heap
/// settings and to match the spec requirement.
const BATCH_SIZE: usize = 100;

// ---------------------------------------------------------------------------
// Hard-coded label / relation-type constants (E4: never from sidecar input)
//
// Labels are passed as &'static str to upsert_nodes_batch so they can be
// safely interpolated into Cypher templates. They are compile-time ASCII
// identifiers — no runtime input ever reaches these strings.
// ---------------------------------------------------------------------------

// Structural node label tokens.
pub(crate) const LABEL_DOCUMENT: &str = "Document";
pub(crate) const LABEL_PAGE: &str = "Page";
pub(crate) const LABEL_SECTION: &str = "Section";
const LABEL_TABLE: &str = "Table";
const LABEL_PICTURE: &str = "Picture";

// Structural relation type tokens — hardcoded, never accepted from user/sidecar input.
//
// Each token is a plain &'static str. They are safe to interpolate into
// Cypher because they are compile-time ASCII identifiers that pass
// `RelationType` validation (alphanumeric + underscore).  We do NOT call
// `RelationType::new()` here because the format!() interpolation is of
// hand-audited constant strings, not of any runtime value.
const REL_HAS_PAGE: &str = "HAS_PAGE";
const REL_HAS_SECTION: &str = "HAS_SECTION";
const REL_HAS_TABLE: &str = "HAS_TABLE";
const REL_HAS_PICTURE: &str = "HAS_PICTURE";
const REL_CONTAINS_CHUNK: &str = "CONTAINS_CHUNK";
const REL_ON_PAGE: &str = "ON_PAGE";
const REL_IN_SECTION: &str = "IN_SECTION";
const REL_REFERENCES_TABLE: &str = "REFERENCES_TABLE";
const REL_REFERENCES_PICTURE: &str = "REFERENCES_PICTURE";
const REL_FOLLOWS: &str = "FOLLOWS";
const REL_PARENT_OF: &str = "PARENT_OF";

// ---------------------------------------------------------------------------
// Public input types
// ---------------------------------------------------------------------------

/// Snapshot of the document header provided by the Rust ingest orchestrator
/// after reading it from the sidecar's `DocumentHeader` proto message.
#[derive(Debug, Clone)]
pub struct DocumentHeaderSnapshot {
    /// Stable document identity (UUIDv5 derived from source + sha256).
    pub document_id: Uuid,
    /// Run-specific identifier for rollback targeting.
    pub ingest_run_id: Uuid,
    /// Owning user — written as `user_id` on the `Document` node via
    /// `ON CREATE SET` (idempotent on re-ingest for the same `document_id`).
    pub user_id: UserId,
    /// SHA-256 hex digest of the original file.
    pub sha256: String,
    /// Human-readable source URI / file path.
    pub source: String,
    /// Docling version string, if available.
    pub docling_version: Option<String>,
    /// Total page count.
    pub n_pages: i64,
    /// Total chunk count.
    pub n_chunks: i64,
    /// Total table count.
    pub n_tables: i64,
    /// Total picture count.
    pub n_pictures: i64,
    /// Detected language code (e.g. `"en"`).
    pub language: Option<String>,
    /// Document title, if extractable.
    pub title: Option<String>,
    /// MIME type of the source file.
    pub mimetype: Option<String>,
    /// Per-page geometry metadata.
    pub pages: Vec<PageInfo>,
}

/// Physical page dimensions from the sidecar.
#[derive(Debug, Clone)]
pub struct PageInfo {
    /// 1-based page number.
    pub page_no: i64,
    pub width: f64,
    pub height: f64,
}

/// Structural metadata for one chunk, assembled from sidecar output.
///
/// The `chunk_id` must match the node id already written by the LLM KG path
/// in `pipeline.rs`. This module only enriches / upserts structural properties
/// onto the pre-existing Chunk node.
#[derive(Debug, Clone)]
pub struct StructuralChunk {
    /// Must match the `id` property on the existing `Chunk` node.
    pub chunk_id: Uuid,
    /// First page the chunk spans (1-based).
    pub page_start: i64,
    /// Last page the chunk spans (1-based).
    pub page_end: i64,
    /// UUID of the deepest containing section, if any.
    pub section_id: Option<Uuid>,
    /// Section heading text.
    pub section_title: Option<String>,
    /// Heading depth (1 = h1, 2 = h2, …).
    pub heading_level: Option<i64>,
    /// Ordered list of ancestor heading titles, outermost first.
    pub heading_path: Vec<String>,
    /// Chunk contains at least one table element.
    pub has_table: bool,
    /// Chunk contains at least one list element.
    pub has_list: bool,
    /// Chunk contains at least one code block.
    pub has_code: bool,
    /// Chunk contains at least one formula.
    pub has_formula: bool,
    /// Chunk references at least one picture by self-ref.
    pub has_picture_ref: bool,
    /// Chunk contains at least one footnote.
    pub has_footnote: bool,
    /// Chunk contains at least one caption.
    pub has_caption: bool,
    /// Table node IDs referenced by this chunk.
    pub table_refs: Vec<Uuid>,
    /// Picture node IDs referenced by this chunk.
    pub picture_refs: Vec<Uuid>,
    /// Previous chunk's ID (for `FOLLOWS` edge); `None` for the first chunk.
    pub prev_chunk_id: Option<Uuid>,
}

/// Structural metadata for one table in the document.
#[derive(Debug, Clone)]
pub struct StructuralTable {
    /// Stable identity (UUIDv5 derived from doc_id + self_ref).
    pub table_id: Uuid,
    /// Docling `self_ref` string (e.g. `"#/tables/0"`).
    pub doc_item_self_ref: String,
    /// 1-based page number.
    pub page_no: i64,
    pub n_rows: i64,
    pub n_cols: i64,
    pub caption: Option<String>,
}

/// Structural metadata for one picture in the document.
#[derive(Debug, Clone)]
pub struct StructuralPicture {
    /// Stable identity (UUIDv5 derived from doc_id + self_ref).
    pub picture_id: Uuid,
    /// Docling `self_ref` string (e.g. `"#/pictures/0"`).
    pub doc_item_self_ref: String,
    /// 1-based page number.
    pub page_no: i64,
    pub caption: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal row types (serialized to JSON, then to BoltType for UNWIND batches)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct NodeRow {
    id: String,
    run_id: String,
    props: Value,
}

#[derive(serde::Serialize)]
struct EdgeRow {
    src: String,
    dst: String,
    run_id: String,
}

// ---------------------------------------------------------------------------
// Neo4jStore impl
// ---------------------------------------------------------------------------

impl Neo4jStore {
    /// Create (or verify) the six structural constraints in Neo4j.
    ///
    /// Uses `IF NOT EXISTS` so this is safe to call on every startup and from
    /// multiple agent-rs instances simultaneously. Existing constraints are
    /// never replaced.
    ///
    /// Constraints created:
    /// - `document_id_unique`  — `(d:Document) REQUIRE d.id IS UNIQUE`
    /// - `page_id_unique`      — `(p:Page)     REQUIRE p.id IS UNIQUE`
    /// - `section_id_unique`   — `(s:Section)  REQUIRE s.id IS UNIQUE`
    /// - `table_id_unique`     — `(t:Table)    REQUIRE t.id IS UNIQUE`
    /// - `picture_id_unique`   — `(p:Picture)  REQUIRE p.id IS UNIQUE`
    /// - `chunk_id_unique`     — `(c:Chunk)    REQUIRE c.id IS UNIQUE`
    #[instrument(name = "neo4j_store.ensure_structural_constraints", skip(self))]
    pub async fn ensure_structural_constraints(&self) -> Result<(), StoreError> {
        let stmts: &[&str] = &[
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS \
             FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT page_id_unique IF NOT EXISTS \
             FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS \
             FOR (s:Section) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT table_id_unique IF NOT EXISTS \
             FOR (t:Table) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT picture_id_unique IF NOT EXISTS \
             FOR (pic:Picture) REQUIRE pic.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS \
             FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ];

        for stmt in stmts {
            self.graph
                .run(query(stmt))
                .await
                .map_err(StoreError::from)?;
        }

        tracing::info!("structural constraints verified (6 constraints)");
        Ok(())
    }

    /// Upsert the full structural graph for one document in batched transactions.
    ///
    /// All nodes created in this call are tagged with `header.ingest_run_id`
    /// via `ON CREATE SET`, making them eligible for atomic rollback via
    /// [`Self::rollback_by_run_id`].
    ///
    /// # Order of operations
    ///
    /// 1. Upsert `Document` node
    /// 2. Upsert `Page` nodes in batches of 100
    /// 3. Collect unique `Section` nodes from `chunks` → upsert in batches of 100
    /// 4. Upsert `Table` nodes in batches of 100
    /// 5. Upsert `Picture` nodes in batches of 100
    /// 6. Update `Chunk` structural properties (SET on existing nodes — no `ingest_run_id` touch)
    /// 7. Upsert all edge types in batches of 100 (with count verification per batch)
    ///
    /// # Errors
    ///
    /// Returns the first error encountered. On error the caller must invoke
    /// `rollback_by_run_id` to clean up partial writes.
    #[instrument(
        name = "neo4j_store.upsert_document_structure",
        skip(self, header, chunks, tables, pictures),
        fields(
            doc_id  = %header.document_id,
            run_id  = %header.ingest_run_id,
            user_id = %header.user_id,
            n_chunks = chunks.len(),
            n_tables = tables.len(),
            n_pictures = pictures.len(),
        )
    )]
    pub async fn upsert_document_structure(
        &self,
        header: &DocumentHeaderSnapshot,
        chunks: &[StructuralChunk],
        tables: &[StructuralTable],
        pictures: &[StructuralPicture],
    ) -> Result<(), StoreError> {
        let doc_id = header.document_id.to_string();
        let run_id = header.ingest_run_id.to_string();
        let user_id_str = header.user_id.as_str().to_owned();

        // ------------------------------------------------------------------
        // Phase 1: Document node
        //
        // `user_id` is written via ON CREATE SET (idempotent on re-ingest —
        // the owning user never changes for a given document_id).
        // ------------------------------------------------------------------
        upsert_document_node(
            &self.graph,
            &doc_id,
            &run_id,
            &user_id_str,
            json!({
                "sha256":           header.sha256,
                "source":           header.source,
                "n_pages":          header.n_pages,
                "n_chunks":         header.n_chunks,
                "n_tables":         header.n_tables,
                "n_pictures":       header.n_pictures,
                "language":         header.language,
                "title":            header.title,
                "mimetype":         header.mimetype,
                "docling_version":  header.docling_version,
            }),
        )
        .await?;

        // ------------------------------------------------------------------
        // Phase 2: Page nodes
        // ------------------------------------------------------------------
        let page_rows: Vec<NodeRow> = header
            .pages
            .iter()
            .map(|p| NodeRow {
                id: page_id(header.document_id, p.page_no),
                run_id: run_id.clone(),
                props: json!({
                    "page_no":     p.page_no,
                    "width":       p.width,
                    "height":      p.height,
                    "document_id": doc_id,
                }),
            })
            .collect();

        upsert_nodes_batch(&self.graph, LABEL_PAGE, &page_rows).await?;

        // ------------------------------------------------------------------
        // Phase 3: Section nodes (deduplicated from chunks)
        // ------------------------------------------------------------------
        let section_rows = collect_section_rows(chunks, &run_id, &doc_id);
        upsert_nodes_batch(&self.graph, LABEL_SECTION, &section_rows).await?;

        // ------------------------------------------------------------------
        // Phase 4: Table nodes
        // ------------------------------------------------------------------
        let table_rows: Vec<NodeRow> = tables
            .iter()
            .map(|t| NodeRow {
                id: t.table_id.to_string(),
                run_id: run_id.clone(),
                props: json!({
                    "self_ref":    t.doc_item_self_ref,
                    "page_no":     t.page_no,
                    "n_rows":      t.n_rows,
                    "n_cols":      t.n_cols,
                    "caption":     t.caption,
                    "document_id": doc_id,
                }),
            })
            .collect();

        upsert_nodes_batch(&self.graph, LABEL_TABLE, &table_rows).await?;

        // ------------------------------------------------------------------
        // Phase 5: Picture nodes
        // ------------------------------------------------------------------
        let picture_rows: Vec<NodeRow> = pictures
            .iter()
            .map(|p| NodeRow {
                id: p.picture_id.to_string(),
                run_id: run_id.clone(),
                props: json!({
                    "self_ref":    p.doc_item_self_ref,
                    "page_no":     p.page_no,
                    "caption":     p.caption,
                    "document_id": doc_id,
                }),
            })
            .collect();

        upsert_nodes_batch(&self.graph, LABEL_PICTURE, &picture_rows).await?;

        // ------------------------------------------------------------------
        // Phase 6: Enrich Chunk nodes with structural properties
        //
        // These nodes were already created by the LLM KG path. We SET the
        // structural fields without touching ingest_run_id (additive-only
        // invariant: only ON CREATE SET is allowed to set ingest_run_id).
        // ------------------------------------------------------------------
        update_chunk_structural_props(&self.graph, chunks).await?;

        // ------------------------------------------------------------------
        // Phase 7: Relationships
        // ------------------------------------------------------------------

        // Document → Page  (HAS_PAGE)
        let doc_page_edges: Vec<EdgeRow> = header
            .pages
            .iter()
            .map(|p| EdgeRow {
                src: doc_id.clone(),
                dst: page_id(header.document_id, p.page_no),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_HAS_PAGE, &doc_page_edges).await?;

        // Document → Section  (HAS_SECTION)
        let doc_section_edges: Vec<EdgeRow> = section_rows
            .iter()
            .map(|s| EdgeRow {
                src: doc_id.clone(),
                dst: s.id.clone(),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_HAS_SECTION, &doc_section_edges).await?;

        // Document → Table  (HAS_TABLE)
        let doc_table_edges: Vec<EdgeRow> = tables
            .iter()
            .map(|t| EdgeRow {
                src: doc_id.clone(),
                dst: t.table_id.to_string(),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_HAS_TABLE, &doc_table_edges).await?;

        // Document → Picture  (HAS_PICTURE)
        let doc_picture_edges: Vec<EdgeRow> = pictures
            .iter()
            .map(|p| EdgeRow {
                src: doc_id.clone(),
                dst: p.picture_id.to_string(),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_HAS_PICTURE, &doc_picture_edges).await?;

        // Document → Chunk  (CONTAINS_CHUNK)
        let doc_chunk_edges: Vec<EdgeRow> = chunks
            .iter()
            .map(|c| EdgeRow {
                src: doc_id.clone(),
                dst: c.chunk_id.to_string(),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_CONTAINS_CHUNK, &doc_chunk_edges).await?;

        // Chunk → Page  (ON_PAGE — use page_start)
        let chunk_page_edges: Vec<EdgeRow> = chunks
            .iter()
            .map(|c| EdgeRow {
                src: c.chunk_id.to_string(),
                dst: page_id(header.document_id, c.page_start),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_ON_PAGE, &chunk_page_edges).await?;

        // Chunk → Section  (IN_SECTION — only chunks that have a section)
        let chunk_section_edges: Vec<EdgeRow> = chunks
            .iter()
            .filter_map(|c| {
                c.section_id.map(|sid| EdgeRow {
                    src: c.chunk_id.to_string(),
                    dst: sid.to_string(),
                    run_id: run_id.clone(),
                })
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_IN_SECTION, &chunk_section_edges).await?;

        // Chunk → Table  (REFERENCES_TABLE)
        let chunk_table_edges: Vec<EdgeRow> = chunks
            .iter()
            .flat_map(|c| {
                c.table_refs.iter().map(|tid| EdgeRow {
                    src: c.chunk_id.to_string(),
                    dst: tid.to_string(),
                    run_id: run_id.clone(),
                })
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_REFERENCES_TABLE, &chunk_table_edges).await?;

        // Chunk → Picture  (REFERENCES_PICTURE)
        let chunk_picture_edges: Vec<EdgeRow> = chunks
            .iter()
            .flat_map(|c| {
                c.picture_refs.iter().map(|pid| EdgeRow {
                    src: c.chunk_id.to_string(),
                    dst: pid.to_string(),
                    run_id: run_id.clone(),
                })
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_REFERENCES_PICTURE, &chunk_picture_edges).await?;

        // Chunk_i → Chunk_{i+1}  (FOLLOWS)
        let follows_edges: Vec<EdgeRow> = chunks
            .iter()
            .filter_map(|c| {
                c.prev_chunk_id.map(|prev| EdgeRow {
                    src: prev.to_string(),
                    dst: c.chunk_id.to_string(),
                    run_id: run_id.clone(),
                })
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_FOLLOWS, &follows_edges).await?;

        // Section → Section  (PARENT_OF — derives from heading_path depth)
        let parent_of_edges = collect_parent_of_edges(chunks, &run_id);
        upsert_edges_batch(&self.graph, REL_PARENT_OF, &parent_of_edges).await?;

        tracing::info!(
            doc_id = %header.document_id,
            run_id = %header.ingest_run_id,
            "structural graph upsert complete"
        );
        Ok(())
    }

    /// Delete every node (and its incident edges) tagged with `ingest_run_id`.
    ///
    /// # Safety
    ///
    /// This query only matches nodes where `ingest_run_id = $run_id`. Because
    /// `ingest_run_id` is set via `ON CREATE SET` only, pre-existing nodes
    /// from the LLM KG path (`Document`, `Chunk`, `Entity`) that were NOT
    /// created in this run will never carry this run's `ingest_run_id`, and
    /// will therefore never be deleted by this query.
    ///
    /// Returns the count of deleted nodes.
    #[instrument(name = "neo4j_store.rollback_by_run_id", skip(self), fields(run_id = %ingest_run_id))]
    pub async fn rollback_by_run_id(&self, ingest_run_id: Uuid) -> Result<u64, StoreError> {
        let run_id_str = ingest_run_id.to_string();

        // E2: always pass ingest_run_id as String, never as Uuid directly.
        //
        // Two-step approach: count first, then delete.  The prior single-query
        // `WITH n, count(n) AS total DETACH DELETE n RETURN total` grouped
        // per-row and always returned 1 or 0, regardless of how many nodes
        // were actually deleted (W7 finding #5).
        let mut count_result = self
            .graph
            .execute(
                query("MATCH (n {ingest_run_id: $run_id}) RETURN count(n) AS total")
                    .param("run_id", run_id_str.clone()),
            )
            .await
            .map_err(StoreError::from)?;

        let deleted: u64 = match count_result.next().await.map_err(StoreError::from)? {
            Some(row) => {
                let n: i64 = row.get("total").map_err(|e| {
                    StoreError::QueryFailed(format!("rollback count deserialize: {e}"))
                })?;
                n.max(0) as u64
            }
            None => 0,
        };

        // Now actually delete.
        if deleted > 0 {
            self.graph
                .run(
                    query("MATCH (n {ingest_run_id: $run_id}) DETACH DELETE n")
                        .param("run_id", run_id_str.clone()),
                )
                .await
                .map_err(StoreError::from)?;
        }

        tracing::info!(run_id = %run_id_str, deleted_nodes = deleted, "rollback complete");
        Ok(deleted)
    }

    /// Delete a document subgraph by its MinIO object key.
    ///
    /// Detach-deletes the `Document` node matching `(id = $minio_key,
    /// user_id = $user_id)` and every descendant reachable through the
    /// `HAS_PAGE` / `HAS_SECTION` / `HAS_TABLE` / `HAS_PICTURE` edges
    /// (i.e. `Page`, `Section`, `Chunk`, `Table`, `Picture`). Cross-user
    /// deletes are a no-op because the `user_id` predicate is on the root
    /// node.
    ///
    /// Returns the total number of nodes detached and deleted (for logs
    /// and the `DeleteDocumentResponse.removed_sections` field — we use
    /// total descendants as an approximation since the proto does not
    /// separate chunks / sections / pages).
    ///
    /// # Errors
    ///
    /// - [`StoreError`] on Neo4j query failure.
    #[instrument(skip(self), fields(user_id = %user_id, minio_object_key = %minio_object_key))]
    pub async fn delete_document_by_minio_key(
        &self,
        user_id: &UserId,
        minio_object_key: &str,
    ) -> Result<u64, StoreError> {
        // Count first so we can return an accurate number — Neo4j's DETACH
        // DELETE does not report a node count at the driver layer.
        let count_cypher = format!(
            "MATCH (d:{LABEL_DOCUMENT} {{id: $minio_key, user_id: $user_id}}) \
             OPTIONAL MATCH (d)-[:{REL_HAS_PAGE}|{REL_HAS_SECTION}|{REL_HAS_TABLE}|{REL_HAS_PICTURE}*0..]->(n) \
             RETURN count(DISTINCT n) AS total"
        );
        let mut rs = self
            .graph
            .execute(
                query(&count_cypher)
                    .param("minio_key", minio_object_key)
                    .param("user_id", user_id.as_str()),
            )
            .await
            .map_err(StoreError::from)?;
        let deleted: u64 = match rs.next().await.map_err(StoreError::from)? {
            Some(row) => {
                let n: i64 = row.get("total").map_err(|e| {
                    StoreError::QueryFailed(format!("delete count deserialize: {e}"))
                })?;
                n.max(0) as u64
            }
            None => 0,
        };

        if deleted == 0 {
            return Ok(0);
        }

        let delete_cypher = format!(
            "MATCH (d:{LABEL_DOCUMENT} {{id: $minio_key, user_id: $user_id}}) \
             OPTIONAL MATCH (d)-[:{REL_HAS_PAGE}|{REL_HAS_SECTION}|{REL_HAS_TABLE}|{REL_HAS_PICTURE}*0..]->(n) \
             DETACH DELETE d, n"
        );
        self.graph
            .run(
                query(&delete_cypher)
                    .param("minio_key", minio_object_key)
                    .param("user_id", user_id.as_str()),
            )
            .await
            .map_err(StoreError::from)?;

        tracing::info!(
            user_id = %user_id,
            minio_object_key = %minio_object_key,
            deleted_nodes = deleted,
            "document subgraph deleted"
        );
        Ok(deleted)
    }

    /// Minimal structural upsert used by the RabbitMQ `IndexerWorker`
    /// MVP path (pre-Docling integration).
    ///
    /// Writes:
    ///   - One `Document { id: minio_object_key, user_id, sha256, source,
    ///     n_chunks }` node.
    ///   - One `Section { id: <uuidv5(minio_key, "section-{i}")>,
    ///     doc_id: minio_object_key, heading: <first ~80 chars of chunk
    ///     text>, section_order: i, first_chunk_index: i,
    ///     is_preamble: false }` per chunk.
    ///   - `(Document)-[:HAS_SECTION]->(Section)` edges so
    ///     `list_sections_for_doc(user_id, minio_key)` returns one row
    ///     per chunk — enough for `coverage_planner` and
    ///     `outline_corpus` to emit a flat TOC.
    ///
    /// Rich Docling-derived structure (real sections, pages, tables,
    /// pictures) continues to flow through [`Self::upsert_document_structure`];
    /// this method is the fallback wired when the Docling sidecar is
    /// unavailable or when indexing plain text.
    ///
    /// Idempotent via `MERGE`. Re-runs overwrite the node/section props
    /// in place but preserve any prior `ingest_run_id`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Neo4j query failure or props serialisation.
    #[instrument(
        skip(self, chunk_headings),
        fields(
            user_id = %user_id,
            minio_object_key = %minio_object_key,
            n_sections = chunk_headings.len(),
        )
    )]
    pub async fn upsert_minimal_document(
        &self,
        user_id: &UserId,
        minio_object_key: &str,
        sha256: &str,
        source: &str,
        chunk_headings: &[String],
    ) -> Result<(), StoreError> {
        let run_id = Uuid::new_v4().to_string();

        // ---- Document node ---------------------------------------------
        upsert_document_node(
            &self.graph,
            minio_object_key,
            &run_id,
            user_id.as_str(),
            json!({
                "sha256": sha256,
                "source": source,
                "n_chunks": chunk_headings.len() as i64,
            }),
        )
        .await?;

        if chunk_headings.is_empty() {
            return Ok(());
        }

        // ---- Section nodes ---------------------------------------------
        //
        // Derive stable section ids from (minio_key, index) so re-indexing
        // the same document hits the same Section nodes under MERGE
        // instead of creating duplicates.
        let doc_namespace = synthetic_doc_namespace(minio_object_key);
        let section_rows: Vec<NodeRow> = chunk_headings
            .iter()
            .enumerate()
            .map(|(i, heading)| {
                let short = heading_preview(heading);
                NodeRow {
                    id: Uuid::new_v5(&doc_namespace, format!("section-{i}").as_bytes())
                        .to_string(),
                    run_id: run_id.clone(),
                    props: json!({
                        "doc_id": minio_object_key,
                        "heading": short,
                        "title": short,
                        "section_order": i as i64,
                        "first_chunk_index": i as i64,
                        "page_start": 1,
                        "is_preamble": false,
                    }),
                }
            })
            .collect();
        upsert_nodes_batch(&self.graph, LABEL_SECTION, &section_rows).await?;

        // ---- HAS_SECTION edges -----------------------------------------
        let has_section_rows: Vec<EdgeRow> = section_rows
            .iter()
            .map(|s| EdgeRow {
                src: minio_object_key.to_owned(),
                dst: s.id.clone(),
                run_id: run_id.clone(),
            })
            .collect();
        upsert_edges_batch(&self.graph, REL_HAS_SECTION, &has_section_rows).await?;

        tracing::info!(
            user_id = %user_id,
            minio_object_key = %minio_object_key,
            n_sections = section_rows.len(),
            "minimal structure upserted"
        );
        Ok(())
    }
}

/// Build the UUIDv5 namespace for synthetic Section ids derived from a
/// MinIO object key. Uses the `URL` namespace as the anchor — the key
/// is already a URL-safe path.
fn synthetic_doc_namespace(minio_object_key: &str) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_URL, minio_object_key.as_bytes())
}

/// First line (or up to 80 characters) of chunk text, trimmed, used as a
/// human-readable Section heading.
fn heading_preview(text: &str) -> String {
    const MAX: usize = 80;
    let first_line = text.lines().next().unwrap_or(text).trim();
    if first_line.chars().count() <= MAX {
        return first_line.to_owned();
    }
    // Truncate on a char boundary near MAX. String::chars().take() is
    // safe because `String::from_iter(chars)` preserves UTF-8.
    let trimmed: String = first_line.chars().take(MAX).collect();
    format!("{trimmed}…")
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Derive a deterministic page node ID from document UUID + page number.
///
/// Uses a namespace so page IDs from different documents never collide.
fn page_id(document_id: Uuid, page_no: i64) -> String {
    // UUIDv5 in the document's own namespace, keyed on page number string.
    Uuid::new_v5(&document_id, page_no.to_string().as_bytes()).to_string()
}

/// Stable id for the synthetic preamble section of a document. Used when
/// the doc has chunks not attached to any heading (introduction, abstract,
/// front matter). Idempotent across re-ingest.
pub fn synthetic_preamble_section_id(doc_id: &str) -> String {
    format!("preamble:{doc_id}")
}

/// Collect unique `Section` `NodeRow`s from the chunk slice.
///
/// Each distinct `section_id` in the chunk list becomes one row. Chunks
/// without a section (preamble chunks — introduction, abstract, front matter)
/// are attached to a synthetic preamble section instead of being skipped.
/// The preamble section id is `preamble:<doc_id>`, which is stable across
/// re-ingest so the MERGE is idempotent.
///
/// New Section properties written on creation:
/// - `first_chunk_index` — position of the first chunk seen for this section
/// - `page_start` — `page_start` of that first chunk
/// - `section_order` — same as `first_chunk_index` (monotonic document order)
/// - `is_preamble` — `true` for the synthetic preamble section
fn collect_section_rows(chunks: &[StructuralChunk], run_id: &str, doc_id: &str) -> Vec<NodeRow> {
    use std::collections::HashMap;

    // Track: section_id -> (NodeRow, first_chunk_index, page_start).
    // We use a Vec to preserve insertion order (= document order).
    let mut order: Vec<String> = Vec::new();
    let mut seen: HashMap<String, NodeRow> = HashMap::new();

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let (id_str, is_preamble) = match chunk.section_id {
            Some(sid) => (sid.to_string(), false),
            None => (synthetic_preamble_section_id(doc_id), true),
        };

        seen.entry(id_str.clone()).or_insert_with(|| {
            order.push(id_str.clone());
            let first_chunk_index = chunk_index as i64;
            let page_start = chunk.page_start;
            let section_order = if is_preamble {
                -1i64
            } else {
                first_chunk_index
            };

            let (heading, depth) = if is_preamble {
                (Some("(preamble)".to_owned()), Some(0i64))
            } else {
                (chunk.section_title.clone(), chunk.heading_level)
            };

            let path_json = Value::Array(
                chunk
                    .heading_path
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            );

            NodeRow {
                id: id_str,
                run_id: run_id.to_owned(),
                props: json!({
                    "title":             heading,
                    "heading_path":      path_json,
                    "depth":             depth,
                    "first_chunk_index": first_chunk_index,
                    "page_start":        page_start,
                    "section_order":     section_order,
                    "is_preamble":       is_preamble,
                }),
            }
        });
    }

    // Return in document order (insertion order into `order`).
    order
        .into_iter()
        .filter_map(|id| seen.remove(&id))
        .collect()
}

/// Collect `PARENT_OF` edges from the heading path hierarchy.
///
/// For each chunk that has a section with `heading_level >= 2`, we look for
/// another chunk whose section at depth `level - 1` is an ancestor. Since
/// the parent section UUID must be deterministically derivable from the
/// heading path, we use a UUIDv5 of the parent section title chain.
///
/// This implementation builds a map of `(heading_path[0..depth-1] joined)`
/// → section_id for all sections, then emits `PARENT_OF` edges between
/// adjacent depth levels.
fn collect_parent_of_edges(chunks: &[StructuralChunk], run_id: &str) -> Vec<EdgeRow> {
    use std::collections::HashMap;

    // Map: heading_path (joined) -> section_id, for depth matching.
    let mut path_to_id: HashMap<String, String> = HashMap::new();
    for chunk in chunks {
        let Some(sid) = chunk.section_id else {
            continue;
        };
        if chunk.heading_path.is_empty() {
            continue;
        }
        let key = chunk.heading_path.join("\x00");
        path_to_id.entry(key).or_insert_with(|| sid.to_string());
    }

    let mut edges: Vec<EdgeRow> = Vec::new();
    // Emit parent → child for each section that has a parent.
    let mut seen_pairs: std::collections::HashSet<(String, String)> = Default::default();
    for chunk in chunks {
        let Some(sid) = chunk.section_id else {
            continue;
        };
        if chunk.heading_path.len() < 2 {
            continue;
        }
        // Parent path is heading_path[0..len-1].
        let parent_key = chunk.heading_path[..chunk.heading_path.len() - 1].join("\x00");
        if let Some(parent_id) = path_to_id.get(&parent_key) {
            let pair = (parent_id.clone(), sid.to_string());
            if seen_pairs.insert(pair.clone()) {
                edges.push(EdgeRow {
                    src: pair.0,
                    dst: pair.1,
                    run_id: run_id.to_owned(),
                });
            }
        }
    }
    edges
}

/// Update structural properties on pre-existing `Chunk` nodes.
///
/// These nodes already exist (created by the LLM KG path). We SET the
/// new structural fields without altering `ingest_run_id` — consistent
/// with the additive-only invariant.
async fn update_chunk_structural_props(
    graph: &Graph,
    chunks: &[StructuralChunk],
) -> Result<(), StoreError> {
    // Build rows: each row maps chunk_id -> structural props.
    // We UNWIND and SET (not MERGE+ON CREATE) because we are enriching
    // an existing node, not creating a new one.
    const CYPHER: &str = "\
        WITH $rows AS rows \
        UNWIND rows AS row \
        MATCH (c:Chunk {id: row.id}) \
        SET c.page_start    = row.page_start, \
            c.page_end      = row.page_end, \
            c.has_table     = row.has_table, \
            c.has_list      = row.has_list, \
            c.has_code      = row.has_code, \
            c.has_formula   = row.has_formula, \
            c.has_picture_ref = row.has_picture_ref, \
            c.has_footnote  = row.has_footnote, \
            c.has_caption   = row.has_caption";

    let rows: Vec<Value> = chunks
        .iter()
        .map(|c| {
            json!({
                "id":             c.chunk_id.to_string(),
                "page_start":     c.page_start,
                "page_end":       c.page_end,
                "has_table":      c.has_table,
                "has_list":       c.has_list,
                "has_code":       c.has_code,
                "has_formula":    c.has_formula,
                "has_picture_ref":c.has_picture_ref,
                "has_footnote":   c.has_footnote,
                "has_caption":    c.has_caption,
            })
        })
        .collect();

    for batch in rows.chunks(BATCH_SIZE) {
        let batch_val = Value::Array(batch.to_vec());
        let bolt_batch = neo4rs::BoltType::try_from(batch_val)
            .map_err(|e| StoreError::Serialization(format!("chunk props batch serialize: {e}")))?;
        let mut txn = graph.start_txn().await.map_err(StoreError::from)?;
        txn.run(query(CYPHER).param("rows", bolt_batch))
            .await
            .map_err(StoreError::from)?;
        txn.commit().await.map_err(StoreError::from)?;
    }
    Ok(())
}

/// Upsert the `Document` node with its `user_id` property.
///
/// Uses `MERGE … ON CREATE SET` so that both `user_id` and `ingest_run_id`
/// are written only when the node is first created. Re-ingesting the same
/// `doc_id` leaves the original `user_id` intact (idempotent per the
/// additive-only invariant).
///
/// Extra payload properties (sha256, title, etc.) are merged separately via
/// `ON CREATE SET n += props` so they follow the same one-shot pattern.
async fn upsert_document_node(
    graph: &Graph,
    doc_id: &str,
    run_id: &str,
    user_id: &str,
    props: serde_json::Value,
) -> Result<(), StoreError> {
    const CYPHER: &str = "\
        MERGE (n:Document {id: $doc_id}) \
        ON CREATE SET \
            n += $props, \
            n.user_id = $user_id, \
            n.ingest_run_id = $run_id";

    let bolt_props = neo4rs::BoltType::try_from(props)
        .map_err(|e| StoreError::Serialization(format!("Document props serialize: {e}")))?;

    let mut txn = graph.start_txn().await.map_err(StoreError::from)?;
    txn.run(
        query(CYPHER)
            .param("doc_id", doc_id)
            .param("props", bolt_props)
            .param("user_id", user_id)
            .param("run_id", run_id),
    )
    .await
    .map_err(StoreError::from)?;
    txn.commit().await.map_err(StoreError::from)?;
    Ok(())
}

/// Upsert a batch of nodes of a single label.
///
/// Uses `MERGE (n:Label {id: row.id}) ON CREATE SET n += row.props, n.ingest_run_id = row.run_id`
/// which means `ingest_run_id` is only written on node creation (E5 invariant).
///
/// The label string is a hard-coded ASCII identifier — safe to interpolate
/// into the Cypher template. It never originates from user/sidecar input.
async fn upsert_nodes_batch(
    graph: &Graph,
    label: &'static str,
    rows: &[NodeRow],
) -> Result<(), StoreError> {
    if rows.is_empty() {
        return Ok(());
    }

    // The label is a compile-time constant string of ASCII alphanumeric chars.
    // Format! is safe here: it interpolates a hard-coded constant, not runtime input.
    let cypher = format!(
        "WITH $rows AS rows \
         UNWIND rows AS row \
         MERGE (n:{label} {{id: row.id}}) \
         ON CREATE SET n += row.props, n.ingest_run_id = row.run_id"
    );

    let row_values: Vec<Value> = rows
        .iter()
        .map(|r| {
            serde_json::to_value(r).map_err(|e| StoreError::Serialization(format!("NodeRow: {e}")))
        })
        .collect::<Result<_, _>>()?;

    for batch in row_values.chunks(BATCH_SIZE) {
        let batch_val = Value::Array(batch.to_vec());
        let bolt = neo4rs::BoltType::try_from(batch_val)
            .map_err(|e| StoreError::Serialization(format!("{label} batch serialize: {e}")))?;
        let mut txn = graph.start_txn().await.map_err(StoreError::from)?;
        txn.run(query(&cypher).param("rows", bolt))
            .await
            .map_err(StoreError::from)?;
        txn.commit().await.map_err(StoreError::from)?;
    }
    Ok(())
}

/// Upsert a batch of edges of a single relationship type, with batch count verification.
///
/// After each batch the function runs a verification query (E3) to confirm
/// that every source and target node existed and the MERGE completed.  If
/// `created_count < batch.len()`, `StoreError::IncompleteBatch` is returned
/// immediately — the caller must rollback.
///
/// The `rel_type` string is always a hard-coded `REL_*` constant — never
/// runtime input. Format! interpolation is safe because all `REL_*` constants
/// are pure ASCII alphanumeric + underscore.
async fn upsert_edges_batch(
    graph: &Graph,
    rel_type: &'static str,
    rows: &[EdgeRow],
) -> Result<(), StoreError> {
    if rows.is_empty() {
        return Ok(());
    }

    // E3: verification query — counts created/found relationships.
    // `MERGE` on an existing edge is a no-op but still "returns" the rel;
    // the count reflects how many (a,b) pairs were matched, not how many
    // new rels were created. A count < batch size means some nodes were missing.
    let verify_cypher = format!(
        "WITH $rows AS rows \
         UNWIND rows AS row \
         MATCH (a {{id: row.src}}) \
         MATCH (b {{id: row.dst}}) \
         MERGE (a)-[r:{rel_type}]->(b) \
         ON CREATE SET r.ingest_run_id = row.run_id \
         RETURN count(r) AS created"
    );

    let row_values: Vec<Value> = rows
        .iter()
        .map(|r| {
            serde_json::to_value(r).map_err(|e| StoreError::Serialization(format!("EdgeRow: {e}")))
        })
        .collect::<Result<_, _>>()?;

    for batch in row_values.chunks(BATCH_SIZE) {
        let expected = batch.len();
        let batch_val = Value::Array(batch.to_vec());
        let bolt = neo4rs::BoltType::try_from(batch_val).map_err(|e| {
            StoreError::Serialization(format!("{rel_type} edge batch serialize: {e}"))
        })?;

        // Use execute() here (not run()) so we can read the RETURN count.
        let mut result = graph
            .execute(query(&verify_cypher).param("rows", bolt))
            .await
            .map_err(StoreError::from)?;

        let created: i64 = match result.next().await.map_err(StoreError::from)? {
            Some(row) => row.get("created").map_err(|e| {
                StoreError::QueryFailed(format!("{rel_type} count deserialize: {e}"))
            })?,
            None => 0,
        };

        if created < expected as i64 {
            return Err(StoreError::IncompleteBatch { expected, created });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preamble_section_id_is_stable_per_doc() {
        let id1 = synthetic_preamble_section_id("doc-abc");
        let id2 = synthetic_preamble_section_id("doc-abc");
        assert_eq!(id1, id2);
        assert_eq!(id1, "preamble:doc-abc");
        assert_ne!(id1, synthetic_preamble_section_id("doc-xyz"));
    }

    #[test]
    fn collect_section_rows_preamble_chunk_gets_synthetic_section() {
        let chunk = StructuralChunk {
            chunk_id: uuid::Uuid::new_v4(),
            page_start: 1,
            page_end: 1,
            section_id: None, // preamble — no heading
            section_title: None,
            heading_level: None,
            heading_path: vec![],
            has_table: false,
            has_list: false,
            has_code: false,
            has_formula: false,
            has_picture_ref: false,
            has_footnote: false,
            has_caption: false,
            table_refs: vec![],
            picture_refs: vec![],
            prev_chunk_id: None,
        };

        let rows = collect_section_rows(&[chunk], "run-1", "doc-abc");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.id, "preamble:doc-abc");
        assert_eq!(row.props["is_preamble"], serde_json::json!(true));
        assert_eq!(row.props["title"], serde_json::json!("(preamble)"));
        assert_eq!(row.props["depth"], serde_json::json!(0));
        assert_eq!(row.props["section_order"], serde_json::json!(-1));
        assert_eq!(row.props["first_chunk_index"], serde_json::json!(0));
        assert_eq!(row.props["page_start"], serde_json::json!(1));
    }

    #[test]
    fn collect_section_rows_non_preamble_gets_ordering_props() {
        let sid = uuid::Uuid::new_v4();
        let chunk = StructuralChunk {
            chunk_id: uuid::Uuid::new_v4(),
            page_start: 3,
            page_end: 4,
            section_id: Some(sid),
            section_title: Some("Introduction".to_owned()),
            heading_level: Some(1),
            heading_path: vec!["Introduction".to_owned()],
            has_table: false,
            has_list: false,
            has_code: false,
            has_formula: false,
            has_picture_ref: false,
            has_footnote: false,
            has_caption: false,
            table_refs: vec![],
            picture_refs: vec![],
            prev_chunk_id: None,
        };

        let rows = collect_section_rows(&[chunk], "run-1", "doc-xyz");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.id, sid.to_string());
        assert_eq!(row.props["is_preamble"], serde_json::json!(false));
        assert_eq!(row.props["title"], serde_json::json!("Introduction"));
        assert_eq!(row.props["depth"], serde_json::json!(1));
        assert_eq!(row.props["section_order"], serde_json::json!(0));
        assert_eq!(row.props["first_chunk_index"], serde_json::json!(0));
        assert_eq!(row.props["page_start"], serde_json::json!(3));
    }
}
