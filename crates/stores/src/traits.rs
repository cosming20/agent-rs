//! Trait definitions for storage backends.
//!
//! Every cross-crate boundary is a trait. Concrete implementations
//! (Qdrant, Neo4j, Postgres) implement these traits in their
//! respective submodules.

use agent_rs_core::{CollectionName, Embedding, StoreError, UserId};
use async_trait::async_trait;
use uuid::Uuid;

use crate::types::{
    CollectionConfig, EntityMatch, EvidenceItem, FilterSpec, GraphEdge, GraphNode, GraphPath,
    Memory, NodeLabel, PointPayload, QueryRun, RelationType, SearchResult, VectorPoint,
};

// ---- Vector Store ----

/// Abstraction over a vector similarity search backend.
///
/// Supports both dense-only and hybrid (dense + sparse) search modes.
/// Named vectors are used to distinguish dense and sparse embeddings
/// within the same collection.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Create a collection with the given configuration.
    ///
    /// Supports named vectors (dense + optionally sparse) for hybrid search.
    async fn create_collection(
        &self,
        name: &CollectionName,
        config: &CollectionConfig,
    ) -> Result<(), StoreError>;

    /// Upsert vector points into a collection.
    ///
    /// Points may include both dense and sparse vectors for hybrid ingestion.
    async fn upsert(
        &self,
        collection: &CollectionName,
        points: Vec<VectorPoint>,
    ) -> Result<(), StoreError>;

    /// Delete points by their string IDs.
    async fn delete(&self, collection: &CollectionName, ids: Vec<String>)
    -> Result<(), StoreError>;

    /// Delete all points matching a filter from the given collection.
    ///
    /// Returns the count of points deleted by the operation.
    ///
    /// Used by the `IngestCoordinator` for rollback-by-run-id: all Qdrant
    /// points tagged `ingest_run_id = <uuid>` are purged in one call without
    /// having to enumerate their IDs first.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the backend call fails.
    async fn delete_by_filter(
        &self,
        collection: &CollectionName,
        filter: FilterSpec,
    ) -> Result<u64, StoreError>;

    /// Check if a collection exists.
    async fn collection_exists(&self, name: &CollectionName) -> Result<bool, StoreError>;

    /// Dense vector similarity search scoped to a specific user.
    ///
    /// Implementations MUST apply a mandatory `user_id` payload filter so that
    /// no cross-tenant data can be returned. This is the only permitted
    /// read path for chunk data — unscoped `search` / `search_hybrid` /
    /// `search_with_filter` / `get_by_ids` are deliberately absent from this
    /// trait, making accidental bypass a compile error.
    ///
    /// `extra_filter_json` conditions are merged with the user scope filter
    /// when provided (same `{"must": [...]}` schema).
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on backend failure.
    async fn search_for_user(
        &self,
        collection: &CollectionName,
        user_id: &UserId,
        query: &Embedding,
        limit: usize,
        extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError>;

    /// Sparse BM25 keyword search scoped to a specific user.
    ///
    /// Dispatches a server-side BM25 inference query — the backend
    /// tokenises `query_text`, weighs terms against corpus statistics, and
    /// scores points via their stored sparse named vector. No client-side
    /// embedding is performed.
    ///
    /// # Graceful degradation
    ///
    /// Collections created without a sparse named vector (dense-only)
    /// MUST return `Ok(Vec::new())` rather than an error. This lets hybrid
    /// retrieval paths call `keyword_search_for_user` unconditionally and
    /// lets RRF fusion treat a missing lexical signal as a no-op rather
    /// than a hard failure. The same contract applies when `query_text`
    /// trims to empty — return an empty vector, never error.
    ///
    /// # Scope
    ///
    /// Implementations MUST apply a mandatory `user_id` payload filter so
    /// that no cross-tenant data can be returned — identical invariant to
    /// [`VectorStore::search_for_user`].
    ///
    /// `extra_filter_json` conditions are merged with the user scope
    /// filter when provided (same `{"must": [...]}` schema).
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::InvalidData`] when `user_id` is empty —
    /// [`UserId::from_string`](agent_rs_core::UserId::from_string)
    /// bypasses validation, so every trait impl MUST reject an empty
    /// newtype at the API boundary rather than rely on `UserId`'s
    /// construction-time invariants. Returns [`StoreError`] on backend
    /// failure *other than* a missing sparse vector configuration.
    async fn keyword_search_for_user(
        &self,
        collection: &CollectionName,
        user_id: &UserId,
        query_text: &str,
        limit: usize,
        extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError>;

    /// Fetch payload-only projections for a list of chunk IDs, scoped to a user.
    ///
    /// Implementations MUST drop any returned point whose payload `user_id`
    /// does not match the requested `user_id`. IDs that do not exist are
    /// silently dropped. This is the only permitted payload-fetch path —
    /// the unscoped `get_by_ids` is absent from this trait.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::InvalidData`] when `user_id` is empty and
    /// [`StoreError`] on backend failure. Empty user scope is rejected
    /// at the API boundary because it would bypass the mandatory
    /// defence-in-depth payload filter.
    async fn retrieve_chunks_for_user(
        &self,
        collection: &CollectionName,
        user_id: &UserId,
        ids: &[String],
    ) -> Result<Vec<PointPayload>, StoreError>;
}

// ---- Graph Store ----

/// Abstraction over a graph database backend.
///
/// Node labels and relationship types are validated newtypes that are
/// safe to interpolate into Cypher queries.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Add a single node to the graph.
    async fn add_node(&self, node: &GraphNode) -> Result<(), StoreError>;

    /// Add multiple nodes in a batch.
    async fn add_nodes(&self, nodes: &[GraphNode]) -> Result<(), StoreError>;

    /// Add a single edge between two nodes.
    async fn add_edge(&self, edge: &GraphEdge) -> Result<(), StoreError>;

    /// Add multiple edges in a batch.
    async fn add_edges(&self, edges: &[GraphEdge]) -> Result<(), StoreError>;

    /// Get a node by its ID, returning `None` if not found.
    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, StoreError>;

    /// Get neighbors of a node up to a given depth.
    async fn get_neighbors(
        &self,
        node_id: &str,
        depth: usize,
    ) -> Result<Vec<GraphNode>, StoreError>;

    /// Get neighbors of a node up to `depth` hops, restricted to the given
    /// labels.
    ///
    /// An empty `labels` slice preserves the behaviour of [`get_neighbors`]
    /// (no filter). Callers supply a slice so multi-label queries stay
    /// expressible without allocation on the common single-label path.
    ///
    /// The default implementation delegates to [`get_neighbors`] and filters
    /// client-side; backends that can push the predicate into the query
    /// language (e.g. Neo4j via `neighbor:Label` in Cypher) should override
    /// this method for efficiency.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on backend failure.
    async fn get_neighbors_of_labels(
        &self,
        node_id: &str,
        depth: usize,
        labels: &[NodeLabel],
    ) -> Result<Vec<GraphNode>, StoreError> {
        let all = self.get_neighbors(node_id, depth).await?;
        if labels.is_empty() {
            return Ok(all);
        }
        let wanted: std::collections::HashSet<&str> =
            labels.iter().map(NodeLabel::as_str).collect();
        Ok(all
            .into_iter()
            .filter(|n| wanted.contains(n.label.as_str()))
            .collect())
    }

    /// Traverse the graph from a starting node, filtering by relationship types.
    async fn traverse(
        &self,
        start_id: &str,
        relation_types: &[RelationType],
        max_depth: usize,
    ) -> Result<Vec<GraphPath>, StoreError>;

    /// Execute a raw Cypher query, returning nodes.
    async fn query_nodes(&self, cypher: &str) -> Result<Vec<GraphNode>, StoreError>;

    /// Execute a raw Cypher query with named string parameters and return each
    /// row as a map of column name → [`serde_json::Value`].
    ///
    /// Unlike [`query_nodes`], this method does not assume any particular column
    /// alias or that results are graph nodes.  It is suited for queries that
    /// return scalar projections such as
    /// `RETURN n.id AS id, n.chunk_text AS text`.
    ///
    /// The caller supplies `columns` — the list of column aliases expected in
    /// the `RETURN` clause.  Columns that cannot be deserialised as JSON are
    /// silently skipped per row.  This is required because `neo4rs::Row` does
    /// not expose a `keys()` iterator; column names must be known at call time.
    async fn query_raw(
        &self,
        cypher: &str,
        params: std::collections::HashMap<String, String>,
        columns: &[&str],
    ) -> Result<Vec<std::collections::HashMap<String, serde_json::Value>>, StoreError>;

    /// Locate knowledge-graph nodes whose name-like properties match a
    /// free-form entity mention.
    ///
    /// Returns up to `limit` matches ranked by backend-defined relevance,
    /// restricted to nodes whose `user_id` property equals `user_id`.
    ///
    /// # Property coverage
    ///
    /// Implementations MUST consult every property that a node in any
    /// indexed label might use for its human-readable name — typically
    /// `name`, `title`, and `heading` — and return the first non-empty
    /// one as [`EntityMatch::display_name`] via a `coalesce`-style
    /// selection. Requiring only `name` was the original
    /// `EntityLinker` bug: the ingester writes structural nodes with
    /// `title` / `heading`, so a name-only lookup misses all of them.
    ///
    /// # Scalability
    ///
    /// Implementations MUST back this by a full-text index (Neo4j
    /// fulltext / Lucene equivalent) rather than a linear
    /// `WHERE CONTAINS` scan. For small corpora both produce the same
    /// result; for any production corpus the linear path becomes the
    /// dominant retrieval latency.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::InvalidData`] when `user_id` is empty and
    /// [`StoreError`] on backend failure. An empty `mention` or a
    /// backend that lacks the fulltext index MUST return
    /// `Ok(Vec::new())` — never an error — so the retrieval router's
    /// entity-linking step can fail open. Empty `user_id` is the single
    /// exception because it is an API-boundary programming error, not a
    /// degraded backend state.
    async fn find_entities_by_mention(
        &self,
        user_id: &UserId,
        mention: &str,
        limit: usize,
    ) -> Result<Vec<EntityMatch>, StoreError>;

    /// Delete a node and its edges by ID.
    async fn delete_node(&self, id: &str) -> Result<(), StoreError>;

    /// Delete all graph nodes (and their edges) whose `ingest_run_id` property
    /// matches `ingest_run_id`.
    ///
    /// Used by `IngestCoordinator::cleanup_partial` to roll back a partial
    /// ingest run without touching nodes created by other runs.
    ///
    /// Returns the count of deleted nodes.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the backend call fails.
    async fn rollback_by_run_id(&self, ingest_run_id: Uuid) -> Result<u64, StoreError>;

    /// Ensure the structural-layer constraints exist (idempotent).
    ///
    /// Safe to call on every pipeline startup. Uses `IF NOT EXISTS`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the backend call fails.
    async fn ensure_structural_constraints(&self) -> Result<(), StoreError>;

    /// Upsert the full structural layer for one document (Document, Page,
    /// Section, Table, Picture nodes + ON_PAGE / IN_SECTION / REFERENCES_TABLE
    /// / REFERENCES_PICTURE / FOLLOWS / PARENT_OF / HAS_PAGE / HAS_SECTION /
    /// HAS_TABLE / HAS_PICTURE / CONTAINS_CHUNK edges), all tagged with
    /// `ingest_run_id`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on any backend failure; caller should call
    /// `rollback_by_run_id(ingest_run_id)` to clean up partial state.
    /// # Errors
    ///
    /// Returns [`StoreError`] on any backend failure; caller should call
    /// `rollback_by_run_id(ingest_run_id)` to clean up partial state.
    ///
    /// `user_id` is authoritative for the `Document` node and is written via
    /// `ON CREATE SET` — re-ingesting the same `document_id` does not alter
    /// the original owner.
    async fn upsert_document_structure(
        &self,
        header: &crate::neo4j::structural::DocumentHeaderSnapshot,
        chunks: &[crate::neo4j::structural::StructuralChunk],
        tables: &[crate::neo4j::structural::StructuralTable],
        pictures: &[crate::neo4j::structural::StructuralPicture],
    ) -> Result<(), StoreError>;
}

// ---- State Store ----

/// Abstraction over persistent state storage (memories, runs, evidence).
///
/// Chat history is NOT here — in the stateless architecture, history is
/// replayed inline from the client via `AskRequest.history` and never
/// persisted on the agent side. Only user-level state (long-term memories)
/// and analytics/telemetry (query runs + evidence) live in this store.
///
/// Backed by a relational database (Postgres via Diesel).
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Get all memories for a user.
    async fn get_memories(&self, user_id: &UserId) -> Result<Vec<Memory>, StoreError>;

    /// Store a memory for a user.
    async fn store_memory(&self, memory: &Memory) -> Result<(), StoreError>;

    /// Persist a query run.
    async fn persist_run(&self, run: &QueryRun) -> Result<(), StoreError>;

    /// Get a query run by ID.
    async fn get_run(&self, id: &uuid::Uuid) -> Result<Option<QueryRun>, StoreError>;

    /// Store a single evidence item linked to a query run.
    async fn store_evidence(&self, item: &EvidenceItem) -> Result<(), StoreError>;

    /// Store multiple evidence items for a query run.
    async fn store_evidence_batch(&self, items: &[EvidenceItem]) -> Result<(), StoreError>;

    /// Get all evidence items for a query run.
    async fn get_evidence(&self, run_id: &uuid::Uuid) -> Result<Vec<EvidenceItem>, StoreError>;
}
