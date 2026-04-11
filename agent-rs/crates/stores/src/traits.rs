//! Trait definitions for storage backends.
//!
//! Every cross-crate boundary is a trait. Concrete implementations
//! (Qdrant, Neo4j, Postgres) implement these traits in their
//! respective submodules.

use agent_rs_core::{CollectionName, Embedding, SessionId, StoreError, UserId};
use async_trait::async_trait;
use uuid::Uuid;

use crate::types::{
    ChatMessage, CollectionConfig, EvidenceItem, FilterSpec, GraphEdge, GraphNode, GraphPath,
    Memory, QueryRun, RelationType, SearchResult, VectorPoint,
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

    /// Dense vector similarity search.
    async fn search(
        &self,
        collection: &CollectionName,
        query: &Embedding,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StoreError>;

    /// Dense vector similarity search with optional payload filter.
    ///
    /// The `filter` is a JSON object describing Qdrant payload conditions.
    /// When `None`, behaves identically to [`search`].
    async fn search_with_filter(
        &self,
        collection: &CollectionName,
        query: &Embedding,
        limit: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError>;

    /// Hybrid search: dense vector + server-side BM25, fused via prefetch + RRF.
    ///
    /// `query_text` is sent to Qdrant as a `Document { model: "qdrant/bm25" }`
    /// for the sparse prefetch leg so Qdrant handles tokenisation server-side.
    async fn search_hybrid(
        &self,
        collection: &CollectionName,
        dense: &Embedding,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StoreError>;

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

    /// Traverse the graph from a starting node, filtering by relationship types.
    async fn traverse(
        &self,
        start_id: &str,
        relation_types: &[RelationType],
        max_depth: usize,
    ) -> Result<Vec<GraphPath>, StoreError>;

    /// Execute a raw Cypher query, returning nodes.
    async fn query_nodes(&self, cypher: &str) -> Result<Vec<GraphNode>, StoreError>;

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
    async fn upsert_document_structure(
        &self,
        header: &crate::neo4j::structural::DocumentHeaderSnapshot,
        chunks: &[crate::neo4j::structural::StructuralChunk],
        tables: &[crate::neo4j::structural::StructuralTable],
        pictures: &[crate::neo4j::structural::StructuralPicture],
    ) -> Result<(), StoreError>;
}

// ---- State Store ----

/// Abstraction over persistent state storage (chat history, memory, runs, evidence).
///
/// Backed by a relational database (Postgres via Diesel).
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Get the last N messages for a session.
    async fn get_messages(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, StoreError>;

    /// Store a chat message.
    async fn store_message(&self, message: &ChatMessage) -> Result<(), StoreError>;

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
