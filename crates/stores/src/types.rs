//! Data types specific to storage backends.
//!
//! These types are used across VectorStore, GraphStore, and StateStore
//! trait methods. They bridge domain types from `agent-rs-core` with
//! the storage-layer representations.

use agent_rs_core::{
    Confidence, Embedding, MessageRole, RetrievalStrategy, Score, SessionId, UserId,
};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---- Vector Store Types ----

/// Sparse vector for BM25/BM42 lexical search.
///
/// Indices and values must have the same length. Each index identifies
/// a vocabulary term; the value is its weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// A point to upsert into the vector store.
///
/// Supports both dense-only and dense+sparse (hybrid) ingestion.
/// When `sparse_vector` is `Some`, the point carries both a dense
/// embedding and a sparse BM25/BM42 vector for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: String,
    pub embedding: Embedding,
    pub sparse_vector: Option<SparseVector>,
    pub payload: serde_json::Value,
}

/// Configuration for creating a vector collection.
///
/// Supports named vectors: a mandatory dense vector and an optional
/// sparse vector for hybrid search.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    /// Dimension of the dense vector (e.g., 1024 for fastembed).
    pub dense_vector_size: usize,
    /// Whether to also create a named sparse vector for BM25/BM42.
    pub enable_sparse: bool,
    /// Name for the dense vector. Default: `"dense"`.
    pub dense_vector_name: String,
    /// Name for the sparse vector. Default: `"sparse"`.
    pub sparse_vector_name: String,
}

impl CollectionConfig {
    /// Create a dense-only collection config.
    pub fn dense_only(vector_size: usize) -> Self {
        Self {
            dense_vector_size: vector_size,
            enable_sparse: false,
            dense_vector_name: "dense".into(),
            sparse_vector_name: "sparse".into(),
        }
    }

    /// Create a hybrid (dense + sparse) collection config.
    pub fn hybrid(vector_size: usize) -> Self {
        Self {
            dense_vector_size: vector_size,
            enable_sparse: true,
            dense_vector_name: "dense".into(),
            sparse_vector_name: "sparse".into(),
        }
    }
}

/// A single search result returned from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: Score,
    pub payload: serde_json::Value,
}

// ---- Graph Store Types ----

/// Validated node label -- only allows ASCII alphanumeric + underscore.
///
/// This is safe to interpolate into Cypher queries because labels
/// cannot be parameterized in Cypher. The ASCII-only restriction
/// prevents Unicode injection attacks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeLabel(String);

impl NodeLabel {
    /// Create a new `NodeLabel`. Returns `Err` if the label is empty
    /// or contains characters other than ASCII alphanumeric and underscore.
    pub fn new(label: &str) -> Result<Self, InvalidLabelError> {
        if label.is_empty() {
            return Err(InvalidLabelError::Empty);
        }
        if !label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(InvalidLabelError::InvalidChars(label.to_string()));
        }
        Ok(Self(label.to_string()))
    }

    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Validated relationship type -- only allows ASCII alphanumeric + underscore.
///
/// Same safety guarantees as [`NodeLabel`] for Cypher interpolation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationType(String);

impl RelationType {
    /// Create a new `RelationType`. Returns `Err` if the type is empty
    /// or contains characters other than ASCII alphanumeric and underscore.
    pub fn new(rel_type: &str) -> Result<Self, InvalidLabelError> {
        if rel_type.is_empty() {
            return Err(InvalidLabelError::Empty);
        }
        if !rel_type
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(InvalidLabelError::InvalidChars(rel_type.to_string()));
        }
        Ok(Self(rel_type.to_string()))
    }

    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Error for invalid node labels or relationship types.
#[derive(Debug, thiserror::Error)]
pub enum InvalidLabelError {
    #[error("label/type must not be empty")]
    Empty,
    #[error("label/type contains invalid characters (only ASCII alphanumeric + underscore allowed): {0}")]
    InvalidChars(String),
}

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: NodeLabel,
    pub properties: serde_json::Value,
}

/// An edge (relationship) between two graph nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source_id: String,
    pub target_id: String,
    pub relation_type: RelationType,
    pub properties: serde_json::Value,
}

/// A path through the graph consisting of alternating nodes and edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

// ---- State Store Types ----

/// A single chat message in a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: Uuid,
    pub session_id: SessionId,
    pub role: MessageRole,
    pub content: String,
    pub created_at: NaiveDateTime,
}

/// A persistent memory entry associated with a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Uuid,
    pub user_id: UserId,
    pub content: String,
    pub created_at: NaiveDateTime,
}

/// A record of a single query execution including its result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRun {
    pub id: Uuid,
    pub session_id: SessionId,
    pub user_id: UserId,
    pub query: String,
    pub answer: String,
    pub strategy: RetrievalStrategy,
    pub confidence: Confidence,
    pub is_grounded: bool,
    pub created_at: NaiveDateTime,
}

/// A single piece of evidence supporting an answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub id: Uuid,
    pub run_id: Uuid,
    pub content: String,
    pub source: String,
    pub score: f64,
    pub created_at: NaiveDateTime,
}
