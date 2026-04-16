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
///
/// Fields are private to enforce the length invariant at construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseVector {
    /// Create a new `SparseVector`, validating that indices and values
    /// have the same length.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self, agent_rs_core::StoreError> {
        if indices.len() != values.len() {
            return Err(agent_rs_core::StoreError::InvalidData(format!(
                "SparseVector indices ({}) and values ({}) must have same length",
                indices.len(),
                values.len()
            )));
        }
        Ok(Self { indices, values })
    }

    /// Accessor for indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Accessor for values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// A point to upsert into the vector store.
///
/// Supports dense-only and hybrid (dense + server-side BM25) ingestion.
/// When `text` is non-empty, the Qdrant store uses server-side
/// `qdrant/bm25` inference for the sparse named vector (`"sparse"`),
/// eliminating the need for a local SPLADE/fastembed sparse model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: String,
    pub embedding: Embedding,
    /// Raw chunk text used for server-side BM25 inference.
    ///
    /// When non-empty, the store upserts a `Document { text, model: "qdrant/bm25" }`
    /// as the sparse named vector so Qdrant handles tokenisation and weighting.
    /// When empty, the sparse named vector is omitted (dense-only upsert).
    pub text: String,
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

// ---- Filter types ----

/// Store-layer filter specification.
///
/// Abstracts over the concrete filter syntax of each backend
/// (Qdrant payload filters, Neo4j WHERE clauses, SQL WHERE clauses)
/// so that callers like [`crate::postgres::documents`] and the
/// [`IngestCoordinator`] can express filters without depending on
/// backend-specific types.
///
/// # Examples
///
/// ```rust,no_run
/// use agent_rs_stores::types::FilterSpec;
///
/// // Delete all Qdrant points for a failed ingest run:
/// let f = FilterSpec::eq("ingest_run_id", "some-uuid-string");
/// ```
#[derive(Debug, Clone)]
pub enum FilterSpec {
    /// Exact-match filter: the named payload field must equal the given string.
    ///
    /// Maps to a Qdrant `Condition::matches(field, value)` in the `must` list.
    Eq { field: String, value: String },
    /// All sub-filters must match (logical AND).
    And(Vec<FilterSpec>),
}

impl FilterSpec {
    /// Construct an exact-string-equality filter.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use agent_rs_stores::types::FilterSpec;
    /// let f = FilterSpec::eq("ingest_run_id", "abc-123");
    /// ```
    pub fn eq(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Eq {
            field: field.into(),
            value: value.into(),
        }
    }
}

// ---- Graph Store Types ----

/// Validated node label -- only allows ASCII alphanumeric + underscore.
///
/// This is safe to interpolate into Cypher queries because labels
/// cannot be parameterized in Cypher. The ASCII-only restriction
/// prevents Unicode injection attacks.
///
/// Deserialization goes through `TryFrom<String>` to enforce validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct NodeLabel(String);

impl NodeLabel {
    /// Create a new `NodeLabel`. Returns `Err` if the label is empty
    /// or contains characters other than ASCII alphanumeric and underscore.
    pub fn new(label: &str) -> Result<Self, InvalidLabelError> {
        if label.is_empty() {
            return Err(InvalidLabelError::Empty);
        }
        if !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
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
///
/// Deserialization goes through `TryFrom<String>` to enforce validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String")]
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
    #[error(
        "label/type contains invalid characters (only ASCII alphanumeric + underscore allowed): {0}"
    )]
    InvalidChars(String),
}

impl TryFrom<String> for NodeLabel {
    type Error = InvalidLabelError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(&value)
    }
}

impl TryFrom<String> for RelationType {
    type Error = InvalidLabelError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(&value)
    }
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
