//! Domain newtypes and enums for the agent-rs system.
//!
//! All public API boundaries use these types instead of raw primitives.
//! Validation happens at construction time, not at use sites.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::{Uuid, uuid};

// ---- Constants ----

/// UUIDv5 namespace for deterministic document / chunk / ingest IDs.
///
/// This is the fixed OID namespace from RFC 4122 Appendix C:
/// `6ba7b812-9dad-11d1-80b4-00c04fd430c8`.
///
/// All stable IDs in the ingest pipeline are derived as:
/// `Uuid::new_v5(&NAMESPACE_OID, seed_bytes)`.
///
/// Defined once here and imported wherever deterministic IDs are needed,
/// so that any change to the namespace automatically invalidates all
/// derived IDs (compile-time discovery of all derivation sites).
pub const NAMESPACE_OID: Uuid = uuid!("6ba7b812-9dad-11d1-80b4-00c04fd430c8");

// ---- Newtypes ----

/// Unique identifier for a conversation session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    /// Create a new random session ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a user.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub String);

impl fmt::Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A confidence score in the range `[0.0, 1.0]`.
///
/// Construction via [`Confidence::new`] validates the range and returns
/// an error if the value is out of bounds. This type never silently
/// defaults to 0.0.
///
/// Deserialization goes through `TryFrom<f64>` to enforce validation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64")]
pub struct Confidence(f64);

impl Confidence {
    /// Create a new confidence score, returning an error if the value
    /// is outside `[0.0, 1.0]`.
    pub fn new(value: f64) -> Result<Self, ConfidenceError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(ConfidenceError::OutOfRange(value));
        }
        Ok(Self(value))
    }

    /// Return the inner f64 value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check whether this confidence exceeds a given threshold.
    pub fn is_above_threshold(&self, threshold: f64) -> bool {
        self.0 >= threshold
    }
}

/// Error returned when constructing a [`Confidence`] with an invalid value.
#[derive(Debug, thiserror::Error)]
pub enum ConfidenceError {
    #[error("confidence {0} out of range [0.0, 1.0]")]
    OutOfRange(f64),
}

impl TryFrom<f64> for Confidence {
    type Error = ConfidenceError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// A numeric score (e.g., search relevance, RRF rank).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Score(pub f64);

/// A dense embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Return the number of dimensions in this embedding.
    pub fn dimensions(&self) -> usize {
        self.0.len()
    }
}

/// The name of a dataset used for retrieval routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatasetName(pub String);

impl fmt::Display for DatasetName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The name of a vector collection (e.g., Qdrant collection).
///
/// Implements [`AsRef<str>`] for easy interop with APIs expecting `&str`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionName(pub String);

impl CollectionName {
    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for CollectionName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---- Enums ----

/// Retrieval strategies supported by the system.
///
/// Each variant maps to a specific search mode or combination of modes
/// used by the retrieval router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalStrategy {
    Semantic,
    Keyword,
    Hybrid,
    Graph,
    GraphHybrid,
    RagCompletion,
    Summaries,
    Temporal,
    GraphCoT,
    GraphConstrained,
}

impl RetrievalStrategy {
    /// Convert to the string representation used in the database.
    pub fn as_db_str(&self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Keyword => "keyword",
            Self::Hybrid => "hybrid",
            Self::Graph => "graph",
            Self::GraphHybrid => "graph_hybrid",
            Self::RagCompletion => "rag_completion",
            Self::Summaries => "summaries",
            Self::Temporal => "temporal",
            Self::GraphCoT => "graph_cot",
            Self::GraphConstrained => "graph_constrained",
        }
    }

    /// Parse from the database string representation.
    pub fn from_db_str(s: &str) -> Result<Self, StrategyParseError> {
        match s {
            "semantic" => Ok(Self::Semantic),
            "keyword" => Ok(Self::Keyword),
            "hybrid" => Ok(Self::Hybrid),
            "graph" => Ok(Self::Graph),
            "graph_hybrid" => Ok(Self::GraphHybrid),
            "rag_completion" => Ok(Self::RagCompletion),
            "summaries" => Ok(Self::Summaries),
            "temporal" => Ok(Self::Temporal),
            "graph_cot" => Ok(Self::GraphCoT),
            "graph_constrained" => Ok(Self::GraphConstrained),
            other => Err(StrategyParseError::Unknown(other.to_string())),
        }
    }
}

/// Error returned when parsing a [`RetrievalStrategy`] from a string.
#[derive(Debug, thiserror::Error)]
pub enum StrategyParseError {
    #[error("unknown retrieval strategy: {0}")]
    Unknown(String),
}

impl fmt::Display for RetrievalStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_db_str())
    }
}

/// Role of a message in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

// ---- Ingest state machine ----

/// State of a document in the ingestion pipeline.
///
/// Stored as TEXT in Postgres with a CHECK constraint.  Use [`IngestStatus::as_str`]
/// for serialisation and [`IngestStatus::from_str`] for deserialisation —
/// both are exhaustive and reject unknown values loudly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestStatus {
    /// Ingestion is currently running.
    InProgress,
    /// Ingestion started but failed part-way through (rollback performed).
    PartialFailure,
    /// Ingestion finished successfully.
    Complete,
}

impl IngestStatus {
    /// Return the canonical lowercase string stored in the database.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::PartialFailure => "partial_failure",
            Self::Complete => "complete",
        }
    }
}

impl fmt::Display for IngestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when parsing an [`IngestStatus`] from an unknown string.
#[derive(Debug, thiserror::Error)]
#[error("unknown ingest_status value: '{0}'")]
pub struct InvalidIngestStatus(pub String);

impl FromStr for IngestStatus {
    type Err = InvalidIngestStatus;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "in_progress" => Ok(Self::InProgress),
            "partial_failure" => Ok(Self::PartialFailure),
            "complete" => Ok(Self::Complete),
            other => Err(InvalidIngestStatus(other.to_string())),
        }
    }
}

// ---- Utilities ----

/// Truncate a string to at most `max_chars` Unicode characters, returning
/// a sub-slice that always ends on a valid char boundary.
///
/// This is safe for multi-byte UTF-8 content -- unlike `&s[..n]` which
/// panics when `n` falls inside a multi-byte character.
pub fn truncate_to_char_boundary(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let byte_idx = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..byte_idx]
}
