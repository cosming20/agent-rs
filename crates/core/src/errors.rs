//! Structured error hierarchy for the agent-rs system.
//!
//! Uses `thiserror` for all error types. Typed `#[from]` conversions
//! are used wherever the source error type is available in this crate.
//! String wrappers are used only for errors whose source types live
//! in downstream crates.

use thiserror::Error;

// ---- Top-Level Error ----

/// Root error type that unifies all subsystem errors.
#[derive(Debug, Error)]
pub enum AgentRsError {
    #[error("config: {0}")]
    Config(#[from] ConfigError),
    #[error("store: {0}")]
    Store(#[from] StoreError),
    #[error("retrieval: {0}")]
    Retrieval(#[from] RetrievalError),
    #[error("agent: {0}")]
    Agent(#[from] AgentError),
    #[error("tool: {0}")]
    Tool(#[from] ToolError),
    #[error("auth: {0}")]
    Auth(#[from] AuthError),
}

// ---- Config ----

/// Errors during configuration loading and validation.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing required config: {field}")]
    MissingField { field: String },
    #[error("invalid config value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("config load failed: {0}")]
    LoadFailed(String),
}

// ---- Store ----

/// Errors from storage backends (Qdrant, Neo4j, Postgres).
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("qdrant: {0}")]
    Qdrant(Box<qdrant_client::QdrantError>),
    #[error("neo4j: {0}")]
    Neo4j(Box<neo4rs::Error>),
    #[error("postgres: {0}")]
    Postgres(#[from] diesel::result::Error),
    #[error("connection pool: {0}")]
    ConnectionPool(String),
    #[error("connection failed to {service}: {reason}")]
    ConnectionFailed { service: String, reason: String },
    #[error("query failed: {0}")]
    QueryFailed(String),
    #[error("not found: {entity} with id {id}")]
    NotFound { entity: String, id: String },
    #[error("serialization: {0}")]
    Serialization(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("invalid state: row is in an unexpected ingest_status")]
    InvalidState,
    /// A batched Neo4j write produced fewer relationships/nodes than the
    /// batch contained, indicating that one or more source or target nodes
    /// were missing when the edge MERGE ran.
    ///
    /// The ingest coordinator must invoke `rollback_by_run_id` after
    /// receiving this error.
    #[error("incomplete batch write: expected {expected} rows, Neo4j reported {created}")]
    IncompleteBatch { expected: usize, created: i64 },

    /// Session does not exist, OR belongs to a different user.
    ///
    /// Deliberately collapses both cases into one error to prevent
    /// enumeration attacks: a caller cannot distinguish "this session_id
    /// was never issued" from "this session_id belongs to someone else".
    #[error("session {session_id} not found for user {user_id}")]
    SessionNotFound { session_id: String, user_id: String },

    /// Caller attempted to activate / read / delete a document that does
    /// not belong to the provided user.
    #[error("document {document_id} not owned by user {user_id}")]
    DocumentNotOwnedByUser {
        document_id: String,
        user_id: String,
    },

    /// Error from the MinIO / S3 client. Message is the wrapped cause.
    #[error("MinIO error: {0}")]
    Minio(String),
}

// ---- Retrieval ----

/// Errors during retrieval operations (embedding, routing, ranking).
#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("embedding failed: {reason}")]
    Embedding { reason: String },
    #[error("routing failed for strategy {strategy}: {reason}")]
    Routing { strategy: String, reason: String },
    #[error("ranking failed: {0}")]
    Ranking(String),
    #[error("search returned no results for query: {query}")]
    NoResults { query: String },
    #[error("store error during retrieval: {0}")]
    Store(#[from] StoreError),
    #[error("ingestion failed: {reason}")]
    Ingestion { reason: String },
    #[error("pdf extraction failed: {reason}")]
    PdfExtraction { reason: String },
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("docling ingest failed: {0}")]
    DoclingIngestFailed(String),
    #[error("ingest already in progress: {0}")]
    IngestInProgress(String),
    #[error("postgres commit failed: {0}")]
    PostgresCommitFailed(String),
}

// ---- Agent ----

/// Errors from the agent orchestration layer.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("orchestrator: {0}")]
    Orchestrator(String),
    #[error("verification failed: {0}")]
    Verification(String),
    #[error("synthesis: {0}")]
    Synthesis(String),
    #[error("memory store: {0}")]
    MemoryStore(String),
    /// Invalid user id — rejected by [`crate::types::UserId::try_new`].
    ///
    /// Message is verbose for logs but callers must NOT propagate it back to
    /// user-facing output; return a generic "invalid user id" at API
    /// boundaries instead.
    #[error("invalid user id: {0}")]
    InvalidUserId(String),
    #[error("max retries ({max}) exceeded")]
    MaxRetriesExceeded { max: usize },
    #[error("retrieval error: {0}")]
    Retrieval(#[from] RetrievalError),
    #[error("store error: {0}")]
    Store(#[from] StoreError),
}

// ---- Tool ----

/// Errors from tool invocations (MCP, web search, fetch).
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("MCP client: {0}")]
    McpClient(String),
    #[error("web search: {0}")]
    WebSearch(String),
    #[error("SSRF blocked: attempted access to {url}")]
    Ssrf { url: String },
    #[error("web budget exhausted: {resource} (used {used}/{max})")]
    WebBudgetExhausted {
        resource: String,
        used: usize,
        max: usize,
    },
    #[error("fetch failed for {url}: {reason}")]
    FetchFailed { url: String, reason: String },
}

// ---- Auth ----

/// Errors from authentication and authorization (OPA).
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("OPA policy denied: {reason}")]
    OpaDenied { reason: String },
    #[error("OPA unreachable: {0}")]
    OpaUnreachable(String),
}

// ---- Manual From impls for boxed error variants ----

impl From<qdrant_client::QdrantError> for StoreError {
    fn from(err: qdrant_client::QdrantError) -> Self {
        Self::Qdrant(Box::new(err))
    }
}

impl From<neo4rs::Error> for StoreError {
    fn from(err: neo4rs::Error) -> Self {
        Self::Neo4j(Box::new(err))
    }
}

#[cfg(test)]
mod multi_tenant_error_tests {
    use super::*;

    #[test]
    fn store_error_session_not_found_does_not_leak_existence() {
        // The Display impl must NOT reveal whether a session exists for a
        // different user — both "no such session" and "session belongs to
        // different user" collapse to the same message text.
        let err = StoreError::SessionNotFound {
            session_id: "sess-123".into(),
            user_id: "alice".into(),
        };
        let s = format!("{err}");
        assert!(s.contains("not found"));
        assert!(!s.to_lowercase().contains("different user"));
        assert!(!s.to_lowercase().contains("belongs to"));
    }

    #[test]
    fn store_error_document_not_owned_has_required_fields() {
        let err = StoreError::DocumentNotOwnedByUser {
            document_id: "doc-1".into(),
            user_id: "alice".into(),
        };
        let s = format!("{err}");
        assert!(s.contains("doc-1"));
    }

    #[test]
    fn store_error_minio_wraps_message() {
        let err = StoreError::Minio("bucket not found".into());
        let s = format!("{err}");
        assert!(s.contains("MinIO"));
        assert!(s.contains("bucket not found"));
    }
}
