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
