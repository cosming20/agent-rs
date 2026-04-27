//! Core crate for agent-rs: shared types, configuration, errors, and logging.
//!
//! This crate has no dependencies on other workspace crates.
//! All cross-crate contracts reference types defined here.

pub mod config;
pub mod doc_handle;
pub mod errors;
pub mod logging;
pub mod payload;
pub mod types;

// Re-export commonly used items at crate root for ergonomic imports.
pub use config::{
    DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT, DoclingConfig, IngestBackend, LatencyMode,
    RerankerConfig, RerankerProvider, Settings, temperature_for,
};
pub use doc_handle::{DocHandle, DocHandleError};
pub use errors::{AgentError, AgentRsError, RetrievalError, StoreError, ToolError};
pub use types::{
    CollectionName, Confidence, DatasetName, Embedding, IngestStatus, InvalidIngestStatus,
    MessageRole, NAMESPACE_OID, RetrievalStrategy, Score, SessionId, UserId,
    truncate_to_char_boundary,
};
