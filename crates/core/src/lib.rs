//! Core crate for agent-rs: shared types, configuration, errors, and logging.
//!
//! This crate has no dependencies on other workspace crates.
//! All cross-crate contracts reference types defined here.

pub mod config;
pub mod errors;
pub mod logging;
pub mod types;

// Re-export commonly used items at crate root for ergonomic imports.
pub use config::Settings;
pub use errors::{AgentError, AgentRsError, RetrievalError, StoreError, ToolError};
pub use types::{
    CollectionName, Confidence, DatasetName, Embedding, MessageRole, RetrievalStrategy, Score,
    SessionId, UserId,
};
