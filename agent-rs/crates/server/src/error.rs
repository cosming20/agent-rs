//! Server-layer error types.
//!
//! Maps domain errors to gRPC [`tonic::Status`] codes so callers receive
//! well-formed error responses rather than opaque INTERNAL failures.

use agent_rs_core::errors::{AgentError, ConfigError};
use tonic::Status;

/// Errors that can occur in the gRPC server layer.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Address parse failure during server startup.
    #[error("invalid bind address '{addr}': {reason}")]
    InvalidAddress { addr: String, reason: String },

    /// Server transport failure.
    #[error("server transport error: {0}")]
    Transport(String),

    /// Configuration error propagated from core.
    #[error("config: {0}")]
    Config(#[from] ConfigError),
}

/// Convert an [`AgentError`] into a gRPC [`Status`].
///
/// Each variant maps to the most appropriate gRPC status code.
pub(crate) fn agent_error_to_status(err: AgentError) -> Status {
    use AgentError::*;
    match err {
        MaxRetriesExceeded { .. } => Status::resource_exhausted(err.to_string()),
        Retrieval(ref inner) => {
            tracing::warn!(error = %inner, "retrieval error during query");
            Status::unavailable(err.to_string())
        }
        Store(ref inner) => {
            tracing::error!(error = %inner, "store error during query");
            Status::internal(err.to_string())
        }
        Orchestrator(ref msg) => {
            tracing::error!(error = %msg, "orchestrator error");
            Status::internal(err.to_string())
        }
        Verification(ref msg) => {
            tracing::warn!(error = %msg, "verification error");
            Status::internal(err.to_string())
        }
        Synthesis(ref msg) => {
            tracing::error!(error = %msg, "synthesis error");
            Status::internal(err.to_string())
        }
        MemoryStore(ref msg) => {
            tracing::error!(error = %msg, "memory store error");
            Status::internal(err.to_string())
        }
    }
}
