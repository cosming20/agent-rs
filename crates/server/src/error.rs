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
    // Generate a correlation ID for cross-referencing client errors with
    // server logs.  The full error detail stays in structured logs only —
    // never exposed to the gRPC client.
    let correlation_id = uuid::Uuid::new_v4();

    use AgentError::*;
    match err {
        MaxRetriesExceeded { max } => {
            tracing::warn!(correlation_id = %correlation_id, max, "max retries exceeded");
            Status::resource_exhausted(format!(
                "Query exceeded maximum retry attempts. Correlation ID: {correlation_id}"
            ))
        }
        Retrieval(ref inner) => {
            tracing::warn!(correlation_id = %correlation_id, error = %inner, "retrieval error during query");
            Status::unavailable(format!(
                "Retrieval service temporarily unavailable. Correlation ID: {correlation_id}"
            ))
        }
        Store(ref inner) => {
            tracing::error!(correlation_id = %correlation_id, error = %inner, "store error during query");
            Status::internal(format!(
                "An internal error occurred. Correlation ID: {correlation_id}"
            ))
        }
        Orchestrator(ref msg) => {
            tracing::error!(correlation_id = %correlation_id, error = %msg, "orchestrator error");
            Status::internal(format!(
                "An internal error occurred. Correlation ID: {correlation_id}"
            ))
        }
        Verification(ref msg) => {
            tracing::warn!(correlation_id = %correlation_id, error = %msg, "verification error");
            Status::internal(format!(
                "An internal error occurred. Correlation ID: {correlation_id}"
            ))
        }
        Synthesis(ref msg) => {
            tracing::error!(correlation_id = %correlation_id, error = %msg, "synthesis error");
            Status::internal(format!(
                "An internal error occurred. Correlation ID: {correlation_id}"
            ))
        }
        MemoryStore(ref msg) => {
            tracing::error!(correlation_id = %correlation_id, error = %msg, "memory store error");
            Status::internal(format!(
                "An internal error occurred. Correlation ID: {correlation_id}"
            ))
        }
        InvalidUserId(ref raw) => {
            tracing::warn!(correlation_id = %correlation_id, raw = %raw, "invalid user_id in request");
            Status::invalid_argument(format!(
                "Invalid user_id. Correlation ID: {correlation_id}"
            ))
        }
    }
}
