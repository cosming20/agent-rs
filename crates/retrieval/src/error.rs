//! Retrieval-crate-local error type.
//!
//! [`RetrievalError`] mirrors the variant from `agent-rs-core` but is defined
//! here so the retrieval crate can use it without a circular dependency.
//! The core crate's `RetrievalError` is the canonical type surfaced to callers
//! outside this crate; this local type converts into it via `From`.

use thiserror::Error;

/// Errors that can occur within the retrieval crate.
#[derive(Debug, Error)]
pub enum RetrievalError {
    /// An embedding model call failed.
    #[error("embedding failed: {reason}")]
    Embedding { reason: String },

    /// A retrieval strategy could not be routed.
    #[error("routing failed for strategy {strategy}: {reason}")]
    Routing { strategy: String, reason: String },

    /// Rank fusion or evidence assembly failed.
    #[error("ranking failed: {0}")]
    Ranking(String),

    /// The search returned no results.
    #[error("search returned no results for query: {query}")]
    NoResults { query: String },

    /// A storage backend returned an error during a search operation.
    #[error("store error during retrieval: {0}")]
    Store(#[from] agent_rs_core::StoreError),

    /// An ingestion operation failed (chunking, metadata extraction, pipeline).
    #[error("ingestion failed: {reason}")]
    Ingestion { reason: String },

    /// PDF text extraction failed.
    #[error("pdf extraction failed: {reason}")]
    PdfExtraction { reason: String },

    /// Feature not yet implemented (used only during phased development).
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

impl From<RetrievalError> for agent_rs_core::errors::RetrievalError {
    fn from(e: RetrievalError) -> Self {
        match e {
            RetrievalError::Embedding { reason } => {
                agent_rs_core::errors::RetrievalError::Embedding { reason }
            }
            RetrievalError::Routing { strategy, reason } => {
                agent_rs_core::errors::RetrievalError::Routing { strategy, reason }
            }
            RetrievalError::Ranking(msg) => agent_rs_core::errors::RetrievalError::Ranking(msg),
            RetrievalError::NoResults { query } => {
                agent_rs_core::errors::RetrievalError::NoResults { query }
            }
            RetrievalError::Store(e) => agent_rs_core::errors::RetrievalError::Store(e),
            RetrievalError::Ingestion { reason } => {
                agent_rs_core::errors::RetrievalError::Ranking(format!("ingestion: {reason}"))
            }
            RetrievalError::PdfExtraction { reason } => {
                agent_rs_core::errors::RetrievalError::Ranking(format!("pdf extraction: {reason}"))
            }
            RetrievalError::NotImplemented(msg) => {
                agent_rs_core::errors::RetrievalError::Ranking(format!("not implemented: {msg}"))
            }
        }
    }
}
