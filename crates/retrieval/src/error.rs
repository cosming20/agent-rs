//! Retrieval-crate-local error type.
//!
//! [`RetrievalError`] mirrors the variant from `agent-rs-core` but is defined
//! here so the retrieval crate can use it without a circular dependency.
//! The core crate's `RetrievalError` is the canonical type surfaced to callers
//! outside this crate; this local type converts into it via `From`.

use thiserror::Error;

use crate::ingestion::types::IngestErrorCode;

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

    /// PDF text extraction failed (legacy `pdf_extract` path).
    #[error("pdf extraction failed: {reason}")]
    PdfExtraction { reason: String },

    /// Feature not yet implemented (used only during phased development).
    #[error("not implemented: {0}")]
    NotImplemented(String),

    // ---- Docling sidecar errors ----
    /// The Docling sidecar reported a structured ingest error.
    ///
    /// `code` identifies the failure class; `retryable` signals whether a
    /// retry with backoff may succeed; `triggers_legacy_fallback` indicates
    /// that the `DoclingWithLegacyFallback` backend may attempt the legacy
    /// `pdf_extract` path.
    #[error("docling ingest failed [{code:?}]: {message}")]
    DoclingIngestFailed {
        code: IngestErrorCode,
        retryable: bool,
        triggers_legacy_fallback: bool,
        message: String,
    },

    /// Another worker is already ingesting a document with the given sha256.
    ///
    /// The caller may wait and retry, or surface the conflict to the user.
    #[error("ingest already in progress for sha256 {sha256}")]
    IngestInProgress { sha256: String },

    /// Postgres `mark_complete` failed after exhausting all retries.
    ///
    /// Qdrant and Neo4j are in a consistent state (writes succeeded); only
    /// the Postgres row is inconsistent.  Operator reconciliation is required.
    #[error("postgres commit failed after retries (ingest_run_id={ingest_run_id}): {reason}")]
    PostgresCommitFailed {
        ingest_run_id: String,
        reason: String,
    },
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
            RetrievalError::DoclingIngestFailed { code, message, .. } => {
                agent_rs_core::errors::RetrievalError::Ranking(format!(
                    "docling ingest failed [{code:?}]: {message}"
                ))
            }
            RetrievalError::IngestInProgress { sha256 } => {
                agent_rs_core::errors::RetrievalError::Ranking(format!(
                    "ingest in progress for sha256={sha256}"
                ))
            }
            RetrievalError::PostgresCommitFailed {
                ingest_run_id,
                reason,
            } => agent_rs_core::errors::RetrievalError::Ranking(format!(
                "postgres commit failed (run={ingest_run_id}): {reason}"
            )),
        }
    }
}
