//! Tonic streaming client for the Marker sidecar gRPC service.
//!
//! [`MarkerSidecarClient`] is a thin newtype over
//! [`crate::ingestion::docling_client::DoclingSidecarClient`]. Both
//! sidecars implement the **same** `agentic_rag.v1.DoclingChunker`
//! proto service — the wire protocol, state machine, error payload, and
//! response shape are identical. Only the endpoint and the backend's
//! internal parse engine differ.
//!
//! Keeping the wrapper as its own type (rather than reusing
//! `DoclingSidecarClient` directly) buys us:
//!
//! - Type-level separation in the indexer — `AgentIndexHandler` can hold
//!   `docling: Option<Arc<DoclingSidecarClient>>` and
//!   `marker: Option<Arc<MarkerSidecarClient>>` as distinct fields, so
//!   backend routing is a compile-time check, not an endpoint-guess.
//! - A place to hang Marker-specific telemetry, warmup probes, or
//!   graceful-degradation policies in the future without touching the
//!   Docling client.
//!
//! # Usage
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use agent_rs_retrieval::ingestion::marker_client::MarkerSidecarClient;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = MarkerSidecarClient::connect(
//!     "http://localhost:1069",
//!     Duration::from_secs(10),
//!     Duration::from_secs(600),
//! ).await?;
//!
//! let response = client.chunk_document(
//!     b"...pdf bytes...".to_vec(),
//!     "doc-id-uuid",
//!     "report.pdf",
//!     "application/pdf",
//! ).await?;
//!
//! println!("got {} chunks via marker", response.chunks.len());
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use crate::error::RetrievalError;
use crate::ingestion::docling_client::{DoclingResponse, DoclingSidecarClient};

/// Streaming gRPC client for the Marker sidecar.
///
/// Share via `Arc` — the underlying tonic `Channel` is already
/// `Arc`-backed and HTTP/2 multiplexes concurrent streams.
pub struct MarkerSidecarClient {
    inner: DoclingSidecarClient,
}

/// Response shape is identical to Docling's — both backends emit the
/// same `DocumentHeader → Chunk* → DocumentFooter` stream.
pub type MarkerResponse = DoclingResponse;

impl MarkerSidecarClient {
    /// Connect to the Marker sidecar at `endpoint`.
    ///
    /// The channel is lazily connected; errors surface on the first
    /// `chunk_document` call rather than here.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::DoclingIngestFailed`] if the endpoint
    /// URI cannot be parsed. (The error variant is named after Docling
    /// for historical reasons but covers every sidecar implementing the
    /// shared proto.)
    pub async fn connect(
        endpoint: impl Into<String>,
        connect_timeout: Duration,
        stream_idle_timeout: Duration,
    ) -> Result<Self, RetrievalError> {
        let inner =
            DoclingSidecarClient::connect(endpoint, connect_timeout, stream_idle_timeout).await?;
        Ok(Self { inner })
    }

    /// Stream a document through the sidecar and collect the response.
    ///
    /// See [`DoclingSidecarClient::chunk_document`] for the full state
    /// machine — this method is a pass-through.
    ///
    /// # Errors
    ///
    /// See [`DoclingSidecarClient::chunk_document`].
    pub async fn chunk_document(
        &self,
        file_bytes: Vec<u8>,
        doc_id: &str,
        source: &str,
        format: &str,
    ) -> Result<MarkerResponse, RetrievalError> {
        self.inner
            .chunk_document(file_bytes, doc_id, source, format)
            .await
    }
}
