//! Tonic streaming client for the Docling sidecar gRPC service.
//!
//! [`DoclingSidecarClient`] wraps the generated [`DoclingChunkerClient`] and
//! drives the `ChunkDocument` server-streaming RPC.  It handles:
//!
//! - Stream state-machine validation (Header → Chunks* → Footer or Error).
//! - Idle-stream timeout cancellation via [`tokio::select!`].
//! - Typed error conversion to [`RetrievalError::DoclingIngestFailed`].
//! - Schema-version validation (rejects incompatible sidecars early).
//!
//! # Usage
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use agent_rs_retrieval::ingestion::docling_client::DoclingSidecarClient;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut client = DoclingSidecarClient::connect(
//!     "http://localhost:1068",
//!     Duration::from_secs(10),
//!     Duration::from_secs(600),
//! ).await?;
//!
//! let response = client.chunk_document(
//!     b"...pdf bytes...".to_vec(),
//!     "doc-id-uuid",
//!     "my_report.pdf",
//!     "application/pdf",
//! ).await?;
//!
//! println!("got {} chunks", response.chunks.len());
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
use tracing::{debug, info, warn};

use crate::error::RetrievalError;
use crate::ingestion::types::IngestErrorCode;
use crate::proto::agentic_rag::v1::Chunk;
use crate::proto::agentic_rag::v1::{
    ChunkDocumentRequest, DocumentFooter, DocumentHeader, RequestLimits,
    chunk_document_response::Payload, docling_chunker_client::DoclingChunkerClient,
};

/// The highest schema_version this client understands.
const MAX_SUPPORTED_SCHEMA_VERSION: u32 = 1;

// ---- Public API ----

/// Streaming gRPC client for the Docling sidecar.
///
/// Wraps the generated [`DoclingChunkerClient`] with typed error handling
/// and idle-stream timeout cancellation.
pub struct DoclingSidecarClient {
    inner: DoclingChunkerClient<Channel>,
    stream_idle_timeout: Duration,
}

/// The parsed response from a successful `ChunkDocument` stream.
///
/// All three components are present only when the sidecar sent a valid
/// Header → Chunk* → Footer sequence without any error payload.
pub struct DoclingResponse {
    /// Document-level metadata from the sidecar's `DocumentHeader`.
    pub header: DocumentHeader,
    /// All chunks in stream order.
    pub chunks: Vec<Chunk>,
    /// Stream footer (aggregate counters, latency).
    pub footer: DocumentFooter,
}

impl DoclingSidecarClient {
    /// Connect to the Docling sidecar at `endpoint`.
    ///
    /// # Parameters
    ///
    /// - `endpoint` — full gRPC URL, e.g. `http://docling-sidecar:1068`.
    /// - `connect_timeout` — timeout for the initial TCP/TLS connection.
    /// - `stream_idle_timeout` — maximum idle time between stream messages
    ///   before the client cancels the RPC.
    ///
    /// The channel is lazily connected; errors surface on the first RPC.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::DoclingIngestFailed`] with
    /// `code = InternalError` if the endpoint URI cannot be parsed.
    pub async fn connect(
        endpoint: impl Into<String>,
        connect_timeout: Duration,
        stream_idle_timeout: Duration,
    ) -> Result<Self, RetrievalError> {
        let endpoint_str = endpoint.into();

        let channel = Endpoint::new(endpoint_str.clone())
            .map_err(|e| RetrievalError::DoclingIngestFailed {
                code: IngestErrorCode::InternalError,
                retryable: false,
                triggers_legacy_fallback: false,
                message: format!("invalid sidecar endpoint URI '{}': {}", endpoint_str, e),
            })?
            .connect_timeout(connect_timeout)
            .connect_lazy();

        Ok(Self {
            inner: DoclingChunkerClient::new(channel),
            stream_idle_timeout,
        })
    }

    /// Stream a document through the sidecar and collect the response.
    ///
    /// Sends a [`ChunkDocumentRequest`] and drives the server-streaming RPC
    /// until the stream closes.
    ///
    /// # State machine
    ///
    /// ```text
    /// Initial
    ///   → Header  received              → ExpectingChunksOrFooter
    ///   → Chunk   received before Header → DoclingIngestFailed (protocol error)
    ///   → Footer  received before Header → DoclingIngestFailed (protocol error)
    ///   → Error   received               → DoclingIngestFailed (error payload)
    ///
    /// ExpectingChunksOrFooter
    ///   → Chunk   received   → ExpectingChunksOrFooter (accumulate)
    ///   → Footer  received   → Done (return Ok)
    ///   → Header  received   → DoclingIngestFailed (duplicate header)
    ///   → Error   received   → DoclingIngestFailed (error payload)
    ///
    /// Done / stream closed without Footer → DoclingIngestFailed
    /// ```
    ///
    /// The idle timeout is applied per-message via `tokio::select!`.
    ///
    /// # Parameters
    ///
    /// - `file_bytes` — raw document bytes.
    /// - `doc_id` — UUIDv5 identifier for this document (echoed in the header).
    /// - `source` — originating filename, e.g. `"annual_report.pdf"`.
    /// - `format` — MIME type hint, e.g. `"application/pdf"`.  Empty string
    ///   lets the sidecar sniff from the bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::DoclingIngestFailed`] on any of:
    /// - gRPC `Status::Unavailable` (sidecar not reachable).
    /// - Stream idle timeout exceeded.
    /// - Out-of-order or missing stream messages.
    /// - Sidecar-reported `IngestError` payload.
    pub async fn chunk_document(
        &mut self,
        file_bytes: Vec<u8>,
        doc_id: &str,
        source: &str,
        format: &str,
    ) -> Result<DoclingResponse, RetrievalError> {
        use tokio_stream::StreamExt as _;

        // Compute sha256 for request integrity check.  The sidecar verifies
        // this and returns INVALID_METADATA on a mismatch.
        let sha256 = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&file_bytes);
            format!("{:x}", hasher.finalize())
        };

        let request = ChunkDocumentRequest {
            content: file_bytes,
            filename: source.to_owned(),
            mimetype: format.to_owned(),
            document_id: doc_id.to_owned(),
            sha256,
            limits: Some(RequestLimits::default()),
        };

        debug!(
            doc_id,
            source, "sending ChunkDocumentRequest to Docling sidecar"
        );

        let response = self
            .inner
            .chunk_document(request)
            .await
            .map_err(|status| map_tonic_status(status, doc_id))?;

        let mut stream = response.into_inner();
        let idle_timeout = self.stream_idle_timeout;

        // Stream state.
        let mut header: Option<DocumentHeader> = None;
        let mut chunks: Vec<Chunk> = Vec::new();

        loop {
            let next_message = tokio::select! {
                msg = stream.next() => msg,
                () = tokio::time::sleep(idle_timeout) => {
                    return Err(RetrievalError::DoclingIngestFailed {
                        code: IngestErrorCode::InternalError,
                        retryable: true,
                        triggers_legacy_fallback: false,
                        message: format!(
                            "stream idle timeout ({} s) exceeded for doc_id={doc_id}",
                            idle_timeout.as_secs()
                        ),
                    });
                }
            };

            match next_message {
                None => {
                    // Stream closed without a Footer message.
                    return Err(RetrievalError::DoclingIngestFailed {
                        code: IngestErrorCode::InternalError,
                        retryable: true,
                        triggers_legacy_fallback: false,
                        message: format!("stream closed without footer for doc_id={doc_id}"),
                    });
                }
                Some(Err(status)) => {
                    return Err(map_tonic_status(status, doc_id));
                }
                Some(Ok(response_msg)) => {
                    match response_msg.payload {
                        None => {
                            return Err(RetrievalError::DoclingIngestFailed {
                                code: IngestErrorCode::InternalError,
                                retryable: true,
                                triggers_legacy_fallback: false,
                                message: format!(
                                    "received stream message with no payload for doc_id={doc_id}"
                                ),
                            });
                        }

                        Some(Payload::Header(h)) => {
                            if header.is_some() {
                                return Err(RetrievalError::DoclingIngestFailed {
                                    code: IngestErrorCode::InternalError,
                                    retryable: false,
                                    triggers_legacy_fallback: false,
                                    message: format!(
                                        "received duplicate DocumentHeader for doc_id={doc_id}"
                                    ),
                                });
                            }
                            // Validate schema version early so we don't parse
                            // a format we don't understand.
                            if h.schema_version > MAX_SUPPORTED_SCHEMA_VERSION {
                                return Err(RetrievalError::DoclingIngestFailed {
                                    code: IngestErrorCode::InternalError,
                                    retryable: false,
                                    triggers_legacy_fallback: false,
                                    message: format!(
                                        "unsupported schema_version {} (max supported: {}) \
                                         for doc_id={doc_id}",
                                        h.schema_version, MAX_SUPPORTED_SCHEMA_VERSION
                                    ),
                                });
                            }
                            info!(
                                doc_id,
                                docling_version = %h.docling_version,
                                schema_version = h.schema_version,
                                n_pages = h.n_pages,
                                "received DocumentHeader from sidecar"
                            );
                            header = Some(h);
                        }

                        Some(Payload::Chunk(c)) => {
                            if header.is_none() {
                                return Err(RetrievalError::DoclingIngestFailed {
                                    code: IngestErrorCode::InternalError,
                                    retryable: false,
                                    triggers_legacy_fallback: false,
                                    message: format!(
                                        "received Chunk before DocumentHeader for doc_id={doc_id}"
                                    ),
                                });
                            }
                            chunks.push(c);
                        }

                        Some(Payload::Footer(f)) => {
                            let Some(h) = header else {
                                return Err(RetrievalError::DoclingIngestFailed {
                                    code: IngestErrorCode::InternalError,
                                    retryable: false,
                                    triggers_legacy_fallback: false,
                                    message: format!(
                                        "received Footer before DocumentHeader for doc_id={doc_id}"
                                    ),
                                });
                            };
                            info!(
                                doc_id,
                                chunks_received = chunks.len(),
                                total_handle_latency_ms = f.total_handle_latency_ms,
                                "received DocumentFooter; stream complete"
                            );
                            return Ok(DoclingResponse {
                                header: h,
                                chunks,
                                footer: f,
                            });
                        }

                        Some(Payload::Error(e)) => {
                            let code = IngestErrorCode::from(e.code);
                            warn!(
                                doc_id,
                                ?code,
                                message = %e.message,
                                retryable = e.retryable,
                                "Docling sidecar returned IngestError"
                            );
                            return Err(RetrievalError::DoclingIngestFailed {
                                triggers_legacy_fallback: code.triggers_legacy_fallback(),
                                retryable: e.retryable,
                                code,
                                message: e.message,
                            });
                        }
                    }
                }
            }
        }
    }
}

// ---- Private helpers ----

/// Map a [`tonic::Status`] to a typed [`RetrievalError`].
///
/// `Unavailable` is treated as a retryable connection failure that also
/// triggers the legacy fallback path (the sidecar is down, not broken).
fn map_tonic_status(status: tonic::Status, doc_id: &str) -> RetrievalError {
    use tonic::Code;

    let (retryable, triggers_legacy_fallback) = match status.code() {
        Code::Unavailable => (true, true),
        Code::DeadlineExceeded => (true, true),
        Code::ResourceExhausted => (false, false),
        _ => (false, false),
    };

    warn!(
        doc_id,
        grpc_code = ?status.code(),
        message = %status.message(),
        "gRPC call to Docling sidecar failed"
    );

    RetrievalError::DoclingIngestFailed {
        code: IngestErrorCode::InternalError,
        retryable,
        triggers_legacy_fallback,
        message: format!(
            "gRPC error [{:?}] for doc_id={doc_id}: {}",
            status.code(),
            status.message()
        ),
    }
}
