//! Domain types for the Docling sidecar ingest pipeline.
//!
//! [`IngestErrorCode`] mirrors the proto-generated enum and adds
//! application-level semantics (retryability, legacy-fallback eligibility)
//! so that the ingest state machine can dispatch without consulting the raw
//! proto type at every call site.

use crate::proto::agentic_rag::v1::IngestErrorCode as ProtoIngestErrorCode;

// ---- IngestErrorCode ----

/// Rust-domain mirror of the proto `IngestErrorCode` enum.
///
/// Derives `From<proto::IngestErrorCode>` for lossless, exhaustive conversion
/// at the gRPC boundary.  Any proto variant not yet listed here maps to
/// [`Self::InternalError`] so that new sidecar codes don't break older Rust
/// binaries.
///
/// Use [`Self::triggers_legacy_fallback`] and [`Self::is_retryable`] to
/// make dispatch decisions without match-exhaustiveness surprises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestErrorCode {
    /// Should never appear in practice; treat as [`Self::InternalError`].
    Unspecified,
    /// Docling failed to parse the document (corrupted PDF, layout failure).
    /// `retryable = false`.  Triggers legacy fallback.
    ParseError,
    /// Document type not supported by Docling (e.g., plain-text HTML).
    /// `retryable = false`.  Triggers legacy fallback.
    UnsupportedMimetype,
    /// OCR timed out.  `retryable = true` with a longer `parse_timeout_s`.
    /// Triggers legacy fallback after 1 retry.
    OcrTimeout,
    /// Embedding model not yet downloaded / warmed up.
    /// `retryable = true`.  Does NOT trigger legacy fallback — the whole point
    /// is Docling-quality parsing; falling back silently hides the outage.
    ModelDownloadUnavailable,
    /// `HybridChunker` raised an error — likely a config bug.
    /// `retryable = false`.  Does NOT trigger legacy fallback.
    ChunkingError,
    /// sha256 mismatch or missing required request fields.
    /// `retryable = false`.  Does NOT trigger legacy fallback (client bug).
    InvalidMetadata,
    /// Document exceeds `max_pdf_bytes` or `max_pages`.
    /// `retryable = false`.  Does NOT trigger legacy fallback.
    RequestTooLarge,
    /// `parse_timeout_s` exceeded.
    /// `retryable = true`.  Triggers legacy fallback after 1 retry.
    RequestTimeout,
    /// Catch-all for unexpected sidecar bugs.
    /// `retryable = true` (cap at 1 retry).  Does NOT trigger legacy fallback.
    InternalError,
}

impl IngestErrorCode {
    /// Return `true` if the caller should attempt a retry with backoff.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::OcrTimeout
                | Self::ModelDownloadUnavailable
                | Self::RequestTimeout
                | Self::InternalError
                | Self::Unspecified
        )
    }

    /// Return `true` if a failure with this code may succeed via the legacy
    /// `pdf_extract` backend.
    ///
    /// Used by `IngestBackend::DoclingWithLegacyFallback` to decide whether
    /// to invoke the legacy path after a failed Docling attempt.
    pub fn triggers_legacy_fallback(self) -> bool {
        matches!(
            self,
            Self::UnsupportedMimetype | Self::ParseError | Self::OcrTimeout | Self::RequestTimeout
        )
    }
}

impl From<ProtoIngestErrorCode> for IngestErrorCode {
    fn from(code: ProtoIngestErrorCode) -> Self {
        match code {
            ProtoIngestErrorCode::Unspecified => Self::Unspecified,
            ProtoIngestErrorCode::ParseError => Self::ParseError,
            ProtoIngestErrorCode::UnsupportedMimetype => Self::UnsupportedMimetype,
            ProtoIngestErrorCode::OcrTimeout => Self::OcrTimeout,
            ProtoIngestErrorCode::ModelDownloadUnavailable => Self::ModelDownloadUnavailable,
            ProtoIngestErrorCode::ChunkingError => Self::ChunkingError,
            ProtoIngestErrorCode::InvalidMetadata => Self::InvalidMetadata,
            ProtoIngestErrorCode::RequestTooLarge => Self::RequestTooLarge,
            ProtoIngestErrorCode::RequestTimeout => Self::RequestTimeout,
            ProtoIngestErrorCode::InternalError => Self::InternalError,
        }
    }
}

/// Lossless conversion from the raw `i32` tag that prost stores in
/// `IngestError.code`.  Unknown values map to [`IngestErrorCode::InternalError`]
/// so that new sidecar codes don't panic older clients.
impl From<i32> for IngestErrorCode {
    fn from(v: i32) -> Self {
        ProtoIngestErrorCode::try_from(v)
            .map(Self::from)
            .unwrap_or(Self::InternalError)
    }
}
