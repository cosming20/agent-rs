//! Message types exchanged between producer and consumer.

use serde::{Deserialize, Serialize};

/// Canonical queue name. Both the producer and the consumer point at the
/// same string; centralising it here prevents drift.
pub const QUEUE_NAME: &str = "agent-rs.index";

/// Indexing job published by the gRPC service on `EnqueueIndex`.
///
/// The consumer pulls bytes from MinIO using [`Self::minio_object_key`] and
/// reports progress back to the `document_index_state` table; the web app
/// polls `GetDocumentStatus` to follow along.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexJob {
    /// Owner of the document. The consumer re-validates this against the
    /// key prefix before reading bytes — defence-in-depth against a forged
    /// queue message.
    pub user_id: String,
    /// Full MinIO object key, e.g. `users/alice/docs/{uuid}.pdf`.
    pub minio_object_key: String,
    /// Opaque correlation id for log/trace joins across hops. The producer
    /// generates a v4 UUID; the consumer logs it on every state transition.
    pub job_id: String,
}
