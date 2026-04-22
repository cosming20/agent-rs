//! Production [`IndexHandler`] implementation for the RabbitMQ
//! `IndexerWorker` consumer.
//!
//! One handler instance is shared across the worker fleet (every replica
//! holds an `Arc<AgentIndexHandler>` and passes it into
//! [`agent_rs_queue::run_indexer`]). Each call to [`Self::handle`] owns a
//! single document from MinIO-read to "state = complete".
//!
//! ## MVP pipeline (no Docling)
//!
//! 1. Validate the `user_id` newtype and claim the state row via
//!    `claim_pending_for_indexing` (CAS `pending → indexing`). If the CAS
//!    fails another worker already took the job — we return `Ok(())` so
//!    the message is acked.
//! 2. Pull bytes from MinIO via [`MinioStore::get_bytes_by_key`]; the
//!    ownership guard on the key prefix prevents cross-user fetches even
//!    with a hand-crafted queue message.
//! 3. Extract plain text: `pdf-extract` for PDFs, UTF-8 for `.txt` / `.md`.
//! 4. Chunk via `agent_rs_retrieval::ingestion::chunker::chunk_document`
//!    (`RecursiveCharacter` default) keyed by MinIO object key.
//! 5. Embed chunks through the shared `Arc<dyn Embedder>`.
//! 6. Upsert to the primary chunks collection in Qdrant, with a payload
//!    carrying `user_id`, `minio_object_key`, `chunk_id`, `chunk_text`,
//!    and `chunk_index`.
//! 7. Mark the state row `complete` with `sha256`, `n_pages`, `n_chunks`.
//!
//! Any failure along the way transitions the state row to `failed` with
//! the error message. The queue message is nacked so lapin redelivers
//! (rust-rabbit's default retry policy decides the backoff).
//!
//! Neo4j structural writes + KG extraction + per-section summaries are
//! intentionally out of scope for this MVP — they require the Docling
//! sidecar + LLM API credentials and will land in a follow-up PR.

use std::sync::Arc;

use agent_rs_core::types::{CollectionName, UserId};
use agent_rs_queue::{IndexHandler, IndexJob};
use agent_rs_retrieval::embedder::Embedder;
use agent_rs_retrieval::ingestion::{ChunkerConfig, chunk_document};
use agent_rs_stores::minio::MinioStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::VectorPoint;
use async_trait::async_trait;
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{error, info, instrument, warn};

/// Concrete [`IndexHandler`] wiring the agent-rs storage stack.
pub struct AgentIndexHandler {
    minio: Arc<MinioStore>,
    embedder: Arc<dyn Embedder + Send + Sync>,
    qdrant: Arc<QdrantStore>,
    pg: Arc<PgStore>,
    /// Primary chunks collection (same as `server::AgentServer.chunk_collection`).
    chunk_collection: CollectionName,
    chunker_config: ChunkerConfig,
}

impl AgentIndexHandler {
    /// Create a new handler. Every dependency is already `Arc`-wrapped to
    /// make the worker pool cheap: `Arc::clone` per job, no re-allocation.
    pub fn new(
        minio: Arc<MinioStore>,
        embedder: Arc<dyn Embedder + Send + Sync>,
        qdrant: Arc<QdrantStore>,
        pg: Arc<PgStore>,
        chunk_collection: CollectionName,
        chunker_config: ChunkerConfig,
    ) -> Self {
        Self {
            minio,
            embedder,
            qdrant,
            pg,
            chunk_collection,
            chunker_config,
        }
    }

    /// Run the full indexing pipeline, surfacing the structured error so
    /// the outer `handle` impl can mark the state row `failed` before
    /// propagating the error to the queue consumer loop.
    #[instrument(
        skip(self, job),
        fields(
            user_id = %job.user_id,
            minio_object_key = %job.minio_object_key,
            job_id = %job.job_id,
        )
    )]
    async fn index(&self, job: &IndexJob) -> Result<IndexSummary, IndexError> {
        let user_id = UserId::try_new(&job.user_id)
            .map_err(|e| IndexError::InvalidUser(e.to_string()))?;

        // CAS pending → indexing. If another worker already claimed it
        // (`Ok(false)`), acknowledge and skip.
        let claimed = self
            .pg
            .claim_pending_for_indexing(&job.minio_object_key)
            .await
            .map_err(|e| IndexError::StateMachine(e.to_string()))?;
        if !claimed {
            warn!("job not in pending state; another worker claimed it or it is already done");
            return Ok(IndexSummary::already_claimed());
        }

        // ---- 1. read bytes -----------------------------------------------
        let bytes = self
            .minio
            .get_bytes_by_key(&user_id, &job.minio_object_key)
            .await
            .map_err(|e| IndexError::Fetch(e.to_string()))?;

        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        info!(bytes = bytes.len(), sha256 = %sha256, "document fetched from MinIO");

        // ---- 2. extract text --------------------------------------------
        let (text, n_pages_estimate) = extract_text(&job.minio_object_key, &bytes)?;
        if text.trim().is_empty() {
            return Err(IndexError::Parse(
                "document produced no extractable text".into(),
            ));
        }

        // ---- 3. chunk ----------------------------------------------------
        let chunks = chunk_document(
            &text,
            &job.minio_object_key,
            &job.minio_object_key,
            &self.chunker_config,
        );
        if chunks.is_empty() {
            return Err(IndexError::Parse(
                "chunker produced zero chunks — document below minimum size".into(),
            ));
        }
        let n_chunks = chunks.len() as i32;

        // ---- 4. embed ----------------------------------------------------
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self
            .embedder
            .embed_documents(&texts)
            .await
            .map_err(|e| IndexError::Embed(e.to_string()))?;
        if embeddings.len() != chunks.len() {
            return Err(IndexError::Embed(format!(
                "embedder returned {} vectors for {} chunks",
                embeddings.len(),
                chunks.len()
            )));
        }

        // ---- 5. build VectorPoints --------------------------------------
        let points: Vec<VectorPoint> = chunks
            .iter()
            .zip(embeddings.into_iter())
            .map(|(c, emb)| VectorPoint {
                id: c.chunk_id.clone(),
                embedding: emb,
                text: c.content.clone(),
                payload: json!({
                    "user_id": user_id.as_str(),
                    "minio_object_key": job.minio_object_key,
                    "chunk_id": c.chunk_id,
                    "chunk_text": c.content,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                }),
            })
            .collect();

        // ---- 6. Qdrant upsert -------------------------------------------
        self.qdrant
            .upsert(&self.chunk_collection, points)
            .await
            .map_err(|e| IndexError::Upsert(e.to_string()))?;

        // ---- 7. mark complete -------------------------------------------
        self.pg
            .mark_index_complete(&job.minio_object_key, &sha256, n_pages_estimate, n_chunks)
            .await
            .map_err(|e| IndexError::StateMachine(e.to_string()))?;

        info!(
            sha256 = %sha256,
            n_pages = n_pages_estimate,
            n_chunks,
            "indexing complete"
        );
        Ok(IndexSummary {
            claimed: true,
            sha256,
            n_pages: n_pages_estimate,
            n_chunks,
        })
    }
}

#[async_trait]
impl IndexHandler for AgentIndexHandler {
    async fn handle(
        &self,
        job: IndexJob,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self.index(&job).await {
            Ok(summary) => {
                if summary.claimed {
                    info!(
                        minio_object_key = %job.minio_object_key,
                        sha256 = %summary.sha256,
                        n_pages = summary.n_pages,
                        n_chunks = summary.n_chunks,
                        "job completed"
                    );
                }
                Ok(())
            }
            Err(e) => {
                error!(
                    minio_object_key = %job.minio_object_key,
                    job_id = %job.job_id,
                    error = %e,
                    "indexing failed; marking state row and nacking"
                );
                let _ = self
                    .pg
                    .mark_index_failed(&job.minio_object_key, &e.to_string())
                    .await;
                Err(Box::new(e))
            }
        }
    }
}

// ---- helpers ----------------------------------------------------------------

/// Determine text content from the raw bytes using the object key's
/// extension as the routing hint. Returns `(text, estimated_n_pages)` so
/// the state row carries an approximate page count even when the backend
/// (plain text / markdown) has no native notion of pages.
fn extract_text(key: &str, bytes: &[u8]) -> Result<(String, i32), IndexError> {
    let ext = key
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "pdf" => {
            let text = pdf_extract::extract_text_from_mem(bytes)
                .map_err(|e| IndexError::Parse(format!("pdf_extract: {e}")))?;
            // pdf-extract does not report page count; approximate from `\x0c`
            // form-feed separators the crate inserts between pages.
            let n_pages = (text.matches('\x0c').count() as i32).max(1);
            Ok((text, n_pages))
        }
        "txt" | "md" | "markdown" | "html" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| IndexError::Parse(format!("utf-8: {e}")))?
                .to_owned();
            Ok((text, 1))
        }
        other => Err(IndexError::Parse(format!(
            "unsupported extension '{other}' on key '{key}'"
        ))),
    }
}

/// What we've accomplished on a successful pass — used for debug logs and
/// potentially tests.
#[derive(Debug)]
struct IndexSummary {
    /// `false` when the CAS claim failed (another worker took the job);
    /// `true` when this invocation did the work.
    claimed: bool,
    sha256: String,
    n_pages: i32,
    n_chunks: i32,
}

impl IndexSummary {
    fn already_claimed() -> Self {
        Self {
            claimed: false,
            sha256: String::new(),
            n_pages: 0,
            n_chunks: 0,
        }
    }
}

/// Typed error enum so `handle` can decide retry semantics per variant
/// when we stop erasing it to `Box<dyn Error>` (future work).
#[derive(Debug, thiserror::Error)]
enum IndexError {
    #[error("invalid user_id: {0}")]
    InvalidUser(String),
    #[error("state machine: {0}")]
    StateMachine(String),
    #[error("fetch from MinIO: {0}")]
    Fetch(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("embed: {0}")]
    Embed(String),
    #[error("upsert to Qdrant: {0}")]
    Upsert(String),
}
