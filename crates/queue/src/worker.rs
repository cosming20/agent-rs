//! Consumer side — the `IndexerWorker` pulls jobs and runs indexing.
//!
//! The consumer loop lives in this crate; the actual indexing logic
//! (MinIO read → Docling → chunk → embed → Qdrant/Neo4j upsert) is
//! injected via the [`IndexHandler`] trait so this crate stays agnostic
//! of retrieval / storage internals.
//!
//! `run_indexer` blocks until the consumer loop exits (broker drop, SIGINT,
//! or an unrecoverable error).

use std::sync::Arc;

use async_trait::async_trait;
use rust_rabbit::{Connection, Consumer};
use tracing::{error, info, instrument};

use crate::error::QueueError;
use crate::types::{IndexJob, QUEUE_NAME};

/// Business-logic hook for the indexer.
///
/// One implementation lives in `agent-service` (the production path) and
/// any future integration tests can supply a mock that asserts on the
/// message shape without touching MinIO / Qdrant / Neo4j.
#[async_trait]
pub trait IndexHandler: Send + Sync + 'static {
    /// Process a single [`IndexJob`].
    ///
    /// Returning an error triggers a negative-ack on the queue so the
    /// broker can redeliver the message after its retry policy kicks in.
    async fn handle(&self, job: IndexJob) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Connect to `endpoint` and run the consumer loop, routing each job to
/// `handler`.
///
/// `prefetch` caps the number of in-flight messages — tune based on how
/// memory-heavy the indexing pipeline is per document.
///
/// # Errors
///
/// - [`QueueError::Connect`] — AMQP dial failed.
/// - [`QueueError::Consume`] — the consumer loop exited with an error.
#[instrument(skip(handler), fields(endpoint = %endpoint, prefetch))]
pub async fn run_indexer<H: IndexHandler>(
    endpoint: &str,
    prefetch: u16,
    handler: Arc<H>,
) -> Result<(), QueueError> {
    let connection = Connection::new(endpoint)
        .await
        .map_err(|e| QueueError::Connect {
            endpoint: endpoint.to_owned(),
            reason: e.to_string(),
        })?;

    let consumer = Consumer::builder(connection, QUEUE_NAME)
        .with_prefetch(prefetch)
        .build();

    info!(queue = QUEUE_NAME, prefetch, "indexer consumer ready");

    consumer
        .consume(move |job: IndexJob| {
            let handler = Arc::clone(&handler);
            async move {
                let job_id = job.job_id.clone();
                let key = job.minio_object_key.clone();
                match handler.handle(job).await {
                    Ok(()) => {
                        info!(job_id = %job_id, minio_object_key = %key, "index job ok");
                        Ok(())
                    }
                    Err(e) => {
                        error!(
                            job_id = %job_id,
                            minio_object_key = %key,
                            error = %e,
                            "index job failed — message will be redelivered"
                        );
                        Err(e)
                    }
                }
            }
        })
        .await
        .map_err(|e| QueueError::Consume {
            reason: e.to_string(),
        })?;

    Ok(())
}
