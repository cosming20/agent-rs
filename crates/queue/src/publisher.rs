//! Publisher side — the gRPC service hands off a job and returns.

use rust_rabbit::{Connection, Publisher};
use tracing::{info, instrument};

use crate::error::QueueError;
use crate::types::{IndexJob, QUEUE_NAME};

/// Thin wrapper over [`rust_rabbit::Publisher`] pinned to the indexing queue.
pub struct IndexPublisher {
    inner: Publisher,
    endpoint_label: String,
}

impl IndexPublisher {
    /// Open an AMQP connection to `endpoint` and attach a publisher.
    ///
    /// `endpoint` is stashed unredacted — callers are responsible for
    /// passing a safe string (no inline credentials in logs).
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::Connect`] if the AMQP dial fails.
    #[instrument(skip_all, fields(endpoint = %endpoint))]
    pub async fn connect(endpoint: &str) -> Result<Self, QueueError> {
        let connection = Connection::new(endpoint)
            .await
            .map_err(|e| QueueError::Connect {
                endpoint: endpoint.to_owned(),
                reason: e.to_string(),
            })?;
        let inner = Publisher::new(connection);
        info!(endpoint = %endpoint, queue = QUEUE_NAME, "index publisher ready");
        Ok(Self {
            inner,
            endpoint_label: endpoint.to_owned(),
        })
    }

    /// Publish an [`IndexJob`] to the canonical queue.
    ///
    /// The `Publisher::publish_to_queue` call auto-declares the queue on
    /// first use, so there is no bootstrap step.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::Publish`] on any broker-side failure.
    #[instrument(skip(self, job), fields(
        endpoint = %self.endpoint_label,
        user_id = %job.user_id,
        minio_object_key = %job.minio_object_key,
        job_id = %job.job_id,
    ))]
    pub async fn publish(&self, job: &IndexJob) -> Result<(), QueueError> {
        self.inner
            .publish_to_queue(QUEUE_NAME, job, None)
            .await
            .map_err(|e| QueueError::Publish {
                queue: QUEUE_NAME.to_owned(),
                reason: e.to_string(),
            })?;
        info!("index job enqueued");
        Ok(())
    }
}
