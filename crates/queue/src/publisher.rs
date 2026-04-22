//! Publisher side — the gRPC service hands off a job and returns.

use std::sync::atomic::{AtomicUsize, Ordering};

use rust_rabbit::{Connection, Publisher};
use tracing::{info, instrument};

use crate::error::QueueError;
use crate::types::{IndexJob, QUEUE_NAME};

/// Default number of parallel AMQP channels kept in the publisher pool.
///
/// Each channel serialises its own publishes (lapin property), so scaling
/// write throughput past a single channel requires either multiple
/// channels on one connection or multiple connections. We choose multiple
/// channels on one TCP connection — minimises socket count while still
/// removing the single-channel serialisation bottleneck documented in the
/// "1000 concurrent uploads" scaling review.
pub const DEFAULT_CHANNEL_POOL_SIZE: usize = 8;

/// Connection pool wrapper over [`rust_rabbit::Publisher`].
///
/// Holds N publishers (each backed by its own AMQP channel on the same
/// underlying TCP connection) and dispatches each call through an atomic
/// round-robin counter so a burst of concurrent `publish` calls uses every
/// available channel in parallel rather than serialising behind a single
/// one.
pub struct IndexPublisher {
    /// Publishers sized to [`DEFAULT_CHANNEL_POOL_SIZE`] by
    /// [`IndexPublisher::connect`] (or caller-supplied via
    /// [`IndexPublisher::connect_with_pool`]).
    publishers: Vec<Publisher>,
    /// Round-robin cursor. Wraps on u32/u64 overflow via `%` so we never
    /// need to clamp.
    cursor: AtomicUsize,
    endpoint_label: String,
}

impl IndexPublisher {
    /// Open an AMQP connection and attach [`DEFAULT_CHANNEL_POOL_SIZE`]
    /// channels.
    ///
    /// `endpoint` is stashed unredacted — callers are responsible for
    /// passing a safe string (no inline credentials in logs).
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::Connect`] if the AMQP dial fails.
    pub async fn connect(endpoint: &str) -> Result<Self, QueueError> {
        Self::connect_with_pool(endpoint, DEFAULT_CHANNEL_POOL_SIZE).await
    }

    /// Open an AMQP connection with a caller-chosen channel pool size.
    ///
    /// Size 0 is clamped to 1 to keep the invariant that the round-robin
    /// cursor has a valid target.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::Connect`] if the AMQP dial fails.
    #[instrument(skip_all, fields(endpoint = %endpoint, pool_size))]
    pub async fn connect_with_pool(
        endpoint: &str,
        pool_size: usize,
    ) -> Result<Self, QueueError> {
        let pool_size = pool_size.max(1);
        tracing::Span::current().record("pool_size", pool_size);

        let connection = Connection::new(endpoint)
            .await
            .map_err(|e| QueueError::Connect {
                endpoint: endpoint.to_owned(),
                reason: e.to_string(),
            })?;

        let mut publishers = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            publishers.push(Publisher::new(connection.clone()));
        }

        info!(
            endpoint = %endpoint,
            queue = QUEUE_NAME,
            pool_size,
            "index publisher pool ready"
        );
        Ok(Self {
            publishers,
            cursor: AtomicUsize::new(0),
            endpoint_label: endpoint.to_owned(),
        })
    }

    /// Publish an [`IndexJob`] on the next channel in the round-robin
    /// rotation.
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
        channel,
    ))]
    pub async fn publish(&self, job: &IndexJob) -> Result<(), QueueError> {
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed) % self.publishers.len();
        tracing::Span::current().record("channel", idx);

        self.publishers[idx]
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
