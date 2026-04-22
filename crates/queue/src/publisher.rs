//! Publisher side — the gRPC service hands off a job and returns.
//!
//! The publisher is resilient to broker restarts. `rust_rabbit 1.2` does
//! not auto-reconnect: once the underlying lapin channel is closed every
//! subsequent `publish_to_queue` returns an error forever. We wrap the
//! pool in a `RwLock` so that on a transient publish failure we swap the
//! whole pool in-place on the write path while readers keep publishing on
//! the old pool. Under the 1000-concurrent-uploads scenario the
//! round-robin cursor continues to parallelise across the 8 channels,
//! and a broker bounce turns into a single reconnect + retry per
//! in-flight call rather than a permanent outage.

use std::sync::atomic::{AtomicUsize, Ordering};

use rust_rabbit::{Connection, Publisher};
use tokio::sync::RwLock;
use tracing::{info, instrument, warn};

use crate::error::QueueError;
use crate::types::{IndexJob, QUEUE_NAME};

/// Default number of parallel AMQP channels kept in the publisher pool.
///
/// Each channel serialises its own publishes (lapin property), so scaling
/// write throughput past a single channel requires multiple channels on
/// one connection. 8 channels fit comfortably inside the default lapin
/// channel_max (2048) while matching typical CPU core counts.
pub const DEFAULT_CHANNEL_POOL_SIZE: usize = 8;

/// Connection pool wrapper over [`rust_rabbit::Publisher`] with automatic
/// reconnect on transient broker failures.
pub struct IndexPublisher {
    /// Current pool — swapped wholesale on reconnect.
    publishers: RwLock<Vec<Publisher>>,
    /// Round-robin cursor. Wraps via `%` so no clamping is needed.
    cursor: AtomicUsize,
    /// AMQP URI, stashed so reconnect uses the same endpoint.
    endpoint: String,
    /// Size of the pool, preserved on reconnect.
    pool_size: usize,
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

        let publishers = build_pool(endpoint, pool_size).await?;

        info!(
            endpoint = %endpoint,
            queue = QUEUE_NAME,
            pool_size,
            "index publisher pool ready"
        );
        Ok(Self {
            publishers: RwLock::new(publishers),
            cursor: AtomicUsize::new(0),
            endpoint: endpoint.to_owned(),
            pool_size,
        })
    }

    /// Publish an [`IndexJob`] on the next channel in the round-robin
    /// rotation.
    ///
    /// If the first attempt fails with a transient error (connection
    /// closed, broker bounce), the whole pool is rebuilt and the publish
    /// is retried exactly once. A second failure surfaces to the caller
    /// with [`QueueError::Publish`].
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::Publish`] on any broker-side failure that
    /// survives one reconnect attempt, or [`QueueError::Connect`] if the
    /// reconnect itself fails.
    #[instrument(skip(self, job), fields(
        endpoint = %self.endpoint,
        user_id = %job.user_id,
        minio_object_key = %job.minio_object_key,
        job_id = %job.job_id,
        retried,
    ))]
    pub async fn publish(&self, job: &IndexJob) -> Result<(), QueueError> {
        // First attempt on the existing pool.
        match self.try_publish(job).await {
            Ok(()) => {
                tracing::Span::current().record("retried", false);
                Ok(())
            }
            Err(e) if e.is_transient() => {
                warn!(error = %e, "transient publish error; rebuilding pool and retrying once");
                self.reconnect().await?;
                tracing::Span::current().record("retried", true);
                self.try_publish(job).await
            }
            Err(e) => Err(e),
        }
    }

    /// Single publish attempt on the current pool. Never reconnects.
    async fn try_publish(&self, job: &IndexJob) -> Result<(), QueueError> {
        let guard = self.publishers.read().await;
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed) % guard.len();
        guard[idx]
            .publish_to_queue(QUEUE_NAME, job, None)
            .await
            .map_err(|e| QueueError::Publish {
                queue: QUEUE_NAME.to_owned(),
                reason: e.to_string(),
            })
    }

    /// Swap the pool for a freshly-connected one. Callers racing through
    /// `try_publish` either see the new pool on the next read or the old
    /// pool on the current read — either is safe; they'll retry once more
    /// on their own if the old one is terminally broken.
    async fn reconnect(&self) -> Result<(), QueueError> {
        let fresh = build_pool(&self.endpoint, self.pool_size).await?;
        let mut guard = self.publishers.write().await;
        *guard = fresh;
        info!(
            endpoint = %self.endpoint,
            pool_size = self.pool_size,
            "publisher pool reconnected"
        );
        Ok(())
    }
}

/// Build a fresh connection + N publishers.
async fn build_pool(endpoint: &str, pool_size: usize) -> Result<Vec<Publisher>, QueueError> {
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
    Ok(publishers)
}
