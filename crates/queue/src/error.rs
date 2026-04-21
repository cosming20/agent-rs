//! Error type for the queue crate.

/// Errors produced by the publisher / consumer.
///
/// `rust_rabbit::Error` does not implement `StdError` in the current 1.2.x
/// series, so we capture it as `String` after the call returns. The string
/// already carries the underlying lapin context via `Display`.
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    /// Failed to open an AMQP connection.
    #[error("connect {endpoint}: {reason}")]
    Connect {
        /// AMQP URI, with credentials redacted by the caller.
        endpoint: String,
        /// Text reason from `rust_rabbit`.
        reason: String,
    },

    /// Failed to publish a message.
    #[error("publish {queue}: {reason}")]
    Publish {
        /// Target queue name.
        queue: String,
        /// Text reason from `rust_rabbit`.
        reason: String,
    },

    /// Consumer loop ended abnormally.
    #[error("consumer loop: {reason}")]
    Consume {
        /// Text reason from `rust_rabbit`.
        reason: String,
    },
}
