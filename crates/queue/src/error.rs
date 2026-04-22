//! Error type for the queue crate.

/// Errors produced by the publisher / consumer.
///
/// `rust_rabbit::Error` does not implement `StdError` in the current 1.2.x
/// series, so we capture it as `String` after the call returns. The string
/// already carries the underlying lapin context via `Display`; callers
/// classify it via [`QueueError::is_transient`] to decide retry semantics.
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

impl QueueError {
    /// Best-effort classification of the underlying lapin / rust-rabbit
    /// failure.
    ///
    /// We string-match the captured reason against known transient
    /// markers (connection drops, channel closures, AMQP frame reset).
    /// Transient failures warrant a reconnect + retry; terminal failures
    /// (serialisation error, unknown queue type, auth refused) should
    /// surface to the caller without spinning the loop.
    ///
    /// The classifier is conservative: unknown reason strings default to
    /// **non-transient** so we never hide a novel bug behind infinite
    /// retries. Operators see the log and can extend the table here.
    pub fn is_transient(&self) -> bool {
        let reason = match self {
            QueueError::Connect { reason, .. }
            | QueueError::Publish { reason, .. }
            | QueueError::Consume { reason } => reason.as_str(),
        };
        is_transient_reason(reason)
    }
}

/// Keywords in `rust_rabbit::Error::to_string()` that reliably signal a
/// transient, reconnect-recoverable failure.
///
/// Sourced from lapin's `Error` variants and their nested causes:
/// `IOError`, `ConnectionClosed`, `ChannelClosed`, `ConnectionRefused`,
/// frame protocol resets. Extend this list when a new string surfaces in
/// `QueueError.reason` that the existing patterns miss.
const TRANSIENT_MARKERS: &[&str] = &[
    "connection closed",
    "connectionclosed",
    "connection refused",
    "connection reset",
    "channel closed",
    "channelclosed",
    "io error",
    "ioerror",
    "broken pipe",
    "not connected",
    "socket",
    "tcp",
    "timed out",
    "timeout",
];

fn is_transient_reason(reason: &str) -> bool {
    let lc = reason.to_ascii_lowercase();
    TRANSIENT_MARKERS.iter().any(|m| lc.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_closed_is_transient() {
        let e = QueueError::Publish {
            queue: "q".into(),
            reason: "ChannelClosed: broker dropped".into(),
        };
        assert!(e.is_transient());
    }

    #[test]
    fn auth_refused_is_not_transient() {
        let e = QueueError::Connect {
            endpoint: "amqp://bad".into(),
            reason: "ACCESS_REFUSED - Login was refused".into(),
        };
        assert!(!e.is_transient());
    }

    #[test]
    fn io_error_is_transient() {
        let e = QueueError::Publish {
            queue: "q".into(),
            reason: "IOError: broken pipe".into(),
        };
        assert!(e.is_transient());
    }

    #[test]
    fn unknown_reason_defaults_to_terminal() {
        // Conservative: don't swallow novel bugs behind retries.
        let e = QueueError::Publish {
            queue: "q".into(),
            reason: "unexpected-serialisation-error-123".into(),
        };
        assert!(!e.is_transient());
    }
}
