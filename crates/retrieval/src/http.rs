//! Shared HTTP client factory for the retrieval crate.
//!
//! All modules that make outbound HTTP calls (OpenAI API, etc.) should obtain
//! their [`reqwest::Client`] from [`build_http_client`] rather than calling
//! [`reqwest::Client::new`] directly.  A single client is internally pooled
//! (keep-alive, TLS session reuse) and should be wrapped in [`Arc`] and shared
//! across the application lifetime.
//!
//! # Connection settings
//!
//! | Setting | Value |
//! |---|---|
//! | Connect timeout | 10 s |
//! | Request timeout | caller-supplied (default 60 s) |
//! | Max idle connections per host | 32 |
//! | HTTP/2 keep-alive interval | 30 s |
//! | HTTP/2 keep-alive timeout | 10 s |
//! | HTTP/2 keep-alive while idle | true |

use std::sync::Arc;
use std::time::Duration;

use crate::error::RetrievalError;

/// Default connect timeout for new HTTP connections.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum idle connections kept alive per remote host.
const MAX_IDLE_PER_HOST: usize = 32;

/// HTTP/2 keep-alive ping interval.
const HTTP2_KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(30);

/// HTTP/2 keep-alive timeout (time to wait for a pong before closing).
const HTTP2_KEEP_ALIVE_TIMEOUT: Duration = Duration::from_secs(10);

/// Build a shared [`reqwest::Client`] suitable for all outbound OpenAI / LLM
/// API calls in the retrieval crate.
///
/// # Parameters
///
/// - `request_timeout` — Maximum time to wait for a complete response.
///   Callers should supply a value from their configuration struct; 60 s is a
///   reasonable default for embedding / completion APIs.
///
/// # Errors
///
/// Returns [`RetrievalError::Ingestion`] if the underlying TLS backend cannot
/// initialise (extremely unlikely in practice).
pub fn build_http_client(
    request_timeout: Duration,
) -> Result<Arc<reqwest::Client>, RetrievalError> {
    let client = reqwest::Client::builder()
        .timeout(request_timeout)
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .pool_max_idle_per_host(MAX_IDLE_PER_HOST)
        .http2_keep_alive_interval(Some(HTTP2_KEEP_ALIVE_INTERVAL))
        .http2_keep_alive_timeout(HTTP2_KEEP_ALIVE_TIMEOUT)
        .http2_keep_alive_while_idle(true)
        .build()
        .map_err(|e| RetrievalError::Ingestion {
            reason: format!("failed to build shared HTTP client: {e}"),
        })?;

    Ok(Arc::new(client))
}
