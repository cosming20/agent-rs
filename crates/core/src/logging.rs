//! Structured logging initialization using `tracing-subscriber`.
//!
//! Supports two output modes:
//! - **JSON** (`json = true`): machine-readable, for production
//! - **Pretty** (`json = false`): human-readable, for development
//!
//! Filter level is controlled via the `RUST_LOG` environment variable
//! (defaults to `info`).

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize the global tracing subscriber.
///
/// Call this once at application startup. Subsequent calls will panic
/// because `tracing` only allows one global subscriber.
///
/// # Arguments
///
/// * `json` - When `true`, emit structured JSON logs suitable for log
///   aggregation systems. When `false`, emit pretty-printed logs for
///   local development.
pub fn init_logging(json: bool) {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if json {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json().with_target(true).with_thread_ids(true))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().pretty().with_target(true))
            .init();
    }

    tracing::info!("logging initialized (json={json})");
}
