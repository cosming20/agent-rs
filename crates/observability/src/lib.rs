//! Observability crate for agent-rs.
//!
//! Provides OpenTelemetry tracing and custom metrics integration.
//!
//! # Usage
//!
//! At application startup:
//!
//! ```no_run
//! use agent_rs_core::config::OtelConfig;
//! use agent_rs_observability::tracing::init_tracing;
//! use agent_rs_observability::metrics::RetrievalMetrics;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = OtelConfig { host: "localhost".into(), collector_port: 1057 };
//!     let _guard = init_tracing(&config, "agent-rs").expect("tracing init failed");
//!     let metrics = RetrievalMetrics::new("agent-rs");
//!     // ... application code ...
//! }
//! ```

pub mod circuit_breaker;
pub mod metrics;
pub mod tracing;

pub use circuit_breaker::{BreakerState, CircuitBreaker};
