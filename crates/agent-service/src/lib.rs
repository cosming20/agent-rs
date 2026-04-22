//! Agent gRPC service — tonic server wrapping [`AgentOrchestrator`],
//! [`PgStore`], [`MinioStore`], and friends.
//!
//! This crate exposes the runtime needed by `src/main.rs`; the binary
//! wires configuration loading + tracing + a graceful shutdown signal
//! around [`server::AgentServer`].

pub mod indexer;
pub mod pb;
pub mod server;
