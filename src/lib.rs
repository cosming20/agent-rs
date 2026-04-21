//! agent-rs — reusable wiring helpers for the agentic RAG system.
//!
//! This library provides builder functions that compose the crates in the
//! workspace into a runnable system. After the 2026-04-21 stateless pivot
//! the only entry point is `crates/agent-service` (the gRPC service that
//! speaks the stateless `agent.v1.Agent` contract).
//!
//! # Modules
//!
//! - [`stores`] — connect to Qdrant, Neo4j, Postgres, MinIO
//! - [`embedder`] — build OpenAI or local fastembed embedder
//! - [`router`] — assemble the retrieval router with all cognitive features

pub mod embedder;
pub mod router;
pub mod stores;
