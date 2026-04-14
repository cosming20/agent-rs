//! agent-rs — reusable wiring functions for the agentic RAG system.
//!
//! This library provides builder functions that compose the crates in the
//! workspace into a runnable system. Both `src/main.rs` (the gRPC server)
//! and `examples/eval_harness.rs` (the evaluation tool) import from here.
//!
//! # Modules
//!
//! - [`stores`] — connect to Qdrant, Neo4j, Postgres
//! - [`embedder`] — build OpenAI or local fastembed embedder
//! - [`router`] — assemble the retrieval router with all cognitive features
//! - [`ingest`] — build the ingestion pipeline with Docling sidecar

pub mod embedder;
pub mod ingest;
pub mod router;
pub mod stores;
