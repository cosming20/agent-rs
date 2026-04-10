//! Concrete search back-end implementations.
//!
//! Each sub-module implements one logical search back-end:
//!
//! - `vector`  — dense vector (ANN) search via Qdrant
//! - `keyword` — BM25 / lexical search via Qdrant sparse vectors
//! - `graph`   — Cypher graph traversal via Neo4j
//! - `hybrid`  — parallel multi-backend dispatch with RRF fusion
//!
//! # Status
//!
//! Stub implementations — full logic will be added in Phase 3 Steps 8–9.

pub mod graph;
pub mod hybrid;
pub mod keyword;
pub mod vector;
