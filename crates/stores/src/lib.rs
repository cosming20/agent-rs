//! Storage abstractions and implementations for agent-rs.
//!
//! Defines trait-based contracts ([`VectorStore`], [`GraphStore`],
//! [`StateStore`]) with concrete implementations for:
//! - Qdrant (vector similarity search)
//! - Neo4j (knowledge graph)
//! - Postgres (persistent state via Diesel)

pub mod traits;
pub mod types;

pub mod minio;
pub mod neo4j;
pub mod postgres;
pub mod qdrant;

// Re-export structural graph types so callers can use `agent_rs_stores::DocumentHeaderSnapshot`
// without reaching into the nested module path.
pub use neo4j::structural::{
    DocumentHeaderSnapshot, PageInfo, StructuralChunk, StructuralPicture, StructuralTable,
};
