//! Storage abstractions and implementations for agent-rs.
//!
//! Defines trait-based contracts ([`VectorStore`], [`GraphStore`],
//! [`StateStore`]) with concrete implementations for:
//! - Qdrant (vector similarity search)
//! - Neo4j (knowledge graph)
//! - Postgres (persistent state via Diesel)

pub mod traits;
pub mod types;

pub mod neo4j;
pub mod postgres;
pub mod qdrant;
