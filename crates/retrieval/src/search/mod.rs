//! Concrete search back-end implementations.
//!
//! Each sub-module implements one logical search back-end:
//!
//! - [`vector`]  — dense vector (ANN) search via Qdrant
//! - [`keyword`] — BM25 / lexical search via Qdrant sparse vectors
//! - [`graph`]   — Cypher graph traversal via Neo4j
//! - [`hybrid`]  — parallel multi-backend dispatch returning unfused results
//!
//! # Usage
//!
//! All public functions take trait-object references and return typed results.
//! Fusion is the caller's responsibility (see [`crate::ranker::ResultRanker`]).
//!
//! ```rust,no_run
//! use agent_rs_retrieval::search::hybrid::hybrid_search;
//! ```

pub mod graph;
pub mod hybrid;
pub mod keyword;
pub mod vector;

// Re-export frequently used items at the search module level.
pub use graph::{
    graph_constrained_search, graph_cot_search, graph_rag_context_search, graph_search,
};
pub use hybrid::{GraphHybridParams, HybridResults, graph_hybrid_search, hybrid_search};
pub use keyword::keyword_search;
pub use vector::{summaries_search, temporal_search, vector_search};
