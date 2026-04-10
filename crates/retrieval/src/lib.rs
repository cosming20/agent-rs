//! Retrieval engine for agent-rs.
//!
//! Provides embedding, search strategy dispatch, rank fusion, and evidence
//! pack assembly over the storage backends defined in `agent-rs-stores`.
//!
//! # Module layout
//!
//! - [`embedder`] — [`Embedder`] trait + [`FastEmbedder`] implementation (fastembed/ONNX)
//! - [`strategy`] — [`RetrievalStrategy`] registry and per-strategy configuration
//! - [`router`] — query rewriting and strategy dispatch
//! - [`ranker`] — reciprocal rank fusion, dedup, and [`EvidencePack`] assembly
//! - [`search`] — concrete search back-ends (vector, keyword, graph, hybrid)

pub mod embedder;
pub mod error;
pub mod ranker;
pub mod router;
pub mod search;
pub mod strategy;

pub use embedder::{Embedder, FastEmbedder, FastEmbedderConfig, SparseVector};
pub use error::RetrievalError;
pub use ranker::{EvidencePack, ResultRanker};
pub use strategy::StrategyRegistry;
