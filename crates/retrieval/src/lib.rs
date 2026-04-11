//! Retrieval engine for agent-rs.
//!
//! Provides embedding, search strategy dispatch, rank fusion, and evidence
//! pack assembly over the storage backends defined in `agent-rs-stores`.
//!
//! # Module layout
//!
//! - [`embedder`] — [`Embedder`] trait + [`FastEmbedder`] implementation (fastembed/ONNX)
//! - [`openai_embedder`] — [`OpenAIEmbedder`] implementation via OpenAI Embeddings API
//! - [`strategy`] — [`RetrievalStrategy`] registry and per-strategy configuration
//! - [`router`] — query rewriting and strategy dispatch
//! - [`ranker`] — reciprocal rank fusion, dedup, and [`EvidencePack`] assembly
//! - [`search`] — concrete search back-ends (vector, keyword, graph, hybrid)
//! - [`ingestion`] — universal document ingestion pipeline (chunking, metadata, embedding, graph)

pub mod embedder;
pub mod error;
pub mod hybrid_embedder;
pub mod ingestion;
pub mod openai_embedder;
pub mod ranker;
pub mod reranker;
pub mod router;
pub mod search;
pub mod strategy;

pub use embedder::{Embedder, FastEmbedder, FastEmbedderConfig, SparseVector};
pub use error::RetrievalError;
pub use hybrid_embedder::HybridEmbedder;
pub use ingestion::{
    ChunkStrategy, ChunkerConfig, DocumentChunk, ExtractedEdge, ExtractedNode, IngestionPipeline,
    IngestionPipelineConfig, IngestionResult, KnowledgeGraph, LlmGraphExtractor, Triplet,
};
pub use openai_embedder::{OpenAIEmbedder, OpenAIEmbedderConfig};
pub use ranker::{EvidencePack, ResultRanker};
pub use reranker::CrossEncoderReranker;
pub use router::RetrievalRouter;
pub use strategy::StrategyRegistry;
