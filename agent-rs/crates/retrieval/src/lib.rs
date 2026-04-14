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
//! - [`proto`] — generated protobuf types for the Docling sidecar gRPC contract
//! - [`query_rewriter`] — LLM-based query decomposition into focused sub-queries
//! - [`context_expander`] — post-retrieval context expansion via Neo4j FOLLOWS / IN_SECTION edges
//! - [`metadata_filter`] — LLM-based metadata hint extraction for query-time Qdrant filtering

pub mod context_expander;
pub mod embedder;
pub mod entity_linker;
pub mod error;
pub mod graph_cot;
pub mod ingestion;
pub mod metadata_filter;
pub mod openai_embedder;
pub mod proto;
pub mod query_analyzer;
pub mod query_rewriter;
pub mod ranker;
pub mod reranker;
pub mod router;
pub mod search;
pub mod strategy;

pub use context_expander::{ContextChunk, ContextExpander, ContextExpanderConfig, ExpandedChunk};
pub use embedder::{Embedder, FastEmbedder, FastEmbedderConfig, SparseVector};
pub use entity_linker::{EntityLinkResult, EntityLinker, EntityLinkerConfig, LinkedEntity};
pub use error::RetrievalError;
pub use graph_cot::{GraphCotConfig, GraphCotRetriever, IterationLog, SufficiencyCheck};
pub use ingestion::{
    ChunkStrategy, ChunkerConfig, DocumentChunk, ExtractedEdge, ExtractedNode, IngestionPipeline,
    IngestionPipelineConfig, IngestionResult, KnowledgeGraph, LlmGraphExtractor, Triplet,
};
pub use metadata_filter::{
    FilterCondition, MetadataFilterConfig, MetadataFilterExtractor, QueryMetadataHints,
};
pub use openai_embedder::{OpenAIEmbedder, OpenAIEmbedderConfig};
pub use query_analyzer::{QueryAnalysis, QueryAnalyzer, QueryAnalyzerConfig, QueryIntent};
pub use query_rewriter::{
    QueryRewriter, QueryRewriterConfig, RewrittenQuery, SubQuery, SubQueryType,
};
pub use ranker::{EvidencePack, ResultRanker};
pub use reranker::CrossEncoderReranker;
pub use router::RetrievalRouter;
pub use strategy::StrategyRegistry;
