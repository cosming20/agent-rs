//! Universal document ingestion pipeline.
//!
//! Orchestrates the full ingestion flow: text chunking, metadata extraction,
//! embedding, vector upsert, entity extraction, and graph storage.
//!
//! # Modules
//!
//! - [`chunker`] -- configurable text chunking (recursive, section-based, sliding window, token-aware paragraph)
//! - [`metadata`] -- automatic metadata extraction from text content
//! - [`graph_extractor`] -- LLM-based entity/relation extraction (OpenAI structured outputs)
//! - [`pipeline`] -- end-to-end ingestion pipeline wiring all stages together

pub mod chunker;
pub mod graph_extractor;
pub mod metadata;
pub mod pipeline;

pub use chunker::{ChunkStrategy, ChunkerConfig, DocumentChunk};
pub use graph_extractor::{
    DEFAULT_BATCH_CONCURRENCY, ExtractedEdge, ExtractedNode, KnowledgeGraph, LlmGraphExtractor,
};
pub use metadata::{ChunkExtras, ChunkMetadata};
pub use pipeline::{IngestionPipeline, IngestionPipelineConfig, IngestionResult};
