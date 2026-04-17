//! Universal document ingestion pipeline.
//!
//! Orchestrates the full ingestion flow: text chunking, metadata extraction,
//! embedding, vector upsert, entity extraction, and graph storage.
//!
//! # Modules
//!
//! - [`chunker`] -- configurable text chunking (recursive, section-based, sliding window, token-aware paragraph)
//! - [`coordinator`] -- cross-store cleanup coordinator for partial ingest failures
//! - [`docling_client`] -- tonic streaming client for the Docling sidecar gRPC service
//! - [`metadata`] -- automatic metadata extraction from text content
//! - [`graph_extractor`] -- LLM-based entity/relation extraction (OpenAI structured outputs)
//! - [`mock_docling`] -- mock Docling sidecar stream for W3 development
//! - [`pipeline`] -- end-to-end ingestion pipeline wiring all stages together
//! - [`summarizer`] -- LLM-based chunk summarization (powers `RetrievalStrategy::Summaries`)
//! - [`types`] -- domain types for the Docling ingest pipeline (`IngestErrorCode`)

pub mod chunker;
pub mod coordinator;
pub mod docling_client;
pub mod entity_id;
pub mod graph_extractor;
pub mod metadata;
pub mod mock_docling;
pub mod pipeline;
pub mod summarizer;
pub mod text_normalize;
pub mod types;

pub use chunker::{ChunkStrategy, ChunkerConfig, DocumentChunk};
pub use coordinator::IngestCoordinator;
pub use docling_client::{DoclingResponse, DoclingSidecarClient};
pub use graph_extractor::{
    DEFAULT_BATCH_CONCURRENCY, ExtractedEdge, ExtractedNode, KnowledgeGraph, LlmGraphExtractor,
    Triplet,
};
pub use metadata::{ChunkExtras, ChunkMetadata};
pub use pipeline::{IngestionPipeline, IngestionPipelineConfig, IngestionResult};
pub use summarizer::{ChunkSummarizer, ChunkSummary, DEFAULT_SUMMARIZER_CONCURRENCY};
pub use types::IngestErrorCode;
