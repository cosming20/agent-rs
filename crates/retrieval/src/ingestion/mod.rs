//! Document ingestion building blocks.
//!
//! After the 2026-04-21 stateless pivot, orchestration moved out of this
//! crate: the RabbitMQ `IndexerWorker` (in `crates/queue`) now drives the
//! end-to-end pipeline (MinIO read → Docling → chunk → embed → Qdrant +
//! Neo4j upsert) and writes progress back via the `document_index_state`
//! table on agent-rs. This module retains the reusable stage primitives.
//!
//! # Modules
//!
//! - [`chunker`] — configurable text chunking (recursive, section-based, sliding window, token-aware paragraph)
//! - [`docling_client`] — tonic streaming client for the Docling sidecar gRPC service
//! - [`entity_id`] — stable entity ID derivation (used by the graph extractor)
//! - [`graph_extractor`] — LLM-based entity/relation extraction
//! - [`metadata`] — automatic metadata extraction from text content
//! - [`mock_docling`] — mock Docling sidecar stream for integration tests
//! - [`summarizer`] — LLM-based chunk summarisation (powers `RetrievalStrategy::Summaries`)
//! - [`text_normalize`] — whitespace and punctuation normalisation helpers
//! - [`types`] — domain types for the Docling ingest pipeline (`IngestErrorCode`)

pub mod chunker;
pub mod docling_client;
pub mod entity_id;
pub mod graph_extractor;
pub mod metadata;
pub mod mock_docling;
pub mod summarizer;
pub mod text_normalize;
pub mod types;

pub use chunker::{ChunkStrategy, ChunkerConfig, DocumentChunk};
pub use docling_client::{DoclingResponse, DoclingSidecarClient};
pub use graph_extractor::{
    DEFAULT_BATCH_CONCURRENCY, ExtractedEdge, ExtractedNode, KnowledgeGraph, LlmGraphExtractor,
    Triplet,
};
pub use metadata::{ChunkExtras, ChunkMetadata};
pub use summarizer::{ChunkSummarizer, ChunkSummary, DEFAULT_SUMMARIZER_CONCURRENCY};
pub use types::IngestErrorCode;
