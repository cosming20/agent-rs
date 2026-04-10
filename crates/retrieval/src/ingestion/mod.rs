//! Universal document ingestion pipeline.
//!
//! Orchestrates the full ingestion flow: text chunking, metadata extraction,
//! embedding, vector upsert, entity extraction, and graph storage.
//!
//! # Modules
//!
//! - [`chunker`] -- configurable text chunking (recursive, section-based, sliding window)
//! - [`metadata`] -- automatic metadata extraction from text content
//! - [`pipeline`] -- end-to-end ingestion pipeline wiring all stages together

pub mod chunker;
pub mod metadata;
pub mod pipeline;

pub use chunker::{ChunkStrategy, ChunkerConfig, DocumentChunk};
pub use metadata::ChunkMetadata;
pub use pipeline::{IngestionPipeline, IngestionResult};
