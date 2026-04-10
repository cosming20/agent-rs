//! Embedding pipeline — stub. Full implementation by W3-embedder worker.

use agent_rs_core::{Embedding, RetrievalError};

/// Trait for generating dense embedding vectors from text.
pub trait Embedder: Send + Sync {
    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Embedding, RetrievalError>;

    /// Embed a batch of text strings.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError>;
}

/// fastembed-backed embedder configuration.
#[derive(Debug, Clone)]
pub struct FastEmbedderConfig {
    /// Model name to use with fastembed.
    pub model_name: String,
}

/// fastembed-backed embedder — stub until W3-embedder delivers implementation.
pub struct FastEmbedder;

impl Embedder for FastEmbedder {
    fn embed(&self, _text: &str) -> Result<Embedding, RetrievalError> {
        Err(RetrievalError::Embedding {
            reason: "FastEmbedder: stub — W3-embedder implementation pending".into(),
        })
    }

    fn embed_batch(&self, _texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        Err(RetrievalError::Embedding {
            reason: "FastEmbedder: stub — W3-embedder implementation pending".into(),
        })
    }
}
