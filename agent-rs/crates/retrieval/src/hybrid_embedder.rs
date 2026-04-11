//! Hybrid embedder: delegates dense to any [`Embedder`] back-end, sparse to a
//! local [`FastEmbedder`].
//!
//! # Why
//!
//! OpenAI's Embeddings API supports only dense vectors.  Hybrid / keyword
//! search in Qdrant requires sparse vectors (BM42 / SPLADE).  `HybridEmbedder`
//! solves this by wiring a remote dense back-end (e.g. [`OpenAIEmbedder`]) with
//! a local `FastEmbedder` that handles sparse inference.
//!
//! # Usage
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use agent_rs_retrieval::hybrid_embedder::HybridEmbedder;
//! # use agent_rs_retrieval::embedder::{FastEmbedder, FastEmbedderConfig};
//! # use agent_rs_retrieval::openai_embedder::{OpenAIEmbedder, OpenAIEmbedderConfig};
//! # async fn example() -> anyhow::Result<()> {
//! let dense = OpenAIEmbedder::new(OpenAIEmbedderConfig::new("sk-..."));
//! let sparse = FastEmbedder::try_new(FastEmbedderConfig::default()).await?;
//! let hybrid = HybridEmbedder::new(Arc::new(dense), Arc::new(sparse));
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use crate::embedder::{Embedder, FastEmbedder, SparseVector};
use crate::error::RetrievalError;
use agent_rs_core::types::Embedding;

// ---- HybridEmbedder ----

/// Combines any dense [`Embedder`] with a local [`FastEmbedder`] for sparse.
///
/// Dense methods (`embed_text`, `embed_batch`, `embed_documents`, `dimensions`)
/// are forwarded to the `dense` back-end.  Sparse methods (`embed_sparse`,
/// `embed_sparse_batch`) are forwarded to the local `sparse` back-end which
/// always supports them.
#[derive(Clone)]
pub struct HybridEmbedder {
    dense: Arc<dyn Embedder + Send + Sync>,
    sparse: Arc<FastEmbedder>,
}

impl std::fmt::Debug for HybridEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridEmbedder")
            .field("dense_dims", &self.dense.dimensions())
            .finish_non_exhaustive()
    }
}

impl HybridEmbedder {
    /// Create a new [`HybridEmbedder`].
    ///
    /// # Parameters
    ///
    /// * `dense` — any [`Embedder`] implementation used for dense vectors (e.g.
    ///   [`OpenAIEmbedder`] or [`FastEmbedder`]).
    /// * `sparse` — a [`FastEmbedder`] used exclusively for sparse vectors.
    pub fn new(dense: Arc<dyn Embedder + Send + Sync>, sparse: Arc<FastEmbedder>) -> Self {
        tracing::info!(dense_dims = dense.dimensions(), "HybridEmbedder created");
        Self { dense, sparse }
    }
}

// ---- Embedder impl ----

#[async_trait]
impl Embedder for HybridEmbedder {
    /// Delegates to the dense back-end.
    async fn embed_text(&self, text: &str) -> Result<Embedding, RetrievalError> {
        self.dense.embed_text(text).await
    }

    /// Delegates to the dense back-end.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        self.dense.embed_batch(texts).await
    }

    /// Delegates to the dense back-end.
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        self.dense.embed_documents(texts).await
    }

    /// Delegates to the local sparse back-end.
    async fn embed_sparse(&self, text: &str) -> Result<SparseVector, RetrievalError> {
        self.sparse.embed_sparse(text).await
    }

    /// Delegates to the local sparse back-end.
    async fn embed_sparse_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<SparseVector>, RetrievalError> {
        self.sparse.embed_sparse_batch(texts).await
    }

    /// Returns the dense dimensionality (from the dense back-end).
    fn dimensions(&self) -> usize {
        self.dense.dimensions()
    }
}
