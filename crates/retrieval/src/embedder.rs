//! Embedding pipeline for the agent-rs retrieval engine.
//!
//! # Overview
//!
//! This module exposes the [`Embedder`] trait and the production implementation
//! [`FastEmbedder`], which uses `fastembed` ONNX models for fully local
//! inference — no external API calls required.
//!
//! ## Thread-safety contract
//!
//! `fastembed`'s [`TextEmbedding`] and [`SparseTextEmbedding`] are **not**
//! `Send + Sync` because the underlying ONNX session holds non-`Send` state.
//! Both models are wrapped in `Arc<std::sync::Mutex<T>>`, and every embedding
//! call is offloaded to the blocking thread-pool via
//! [`tokio::task::spawn_blocking`] so the async runtime is never blocked.
//!
//! ## Batch parallelism
//!
//! fastembed handles internal batch parallelism via its ONNX threading. The
//! `rayon` crate is available in this crate for CPU-bound post-processing
//! stages (e.g. RRF scoring across large result sets in the ranker).
//!
//! # Configuration
//!
//! See [`FastEmbedderConfig`] for all tunables.

use std::sync::{Arc, Mutex};

use agent_rs_core::types::Embedding;
use async_trait::async_trait;
use fastembed::{
    EmbeddingModel, InitOptions, SparseInitOptions, SparseTextEmbedding, TextEmbedding,
};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::RetrievalError;

// ---- SparseVector ----

/// A sparse embedding vector returned by BM42 / SPLADE models.
///
/// Non-zero dimensions are identified by `indices`; corresponding activation
/// magnitudes are stored in `values`. Both slices are always the same length.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero dimension indices (vocabulary token IDs for SPLADE models).
    pub indices: Vec<u32>,
    /// Activation magnitude for each non-zero index.
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Return the number of non-zero components.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Return `true` when all components are zero (empty sparse vector).
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ---- Embedder trait ----

/// Contract that all embedding back-ends must satisfy.
///
/// # Design
///
/// Methods are `async` because production implementations offload CPU-bound
/// ONNX inference to a blocking thread-pool. Implementors must be
/// `Send + Sync` so they can be stored behind `Arc<dyn Embedder>` and shared
/// across concurrent request handlers.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a dense [`Embedding`].
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when the ONNX inference session
    /// fails or returns unexpected output.
    async fn embed_text(&self, text: &str) -> Result<Embedding, RetrievalError>;

    /// Embed a batch of text strings into dense [`Embedding`]s.
    ///
    /// The order of the returned embeddings matches the order of `texts`.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when the ONNX session fails for
    /// any element in the batch.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError>;

    /// Embed a single text string into a [`SparseVector`] via SPLADE / BM42.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when the sparse model session
    /// fails or returns malformed output.
    async fn embed_sparse(&self, text: &str) -> Result<SparseVector, RetrievalError>;

    /// Embed a batch of text strings into [`SparseVector`]s.
    ///
    /// The order of the returned vectors matches the order of `texts`.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when the sparse model session
    /// fails for any element in the batch.
    async fn embed_sparse_batch(&self, texts: &[&str])
    -> Result<Vec<SparseVector>, RetrievalError>;

    /// Return the number of dimensions in the dense embedding space.
    ///
    /// Constant for the lifetime of a given [`Embedder`] instance; detected
    /// at construction time by a probe embedding.
    fn dimensions(&self) -> usize;
}

// ---- FastEmbedderConfig ----

/// Configuration for [`FastEmbedder`].
///
/// All fields have documented defaults. Use [`FastEmbedderConfig::default()`]
/// for the recommended production configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastEmbedderConfig {
    /// Dense embedding model to load.
    ///
    /// Default: `BGESmallENV15` — 33 M params, 384 dims, strong quality-to-
    /// latency trade-off for English RAG workloads.
    #[serde(skip, default = "default_dense_model")]
    pub dense_model: EmbeddingModel,

    /// Number of documents per dense inference batch.
    ///
    /// `None` uses the fastembed library default (256).
    pub dense_batch_size: Option<usize>,

    /// Number of documents per sparse inference batch.
    ///
    /// `None` uses the fastembed library default.
    pub sparse_batch_size: Option<usize>,

    /// Whether to print model download progress bars to stderr.
    ///
    /// Enable during local development; disable in production to avoid
    /// polluting structured logs.
    pub show_download_progress: bool,
}

fn default_dense_model() -> EmbeddingModel {
    EmbeddingModel::BGESmallENV15
}

impl Default for FastEmbedderConfig {
    fn default() -> Self {
        Self {
            dense_model: EmbeddingModel::BGESmallENV15,
            dense_batch_size: None,
            sparse_batch_size: None,
            show_download_progress: false,
        }
    }
}

// ---- FastEmbedder ----

/// Production embedding back-end powered by local ONNX models via `fastembed`.
///
/// ## Initialization
///
/// ```rust,no_run
/// use agent_rs_retrieval::embedder::{FastEmbedder, FastEmbedderConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// let embedder = FastEmbedder::try_new(FastEmbedderConfig::default()).await?;
/// let embedding = embedder.embed_text("Hello, world!").await?;
/// println!("dims: {}", embedder.dimensions());
/// # Ok(())
/// # }
/// ```
///
/// ## Send + Sync safety
///
/// `FastEmbedder` is `Send + Sync` even though the underlying fastembed models
/// are not, because both are wrapped in `Arc<Mutex<T>>` and all calls that
/// access the inner state are dispatched via `tokio::task::spawn_blocking`.
///
/// ## Dimension detection
///
/// Dimensions are detected at construction time by embedding a single probe
/// string and measuring the returned vector length. This avoids hard-coding
/// model-specific constants that could silently become stale.
#[derive(Clone)]
pub struct FastEmbedder {
    /// Thread-safe wrapper around the dense ONNX model.
    dense: Arc<Mutex<TextEmbedding>>,
    /// Thread-safe wrapper around the sparse ONNX model.
    sparse: Arc<Mutex<SparseTextEmbedding>>,
    /// Dense embedding dimensionality detected at construction time.
    dims: usize,
    /// Batch size forwarded to the dense model (`None` = fastembed default 256).
    dense_batch_size: Option<usize>,
    /// Batch size forwarded to the sparse model (`None` = fastembed default).
    sparse_batch_size: Option<usize>,
}

impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("dims", &self.dims)
            .field("dense_batch_size", &self.dense_batch_size)
            .field("sparse_batch_size", &self.sparse_batch_size)
            .finish_non_exhaustive()
    }
}

impl FastEmbedder {
    /// Initialise both the dense and sparse ONNX models and return a ready
    /// [`FastEmbedder`].
    ///
    /// Model weights are downloaded on first use to the fastembed cache
    /// directory (`~/.cache/fastembed` by default). Subsequent starts reuse
    /// the on-disk cache and require no network access.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when either model fails to
    /// initialise or the dimension probe fails.
    #[instrument(skip(config), fields(
        show_progress = config.show_download_progress,
    ))]
    pub async fn try_new(config: FastEmbedderConfig) -> Result<Self, RetrievalError> {
        let dense_batch_size = config.dense_batch_size;
        let sparse_batch_size = config.sparse_batch_size;
        let show_progress = config.show_download_progress;

        // --- Dense model initialisation (blocking; may download weights) ---
        let dense_init_opts =
            InitOptions::new(config.dense_model).with_show_download_progress(show_progress);

        let dense_model = tokio::task::spawn_blocking(move || {
            TextEmbedding::try_new(dense_init_opts).map_err(|e| RetrievalError::Embedding {
                reason: e.to_string(),
            })
        })
        .await
        .map_err(|join_err| RetrievalError::Embedding {
            reason: format!("dense model init task panicked: {join_err}"),
        })??;

        // --- Sparse model initialisation (blocking; may download weights) ---
        // The default SparseInitOptions selects SPLADE++.
        let sparse_init_opts =
            SparseInitOptions::default().with_show_download_progress(show_progress);

        let sparse_model = tokio::task::spawn_blocking(move || {
            SparseTextEmbedding::try_new(sparse_init_opts).map_err(|e| RetrievalError::Embedding {
                reason: e.to_string(),
            })
        })
        .await
        .map_err(|join_err| RetrievalError::Embedding {
            reason: format!("sparse model init task panicked: {join_err}"),
        })??;

        let dense = Arc::new(Mutex::new(dense_model));
        let sparse = Arc::new(Mutex::new(sparse_model));

        // --- Detect dense dimensionality via a single probe embedding ---
        let dims = {
            let probe_dense = Arc::clone(&dense);
            let probe_batch = dense_batch_size;
            tokio::task::spawn_blocking(move || {
                let mut guard = probe_dense.lock().map_err(|_| RetrievalError::Embedding {
                    reason: "dense model mutex poisoned during dimension probe".to_owned(),
                })?;
                guard
                    .embed(vec!["probe"], probe_batch)
                    .map_err(|e| RetrievalError::Embedding {
                        reason: e.to_string(),
                    })?
                    .into_iter()
                    .next()
                    .map(|v| v.len())
                    .ok_or_else(|| RetrievalError::Embedding {
                        reason: "dimension probe returned empty result".to_owned(),
                    })
            })
            .await
            .map_err(|join_err| RetrievalError::Embedding {
                reason: format!("dimension probe task panicked: {join_err}"),
            })??
        };

        tracing::info!(dims, "FastEmbedder initialised");

        Ok(Self {
            dense,
            sparse,
            dims,
            dense_batch_size,
            sparse_batch_size,
        })
    }
}

// ---- Embedder impl ----

#[async_trait]
impl Embedder for FastEmbedder {
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    async fn embed_text(&self, text: &str) -> Result<Embedding, RetrievalError> {
        // Delegate to embed_batch to avoid duplicating the spawn_blocking logic.
        let slice = [text];
        let mut results = self.embed_batch(&slice[..]).await?;
        results.pop().ok_or_else(|| RetrievalError::Embedding {
            reason: "embed_text: batch returned empty result".to_owned(),
        })
    }

    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Clone to owned strings so they can cross the spawn_blocking boundary.
        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let model = Arc::clone(&self.dense);
        let batch_size = self.dense_batch_size;

        let raw_vecs: Vec<Vec<f32>> = tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().map_err(|_| RetrievalError::Embedding {
                reason: "dense model mutex poisoned".to_owned(),
            })?;
            let str_refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            guard
                .embed(str_refs, batch_size)
                .map_err(|e| RetrievalError::Embedding {
                    reason: e.to_string(),
                })
        })
        .await
        .map_err(|join_err| RetrievalError::Embedding {
            reason: format!("embed_batch task panicked: {join_err}"),
        })??;

        Ok(raw_vecs.into_iter().map(Embedding).collect())
    }

    #[instrument(skip(self, text), fields(text_len = text.len()))]
    async fn embed_sparse(&self, text: &str) -> Result<SparseVector, RetrievalError> {
        let slice = [text];
        let mut results = self.embed_sparse_batch(&slice[..]).await?;
        results.pop().ok_or_else(|| RetrievalError::Embedding {
            reason: "embed_sparse: batch returned empty result".to_owned(),
        })
    }

    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_sparse_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<SparseVector>, RetrievalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let model = Arc::clone(&self.sparse);
        let batch_size = self.sparse_batch_size;

        tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().map_err(|_| RetrievalError::Embedding {
                reason: "sparse model mutex poisoned".to_owned(),
            })?;
            let str_refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            // fastembed SparseEmbedding uses Vec<usize> for indices; we store
            // them as Vec<u32> because Qdrant's sparse vector API expects u32.
            // Vocabulary token IDs are always < 2^32, so the cast is safe.
            guard
                .embed(str_refs, batch_size)
                .map_err(|e| RetrievalError::Embedding {
                    reason: e.to_string(),
                })?
                .into_iter()
                .map(|se| {
                    Ok(SparseVector {
                        indices: se.indices.into_iter().map(|i| i as u32).collect(),
                        values: se.values,
                    })
                })
                .collect::<Result<Vec<SparseVector>, RetrievalError>>()
        })
        .await
        .map_err(|join_err| RetrievalError::Embedding {
            reason: format!("embed_sparse_batch task panicked: {join_err}"),
        })?
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}
