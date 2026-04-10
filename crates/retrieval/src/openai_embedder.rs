//! OpenAI embedding backend using the `text-embedding-3-large` model.
//!
//! # Overview
//!
//! [`OpenAIEmbedder`] implements the [`Embedder`] trait by calling the
//! [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).
//! Unlike the local [`FastEmbedder`](super::FastEmbedder), this backend requires
//! network access and a valid OpenAI API key.
//!
//! # Batch handling
//!
//! The OpenAI API accepts up to 2048 texts per request, but large batches risk
//! exceeding per-request token limits. This implementation splits batches into
//! chunks of [`BATCH_CHUNK_SIZE`] (100) and processes them sequentially to
//! respect rate limits.
//!
//! # Sparse embeddings
//!
//! OpenAI does not support sparse (SPLADE / BM42) embeddings. The
//! `embed_sparse` and `embed_sparse_batch` methods return
//! [`RetrievalError::Embedding`] with a descriptive message. Use Qdrant's
//! built-in sparse indexing for hybrid search instead.
//!
//! # Rate-limit handling
//!
//! HTTP 429 responses are retried with exponential backoff via the `backon`
//! crate. Other transient errors (5xx) are also retried.

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use agent_rs_core::types::Embedding;

use crate::embedder::{Embedder, SparseVector};
use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default OpenAI embedding model.
const DEFAULT_MODEL: &str = "text-embedding-3-large";

/// Default dimensionality for `text-embedding-3-large`.
const DEFAULT_DIMENSIONS: usize = 3072;

/// Maximum texts per API request chunk (conservative to stay within token limits).
const BATCH_CHUNK_SIZE: usize = 100;

/// OpenAI embeddings API endpoint.
const EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";

// ---------------------------------------------------------------------------
// API request / response types
// ---------------------------------------------------------------------------

/// Request body for the OpenAI Embeddings API.
#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
    dimensions: usize,
}

/// Top-level API response.
#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
    usage: Usage,
}

/// A single embedding entry in the response.
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Token usage reported by the API.
#[derive(Debug, Deserialize)]
struct Usage {
    total_tokens: u64,
}

/// OpenAI API error envelope.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

/// Error detail within the API error envelope.
#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAIEmbedderConfig
// ---------------------------------------------------------------------------

/// Configuration for [`OpenAIEmbedder`].
///
/// # Defaults
///
/// | Field        | Default                       |
/// |-------------|-------------------------------|
/// | `model`      | `text-embedding-3-large`      |
/// | `dimensions` | `3072`                        |
#[derive(Debug, Clone)]
pub struct OpenAIEmbedderConfig {
    /// OpenAI API key (required).
    pub api_key: String,
    /// Model name to use for embedding.
    pub model: String,
    /// Number of output dimensions.
    pub dimensions: usize,
}

impl OpenAIEmbedderConfig {
    /// Create a config with the given API key and default model/dimensions.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            dimensions: DEFAULT_DIMENSIONS,
        }
    }

    /// Override the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output dimensionality.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }
}

// ---------------------------------------------------------------------------
// OpenAIEmbedder
// ---------------------------------------------------------------------------

/// Embedding backend that calls the OpenAI Embeddings API.
///
/// ## Initialization
///
/// ```rust,no_run
/// use agent_rs_retrieval::openai_embedder::{OpenAIEmbedder, OpenAIEmbedderConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// let embedder = OpenAIEmbedder::new(OpenAIEmbedderConfig::new("sk-..."));
/// let embedding = embedder.embed_text("Hello, world!").await?;
/// println!("dims: {}", embedder.dimensions());
/// # Ok(())
/// # }
/// ```
///
/// ## Thread safety
///
/// `OpenAIEmbedder` is `Send + Sync` because `reqwest::Client` is `Send + Sync`
/// and all fields are owned, immutable data.
#[derive(Clone)]
pub struct OpenAIEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dims: usize,
}

impl std::fmt::Debug for OpenAIEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbedder")
            .field("model", &self.model)
            .field("dims", &self.dims)
            .field("api_key", &"***")
            .finish()
    }
}

impl OpenAIEmbedder {
    /// Create a new [`OpenAIEmbedder`] with the given configuration.
    ///
    /// No network call is made during construction. The first embedding
    /// request will validate the API key.
    pub fn new(config: OpenAIEmbedderConfig) -> Self {
        let client = reqwest::Client::new();
        tracing::info!(
            model = %config.model,
            dims = config.dimensions,
            "OpenAIEmbedder created"
        );
        Self {
            client,
            api_key: config.api_key,
            model: config.model,
            dims: config.dimensions,
        }
    }

    /// Call the OpenAI Embeddings API for a single chunk of texts.
    ///
    /// Returns embeddings sorted by the original input order.
    ///
    /// Retries on HTTP 429 (rate limit) and 5xx (server error) with
    /// exponential backoff.
    #[instrument(skip(self, texts), fields(chunk_size = texts.len(), model = %self.model))]
    async fn call_api(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RetrievalError> {
        let body = EmbeddingsRequest {
            input: texts.to_vec(),
            model: &self.model,
            dimensions: self.dims,
        };

        let api_key = self.api_key.clone();
        let client = self.client.clone();
        let url = EMBEDDINGS_URL;

        // Serialize the body once for retry re-use.
        let body_json = serde_json::to_string(&body).map_err(|e| RetrievalError::Embedding {
            reason: format!("failed to serialize embeddings request: {e}"),
        })?;

        let response_bytes = (|| {
            let client = client.clone();
            let api_key = api_key.clone();
            let body_json = body_json.clone();
            async move {
                let resp = client
                    .post(url)
                    .header("Authorization", format!("Bearer {api_key}"))
                    .header("Content-Type", "application/json")
                    .body(body_json)
                    .send()
                    .await
                    .map_err(|e| RetrievalError::Embedding {
                        reason: format!("HTTP request failed: {e}"),
                    })?;

                let status = resp.status();
                let bytes = resp.bytes().await.map_err(|e| RetrievalError::Embedding {
                    reason: format!("failed to read response body: {e}"),
                })?;

                if status.is_success() {
                    Ok(bytes)
                } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    // Parse error message for logging, then return retryable error.
                    let msg = parse_api_error(&bytes, status);
                    tracing::warn!(status = %status, error = %msg, "retryable API error");
                    Err(RetrievalError::Embedding { reason: msg })
                } else {
                    // Non-retryable client error (400, 401, 403, etc.)
                    let msg = parse_api_error(&bytes, status);
                    Err(RetrievalError::Embedding { reason: msg })
                }
            }
        })
        .retry(
            ExponentialBuilder::default()
                .with_min_delay(std::time::Duration::from_millis(500))
                .with_max_delay(std::time::Duration::from_secs(30))
                .with_max_times(3),
        )
        .when(|e: &RetrievalError| {
            // Only retry rate-limit and server errors.
            match e {
                RetrievalError::Embedding { reason } => {
                    reason.contains("429")
                        || reason.contains("5xx")
                        || reason.contains("server error")
                }
                _ => false,
            }
        })
        .await?;

        // Parse the successful response.
        let mut api_resp: EmbeddingsResponse =
            serde_json::from_slice(&response_bytes).map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to parse embeddings response: {e}"),
            })?;

        tracing::debug!(
            total_tokens = api_resp.usage.total_tokens,
            embeddings_count = api_resp.data.len(),
            "OpenAI embeddings received"
        );

        // Sort by index to guarantee order matches input.
        api_resp.data.sort_by_key(|d| d.index);

        let embeddings: Vec<Vec<f32>> = api_resp.data.into_iter().map(|d| d.embedding).collect();

        if embeddings.len() != texts.len() {
            return Err(RetrievalError::Embedding {
                reason: format!(
                    "expected {} embeddings, got {}",
                    texts.len(),
                    embeddings.len()
                ),
            });
        }

        Ok(embeddings)
    }

    /// Embed a batch of texts, splitting into chunks of [`BATCH_CHUNK_SIZE`]
    /// and processing sequentially to respect rate limits.
    #[instrument(skip(self, texts), fields(total = texts.len()))]
    async fn embed_chunked(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for (chunk_idx, chunk) in texts.chunks(BATCH_CHUNK_SIZE).enumerate() {
            tracing::debug!(
                chunk = chunk_idx + 1,
                total_chunks = texts.len().div_ceil(BATCH_CHUNK_SIZE),
                chunk_size = chunk.len(),
                "processing embedding chunk"
            );

            let raw_vecs = self.call_api(chunk).await?;
            all_embeddings.extend(raw_vecs.into_iter().map(Embedding));
        }

        Ok(all_embeddings)
    }
}

// ---------------------------------------------------------------------------
// Embedder trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Embedder for OpenAIEmbedder {
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    async fn embed_text(&self, text: &str) -> Result<Embedding, RetrievalError> {
        let texts = [text];
        let mut results = self.embed_chunked(&texts[..]).await?;
        results.pop().ok_or_else(|| RetrievalError::Embedding {
            reason: "embed_text: API returned empty result".to_owned(),
        })
    }

    /// Embed a batch of query texts.
    ///
    /// OpenAI `text-embedding-3-large` handles asymmetric search natively --
    /// no "query: " prefix is needed.
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        self.embed_chunked(texts).await
    }

    /// Embed a batch of document/passage texts.
    ///
    /// OpenAI `text-embedding-3-large` handles asymmetric search natively --
    /// no "passage: " prefix is needed.
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        self.embed_chunked(texts).await
    }

    /// OpenAI does not support sparse embeddings.
    ///
    /// Returns [`RetrievalError::Embedding`]. Use Qdrant's built-in sparse
    /// indexing (BM42) for hybrid search when using the OpenAI embedder.
    #[instrument(skip(self, _text))]
    async fn embed_sparse(&self, _text: &str) -> Result<SparseVector, RetrievalError> {
        Err(RetrievalError::Embedding {
            reason: "OpenAI does not support sparse embeddings; use Qdrant built-in sparse indexing for hybrid search".to_owned(),
        })
    }

    /// OpenAI does not support sparse embeddings.
    ///
    /// Returns [`RetrievalError::Embedding`]. Use Qdrant's built-in sparse
    /// indexing (BM42) for hybrid search when using the OpenAI embedder.
    #[instrument(skip(self, _texts))]
    async fn embed_sparse_batch(
        &self,
        _texts: &[&str],
    ) -> Result<Vec<SparseVector>, RetrievalError> {
        Err(RetrievalError::Embedding {
            reason: "OpenAI does not support sparse embeddings; use Qdrant built-in sparse indexing for hybrid search".to_owned(),
        })
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an OpenAI API error response into a human-readable message.
fn parse_api_error(bytes: &[u8], status: StatusCode) -> String {
    match serde_json::from_slice::<ApiErrorResponse>(bytes) {
        Ok(err_resp) => {
            let mut msg = format!(
                "OpenAI API error (HTTP {}): {}",
                status.as_u16(),
                err_resp.error.message
            );
            if let Some(ref code) = err_resp.error.code {
                msg.push_str(&format!(" [code: {code}]"));
            }
            if let Some(ref error_type) = err_resp.error.error_type {
                msg.push_str(&format!(" [type: {error_type}]"));
            }
            msg
        }
        Err(_) => {
            let body_preview = String::from_utf8_lossy(&bytes[..bytes.len().min(200)]);
            format!(
                "OpenAI API error (HTTP {}): {}",
                status.as_u16(),
                body_preview
            )
        }
    }
}
