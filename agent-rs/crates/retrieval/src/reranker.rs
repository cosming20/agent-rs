//! Cross-encoder reranking stage for the retrieval pipeline.
//!
//! # Overview
//!
//! [`CrossEncoderReranker`] takes the top-N results from RRF fusion and
//! reranks them against the original query using Jina's hosted reranker API.
//! Unlike bi-encoder vector similarity (used in the primary retrieval stage),
//! a cross-encoder scores query+document pairs jointly, producing more
//! accurate relevance estimates.
//!
//! # Model
//!
//! Uses Jina's `jina-reranker-v2-base-multilingual` by default — a 278M
//! parameter cross-encoder with strong cross-lingual transfer. Fully
//! multilingual, handles Romanian and 100+ other languages.
//!
//! # API
//!
//! POST https://api.jina.ai/v1/rerank
//! Headers: Authorization: Bearer {JINA_API_KEY}, Content-Type: application/json
//! Body: {"model": "...", "query": "...", "documents": ["..."], "top_n": N, "return_documents": false}
//! Response: {"results": [{"index": 0, "relevance_score": 0.9}], "usage": {...}}
//!
//! # Integration
//!
//! The reranker is optional on [`crate::router::RetrievalRouter`]. When
//! configured, `retrieve` passes the top-50 RRF results through
//! [`CrossEncoderReranker::rerank`] before truncating to the caller's `limit`.

use std::sync::Arc;
use std::time::Duration;

use agent_rs_core::Score;
use agent_rs_stores::types::SearchResult;
use backon::{ExponentialBuilder, Retryable};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::RetrievalError;

// ---- Constants ----

/// Number of top RRF results forwarded to the cross-encoder.
pub const RERANK_CANDIDATE_COUNT: usize = 50;

/// Payload key that carries the chunk text used as the reranker document body.
const CHUNK_TEXT_KEY: &str = "chunk_text";

/// Default Jina reranker model (multilingual, 278M params).
const DEFAULT_MODEL: &str = "jina-reranker-v2-base-multilingual";

/// Jina rerank API endpoint.
const JINA_RERANK_URL: &str = "https://api.jina.ai/v1/rerank";

// ---- Request/Response types ----

#[derive(Debug, Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
    return_documents: bool,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f64,
}

// ---- CrossEncoderReranker ----

/// Cross-encoder reranker backed by Jina's hosted API.
#[derive(Clone)]
pub struct CrossEncoderReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("model", &self.model)
            .field("api_key", &"***")
            .finish()
    }
}

impl CrossEncoderReranker {
    /// Construct a reranker with the given Jina API key and default model.
    ///
    /// Reads from `JINA_API_KEY` env var if `api_key` is empty.
    pub fn new(api_key: String) -> Result<Self, RetrievalError> {
        let api_key = if api_key.is_empty() {
            std::env::var("JINA_API_KEY").map_err(|_| RetrievalError::Embedding {
                reason: "JINA_API_KEY not set and no api_key provided".to_owned(),
            })?
        } else {
            api_key
        };

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to build reqwest client: {e}"),
            })?;

        tracing::info!(model = %DEFAULT_MODEL, "CrossEncoderReranker initialised (Jina API)");

        Ok(Self {
            client,
            api_key,
            model: DEFAULT_MODEL.to_owned(),
        })
    }

    /// Use a non-default Jina reranker model (e.g. "jina-reranker-v1-turbo-en").
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// Try to create from env var `JINA_API_KEY`. Returns `None` if not set.
    pub fn try_from_env() -> Option<Result<Self, RetrievalError>> {
        std::env::var("JINA_API_KEY").ok().map(Self::new)
    }

    /// Rerank documents by semantic relevance to `query`.
    ///
    /// Returns up to `top_k` results sorted by descending cross-encoder score.
    /// The [`SearchResult::score`] field is replaced with the relevance score.
    #[instrument(skip(self, documents), fields(
        query_len = query.len(),
        candidates = documents.len(),
        top_k,
    ))]
    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, RetrievalError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Extract text bodies for the API. Keep the original documents around
        // so we can map scores back by index.
        let texts: Vec<String> = documents
            .iter()
            .map(|doc| {
                doc.payload
                    .get(CHUNK_TEXT_KEY)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned()
            })
            .collect();

        // Skip the API call if all documents are empty.
        if texts.iter().all(|t| t.is_empty()) {
            tracing::warn!("all documents have empty chunk_text; skipping reranker");
            return Ok(documents.into_iter().take(top_k).collect());
        }

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let request = RerankRequest {
            model: &self.model,
            query,
            documents: text_refs,
            top_n: top_k.min(documents.len()),
            return_documents: false,
        };

        let body = serde_json::to_string(&request).map_err(|e| RetrievalError::Embedding {
            reason: format!("failed to serialize rerank request: {e}"),
        })?;

        // Retry on transient errors (429, 5xx).
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let body_ref = body.clone();

        let call = || {
            let client = client.clone();
            let api_key = api_key.clone();
            let body_ref = body_ref.clone();
            async move {
                let resp = client
                    .post(JINA_RERANK_URL)
                    .bearer_auth(&api_key)
                    .header("Content-Type", "application/json")
                    .body(body_ref)
                    .send()
                    .await
                    .map_err(|e| RetrievalError::Embedding {
                        reason: format!("jina rerank HTTP request failed: {e}"),
                    })?;

                let status = resp.status();
                if !status.is_success() {
                    let body_text = resp.text().await.unwrap_or_else(|_| "<no body>".into());
                    let reason = format!("jina rerank API error (HTTP {status}): {body_text}");
                    if status.as_u16() == 429 || status.is_server_error() {
                        tracing::warn!(
                            status = %status,
                            attempt_will_retry = true,
                            "jina rerank retryable error"
                        );
                    }
                    return Err(RetrievalError::Embedding { reason });
                }

                let parsed: RerankResponse =
                    resp.json().await.map_err(|e| RetrievalError::Embedding {
                        reason: format!("failed to parse jina rerank response: {e}"),
                    })?;

                Ok::<_, RetrievalError>(parsed)
            }
        };

        let rerank_response = call
            .retry(
                ExponentialBuilder::default()
                    .with_min_delay(Duration::from_secs(2))
                    .with_max_delay(Duration::from_secs(8))
                    .with_max_times(3),
            )
            .when(|e| {
                matches!(e, RetrievalError::Embedding { reason }
                    if reason.contains("HTTP 429") || reason.contains("HTTP 5"))
            })
            .notify(|e, dur| {
                tracing::warn!(
                    error = %e,
                    retry_after = ?dur,
                    "jina reranker rate-limited; retrying after backoff"
                );
            })
            .await?;

        // Map results back onto original SearchResult objects.
        let n = documents.len();
        let mut doc_slots: Vec<Option<SearchResult>> = documents.into_iter().map(Some).collect();

        let mut reranked: Vec<SearchResult> = Vec::with_capacity(top_k);
        for result in rerank_response.results {
            if result.index >= n {
                return Err(RetrievalError::Embedding {
                    reason: format!(
                        "jina returned out-of-range index {} (candidates: {n})",
                        result.index
                    ),
                });
            }
            if let Some(mut doc) = doc_slots[result.index].take() {
                doc.score = Score::new(result.relevance_score);
                reranked.push(doc);
                if reranked.len() >= top_k {
                    break;
                }
            }
        }

        tracing::debug!(
            input_count = n,
            output_count = reranked.len(),
            usage = ?rerank_response.usage,
            "jina rerank complete"
        );

        Ok(reranked)
    }
}

// Arc<Self> compatibility is handled at the router level.
impl CrossEncoderReranker {
    /// Create an Arc-wrapped reranker — convenience for router integration.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}
