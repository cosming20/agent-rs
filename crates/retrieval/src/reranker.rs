//! Cross-encoder reranking stage for the retrieval pipeline.
//!
//! # Overview
//!
//! [`CrossEncoderReranker`] takes the top-N results from RRF fusion and
//! reranks them against the original query using a hosted cross-encoder API.
//! Unlike bi-encoder vector similarity (used in the primary retrieval stage),
//! a cross-encoder scores query+document pairs jointly, producing more
//! accurate relevance estimates.
//!
//! # Providers
//!
//! Two providers are supported behind a single struct, selected via
//! [`agent_rs_core::RerankerProvider`]:
//!
//! - **Jina** (`jina-reranker-v2-base-multilingual`): 278M-param cross-encoder,
//!   multilingual (100+ languages), raw similarity scores (not calibrated).
//!   Endpoint: `https://api.jina.ai/v1/rerank`.
//! - **ZeroEntropy** (`zerank-2`): instruction-following multilingual
//!   reranker, returns **calibrated** relevance scores — enables
//!   `score_threshold` filtering to drop low-signal candidates before the
//!   agent sees them. Endpoint: `https://api.zeroentropy.dev/v1/models/rerank`.
//!   Supports a `latency` parameter (`fast` sub-second vs `slow` >10s with
//!   much higher rate limits).
//!
//! # Integration
//!
//! The reranker is optional on [`crate::router::RetrievalRouter`]. When
//! configured, `retrieve` passes the top-50 RRF results through
//! [`CrossEncoderReranker::rerank`] before truncating to the caller's `limit`.
//! Router already falls back to RRF-ranked results on total reranker failure,
//! so switching providers carries low blast radius.

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_rs_core::{LatencyMode, RerankerConfig, RerankerProvider, Score};
use agent_rs_stores::types::SearchResult;
use backon::{ExponentialBuilder, Retryable};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::instrument;

use crate::error::RetrievalError;

// ---- Constants ----

/// Number of top RRF results forwarded to the cross-encoder.
pub const RERANK_CANDIDATE_COUNT: usize = 50;

/// Payload key that carries the chunk text used as the reranker document body.
const CHUNK_TEXT_KEY: &str = "chunk_text";

/// Per-request timeout. Shorter than the provider's inherent limit: the
/// reranker is optional and the router falls back cleanly on failure, so we
/// prefer to bail fast and keep the overall query latency bounded.
const RERANK_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

// ---- Provider-specific constants ----

const JINA_URL: &str = "https://api.jina.ai/v1/rerank";
const JINA_DEFAULT_MODEL: &str = "jina-reranker-v2-base-multilingual";
const JINA_ENV_KEY: &str = "JINA_API_KEY";

const ZEROENTROPY_URL: &str = "https://api.zeroentropy.dev/v1/models/rerank";
const ZEROENTROPY_DEFAULT_MODEL: &str = "zerank-2";
const ZEROENTROPY_ENV_KEY: &str = "ZEROENTROPY_API_KEY";

// ---- Request/Response types ----

/// Request body shared across providers. `latency` is only serialized for
/// ZeroEntropy (and only in `fast`/`slow` variants — `None` = provider
/// default, which is "auto" = fast with slow-fallback on rate limit).
#[derive(Debug, Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
    /// Jina-only: omit `return_documents` to save bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
    /// ZeroEntropy-only latency hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    latency: Option<&'static str>,
}

/// Common response shape: both providers return `{results: [{index, relevance_score}, ...]}`.
/// ZeroEntropy additionally returns `total_tokens`, `total_bytes`,
/// `actual_latency_mode`, `e2e_latency`, `inference_latency` — all captured
/// in `extra` for observability without tying the struct to one provider.
#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
    #[serde(default, flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f64,
}

// ---- CrossEncoderReranker ----

/// Cross-encoder reranker backed by a hosted API (Jina or ZeroEntropy).
///
/// The api_key and model are held in `Arc<str>` so that retry-closure clones
/// are pointer-sized instead of full-string copies. The concurrency semaphore
/// is shared across every clone of this struct, so global concurrency is
/// enforced regardless of how many `Arc<CrossEncoderReranker>` references
/// the router holds.
#[derive(Clone)]
pub struct CrossEncoderReranker {
    client: reqwest::Client,
    api_key: Arc<str>,
    model: Arc<str>,
    provider: RerankerProvider,
    latency_mode: LatencyMode,
    score_threshold: Option<f64>,
    semaphore: Arc<Semaphore>,
}

impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &"***")
            .field("latency_mode", &self.latency_mode)
            .field("score_threshold", &self.score_threshold)
            .field("available_permits", &self.semaphore.available_permits())
            .finish()
    }
}

impl CrossEncoderReranker {
    /// Construct a Jina reranker with the given API key — backward-compatible
    /// with prior API. Reads `JINA_API_KEY` env var if `api_key` is empty.
    pub fn new(api_key: String) -> Result<Self, RetrievalError> {
        Self::from_config(&RerankerConfig {
            provider: RerankerProvider::Jina,
            api_key,
            ..Default::default()
        })
    }

    /// Construct from a full config struct — the preferred entry point.
    pub fn from_config(config: &RerankerConfig) -> Result<Self, RetrievalError> {
        let (default_model, env_key) = match config.provider {
            RerankerProvider::Jina => (JINA_DEFAULT_MODEL, JINA_ENV_KEY),
            RerankerProvider::ZeroEntropy => (ZEROENTROPY_DEFAULT_MODEL, ZEROENTROPY_ENV_KEY),
        };

        let api_key = if config.api_key.is_empty() {
            std::env::var(env_key).map_err(|_| RetrievalError::Embedding {
                reason: format!("{env_key} not set and no api_key provided in config"),
            })?
        } else {
            config.api_key.clone()
        };

        let model = config
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_owned());

        let max_concurrency = config.max_concurrency.max(1);
        let client = reqwest::Client::builder()
            .timeout(RERANK_REQUEST_TIMEOUT)
            .pool_max_idle_per_host(max_concurrency)
            .build()
            .map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to build reqwest client: {e}"),
            })?;

        tracing::info!(
            provider = ?config.provider,
            model = %model,
            max_concurrency,
            timeout_s = RERANK_REQUEST_TIMEOUT.as_secs(),
            latency_mode = ?config.latency_mode,
            score_threshold = ?config.score_threshold,
            "CrossEncoderReranker initialised"
        );

        Ok(Self {
            client,
            api_key: Arc::<str>::from(api_key),
            model: Arc::<str>::from(model),
            provider: config.provider,
            latency_mode: config.latency_mode,
            score_threshold: config.score_threshold,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        })
    }

    /// Use a non-default model for the currently-selected provider.
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Arc::<str>::from(model);
        self
    }

    /// Try to construct from env vars: prefers `ZEROENTROPY_API_KEY` when set,
    /// falls back to `JINA_API_KEY`. Returns `None` if neither is set.
    pub fn try_from_env() -> Option<Result<Self, RetrievalError>> {
        if std::env::var(ZEROENTROPY_ENV_KEY).is_ok() {
            Some(Self::from_config(&RerankerConfig {
                provider: RerankerProvider::ZeroEntropy,
                ..Default::default()
            }))
        } else if std::env::var(JINA_ENV_KEY).is_ok() {
            Some(Self::from_config(&RerankerConfig {
                provider: RerankerProvider::Jina,
                ..Default::default()
            }))
        } else {
            None
        }
    }

    fn api_url(&self) -> &'static str {
        match self.provider {
            RerankerProvider::Jina => JINA_URL,
            RerankerProvider::ZeroEntropy => ZEROENTROPY_URL,
        }
    }

    /// Return the `latency` field value for this request, if any.
    /// Jina ignores the field; ZeroEntropy uses it to pick its rate-limit tier.
    fn latency_param(&self, override_mode: Option<LatencyMode>) -> Option<&'static str> {
        if !matches!(self.provider, RerankerProvider::ZeroEntropy) {
            return None;
        }
        let mode = override_mode.unwrap_or(self.latency_mode);
        Some(match mode {
            LatencyMode::Fast => "fast",
            LatencyMode::Slow => "slow",
        })
    }

    /// Rerank documents by semantic relevance to `query`. Uses the reranker's
    /// configured latency mode.
    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, RetrievalError> {
        self.rerank_with_latency(query, documents, top_k, None)
            .await
    }

    /// Rerank with an explicit latency override. Useful for the grounding-gate
    /// retry path: the first attempt uses `Fast` (subsecond, lower rate
    /// limit), and retries escalate to `Slow` (>10s with orders-of-magnitude
    /// higher rate limit). On providers without a latency tier the override
    /// is silently ignored.
    ///
    /// Documents with empty `chunk_text` are filtered out before the API call
    /// (the cross-encoder would return spurious low scores for empty input,
    /// wasting API budget and polluting the ranking). The filtered docs are
    /// appended to the end of the result in their original order so they
    /// still surface if `top_k` is large enough.
    ///
    /// Requests are serialised through a workspace-wide semaphore to cap
    /// concurrent calls to the provider; this prevents traffic spikes from
    /// triggering a rate-limit cascade that would degrade every concurrent
    /// query.
    ///
    /// When `score_threshold` is set on the reranker config, results with
    /// `relevance_score < threshold` are dropped — this relies on the
    /// provider returning calibrated scores (ZeroEntropy does; Jina does not).
    #[instrument(skip(self, documents), fields(
        provider = ?self.provider,
        query_len = query.len(),
        candidates = documents.len(),
        top_k,
    ))]
    pub async fn rerank_with_latency(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_k: usize,
        latency_override: Option<LatencyMode>,
    ) -> Result<Vec<SearchResult>, RetrievalError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Partition documents by whether they have text to rerank. Indexing
        // preserves original order for the skipped set, which is appended at
        // the end of the output if room remains after the reranked batch.
        let (candidates, skipped): (Vec<(usize, SearchResult, String)>, Vec<SearchResult>) =
            documents
                .into_iter()
                .fold((Vec::new(), Vec::new()), |(mut kept, mut skipped), doc| {
                    let text = doc
                        .payload
                        .get(CHUNK_TEXT_KEY)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                    if text.is_empty() {
                        skipped.push(doc);
                    } else {
                        kept.push((kept.len(), doc, text));
                    }
                    (kept, skipped)
                });

        if candidates.is_empty() {
            tracing::warn!(
                skipped = skipped.len(),
                "all documents had empty chunk_text; skipping reranker"
            );
            return Ok(skipped.into_iter().take(top_k).collect());
        }

        if !skipped.is_empty() {
            tracing::debug!(
                with_text = candidates.len(),
                empty_text = skipped.len(),
                "rerank: filtered docs without chunk_text"
            );
        }

        let text_refs: Vec<&str> = candidates.iter().map(|(_, _, t)| t.as_str()).collect();
        let return_documents = match self.provider {
            RerankerProvider::Jina => Some(false),
            RerankerProvider::ZeroEntropy => None,
        };
        let request = RerankRequest {
            model: &self.model,
            query,
            documents: text_refs,
            top_n: top_k.min(candidates.len()),
            return_documents,
            latency: self.latency_param(latency_override),
        };

        let body = serde_json::to_vec(&request).map_err(|e| RetrievalError::Embedding {
            reason: format!("failed to serialize rerank request: {e}"),
        })?;
        // Ref-counted body so retry-closure clones are pointer-sized, not a
        // full memcpy of the JSON payload (can be 10-50KB for 50 candidates).
        let body: Arc<[u8]> = Arc::from(body);

        // Bound total rerank concurrency. Held for the duration of the retry
        // loop to ensure a single query never occupies more than one permit.
        let _permit = self.semaphore.clone().acquire_owned().await.map_err(|e| {
            RetrievalError::Embedding {
                reason: format!("rerank semaphore closed: {e}"),
            }
        })?;

        let started = Instant::now();
        let url = self.api_url();
        let provider_label = match self.provider {
            RerankerProvider::Jina => "jina",
            RerankerProvider::ZeroEntropy => "zeroentropy",
        };

        let rerank_response = {
            let client = &self.client;
            let api_key = self.api_key.clone();
            let body = body.clone();

            let call = || {
                let client = client.clone();
                let api_key = api_key.clone();
                let body = body.clone();
                async move {
                    let resp = client
                        .post(url)
                        .bearer_auth(api_key.as_ref())
                        .header("Content-Type", "application/json")
                        .body(body.to_vec())
                        .send()
                        .await
                        .map_err(|e| RetrievalError::Embedding {
                            reason: format!("{provider_label} rerank HTTP request failed: {e}"),
                        })?;

                    let status = resp.status();
                    if !status.is_success() {
                        let body_text = resp.text().await.unwrap_or_else(|_| "<no body>".into());
                        // Log only status + truncated body; provider error messages can echo
                        // request fragments which may include PII from chunk text.
                        let snippet = body_text.chars().take(256).collect::<String>();
                        let reason =
                            format!("{provider_label} rerank API error (HTTP {status}): {snippet}");
                        if status.as_u16() == 429 || status.is_server_error() {
                            tracing::warn!(
                                provider = provider_label,
                                status = %status,
                                attempt_will_retry = true,
                                "rerank retryable error"
                            );
                        }
                        return Err(RetrievalError::Embedding { reason });
                    }

                    let parsed: RerankResponse =
                        resp.json().await.map_err(|e| RetrievalError::Embedding {
                            reason: format!(
                                "failed to parse {provider_label} rerank response: {e}"
                            ),
                        })?;

                    Ok::<_, RetrievalError>(parsed)
                }
            };

            call.retry(
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
                    "reranker rate-limited; retrying after backoff"
                );
            })
            .await?
        };
        let latency_ms = started.elapsed().as_millis() as u64;

        // Map returned indices back onto the original SearchResult objects,
        // applying the optional calibrated-score threshold. Scores below the
        // threshold are dropped BEFORE the top_k cap so we don't include
        // low-relevance docs just to fill the quota.
        let n = candidates.len();
        let mut doc_slots: Vec<Option<SearchResult>> = candidates
            .into_iter()
            .map(|(_, doc, _)| Some(doc))
            .collect();

        let mut reranked: Vec<SearchResult> = Vec::with_capacity(top_k);
        let mut dropped_oor = 0usize;
        let mut dropped_below_threshold = 0usize;
        for result in rerank_response.results {
            if result.index >= n {
                // Soft-skip: one bad index shouldn't fail the whole rerank.
                // Router already has a hard-fallback path for total failure.
                dropped_oor += 1;
                tracing::warn!(
                    provider = provider_label,
                    index = result.index,
                    candidates = n,
                    "provider returned out-of-range index; skipping"
                );
                continue;
            }

            // Apply calibrated-score threshold when configured. Only
            // meaningful on providers that return calibrated scores.
            if let Some(threshold) = self.score_threshold {
                if result.relevance_score < threshold {
                    dropped_below_threshold += 1;
                    continue;
                }
            }

            if let Some(mut doc) = doc_slots[result.index].take() {
                // Score::new coerces non-finite to 0.0, so a malformed score
                // never poisons RRF's unwrap_or(Equal) sort downstream.
                doc.score = Score::new(result.relevance_score);
                reranked.push(doc);
                if reranked.len() >= top_k {
                    break;
                }
            }
        }

        // Backfill from empty-text skipped docs if the caller asked for more
        // than the provider scored. Preserves original ordering.
        if reranked.len() < top_k {
            for doc in skipped {
                if reranked.len() >= top_k {
                    break;
                }
                reranked.push(doc);
            }
        }

        tracing::info!(
            provider = provider_label,
            input_count = n,
            output_count = reranked.len(),
            dropped_out_of_range = dropped_oor,
            dropped_below_threshold,
            threshold = ?self.score_threshold,
            latency_ms,
            extra = ?rerank_response.extra,
            "rerank complete"
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
