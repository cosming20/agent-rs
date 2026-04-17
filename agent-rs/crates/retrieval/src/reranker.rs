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
use std::time::{Duration, Instant};

use agent_rs_core::Score;
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

/// Default Jina reranker model (multilingual, 278M params).
const DEFAULT_MODEL: &str = "jina-reranker-v2-base-multilingual";

/// Jina rerank API endpoint.
const JINA_RERANK_URL: &str = "https://api.jina.ai/v1/rerank";

/// Maximum concurrent rerank calls across all queries. Prevents a thundering
/// herd from slamming Jina on traffic spikes and triggering a rate-limit
/// cascade that would degrade every in-flight query.
const DEFAULT_MAX_CONCURRENCY: usize = 8;

/// Per-request timeout. Shorter than the provider's inherent limit: the
/// reranker is optional and the router falls back cleanly on failure, so we
/// prefer to bail fast and keep the overall query latency bounded.
const RERANK_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

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
///
/// The api_key is held in an `Arc<str>` so that retry-closure clones are
/// pointer-sized instead of full-string copies. The concurrency semaphore
/// is shared across every clone of this struct, so global concurrency is
/// enforced regardless of how many `Arc<CrossEncoderReranker>` references
/// the router holds.
#[derive(Clone)]
pub struct CrossEncoderReranker {
    client: reqwest::Client,
    api_key: Arc<str>,
    model: Arc<str>,
    semaphore: Arc<Semaphore>,
}

impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("model", &self.model)
            .field("api_key", &"***")
            .field(
                "available_permits",
                &self.semaphore.available_permits(),
            )
            .finish()
    }
}

impl CrossEncoderReranker {
    /// Construct a reranker with the given Jina API key and default model.
    ///
    /// Reads from `JINA_API_KEY` env var if `api_key` is empty.
    pub fn new(api_key: String) -> Result<Self, RetrievalError> {
        Self::with_max_concurrency(api_key, DEFAULT_MAX_CONCURRENCY)
    }

    /// Construct a reranker with an explicit max-concurrency cap.
    pub fn with_max_concurrency(
        api_key: String,
        max_concurrency: usize,
    ) -> Result<Self, RetrievalError> {
        let api_key = if api_key.is_empty() {
            std::env::var("JINA_API_KEY").map_err(|_| RetrievalError::Embedding {
                reason: "JINA_API_KEY not set and no api_key provided".to_owned(),
            })?
        } else {
            api_key
        };

        let client = reqwest::Client::builder()
            .timeout(RERANK_REQUEST_TIMEOUT)
            .pool_max_idle_per_host(max_concurrency)
            .build()
            .map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to build reqwest client: {e}"),
            })?;

        let permits = max_concurrency.max(1);
        tracing::info!(
            model = %DEFAULT_MODEL,
            max_concurrency = permits,
            timeout_s = RERANK_REQUEST_TIMEOUT.as_secs(),
            "CrossEncoderReranker initialised (Jina API)"
        );

        Ok(Self {
            client,
            api_key: Arc::<str>::from(api_key),
            model: Arc::<str>::from(DEFAULT_MODEL),
            semaphore: Arc::new(Semaphore::new(permits)),
        })
    }

    /// Use a non-default Jina reranker model (e.g. "jina-reranker-v1-turbo-en").
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Arc::<str>::from(model);
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
    ///
    /// Documents with empty `chunk_text` are filtered out before the API call
    /// (the cross-encoder would return spurious low scores for empty input,
    /// wasting API budget and polluting the ranking). The filtered docs are
    /// appended to the end of the result in their original order so they
    /// still surface if `top_k` is large enough.
    ///
    /// Requests are serialised through a workspace-wide semaphore to cap
    /// concurrent calls to Jina; this prevents traffic spikes from triggering
    /// a rate-limit cascade that would degrade every concurrent query.
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

        // Partition documents by whether they have text to rerank. Indexing
        // preserves original order for the skipped set, which is appended at
        // the end of the output if room remains after the reranked batch.
        let (candidates, skipped): (Vec<(usize, SearchResult, String)>, Vec<SearchResult>) =
            documents.into_iter().fold(
                (Vec::new(), Vec::new()),
                |(mut kept, mut skipped), doc| {
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
                },
            );

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
        let request = RerankRequest {
            model: &self.model,
            query,
            documents: text_refs,
            top_n: top_k.min(candidates.len()),
            return_documents: false,
        };

        let body = serde_json::to_vec(&request).map_err(|e| RetrievalError::Embedding {
            reason: format!("failed to serialize rerank request: {e}"),
        })?;
        // Ref-counted body so retry-closure clones are pointer-sized, not a
        // full memcpy of the JSON payload (can be 10-50KB for 50 candidates).
        let body: Arc<[u8]> = Arc::from(body);

        // Bound total rerank concurrency. Held for the duration of the retry
        // loop to ensure a single query never occupies more than one permit.
        let _permit =
            self.semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| RetrievalError::Embedding {
                    reason: format!("rerank semaphore closed: {e}"),
                })?;

        let started = Instant::now();
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
                        .post(JINA_RERANK_URL)
                        .bearer_auth(api_key.as_ref())
                        .header("Content-Type", "application/json")
                        .body(body.to_vec())
                        .send()
                        .await
                        .map_err(|e| RetrievalError::Embedding {
                            reason: format!("jina rerank HTTP request failed: {e}"),
                        })?;

                    let status = resp.status();
                    if !status.is_success() {
                        let body_text = resp.text().await.unwrap_or_else(|_| "<no body>".into());
                        // Log only status + truncated body; Jina error messages can echo
                        // request fragments which may include PII from chunk text.
                        let snippet = body_text.chars().take(256).collect::<String>();
                        let reason = format!("jina rerank API error (HTTP {status}): {snippet}");
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
                    "jina reranker rate-limited; retrying after backoff"
                );
            })
            .await?
        };
        let latency_ms = started.elapsed().as_millis() as u64;

        // Map Jina's returned indices back onto the original SearchResult objects.
        let n = candidates.len();
        let mut doc_slots: Vec<Option<SearchResult>> =
            candidates.into_iter().map(|(_, doc, _)| Some(doc)).collect();

        let mut reranked: Vec<SearchResult> = Vec::with_capacity(top_k);
        let mut dropped_oor = 0usize;
        for result in rerank_response.results {
            if result.index >= n {
                // Soft-skip: one bad index shouldn't fail the whole rerank.
                // Router already has a hard-fallback path for total failure.
                dropped_oor += 1;
                tracing::warn!(
                    index = result.index,
                    candidates = n,
                    "jina returned out-of-range index; skipping"
                );
                continue;
            }
            if let Some(mut doc) = doc_slots[result.index].take() {
                // Score::new coerces non-finite to 0.0 (post-Phase A), so a
                // malformed score never poisons RRF's unwrap_or(Equal) sort.
                doc.score = Score::new(result.relevance_score);
                reranked.push(doc);
                if reranked.len() >= top_k {
                    break;
                }
            }
        }

        // Backfill from empty-text skipped docs if the caller asked for more
        // than Jina could score. Preserves original ordering.
        if reranked.len() < top_k {
            for doc in skipped {
                if reranked.len() >= top_k {
                    break;
                }
                reranked.push(doc);
            }
        }

        tracing::info!(
            input_count = n,
            output_count = reranked.len(),
            dropped_out_of_range = dropped_oor,
            latency_ms,
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
