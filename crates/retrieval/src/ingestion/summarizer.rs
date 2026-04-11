//! LLM-based chunk summarization for the ingestion pipeline.
//!
//! Each ingested chunk is summarized to 1-2 sentences via the OpenAI chat
//! completions API, then its summary embedding is stored in a dedicated
//! Qdrant collection. This enables the [`RetrievalStrategy::Summaries`]
//! search path: queries are matched against concise semantic summaries rather
//! than raw chunk text, yielding broader / coarser recall with lower noise.
//!
//! # Design
//!
//! - Uses the same `/v1/chat/completions` endpoint and `backon`-based retry
//!   pattern as [`super::graph_extractor::LlmGraphExtractor`].
//! - Plain-text completion (no structured output required; summaries are
//!   free-form text). Temperature `0.0` for determinism.
//! - Concurrency-bounded via [`tokio::sync::Semaphore`], matching the batch
//!   extraction pattern in `graph_extractor.rs`.
//! - UUID v5 summary IDs derived from `{chunk_id}+"summary"` ensure repeated
//!   ingestion of the same chunk produces idempotent upserts (mirrors Cognee's
//!   `uuid5(chunk.id, "TextSummary")`).
//!
//! # Integration
//!
//! [`IngestionPipeline`](super::pipeline::IngestionPipeline) accepts an optional
//! `Arc<ChunkSummarizer>` via
//! [`IngestionPipelineConfig`](super::pipeline::IngestionPipelineConfig).
//! When present and a `summaries_collection` is also configured, the pipeline
//! summarises every chunk, embeds the summary text, and upserts the embeddings
//! into the summaries collection after the main chunk upsert.

use std::sync::Arc;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::instrument;

use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default OpenAI model for summarization. Chosen for cost/latency.
const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// OpenAI chat completions endpoint.
const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default retry budget for transient failures.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default concurrency for batch summarization calls.
pub const DEFAULT_SUMMARIZER_CONCURRENCY: usize = 5;

/// Maximum characters forwarded to the LLM for summarization.
/// Very large chunks are truncated at a character boundary.
const MAX_CHUNK_CHARS: usize = 8_000;

/// Preview length stored in the payload for debugging / display.
const PREVIEW_CHARS: usize = 200;

/// Namespace UUID used for deterministic summary ID generation
/// (UUID v5 over `{chunk_id}|summary`).
const SUMMARY_UUID_NAMESPACE: uuid::Uuid = uuid::Uuid::NAMESPACE_OID;

/// System prompt for chunk summarization.
const SYSTEM_PROMPT: &str = "Summarize the following text in 1-2 sentences, focusing on the key concepts and entities. \
     Be concise and factual. Return only the summary — no preamble, no labels.";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The summary produced for a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSummary {
    /// The chunk that was summarized.
    pub chunk_id: String,
    /// Deterministic UUID v5 derived from `{chunk_id}|summary`.
    ///
    /// Used as the Qdrant point id so repeated ingestion of the same chunk
    /// produces an idempotent upsert.
    pub summary_id: String,
    /// The 1-2 sentence summary produced by the LLM.
    pub summary_text: String,
    /// First [`PREVIEW_CHARS`] characters of the source chunk for inspection.
    pub chunk_text_preview: String,
}

// ---------------------------------------------------------------------------
// OpenAI wire types (minimal — plain text completion)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: Option<String>,
    #[allow(dead_code)]
    code: Option<String>,
}

// ---------------------------------------------------------------------------
// ChunkSummarizer
// ---------------------------------------------------------------------------

/// LLM-based chunk summarizer.
///
/// Cheap to clone (inner state is `Arc`-wrapped). Designed to be constructed
/// once at startup and shared across the ingestion pipeline.
#[derive(Clone)]
pub struct ChunkSummarizer {
    client: reqwest::Client,
    api_key: Arc<String>,
    model: Arc<String>,
    max_retries: u32,
}

impl std::fmt::Debug for ChunkSummarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkSummarizer")
            .field("model", &self.model)
            .field("max_retries", &self.max_retries)
            .field("api_key", &"***")
            .finish()
    }
}

impl ChunkSummarizer {
    /// Create a new summarizer with the given OpenAI API key and model.
    pub fn new(api_key: String, model: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        tracing::info!(
            model = %model,
            max_retries = DEFAULT_MAX_RETRIES,
            "ChunkSummarizer created"
        );
        Self {
            client,
            api_key: Arc::new(api_key),
            model: Arc::new(model),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Create a summarizer using the default model (`gpt-4o-mini`).
    pub fn with_default_model(api_key: String) -> Self {
        Self::new(api_key, DEFAULT_MODEL.to_owned())
    }

    /// Override the retry budget. The default is [`DEFAULT_MAX_RETRIES`].
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Generate a concise 1-2 sentence summary for a single chunk of text.
    ///
    /// Returns an empty string for empty / whitespace-only input without
    /// making an API call.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Ingestion`] on non-retryable transport or
    /// authentication failures. Malformed responses are treated as soft
    /// failures: the method logs a WARN and returns an empty summary string.
    #[instrument(skip(self, chunk_text), fields(chars = chunk_text.len(), model = %self.model))]
    pub async fn summarize(&self, chunk_text: &str) -> Result<String, RetrievalError> {
        let trimmed = chunk_text.trim();
        if trimmed.is_empty() {
            tracing::debug!("skipping empty chunk for summarization");
            return Ok(String::new());
        }

        // Truncate very large chunks at a character boundary.
        let payload_text: &str = if trimmed.chars().count() > MAX_CHUNK_CHARS {
            let mut end = 0usize;
            for (i, (byte_idx, _)) in trimmed.char_indices().enumerate() {
                if i == MAX_CHUNK_CHARS {
                    break;
                }
                end = byte_idx;
            }
            tracing::warn!(
                original_bytes = trimmed.len(),
                truncated_to = end,
                "chunk exceeds MAX_CHUNK_CHARS; truncating for summarization"
            );
            &trimmed[..end]
        } else {
            trimmed
        };

        let body = ChatRequest {
            model: self.model.as_str(),
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: SYSTEM_PROMPT,
                },
                ChatMessage {
                    role: "user",
                    content: payload_text,
                },
            ],
            temperature: 0.0,
        };

        let body_bytes = serde_json::to_vec(&body).map_err(|e| RetrievalError::Ingestion {
            reason: format!("serialize summarization request: {e}"),
        })?;

        let response_bytes = self.call_api_with_retry(body_bytes).await?;

        let chat: ChatResponse = match serde_json::from_slice(&response_bytes) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to parse summarization response; returning empty summary"
                );
                return Ok(String::new());
            }
        };

        let summary = chat
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        let trimmed_summary = summary.trim().to_owned();
        tracing::debug!(
            summary_chars = trimmed_summary.len(),
            "summarization complete for chunk"
        );
        Ok(trimmed_summary)
    }

    /// Generate summaries for a batch of `(chunk_id, chunk_text)` pairs with
    /// bounded concurrency.
    ///
    /// The returned `Vec` has the same length and ordering as `chunks`.
    /// A failed or empty summary entry records an empty `summary_text` so the
    /// caller can decide whether to skip upsert for that chunk.
    ///
    /// `concurrency` bounds the number of in-flight LLM requests and should
    /// be tuned to match provider rate limits.
    #[instrument(skip(self, chunks), fields(total = chunks.len(), concurrency))]
    pub async fn summarize_batch(
        &self,
        chunks: &[(&str, &str)], // (chunk_id, chunk_text)
        concurrency: usize,
    ) -> Vec<Result<ChunkSummary, RetrievalError>> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let limit = concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(limit));
        let mut handles = Vec::with_capacity(chunks.len());

        for (idx, &(chunk_id, chunk_text)) in chunks.iter().enumerate() {
            let summarizer = self.clone();
            let chunk_id_owned = chunk_id.to_owned();
            let chunk_text_owned = chunk_text.to_owned();
            let permit_sem = Arc::clone(&semaphore);

            let handle = tokio::spawn(async move {
                let _permit = match permit_sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(e) => {
                        return (
                            idx,
                            Err(RetrievalError::Ingestion {
                                reason: format!("semaphore closed during summarization: {e}"),
                            }),
                        );
                    }
                };

                let result = summarizer.summarize(&chunk_text_owned).await.map(|text| {
                    let key = format!("{chunk_id_owned}|summary");
                    let summary_id =
                        uuid::Uuid::new_v5(&SUMMARY_UUID_NAMESPACE, key.as_bytes()).to_string();
                    let preview: String = chunk_text_owned.chars().take(PREVIEW_CHARS).collect();
                    ChunkSummary {
                        chunk_id: chunk_id_owned.clone(),
                        summary_id,
                        summary_text: text,
                        chunk_text_preview: preview,
                    }
                });
                (idx, result)
            });
            handles.push(handle);
        }

        // Collect into an ordered vec.
        let mut results: Vec<Result<ChunkSummary, RetrievalError>> = (0..chunks.len())
            .map(|_| {
                Err(RetrievalError::Ingestion {
                    reason: "summarization task did not complete".to_owned(),
                })
            })
            .collect();

        for handle in handles {
            match handle.await {
                Ok((idx, result)) => {
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = result;
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "summarization task panicked");
                }
            }
        }

        results
    }

    /// POST to `/v1/chat/completions` with retry on 429 / 5xx.
    async fn call_api_with_retry(&self, body_bytes: Vec<u8>) -> Result<Vec<u8>, RetrievalError> {
        let client = self.client.clone();
        let api_key = Arc::clone(&self.api_key);
        let body_arc = Arc::new(body_bytes);

        let op = move || {
            let client = client.clone();
            let api_key = Arc::clone(&api_key);
            let body = Arc::clone(&body_arc);
            async move {
                let resp = client
                    .post(CHAT_COMPLETIONS_URL)
                    .header("Authorization", format!("Bearer {}", api_key.as_str()))
                    .header("Content-Type", "application/json")
                    .body((*body).clone())
                    .send()
                    .await
                    .map_err(|e| RetrievalError::Ingestion {
                        reason: format!("summarizer HTTP request failed: {e}"),
                    })?;

                let status = resp.status();
                let bytes = resp.bytes().await.map_err(|e| RetrievalError::Ingestion {
                    reason: format!("read summarizer response body: {e}"),
                })?;

                if status.is_success() {
                    Ok(bytes.to_vec())
                } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    let msg = parse_api_error(&bytes, status);
                    tracing::warn!(status = %status, error = %msg, "retryable summarizer API error");
                    Err(RetrievalError::Ingestion { reason: msg })
                } else {
                    let msg = parse_api_error(&bytes, status);
                    Err(RetrievalError::Ingestion { reason: msg })
                }
            }
        };

        op.retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_millis(500))
                .with_max_delay(Duration::from_secs(30))
                .with_max_times(self.max_retries as usize),
        )
        .when(is_retryable_error)
        .await
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decide whether an ingestion error should be retried.
fn is_retryable_error(e: &RetrievalError) -> bool {
    match e {
        RetrievalError::Ingestion { reason } => {
            reason.contains("HTTP 429")
                || reason.contains("HTTP 5")
                || reason.contains("server error")
        }
        _ => false,
    }
}

/// Extract a human-readable error message from an OpenAI error response body.
fn parse_api_error(bytes: &[u8], status: StatusCode) -> String {
    match serde_json::from_slice::<ApiErrorResponse>(bytes) {
        Ok(err) => format!("HTTP {status}: {}", err.error.message),
        Err(_) => {
            let preview: String = String::from_utf8_lossy(bytes).chars().take(200).collect();
            format!("HTTP {status}: {preview}")
        }
    }
}
