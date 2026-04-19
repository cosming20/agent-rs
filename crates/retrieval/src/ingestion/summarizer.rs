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
///
/// Bumped from 5 to 20 — matches KG extraction concurrency. Reduces
/// chunk summarization from ~164s to ~42s for 412 chunks.
pub const DEFAULT_SUMMARIZER_CONCURRENCY: usize = 20;

/// Maximum characters forwarded to the LLM for summarization.
/// Very large chunks are truncated at a character boundary.
///
/// Sized well above the chunker's default (~4K chars per chunk) so this
/// bound almost never fires in practice. gpt-5.4-mini handles 200K context.
const MAX_CHUNK_CHARS: usize = 40_000;

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

/// Outcome of a per-chunk summary attempt. Recorded in the summaries payload
/// so the outline tool can surface `[summary unavailable]` instead of silently
/// hiding chunks whose summary failed to generate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SummaryStatus {
    /// LLM returned a non-empty summary.
    Ok,
    /// LLM returned an empty / whitespace-only response.
    Empty,
    /// LLM call failed (after retries).
    Failed,
}

impl SummaryStatus {
    /// Serde default used for legacy payloads that pre-date this field.
    fn default_ok() -> Self {
        Self::Ok
    }
}

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
    /// Document id of the source chunk; populated by `summarize_batch`
    /// from the chunk metadata. `None` only for legacy summaries from
    /// before this field was added.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// Section id of the source chunk; same provenance / nullability as
    /// `doc_id`.
    #[serde(default)]
    pub section_id: Option<String>,
    /// Ingest run that produced this summary; used by rollback to delete
    /// orphan summaries when an ingest run fails.
    #[serde(default)]
    pub ingest_run_id: Option<String>,
    /// Outcome marker (`Ok` / `Empty` / `Failed`). Replaces the previous
    /// silent-drop behaviour where empty/failed summaries vanished.
    #[serde(default = "SummaryStatus::default_ok")]
    pub summary_status: SummaryStatus,
}

/// Metadata supplied per chunk to [`ChunkSummarizer::summarize_batch`].
///
/// Carries identifiers that are propagated into the resulting [`ChunkSummary`]
/// payload so downstream tools (rollback, outline corpus) can join on them.
pub struct SummarizeItem<'a> {
    /// Unique identifier for this chunk.
    pub chunk_id: &'a str,
    /// Raw text content to summarize.
    pub chunk_text: &'a str,
    /// Parent document identifier; forwarded into [`ChunkSummary::doc_id`].
    pub doc_id: Option<&'a str>,
    /// Section identifier (e.g. a Neo4j section UUID); forwarded into
    /// [`ChunkSummary::section_id`]. May be `None` when section structure is
    /// not available (e.g. plain-text ingestion path).
    pub section_id: Option<&'a str>,
    /// Ingest run identifier; forwarded into [`ChunkSummary::ingest_run_id`]
    /// so rollback can delete orphan summaries by run.
    pub ingest_run_id: Option<&'a str>,
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
    #[serde(default)]
    usage: Option<Usage>,
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
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
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
    client: Arc<reqwest::Client>,
    api_key: Arc<String>,
    model: Arc<String>,
    max_retries: u32,
    metrics: Option<Arc<agent_rs_observability::metrics::RetrievalMetrics>>,
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
    pub fn new(api_key: String, model: String, client: Arc<reqwest::Client>) -> Self {
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
            metrics: None,
        }
    }

    /// Create a summarizer using the default model (`gpt-4o-mini`).
    pub fn with_default_model(api_key: String, client: Arc<reqwest::Client>) -> Self {
        Self::new(api_key, DEFAULT_MODEL.to_owned(), client)
    }

    /// Override the retry budget. The default is [`DEFAULT_MAX_RETRIES`].
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Attach a metrics handle so per-call cost / tokens / latency are
    /// emitted to the OTel pipeline.
    pub fn with_metrics(
        mut self,
        metrics: Arc<agent_rs_observability::metrics::RetrievalMetrics>,
    ) -> Self {
        self.metrics = Some(metrics);
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

        let llm_started = std::time::Instant::now();
        let response_bytes = self.call_api_with_retry(body_bytes).await?;
        let llm_elapsed_ms = llm_started.elapsed().as_secs_f64() * 1000.0;

        let chat: ChatResponse = match serde_json::from_slice(&response_bytes) {
            Ok(c) => c,
            Err(e) => {
                if let Some(m) = self.metrics.as_ref() {
                    m.record_llm_failure("summarizer", "malformed_json");
                }
                tracing::warn!(
                    error = %e,
                    "failed to parse summarization response; returning empty summary"
                );
                return Ok(String::new());
            }
        };

        if let (Some(m), Some(u)) = (self.metrics.as_ref(), chat.usage.as_ref()) {
            m.record_llm_call(
                "summarizer",
                self.model.as_str(),
                u.prompt_tokens,
                u.completion_tokens,
                llm_elapsed_ms,
            );
        }

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

    /// Generate summaries for a batch of [`SummarizeItem`]s with bounded concurrency.
    ///
    /// The returned `Vec` has the same length and ordering as `items`.
    /// Every entry is `Ok(ChunkSummary)` — failures and empty responses are
    /// recorded as `SummaryStatus::Failed` / `SummaryStatus::Empty` respectively
    /// rather than being dropped, so callers can distinguish "never tried" from
    /// "tried and got nothing".
    ///
    /// `concurrency` bounds the number of in-flight LLM requests and should
    /// be tuned to match provider rate limits.
    #[instrument(skip(self, items), fields(total = items.len(), concurrency))]
    pub async fn summarize_batch<'a>(
        &self,
        items: &[SummarizeItem<'a>],
        concurrency: usize,
    ) -> Vec<ChunkSummary> {
        if items.is_empty() {
            return Vec::new();
        }

        let limit = concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(limit));
        let mut handles = Vec::with_capacity(items.len());

        for (idx, item) in items.iter().enumerate() {
            let summarizer = self.clone();
            let chunk_id_owned = item.chunk_id.to_owned();
            let chunk_text_owned = item.chunk_text.to_owned();
            let doc_id_owned = item.doc_id.map(str::to_owned);
            let section_id_owned = item.section_id.map(str::to_owned);
            let ingest_run_id_owned = item.ingest_run_id.map(str::to_owned);
            let permit_sem = Arc::clone(&semaphore);

            let handle = tokio::spawn(async move {
                let _permit = match permit_sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "semaphore closed during summarization");
                        let key = format!("{chunk_id_owned}|summary");
                        let summary_id =
                            uuid::Uuid::new_v5(&SUMMARY_UUID_NAMESPACE, key.as_bytes())
                                .to_string();
                        let preview: String =
                            chunk_text_owned.chars().take(PREVIEW_CHARS).collect();
                        return (
                            idx,
                            ChunkSummary {
                                chunk_id: chunk_id_owned,
                                summary_id,
                                summary_text: String::new(),
                                chunk_text_preview: preview,
                                doc_id: doc_id_owned,
                                section_id: section_id_owned,
                                ingest_run_id: ingest_run_id_owned,
                                summary_status: SummaryStatus::Failed,
                            },
                        );
                    }
                };

                let key = format!("{chunk_id_owned}|summary");
                let summary_id =
                    uuid::Uuid::new_v5(&SUMMARY_UUID_NAMESPACE, key.as_bytes()).to_string();
                let preview: String = chunk_text_owned.chars().take(PREVIEW_CHARS).collect();

                let (summary_text, summary_status) =
                    match summarizer.summarize(&chunk_text_owned).await {
                        Ok(text) if text.is_empty() => (String::new(), SummaryStatus::Empty),
                        Ok(text) => (text, SummaryStatus::Ok),
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                chunk_id = %chunk_id_owned,
                                "chunk summarization failed"
                            );
                            (String::new(), SummaryStatus::Failed)
                        }
                    };

                (
                    idx,
                    ChunkSummary {
                        chunk_id: chunk_id_owned,
                        summary_id,
                        summary_text,
                        chunk_text_preview: preview,
                        doc_id: doc_id_owned,
                        section_id: section_id_owned,
                        ingest_run_id: ingest_run_id_owned,
                        summary_status,
                    },
                )
            });
            handles.push(handle);
        }

        // Collect into an ordered vec. Slots are pre-filled with a Failed
        // sentinel so any panicked task produces a recorded failure instead
        // of a gap in the output.
        let item_count = items.len();
        let mut results: Vec<Option<ChunkSummary>> = (0..item_count).map(|_| None).collect();

        for handle in handles {
            match handle.await {
                Ok((idx, summary)) => {
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = Some(summary);
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "summarization task panicked");
                }
            }
        }

        // Any slot still None means the spawn panicked — unwrap to a Failed
        // sentinel using an empty chunk_id (should be extremely rare).
        results
            .into_iter()
            .enumerate()
            .map(|(idx, opt)| {
                opt.unwrap_or_else(|| ChunkSummary {
                    chunk_id: format!("unknown-{idx}"),
                    summary_id: String::new(),
                    summary_text: String::new(),
                    chunk_text_preview: String::new(),
                    doc_id: None,
                    section_id: None,
                    ingest_run_id: None,
                    summary_status: SummaryStatus::Failed,
                })
            })
            .collect()
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_summary_serializes_doc_section_ingest_run_id_and_status() {
        let summary = ChunkSummary {
            summary_id: "uuid-s".into(),
            chunk_id: "uuid-c".into(),
            summary_text: "...".into(),
            chunk_text_preview: String::new(),
            doc_id: Some("d1".into()),
            section_id: Some("sec1".into()),
            ingest_run_id: Some("run-7".into()),
            summary_status: SummaryStatus::Ok,
        };
        let json = serde_json::to_value(&summary).expect("serialize");
        assert_eq!(json["doc_id"], "d1");
        assert_eq!(json["section_id"], "sec1");
        assert_eq!(json["ingest_run_id"], "run-7");
        assert_eq!(json["summary_status"], "ok");
    }

    #[test]
    fn summary_status_empty_serializes_correctly() {
        let status = SummaryStatus::Empty;
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json, "empty");
    }

    #[test]
    fn summary_status_failed_serializes_correctly() {
        let status = SummaryStatus::Failed;
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json, "failed");
    }

    #[test]
    fn chunk_summary_deserializes_with_serde_defaults() {
        let json = serde_json::json!({
            "chunk_id": "c1",
            "summary_id": "s1",
            "summary_text": "text",
            "chunk_text_preview": "preview",
        });
        let summary: ChunkSummary = serde_json::from_value(json).expect("deserialize");
        assert_eq!(summary.doc_id, None);
        assert_eq!(summary.section_id, None);
        assert_eq!(summary.ingest_run_id, None);
        assert_eq!(summary.summary_status, SummaryStatus::Ok);
    }
}
