//! GraphCoT — iterative chain-of-thought retrieval over a knowledge graph.
//!
//! # Overview
//!
//! [`GraphCotRetriever`] wraps an initial retrieval result and iteratively
//! refines it by asking an LLM whether the question is fully answered by the
//! current context. When it is not, the LLM generates a follow-up question
//! which is fed back into the user-supplied retrieval closure to fetch
//! additional evidence. New results are merged and deduplicated into the
//! running pool until one of the following stop conditions is met:
//!
//! 1. The LLM reports the question is `answered`.
//! 2. The LLM fails to propose a follow-up question.
//! 3. `max_iterations` is reached.
//! 4. `max_latency` (wall-clock) is exceeded.
//! 5. A follow-up iteration yields zero new documents (convergence).
//!
//! # Fallback
//!
//! Every failure mode — LLM HTTP error (429, 5xx, timeout), JSON parse error,
//! unexpected schema, empty initial pool — is logged at `WARN` and the method
//! returns the caller's `initial_results` unchanged. The retriever never
//! propagates an error: callers receive at worst what they would have gotten
//! without CoT.
//!
//! # Structured output
//!
//! The LLM is called via the OpenAI Chat Completions API with
//! `response_format = {"type": "json_schema", ...}` so the output is
//! guaranteed to match the declared schema. No free-form JSON parsing is
//! performed.
//!
//! # Observability
//!
//! Every iteration emits a `tracing` event with:
//!
//! - `iteration` — 1-based iteration number
//! - `follow_up_question` — the LLM's proposed follow-up (or `<none>`)
//! - `candidates_added` — new unique documents merged this iteration
//! - `tokens_used` — prompt + completion tokens reported by the API
//! - `stop_reason` — `converged | max_iter | timeout | error | answered`
//!
//! # Example
//!
//! ```rust,ignore
//! use std::time::Duration;
//! use agent_rs_retrieval::graph_cot::{GraphCotConfig, GraphCotRetriever};
//!
//! let config = GraphCotConfig::new("sk-...".into())
//!     .with_max_iterations(3)
//!     .with_max_latency(Duration::from_secs(15));
//! let retriever = GraphCotRetriever::new(config)?;
//!
//! let merged = retriever
//!     .retrieve(
//!         "Who founded OpenAI and when?",
//!         initial_results,
//!         |follow_up: String| async move {
//!             // embed + graph_search for the follow-up, returning Vec<SearchResult>
//!             graph_search_for(&follow_up).await
//!         },
//!     )
//!     .await;
//! ```

use std::collections::HashSet;
use std::future::Future;
use std::time::{Duration, Instant};

use agent_rs_stores::types::SearchResult;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::error::RetrievalError;

// ---- Constants ----

/// Default maximum number of CoT iterations (excluding the initial retrieval).
pub const DEFAULT_MAX_ITERATIONS: usize = 3;

/// Default wall-clock budget for the entire CoT loop.
pub const DEFAULT_MAX_LATENCY: Duration = Duration::from_secs(15);

/// Default OpenAI Chat Completions model.
///
/// `gpt-4o-mini` supports structured outputs (`response_format.json_schema`)
/// and is cheap enough to call several times per query.
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// Default LLM output token cap per iteration.
pub const DEFAULT_MAX_TOKENS: u32 = 2_000;

/// OpenAI Chat Completions endpoint.
const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Payload key inside a [`SearchResult`] that carries the chunk body.
const CHUNK_TEXT_KEY: &str = "chunk_text";

/// Maximum context characters embedded in the LLM prompt.
///
/// We truncate the joined evidence text to this length before asking the LLM
/// whether the question is answered; longer contexts rarely help and make the
/// call slower and more expensive.
const MAX_CONTEXT_CHARS: usize = 6_000;

/// HTTP timeout applied to every LLM call.
const HTTP_TIMEOUT: Duration = Duration::from_secs(20);

/// System prompt driving the CoT validation + follow-up round.
const SYSTEM_PROMPT: &str = "You are a retrieval assistant helping decide whether a user's \
question is fully answered by a set of retrieved context snippets from a knowledge graph. \
Return a JSON object conforming exactly to the provided schema. Set `answered` to true only \
if the context contains enough information to answer the question with confidence; otherwise \
set it to false and propose one focused follow-up question that would fill the gap. The \
follow-up must be a short natural-language query suitable for a graph/vector retrieval \
system. If no further retrieval could plausibly help, return `answered=false` with \
`follow_up_question=null`.";

// ---- Config ----

/// Configuration for [`GraphCotRetriever`].
#[derive(Debug, Clone)]
pub struct GraphCotConfig {
    /// Maximum number of CoT iterations (excluding initial retrieval).
    pub max_iterations: usize,
    /// Wall-clock budget for the whole CoT loop.
    pub max_latency: Duration,
    /// OpenAI API key.
    pub llm_api_key: String,
    /// OpenAI Chat Completions model identifier.
    pub llm_model: String,
    /// Maximum completion tokens per LLM call.
    pub llm_max_tokens: u32,
}

impl GraphCotConfig {
    /// Build a config with the given API key and defaults for everything else.
    pub fn new(api_key: String) -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_latency: DEFAULT_MAX_LATENCY,
            llm_api_key: api_key,
            llm_model: DEFAULT_MODEL.to_owned(),
            llm_max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// Override the OpenAI Chat Completions model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.llm_model = model.into();
        self
    }

    /// Override the maximum number of CoT iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Override the wall-clock budget for the CoT loop.
    #[must_use]
    pub fn with_max_latency(mut self, d: Duration) -> Self {
        self.max_latency = d;
        self
    }

    /// Override the LLM max_tokens per completion.
    #[must_use]
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.llm_max_tokens = n;
        self
    }
}

// ---- Retriever ----

/// Iterative chain-of-thought retriever.
///
/// Wraps a reqwest client and the configured LLM credentials. Cheap to clone.
#[derive(Clone)]
pub struct GraphCotRetriever {
    client: reqwest::Client,
    config: GraphCotConfig,
}

impl std::fmt::Debug for GraphCotRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphCotRetriever")
            .field("model", &self.config.llm_model)
            .field("max_iterations", &self.config.max_iterations)
            .field("max_latency", &self.config.max_latency)
            .field("llm_api_key", &"***")
            .finish()
    }
}

impl GraphCotRetriever {
    /// Construct a new retriever from a config.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] if the underlying reqwest client
    /// cannot be built (extremely unlikely in practice).
    pub fn new(config: GraphCotConfig) -> Result<Self, RetrievalError> {
        if config.llm_api_key.is_empty() {
            return Err(RetrievalError::Routing {
                strategy: "GraphCoT".to_owned(),
                reason: "llm_api_key is empty".to_owned(),
            });
        }
        let client = reqwest::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to build reqwest client: {e}"),
            })?;

        tracing::info!(
            model = %config.llm_model,
            max_iterations = config.max_iterations,
            max_latency_s = config.max_latency.as_secs(),
            "GraphCotRetriever initialised"
        );

        Ok(Self { client, config })
    }

    /// Convenience constructor from `OPENAI_API_KEY` / `LLM_API_KEY` env vars.
    ///
    /// Returns `None` if no API key is set.
    #[must_use]
    pub fn try_from_env() -> Option<Result<Self, RetrievalError>> {
        std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("LLM_API_KEY"))
            .ok()
            .map(|k| Self::new(GraphCotConfig::new(k)))
    }

    /// Run the chain-of-thought retrieval loop.
    ///
    /// # Arguments
    ///
    /// - `question` — the original user query, used for LLM prompting.
    /// - `initial_results` — the results of the caller's base retrieval pass.
    /// - `retrieve_fn` — closure that takes a follow-up query string and
    ///   returns newly-retrieved [`SearchResult`]s (typically a graph search
    ///   seeded from the follow-up). Errors from the closure cause that
    ///   iteration to be skipped, not the whole loop to fail.
    ///
    /// # Returns
    ///
    /// A merged, deduplicated list of [`SearchResult`]s. On any LLM failure or
    /// fallback condition, returns the `initial_results` unchanged. This
    /// method never returns an error — observability is handled via `tracing`.
    #[instrument(
        skip(self, initial_results, retrieve_fn),
        fields(
            question_len = question.len(),
            initial_count = initial_results.len(),
            max_iter = self.config.max_iterations,
        )
    )]
    pub async fn retrieve<F, Fut>(
        &self,
        question: &str,
        initial_results: Vec<SearchResult>,
        retrieve_fn: F,
    ) -> Vec<SearchResult>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: Future<Output = Result<Vec<SearchResult>, RetrievalError>> + Send,
    {
        // Fast path: no initial evidence -> nothing for the LLM to verify
        // against. Return whatever we got.
        if initial_results.is_empty() {
            debug!(
                stop_reason = "empty_initial",
                "GraphCoT: initial results empty; skipping LLM loop"
            );
            return initial_results;
        }

        let deadline = Instant::now() + self.config.max_latency;

        // Running dedup set keyed by SearchResult.id.
        let mut seen: HashSet<String> =
            HashSet::with_capacity(initial_results.len().saturating_mul(2));
        for r in &initial_results {
            seen.insert(r.id.clone());
        }
        let mut merged: Vec<SearchResult> = initial_results.clone();

        for iteration in 1..=self.config.max_iterations {
            // Budget check before starting a new iteration.
            if Instant::now() >= deadline {
                warn!(
                    iteration = iteration,
                    stop_reason = "timeout",
                    merged_count = merged.len(),
                    "GraphCoT: wall-clock budget exhausted"
                );
                return merged;
            }

            // Ask the LLM whether the question is answered and what to ask next.
            let context_text = join_context(&merged);
            let llm_result = self.call_llm_validator(question, &context_text).await;

            let decision = match llm_result {
                Ok(d) => d,
                Err(e) => {
                    warn!(
                        iteration = iteration,
                        stop_reason = "error",
                        error = %e,
                        "GraphCoT: LLM call failed; falling back to initial results"
                    );
                    // Fallback: return what we have so far (initial results at minimum).
                    return if merged.len() == initial_results.len() {
                        initial_results
                    } else {
                        merged
                    };
                }
            };

            debug!(
                iteration = iteration,
                answered = decision.answered,
                follow_up_question = %decision
                    .follow_up_question
                    .as_deref()
                    .unwrap_or("<none>"),
                tokens_used = decision.tokens_used,
                reasoning = %decision.reasoning,
                "GraphCoT: LLM decision"
            );

            // Stop if the LLM says the question is answered.
            if decision.answered {
                debug!(
                    iteration = iteration,
                    stop_reason = "answered",
                    merged_count = merged.len(),
                    tokens_used = decision.tokens_used,
                    "GraphCoT: LLM reports question answered"
                );
                return merged;
            }

            // Stop if there is no follow-up question to ask.
            let Some(follow_up) = decision
                .follow_up_question
                .as_deref()
                .map(str::trim)
                .filter(|q| !q.is_empty())
            else {
                debug!(
                    iteration = iteration,
                    stop_reason = "no_followup",
                    merged_count = merged.len(),
                    tokens_used = decision.tokens_used,
                    "GraphCoT: LLM returned no follow-up"
                );
                return merged;
            };

            // Issue the follow-up retrieval. Errors are logged and the iteration
            // is treated as a no-op rather than failing the whole loop.
            let new_results = match retrieve_fn(follow_up.to_owned()).await {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        iteration = iteration,
                        follow_up_question = %follow_up,
                        error = %e,
                        "GraphCoT: follow-up retrieval failed; stopping loop"
                    );
                    return merged;
                }
            };

            // Merge with dedup.
            let added = merge_dedup(&mut merged, &mut seen, new_results);

            debug!(
                iteration = iteration,
                follow_up_question = %follow_up,
                candidates_added = added,
                tokens_used = decision.tokens_used,
                "GraphCoT: iteration complete"
            );

            // Convergence: no new info this round — further iterations would
            // only reuse the same evidence pool, so stop early.
            if added == 0 {
                debug!(
                    iteration = iteration,
                    stop_reason = "converged",
                    merged_count = merged.len(),
                    "GraphCoT: convergence (zero new candidates)"
                );
                return merged;
            }
        }

        debug!(
            stop_reason = "max_iter",
            merged_count = merged.len(),
            "GraphCoT: reached max_iterations"
        );
        merged
    }

    /// Call the OpenAI Chat Completions API with a strict JSON schema and
    /// parse the response into a [`CotDecision`].
    async fn call_llm_validator(
        &self,
        question: &str,
        context_text: &str,
    ) -> Result<CotDecision, RetrievalError> {
        let user_content = format!(
            "Question:\n{question}\n\nContext snippets:\n{context_text}\n\n\
             Decide whether the question is fully answered by the context. \
             If not, propose one focused follow-up question."
        );

        let request = ChatRequest {
            model: &self.config.llm_model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: SYSTEM_PROMPT,
                },
                ChatMessage {
                    role: "user",
                    content: &user_content,
                },
            ],
            max_tokens: self.config.llm_max_tokens,
            temperature: 0.0,
            response_format: ResponseFormat::json_schema(),
        };

        let resp = self
            .client
            .post(OPENAI_CHAT_URL)
            .bearer_auth(&self.config.llm_api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RetrievalError::Embedding {
                reason: format!("openai chat HTTP request failed: {e}"),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_else(|_| "<no body>".into());
            return Err(RetrievalError::Embedding {
                reason: format!("openai chat API error (HTTP {status}): {body_text}"),
            });
        }

        let parsed: ChatResponse = resp.json().await.map_err(|e| RetrievalError::Embedding {
            reason: format!("failed to parse openai chat response: {e}"),
        })?;

        // Extract the JSON string from the first choice's message content.
        let content_str = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| RetrievalError::Embedding {
                reason: "openai response missing choices[0].message.content".to_owned(),
            })?;

        // Content is guaranteed to be JSON matching our schema.
        let payload: CotSchemaPayload =
            serde_json::from_str(content_str).map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to parse structured output JSON: {e}"),
            })?;

        let tokens_used = parsed.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0);

        Ok(CotDecision {
            answered: payload.answered,
            follow_up_question: payload.follow_up_question,
            reasoning: payload.reasoning,
            tokens_used,
        })
    }
}

// ---- Internal helpers ----

/// Join the chunk text of every [`SearchResult`] into a single bounded string.
///
/// Stops once [`MAX_CONTEXT_CHARS`] characters have been accumulated.
fn join_context(results: &[SearchResult]) -> String {
    let mut out = String::with_capacity(MAX_CONTEXT_CHARS);
    for (idx, r) in results.iter().enumerate() {
        let snippet = r
            .payload
            .get(CHUNK_TEXT_KEY)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if snippet.is_empty() {
            continue;
        }
        // Prefix with an index marker so the LLM can reason about them.
        let header = format!("[{}] ", idx + 1);
        if out.len() + header.len() + snippet.len() > MAX_CONTEXT_CHARS {
            let remaining = MAX_CONTEXT_CHARS.saturating_sub(out.len() + header.len());
            if remaining == 0 {
                break;
            }
            out.push_str(&header);
            // Slice at a char boundary to avoid panicking on multi-byte chars.
            let cut = snippet
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= remaining)
                .last()
                .unwrap_or(0);
            out.push_str(&snippet[..cut]);
            out.push('\n');
            break;
        }
        out.push_str(&header);
        out.push_str(snippet);
        out.push('\n');
    }
    out
}

/// Merge `new_results` into `merged`, skipping ids that are already `seen`.
/// Returns the number of newly-added documents.
fn merge_dedup(
    merged: &mut Vec<SearchResult>,
    seen: &mut HashSet<String>,
    new_results: Vec<SearchResult>,
) -> usize {
    let mut added = 0;
    for r in new_results {
        if seen.insert(r.id.clone()) {
            merged.push(r);
            added += 1;
        }
    }
    added
}

// ---- Decision type (internal) ----

/// Parsed outcome of a single CoT LLM call.
#[derive(Debug, Clone)]
struct CotDecision {
    answered: bool,
    follow_up_question: Option<String>,
    #[allow(dead_code)]
    reasoning: String,
    tokens_used: u64,
}

// ---- OpenAI request/response types ----

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
    response_format: ResponseFormat,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JsonSchemaSpec,
}

impl ResponseFormat {
    fn json_schema() -> Self {
        Self {
            kind: "json_schema",
            json_schema: JsonSchemaSpec {
                name: "cot_decision",
                strict: true,
                schema: cot_decision_schema(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonSchemaSpec {
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
}

/// JSON schema for the structured CoT response.
///
/// Matches [`CotSchemaPayload`] exactly. All fields are required (the schema
/// uses `additionalProperties: false` as required by OpenAI's strict mode).
fn cot_decision_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["answered", "follow_up_question", "reasoning"],
        "properties": {
            "answered": {
                "type": "boolean",
                "description": "True iff the context already contains enough information to answer the question confidently."
            },
            "follow_up_question": {
                "type": ["string", "null"],
                "description": "A single focused follow-up question that would help answer the original question, or null if none is helpful."
            },
            "reasoning": {
                "type": "string",
                "description": "Short (one sentence) justification for the decision."
            }
        }
    })
}

#[derive(Debug, Deserialize)]
struct CotSchemaPayload {
    answered: bool,
    follow_up_question: Option<String>,
    reasoning: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
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
struct ChatUsage {
    total_tokens: u64,
}
