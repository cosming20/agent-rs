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
use std::sync::Arc;
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

/// Default confidence threshold above which evidence is considered sufficient.
///
/// When the sufficiency check returns a confidence >= this value the loop exits
/// early without consuming the remaining iteration budget.
pub const DEFAULT_SUFFICIENCY_CONFIDENCE_THRESHOLD: f64 = 0.8;

/// OpenAI Chat Completions endpoint.
const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Payload key inside a [`SearchResult`] that carries the chunk body.
const CHUNK_TEXT_KEY: &str = "chunk_text";

/// Maximum context characters embedded in the LLM prompt.
///
/// Sized for gpt-5.4-mini (200K token context). We pass the full reranked
/// evidence so the LLM has complete context for grounding decisions.
const MAX_CONTEXT_CHARS: usize = 60_000;

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

/// System prompt for the evidence sufficiency evaluation stage.
///
/// This prompt is designed to produce a structured JSON assessment of whether
/// the retrieved evidence is sufficient to fully answer the user's query.
/// It handles several common hard cases explicitly:
///
/// - **Comparative queries**: both sides of the comparison must be present.
/// - **Negation / exception queries**: the rule AND its exceptions must appear.
/// - **Multi-hop reasoning**: the full reasoning chain must be traceable.
///
/// The `refined_query` field must be a targeted retrieval query (not a
/// conversational follow-up) aimed at filling the exact gap identified.
const SUFFICIENCY_SYSTEM_PROMPT: &str = "\
You are an expert evidence evaluator for a retrieval-augmented generation system. \
Your task is to assess whether a set of retrieved evidence chunks provides \
sufficient information to fully and accurately answer the user's original query.\n\
\n\
Evaluation guidelines:\n\
1. COMPLETENESS: The evidence must cover ALL aspects of the query, not just part of it.\n\
2. COMPARATIVE QUERIES: Both (or all) sides of any comparison must be present in the evidence. \
   If the query asks 'compare X and Y', evidence about only X is insufficient.\n\
3. NEGATION & EXCEPTIONS: If the query involves a rule, the evidence must include both \
   the rule and any relevant exceptions mentioned in the query.\n\
4. MULTI-HOP REASONING: For queries requiring a chain of reasoning (A → B → C), \
   ALL steps in the chain must be supported by evidence.\n\
5. SPECIFICITY: Generic background information is not sufficient if the query asks \
   for specific facts, numbers, dates, or named entities.\n\
6. CONFIDENCE SCORING: Score 0.0 (no useful evidence) to 1.0 (complete, unambiguous answer). \
   Use 0.8+ only if the evidence fully covers the query with no material gaps.\n\
\n\
Return a JSON object with these exact fields:\n\
- sufficient (boolean): true only if confidence >= 0.8 and no material gaps remain.\n\
- confidence (number 0.0-1.0): your confidence that the evidence answers the query fully.\n\
- missing_info (string): describe specifically what information is still needed, \
  or 'none' if sufficient.\n\
- refined_query (string or null): if not sufficient, provide a SPECIFIC retrieval query \
  that targets the exact missing information. This must be a search query for a \
  vector/graph retrieval system, not a conversational question. Return null if sufficient.\n\
- reasoning (string): one to two sentences explaining why the evidence is or is not \
  sufficient, referencing specific gaps or coverage.";

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
    /// Confidence threshold (0.0–1.0) above which the loop exits early.
    ///
    /// When `evaluate_sufficiency` returns a confidence >= this value the
    /// retrieval loop terminates without consuming the remaining iteration
    /// budget. Defaults to [`DEFAULT_SUFFICIENCY_CONFIDENCE_THRESHOLD`].
    pub sufficiency_confidence_threshold: f64,
    /// Whether to run the sufficiency check after each retrieval pass.
    ///
    /// When `false` the loop always runs all `max_iterations` and returns
    /// the merged pool without LLM sufficiency checks, which is equivalent
    /// to the original follow-up-question-only behaviour.
    pub enable_sufficiency_check: bool,
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
            sufficiency_confidence_threshold: DEFAULT_SUFFICIENCY_CONFIDENCE_THRESHOLD,
            enable_sufficiency_check: true,
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

    /// Override the sufficiency confidence threshold.
    ///
    /// The value is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn with_sufficiency_confidence_threshold(mut self, threshold: f64) -> Self {
        self.sufficiency_confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Disable LLM-based sufficiency checks.
    ///
    /// When disabled the loop always runs all `max_iterations` using
    /// follow-up questions only (original behaviour).
    #[must_use]
    pub fn with_sufficiency_check_disabled(mut self) -> Self {
        self.enable_sufficiency_check = false;
        self
    }
}

// ---- Public sufficiency types ----

/// Structured result of an LLM evidence sufficiency check.
///
/// Returned by [`GraphCotRetriever::evaluate_sufficiency`] and embedded in
/// each [`IterationLog`] produced by the enhanced `retrieve` loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SufficiencyCheck {
    /// `true` if the evidence is considered sufficient to answer the query.
    pub sufficient: bool,
    /// Confidence in the sufficiency assessment (0.0 = none, 1.0 = certain).
    pub confidence: f64,
    /// Description of what information is still needed, or `"none"` if sufficient.
    pub missing_info: String,
    /// A targeted retrieval query that would obtain the missing information,
    /// or `None` if the evidence is already sufficient.
    pub refined_query: Option<String>,
    /// Short explanation of the assessment.
    pub reasoning: String,
}

impl SufficiencyCheck {
    /// Fallback instance used when the LLM call fails.
    ///
    /// Assumes the evidence is sufficient at moderate confidence so the
    /// caller is never blocked by an LLM outage.
    fn fallback() -> Self {
        Self {
            sufficient: true,
            confidence: 0.5,
            missing_info: "unknown — sufficiency check failed".to_owned(),
            refined_query: None,
            reasoning:
                "LLM sufficiency check failed; assuming sufficient to avoid blocking the caller."
                    .to_owned(),
        }
    }
}

/// Per-iteration diagnostics logged by the enhanced `retrieve` loop.
#[derive(Debug, Clone)]
pub struct IterationLog {
    /// 1-based iteration index.
    pub iteration: usize,
    /// The query submitted to the retrieval function this iteration.
    pub query_used: String,
    /// Number of new (deduplicated) chunks returned.
    pub chunks_added: usize,
    /// Sufficiency assessment for this iteration, if the check was enabled.
    pub sufficiency: Option<SufficiencyCheck>,
}

// ---- Retriever ----

/// Iterative chain-of-thought retriever.
///
/// Wraps a reqwest client and the configured LLM credentials. Cheap to clone.
#[derive(Clone)]
pub struct GraphCotRetriever {
    client: Arc<reqwest::Client>,
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
    pub fn new(
        config: GraphCotConfig,
        client: Arc<reqwest::Client>,
    ) -> Result<Self, RetrievalError> {
        if config.llm_api_key.is_empty() {
            return Err(RetrievalError::Routing {
                strategy: "GraphCoT".to_owned(),
                reason: "llm_api_key is empty".to_owned(),
            });
        }

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
    /// Builds a shared HTTP client with a 20-second request timeout.
    /// Returns `None` if no API key is set.
    #[must_use]
    pub fn try_from_env() -> Option<Result<Self, RetrievalError>> {
        use crate::http::build_http_client;
        std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("LLM_API_KEY"))
            .ok()
            .map(|k| {
                let client = build_http_client(HTTP_TIMEOUT)?;
                Self::new(GraphCotConfig::new(k), client)
            })
    }

    /// Evaluate whether the retrieved evidence is sufficient to answer the query.
    ///
    /// Calls the OpenAI Chat Completions API with a structured JSON schema
    /// designed for sufficiency assessment. The check handles comparative
    /// queries, negation/exception queries, and multi-hop reasoning chains.
    ///
    /// # Arguments
    ///
    /// - `original_query` — the user's original query.
    /// - `evidence_texts` — the evidence text snippets to evaluate.
    ///
    /// # Returns
    ///
    /// A [`SufficiencyCheck`] with confidence, missing-info description, and
    /// an optional refined retrieval query. On any LLM failure, returns a
    /// fallback instance that assumes sufficiency at moderate confidence so
    /// the caller is never blocked by an outage.
    #[instrument(skip(self, evidence_texts), fields(query_len = original_query.len(), evidence_count = evidence_texts.len()))]
    pub async fn evaluate_sufficiency(
        &self,
        original_query: &str,
        evidence_texts: &[&str],
    ) -> Result<SufficiencyCheck, RetrievalError> {
        let evidence_block = evidence_texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("[{}] {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n");

        // Guard against excessively large prompts by truncating the evidence block.
        let truncated: &str = if evidence_block.len() > MAX_CONTEXT_CHARS {
            let cut = evidence_block
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= MAX_CONTEXT_CHARS)
                .last()
                .unwrap_or(MAX_CONTEXT_CHARS);
            &evidence_block[..cut]
        } else {
            &evidence_block
        };

        let user_content = format!(
            "Original query:\n{original_query}\n\nRetrieved evidence:\n{truncated}\n\n\
             Evaluate whether this evidence is sufficient to fully answer the query."
        );

        let request = ChatRequest {
            model: &self.config.llm_model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: SUFFICIENCY_SYSTEM_PROMPT,
                },
                ChatMessage {
                    role: "user",
                    content: &user_content,
                },
            ],
            max_completion_tokens: self.config.llm_max_tokens,
            temperature: 0.0,
            seed: crate::llm_determinism::RANKING_LLM_SEED,
            response_format: ResponseFormat::sufficiency_schema(),
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
                reason: format!("sufficiency check HTTP request failed: {e}"),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_else(|_| "<no body>".into());
            return Err(RetrievalError::Embedding {
                reason: format!("sufficiency check API error (HTTP {status}): {body_text}"),
            });
        }

        let parsed: ChatResponse = resp.json().await.map_err(|e| RetrievalError::Embedding {
            reason: format!("failed to parse sufficiency check response: {e}"),
        })?;

        let content_str = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| RetrievalError::Embedding {
                reason: "sufficiency check response missing choices[0].message.content".to_owned(),
            })?;

        let payload: SufficiencySchemaPayload =
            serde_json::from_str(content_str).map_err(|e| RetrievalError::Embedding {
                reason: format!("failed to parse sufficiency structured output: {e}"),
            })?;

        Ok(SufficiencyCheck {
            sufficient: payload.sufficient,
            confidence: payload.confidence.clamp(0.0, 1.0),
            missing_info: payload.missing_info,
            refined_query: payload.refined_query.filter(|q| !q.trim().is_empty()),
            reasoning: payload.reasoning,
        })
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
    ///
    /// # Sufficiency checks
    ///
    /// When [`GraphCotConfig::enable_sufficiency_check`] is `true` (default),
    /// after each retrieval pass the LLM evaluates whether the accumulated
    /// evidence is sufficient:
    ///
    /// - If `sufficient == true` **or** `confidence >= sufficiency_confidence_threshold`:
    ///   the loop exits immediately.
    /// - Otherwise, the `refined_query` from the sufficiency check (if present)
    ///   is used as the next iteration's query instead of the follow-up question
    ///   from the CoT validator, targeting the exact information gap.
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

        // Queries already issued through `retrieve_fn` — the loop skips a
        // follow-up whose query string has been seen before, avoiding the
        // embed + seed-derivation + graph-traversal round-trip that would
        // rediscover the same candidates. Saves ~0.5–1 s per redundant
        // iteration on stable-seeds cases.
        let mut queries_issued: HashSet<String> = HashSet::new();
        queries_issued.insert(question.to_owned());

        // Per-iteration diagnostics, exposed via debug logging.
        let mut iteration_logs: Vec<IterationLog> = Vec::with_capacity(self.config.max_iterations);

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

            // ---- Sufficiency check (pre-iteration) ----
            //
            // Evaluate after the initial retrieval pass (or after each
            // additional pass) before deciding whether to issue another query.
            // We run this BEFORE calling the CoT validator so that if the
            // evidence is already sufficient we skip the follow-up entirely.
            let sufficiency = if self.config.enable_sufficiency_check {
                let evidence_texts: Vec<&str> = merged
                    .iter()
                    .filter_map(|r| r.payload.get(CHUNK_TEXT_KEY).and_then(|v| v.as_str()))
                    .collect();

                let check = match self.evaluate_sufficiency(question, &evidence_texts).await {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(
                            iteration = iteration,
                            error = %e,
                            "GraphCoT: sufficiency check failed; using fallback (assume sufficient)"
                        );
                        SufficiencyCheck::fallback()
                    }
                };

                debug!(
                    iteration = iteration,
                    sufficient = check.sufficient,
                    confidence = check.confidence,
                    missing_info = %check.missing_info,
                    refined_query = ?check.refined_query,
                    "GraphCoT: sufficiency check result"
                );

                let is_sufficient = check.sufficient
                    || check.confidence >= self.config.sufficiency_confidence_threshold;

                if is_sufficient {
                    debug!(
                        iteration = iteration,
                        stop_reason = "sufficient",
                        confidence = check.confidence,
                        merged_count = merged.len(),
                        "GraphCoT: evidence deemed sufficient; exiting loop early"
                    );
                    // Record the log entry before returning.
                    iteration_logs.push(IterationLog {
                        iteration,
                        query_used: question.to_owned(),
                        chunks_added: 0,
                        sufficiency: Some(check),
                    });
                    debug!(
                        total_iterations = iteration_logs.len(),
                        "GraphCoT: iteration history"
                    );
                    return merged;
                }

                Some(check)
            } else {
                None
            };

            // ---- CoT validator ----
            //
            // Ask the LLM whether the question is answered and what follow-up
            // question to pose next. If a sufficiency check produced a
            // `refined_query`, we prefer that over the CoT follow-up.
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
                iteration_logs.push(IterationLog {
                    iteration,
                    query_used: question.to_owned(),
                    chunks_added: 0,
                    sufficiency,
                });
                debug!(
                    total_iterations = iteration_logs.len(),
                    "GraphCoT: iteration history"
                );
                return merged;
            }

            // Determine the next query: prefer the sufficiency check's
            // `refined_query` (targets the exact gap) over the CoT follow-up.
            let next_query: Option<String> = sufficiency
                .as_ref()
                .and_then(|s| s.refined_query.clone())
                .filter(|q| !q.trim().is_empty())
                .or_else(|| {
                    decision
                        .follow_up_question
                        .as_deref()
                        .map(str::trim)
                        .filter(|q| !q.is_empty())
                        .map(str::to_owned)
                });

            let Some(next_query) = next_query else {
                debug!(
                    iteration = iteration,
                    stop_reason = "no_followup",
                    merged_count = merged.len(),
                    tokens_used = decision.tokens_used,
                    "GraphCoT: no refined query or follow-up available"
                );
                iteration_logs.push(IterationLog {
                    iteration,
                    query_used: question.to_owned(),
                    chunks_added: 0,
                    sufficiency,
                });
                return merged;
            };

            let used_refined = sufficiency
                .as_ref()
                .and_then(|s| s.refined_query.as_deref())
                .map(|rq| rq == next_query.as_str())
                .unwrap_or(false);

            debug!(
                iteration = iteration,
                next_query = %next_query,
                used_refined_query = used_refined,
                "GraphCoT: selected next query for follow-up retrieval"
            );

            // Early-exit: if this exact query was already issued in a prior
            // iteration (or is the original question), the retrieval would
            // deterministically return the same candidates. Short-circuit.
            if !queries_issued.insert(next_query.clone()) {
                debug!(
                    iteration = iteration,
                    stop_reason = "query_repeated",
                    next_query = %next_query,
                    merged_count = merged.len(),
                    "GraphCoT: query already issued; skipping redundant follow-up"
                );
                iteration_logs.push(IterationLog {
                    iteration,
                    query_used: next_query,
                    chunks_added: 0,
                    sufficiency,
                });
                return merged;
            }

            // Issue the follow-up retrieval. Errors are logged and the iteration
            // is treated as a no-op rather than failing the whole loop.
            let new_results = match retrieve_fn(next_query.clone()).await {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        iteration = iteration,
                        next_query = %next_query,
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
                next_query = %next_query,
                used_refined_query = used_refined,
                candidates_added = added,
                tokens_used = decision.tokens_used,
                "GraphCoT: iteration complete"
            );

            iteration_logs.push(IterationLog {
                iteration,
                query_used: next_query,
                chunks_added: added,
                sufficiency,
            });

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
            total_iterations = iteration_logs.len(),
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
            max_completion_tokens: self.config.llm_max_tokens,
            temperature: 0.0,
            seed: crate::llm_determinism::RANKING_LLM_SEED,
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

/// Chat Completions request body.
///
/// The token cap is serialised as `max_completion_tokens`, the name required
/// by gpt-5.x and the o-series reasoning models. The legacy `max_tokens`
/// field was deprecated for Chat Completions in favour of this one, and
/// newer models reject requests that send the old name.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_completion_tokens: u32,
    temperature: f32,
    /// Ranking-signal determinism: the sufficiency check and CoT decision
    /// both feed into which candidates enter the evidence pack, so we pin
    /// the seed. Best-effort on reasoning models; exact on others.
    seed: u32,
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

    fn sufficiency_schema() -> Self {
        Self {
            kind: "json_schema",
            json_schema: JsonSchemaSpec {
                name: "sufficiency_check",
                strict: true,
                schema: sufficiency_check_schema(),
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

/// JSON schema for the structured sufficiency-check response.
///
/// Matches [`SufficiencySchemaPayload`] exactly.
fn sufficiency_check_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["sufficient", "confidence", "missing_info", "refined_query", "reasoning"],
        "properties": {
            "sufficient": {
                "type": "boolean",
                "description": "True only if confidence >= 0.8 and the evidence fully covers all aspects of the query."
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 (no useful evidence) to 1.0 (complete, unambiguous answer)."
            },
            "missing_info": {
                "type": "string",
                "description": "Specific description of what information is still needed, or 'none' if sufficient."
            },
            "refined_query": {
                "type": ["string", "null"],
                "description": "A targeted retrieval query for a vector/graph search system that would obtain the missing information. Must be a search query, not a conversational question. Null if sufficient."
            },
            "reasoning": {
                "type": "string",
                "description": "One to two sentences explaining why the evidence is or is not sufficient, referencing specific gaps or coverage."
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

/// Deserialization target for the sufficiency check structured output.
#[derive(Debug, Deserialize)]
struct SufficiencySchemaPayload {
    sufficient: bool,
    confidence: f64,
    missing_info: String,
    refined_query: Option<String>,
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
