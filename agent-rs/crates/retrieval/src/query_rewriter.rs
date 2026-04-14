//! LLM-based query decomposition for the retrieval router.
//!
//! Complex or multi-faceted user queries often need more than one focused
//! retrieval pass to gather all relevant evidence. This module turns a raw
//! query into 1–3 targeted sub-queries via an OpenAI structured-output call,
//! so each sub-query can be dispatched independently and the results fused by
//! the ranker.
//!
//! # Design
//!
//! - **Structured outputs** — Uses `/v1/chat/completions` with
//!   `response_format = { type: "json_schema" }` for schema-constrained JSON.
//!   No markdown-fence parsing, no regex fragility.
//! - **Graceful fallback** — On any LLM error (network, quota, bad JSON) the
//!   module logs a warning and returns the original query verbatim as a single
//!   `Factual` sub-query. The retrieval pipeline is never blocked because
//!   rewriting failed.
//! - **Romanian legal awareness** — The system prompt includes explicit
//!   guidance for Romanian legal queries: article cross-references, crime
//!   definitions, penalty lookups, negation rewriting.
//! - **Prompt-cache friendly** — System prompt and JSON schema are static so
//!   the static prefix (≥1024 tokens) reliably hits OpenAI's automatic prompt
//!   cache, reducing cost on high-volume workloads.
//! - **Retry on transient failures** — 429 / 5xx responses are retried with
//!   exponential backoff via `backon`, matching the graph-extractor pattern.
//!
//! # Integration
//!
//! The router calls [`QueryRewriter::rewrite`] before dispatching to search
//! back-ends. The returned [`RewrittenQuery::sub_queries`] slice is iterated;
//! each [`SubQuery::text`] is embedded and searched independently. Results are
//! pooled and deduplicated by the ranker.
//!
//! ```rust,ignore
//! let rewriter = QueryRewriter::new(QueryRewriterConfig { ... });
//! let rewritten = rewriter.rewrite("Ce diferență este între furt și tâlhărie?").await?;
//! for sq in &rewritten.sub_queries {
//!     // dispatch sq.text to the retrieval back-end
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default OpenAI model for query rewriting.
///
/// Matches the model used by [`crate::ingestion::graph_extractor`] for cost
/// and caching consistency.
const DEFAULT_MODEL: &str = "gpt-5.4-mini-2026-03-17";

/// OpenAI chat completions endpoint.
const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default retry budget for transient failures (429 / 5xx).
const DEFAULT_MAX_RETRIES: u32 = 2;

/// Hard cap on the number of sub-queries emitted per rewriting call.
/// Setting this to 3 keeps retrieval latency bounded while covering most
/// multi-hop queries. Callers may lower this via [`QueryRewriterConfig`].
const MAX_SUB_QUERIES_HARD_LIMIT: usize = 3;

/// System prompt for query decomposition.
///
/// The prompt is intentionally verbose so that it exceeds the 1024-token
/// OpenAI prompt-cache threshold on the FIRST call. The static portion
/// (system prompt + JSON schema) never changes between calls, so the
/// prefix is cached server-side and subsequent calls enjoy ~50-80% savings
/// on prompt tokens.
///
/// Design goals:
/// 1. Decide if a query is simple (single-hop) and can pass through unchanged.
/// 2. For complex queries: produce 2-3 targeted sub-queries, each addressing
///    one aspect of the original intent.
/// 3. Always include the ORIGINAL query as one of the sub-queries so that
///    direct-match retrieval is never lost.
/// 4. Handle Romanian legal queries: article references, crime definitions,
///    cross-article penalties, comparative questions, negation rewriting.
const SYSTEM_PROMPT: &str = r#"You are an expert information-retrieval query analyst.

Your task is to analyse a user query and decide whether to pass it through
unchanged or decompose it into 2-3 focused sub-queries that together cover
all aspects of the original intent.

# Decision rules

## Keep the query as-is (should_decompose: false) when:
- The query has a single clear retrieval need (factual look-up, simple
  definition, one specific article, one specific person/event).
- Examples: "What is tâlhăria?", "Define robbery", "Art. 233 Cod Penal",
  "Who is Ion Creangă?", "What year was Romania's constitution adopted?".

## Decompose the query (should_decompose: true) when:
- The query asks to COMPARE two or more concepts, crimes, laws, or entities.
- The query contains MULTIPLE retrieval needs joined by "și" / "and" / "or".
- The query asks about both a definition AND a consequence / penalty.
- The query asks what something is NOT (negation) — rewrite to:
    (a) definition of the positive concept, and
    (b) exceptions / exclusions from that concept.
- The query implies a multi-hop lookup (e.g. "what punishment applies to X"
  where X is itself defined in another article).
- Examples:
    "Ce diferență este între furt și tâlhărie?" → 3 sub-queries:
      • original query (comparative anchor)
      • "definiția furtului Cod Penal"
      • "definiția tâlhăriei Cod Penal"
    "Ce NU este tâlhărie?" → 2 sub-queries:
      • "definiția tâlhăriei"
      • "excepțiile și limitele noțiunii de tâlhărie"
    "What are the penalties and elements of robbery?" → 2 sub-queries:
      • "definition of robbery criminal elements"
      • "penalties for robbery"

# Romanian legal query guidance

- Article references: "Art. 233 Cod Penal" → keep as one factual sub-query.
- Crime definitions: always emit a sub-query prefixed "definiția {infracțiunea}".
- Cross-article penalties: emit a sub-query "pedeapsa pentru {infracțiunea}".
- Comparative crimes (e.g. furt vs. tâlhărie): emit one sub-query per crime
  plus the original comparative query.
- Aggravating circumstances (forme agravate): emit sub-queries for both the
  basic form and the aggravated form.
- Constitutional/civil law queries follow the same decomposition rules.

# Output rules

1. ALWAYS include the original query verbatim as one of the sub-queries.
   This ensures direct-match retrieval is never discarded.
2. Sub-queries must be SHORTER and more focused than the original, not longer.
3. Maximum 3 sub-queries (enforced by schema).
4. If should_decompose is false, emit exactly 1 sub-query equal to the original.
5. query_type classification:
   - factual   — single-hop factual look-up (definition, article, date, name)
   - conceptual — abstract concept that requires understanding, not just recall
   - comparative — requires retrieving multiple items and comparing them
   - negation  — asks what something is NOT; rewrite to positive + exclusions
6. intent — one short phrase explaining WHY this sub-query was generated.
7. reasoning — a 1-2 sentence explanation of the overall decomposition decision.
8. Return ONLY valid JSON matching the provided schema. No prose, no markdown.

# Few-shot examples

Query: "Ce este tâlhăria?"
Output:
{
  "reasoning": "Single-concept definition query. No decomposition needed.",
  "should_decompose": false,
  "sub_queries": [
    {
      "text": "Ce este tâlhăria?",
      "intent": "Direct definition lookup for tâlhărie",
      "query_type": "factual"
    }
  ]
}

---

Query: "Ce diferență este între furt și tâlhărie?"
Output:
{
  "reasoning": "Comparative query requiring independent definitions of furt and tâlhărie before comparison can be grounded.",
  "should_decompose": true,
  "sub_queries": [
    {
      "text": "Ce diferență este între furt și tâlhărie?",
      "intent": "Original comparative anchor — preserves direct-match retrieval",
      "query_type": "comparative"
    },
    {
      "text": "definiția și elementele constitutive ale furtului Cod Penal",
      "intent": "Retrieve full definition of furt to support comparison",
      "query_type": "factual"
    },
    {
      "text": "definiția și elementele constitutive ale tâlhăriei Cod Penal",
      "intent": "Retrieve full definition of tâlhărie to support comparison",
      "query_type": "factual"
    }
  ]
}

---

Query: "What are the penalties for robbery and how does it differ from theft?"
Output:
{
  "reasoning": "Mixed query: penalties (factual) plus comparison (comparative). Decompose into three focused lookups.",
  "should_decompose": true,
  "sub_queries": [
    {
      "text": "What are the penalties for robbery and how does it differ from theft?",
      "intent": "Original query preserved for direct retrieval",
      "query_type": "comparative"
    },
    {
      "text": "penalties and sentencing for robbery",
      "intent": "Isolate the penalty aspect for targeted retrieval",
      "query_type": "factual"
    },
    {
      "text": "definition of robbery vs theft elements and distinctions",
      "intent": "Isolate the comparative definitional aspect",
      "query_type": "comparative"
    }
  ]
}

---

Query: "Ce NU este considerat tâlhărie în Codul Penal?"
Output:
{
  "reasoning": "Negation query. Rewrite as positive definition plus exclusions/exceptions.",
  "should_decompose": true,
  "sub_queries": [
    {
      "text": "Ce NU este considerat tâlhărie în Codul Penal?",
      "intent": "Original negation query preserved",
      "query_type": "negation"
    },
    {
      "text": "definiția tâlhăriei și elementele sale constitutive Cod Penal",
      "intent": "Retrieve positive definition to anchor the negation",
      "query_type": "factual"
    },
    {
      "text": "excepțiile și limitele infracțiunii de tâlhărie situații care nu constituie tâlhărie",
      "intent": "Retrieve exclusions and boundary cases for tâlhărie",
      "query_type": "conceptual"
    }
  ]
}
"#;

// ---------------------------------------------------------------------------
// Public domain types
// ---------------------------------------------------------------------------

/// The broad retrieval intent of a sub-query.
///
/// Used by the retrieval router to weight or bias strategy selection —
/// for example, `Comparative` sub-queries may benefit from a wider
/// [`RetrievalStrategy`] fan-out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubQueryType {
    /// A single-hop factual look-up: definition, article, date, name.
    Factual,
    /// An abstract concept that requires understanding, not only recall.
    Conceptual,
    /// A question that requires retrieving multiple items and comparing them.
    Comparative,
    /// A query about what something is NOT; typically rewritten into positive
    /// definition + exclusion sub-queries.
    Negation,
}

/// A single focused sub-query produced by [`QueryRewriter::rewrite`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    /// The rewritten query text to be embedded and dispatched for retrieval.
    pub text: String,

    /// Brief rationale for why this sub-query was generated.
    pub intent: String,

    /// Broad retrieval intent classification.
    pub query_type: SubQueryType,
}

/// The complete output of a [`QueryRewriter::rewrite`] call.
#[derive(Debug, Clone)]
pub struct RewrittenQuery {
    /// The original user query, unchanged.
    pub original: String,

    /// One to three focused sub-queries covering all aspects of [`original`].
    ///
    /// When [`was_decomposed`] is `false` this slice contains exactly one
    /// element whose [`SubQuery::text`] equals [`original`].
    pub sub_queries: Vec<SubQuery>,

    /// LLM-generated explanation of the decomposition decision.
    pub reasoning: String,

    /// `true` when the LLM decided the query needed decomposition and emitted
    /// more than one sub-query; `false` for simple pass-through queries.
    pub was_decomposed: bool,
}

impl RewrittenQuery {
    /// Construct a trivial pass-through result from a raw query string.
    ///
    /// Used as the fallback when the LLM call fails so the retrieval pipeline
    /// is never blocked by a rewriting failure.
    fn passthrough(original: impl Into<String>) -> Self {
        let text = original.into();
        Self {
            original: text.clone(),
            sub_queries: vec![SubQuery {
                text,
                intent: "pass-through (rewriter fallback)".to_owned(),
                query_type: SubQueryType::Factual,
            }],
            reasoning: String::new(),
            was_decomposed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`QueryRewriter`].
#[derive(Debug, Clone)]
pub struct QueryRewriterConfig {
    /// OpenAI API key. Reads `OPENAI_API_KEY` or `LLM_API_KEY` if not set
    /// explicitly (callers are encouraged to provide a value here rather than
    /// relying on env-var fallback).
    pub api_key: String,

    /// Model to use for rewriting. Defaults to [`DEFAULT_MODEL`].
    pub model: String,

    /// Maximum number of sub-queries to accept from the LLM.
    ///
    /// The LLM is instructed (via prompt) to emit at most this many, but the
    /// result is also truncated here as a hard safety cap.
    /// Values outside `1..=3` are clamped to that range at construction time.
    pub max_sub_queries: usize,

    /// Sampling temperature. Keep low (0.0–0.2) for deterministic rewriting.
    pub temperature: f32,
}

impl QueryRewriterConfig {
    /// Create a config with sensible defaults given just an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            max_sub_queries: 3,
            temperature: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI wire types (private)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
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
    format_type: &'static str,
    json_schema: JsonSchemaWrapper,
}

#[derive(Debug, Serialize)]
struct JsonSchemaWrapper {
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
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
}

/// Internal LLM response model; must match the JSON schema returned in
/// [`query_rewrite_schema`].
#[derive(Debug, Deserialize)]
struct LlmRewriteResponse {
    reasoning: String,
    should_decompose: bool,
    sub_queries: Vec<LlmSubQuery>,
}

#[derive(Debug, Deserialize)]
struct LlmSubQuery {
    text: String,
    intent: String,
    query_type: String,
}

// ---------------------------------------------------------------------------
// QueryRewriter
// ---------------------------------------------------------------------------

/// LLM-based query decomposition engine.
///
/// Cheap to clone — wraps an `Arc<reqwest::Client>` internally.
/// Construct once at startup and share across request handlers via `Arc<Self>`
/// or by cloning.
#[derive(Clone)]
pub struct QueryRewriter {
    client: reqwest::Client,
    api_key: Arc<String>,
    model: Arc<String>,
    max_sub_queries: usize,
    temperature: f32,
    max_retries: u32,
}

impl std::fmt::Debug for QueryRewriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryRewriter")
            .field("model", &self.model)
            .field("max_sub_queries", &self.max_sub_queries)
            .field("temperature", &self.temperature)
            .field("max_retries", &self.max_retries)
            .field("api_key", &"***")
            .finish()
    }
}

impl QueryRewriter {
    /// Create a new rewriter from the provided configuration.
    ///
    /// The `max_sub_queries` value is clamped to `1..=3` and `temperature` is
    /// clamped to `0.0..=1.0` to prevent misconfiguration.
    pub fn new(config: QueryRewriterConfig) -> Self {
        let max_sub_queries = config.max_sub_queries.clamp(1, MAX_SUB_QUERIES_HARD_LIMIT);
        let temperature = config.temperature.clamp(0.0_f32, 1.0_f32);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        tracing::info!(
            model = %config.model,
            max_sub_queries,
            temperature,
            "QueryRewriter created"
        );

        Self {
            client,
            api_key: Arc::new(config.api_key),
            model: Arc::new(config.model),
            max_sub_queries,
            temperature,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Rewrite `query` into one to three focused sub-queries via LLM.
    ///
    /// Simple queries return a single sub-query whose text matches the
    /// original. Complex queries return 2-3 targeted sub-queries.
    ///
    /// # Fallback behaviour
    ///
    /// On ANY failure (network error, quota exceeded, malformed JSON response)
    /// this method logs a `WARN` and returns a [`RewrittenQuery`] whose
    /// `sub_queries` contains only the original query verbatim. The retrieval
    /// pipeline is never interrupted by a rewriting failure.
    ///
    /// # Errors
    ///
    /// This method **never returns `Err`** — failures are swallowed and
    /// surfaced as the fallback pass-through result. The `Result` wrapper is
    /// kept for API symmetry with other async pipeline methods.
    #[instrument(skip(self), fields(query_len = query.len(), model = %self.model))]
    pub async fn rewrite(&self, query: &str) -> Result<RewrittenQuery, RetrievalError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            tracing::debug!("empty query — returning pass-through");
            return Ok(RewrittenQuery::passthrough(""));
        }

        let start = std::time::Instant::now();

        let result = self.rewrite_inner(trimmed).await;

        let elapsed = start.elapsed();

        match result {
            Ok(rewritten) => {
                tracing::info!(
                    was_decomposed = rewritten.was_decomposed,
                    sub_query_count = rewritten.sub_queries.len(),
                    latency_ms = elapsed.as_millis(),
                    "query rewriting complete"
                );
                Ok(rewritten)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    latency_ms = elapsed.as_millis(),
                    "query rewriting failed — falling back to original query"
                );
                Ok(RewrittenQuery::passthrough(trimmed))
            }
        }
    }

    // ------------------------------------------------------------------
    // Private implementation
    // ------------------------------------------------------------------

    /// Inner rewrite call — propagates errors so [`rewrite`] can catch them.
    async fn rewrite_inner(&self, query: &str) -> Result<RewrittenQuery, RetrievalError> {
        let body = build_request_body(
            self.model.as_str(),
            query,
            self.temperature,
            self.max_sub_queries,
        );

        let body_bytes = serde_json::to_vec(&body).map_err(|e| RetrievalError::Ingestion {
            reason: format!("serialize query rewrite request: {e}"),
        })?;

        let response_bytes = self.call_api_with_retry(body_bytes).await?;

        let chat: ChatResponse =
            serde_json::from_slice(&response_bytes).map_err(|e| RetrievalError::Ingestion {
                reason: format!("parse chat completion envelope: {e}"),
            })?;

        let content = chat
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| RetrievalError::Ingestion {
                reason: "chat completion returned no content".to_owned(),
            })?;

        let llm: LlmRewriteResponse =
            serde_json::from_str(&content).map_err(|e| RetrievalError::Ingestion {
                reason: format!(
                    "parse LLM rewrite response ({}): {}",
                    content.chars().take(200).collect::<String>(),
                    e
                ),
            })?;

        self.build_rewritten_query(query.to_owned(), llm)
    }

    /// Convert the raw LLM output into the public [`RewrittenQuery`] type,
    /// applying safety caps and ensuring the original query is always present.
    fn build_rewritten_query(
        &self,
        original: String,
        llm: LlmRewriteResponse,
    ) -> Result<RewrittenQuery, RetrievalError> {
        // Convert LLM string enum to typed variant; unknown values fall back
        // to Factual so schema changes in the prompt don't break the pipeline.
        let mut sub_queries: Vec<SubQuery> = llm
            .sub_queries
            .into_iter()
            .filter(|sq| !sq.text.trim().is_empty())
            .take(self.max_sub_queries)
            .map(|sq| SubQuery {
                text: sq.text.trim().to_owned(),
                intent: sq.intent.trim().to_owned(),
                query_type: parse_query_type(&sq.query_type),
            })
            .collect();

        // Safety invariant: the original query must be in the list.
        // The LLM is instructed to include it, but we enforce it here in case
        // it hallucinates or the model was fine-tuned differently.
        if !sub_queries
            .iter()
            .any(|sq| sq.text.trim() == original.trim())
        {
            tracing::debug!("LLM did not include original query — prepending it");
            sub_queries.insert(
                0,
                SubQuery {
                    text: original.clone(),
                    intent: "original query (safety re-insertion)".to_owned(),
                    query_type: SubQueryType::Factual,
                },
            );
            // Re-apply cap after insertion.
            sub_queries.truncate(self.max_sub_queries);
        }

        // If the LLM said should_decompose=false but somehow returned >1
        // sub-query, trust the count and flip the flag.
        let was_decomposed = llm.should_decompose || sub_queries.len() > 1;

        Ok(RewrittenQuery {
            original,
            sub_queries,
            reasoning: llm.reasoning,
            was_decomposed,
        })
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
                        reason: format!("query rewriter HTTP request failed: {e}"),
                    })?;

                let status = resp.status();
                let bytes = resp.bytes().await.map_err(|e| RetrievalError::Ingestion {
                    reason: format!("read query rewriter response body: {e}"),
                })?;

                if status.is_success() {
                    Ok(bytes.to_vec())
                } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    let msg = parse_api_error(&bytes, status);
                    tracing::warn!(
                        status = %status,
                        error = %msg,
                        "retryable query rewriter API error"
                    );
                    Err(RetrievalError::Ingestion { reason: msg })
                } else {
                    Err(RetrievalError::Ingestion {
                        reason: parse_api_error(&bytes, status),
                    })
                }
            }
        };

        op.retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_millis(400))
                .with_max_delay(Duration::from_secs(20))
                .with_max_times(self.max_retries as usize),
        )
        .when(is_retryable_error)
        .await
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decide whether a retrieval error from [`QueryRewriter`] is retryable.
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

/// Build the JSON-schema-constrained chat completion request body.
fn build_request_body<'a>(
    model: &'a str,
    query: &'a str,
    temperature: f32,
    max_sub_queries: usize,
) -> ChatRequest<'a> {
    ChatRequest {
        model,
        messages: vec![
            ChatMessage {
                role: "system",
                content: SYSTEM_PROMPT,
            },
            ChatMessage {
                role: "user",
                content: query,
            },
        ],
        temperature,
        response_format: ResponseFormat {
            format_type: "json_schema",
            json_schema: JsonSchemaWrapper {
                name: "query_rewrite",
                strict: true,
                schema: query_rewrite_schema(max_sub_queries),
            },
        },
    }
}

/// JSON schema for the LLM rewrite response, compatible with OpenAI structured
/// outputs (strict mode).
///
/// The `maxItems` constraint mirrors `max_sub_queries` so the LLM never exceeds
/// the caller's budget even before the Rust-side cap is applied.
fn query_rewrite_schema(max_items: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["reasoning", "should_decompose", "sub_queries"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "1-2 sentences explaining the decomposition decision"
            },
            "should_decompose": {
                "type": "boolean",
                "description": "true if the query was decomposed into multiple sub-queries"
            },
            "sub_queries": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["text", "intent", "query_type"],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The sub-query text to be used for retrieval"
                        },
                        "intent": {
                            "type": "string",
                            "description": "Brief rationale for this sub-query"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["factual", "conceptual", "comparative", "negation"],
                            "description": "Broad retrieval intent classification"
                        }
                    }
                }
            }
        }
    })
}

/// Parse an OpenAI error response body for a human-readable message.
fn parse_api_error(bytes: &[u8], status: StatusCode) -> String {
    match serde_json::from_slice::<ApiErrorResponse>(bytes) {
        Ok(err) => format!("HTTP {status}: {}", err.error.message),
        Err(_) => {
            let body_preview: String = String::from_utf8_lossy(bytes).chars().take(200).collect();
            format!("HTTP {status}: {body_preview}")
        }
    }
}

/// Map the LLM-emitted string enum to [`SubQueryType`].
///
/// Unknown values fall back to [`SubQueryType::Factual`] so schema changes in
/// future prompt versions don't panic.
fn parse_query_type(s: &str) -> SubQueryType {
    match s {
        "conceptual" => SubQueryType::Conceptual,
        "comparative" => SubQueryType::Comparative,
        "negation" => SubQueryType::Negation,
        _ => SubQueryType::Factual,
    }
}
