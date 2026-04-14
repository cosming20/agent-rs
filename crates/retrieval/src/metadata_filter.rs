//! LLM-based metadata hint extraction for query-time Qdrant filter narrowing.
//!
//! When a user query contains explicit structural references — page numbers,
//! article numbers, section titles, or content-type keywords like "tabelul"
//! (table) or "formula" — this module extracts those hints via an LLM
//! structured-output call, then converts them into [`FilterCondition`] values
//! that callers can apply to Qdrant point payloads before (or during) search.
//!
//! # Why this matters
//!
//! Ingested documents carry a rich 27-field metadata payload (page_start,
//! page_end, section_title, has_table, has_formula, has_code, …).  Without
//! filter narrowing, every query scans all indexed chunks.  With it, a query
//! like "Ce prevede tabelul din articolul 188?" narrows the search space to
//! chunks where `has_table = true` AND `section_title` contains "188".
//!
//! # Resilience
//!
//! Metadata filtering is strictly **best-effort**.  If the LLM call fails, the
//! network is unavailable, or the response cannot be parsed, the extractor logs
//! a WARN and returns [`QueryMetadataHints::default()`] — no filters — so
//! retrieval continues unimpeded with the full chunk set.
//!
//! # Integration
//!
//! ```rust,ignore
//! let extractor = MetadataFilterExtractor::new(MetadataFilterConfig {
//!     api_key: "sk-...".to_owned(),
//!     model: "gpt-4o-mini".to_owned(),
//!     temperature: 0.0,
//! });
//!
//! let hints = extractor.extract("Ce prevede articolul 188 cu privire la tabel?").await;
//! let filters = MetadataFilterExtractor::to_qdrant_filter(&hints);
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

/// Default OpenAI model for metadata hint extraction.
///
/// `gpt-4o-mini` is fast and cheap — appropriate for a best-effort pre-filter
/// step that must not add measurable latency to the retrieval path.
const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// OpenAI chat completions endpoint.
const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default retry budget for transient failures.
const DEFAULT_MAX_RETRIES: u32 = 2;

/// Maximum query characters forwarded to the LLM.  Metadata extraction only
/// needs the query text, which is nearly always short; this cap prevents
/// accidental large payloads.
const MAX_QUERY_CHARS: usize = 2_000;

/// Stable prompt-cache key — shared across all metadata extraction requests
/// so the static system prompt hits the cache after the first call.
const PROMPT_CACHE_KEY: &str = "agent-rs-metadata-filter-v1";

/// System prompt for metadata hint extraction.
///
/// Designed for OpenAI prompt caching:
/// - All static instructions live in the system message so the prefix is
///   identical across every call, keeping the cache key stable.
/// - Dynamic content (the query) is injected only as the user message.
///
/// Handles Romanian and English queries, covering the metadata fields present
/// in the enriched Docling payload schema.
const SYSTEM_PROMPT: &str = r#"You are a metadata extraction assistant.
Your task: analyse a user query and extract structural metadata hints that
indicate which document chunks are most likely to answer the question.

You MUST return strict JSON matching the provided schema. No prose, no markdown.

# Fields to extract

## page_start / page_end
Extract if the query references a specific page or page range.
- "pagina 47" → page_start: 47, page_end: 47
- "paginile 50-55" → page_start: 50, page_end: 55
- "pagina 10 până la 15" → page_start: 10, page_end: 15
- "page 10" → page_start: 10, page_end: 10
- "pages 3 to 7" → page_start: 3, page_end: 7
If NO page reference → page_start: null, page_end: null

## article_number
Extract if the query references a specific article, section, or numbered clause.
- "articolul 188" → "188"
- "art. 345" → "345"
- "Article 15" → "15"
- "secțiunea 3.4" → "3.4"
- "alineatul (2)" → "(2)"
If NO article/section reference → null

## section_contains
Extract a short keyword or phrase that should appear in the section title if
the query describes a named section or topic heading. Use only if the user
names a specific section or chapter. Keep it short (1-3 words max).
- "capitolul despre furt" → "furt"
- "secțiunea privind pedepsele" → "pedepse"
- "chapter on penalties" → "penalties"
If NO section name is given → null

## requires_table
Set to true ONLY if the query explicitly asks about a table, grid, list
presented in table form, or uses words like: tabel, tabelul, tabelar,
table, tablou, matrice, grilă.
Otherwise false.

## requires_formula
Set to true ONLY if the query explicitly asks about a formula, equation,
mathematical expression, or calculation. Words: formulă, formula, ecuație,
equation, calcul.
Otherwise false.

## requires_code
Set to true ONLY if the query explicitly asks about code, a code example,
program listing, or algorithm. Words: cod, code, exemplu de cod, listing,
algoritm, program, funcție, function.
Otherwise false.

# Important rules

- Most queries are plain natural language with NO structural hints.
  For those, return all nulls and all false values — do NOT invent hints.
- Do NOT extract article numbers from incidental numbers in a query
  (e.g. "give me 3 examples" must NOT produce article_number: "3").
- Only extract article_number when the query contains an explicit article,
  section, or clause reference marker (articolul, art., Article, §, alin., etc.).
- Return empty/default hints when nothing structural is detected.

# Output
Return ONLY valid JSON matching the provided schema. No additional text."#;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Metadata hints extracted from a natural language query.
///
/// All fields are optional / defaulted to `false`.  An all-default instance
/// means "no structural hints found" — callers should skip filtering entirely.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct QueryMetadataHints {
    /// Page range the user is asking about (inclusive, 1-based).
    ///
    /// `Some((47, 47))` means exactly page 47.
    /// `Some((50, 55))` means pages 50 through 55.
    pub page_range: Option<(u32, u32)>,

    /// Article or section number extracted from the query.
    ///
    /// `Some("188")` means the query explicitly references "articolul 188".
    pub article_number: Option<String>,

    /// Short fragment that should appear in the `section_title` payload field.
    ///
    /// `Some("furt")` narrows to sections whose title contains "furt".
    pub section_contains: Option<String>,

    /// The query explicitly mentions a table or tabular data.
    pub requires_table: bool,

    /// The query explicitly mentions a formula or equation.
    pub requires_formula: bool,

    /// The query explicitly mentions source code or a code example.
    pub requires_code: bool,
}

impl QueryMetadataHints {
    /// Return `true` if no structural hints were found (all fields are empty /
    /// false).  Callers can skip the filter-building step entirely in this case.
    pub fn is_empty(&self) -> bool {
        self.page_range.is_none()
            && self.article_number.is_none()
            && self.section_contains.is_none()
            && !self.requires_table
            && !self.requires_formula
            && !self.requires_code
    }
}

/// A single Qdrant payload filter condition.
///
/// Callers translate these into the Qdrant SDK's `Condition` type.  The
/// abstraction exists so this module stays free of a hard dependency on the
/// `qdrant-client` crate.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterCondition {
    /// Payload field must exactly match a scalar value.
    Match {
        /// Qdrant payload field name.
        field: String,
        /// Value to match (bool, string, or integer wrapped in JSON).
        value: serde_json::Value,
    },

    /// Payload field (numeric) must fall within `[gte, lte]`.
    ///
    /// Either bound may be `None` (open-ended range).
    Range {
        /// Qdrant payload field name.
        field: String,
        /// Inclusive lower bound (`>=`).
        gte: Option<f64>,
        /// Inclusive upper bound (`<=`).
        lte: Option<f64>,
    },

    /// Payload field (string) must contain the given substring (case-insensitive
    /// where the Qdrant filter backend supports it; otherwise exact substring).
    Contains {
        /// Qdrant payload field name.
        field: String,
        /// Substring the field value must contain.
        text: String,
    },
}

/// Configuration for [`MetadataFilterExtractor`].
#[derive(Debug, Clone)]
pub struct MetadataFilterConfig {
    /// OpenAI API key (`sk-...`).
    pub api_key: String,
    /// Model identifier.  Defaults to `gpt-4o-mini`.
    pub model: String,
    /// Sampling temperature.  Use `0.0` for deterministic extraction.
    pub temperature: f32,
}

impl Default for MetadataFilterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_MODEL.to_owned(),
            temperature: 0.0,
        }
    }
}

impl MetadataFilterConfig {
    /// Create a config with the given API key and the default model.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// LLM-backed metadata hint extractor.
///
/// Cheap to clone (internal HTTP client is `Arc`-wrapped).  Create once at
/// startup and share across request handlers via `Arc<MetadataFilterExtractor>`.
#[derive(Clone)]
pub struct MetadataFilterExtractor {
    client: reqwest::Client,
    api_key: Arc<String>,
    model: Arc<String>,
    temperature: f32,
    max_retries: u32,
}

impl std::fmt::Debug for MetadataFilterExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataFilterExtractor")
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("max_retries", &self.max_retries)
            .field("api_key", &"***")
            .finish()
    }
}

impl MetadataFilterExtractor {
    /// Create a new extractor from a [`MetadataFilterConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Ingestion`] if the HTTP client cannot be
    /// built (extremely rare — only when TLS libraries are missing).
    pub fn new(config: MetadataFilterConfig) -> Result<Self, RetrievalError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| RetrievalError::Ingestion {
                reason: format!("build MetadataFilterExtractor HTTP client: {e}"),
            })?;

        tracing::info!(
            model = %config.model,
            temperature = config.temperature,
            "MetadataFilterExtractor created"
        );

        Ok(Self {
            client,
            api_key: Arc::new(config.api_key),
            model: Arc::new(config.model),
            temperature: config.temperature,
            max_retries: DEFAULT_MAX_RETRIES,
        })
    }

    /// Override the retry budget (default: [`DEFAULT_MAX_RETRIES`]).
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Extract metadata hints from a natural language query.
    ///
    /// On any failure (network, auth, parse error) the method logs at WARN and
    /// returns `Ok(QueryMetadataHints::default())` — retrieval must never be
    /// blocked by a best-effort metadata extraction step.
    ///
    /// # Errors
    ///
    /// This method is infallible from the caller's perspective: it always
    /// returns `Ok(_)`.  The `Result` wrapper is kept for API consistency with
    /// the rest of the retrieval crate.
    #[instrument(skip(self), fields(query_len = query.len(), model = %self.model))]
    pub async fn extract(&self, query: &str) -> QueryMetadataHints {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            tracing::debug!("empty query — skipping metadata extraction");
            return QueryMetadataHints::default();
        }

        // Cap query length so we never forward unexpectedly large inputs.
        let bounded: &str = if trimmed.len() > MAX_QUERY_CHARS {
            // Truncate at a valid UTF-8 character boundary.
            let mut end = MAX_QUERY_CHARS;
            while end > 0 && !trimmed.is_char_boundary(end) {
                end -= 1;
            }
            tracing::warn!(
                original_bytes = trimmed.len(),
                truncated_to = end,
                "query exceeds MAX_QUERY_CHARS; truncating for metadata extraction"
            );
            &trimmed[..end]
        } else {
            trimmed
        };

        let body = build_request_body(self.model.as_str(), bounded, self.temperature);
        let body_bytes = match serde_json::to_vec(&body) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize metadata extraction request; skipping");
                return QueryMetadataHints::default();
            }
        };

        let response_bytes = match self.call_api_with_retry(body_bytes).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "metadata extraction API call failed; using empty hints");
                return QueryMetadataHints::default();
            }
        };

        // Parse the outer chat completion envelope.
        let chat: ChatResponse = match serde_json::from_slice(&response_bytes) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "failed to parse metadata extraction response envelope; using empty hints");
                return QueryMetadataHints::default();
            }
        };

        // Extract the assistant's text content.
        let content = match chat
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
        {
            Some(c) if !c.trim().is_empty() => c,
            _ => {
                tracing::warn!("metadata extraction returned empty content; using empty hints");
                return QueryMetadataHints::default();
            }
        };

        // Log token usage if available.
        if let Some(usage) = chat.usage {
            tracing::debug!(
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                total_tokens = usage.total_tokens,
                "metadata extraction token usage"
            );
        }

        // Parse the structured JSON the LLM returned.
        let raw: LlmHintsRaw = match serde_json::from_str(&content) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    content = %content,
                    "metadata extraction JSON parse failed; using empty hints"
                );
                return QueryMetadataHints::default();
            }
        };

        let hints = raw.into_hints();
        tracing::debug!(
            page_range = ?hints.page_range,
            article_number = ?hints.article_number,
            section_contains = ?hints.section_contains,
            requires_table = hints.requires_table,
            requires_formula = hints.requires_formula,
            requires_code = hints.requires_code,
            "metadata hints extracted"
        );

        hints
    }

    /// Convert extracted hints into Qdrant filter conditions.
    ///
    /// Returns `None` when `hints.is_empty()` — the caller should not apply
    /// any filter in that case.
    ///
    /// The returned conditions are meant to be combined with AND semantics
    /// (all must match).
    ///
    /// # Qdrant payload field mapping
    ///
    /// | Hint | Condition | Rationale |
    /// |---|---|---|
    /// | `page_range(a, b)` | `Range(page_start, lte=b)` + `Range(page_end, gte=a)` | Overlap: chunk covers any part of the range |
    /// | `article_number("N")` | `Contains(section_title, "N")` | Section titles embed article numbers |
    /// | `section_contains("X")` | `Contains(section_title, "X")` | Topic heading match |
    /// | `requires_table` | `Match(has_table, true)` | Direct bool payload field |
    /// | `requires_formula` | `Match(has_formula, true)` | Direct bool payload field |
    /// | `requires_code` | `Match(has_code, true)` | Direct bool payload field |
    pub fn to_qdrant_filter(hints: &QueryMetadataHints) -> Option<Vec<FilterCondition>> {
        if hints.is_empty() {
            return None;
        }

        let mut conditions: Vec<FilterCondition> = Vec::new();

        // Page range — use overlap logic so a chunk that spans pages 45-50
        // still matches a query asking about page 47.
        // Overlap condition: chunk.page_start <= range_end AND chunk.page_end >= range_start
        if let Some((page_start, page_end)) = hints.page_range {
            conditions.push(FilterCondition::Range {
                field: "page_start".to_owned(),
                gte: None,
                lte: Some(f64::from(page_end)),
            });
            conditions.push(FilterCondition::Range {
                field: "page_end".to_owned(),
                gte: Some(f64::from(page_start)),
                lte: None,
            });
        }

        // Article number — section_title contains the number.
        if let Some(ref article) = hints.article_number {
            conditions.push(FilterCondition::Contains {
                field: "section_title".to_owned(),
                text: article.clone(),
            });
        }

        // Section keyword — section_title contains the fragment.
        // If both article_number and section_contains are set, both apply.
        if let Some(ref section) = hints.section_contains {
            // Avoid duplicating the condition when article_number and
            // section_contains happen to refer to the same text.
            let already_covered = hints
                .article_number
                .as_deref()
                .map(|a| a == section.as_str())
                .unwrap_or(false);
            if !already_covered {
                conditions.push(FilterCondition::Contains {
                    field: "section_title".to_owned(),
                    text: section.clone(),
                });
            }
        }

        // Boolean payload fields.
        if hints.requires_table {
            conditions.push(FilterCondition::Match {
                field: "has_table".to_owned(),
                value: serde_json::Value::Bool(true),
            });
        }
        if hints.requires_formula {
            conditions.push(FilterCondition::Match {
                field: "has_formula".to_owned(),
                value: serde_json::Value::Bool(true),
            });
        }
        if hints.requires_code {
            conditions.push(FilterCondition::Match {
                field: "has_code".to_owned(),
                value: serde_json::Value::Bool(true),
            });
        }

        if conditions.is_empty() {
            None
        } else {
            tracing::debug!(
                num_conditions = conditions.len(),
                "metadata filter conditions built"
            );
            Some(conditions)
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// POST to `/v1/chat/completions` with exponential-backoff retry on 429 / 5xx.
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
                        reason: format!("metadata filter HTTP request failed: {e}"),
                    })?;

                let status = resp.status();
                let bytes = resp.bytes().await.map_err(|e| RetrievalError::Ingestion {
                    reason: format!("read metadata filter response body: {e}"),
                })?;

                if status.is_success() {
                    Ok(bytes.to_vec())
                } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    let msg = parse_api_error(&bytes, status);
                    tracing::warn!(
                        status = %status,
                        error = %msg,
                        "retryable metadata filter API error"
                    );
                    Err(RetrievalError::Ingestion { reason: msg })
                } else {
                    let msg = parse_api_error(&bytes, status);
                    Err(RetrievalError::Ingestion { reason: msg })
                }
            }
        };

        op.retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_millis(300))
                .with_max_delay(Duration::from_secs(15))
                .with_max_times(self.max_retries as usize),
        )
        .when(is_retryable_error)
        .await
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
    prompt_cache_key: &'static str,
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
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
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
    error_type: Option<String>,
    code: Option<String>,
}

// ---------------------------------------------------------------------------
// LLM output deserialization (private)
// ---------------------------------------------------------------------------

/// The raw JSON object the LLM returns.  All fields are nullable so we can
/// tolerate a model that omits optional fields rather than emitting `null`.
#[derive(Debug, Deserialize)]
struct LlmHintsRaw {
    #[serde(default)]
    page_start: Option<u32>,
    #[serde(default)]
    page_end: Option<u32>,
    #[serde(default)]
    article_number: Option<String>,
    #[serde(default)]
    section_contains: Option<String>,
    #[serde(default)]
    requires_table: bool,
    #[serde(default)]
    requires_formula: bool,
    #[serde(default)]
    requires_code: bool,
}

impl LlmHintsRaw {
    /// Convert the raw LLM output into the typed [`QueryMetadataHints`].
    fn into_hints(self) -> QueryMetadataHints {
        // Build page_range from the two nullable integers.
        // If only one is present, use it for both bounds.
        let page_range = match (self.page_start, self.page_end) {
            (Some(s), Some(e)) => Some((s.min(e), s.max(e))),
            (Some(s), None) => Some((s, s)),
            (None, Some(e)) => Some((e, e)),
            (None, None) => None,
        };

        // Normalise empty strings to None.
        let article_number = self
            .article_number
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_owned());

        let section_contains = self
            .section_contains
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_owned());

        QueryMetadataHints {
            page_range,
            article_number,
            section_contains,
            requires_table: self.requires_table,
            requires_formula: self.requires_formula,
            requires_code: self.requires_code,
        }
    }
}

// ---------------------------------------------------------------------------
// Request builder + schema
// ---------------------------------------------------------------------------

fn build_request_body<'a>(model: &'a str, query: &'a str, temperature: f32) -> ChatRequest<'a> {
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
                name: "query_metadata_hints",
                strict: true,
                schema: metadata_hints_schema(),
            },
        },
        prompt_cache_key: PROMPT_CACHE_KEY,
    }
}

/// JSON schema for [`LlmHintsRaw`] compatible with OpenAI structured outputs
/// (strict mode: all properties listed in `required`, `additionalProperties: false`).
fn metadata_hints_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "page_start": {
                "type": ["integer", "null"],
                "description": "Start page number (1-based). Null if the query does not reference a page."
            },
            "page_end": {
                "type": ["integer", "null"],
                "description": "End page number (1-based, inclusive). Null if the query does not reference a page."
            },
            "article_number": {
                "type": ["string", "null"],
                "description": "Article or section number as a string (e.g. '188', '3.4'). Null if none."
            },
            "section_contains": {
                "type": ["string", "null"],
                "description": "Short keyword from a section/chapter name the query refers to. Null if none."
            },
            "requires_table": {
                "type": "boolean",
                "description": "True only if the query explicitly asks about a table or tabular data."
            },
            "requires_formula": {
                "type": "boolean",
                "description": "True only if the query explicitly asks about a formula or equation."
            },
            "requires_code": {
                "type": "boolean",
                "description": "True only if the query explicitly asks about source code or a code example."
            }
        },
        "required": [
            "page_start",
            "page_end",
            "article_number",
            "section_contains",
            "requires_table",
            "requires_formula",
            "requires_code"
        ]
    })
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

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

fn parse_api_error(body: &[u8], status: StatusCode) -> String {
    if let Ok(err_resp) = serde_json::from_slice::<ApiErrorResponse>(body) {
        format!(
            "HTTP {} — {} (type={:?}, code={:?})",
            status, err_resp.error.message, err_resp.error.error_type, err_resp.error.code
        )
    } else {
        let preview = String::from_utf8_lossy(&body[..body.len().min(200)]);
        format!("HTTP {status} — {preview}")
    }
}
