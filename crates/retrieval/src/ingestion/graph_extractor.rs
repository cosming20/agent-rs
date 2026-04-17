//! LLM-based knowledge graph extraction for the ingestion pipeline.
//!
//! This module replaces the purely heuristic entity extractor in
//! [`crate::ingestion::pipeline`] with an LLM-driven pipeline that mirrors
//! Cognee's `generate_graph_prompt` flow: for each chunk of text we ask an
//! OpenAI model (via structured outputs / `response_format: json_schema`) to
//! emit a full [`KnowledgeGraph`] -- a set of typed entities and relationships.
//!
//! # Design
//!
//! - **Structured outputs** -- The extractor uses the `/v1/chat/completions`
//!   endpoint with `response_format = { type: "json_schema" }`, which
//!   guarantees the model's reply is JSON matching our schema. No free-form
//!   JSON parsing of markdown fences or similar hacks.
//! - **Coreference-friendly IDs** -- Node IDs are human-readable `snake_case`
//!   slugs (e.g. `albert_einstein`), which allows cross-chunk and
//!   cross-document deduplication by string equality.
//! - **Resilience** -- Rate-limit (429) and server (5xx) errors are retried
//!   with exponential backoff via `backon`. Malformed JSON from the model is
//!   logged at WARN and returns an empty graph so the rest of ingestion can
//!   continue unaffected.
//! - **Concurrency** -- Batch extraction runs chunks through a bounded
//!   semaphore so we can pipeline LLM calls without breaking provider rate
//!   limits.
//!
//! # Integration
//!
//! [`IngestionPipeline`](super::pipeline::IngestionPipeline) accepts an
//! optional `Arc<LlmGraphExtractor>` via
//! [`IngestionPipelineConfig`](super::pipeline::IngestionPipelineConfig).
//! When present, the pipeline calls [`LlmGraphExtractor::extract_batch`]
//! after chunking and uses the result to build graph nodes and edges.

use std::sync::Arc;
use std::time::Duration;

use agent_rs_observability::metrics::RetrievalMetrics;
use backon::{ExponentialBuilder, Retryable};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::instrument;

use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default OpenAI model for graph extraction. Chosen for cost/quality balance.
///
/// `gpt-5.4-mini-2026-03-17` is the small reasoning-tuned model: better at
/// structured extraction and coreference than `gpt-4o-mini` while staying
/// within ~3x the per-token cost. Override per-call via [`LlmGraphExtractor::with_default_model`]
/// or pass an explicit model to [`LlmGraphExtractor::new`].
const DEFAULT_MODEL: &str = "gpt-5.4-mini-2026-03-17";

/// OpenAI chat completions endpoint.
const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default retry budget for transient failures.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default concurrency for batch extraction.
///
/// Bumped from 5 to 20 — gpt-5.4-mini supports thousands of req/min.
/// At 20 concurrent calls × ~3500 tokens each = ~140K tok/min, well
/// within standard tier limits. Reduces KG extraction from ~164s to ~42s
/// for a 412-chunk document.
pub const DEFAULT_BATCH_CONCURRENCY: usize = 20;

/// Maximum characters of a chunk we forward to the LLM. Very large chunks are
/// truncated at the character boundary to keep request size predictable.
///
/// Sized well above the chunker's default (~4K chars per chunk) so this
/// bound almost never fires in practice. gpt-5.4-mini handles 200K context.
const MAX_CHUNK_CHARS: usize = 40_000;

/// Fallback size when the primary extraction attempt fails and we retry
/// with aggressive character sanitization (strips weird unicode, PDF
/// artifacts). Kept smaller than MAX_CHUNK_CHARS because the failure
/// implies the chunk already had payload issues.
const MAX_SANITIZED_FALLBACK_CHARS: usize = 10_000;

/// System prompt for knowledge graph extraction.
///
/// Static prefix designed for OpenAI prompt caching:
/// - All static instructions live here so the prefix is identical across
///   every chunk extraction call. Combined with the JSON schema (also static)
///   this gives a stable cacheable prefix above the ~1024 token threshold,
///   slashing cost on subsequent requests within the same cache window.
/// - Dynamic content (the chunk text) is injected as the user message,
///   never inside the system prompt, so it doesn't pollute the cache.
///
/// Optimised for multilingual extraction with explicit guidance on
/// scientific, technical, business, and narrative document types. Few-shot
/// examples cover entity normalisation, coreference, implicit relations, and
/// edge-case handling — drawn from multiple neutral domains to avoid
/// anchoring the model to any single domain.
#[doc(hidden)]
pub const SYSTEM_PROMPT_FOR_TESTING: &str = SYSTEM_PROMPT;

const SYSTEM_PROMPT: &str = r#"You are an expert knowledge graph extractor.
Your task: read a text chunk in any language and emit a typed knowledge graph
of entities (nodes) and relationships (edges) that captures the salient facts.

# Universal rules

1. Nodes represent entities and concepts (like Wikipedia pages or ontology terms).
2. Edges represent typed, directed relationships between two nodes.
3. Output strict JSON matching the provided schema. No prose, no markdown fences.
4. If the text contains no extractable entities or relations, return empty lists.

# Node rules

- **Type labels**: Use specific but reusable categories. Prefer:
  Person, Organization, Location, Date, Event, Document, Concept,
  Product, Technology, Discipline, Material, Process, Quantity, Measurement,
  Right, Obligation, Currency, Title, Role, BodyPart, Disease,
  Treatment, Chemical, Species, Vehicle, Building, Award, Phenomenon.
  Avoid generic "Entity", "Thing", "Item".
- **Node IDs**: Lowercase ASCII snake_case slugs (`albert_einstein`,
  `world_war_two`, `section_4_2`, `request_timeout`).
  - Strip diacritics and combining marks to plain ASCII (ă→a, â→a, é→e,
    ü→u, ñ→n, ś→s, ç→c, etc.).
  - Replace spaces and punctuation with underscores. Collapse multiple underscores.
  - Never use integer IDs. IDs are the dedup key across the entire corpus.
- **Names**: The full canonical name as it appears in the text (preserve diacritics).
- **Descriptions**: One sentence grounded in the source text, no speculation.

# Coreference and normalization

- If "Einstein" appears, then later "Albert Einstein" appears, use the SAME id
  `albert_einstein` for both. Always prefer the most complete identifier.
- Pronouns and anaphora: resolve "he/she/it/they" (and cross-lingual
  equivalents) to the most recent matching entity. Never create a node
  for a pronoun.
- Dates: Format `YYYY-MM-DD` when fully specified; otherwise `YYYY-MM` or `YYYY`.
  Parse dates written in any language into this canonical form
  (e.g. "5 June 2014" → `2014-06-05`).
- Numeric quantities: include the unit in the name (e.g. `5_year_sentence`,
  `250_milliseconds`).

# Edge rules

- Use snake_case verb phrases in the present tense or past participle:
  `founded_by`, `located_in`, `acted_in`, `signed_on`, `depends_on`,
  `authored_by`, `regulated_by`, `applies_to`, `succeeded_by`,
  `composed_of`, `reports_to`, `references`.
- Source and target node IDs MUST appear in the nodes list. Never invent IDs.
- Capture IMPLICIT relations the text strongly entails. For example:
  "When X happens, Y becomes Z" implies `x -[causes]-> z` when `z` is a
  defined state of `y`.
- Focus on meaningful, non-trivial relations. Skip incidental mentions
  ("the document is 5 pages long").
- For regulatory / rule-bearing text: extract relations like `prohibits`,
  `permits`, `requires`, `defines`, `aggravates`, `excludes_from`,
  `applies_to` — using the language of the source document.

# Few-shot examples

EXAMPLE 1 — Scientific text:
"Marie Curie was awarded the Nobel Prize in Physics in 1903 jointly with her
husband Pierre Curie and Henri Becquerel for their research on radioactivity."

Expected output:
{
  "nodes": [
    {"id": "marie_curie", "name": "Marie Curie", "type": "Person",
     "description": "Physicist and chemist who pioneered research on radioactivity."},
    {"id": "pierre_curie", "name": "Pierre Curie", "type": "Person",
     "description": "Physicist and husband of Marie Curie."},
    {"id": "henri_becquerel", "name": "Henri Becquerel", "type": "Person",
     "description": "Physicist who shared the 1903 Nobel Prize in Physics."},
    {"id": "nobel_prize_in_physics", "name": "Nobel Prize in Physics", "type": "Award",
     "description": "Annual prize awarded by the Royal Swedish Academy of Sciences."},
    {"id": "1903", "name": "1903", "type": "Date",
     "description": "Year the prize was awarded."},
    {"id": "radioactivity", "name": "radioactivity", "type": "Phenomenon",
     "description": "Spontaneous emission of radiation from atomic nuclei."}
  ],
  "edges": [
    {"source_node_id": "marie_curie", "target_node_id": "nobel_prize_in_physics",
     "relationship_name": "awarded", "description": "Marie Curie received the prize."},
    {"source_node_id": "pierre_curie", "target_node_id": "nobel_prize_in_physics",
     "relationship_name": "awarded", "description": "Pierre Curie shared the prize."},
    {"source_node_id": "henri_becquerel", "target_node_id": "nobel_prize_in_physics",
     "relationship_name": "awarded", "description": "Henri Becquerel shared the prize."},
    {"source_node_id": "nobel_prize_in_physics", "target_node_id": "1903",
     "relationship_name": "awarded_in", "description": "Year of award."},
    {"source_node_id": "marie_curie", "target_node_id": "pierre_curie",
     "relationship_name": "married_to", "description": "Marie and Pierre were spouses."},
    {"source_node_id": "marie_curie", "target_node_id": "radioactivity",
     "relationship_name": "researched", "description": "Curie's primary research topic."}
  ]
}

EXAMPLE 2 — Technical reference text:
"The TCP protocol, defined in RFC 793 (1981), implements reliable ordered
delivery over IP by introducing sequence numbers and acknowledgements.
Van Jacobson's 1988 paper introduced congestion control, which is now part
of RFC 5681. Without congestion control, TCP suffers from congestion
collapse under heavy load."

Expected output:
{
  "nodes": [
    {"id": "tcp_protocol", "name": "TCP protocol", "type": "Concept",
     "description": "Reliable ordered delivery protocol over IP."},
    {"id": "rfc_793", "name": "RFC 793", "type": "Document",
     "description": "1981 specification defining TCP."},
    {"id": "rfc_5681", "name": "RFC 5681", "type": "Document",
     "description": "Specification that includes TCP congestion control."},
    {"id": "ip", "name": "IP", "type": "Concept",
     "description": "Internet Protocol, the layer over which TCP runs."},
    {"id": "sequence_numbers", "name": "sequence numbers", "type": "Concept",
     "description": "Per-byte numbering used by TCP to achieve ordered delivery."},
    {"id": "acknowledgements", "name": "acknowledgements", "type": "Concept",
     "description": "Receiver-side confirmations used by TCP."},
    {"id": "van_jacobson", "name": "Van Jacobson", "type": "Person",
     "description": "Introduced TCP congestion control in a 1988 paper."},
    {"id": "van_jacobson_1988_paper", "name": "Van Jacobson 1988 paper",
     "type": "Document", "description": "Introduced TCP congestion control."},
    {"id": "congestion_control", "name": "congestion control", "type": "Concept",
     "description": "Mechanism to avoid TCP congestion collapse."},
    {"id": "congestion_collapse", "name": "congestion collapse", "type": "Concept",
     "description": "Failure mode under heavy load without congestion control."},
    {"id": "1981", "name": "1981", "type": "Date",
     "description": "Publication year of RFC 793."},
    {"id": "1988", "name": "1988", "type": "Date",
     "description": "Publication year of Van Jacobson's congestion-control paper."}
  ],
  "edges": [
    {"source_node_id": "rfc_793", "target_node_id": "tcp_protocol",
     "relationship_name": "defines",
     "description": "RFC 793 defines the TCP protocol."},
    {"source_node_id": "rfc_793", "target_node_id": "1981",
     "relationship_name": "published_in",
     "description": "RFC 793 was published in 1981."},
    {"source_node_id": "tcp_protocol", "target_node_id": "ip",
     "relationship_name": "runs_over",
     "description": "TCP runs over IP."},
    {"source_node_id": "tcp_protocol", "target_node_id": "sequence_numbers",
     "relationship_name": "uses",
     "description": "TCP uses sequence numbers for ordered delivery."},
    {"source_node_id": "tcp_protocol", "target_node_id": "acknowledgements",
     "relationship_name": "uses",
     "description": "TCP uses acknowledgements for reliable delivery."},
    {"source_node_id": "van_jacobson", "target_node_id": "van_jacobson_1988_paper",
     "relationship_name": "authored",
     "description": "Van Jacobson authored the 1988 paper."},
    {"source_node_id": "van_jacobson_1988_paper", "target_node_id": "congestion_control",
     "relationship_name": "introduces",
     "description": "The paper introduces TCP congestion control."},
    {"source_node_id": "van_jacobson_1988_paper", "target_node_id": "1988",
     "relationship_name": "published_in",
     "description": "Published in 1988."},
    {"source_node_id": "rfc_5681", "target_node_id": "congestion_control",
     "relationship_name": "includes",
     "description": "RFC 5681 includes TCP congestion control."},
    {"source_node_id": "congestion_control", "target_node_id": "congestion_collapse",
     "relationship_name": "prevents",
     "description": "Congestion control prevents congestion collapse under heavy load."}
  ]
}

# Output

Return ONLY valid JSON matching the provided schema. Do not include explanations.
Do not wrap the JSON in markdown code fences. Do not escape quotes inside property
values unless required by JSON. If the text contains no clear entities, return
{"nodes": [], "edges": []}."#;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single node emitted by the LLM graph extractor.
///
/// Mirrors Cognee's [`Node`](https://github.com/topoteretes/cognee/blob/dev/cognee/shared/data_models.py)
/// data model but uses idiomatic Rust naming (`node_type` instead of the
/// reserved `type`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExtractedNode {
    /// Human-readable lowercase `snake_case` identifier.
    ///
    /// Used as the dedup key when merging graphs from multiple chunks.
    pub id: String,

    /// Full entity name as it appears in the source text.
    pub name: String,

    /// Simple type label (e.g. `Person`, `Organization`, `Location`).
    #[serde(rename = "type")]
    pub node_type: String,

    /// One-sentence description grounded in the source text.
    #[serde(default)]
    pub description: String,
}

/// A single relationship emitted by the LLM graph extractor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExtractedEdge {
    /// Node ID of the source entity. Must reference a node in the same graph.
    pub source_node_id: String,

    /// Node ID of the target entity. Must reference a node in the same graph.
    pub target_node_id: String,

    /// `snake_case` verb or verb phrase describing the relationship.
    pub relationship_name: String,

    /// One-sentence description of the relationship, grounded in text.
    #[serde(default)]
    pub description: String,
}

/// A knowledge graph extracted from a single chunk of text.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct KnowledgeGraph {
    /// All entities extracted from the chunk.
    #[serde(default)]
    pub nodes: Vec<ExtractedNode>,

    /// All relationships extracted from the chunk.
    #[serde(default)]
    pub edges: Vec<ExtractedEdge>,
}

impl KnowledgeGraph {
    /// Return `true` if the graph contains no nodes and no edges.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }
}

/// A complete `(source) -[relation]-> (target)` triplet, linearised into a
/// single vectorisable unit.
///
/// Triplets are the atomic retrieval unit used to replicate Cognee's
/// [`brute_force_triplet_search`](https://github.com/topoteretes/cognee/blob/dev/cognee/modules/retrieval/utils/brute_force_triplet_search.py):
/// instead of indexing only chunks, the ingestion pipeline also indexes each
/// edge's source/edge/target triple as an embedded text, so the retriever can
/// score whole relations against the query embedding rather than only chunk
/// content.
///
/// The full [`Triplet`] — both its projected text and its provenance — is
/// later stored as the Qdrant payload so the ranker can reassemble the
/// concrete subject/predicate/object for downstream evidence fusion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Triplet {
    /// Deterministic UUID (v5 over `{source_id}|{relation}|{target_id}`).
    ///
    /// Used as the Qdrant point id so repeated ingestion of the same edge
    /// produces an idempotent upsert.
    pub id: String,

    /// Source node ID as emitted by the LLM extractor (snake_case slug).
    pub source_node_id: String,

    /// Human-readable name of the source node as it appears in text.
    pub source_node_name: String,

    /// Sanitised relationship verb / predicate.
    pub relationship_name: String,

    /// Target node ID as emitted by the LLM extractor (snake_case slug).
    pub target_node_id: String,

    /// Human-readable name of the target node as it appears in text.
    pub target_node_name: String,

    /// Chunk ID that generated this triplet (provenance / de-dup key).
    pub chunk_id: String,

    /// Linearised text used for embedding:
    /// `"{source_name} -> {relationship_name} -> {target_name}"`.
    pub triplet_text: String,
}

impl Triplet {
    /// Build a [`Triplet`] from the source + target names and the sanitised
    /// relationship, deriving a deterministic UUID v5 id from the tuple.
    ///
    /// The `chunk_id` records the chunk this triplet was extracted from so
    /// downstream evidence assembly can link back to the original text.
    pub fn from_parts(
        source_node_id: impl Into<String>,
        source_node_name: impl Into<String>,
        relationship_name: impl Into<String>,
        target_node_id: impl Into<String>,
        target_node_name: impl Into<String>,
        chunk_id: impl Into<String>,
    ) -> Self {
        let source_node_id = source_node_id.into();
        let source_node_name = source_node_name.into();
        let relationship_name = relationship_name.into();
        let target_node_id = target_node_id.into();
        let target_node_name = target_node_name.into();
        let chunk_id = chunk_id.into();

        let key = format!("{source_node_id}|{relationship_name}|{target_node_id}");
        let id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, key.as_bytes()).to_string();

        let triplet_text =
            format!("{source_node_name} -> {relationship_name} -> {target_node_name}");

        Self {
            id,
            source_node_id,
            source_node_name,
            relationship_name,
            target_node_id,
            target_node_name,
            chunk_id,
            triplet_text,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    response_format: ResponseFormat,
    /// OpenAI prompt-cache routing hint. Same key → same cache shard, which
    /// dramatically improves cache hit rate when many requests share the same
    /// system prompt + schema prefix.
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
    /// Token usage breakdown including prompt-cache hit count.
    /// Only set when the upstream API returns it (always for OpenAI gpt-* models).
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
    /// Detailed breakdown of prompt tokens. Contains `cached_tokens` showing
    /// how many of the prompt tokens were served from the cache.
    /// Only present for prompts ≥1024 tokens.
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
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
// LlmGraphExtractor
// ---------------------------------------------------------------------------

/// LLM-based entity + relation extractor.
///
/// Cheap to clone (wraps an `Arc<reqwest::Client>`). Designed to be constructed
/// once at startup and shared across the ingestion pipeline.
#[derive(Clone)]
pub struct LlmGraphExtractor {
    client: Arc<reqwest::Client>,
    api_key: Arc<String>,
    model: Arc<String>,
    max_retries: u32,
    /// Optional shared metric instruments for graph extraction failures.
    metrics: Option<Arc<RetrievalMetrics>>,
}

impl std::fmt::Debug for LlmGraphExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmGraphExtractor")
            .field("model", &self.model)
            .field("max_retries", &self.max_retries)
            .field("api_key", &"***")
            .finish()
    }
}

impl LlmGraphExtractor {
    /// Create a new extractor with the given OpenAI API key and model.
    ///
    /// Use `DEFAULT_MODEL` (`gpt-4o-mini`) when the caller has no preference.
    pub fn new(api_key: String, model: String, client: Arc<reqwest::Client>) -> Self {
        tracing::info!(
            model = %model,
            max_retries = DEFAULT_MAX_RETRIES,
            "LlmGraphExtractor created"
        );
        Self {
            client,
            api_key: Arc::new(api_key),
            model: Arc::new(model),
            max_retries: DEFAULT_MAX_RETRIES,
            metrics: None,
        }
    }

    /// Create a new extractor using the default model (`gpt-4o-mini`).
    pub fn with_default_model(api_key: String, client: Arc<reqwest::Client>) -> Self {
        Self::new(api_key, DEFAULT_MODEL.to_owned(), client)
    }

    /// Override the retry budget. The default is `DEFAULT_MAX_RETRIES`.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Attach shared metric instruments for extraction failure tracking.
    ///
    /// When set, [`extract_batch`] increments the graph extraction failure
    /// counter each time both the primary and sanitized fallback attempts fail
    /// for a chunk.
    pub fn with_metrics(mut self, metrics: Arc<RetrievalMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Extract a knowledge graph from a single text chunk.
    ///
    /// Returns an empty [`KnowledgeGraph`] for empty or whitespace-only input
    /// without making an API call.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Ingestion`] for non-retryable transport,
    /// authentication, or schema validation failures. Malformed JSON in an
    /// otherwise successful response is treated as a soft failure: the method
    /// logs a WARN and returns `Ok(KnowledgeGraph::default())`.
    #[instrument(skip(self, chunk_text), fields(chars = chunk_text.len(), model = %self.model))]
    pub async fn extract_from_chunk(
        &self,
        chunk_text: &str,
    ) -> Result<KnowledgeGraph, RetrievalError> {
        let trimmed = chunk_text.trim();
        if trimmed.is_empty() {
            tracing::debug!("skipping empty chunk");
            return Ok(KnowledgeGraph::default());
        }

        // Truncate very large chunks at a character boundary.
        let bounded: &str = if trimmed.chars().count() > MAX_CHUNK_CHARS {
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
                "chunk exceeds MAX_CHUNK_CHARS; truncating for extraction"
            );
            &trimmed[..end]
        } else {
            trimmed
        };

        // Sanitize control chars (form feed, vertical tab, etc.) that some
        // PDF extractors leave in text and which break OpenAI's JSON parser.
        // Keep \n, \r, \t — strip everything else below ASCII 0x20.
        let payload_owned: String = bounded
            .chars()
            .map(|c| {
                if c == '\n' || c == '\r' || c == '\t' {
                    c
                } else if (c as u32) < 0x20 || c == '\u{007F}' {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        let payload_text: &str = &payload_owned;

        let body = build_request_body(self.model.as_str(), payload_text);

        // Serialize once so retries can cheaply reuse the body.
        let body_bytes = serde_json::to_vec(&body).map_err(|e| RetrievalError::Ingestion {
            reason: format!("serialize graph extraction request: {e}"),
        })?;

        let llm_started = std::time::Instant::now();
        let response_bytes = self.call_api_with_retry(body_bytes).await?;
        let llm_elapsed_ms = llm_started.elapsed().as_secs_f64() * 1000.0;

        // Parse the outer chat completion envelope.
        let chat: ChatResponse = match serde_json::from_slice(&response_bytes) {
            Ok(c) => c,
            Err(e) => {
                if let Some(m) = self.metrics.as_ref() {
                    m.record_llm_failure("graph_extractor", "malformed_json");
                }
                tracing::warn!(
                    error = %e,
                    preview = %bytes_preview(&response_bytes),
                    "failed to parse chat completion envelope; returning empty graph"
                );
                return Ok(KnowledgeGraph::default());
            }
        };

        // Log token usage and prompt-cache hit ratio so we can verify caching
        // is actually working. cached_tokens > 0 means we hit OpenAI's
        // automatic prompt cache (≥1024-token static prefix matched). Also
        // emit OTel metrics so cost + latency land in Prometheus/Grafana.
        if let Some(usage) = chat.usage.as_ref() {
            let cached = usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens)
                .unwrap_or(0);
            let cache_ratio = if usage.prompt_tokens > 0 {
                (cached as f64) / (usage.prompt_tokens as f64)
            } else {
                0.0
            };
            tracing::info!(
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                total_tokens = usage.total_tokens,
                cached_tokens = cached,
                cache_hit_ratio = format!("{:.1}%", cache_ratio * 100.0),
                latency_ms = llm_elapsed_ms,
                "graph_extractor token usage"
            );
            if let Some(m) = self.metrics.as_ref() {
                m.record_llm_call(
                    "graph_extractor",
                    &self.model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    llm_elapsed_ms,
                );
            }
        }

        let Some(content) = chat
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
        else {
            tracing::warn!("chat completion returned no content; returning empty graph");
            return Ok(KnowledgeGraph::default());
        };

        // Parse the JSON-schema constrained content.
        match serde_json::from_str::<KnowledgeGraph>(&content) {
            Ok(graph) => {
                let cleaned = sanitize_graph(graph);
                tracing::debug!(
                    nodes = cleaned.nodes.len(),
                    edges = cleaned.edges.len(),
                    "extracted graph from chunk"
                );
                Ok(cleaned)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    preview = %content.chars().take(200).collect::<String>(),
                    "LLM returned invalid JSON for KnowledgeGraph; returning empty graph"
                );
                Ok(KnowledgeGraph::default())
            }
        }
    }

    /// Extract knowledge graphs for a batch of chunks in parallel.
    ///
    /// The returned `Vec` has the same length and ordering as `chunk_texts`.
    /// A failed chunk yields an empty [`KnowledgeGraph`] (errors are logged at
    /// WARN) so partial extraction failures do not block the ingestion
    /// pipeline.
    ///
    /// `concurrency` bounds the number of in-flight LLM requests and should be
    /// tuned to match provider rate limits (see [`DEFAULT_BATCH_CONCURRENCY`]).
    #[instrument(skip(self, chunk_texts), fields(total = chunk_texts.len(), concurrency))]
    pub async fn extract_batch(
        &self,
        chunk_texts: &[&str],
        concurrency: usize,
    ) -> Vec<KnowledgeGraph> {
        if chunk_texts.is_empty() {
            return Vec::new();
        }

        let limit = concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(limit));
        let mut handles = Vec::with_capacity(chunk_texts.len());

        for (idx, chunk) in chunk_texts.iter().enumerate() {
            let extractor = self.clone();
            let text = (*chunk).to_owned();
            let permit_sem = Arc::clone(&semaphore);

            let handle = tokio::spawn(async move {
                // acquire_owned can only fail if the semaphore is closed, which
                // we never do.
                let _permit = match permit_sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(
                            chunk_index = idx,
                            error = %e,
                            "semaphore closed; skipping chunk"
                        );
                        return (idx, KnowledgeGraph::default());
                    }
                };

                match extractor.extract_from_chunk(&text).await {
                    Ok(graph) => (idx, graph),
                    Err(e) => {
                        tracing::warn!(
                            chunk_index = idx,
                            error = %e,
                            "graph extraction failed; retrying with aggressive sanitization"
                        );
                        // Fallback: strip ALL non-letter/digit/space/basic-punct chars
                        // and cap at MAX_SANITIZED_FALLBACK_CHARS. Handles PDF artifacts,
                        // weird unicode, lone surrogates, etc. that break the JSON body.
                        let sanitized: String = text
                            .chars()
                            .filter(|c| {
                                c.is_alphanumeric()
                                    || c.is_whitespace()
                                    || matches!(
                                        *c,
                                        '.' | ','
                                            | ';'
                                            | ':'
                                            | '!'
                                            | '?'
                                            | '-'
                                            | '('
                                            | ')'
                                            | '['
                                            | ']'
                                            | '/'
                                            | '\''
                                            | '"'
                                    )
                            })
                            .take(MAX_SANITIZED_FALLBACK_CHARS)
                            .collect();
                        match extractor.extract_from_chunk(&sanitized).await {
                            Ok(graph) => {
                                tracing::info!(
                                    chunk_index = idx,
                                    nodes = graph.nodes.len(),
                                    "fallback extraction succeeded after sanitization"
                                );
                                (idx, graph)
                            }
                            Err(e2) => {
                                tracing::warn!(
                                    chunk_index = idx,
                                    error = %e2,
                                    "fallback extraction also failed; using empty graph"
                                );
                                if let Some(m) = &extractor.metrics {
                                    m.record_graph_extraction_failure();
                                }
                                (idx, KnowledgeGraph::default())
                            }
                        }
                    }
                }
            });
            handles.push(handle);
        }

        // Collect into an ordered vec. Any JoinError degrades to an empty graph.
        let mut results: Vec<KnowledgeGraph> = vec![KnowledgeGraph::default(); chunk_texts.len()];
        for handle in handles {
            match handle.await {
                Ok((idx, graph)) => {
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = graph;
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "graph extraction task panicked");
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
                        reason: format!("graph extractor HTTP request failed: {e}"),
                    })?;

                let status = resp.status();
                let bytes = resp.bytes().await.map_err(|e| RetrievalError::Ingestion {
                    reason: format!("read graph extractor response body: {e}"),
                })?;

                if status.is_success() {
                    Ok(bytes.to_vec())
                } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    let msg = parse_api_error(&bytes, status);
                    tracing::warn!(status = %status, error = %msg, "retryable graph extractor API error");
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

/// Decide whether an ingestion error produced by [`LlmGraphExtractor`] should
/// be retried (rate-limit or transient server failure).
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

/// Build the JSON-schema-constrained chat completion request for a chunk.
fn build_request_body<'a>(model: &'a str, chunk_text: &'a str) -> ChatRequest<'a> {
    ChatRequest {
        model,
        messages: vec![
            ChatMessage {
                role: "system",
                content: SYSTEM_PROMPT,
            },
            ChatMessage {
                role: "user",
                content: chunk_text,
            },
        ],
        temperature: 0.0,
        response_format: ResponseFormat {
            format_type: "json_schema",
            json_schema: JsonSchemaWrapper {
                name: "knowledge_graph",
                strict: true,
                schema: knowledge_graph_schema(),
            },
        },
        // Stable key shared by every chunk extraction → routes to the same
        // cache shard so the static prefix (system prompt + schema) hits the
        // cache after the first call. Different from the chunk text which goes
        // in the user message and is the only varying part.
        prompt_cache_key: "agent-rs-graph-extractor-v2",
    }
}

/// JSON schema for [`KnowledgeGraph`] compatible with OpenAI structured outputs.
///
/// Strict mode requires:
/// - every object sets `additionalProperties: false`
/// - every property is listed in the object's `required` array
fn knowledge_graph_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Human-readable lowercase snake_case identifier"
                        },
                        "name": {
                            "type": "string",
                            "description": "Full entity name as it appears in text"
                        },
                        "type": {
                            "type": "string",
                            "description": "Simple type label (Person, Organization, Location, Date, Concept, Event, Document, Law, Article, Product, ...)"
                        },
                        "description": {
                            "type": "string",
                            "description": "One-sentence description grounded in the source text"
                        }
                    },
                    "required": ["id", "name", "type", "description"]
                }
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "source_node_id": { "type": "string" },
                        "target_node_id": { "type": "string" },
                        "relationship_name": {
                            "type": "string",
                            "description": "snake_case verb or verb phrase"
                        },
                        "description": {
                            "type": "string",
                            "description": "One-sentence description of the relation"
                        }
                    },
                    "required": [
                        "source_node_id",
                        "target_node_id",
                        "relationship_name",
                        "description"
                    ]
                }
            }
        },
        "required": ["nodes", "edges"]
    })
}

/// Clean a parsed graph: trim strings, drop empty-id nodes, drop dangling
/// edges, and normalise IDs to lowercase `snake_case`.
fn sanitize_graph(mut graph: KnowledgeGraph) -> KnowledgeGraph {
    use std::collections::HashSet;

    // Normalise and deduplicate nodes by id within the same chunk.
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut cleaned_nodes: Vec<ExtractedNode> = Vec::with_capacity(graph.nodes.len());
    for mut node in graph.nodes.drain(..) {
        node.id = normalize_id(&node.id);
        node.name = node.name.trim().to_owned();
        node.node_type = node.node_type.trim().to_owned();
        node.description = node.description.trim().to_owned();

        if node.id.is_empty() || node.name.is_empty() {
            continue;
        }
        if seen_ids.insert(node.id.clone()) {
            cleaned_nodes.push(node);
        }
    }

    let valid_ids: HashSet<&str> = cleaned_nodes.iter().map(|n| n.id.as_str()).collect();
    let cleaned_edges: Vec<ExtractedEdge> = graph
        .edges
        .drain(..)
        .filter_map(|mut e| {
            e.source_node_id = normalize_id(&e.source_node_id);
            e.target_node_id = normalize_id(&e.target_node_id);
            e.relationship_name = sanitize_relationship(&e.relationship_name);
            e.description = e.description.trim().to_owned();

            if e.source_node_id.is_empty()
                || e.target_node_id.is_empty()
                || e.relationship_name.is_empty()
            {
                return None;
            }
            if e.source_node_id == e.target_node_id {
                // Self-loops on entities are rarely useful and clutter the graph.
                return None;
            }
            if !valid_ids.contains(e.source_node_id.as_str())
                || !valid_ids.contains(e.target_node_id.as_str())
            {
                // Dangling edge referencing a non-existent node.
                return None;
            }
            Some(e)
        })
        .collect();

    KnowledgeGraph {
        nodes: cleaned_nodes,
        edges: cleaned_edges,
    }
}

/// Normalise a node ID to lowercase `snake_case` ASCII alphanumerics, matching
/// the [`RelationType`](agent_rs_stores::types::RelationType) / `NodeLabel`
/// validation rules in the stores crate.
fn normalize_id(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_underscore = true; // suppress leading underscores
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            for lowered in ch.to_lowercase() {
                out.push(lowered);
            }
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

/// Turn a free-form relationship name into an ASCII uppercase `SNAKE_CASE`
/// identifier suitable for Neo4j's `RelationType`.
pub fn sanitize_relationship(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_underscore = true;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            for upper in ch.to_uppercase() {
                out.push(upper);
            }
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        "RELATED_TO".to_owned()
    } else {
        out
    }
}

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
        Err(_) => format!(
            "OpenAI API error (HTTP {}): {}",
            status.as_u16(),
            bytes_preview(bytes)
        ),
    }
}

fn bytes_preview(bytes: &[u8]) -> String {
    let end = bytes.len().min(200);
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
