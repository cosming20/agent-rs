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
const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// OpenAI chat completions endpoint.
const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default retry budget for transient failures.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default concurrency for batch extraction.
pub const DEFAULT_BATCH_CONCURRENCY: usize = 5;

/// Maximum characters of a chunk we forward to the LLM. Very large chunks are
/// truncated at the character boundary to keep request size predictable.
const MAX_CHUNK_CHARS: usize = 12_000;

/// System prompt for knowledge graph extraction.
///
/// Based on Cognee's `generate_graph_prompt.txt` but adapted for JSON-schema
/// structured outputs: the schema itself enforces the shape, so the prompt
/// focuses on *what* to extract and *how* to label nodes/relations.
const SYSTEM_PROMPT: &str = r#"You are extracting a knowledge graph from text.
Extract entities as nodes and relationships as edges.

Rules:
- Nodes represent entities and concepts (like Wikipedia pages).
- Use simple, consistent type labels: Person, Organization, Location, Date, Concept, Event, Document, Law, Article, Product. Avoid overly generic types like "Entity".
- Node IDs MUST be human-readable lowercase snake_case slugs (e.g. "albert_einstein", "world_war_two"). Never use integers. IDs are used for cross-chunk deduplication, so be consistent.
- Every node has a "name" field with the most complete human-readable form of the entity as it appears in text.
- Every node has a 1-sentence description grounded in the source text.
- Coreference: always use the most complete identifier for an entity. If the text mentions "Einstein" and later "Albert Einstein", use "albert_einstein" for both.
- Dates should be formatted YYYY-MM-DD when possible; otherwise year or month-year.
- Relationships use snake_case verbs or verb phrases (e.g. "founded_by", "located_in", "acted_in", "signed_on").
- Edges must reference node IDs that exist in the nodes list. Do not invent IDs.
- Focus on meaningful relations; ignore incidental mentions.
- Never escape quotes inside property values.
- If the text contains no clear entities, return empty lists.

Return ONLY valid JSON matching the provided schema."#;

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
    client: reqwest::Client,
    api_key: Arc<String>,
    model: Arc<String>,
    max_retries: u32,
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
    pub fn new(api_key: String, model: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
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
        }
    }

    /// Create a new extractor using the default model (`gpt-4o-mini`).
    pub fn with_default_model(api_key: String) -> Self {
        Self::new(api_key, DEFAULT_MODEL.to_owned())
    }

    /// Override the retry budget. The default is `DEFAULT_MAX_RETRIES`.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
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
                "chunk exceeds MAX_CHUNK_CHARS; truncating for extraction"
            );
            &trimmed[..end]
        } else {
            trimmed
        };

        let body = build_request_body(self.model.as_str(), payload_text);

        // Serialize once so retries can cheaply reuse the body.
        let body_bytes = serde_json::to_vec(&body).map_err(|e| RetrievalError::Ingestion {
            reason: format!("serialize graph extraction request: {e}"),
        })?;

        let response_bytes = self.call_api_with_retry(body_bytes).await?;

        // Parse the outer chat completion envelope.
        let chat: ChatResponse = match serde_json::from_slice(&response_bytes) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    preview = %bytes_preview(&response_bytes),
                    "failed to parse chat completion envelope; returning empty graph"
                );
                return Ok(KnowledgeGraph::default());
            }
        };

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
                            "graph extraction failed for chunk; using empty graph"
                        );
                        (idx, KnowledgeGraph::default())
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
