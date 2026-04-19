//! `fetch_section_chunks` — exhaustive, no-rank chunk retrieval per section.
//!
//! Bypasses ANN / BM25 ranking. Returns every chunk whose `section_id`
//! matches (or descends from) the requested section, ordered by
//! `chunk_index`. The agent uses this when it has decided exactly which
//! section it wants in full.
//!
//! # Flow
//!
//! 1. Optionally expand `section_id` to include descendants via Neo4j
//!    `PARENT_OF` traversal ([`Neo4jStore::list_descendant_sections`]).
//! 2. Fetch ordered chunk ids from Neo4j ([`Neo4jStore::chunks_in_sections`]).
//! 3. Retrieve full payloads (no vectors) from Qdrant
//!    ([`QdrantStore::retrieve_chunks_by_ids`]).
//! 4. Concatenate `chunk_text` fields in order, respecting `max_chars`.

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_core::types::Score;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::types::SearchResult;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;

// ---- Args ----

/// Arguments for the `fetch_section_chunks` tool.
#[derive(Debug, Deserialize)]
pub struct FetchSectionChunksArgs {
    /// Id of the section to retrieve (as returned by `outline_corpus`).
    pub section_id: String,
    /// When `true`, also fetches all descendant sections recursively.
    /// Useful for pulling an entire chapter (e.g., 5.1 + 5.1.1, 5.1.2, …).
    #[serde(default)]
    pub include_descendants: bool,
    /// Optional hard cap on total character output. Truncation is signalled
    /// with a `[truncated by max_chars]` trailer.
    pub max_chars: Option<usize>,
}

// ---- Tool struct ----

/// Exhaustive, non-ranked chunk retrieval for a single known section.
///
/// All store fields are `#[serde(skip)]` so the struct deserializes from
/// `{}` for contract tests (which only exercise `definition()`).
#[derive(Serialize, Deserialize)]
pub struct FetchSectionChunksTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    qdrant: Option<Arc<QdrantStore>>,
    #[serde(skip)]
    collection: Option<String>,
    /// Optional shared collector for populating `ctx.coverage_evidence`.
    ///
    /// When present, every retrieved [`SearchResult`] is pushed into this
    /// collector so the downstream [`CoverageVerifier`] can inspect what
    /// the tool actually fetched. Mirrors the `evidence_collector` pattern
    /// in [`SearchKnowledgeTool`].
    #[serde(skip)]
    coverage_collector: Option<Arc<Mutex<Vec<SearchResult>>>>,
}

impl FetchSectionChunksTool {
    /// Construct a wired-up [`FetchSectionChunksTool`].
    ///
    /// `collection` is the name of the Qdrant collection that holds chunk
    /// payloads (typically the same collection used by `search_knowledge`).
    pub fn new(neo4j: Arc<Neo4jStore>, qdrant: Arc<QdrantStore>, collection: String) -> Self {
        Self {
            neo4j: Some(neo4j),
            qdrant: Some(qdrant),
            collection: Some(collection),
            coverage_collector: None,
        }
    }

    /// Attach a shared coverage collector.
    ///
    /// Each [`SearchResult`] fetched by this tool is pushed into the collector
    /// so the orchestrator can drain it into `ctx.coverage_evidence` for the
    /// [`CoverageVerifier`] step.
    pub fn with_coverage_collector(
        mut self,
        collector: Arc<Mutex<Vec<SearchResult>>>,
    ) -> Self {
        self.coverage_collector = Some(collector);
        self
    }
}

impl Tool for FetchSectionChunksTool {
    const NAME: &'static str = "fetch_section_chunks";
    type Error = ToolError;
    type Args = FetchSectionChunksArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Exhaustive, NON-RANKED retrieval of ALL chunks in a section. \
                          Use when you need the complete content of a known section_id \
                          (typically discovered via outline_corpus). \
                          Set include_descendants=true to also pull subsections \
                          (e.g., section 5.1 → 5.1.1, 5.1.2). \
                          Output is ordered by chunk_index for linear reading. \
                          Do NOT use this for semantic search — prefer search_knowledge \
                          when you do not already know the exact section_id."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "Id of the section to retrieve (from outline_corpus)."
                    },
                    "include_descendants": {
                        "type": "boolean",
                        "default": false,
                        "description": "Also fetch all descendant sections recursively."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional cap on total output characters."
                    }
                },
                "required": ["section_id"]
            }),
        }
    }

    /// Retrieve every chunk in the requested section, ordered by chunk_index.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when any backing store is uninitialised or a
    /// store query fails.
    #[tracing::instrument(
        skip(self),
        fields(
            tool = Self::NAME,
            section_id = %args.section_id,
            include_descendants = args.include_descendants
        )
    )]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let neo4j = self.neo4j.as_ref().ok_or_else(|| {
            ToolError::McpClient("fetch_section_chunks: Neo4jStore missing".into())
        })?;
        let qdrant = self.qdrant.as_ref().ok_or_else(|| {
            ToolError::McpClient("fetch_section_chunks: QdrantStore missing".into())
        })?;
        let collection = self.collection.as_ref().ok_or_else(|| {
            ToolError::McpClient("fetch_section_chunks: collection missing".into())
        })?;

        // Expand to descendant sections when requested.
        let section_ids: Vec<String> = if args.include_descendants {
            let mut ids = neo4j
                .list_descendant_sections(&args.section_id)
                .await
                .map_err(|e| {
                    ToolError::McpClient(format!("fetch_section_chunks: descendants: {e}"))
                })?;
            // `list_descendant_sections` uses `*0..` so the root is included.
            // Ensure it is present even if the result was empty.
            if !ids.contains(&args.section_id) {
                ids.push(args.section_id.clone());
            }
            ids
        } else {
            vec![args.section_id.clone()]
        };

        // Fetch chunk ids, ordered by chunk_index (Neo4j query sorts them).
        let chunk_ids = neo4j
            .chunks_in_sections(&section_ids)
            .await
            .map_err(|e| ToolError::McpClient(format!("fetch_section_chunks: chunks: {e}")))?;

        if chunk_ids.is_empty() {
            return Ok(format!(
                "No chunks found in section {}.",
                args.section_id
            ));
        }

        // Retrieve full payloads from Qdrant (no vectors).
        let payloads = qdrant
            .retrieve_chunks_by_ids(collection, &chunk_ids)
            .await
            .map_err(|e| ToolError::McpClient(format!("fetch_section_chunks: retrieve: {e}")))?;

        // Accumulate raw results as SearchResults for downstream coverage
        // verification — mirrors the evidence_collector pattern in
        // SearchKnowledgeTool. Score is 1.0 for exhaustive (unranked) fetches.
        if let Some(collector) = &self.coverage_collector {
            let mut guard = collector.lock().await;
            for p in &payloads {
                guard.push(SearchResult {
                    id: p.id.clone(),
                    score: Score::new(1.0),
                    payload: p.payload.clone(),
                });
            }
        }

        // Concatenate chunk_text fields, respecting the optional char cap.
        let max_chars = args.max_chars.unwrap_or(usize::MAX);
        let mut out = String::new();
        let mut written: usize = 0;

        for p in payloads {
            let text = p
                .payload
                .get("chunk_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if written + text.len() > max_chars {
                out.push_str("[truncated by max_chars]\n");
                break;
            }

            written += text.len();
            out.push_str(text);
            out.push_str("\n\n");
        }

        Ok(out)
    }
}
