//! `search_sections` — batched per-section retrieval.
//!
//! Concurrent fan-out: one tool call accepts a list of section_ids and
//! issues per-section retrievals in parallel under a semaphore. Designed
//! to keep exhaustive flows inside `max_turns=5` — without this, the
//! agent would need N+1 turns to cover N sections.
//!
//! # Flow
//!
//! 1. Accept a `section_ids` list from the CoveragePlan.
//! 2. Fan out concurrently (bounded by [`DEFAULT_FAN_OUT_CONCURRENCY`]) —
//!    one [`fetch_one`] task per section.
//! 3. Each task: fetch chunk ids from Neo4j, retrieve payloads from Qdrant,
//!    concatenate `chunk_text` up to `per_section_max_chars`.
//! 4. Stitch results together with `=== section {id} ===` headers.

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::qdrant::QdrantStore;
use futures::future::join_all;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;

/// Maximum number of concurrent per-section retrievals.
const DEFAULT_FAN_OUT_CONCURRENCY: usize = 8;

/// Default hard cap per section, in characters, to bound total output size.
const DEFAULT_PER_SECTION_MAX_CHARS: usize = 8_000;

// ---- Args ----

/// Arguments for the `search_sections` tool.
#[derive(Debug, Deserialize)]
pub struct SearchSectionsArgs {
    /// List of section_ids to retrieve in a single batched call.
    /// Each id should match a value returned by `outline_corpus`.
    pub section_ids: Vec<String>,
    /// Optional per-section character cap. Defaults to
    /// [`DEFAULT_PER_SECTION_MAX_CHARS`].
    pub per_section_max_chars: Option<usize>,
}

// ---- Tool struct ----

/// BATCH retrieval over multiple section_ids in one tool call.
///
/// All store fields are `#[serde(skip)]` so the struct deserialises from
/// `{}` for contract tests (which only exercise `definition()`).
#[derive(Serialize, Deserialize)]
pub struct SearchSectionsTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    qdrant: Option<Arc<QdrantStore>>,
    #[serde(skip)]
    collection: Option<String>,
}

impl SearchSectionsTool {
    /// Construct a wired-up [`SearchSectionsTool`].
    ///
    /// `collection` is the name of the Qdrant collection that holds chunk
    /// payloads (the same collection used by `search_knowledge` and
    /// `fetch_section_chunks`).
    pub fn new(neo4j: Arc<Neo4jStore>, qdrant: Arc<QdrantStore>, collection: String) -> Self {
        Self {
            neo4j: Some(neo4j),
            qdrant: Some(qdrant),
            collection: Some(collection),
        }
    }
}

impl Tool for SearchSectionsTool {
    const NAME: &'static str = "search_sections";
    type Error = ToolError;
    type Args = SearchSectionsArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "BATCH retrieval over MULTIPLE section_ids in a single tool call. \
                          Use when CoveragePlan provides a target_sections list — issuing one \
                          search_sections call (rather than N fetch_section_chunks calls) keeps \
                          the agent under the max_turns budget. Returns concatenated chunks \
                          grouped by section heading. Each section is capped at \
                          per_section_max_chars to prevent context overflow."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "section_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "description": "List of section_ids to retrieve (from outline_corpus or CoveragePlan)."
                    },
                    "per_section_max_chars": {
                        "type": "integer",
                        "description": "Optional per-section character cap (default 8000)."
                    }
                },
                "required": ["section_ids"]
            }),
        }
    }

    /// Retrieve chunks for every requested section concurrently.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when any backing store is uninitialised or a
    /// store query fails at the fan-out level. Per-section errors are embedded
    /// inline in the output as `[error: …]` lines so the agent can act on
    /// partial results.
    #[tracing::instrument(
        skip(self),
        fields(tool = Self::NAME, n_sections = args.section_ids.len())
    )]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let neo4j = self
            .neo4j
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("search_sections: Neo4jStore missing".into()))?;
        let qdrant = self
            .qdrant
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("search_sections: QdrantStore missing".into()))?;
        let collection = self.collection.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_sections: collection missing".into())
        })?;

        if args.section_ids.is_empty() {
            return Ok("No section_ids provided.".into());
        }

        let sem = Arc::new(Semaphore::new(DEFAULT_FAN_OUT_CONCURRENCY));
        let max_chars = args
            .per_section_max_chars
            .unwrap_or(DEFAULT_PER_SECTION_MAX_CHARS);

        let futs = args.section_ids.iter().map(|sid| {
            let sem = Arc::clone(&sem);
            let neo4j = Arc::clone(neo4j);
            let qdrant = Arc::clone(qdrant);
            let collection = collection.clone();
            let sid = sid.clone();
            async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| format!("semaphore closed: {e}"))?;
                fetch_one(&neo4j, &qdrant, &collection, &sid, max_chars).await
            }
        });

        let results = join_all(futs).await;

        let mut out = String::new();
        for (sid, res) in args.section_ids.iter().zip(results.iter()) {
            out.push_str(&format!("=== section {sid} ===\n"));
            match res {
                Ok(text) => out.push_str(text),
                Err(e) => out.push_str(&format!("[error: {e}]\n")),
            }
            out.push('\n');
        }

        Ok(out)
    }
}

/// Fetch all chunks for one `section_id`.
///
/// Returns the concatenated `chunk_text` fields from Qdrant, truncated at
/// `max_chars`. Per-section errors are propagated as `Err(String)` and
/// embedded inline by the caller.
async fn fetch_one(
    neo4j: &Neo4jStore,
    qdrant: &QdrantStore,
    collection: &str,
    section_id: &str,
    max_chars: usize,
) -> Result<String, String> {
    let chunk_ids = neo4j
        .chunks_in_sections(&[section_id.to_string()])
        .await
        .map_err(|e| format!("neo4j chunks_in_sections({section_id}): {e}"))?;

    if chunk_ids.is_empty() {
        return Ok(format!("(no chunks in {section_id})\n"));
    }

    let payloads = qdrant
        .retrieve_chunks_by_ids(collection, &chunk_ids)
        .await
        .map_err(|e| format!("qdrant retrieve_chunks_by_ids({section_id}): {e}"))?;

    let mut s = String::new();
    let mut written = 0usize;

    for p in payloads {
        let text = p
            .payload
            .get("chunk_text")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if written + text.len() > max_chars {
            s.push_str("[truncated by per_section_max_chars]\n");
            break;
        }

        written += text.len();
        s.push_str(text);
        s.push_str("\n\n");
    }

    Ok(s)
}
