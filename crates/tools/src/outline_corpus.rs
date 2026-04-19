//! `outline_corpus` — hierarchical TOC over Doc -> Section -> 1-line summary.
//!
//! Output is intended to be read by the LLM as a navigable index of what
//! the corpus contains. Joins Neo4j section structure (via [`Neo4jStore`])
//! with the summaries Qdrant collection (built by `ChunkSummarizer` at ingest).

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::neo4j::coverage::SectionRow;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Default cap on sections returned per document.
const DEFAULT_MAX_SECTIONS: usize = 200;

// ---- Args ----

/// Arguments for the `outline_corpus` tool.
#[derive(Debug, Deserialize)]
pub struct OutlineCorpusArgs {
    /// Optional document id. If omitted, returns outline for every ingested doc.
    pub doc_id: Option<String>,
    /// Optional max sections to return per document (default 200).
    pub max_sections: Option<usize>,
}

// ---- Tool struct ----

/// Hierarchical TOC tool — maps the corpus structure for the LLM before
/// it decides which sections to drill into.
#[derive(Serialize, Deserialize)]
pub struct OutlineCorpusTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    pg: Option<Arc<PgStore>>,
    #[serde(skip)]
    qdrant: Option<Arc<QdrantStore>>,
    #[serde(skip)]
    summaries_collection: Option<String>,
}

impl OutlineCorpusTool {
    /// Construct an [`OutlineCorpusTool`] with all backing stores wired up.
    ///
    /// `summaries_collection` is the name of the Qdrant collection that holds
    /// per-section summary points written by `ChunkSummarizer` at ingest time.
    pub fn new(
        neo4j: Arc<Neo4jStore>,
        pg: Arc<PgStore>,
        qdrant: Arc<QdrantStore>,
        summaries_collection: String,
    ) -> Self {
        Self {
            neo4j: Some(neo4j),
            pg: Some(pg),
            qdrant: Some(qdrant),
            summaries_collection: Some(summaries_collection),
        }
    }
}

impl Tool for OutlineCorpusTool {
    const NAME: &'static str = "outline_corpus";
    type Error = ToolError;
    type Args = OutlineCorpusArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Hierarchical OUTLINE / table-of-contents of the ingested corpus. \
                          Returns Document -> Section -> 1-line summary for each section. \
                          PREFER this BEFORE search_knowledge for any query that asks for \
                          ALL methods, an enumeration, a cheatsheet, exhaustive coverage, \
                          or 'TOATE' (all). Use it to discover what sections exist, then \
                          target search_knowledge or search_sections at the relevant ids. \
                          Omit doc_id to get the full corpus outline."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Optional document id; omit for full corpus outline."
                    },
                    "max_sections": {
                        "type": "integer",
                        "description": "Optional cap on sections per document (default 200)."
                    }
                },
                "required": []
            }),
        }
    }

    /// Build a hierarchical outline of the corpus, joining Neo4j section
    /// structure with Qdrant summary payloads.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when any backing store is uninitialised or
    /// when a store query fails.
    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let neo4j = self
            .neo4j
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: Neo4jStore missing".into()))?;
        let pg = self
            .pg
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: PgStore missing".into()))?;
        let qdrant = self
            .qdrant
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: QdrantStore missing".into()))?;
        let summaries_coll = self
            .summaries_collection
            .as_ref()
            .ok_or_else(|| {
                ToolError::McpClient("outline_corpus: summaries_collection missing".into())
            })?;

        let max = args.max_sections.unwrap_or(DEFAULT_MAX_SECTIONS);

        let docs = pg
            .list_ingested_documents()
            .await
            .map_err(|e| ToolError::McpClient(format!("outline_corpus: list docs: {e}")))?;

        if docs.is_empty() {
            return Ok("No documents have been ingested yet.".into());
        }

        let mut out = String::new();
        for doc in docs.iter().filter(|d| {
            args.doc_id.as_deref().is_none_or(|want| {
                d.document_id.to_string() == want || d.source == want
            })
        }) {
            out.push_str(&format!(
                "Document: {} ({})\n",
                doc.title.as_deref().unwrap_or("untitled"),
                doc.document_id
            ));

            let mut rows: Vec<SectionRow> = neo4j
                .list_sections_for_doc(&doc.document_id.to_string())
                .await
                .map_err(|e| ToolError::McpClient(format!("outline_corpus: sections: {e}")))?;

            // Honour caller-supplied cap.
            rows.truncate(max);

            for row in &rows {
                let summary =
                    lookup_section_summary(qdrant, summaries_coll, &row.section_id).await;
                let preamble_marker = if row.is_preamble { " [preamble]" } else { "" };
                out.push_str(&format!(
                    "  - [{}] {}{} -> {}\n",
                    row.section_id,
                    row.heading,
                    preamble_marker,
                    summary.as_deref().unwrap_or("[no summary]"),
                ));
            }
            out.push('\n');
        }
        Ok(out)
    }
}

/// Pull a 1-line summary for a section by scrolling the summaries Qdrant
/// collection with a payload filter on `section_id`. Returns the first
/// `summary_text` found (sections may have many summary chunks — we display
/// the first for the outline).
///
/// Returns `None` when no summary exists (failed/empty summary during
/// ingest, pre-this-fix corpus, or Qdrant unavailable). Caller substitutes
/// a placeholder.
async fn lookup_section_summary(
    qdrant: &Arc<QdrantStore>,
    collection: &str,
    section_id: &str,
) -> Option<String> {
    qdrant
        .first_summary_by_section(collection, section_id)
        .await
        .unwrap_or_else(|e| {
            tracing::debug!(
                section_id = section_id,
                error = %e,
                "outline_corpus: summary lookup failed; using placeholder"
            );
            None
        })
}
