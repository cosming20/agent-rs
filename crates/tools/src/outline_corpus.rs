//! `outline_corpus` — hierarchical TOC over the pinned documents
//! (Doc → Section → 1-line summary).
//!
//! Output is intended to be read by the LLM as a navigable index of what
//! the pinned corpus contains. Joins Neo4j section structure (via
//! [`Neo4jStore`]) with the summaries Qdrant collection (written by the
//! ingestion worker).
//!
//! # Stateless pivot note
//!
//! After the 2026-04-21 pivot, the agent never consults a per-user catalogue
//! in Postgres. The pinned document set is provided per request by the web
//! layer as a list of MinIO object keys and flows into this tool through
//! [`OutlineCorpusTool::new_for_user`].

use std::sync::Arc;

use agent_rs_core::{DocHandle, ToolError, UserId};
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::neo4j::coverage::SectionRow;
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
    /// Optional MinIO object key. If omitted, returns an outline for every
    /// document pinned to the current turn.
    pub doc_id: Option<String>,
    /// Optional max sections to return per document (default 200).
    pub max_sections: Option<usize>,
}

// ---- Tool struct ----

/// Hierarchical TOC tool — maps the structure of the pinned documents
/// for the LLM before it decides which sections to drill into.
#[derive(Serialize, Deserialize)]
pub struct OutlineCorpusTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    qdrant: Option<Arc<QdrantStore>>,
    #[serde(skip)]
    summaries_collection: Option<String>,
    /// MinIO object keys pinned to the current turn. Outline is built only
    /// over these documents — agent-rs never sees the user's full library.
    #[serde(skip)]
    active_document_keys: Vec<String>,
    /// Tenant scope baked in at construction time. The LLM never sees or
    /// controls this field — it is excluded from the JSON schema.
    #[serde(skip, default = "default_unset_user_id")]
    user_id: UserId,
}

/// Sentinel `UserId` for contract-test deserialization path.
fn default_unset_user_id() -> UserId {
    UserId::try_new("_unset_test_only").expect("sentinel must pass validator")
}

impl OutlineCorpusTool {
    /// Construct an [`OutlineCorpusTool`] scoped to `user_id` and the
    /// per-turn pinned document set.
    ///
    /// `active_document_keys` is the list of MinIO object keys passed in the
    /// current `AskRequest`; `summaries_collection` names the Qdrant
    /// collection that holds per-section summary points.
    pub fn new_for_user(
        neo4j: Arc<Neo4jStore>,
        qdrant: Arc<QdrantStore>,
        summaries_collection: String,
        user_id: UserId,
        active_document_keys: Vec<String>,
    ) -> Self {
        Self {
            neo4j: Some(neo4j),
            qdrant: Some(qdrant),
            summaries_collection: Some(summaries_collection),
            active_document_keys,
            user_id,
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
            description: "Hierarchical OUTLINE / table-of-contents of the documents pinned to \
                          this turn. Returns Document -> Section -> 1-line summary for each \
                          section. PREFER this BEFORE search_knowledge for any query that asks \
                          for ALL methods, an enumeration, a cheatsheet, exhaustive coverage, \
                          or 'TOATE' (all). Use it to discover what sections exist, then \
                          target search_knowledge or search_sections at the relevant ids. \
                          Omit doc_id to get the outline of every pinned document."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Optional. Restrict the outline to ONE pinned document. Accepts any unique identifier: the full MinIO object key ('users/.../docs/<uuid>.pdf'), the UUID substring, or the filename. Ambiguous or unknown values return an error listing the known handles. Omit (or pass an empty string) for the full pinned-corpus outline."
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

    /// Build a hierarchical outline of the pinned corpus, joining Neo4j
    /// section structure with Qdrant summary payloads.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when any backing store is uninitialised or
    /// when a store query fails.
    #[tracing::instrument(skip(self), fields(tool = Self::NAME, user_id = %self.user_id))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        assert!(
            self.user_id.as_str() != "_unset_test_only",
            "outline_corpus: call() reached with sentinel user_id — use new_for_user()"
        );

        let neo4j = self
            .neo4j
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: Neo4jStore missing".into()))?;
        let qdrant = self
            .qdrant
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: QdrantStore missing".into()))?;
        let summaries_coll = self.summaries_collection.as_ref().ok_or_else(|| {
            ToolError::McpClient("outline_corpus: summaries_collection missing".into())
        })?;

        if self.active_document_keys.is_empty() {
            return Ok("No documents are pinned to this turn.".into());
        }

        let max = args.max_sections.unwrap_or(DEFAULT_MAX_SECTIONS);

        // Resolve a caller-supplied `doc_id` to a canonical key via the same
        // resolver the other tools use. Any prefix / UUID / filename that
        // uniquely identifies a pinned document is accepted; ambiguous or
        // unknown values return an LLM-friendly error string so the next
        // turn can retry with a specific handle instead of an empty outline.
        let doc_id_filter: Option<String> = match args
            .doc_id
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            None => None,
            Some(raw) => match DocHandle::resolve(raw, &self.active_document_keys) {
                Ok(handle) => Some(handle.into_inner()),
                Err(err) => {
                    tracing::warn!(
                        doc_id_input = %raw,
                        active_keys = ?self.active_document_keys,
                        error = %err,
                        "outline_corpus: doc_id could not be resolved"
                    );
                    return Ok(format!("outline_corpus: {err}"));
                }
            },
        };

        let mut out = String::new();
        for key in &self.active_document_keys {
            if let Some(ref want) = doc_id_filter {
                if key != want {
                    continue;
                }
            }

            // Emit the document handle in an explicitly copy-literal form
            // (`[id=<full-key>]`) so the LLM does not tokenise the path and
            // copy only a UUID substring on the next tool call. The leading
            // `filename:` line is a human-readable companion.
            let filename = key.rsplit('/').next().unwrap_or(key);
            out.push_str(&format!("Document [id={key}]\n"));
            out.push_str(&format!("  filename: {filename}\n"));

            let mut rows: Vec<SectionRow> =
                neo4j
                    .list_sections_for_doc(&self.user_id, key)
                    .await
                    .map_err(|e| ToolError::McpClient(format!("outline_corpus: sections: {e}")))?;

            rows.truncate(max);

            for row in &rows {
                let summary =
                    lookup_section_summary(qdrant, summaries_coll, &self.user_id, &row.section_id)
                        .await;
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

        if out.is_empty() {
            return Ok("No matching documents in the pinned set.".into());
        }

        Ok(out)
    }
}

/// Pull a 1-line summary for a section by scrolling the summaries Qdrant
/// collection with a payload filter on `section_id`. Returns the first
/// `summary_text` found.
///
/// Returns `None` when no summary exists (failed/empty summary during
/// ingest, or Qdrant unavailable). Caller substitutes a placeholder.
async fn lookup_section_summary(
    qdrant: &Arc<QdrantStore>,
    collection: &str,
    user_id: &UserId,
    section_id: &str,
) -> Option<String> {
    qdrant
        .first_summary_by_section(collection, user_id, section_id)
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

#[cfg(test)]
mod tests {
    /// Verifies that the `doc_id_filter` normalisation treats an empty string
    /// identically to `None`, so the LLM passing `{"doc_id": ""}` does not
    /// accidentally filter out all documents.
    #[test]
    fn empty_doc_id_string_means_no_filter() {
        let want_some_empty: Option<&str> = Some("");
        let normalized = want_some_empty.filter(|s| !s.is_empty());
        assert!(normalized.is_none(), "empty doc_id must normalize to None");

        let want_some_real: Option<&str> = Some("doc-abc");
        let normalized = want_some_real.filter(|s| !s.is_empty());
        assert_eq!(normalized, Some("doc-abc"));

        let want_none: Option<&str> = None;
        let normalized = want_none.filter(|s| !s.is_empty());
        assert!(normalized.is_none());
    }
}
