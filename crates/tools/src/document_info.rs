//! `document_info` tool — retrieves document-level metadata.
//!
//! Answers questions about ingested documents: page count, table count,
//! language, title, ingestion status, etc.  Data comes from the Postgres
//! `ingested_documents` table populated during Docling ingestion.
//!
//! This tool lets the agent answer "how many pages does the document have?"
//! or "what language is the document in?" without searching the full corpus.

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_stores::postgres::{IngestedDocumentRow, PgStore};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---- Args ----

/// Arguments for the `document_info` tool.
#[derive(Debug, Deserialize)]
pub struct DocumentInfoArgs {
    /// Optional document ID to look up. If omitted, returns info about
    /// all ingested documents.
    pub doc_id: Option<String>,
}

// ---- Tool struct ----

/// Retrieves document-level metadata from the ingestion state store.
#[derive(Serialize, Deserialize)]
pub struct DocumentInfoTool {
    #[serde(skip)]
    pg_store: Option<Arc<PgStore>>,
}

impl DocumentInfoTool {
    pub fn new(pg_store: Arc<PgStore>) -> Self {
        Self {
            pg_store: Some(pg_store),
        }
    }
}

impl Tool for DocumentInfoTool {
    const NAME: &'static str = "document_info";
    type Error = ToolError;
    type Args = DocumentInfoArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Inventory / metadata tool for the ingested corpus. Returns per-document \
                          facts: page count, table count, picture count, language, title, file \
                          type, ingestion status, Docling version. ALWAYS prefer this over \
                          search_knowledge when the user asks: what documents / sources are \
                          available, is X ingested, list the corpus, how many pages in doc Y, \
                          when was Z ingested, is the corpus empty. Do NOT use this tool to \
                          search document content — for that, use search_knowledge. Omit doc_id \
                          to list every ingested document."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Optional document ID. If omitted, returns info about all ingested documents."
                    }
                },
                "required": []
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let pg = self
            .pg_store
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("document_info: PgStore not initialized".into()))?;

        // Query ingested_documents table.
        let rows = pg
            .list_ingested_documents()
            .await
            .map_err(|e| ToolError::McpClient(format!("query ingested_documents: {e}")))?;

        if rows.is_empty() {
            return Ok("No documents have been ingested yet.".to_string());
        }

        // Filter by doc_id if specified.
        let filtered: Vec<_> = if let Some(ref doc_id) = args.doc_id {
            rows.iter()
                .filter(|r| r.document_id.to_string() == *doc_id || r.source == *doc_id)
                .collect()
        } else {
            rows.iter().collect()
        };

        if filtered.is_empty() {
            return Ok(format!(
                "No document found with ID '{}'. {} documents are ingested.",
                args.doc_id.unwrap_or_default(),
                rows.len()
            ));
        }

        let mut output = format!("Ingested documents ({}):\n\n", filtered.len());

        for doc in filtered {
            output.push_str(&format!(
                "Document: {}\n\
                 - Source: {}\n\
                 - Status: {}\n\
                 - Pages: {}\n\
                 - Chunks: {}\n\
                 - Tables: {}\n\
                 - Pictures: {}\n\
                 - Language: {}\n\
                 - Title: {}\n\
                 - MIME type: {}\n\
                 - Docling version: {}\n\
                 - SHA-256: {}\n\
                 - Ingested at: {}\n\n",
                doc.document_id,
                doc.source,
                doc.ingest_status,
                doc.n_pages
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                doc.n_chunks
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                doc.n_tables
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                doc.n_pictures
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                doc.language.as_deref().unwrap_or("unknown"),
                doc.title.as_deref().unwrap_or("untitled"),
                doc.mimetype.as_deref().unwrap_or("unknown"),
                doc.docling_version.as_deref().unwrap_or("unknown"),
                doc.sha256,
                doc.started_at,
            ));
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// resolve_document — free function usable by CoveragePlanner (Task 4.1)
// ---------------------------------------------------------------------------

/// Find the first ingested document whose `source`, `title`, or
/// `document_id` (string form) contains `name_or_alias`
/// (case-insensitive substring). Returns `None` when no document matches.
///
/// Used by `CoveragePlanner` (Task 4.1) to resolve user-supplied document
/// references like "the segmentare PDF" into a concrete `document_id`.
///
/// # Examples
///
/// ```rust,ignore
/// let matched = resolve_document(&docs, "segmentare");
/// assert!(matched.is_some());
/// ```
pub fn resolve_document<'a>(
    docs: &'a [IngestedDocumentRow],
    name_or_alias: &str,
) -> Option<&'a IngestedDocumentRow> {
    let needle = name_or_alias.to_lowercase();
    docs.iter().find(|d| {
        d.source.to_lowercase().contains(&needle)
            || d.title
                .as_deref()
                .map_or(false, |t| t.to_lowercase().contains(&needle))
            || d.document_id.to_string().to_lowercase().contains(&needle)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use chrono::NaiveDateTime;
    use uuid::Uuid;

    use super::*;

    /// Build a minimal [`IngestedDocumentRow`] for testing.
    fn row(source: &str, title: Option<&str>) -> IngestedDocumentRow {
        let now = NaiveDateTime::default();
        IngestedDocumentRow {
            document_id: Uuid::new_v4(),
            sha256: "abc123".to_string(),
            source: source.to_string(),
            ingest_status: "complete".to_string(),
            ingest_run_id: Uuid::new_v4(),
            failure_reason: None,
            docling_version: None,
            docling_core_version: None,
            schema_version: None,
            n_pages: None,
            n_chunks: None,
            n_tables: None,
            n_pictures: None,
            n_formulas: None,
            language: None,
            title: title.map(String::from),
            mimetype: None,
            parse_latency_ms: None,
            total_ingest_latency_ms: None,
            started_at: now,
            completed_at: None,
            updated_at: now,
        }
    }

    #[test]
    fn resolve_document_matches_by_source_substring_case_insensitive() {
        let docs = vec![
            row("segmentare.pdf", Some("Segmentare imagini")),
            row("histograme.pdf", None),
        ];
        let r = resolve_document(&docs, "segmentare").expect("expected match");
        assert_eq!(r.source, "segmentare.pdf");
    }

    #[test]
    fn resolve_document_matches_by_title_substring() {
        let docs = vec![row("foo.pdf", Some("Image Segmentation"))];
        let r = resolve_document(&docs, "image segmentation").expect("expected match");
        assert_eq!(r.source, "foo.pdf");
    }

    #[test]
    fn resolve_document_returns_none_when_no_match() {
        let docs = vec![row("foo.pdf", None)];
        assert!(resolve_document(&docs, "irrelevant").is_none());
    }
}
