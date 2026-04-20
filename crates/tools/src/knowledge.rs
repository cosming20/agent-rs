//! `search_knowledge` tool -- queries the retrieval engine for evidence.
//!
//! This tool is the agent's primary interface to the knowledge base. It
//! calls [`RetrievalRouter::retrieve`] with the given query and strategy,
//! formats the resulting evidence documents, and accumulates raw
//! [`SearchResult`]s in a shared collector for downstream fusion.

use std::sync::Arc;
use tokio::sync::Mutex;

use agent_rs_core::{RetrievalStrategy, ToolError};
use agent_rs_retrieval::{RetrievalRouter, router::SectionFilter};
use agent_rs_stores::types::SearchResult;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Default number of documents to retrieve when the caller does not specify.
const DEFAULT_LIMIT: usize = 10;

// ---- SearchMode ----

/// How to retrieve chunks for a `search_knowledge` call.
///
/// `Ranked` is the default — runs the full retrieval router (vector +
/// keyword + graph fusion + rerank). `AllInSection` bypasses ranking and
/// returns every chunk in the requested `section_id` (requires `section_id`
/// to be set; otherwise falls back to `Ranked` with a tracing warn).
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Full retrieval router: dense + BM25 + graph fusion + cross-encoder rerank.
    Ranked,
    /// Bypass ranking; return every chunk whose payload `section_id` matches.
    /// Falls back to `Ranked` (with a warning) when `section_id` is absent.
    AllInSection,
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Ranked
    }
}

// ---- Args ----

/// Arguments for the `search_knowledge` tool.
#[derive(Debug, Deserialize)]
pub struct SearchKnowledgeArgs {
    /// The search query string.
    pub query: String,
    /// Optional retrieval strategy. Defaults to `"hybrid"` if not specified.
    pub strategy: Option<String>,
    /// Maximum number of results to return. Defaults to 10.
    pub limit: Option<usize>,
    /// Restrict retrieval to chunks whose payload `section_id` matches.
    /// Combine with `mode = AllInSection` to bypass ranking entirely.
    #[serde(default)]
    pub section_id: Option<String>,
    /// Restrict retrieval to chunks whose payload `doc_id` matches.
    /// Useful for "in document X, …" multi-doc queries.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// How to retrieve. Default `Ranked`. `AllInSection` bypasses ranking.
    #[serde(default)]
    pub mode: SearchMode,
}

// ---- Tool struct ----

/// Searches the internal knowledge base via the retrieval engine.
///
/// The tool formats evidence documents as numbered items with source
/// metadata and relevance scores for the LLM to cite.
///
/// # Evidence accumulation
///
/// Raw [`SearchResult`]s are pushed into `evidence_collector` so the
/// orchestrator can fuse them with web results in the evidence fusion step.
#[derive(Serialize, Deserialize)]
pub struct SearchKnowledgeTool {
    #[serde(skip)]
    router: Option<Arc<RetrievalRouter>>,
    #[serde(skip)]
    evidence_collector: Option<Arc<Mutex<Vec<SearchResult>>>>,
}

impl SearchKnowledgeTool {
    /// Create a new `SearchKnowledgeTool`.
    ///
    /// # Parameters
    ///
    /// - `router` -- the shared retrieval router that dispatches to search backends.
    /// - `evidence_collector` -- shared mutable vec where raw results are accumulated.
    pub fn new(
        router: Arc<RetrievalRouter>,
        evidence_collector: Arc<Mutex<Vec<SearchResult>>>,
    ) -> Self {
        Self {
            router: Some(router),
            evidence_collector: Some(evidence_collector),
        }
    }
}

impl Tool for SearchKnowledgeTool {
    const NAME: &'static str = "search_knowledge";
    type Error = ToolError;
    type Args = SearchKnowledgeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Primary knowledge-retrieval tool. Searches the ingested document \
                          corpus and returns the top-matching chunks as numbered evidence \
                          (with page, section, score). Prefer this over general knowledge for \
                          any domain question. Use a focused query in the SAME language as \
                          the ingested corpus (not necessarily the user's language). One call \
                          is usually enough; do not re-query unless the first result is \
                          clearly off-topic."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused retrieval query — typically 2-8 meaningful tokens. For definition questions, use the concept noun plus a hint like 'definiție' / 'definition'. Copy identifiers verbatim (section numbers, codes)."
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Retrieval strategy. Default 'graph_hybrid' is best for 90% of queries (dense + BM25 + graph traversal + triplets + cross-encoder rerank). Use 'keyword' ONLY when the query is an exact-string lookup (section 2.1, code A.2.2). Use 'graph_cot' for multi-hop reasoning that requires iterative graph traversal. Avoid 'semantic' unless debugging.",
                        "enum": [
                            "semantic", "keyword", "hybrid", "graph",
                            "graph_hybrid", "rag_completion", "summaries",
                            "temporal", "graph_cot", "graph_constrained"
                        ]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max evidence chunks to return (default 10). Use 5 for focused definition queries, 10 for exploratory, 15-20 only when you need breadth."
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Optional. Restrict retrieval to chunks from ONE specific document. Pass the UUID you discovered via document_info or outline_corpus. Use when the user names a document ('in the segmentare PDF', 'from doc X') or compares documents."
                    },
                    "section_id": {
                        "type": "string",
                        "description": "Optional. Restrict retrieval to chunks from ONE specific section. Pass a section UUID from outline_corpus. Prefer fetch_section_chunks if you want the WHOLE section; this field narrows a ranked query to that section."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["ranked", "all_in_section"],
                        "description": "Retrieval mode. Default 'ranked' runs the full router (vector + BM25 + graph + rerank). Use 'all_in_section' with a section_id to bypass ranking entirely — returns every chunk in that section, unranked."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // AllInSection mode: if a section_id is provided, warn and fall back
        // to Ranked path until Task 4.6 wires the Neo4j handle.
        // TODO(Task 4.6): when Neo4jStore handle is available, route
        // AllInSection through Neo4jStore::chunks_in_sections + QdrantStore.
        if args.mode == SearchMode::AllInSection {
            if args.section_id.is_some() {
                tracing::warn!(
                    section_id = ?args.section_id,
                    "AllInSection mode not yet fully implemented; falling back to Ranked"
                );
            } else {
                tracing::warn!(
                    mode = "all_in_section",
                    section_id = "missing",
                    fallback = "ranked",
                    "AllInSection mode requested without section_id; falling back to Ranked"
                );
            }
        }

        let router = self.router.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_knowledge: router not initialized".into())
        })?;

        // Default aligns with the system-prompt guidance: `graph_hybrid` is
        // the strongest single-call strategy (dense + BM25 + graph + triplets
        // + cross-encoder rerank) and handles the 90% case.
        let strategy = match args.strategy.as_deref() {
            Some(s) => RetrievalStrategy::from_db_str(s)
                .map_err(|e| ToolError::McpClient(format!("invalid retrieval strategy: {e}")))?,
            None => RetrievalStrategy::GraphHybrid,
        };

        let limit = args.limit.unwrap_or(DEFAULT_LIMIT);

        // Build the hard post-fusion predicate from the caller-supplied
        // doc_id / section_id fields (Task 3.6 — Codex audit gap #10).
        // An empty filter is passed as `None` so the router skips the retain
        // loop entirely in the common case.
        let section_filter = {
            let f = SectionFilter {
                doc_id: args.doc_id.clone(),
                section_id: args.section_id.clone(),
            };
            if f.is_empty() { None } else { Some(f) }
        };

        tracing::info!(
            query = %args.query,
            strategy = %strategy,
            limit,
            doc_id = ?args.doc_id,
            section_id = ?args.section_id,
            "executing knowledge search"
        );

        let evidence_pack = router
            .retrieve_filtered(&args.query, strategy, limit, section_filter)
            .await
            .map_err(|e| ToolError::McpClient(format!("knowledge retrieval failed: {e}")))?;

        // Accumulate raw results for downstream evidence fusion.
        if let Some(collector) = &self.evidence_collector {
            let mut collected = collector.lock().await;
            collected.extend(evidence_pack.documents.iter().cloned());
        }

        if evidence_pack.is_empty() {
            return Ok("No relevant documents found in the knowledge base.".to_string());
        }

        // Format evidence for the LLM.
        let mut output = format!(
            "Found {} documents (strategy: {}):\n\n",
            evidence_pack.len(),
            strategy,
        );

        for (i, doc) in evidence_pack.documents.iter().enumerate() {
            let source = doc
                .payload
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let content = doc
                .payload
                .get("chunk_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Expose structural metadata so the agent can cite page numbers,
            // section titles, and content type in its answer.
            let page_start = doc.payload.get("page_start").and_then(|v| v.as_u64());
            let page_end = doc.payload.get("page_end").and_then(|v| v.as_u64());
            let section = doc
                .payload
                .get("section_title")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let has_table = doc
                .payload
                .get("has_table")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let page_info = match (page_start, page_end) {
                (Some(s), Some(e)) if s == e => format!(", page: {s}"),
                (Some(s), Some(e)) => format!(", pages: {s}-{e}"),
                (Some(s), None) => format!(", page: {s}"),
                _ => String::new(),
            };
            let section_info = if section.is_empty() {
                String::new()
            } else {
                format!(", section: \"{section}\"")
            };
            let table_info = if has_table { ", contains_table" } else { "" };

            output.push_str(&format!(
                "[{}] (score: {:.4}, source: {}{}{}{})\n{}\n\n",
                i + 1,
                doc.score.value(),
                source,
                page_info,
                section_info,
                table_info,
                content,
            ));
        }

        Ok(output)
    }
}
