//! `search_knowledge` tool -- queries the retrieval engine for evidence.
//!
//! This tool is the agent's primary interface to the knowledge base. It
//! calls [`RetrievalRouter::retrieve`] with the given query and strategy,
//! formats the resulting evidence documents, and accumulates raw
//! [`SearchResult`]s in a shared collector for downstream fusion.

use std::sync::{Arc, Mutex};

use agent_rs_core::{RetrievalStrategy, ToolError};
use agent_rs_retrieval::RetrievalRouter;
use agent_rs_stores::types::SearchResult;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Default number of documents to retrieve when the caller does not specify.
const DEFAULT_LIMIT: usize = 10;

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
            description: "Search the internal knowledge base for information relevant to the \
                          query. Returns formatted evidence with sources and scores. Use this \
                          tool to find factual information from ingested documents."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant knowledge."
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Retrieval strategy: semantic, keyword, hybrid, graph, graph_hybrid, rag_completion, summaries, temporal, graph_cot, graph_constrained. Defaults to hybrid.",
                        "enum": [
                            "semantic", "keyword", "hybrid", "graph",
                            "graph_hybrid", "rag_completion", "summaries",
                            "temporal", "graph_cot", "graph_constrained"
                        ]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let router = self.router.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_knowledge: router not initialized".into())
        })?;

        let strategy = match args.strategy.as_deref() {
            Some(s) => RetrievalStrategy::from_db_str(s)
                .map_err(|e| ToolError::WebSearch(format!("invalid retrieval strategy: {e}")))?,
            None => RetrievalStrategy::Hybrid,
        };

        let limit = args.limit.unwrap_or(DEFAULT_LIMIT);

        tracing::info!(
            query = %args.query,
            strategy = %strategy,
            limit,
            "executing knowledge search"
        );

        let evidence_pack = router
            .retrieve(&args.query, strategy, limit)
            .await
            .map_err(|e| ToolError::WebSearch(format!("retrieval failed: {e}")))?;

        // Accumulate raw results for downstream evidence fusion.
        if let Some(collector) = &self.evidence_collector {
            let mut collected = collector.lock().map_err(|_| {
                ToolError::McpClient(
                    "evidence collector mutex poisoned; cannot accumulate evidence".to_string(),
                )
            })?;
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
                doc.score.0,
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
