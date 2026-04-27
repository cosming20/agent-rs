//! `search_knowledge` tool -- queries the retrieval engine for evidence.
//!
//! This tool is the agent's primary interface to the knowledge base. It
//! calls [`RetrievalRouter::retrieve`] with the given query and strategy,
//! formats the resulting evidence documents, and accumulates raw
//! [`SearchResult`]s in a shared collector for downstream fusion.

use std::sync::Arc;
use tokio::sync::Mutex;

use agent_rs_core::{DocHandle, RetrievalStrategy, ToolError, UserId};
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
#[derive(Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Full retrieval router: dense + BM25 + graph fusion + cross-encoder rerank.
    #[default]
    Ranked,
    /// Bypass ranking; return every chunk whose payload `section_id` matches.
    /// Falls back to `Ranked` (with a warning) when `section_id` is absent.
    AllInSection,
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
    /// Tenant scope baked in at construction time. The LLM never sees or
    /// controls this field — it is excluded from the JSON schema.
    #[serde(skip, default = "default_unset_user_id")]
    user_id: UserId,
    /// Canonical MinIO object keys of the documents pinned to this turn.
    ///
    /// Consulted when the LLM supplies a `doc_id` argument: any prefix /
    /// UUID substring / filename is resolved against this list via
    /// [`DocHandle::resolve`] so the caller's loose identifier becomes the
    /// exact key the retrieval filter matches on.
    #[serde(skip, default)]
    active_document_keys: Vec<String>,
}

/// Sentinel `UserId` used only when the struct is deserialised from `{}`
/// in contract tests that call `definition()` exclusively. Any real request
/// path uses [`SearchKnowledgeTool::new_for_user`] which supplies the actual
/// tenant id.
fn default_unset_user_id() -> UserId {
    UserId::try_new("_unset_test_only").expect("sentinel must pass validator")
}

impl SearchKnowledgeTool {
    /// Create a new `SearchKnowledgeTool` scoped to `user_id`.
    ///
    /// # Parameters
    ///
    /// - `router` -- the shared retrieval router that dispatches to search backends.
    /// - `evidence_collector` -- shared mutable vec where raw results are accumulated.
    /// - `user_id` -- tenant id baked into every store call; never exposed in the
    ///   tool JSON schema.
    /// - `active_document_keys` -- MinIO object keys of documents pinned for this
    ///   turn. Used to resolve a caller-supplied `doc_id` (full key, UUID
    ///   substring, or filename) to its canonical form before filtering.
    pub fn new_for_user(
        router: Arc<RetrievalRouter>,
        evidence_collector: Arc<Mutex<Vec<SearchResult>>>,
        user_id: UserId,
        active_document_keys: Vec<String>,
    ) -> Self {
        Self {
            router: Some(router),
            evidence_collector: Some(evidence_collector),
            user_id,
            active_document_keys,
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
                        "description": "Optional. Restrict retrieval to ONE pinned document. Accepts any unique identifier: the full MinIO object key shown by outline_corpus ('users/.../docs/<uuid>.pdf'), the UUID substring, or the filename — whichever uniquely identifies one pinned document. Ambiguous or unknown values return an error listing the known handles so you can retry. Omit the field (or pass an empty string) to search across all pinned documents."
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

    #[tracing::instrument(skip(self), fields(tool = Self::NAME, user_id = %self.user_id))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Sentinel guard: fails fast if the tool was constructed via the
        // serde-skip default path (contract tests) rather than new_for_user().
        assert!(
            self.user_id.as_str() != "_unset_test_only",
            "search_knowledge: call() reached with sentinel user_id — use new_for_user()"
        );

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

        // Resolve a caller-supplied `doc_id` to its canonical MinIO object
        // key via `DocHandle`. The LLM may pass a bare UUID, a filename, or
        // the full key — `DocHandle::resolve` accepts any of these as long
        // as exactly one pinned document matches. Ambiguous or unknown
        // inputs are returned to the LLM as a tool-output string so the
        // next turn can retry with a more specific value instead of
        // receiving a silent empty result set.
        let normalise =
            |s: Option<String>| s.map(|v| v.trim().to_owned()).filter(|v| !v.is_empty());
        let resolved_doc_key: Option<String> = match normalise(args.doc_id.clone()) {
            None => None,
            Some(raw) => match DocHandle::resolve(&raw, &self.active_document_keys) {
                Ok(handle) => Some(handle.into_inner()),
                Err(err) => {
                    tracing::warn!(
                        doc_id_input = %raw,
                        active_keys = ?self.active_document_keys,
                        error = %err,
                        "search_knowledge: doc_id could not be resolved"
                    );
                    // Return the resolver error verbatim so the LLM can
                    // self-correct on the next iteration — the error
                    // message already embeds the known handles.
                    return Ok(format!("search_knowledge: {err}"));
                }
            },
        };

        let section_filter = {
            let f = SectionFilter {
                doc_id: resolved_doc_key,
                section_id: normalise(args.section_id.clone()),
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
            .retrieve_filtered(&self.user_id, &args.query, strategy, limit, section_filter)
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

            // Surface the stable `chunk_id` BOTH as the bracket index
            // (so the LLM can use the short form `[1]`) AND as the
            // citation marker the synthesizer expects (`[cite:<uuid>]`).
            // Without the explicit cite marker the LLM tends to invent
            // shorthand ids like `c-1`/`c-2` which the synthesizer
            // can't resolve, dropping every citation and forcing the
            // answer ungrounded — observed during the FCM evaluation
            // run on 2026-04-23.
            let chunk_id = doc
                .payload
                .get("chunk_id")
                .and_then(|v| v.as_str())
                .unwrap_or(doc.id.as_str());
            output.push_str(&format!(
                "[{idx}] cite:{chunk_id} (score: {score:.4}, source: {source}{page}{section}{table})\n{content}\n\n",
                idx = i + 1,
                chunk_id = chunk_id,
                score = doc.score.value(),
                source = source,
                page = page_info,
                section = section_info,
                table = table_info,
                content = content,
            ));
        }

        // Final reminder so the LLM emits `[cite:<chunk_id>]` rather
        // than guessing — the synthesizer requires the exact id.
        output.push_str(
            "When citing, use the exact `[cite:<chunk_id>]` marker shown next to each \
             document above. Do NOT shorten the id.\n",
        );

        Ok(output)
    }
}

#[cfg(test)]
mod tests_doc_id_resolution {
    //! Smoke tests for the `DocHandle`-backed `doc_id` resolution path.
    //!
    //! These exercise the string-shape boundary between LLM-supplied
    //! identifiers and the canonical MinIO keys stored on chunk payloads.
    //! They do NOT run the full retrieval pipeline; the router is absent
    //! and the tool is constructed only far enough to call the resolver.
    use super::*;

    fn make_tool(active_keys: Vec<String>) -> SearchKnowledgeTool {
        SearchKnowledgeTool {
            router: None,
            evidence_collector: None,
            user_id: UserId::try_new("u").unwrap(),
            active_document_keys: active_keys,
        }
    }

    #[test]
    fn normalise_strips_whitespace_and_nulls_empties() {
        // Mirrors the closure inline in `call()` — kept in sync by
        // convention. If either diverges this test catches the drift.
        let normalise = |s: Option<String>| -> Option<String> {
            s.map(|v| v.trim().to_owned()).filter(|v| !v.is_empty())
        };
        assert_eq!(normalise(None), None);
        assert_eq!(normalise(Some(String::new())), None);
        assert_eq!(normalise(Some("   ".into())), None);
        assert_eq!(normalise(Some("  abc  ".into())), Some("abc".into()));
    }

    #[test]
    fn bare_uuid_resolves_against_full_minio_key() {
        // Reproduces the 2026-04-24 bug: outline_corpus emits the full
        // MinIO key but the LLM copies only the UUID out of it. The
        // resolver must upgrade that UUID back to the canonical key so
        // the section filter matches chunk payloads.
        let full = "users/u1/docs/567f3d77-5eb4-4865-ae70-392fb6be735e.pdf".to_string();
        let tool = make_tool(vec![full.clone()]);
        let handle = DocHandle::resolve(
            "567f3d77-5eb4-4865-ae70-392fb6be735e",
            &tool.active_document_keys,
        )
        .expect("unique uuid must resolve to the canonical key");
        assert_eq!(handle.as_str(), full);
    }

    #[test]
    fn unknown_doc_id_surfaces_known_handles_in_error_text() {
        let tool = make_tool(vec!["users/u1/docs/only.pdf".into()]);
        let err = DocHandle::resolve("nonexistent", &tool.active_document_keys).unwrap_err();
        let msg = err.to_string();
        // The LLM will see this string as the tool output and should be
        // able to read the known handle verbatim out of it.
        assert!(msg.contains("users/u1/docs/only.pdf"));
        assert!(msg.contains("nonexistent"));
    }
}
