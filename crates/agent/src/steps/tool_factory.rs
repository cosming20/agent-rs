//! Tool factory -- builds per-request tool configuration for the Rig agent.
//!
//! Accepts shared references to the retrieval router, state store, and
//! configuration, then constructs concrete tool instances for each request.

use std::sync::Arc;
use tokio::sync::Mutex;

use agent_rs_core::config::{BraveConfig, FetchConfig, WebBudgetConfig};
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::UserId;
use agent_rs_retrieval::RetrievalRouter;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::SearchResult;
use agent_rs_tools::{
    FetchSectionChunksTool, FetchUrlTool, OutlineCorpusTool, RememberTool, SearchKnowledgeTool,
    SearchNewsTool, SearchSectionsTool, SearchWebTool, WebBudgetTracker,
};
use rig::tool::ToolDyn;

/// Per-request tool set: boxed dynamic tools, shared KB evidence collector,
/// and shared coverage evidence collector.
///
/// - The KB collector is drained into `ctx.kb_evidence` by `agent_runner`.
/// - The coverage collector is drained into `ctx.coverage_evidence` by
///   `agent_runner` after the tool-calling loop completes.
pub type ToolSet = (
    Vec<Box<dyn ToolDyn>>,
    Arc<Mutex<Vec<SearchResult>>>,
    Arc<Mutex<Vec<SearchResult>>>,
);

/// Builds per-request tool sets for the Rig agent.
///
/// Constructed once at startup with shared dependencies. Each call to
/// [`build`] creates a fresh set of tools (with a new budget tracker)
/// for a single query.
pub struct ToolFactory {
    router: Arc<RetrievalRouter>,
    store: Arc<dyn StateStore>,
    /// Neo4j graph store used by coverage-aware tools
    /// (`OutlineCorpusTool`, `FetchSectionChunksTool`, `SearchSectionsTool`).
    neo4j: Arc<Neo4jStore>,
    /// Qdrant vector store used by coverage-aware tools.
    qdrant: Arc<QdrantStore>,
    /// Name of the primary Qdrant collection (chunks).
    ///
    /// Shared with `search_knowledge`, `fetch_section_chunks`, and
    /// `search_sections`.
    default_collection: String,
    /// Name of the per-section summaries Qdrant collection.
    ///
    /// Written by `ChunkSummarizer` at ingest time and consumed by
    /// `outline_corpus` to display 1-line section summaries.
    summaries_collection: String,
    brave_config: BraveConfig,
    fetch_config: FetchConfig,
    web_budget_config: WebBudgetConfig,
    /// Shared HTTP client for web/news search tools.
    ///
    /// `reqwest::Client` is internally `Arc`-backed, so cloning is cheap.
    /// Sharing one client enables HTTP/2 connection reuse and avoids
    /// per-request TLS/DNS overhead.
    http_client: reqwest::Client,
}

impl ToolFactory {
    /// Create a new ToolFactory.
    ///
    /// # Parameters
    ///
    /// - `router` -- shared retrieval router for knowledge search.
    /// - `pg_store` -- shared Postgres handle for memory / telemetry tools.
    /// - `neo4j` -- shared Neo4j store for coverage-aware tools.
    /// - `qdrant` -- shared Qdrant store for coverage-aware tools.
    /// - `default_collection` -- primary Qdrant chunk collection name.
    /// - `summaries_collection` -- per-section summaries Qdrant collection name.
    /// - `brave_config` -- Brave Search API configuration.
    /// - `fetch_config` -- URL fetch configuration.
    /// - `web_budget_config` -- per-query web budget limits.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Orchestrator`] if the shared HTTP client cannot
    /// be constructed (e.g. invalid TLS configuration).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router: Arc<RetrievalRouter>,
        pg_store: Arc<PgStore>,
        neo4j: Arc<Neo4jStore>,
        qdrant: Arc<QdrantStore>,
        default_collection: String,
        summaries_collection: String,
        brave_config: BraveConfig,
        fetch_config: FetchConfig,
        web_budget_config: WebBudgetConfig,
    ) -> Result<Self, AgentError> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(brave_config.timeout_s))
            .build()
            .map_err(|e| {
                AgentError::Orchestrator(format!("failed to build shared HTTP client: {e}"))
            })?;
        let store: Arc<dyn StateStore> = pg_store as _;
        Ok(Self {
            router,
            store,
            neo4j,
            qdrant,
            default_collection,
            summaries_collection,
            brave_config,
            fetch_config,
            web_budget_config,
            http_client,
        })
    }

    /// Build a vector of boxed dynamic tools and the shared evidence collectors
    /// for a single query.
    ///
    /// Each call creates a fresh [`WebBudgetTracker`], KB evidence collector,
    /// and coverage evidence collector, ensuring per-query isolation.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `Vec<Box<dyn ToolDyn>>` -- tools to register with the Rig agent.
    /// - `Arc<Mutex<Vec<SearchResult>>>` -- KB evidence collector populated by
    ///   the `search_knowledge` tool.
    /// - `Arc<Mutex<Vec<SearchResult>>>` -- coverage evidence collector populated
    ///   by `fetch_section_chunks` and `search_sections` tools; drained into
    ///   `ctx.coverage_evidence` by `agent_runner` after execution.
    pub fn build(&self, user_id: &UserId, active_document_keys: Vec<String>) -> ToolSet {
        let budget = Arc::new(WebBudgetTracker::new(&self.web_budget_config));
        let evidence_collector: Arc<Mutex<Vec<SearchResult>>> = Arc::new(Mutex::new(Vec::new()));
        let coverage_collector: Arc<Mutex<Vec<SearchResult>>> = Arc::new(Mutex::new(Vec::new()));

        let tools: Vec<Box<dyn ToolDyn>> = vec![
            Box::new(SearchKnowledgeTool::new_for_user(
                Arc::clone(&self.router),
                Arc::clone(&evidence_collector),
                user_id.clone(),
            )),
            Box::new(SearchWebTool::new(
                Arc::clone(&budget),
                self.brave_config.clone(),
                self.http_client.clone(),
            )),
            Box::new(SearchNewsTool::new(
                Arc::clone(&budget),
                self.brave_config.clone(),
                self.http_client.clone(),
            )),
            Box::new(FetchUrlTool::new(
                self.fetch_config.clone(),
                Arc::clone(&budget),
            )),
            Box::new(RememberTool::new_for_user(
                Arc::clone(&self.store),
                user_id.clone(),
            )),
            // outline_corpus — hierarchical TOC (Doc→Section→summary) over
            // the documents pinned to this turn (`active_document_keys`),
            // so the LLM can discover what exists before deciding which
            // sections to pull.
            Box::new(OutlineCorpusTool::new_for_user(
                Arc::clone(&self.neo4j),
                Arc::clone(&self.qdrant),
                self.summaries_collection.clone(),
                user_id.clone(),
                active_document_keys,
            )),
            // fetch_section_chunks — exhaustive, non-ranked chunk retrieval
            // for a single known section_id. Wired with coverage_collector so
            // retrieved chunks reach ctx.coverage_evidence.
            Box::new(
                FetchSectionChunksTool::new_for_user(
                    Arc::clone(&self.neo4j),
                    Arc::clone(&self.qdrant),
                    self.default_collection.clone(),
                    user_id.clone(),
                )
                .with_coverage_collector(Arc::clone(&coverage_collector)),
            ),
            // search_sections — batched parallel retrieval over multiple
            // section_ids in one tool call (keeps exhaustive flows inside
            // the max_turns budget). Wired with coverage_collector so
            // retrieved chunks reach ctx.coverage_evidence.
            Box::new(
                SearchSectionsTool::new_for_user(
                    Arc::clone(&self.neo4j),
                    Arc::clone(&self.qdrant),
                    self.default_collection.clone(),
                    user_id.clone(),
                )
                .with_coverage_collector(Arc::clone(&coverage_collector)),
            ),
        ];

        (tools, evidence_collector, coverage_collector)
    }
}
