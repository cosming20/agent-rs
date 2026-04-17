//! Tool factory -- builds per-request tool configuration for the Rig agent.
//!
//! Accepts shared references to the retrieval router, state store, and
//! configuration, then constructs concrete tool instances for each request.

use std::sync::Arc;
use tokio::sync::Mutex;

use agent_rs_core::config::{BraveConfig, FetchConfig, WebBudgetConfig};
use agent_rs_core::types::UserId;
use agent_rs_retrieval::RetrievalRouter;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::SearchResult;
use agent_rs_tools::{
    DocumentInfoTool, FetchUrlTool, RememberTool, SearchKnowledgeTool, SearchNewsTool,
    SearchWebTool, WebBudgetTracker,
};
use rig::tool::ToolDyn;

/// Per-request tool set: boxed dynamic tools and shared evidence collector.
pub type ToolSet = (Vec<Box<dyn ToolDyn>>, Arc<Mutex<Vec<SearchResult>>>);

/// Builds per-request tool sets for the Rig agent.
///
/// Constructed once at startup with shared dependencies. Each call to
/// [`build`] creates a fresh set of tools (with a new budget tracker)
/// for a single query.
pub struct ToolFactory {
    router: Arc<RetrievalRouter>,
    store: Arc<dyn StateStore>,
    /// Concrete Postgres handle for tools that need PgStore-specific methods
    /// (currently `DocumentInfoTool` which calls `list_ingested_documents`,
    /// not part of the StateStore trait surface).
    pg_store: Arc<PgStore>,
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
    /// - `store` -- shared state store for the remember tool.
    /// - `brave_config` -- Brave Search API configuration.
    /// - `fetch_config` -- URL fetch configuration.
    /// - `web_budget_config` -- per-query web budget limits.
    pub fn new(
        router: Arc<RetrievalRouter>,
        pg_store: Arc<PgStore>,
        brave_config: BraveConfig,
        fetch_config: FetchConfig,
        web_budget_config: WebBudgetConfig,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(brave_config.timeout_s))
            .build()
            .expect("failed to build shared HTTP client");
        let store: Arc<dyn StateStore> = Arc::clone(&pg_store) as _;
        Self {
            router,
            store,
            pg_store,
            brave_config,
            fetch_config,
            web_budget_config,
            http_client,
        }
    }

    /// Build a vector of boxed dynamic tools and the shared evidence collector
    /// for a single query.
    ///
    /// Each call creates a fresh [`WebBudgetTracker`] and evidence collector,
    /// ensuring per-query isolation.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `Vec<Box<dyn ToolDyn>>` -- tools to register with the Rig agent.
    /// - `Arc<Mutex<Vec<SearchResult>>>` -- shared KB evidence collector
    ///   populated by the `search_knowledge` tool during execution.
    pub fn build(&self, user_id: &UserId) -> ToolSet {
        let budget = Arc::new(WebBudgetTracker::new(&self.web_budget_config));
        let evidence_collector: Arc<Mutex<Vec<SearchResult>>> = Arc::new(Mutex::new(Vec::new()));

        let tools: Vec<Box<dyn ToolDyn>> = vec![
            Box::new(SearchKnowledgeTool::new(
                Arc::clone(&self.router),
                Arc::clone(&evidence_collector),
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
            Box::new(RememberTool::new(Arc::clone(&self.store), user_id.clone())),
            // document_info — metadata about ingested docs (page count,
            // language, ingest date, etc.) for meta-questions that don't
            // need a content search.
            Box::new(DocumentInfoTool::new(Arc::clone(&self.pg_store))),
        ];

        (tools, evidence_collector)
    }
}
