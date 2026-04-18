//! `search_web` and `search_news` tools.
//!
//! Both tools check the [`WebBudgetTracker`] before executing and return
//! a budget-exhausted error if the per-query limit has been reached.
//!
//! Web searches use the Brave Search API (GET /web/search).
//! News searches use the Brave News API (GET /news/search).

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_core::config::BraveConfig;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::budget::WebBudgetTracker;

// ---- Brave API response types ----

/// A single web search result from the Brave API.
#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: Option<String>,
    url: Option<String>,
    description: Option<String>,
}

/// Top-level response from the Brave web search API.
#[derive(Debug, Deserialize)]
struct BraveWebResponse {
    web: Option<BraveWebResults>,
}

/// Container for web search results.
#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Option<Vec<BraveWebResult>>,
}

/// A single news result from the Brave API.
#[derive(Debug, Deserialize)]
struct BraveNewsResult {
    title: Option<String>,
    url: Option<String>,
    description: Option<String>,
}

/// Top-level response from the Brave news search API.
#[derive(Debug, Deserialize)]
struct BraveNewsResponse {
    results: Option<Vec<BraveNewsResult>>,
}

// ---- search_web ----

/// Arguments for the `search_web` tool.
#[derive(Debug, Deserialize)]
pub struct SearchWebArgs {
    /// The web search query.
    pub query: String,
}

/// Searches the web for information using the Brave Search API.
///
/// Enforces the per-query web search budget via [`WebBudgetTracker`].
/// Returns formatted results with titles, URLs, and descriptions.
#[derive(Serialize, Deserialize)]
pub struct SearchWebTool {
    // INVARIANT: Always `Some` after construction via `new()`.
    // `Option` required by rig's `Deserialize` bound on `Tool`.
    #[serde(skip)]
    budget: Option<Arc<WebBudgetTracker>>,
    #[serde(skip)]
    config: Option<BraveConfig>,
    #[serde(skip)]
    http_client: Option<reqwest::Client>,
}

impl SearchWebTool {
    /// Create a new `SearchWebTool` with the given budget tracker, config, and shared HTTP client.
    pub fn new(
        budget: Arc<WebBudgetTracker>,
        config: BraveConfig,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            budget: Some(budget),
            config: Some(config),
            http_client: Some(http_client),
        }
    }
}

impl Tool for SearchWebTool {
    const NAME: &'static str = "search_web";
    type Error = ToolError;
    type Args = SearchWebArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Web-search fallback via Brave Search. Reach for this ONLY when the \
                          internal knowledge base is confirmed empty / insufficient on the topic \
                          after a search_knowledge call, OR when the query is explicitly \
                          time-sensitive (news, current events, live prices, today's weather) \
                          and the corpus is static. Never the first retrieval tool in a turn. \
                          Results are URL + snippet triples; follow up with fetch_url if a \
                          specific result's full body is needed."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let budget = self.budget.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_web: budget tracker not initialized".into())
        })?;

        budget.record_search()?;

        let config = self.config.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_web: brave config not initialized".into())
        })?;

        // Fail closed: without an API key the tool cannot function. Returning
        // Ok("not available") let the agent treat the call as a successful
        // empty search and produce confident answers without web context.
        if config.api_key.is_empty() {
            tracing::error!(query = %args.query, "Brave API key not configured; web search failed");
            return Err(ToolError::WebSearch(
                "web search unavailable: BRAVE_API_KEY not configured. \
                 Do not claim that a web search was performed."
                    .into(),
            ));
        }

        tracing::info!(query = %args.query, "executing Brave web search");

        let url = format!("{}/web/search", config.base_url);

        let client = self.http_client.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_web: http_client not initialized".into())
        })?;

        let response = client
            .get(&url)
            .query(&[("q", &args.query)])
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &config.api_key)
            .send()
            .await
            .map_err(|e| ToolError::WebSearch(format!("Brave API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ToolError::WebSearch(format!(
                "Brave API returned HTTP {status}: {body}"
            )));
        }

        let brave_response: BraveWebResponse = response
            .json()
            .await
            .map_err(|e| ToolError::WebSearch(format!("Brave API response parse error: {e}")))?;

        let results = brave_response
            .web
            .and_then(|w| w.results)
            .unwrap_or_default();

        if results.is_empty() {
            return Ok(format!("No web results found for query: '{}'.", args.query));
        }

        let mut output = format!("Web search results for '{}':\n\n", args.query);

        for (i, result) in results.iter().enumerate().take(10) {
            let title = result.title.as_deref().unwrap_or("(no title)");
            let url = result.url.as_deref().unwrap_or("(no url)");
            let desc = result.description.as_deref().unwrap_or("(no description)");

            output.push_str(&format!(
                "[{}] {}\n    URL: {}\n    {}\n\n",
                i + 1,
                title,
                url,
                desc
            ));
        }

        tracing::info!(
            query = %args.query,
            result_count = results.len(),
            "Brave web search complete"
        );

        Ok(output)
    }
}

// ---- search_news ----

/// Arguments for the `search_news` tool.
#[derive(Debug, Deserialize)]
pub struct SearchNewsArgs {
    /// The news search query.
    pub query: String,
}

/// Searches recent news articles using the Brave News API.
///
/// Enforces the per-query news search budget via [`WebBudgetTracker`].
/// Returns formatted results with titles, URLs, and descriptions.
#[derive(Serialize, Deserialize)]
pub struct SearchNewsTool {
    // INVARIANT: Always `Some` after construction via `new()`.
    // `Option` required by rig's `Deserialize` bound on `Tool`.
    #[serde(skip)]
    budget: Option<Arc<WebBudgetTracker>>,
    #[serde(skip)]
    config: Option<BraveConfig>,
    #[serde(skip)]
    http_client: Option<reqwest::Client>,
}

impl SearchNewsTool {
    /// Create a new `SearchNewsTool` with the given budget tracker, config, and shared HTTP client.
    pub fn new(
        budget: Arc<WebBudgetTracker>,
        config: BraveConfig,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            budget: Some(budget),
            config: Some(config),
            http_client: Some(http_client),
        }
    }
}

impl Tool for SearchNewsTool {
    const NAME: &'static str = "search_news";
    type Error = ToolError;
    type Args = SearchNewsArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search recent news articles. Use when you need the latest news \
                          or current events information."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news search query."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let budget = self.budget.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_news: budget tracker not initialized".into())
        })?;

        budget.record_news()?;

        let config = self.config.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_news: brave config not initialized".into())
        })?;

        // Fail closed: without an API key the tool cannot function (see search_web).
        if config.api_key.is_empty() {
            tracing::error!(query = %args.query, "Brave API key not configured; news search failed");
            return Err(ToolError::WebSearch(
                "news search unavailable: BRAVE_API_KEY not configured. \
                 Do not claim that a news search was performed."
                    .into(),
            ));
        }

        tracing::info!(query = %args.query, "executing Brave news search");

        let url = format!("{}/news/search", config.base_url);

        let client = self.http_client.as_ref().ok_or_else(|| {
            ToolError::McpClient("search_news: http_client not initialized".into())
        })?;

        let response = client
            .get(&url)
            .query(&[("q", &args.query)])
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &config.api_key)
            .send()
            .await
            .map_err(|e| ToolError::WebSearch(format!("Brave News API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ToolError::WebSearch(format!(
                "Brave News API returned HTTP {status}: {body}"
            )));
        }

        let brave_response: BraveNewsResponse = response.json().await.map_err(|e| {
            ToolError::WebSearch(format!("Brave News API response parse error: {e}"))
        })?;

        let results = brave_response.results.unwrap_or_default();

        if results.is_empty() {
            return Ok(format!(
                "No news results found for query: '{}'.",
                args.query
            ));
        }

        let mut output = format!("News results for '{}':\n\n", args.query);

        for (i, result) in results.iter().enumerate().take(10) {
            let title = result.title.as_deref().unwrap_or("(no title)");
            let url = result.url.as_deref().unwrap_or("(no url)");
            let desc = result.description.as_deref().unwrap_or("(no description)");

            output.push_str(&format!(
                "[{}] {}\n    URL: {}\n    {}\n\n",
                i + 1,
                title,
                url,
                desc
            ));
        }

        tracing::info!(
            query = %args.query,
            result_count = results.len(),
            "Brave news search complete"
        );

        Ok(output)
    }
}
