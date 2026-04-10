//! `search_web` and `search_news` tools.
//!
//! Both tools check the [`WebBudgetTracker`] before executing and return
//! a budget-exhausted error if the per-query limit has been reached.
//!
//! Phase 5a: stub implementations that return "not available" messages.
//! Phase 5b will integrate Brave Search via the MCP client or direct API.

use std::sync::Arc;

use agent_rs_core::ToolError;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::budget::WebBudgetTracker;

// ---- search_web ----

/// Arguments for the `search_web` tool.
#[derive(Debug, Deserialize)]
pub struct SearchWebArgs {
    /// The web search query.
    pub query: String,
}

/// Searches the web for information using an external search provider.
///
/// Currently a stub -- will integrate with Brave Search in Phase 5b.
/// Enforces the per-query web search budget via [`WebBudgetTracker`].
#[derive(Serialize, Deserialize)]
pub struct SearchWebTool {
    #[serde(skip)]
    budget: Option<Arc<WebBudgetTracker>>,
}

impl SearchWebTool {
    /// Create a new `SearchWebTool` with the given budget tracker.
    pub fn new(budget: Arc<WebBudgetTracker>) -> Self {
        Self {
            budget: Some(budget),
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
            description: "Search the web for current information. Use when the knowledge base \
                          does not contain the answer or when up-to-date information is needed."
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

        tracing::info!(query = %args.query, "web search requested (stub)");

        // Phase 5b: Brave Search integration.
        Ok(format!(
            "Web search is not yet available. Query: '{}'. \
             Please rely on the knowledge base for now.",
            args.query
        ))
    }
}

// ---- search_news ----

/// Arguments for the `search_news` tool.
#[derive(Debug, Deserialize)]
pub struct SearchNewsArgs {
    /// The news search query.
    pub query: String,
}

/// Searches recent news articles using an external search provider.
///
/// Currently a stub -- will integrate with Brave News in Phase 5b.
/// Enforces the per-query news search budget via [`WebBudgetTracker`].
#[derive(Serialize, Deserialize)]
pub struct SearchNewsTool {
    #[serde(skip)]
    budget: Option<Arc<WebBudgetTracker>>,
}

impl SearchNewsTool {
    /// Create a new `SearchNewsTool` with the given budget tracker.
    pub fn new(budget: Arc<WebBudgetTracker>) -> Self {
        Self {
            budget: Some(budget),
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

        tracing::info!(query = %args.query, "news search requested (stub)");

        // Phase 5b: Brave News integration.
        Ok(format!(
            "News search is not yet available. Query: '{}'. \
             Please rely on the knowledge base for now.",
            args.query
        ))
    }
}
