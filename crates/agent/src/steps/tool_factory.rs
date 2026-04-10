//! Tool factory -- builds per-request tool configuration for the Rig agent.
//!
//! In Phase 4, this produces a list of [`ToolSpec`] descriptors.
//! In Phase 5, these specs will be mapped to actual rig-core `Tool`
//! implementations from the `tools` crate.

use serde::{Deserialize, Serialize};

use crate::context::QueryContext;

/// Description of a tool available to the agent for a given request.
///
/// Phase 4: metadata only. Phase 5 maps these to rig-core `Tool` impls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Unique tool name (e.g., "search_knowledge").
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// Whether this tool is enabled for the current request.
    pub enabled: bool,
}

/// Builds per-request tool specifications.
///
/// In Phase 4, this produces the standard set of tool specs.
/// In Phase 5, it will also configure tool-specific state (budgets,
/// evidence accumulators, etc.).
pub struct ToolFactory;

impl ToolFactory {
    /// Create a new ToolFactory.
    pub fn new() -> Self {
        Self
    }

    /// Build tool specifications for the current request.
    ///
    /// Returns the standard set of tools the agent can use.
    /// All tools are enabled by default; Phase 5 will add conditional
    /// enabling based on configuration and available services.
    pub fn build(&self, _ctx: &QueryContext) -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "search_knowledge".into(),
                description: "Search the knowledge base using the specified retrieval strategy."
                    .into(),
                enabled: true,
            },
            ToolSpec {
                name: "search_web".into(),
                description: "Search the web via Brave Search for external information.".into(),
                enabled: true,
            },
            ToolSpec {
                name: "search_news".into(),
                description: "Search recent news via Brave News.".into(),
                enabled: true,
            },
            ToolSpec {
                name: "fetch_url".into(),
                description: "Fetch and extract text content from a URL.".into(),
                enabled: true,
            },
            ToolSpec {
                name: "remember".into(),
                description: "Save important facts to long-term memory.".into(),
                enabled: true,
            },
            ToolSpec {
                name: "ingest".into(),
                description: "Trigger document ingestion into the knowledge base.".into(),
                enabled: true,
            },
        ]
    }
}

impl Default for ToolFactory {
    fn default() -> Self {
        Self::new()
    }
}
