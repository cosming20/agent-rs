//! Agent tool implementations for agent-rs.
//!
//! Each tool implements the [`rig::tool::Tool`] trait and can be
//! registered with a Rig agent for LLM-driven invocation.
//!
//! # Tools
//!
//! | Tool | Module | Description |
//! |------|--------|-------------|
//! | `search_knowledge` | [`knowledge`] | Query the internal knowledge base |
//! | `outline_corpus` | [`outline_corpus`] | Hierarchical TOC over the per-request pinned documents |
//! | `fetch_section_chunks` | [`fetch_section_chunks`] | Exhaustive, non-ranked chunk retrieval per section |
//! | `search_sections` | [`search_sections`] | Batched parallel retrieval over multiple section_ids |
//! | `search_web` | [`web`] | Web search |
//! | `search_news` | [`web`] | News search |
//! | `fetch_url` | [`fetch`] | HTTP GET with SSRF protection |
//! | `remember` | [`remember`] | Store facts in long-term memory |
//!
//! # Budget tracking
//!
//! [`budget::WebBudgetTracker`] enforces per-query limits on web tool
//! usage (searches, news, fetches). Create one tracker per query and
//! share it across the web-facing tools via `Arc`.
//!
//! # MCP client
//!
//! [`mcp_client::McpClient`] is a stub for future integration with
//! external MCP servers via `rust-mcp-sdk`.

pub mod budget;
pub mod fetch;
pub mod fetch_section_chunks;
pub mod knowledge;
pub mod mcp_client;
pub mod outline_corpus;
pub mod remember;
pub mod search_sections;
pub mod web;

pub use budget::WebBudgetTracker;
pub use fetch::FetchUrlTool;
pub use fetch_section_chunks::FetchSectionChunksTool;
pub use knowledge::{SearchKnowledgeArgs, SearchKnowledgeTool, SearchMode};
pub use mcp_client::McpClient;
pub use outline_corpus::OutlineCorpusTool;
pub use remember::RememberTool;
pub use search_sections::SearchSectionsTool;
pub use web::{SearchNewsTool, SearchWebTool};
