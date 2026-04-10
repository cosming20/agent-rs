//! MCP client wrapper -- stub for Phase 5b.
//!
//! Will integrate with `rust-mcp-sdk` for tool discovery and invocation
//! of external MCP servers (e.g., Brave Search MCP).

use agent_rs_core::ToolError;

/// MCP client wrapper for connecting to external MCP servers.
///
/// Phase 5 stub: no actual MCP connection. Phase 5b will use
/// `rust-mcp-sdk` for:
///
/// - Server discovery and capability negotiation
/// - Tool listing and schema retrieval
/// - Tool invocation with structured arguments
/// - Transport management (stdio, HTTP/SSE)
pub struct McpClient {
    /// The MCP server URL or transport configuration.
    _server_url: String,
}

impl McpClient {
    /// Create a new MCP client pointing at the given server.
    ///
    /// Does not establish a connection until [`McpClient::connect`] is called.
    pub fn new(server_url: String) -> Self {
        Self {
            _server_url: server_url,
        }
    }

    /// Establish a connection to the MCP server.
    ///
    /// Phase 5 stub: returns an error indicating MCP is not yet available.
    pub async fn connect(&self) -> Result<(), ToolError> {
        Err(ToolError::McpClient(
            "MCP client not yet implemented".into(),
        ))
    }

    /// Invoke a tool on the connected MCP server.
    ///
    /// Phase 5 stub: returns an error indicating MCP is not yet available.
    pub async fn invoke_tool(
        &self,
        _tool_name: &str,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        Err(ToolError::McpClient(
            "MCP client not yet implemented".into(),
        ))
    }
}
