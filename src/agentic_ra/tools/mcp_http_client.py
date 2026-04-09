"""MCP client for Streamable HTTP transport with connection pooling.

Creates a Streamable HTTP connection per tool invocation. The MCP SDK manages
underlying HTTP connection reuse internally.
One MCPHttpClient instance per remote MCP server per app instance.

See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http
"""


import logging
from typing import Any

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from agentic_ra.core.errors import MCPClientError
from agentic_ra.tools.mcp_client import ToolResult

logger = logging.getLogger(__name__)


class MCPHttpClient:
    """Async MCP client for Streamable HTTP transport.

    Connects to an MCP server via Streamable HTTP transport. Each tool call
    creates a new MCP session. One MCPHttpClient instance per remote MCP
    server per app instance.

    Args:
        base_url: Full URL to the MCP server endpoint (e.g. ``http://brave-search-mcp:1063``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def list_tools(self) -> list[types.Tool]:
        """Discover available tools on the MCP server.

        Returns:
            List of tool definitions with names, descriptions, and schemas.

        Raises:
            MCPClientError: If connection or listing fails.
        """
        try:
            async with streamablehttp_client(
                f"{self._base_url}/mcp",
                timeout=self._timeout,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return list(result.tools)
        except Exception as e:
            raise MCPClientError(
                "Failed to list tools via HTTP",
                base_url=self._base_url,
                operation="list_tools",
                original_error=str(e),
            ) from e

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Invoke a tool on the MCP server.

        Args:
            name: Tool name to invoke.
            arguments: Tool arguments as a dict.

        Returns:
            Structured result with text content and error status.

        Raises:
            MCPClientError: If connection or invocation fails.
        """
        try:
            async with streamablehttp_client(
                f"{self._base_url}/mcp",
                timeout=self._timeout,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments or {})

                    text_parts: list[str] = []
                    for block in result.content:
                        if isinstance(block, types.TextContent):
                            text_parts.append(block.text)

                    return ToolResult(
                        tool_name=name,
                        content="\n".join(text_parts),
                        is_error=result.isError or False,
                        raw=tuple(result.content),
                    )
        except MCPClientError:
            raise
        except Exception as e:
            raise MCPClientError(
                "Failed to call tool via HTTP",
                tool_name=name,
                base_url=self._base_url,
                operation="call_tool",
                original_error=str(e),
            ) from e
