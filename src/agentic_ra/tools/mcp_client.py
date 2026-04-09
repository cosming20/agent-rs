"""MCP client for tool discovery and invocation via the official Python SDK."""


import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client

from agentic_ra.core.errors import MCPClientError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """Structured result from an MCP tool invocation.

    Attributes:
        tool_name: Name of the invoked tool.
        content: Text content returned by the tool.
        is_error: Whether the tool invocation returned an error.
        raw: Raw content blocks from the MCP response.
    """

    tool_name: str
    content: str
    is_error: bool
    raw: tuple[Any, ...] = ()


class MCPToolClient:
    """Async MCP client for discovering and invoking tools on MCP servers.

    Connects to an MCP server via stdio transport, discovers available tools,
    and invokes them with structured arguments.

    .. note::
        Currently spawns a new subprocess per call (list_tools / call_tool).
        For production use, extract the stdio_client + ClientSession lifecycle
        into a shared async context manager so the subprocess is reused across
        calls within one request.

    Args:
        command: Command to launch the MCP server process.
        args: Arguments for the server command.
        env: Optional environment variables for the server process.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

    async def list_tools(self) -> list[types.Tool]:
        """Discover available tools on the MCP server.

        Returns:
            List of tool definitions with names, descriptions, and schemas.
        """
        try:
            async with stdio_client(self._server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return list(result.tools)
        except Exception as e:
            raise MCPClientError(
                "Failed to list tools",
                command=self._server_params.command,
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
        """
        try:
            async with stdio_client(self._server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments or {})

                    # Extract text content from response
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
        except Exception as e:
            raise MCPClientError(
                "Failed to call tool",
                tool_name=name,
                command=self._server_params.command,
                operation="call_tool",
                original_error=str(e),
            ) from e
