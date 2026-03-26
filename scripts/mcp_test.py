"""Test MCP client → server round trip."""

import asyncio

from agentic_ra.tools.mcp_client import MCPToolClient


async def main() -> None:
    client = MCPToolClient(
        command="uv",
        args=["run", "python", "-m", "agentic_ra.tools.mcp_servers.search_server"],
    )
    tools = await client.list_tools()
    print(f"Tools: {[t.name for t in tools]}")

    result = await client.call_tool("search_documents", {"query": "test"})
    print(f"Result: {result.content}")
    print("MCP round trip OK")


if __name__ == "__main__":
    asyncio.run(main())
