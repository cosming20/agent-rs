"""Haystack Agent builder — configures the ReACT tool-calling agent."""

from __future__ import annotations

import logging
from typing import Any

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from haystack.utils import Secret

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert RAG agent. Your job is to answer questions accurately using \
available tools. Follow this approach:

1. ANALYZE the query — determine what information is needed.
2. SEARCH using the most appropriate strategy:
   - Simple factual questions → "semantic" or "keyword"
   - Complex questions needing entity relationships → "graph"
   - Most questions → "hybrid" (default, recommended)
   - Multi-hop reasoning questions → "graph_hybrid"
3. EVALUATE the results — if insufficient, try a different strategy or search_web.
4. ANSWER based on the evidence found.

Rules:
- Always cite your sources by referencing the evidence chunks [1], [2], etc.
- If you cannot find sufficient evidence, say so honestly.
- Use search_knowledge first, search_web only if local knowledge is insufficient.
- For multi-part questions, make multiple search calls with different queries.
- Use remember to save important facts the user asks you to remember.
"""


def build_agent(
    *,
    model: str,
    api_key: str,
    tools_config: list[dict[str, Any]],
    max_steps: int = 10,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Agent:
    """Build a Haystack Agent with the given tools and configuration."""
    haystack_tools: list[Tool] = []

    for tc in tools_config:
        tool = Tool(
            name=tc["name"],
            description=tc["description"],
            function=tc["function"],
            parameters=tc["parameters"],
        )
        haystack_tools.append(tool)

    chat_generator = OpenAIChatGenerator(
        model=model,
        api_key=Secret.from_token(api_key),
        generation_kwargs={"temperature": 0.1},
    )

    agent = Agent(
        chat_generator=chat_generator,
        tools=haystack_tools,
        system_prompt=system_prompt,
        exit_conditions=["text"],
        max_agent_steps=max_steps,
        raise_on_tool_invocation_failure=False,
    )

    logger.info(
        "Haystack Agent built: model=%s, tools=%s, max_steps=%d",
        model,
        [t.name for t in haystack_tools],
        max_steps,
    )

    return agent
