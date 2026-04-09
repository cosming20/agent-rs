"""Tool factory — builds per-request agent with scoped tools."""


import logging
from typing import TYPE_CHECKING, Any

from haystack.components.agents import Agent

from agentic_ra.agent.agent import build_agent_with_tools
from agentic_ra.agent.context import QueryContext
from agentic_ra.agent.tools import (
    make_fetch_url_tool,
    make_ingest_tool,
    make_remember_tool,
    make_search_knowledge_tool,
    make_search_news_tool,
    make_search_web_tool,
)
from agentic_ra.tools.budgets import WebToolBudget
from agentic_ra.tools.url_safety import URLValidator

if TYPE_CHECKING:
    from agentic_ra.agent.memory import MemoryStore
    from agentic_ra.core.config import Settings
    from agentic_ra.tools.mcp_http_client import MCPHttpClient

logger = logging.getLogger(__name__)


class ToolFactory:
    """Builds a per-request Haystack Agent with user-scoped tools.

    Responsibilities:
    - Build WebToolBudget from settings
    - Build all tool configs (search_knowledge, search_web, search_news,
      fetch_url, remember, ingest)
    - Call build_agent_with_tools() to create per-request agent
    - Set ctx.agent, ctx.kb_evidence, ctx.web_evidence
    """

    def __init__(
        self,
        base_agent: Agent,
        router: Any,
        ranker: Any,
        cognee_store: Any,
        document_store: Any,
        brave_client: MCPHttpClient | None,
        url_validator: URLValidator | None,
        memory: MemoryStore,
        settings: Settings,
    ) -> None:
        self._base_agent = base_agent
        self._router = router
        self._ranker = ranker
        self._cognee_store = cognee_store
        self._document_store = document_store
        self._brave_client = brave_client
        self._url_validator = url_validator or URLValidator()
        self._memory = memory
        self._settings = settings

    def build(self, ctx: QueryContext) -> None:
        """Build per-request tools and agent, attach to QueryContext."""
        settings = self._settings
        dataset = settings.cognee_dataset

        budget = WebToolBudget(
            max_web_searches=settings.web_budget_max_searches,
            max_news_searches=settings.web_budget_max_news,
            max_fetch_calls=settings.web_budget_max_fetches,
            total_web_timeout_s=settings.web_budget_timeout_s,
            total_fetch_bytes=settings.web_budget_max_bytes,
        )

        tools_config = [
            make_search_knowledge_tool(
                self._router,
                self._ranker,
                cognee_store=self._cognee_store,
                document_store=self._document_store,
                dataset=dataset,
                role=ctx.role,
                evidence_accumulator=ctx.kb_evidence,
                session_id=str(ctx.session_id),
                conversation_context=ctx.conversation_context,
            ),
            make_search_web_tool(
                brave_client=self._brave_client,
                evidence_accumulator=ctx.web_evidence,
                budget=budget,
            ),
            make_search_news_tool(
                brave_client=self._brave_client,
                evidence_accumulator=ctx.web_evidence,
                budget=budget,
            ),
            make_fetch_url_tool(
                url_validator=self._url_validator,
                evidence_accumulator=ctx.web_evidence,
                budget=budget,
                max_length=settings.fetch_max_length,
                max_bytes=settings.fetch_max_bytes,
                timeout=settings.fetch_timeout,
                user_agent=settings.fetch_user_agent,
            ),
            make_remember_tool(self._memory, ctx.user_id),
        ]
        if self._cognee_store is not None:
            tools_config.append(
                make_ingest_tool(self._cognee_store, dataset=dataset)
            )

        ctx.agent = build_agent_with_tools(self._base_agent, tools_config)
