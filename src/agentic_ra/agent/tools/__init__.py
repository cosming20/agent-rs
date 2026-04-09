"""High-level tool functions for the Haystack Agent.

Each ``make_*`` factory returns a ``ToolConfig`` dict with ``name``,
``description``, ``function`` (sync callable), and ``parameters``
(JSON Schema).
"""

from agentic_ra.agent.tools._types import ToolConfig
from agentic_ra.agent.tools._utils import sanitize_user_id
from agentic_ra.agent.tools.fetch import make_fetch_url_tool
from agentic_ra.agent.tools.ingest import make_ingest_tool
from agentic_ra.agent.tools.knowledge import make_search_knowledge_tool
from agentic_ra.agent.tools.memory import make_remember_tool
from agentic_ra.agent.tools.web import make_search_news_tool, make_search_web_tool

# Backward compat alias
_sanitize_user_id = sanitize_user_id

__all__ = [
    "ToolConfig",
    "sanitize_user_id",
    "make_fetch_url_tool",
    "make_ingest_tool",
    "make_search_knowledge_tool",
    "make_remember_tool",
    "make_search_news_tool",
    "make_search_web_tool",
]
