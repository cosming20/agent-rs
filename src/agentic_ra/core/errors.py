"""Structured error hierarchy for the Agentic RAG system.

All domain exceptions inherit from ``AgenticRAError`` and carry
structured context for observability integration.

Usage::

    raise CogneeError("Graph query failed", dataset="agentic_rag", operation="search")
    # str: "Graph query failed [dataset='agentic_rag', operation='search']"
"""

from __future__ import annotations


class AgenticRAError(Exception):
    """Base exception for the Agentic RAG system.

    Args:
        message: Human-readable error description.
        **context: Structured key-value metadata for logging/tracing.
    """

    def __init__(self, message: str, **context: object) -> None:
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class ConfigError(AgenticRAError):
    """Configuration validation or loading failure."""


class StoreError(AgenticRAError):
    """Base for all data store failures."""


class PostgresError(StoreError):
    """PostgreSQL connection or query failure."""


class RetrievalError(AgenticRAError):
    """Base for all retrieval pipeline failures."""


class EmbeddingError(RetrievalError):
    """Embedding generation failure."""


class RoutingError(RetrievalError):
    """Retrieval routing or dispatch failure."""


class RankingError(RetrievalError):
    """Ranking or evidence assembly failure."""


class AgentError(AgenticRAError):
    """Base for all agent core failures."""


class MemoryStoreError(AgentError):
    """Memory store read/write failure."""


class OrchestratorError(AgentError):
    """Agent orchestration failure."""


class VerificationError(AgentError):
    """Faithfulness/grounding verification failure."""


class SynthesisError(AgentError):
    """Response synthesis failure."""


class ToolError(AgenticRAError):
    """Base for all tool integration failures."""


class MCPClientError(ToolError):
    """MCP client connection or invocation failure."""


class AuthError(AgenticRAError):
    """Base for authentication/authorization failures."""


class OPAError(AuthError):
    """OPA policy evaluation failure."""


class CogneeError(StoreError):
    """Cognee knowledge graph operation failure."""


class GatewayError(AgenticRAError):
    """API gateway failure."""
