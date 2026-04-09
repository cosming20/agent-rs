"""Query context — shared mutable state flowing through orchestrator pipeline steps."""


import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from agentic_ra.agent.verifier import VerificationResult
from agentic_ra.retrieval.ranker import EvidencePack


@dataclass
class QueryContext:
    """Mutable state accumulating through the orchestrator pipeline.

    Created once per process_query() call. Each pipeline step reads
    what it needs and writes its outputs to this context.
    """

    # Immutable request parameters
    query: str
    session_id: uuid.UUID
    user_id: str
    strategy: str
    limit: int
    role: str
    start_time: float = field(default_factory=time.monotonic)

    # Accumulated by ContextBuilder
    chat_messages: list[ChatMessage] = field(default_factory=list)
    query_embedding: list[float] | None = None
    strategy_reason: str = ""
    failure_warning: str = ""
    conversation_context: str = ""

    # Accumulated by ToolFactory + AgentRunner
    kb_evidence: list[Document] = field(default_factory=list)
    web_evidence: list[Document] = field(default_factory=list)
    agent: Agent | None = None
    agent_result: dict[str, Any] | None = None
    agent_latency_ms: int = 0

    # Accumulated by EvidenceFuser
    answer: str = ""
    evidence: list[Document] = field(default_factory=list)
    fused_pack: EvidencePack | None = None

    # Accumulated by GroundingGate
    verification: VerificationResult | None = None
    retries: int = 0

    @property
    def elapsed_ms(self) -> int:
        """Milliseconds elapsed since query started."""
        return int((time.monotonic() - self.start_time) * 1000)
