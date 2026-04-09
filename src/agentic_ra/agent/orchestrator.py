"""Agent orchestrator — slim coordinator over pipeline step classes."""


import logging
import uuid

from agentic_ra.agent.context import QueryContext
from agentic_ra.agent.steps.agent_runner import AgentRunner
from agentic_ra.agent.steps.context_builder import ContextBuilder
from agentic_ra.agent.steps.evidence_fuser import EvidenceFuser
from agentic_ra.agent.steps.grounding_gate import GroundingGate
from agentic_ra.agent.steps.result_persister import ResultPersister
from agentic_ra.agent.steps.tool_factory import ToolFactory
from agentic_ra.agent.synthesis import SynthesizedResponse
from agentic_ra.core.errors import OrchestratorError
from agentic_ra.observability.tracing import get_tracer
from agentic_ra.retrieval.router import RetrievalStrategy

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Runs the Haystack Agent and applies a hardcoded quality gate.

    Flow:
    1. Load conversation history -> build ChatMessage list
    2. Run Haystack Agent (ReACT tool-calling loop)
    3. Extract answer and evidence from agent result
    4. Verify grounding (hardcoded, agent cannot skip)
    5. If not grounded, re-run agent with feedback (max retries)
    6. Synthesize citations
    7. Persist run + store messages in memory
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        tool_factory: ToolFactory,
        agent_runner: AgentRunner,
        evidence_fuser: EvidenceFuser,
        grounding_gate: GroundingGate,
        result_persister: ResultPersister,
    ) -> None:
        self._context_builder = context_builder
        self._tool_factory = tool_factory
        self._agent_runner = agent_runner
        self._evidence_fuser = evidence_fuser
        self._grounding_gate = grounding_gate
        self._result_persister = result_persister

    async def process_query(
        self,
        query: str,
        session_id: uuid.UUID,
        user_id: str,
        strategy: str = "hybrid",
        limit: int = 10,
        role: str = "user",
    ) -> SynthesizedResponse:
        """Process a user query through the agent + quality gate pipeline."""
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "orchestrator.process_query",
            attributes={
                "query": query[:200],
                "session_id": str(session_id),
                "user_id": user_id,
                "strategy": strategy,
            },
        ) as root_span:
            try:
                # Validate strategy against enum to prevent prompt injection
                try:
                    RetrievalStrategy(strategy)
                except ValueError:
                    logger.warning(
                        "Invalid strategy %r, falling back to hybrid", strategy
                    )
                    strategy = "hybrid"

                ctx = QueryContext(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    strategy=strategy,
                    limit=limit,
                    role=role,
                )

                await self._context_builder.build(ctx)
                self._tool_factory.build(ctx)
                await self._agent_runner.run(ctx)

                if ctx.agent_result is None:
                    root_span.set_attribute("agent_crashed", True)

                self._evidence_fuser.fuse(ctx)
                await self._grounding_gate.verify_and_retry(ctx)
                response = await self._result_persister.persist(ctx)

                # Record final outcome on root span
                root_span.set_attribute("result.confidence", response.confidence)
                root_span.set_attribute("result.is_grounded", response.is_grounded)
                root_span.set_attribute("result.citations", len(response.citations))
                root_span.set_attribute("result.evidence", len(ctx.evidence))
                root_span.set_attribute("result.retries", ctx.retries)
                root_span.set_attribute("result.strategy", ctx.strategy)
                root_span.set_attribute("result.elapsed_ms", ctx.elapsed_ms)

                return response

            except OrchestratorError:
                raise
            except Exception as e:
                raise OrchestratorError(
                    "Query processing failed",
                    query=query,
                    session_id=str(session_id),
                    operation="process_query",
                    original_error=str(e),
                ) from e
