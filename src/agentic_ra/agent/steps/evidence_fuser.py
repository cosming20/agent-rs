"""Evidence fuser — merges KB and web evidence via ranked fusion."""


import logging
from typing import TYPE_CHECKING

from agentic_ra.agent.context import QueryContext
from agentic_ra.observability.tracing import get_tracer

if TYPE_CHECKING:
    from agentic_ra.observability.metrics import RAGMetrics
    from agentic_ra.retrieval.ranker import ResultRanker

logger = logging.getLogger(__name__)


class EvidenceFuser:
    """Fuses KB and web evidence into a single ranked evidence pack.

    Responsibilities:
    - Call ranker.fuse_evidence(kb_documents, web_documents, query)
    - Set ctx.evidence, ctx.fused_pack
    - Log evidence counts
    - Record metrics
    """

    def __init__(
        self,
        ranker: ResultRanker,
        metrics: RAGMetrics | None,
    ) -> None:
        self._ranker = ranker
        self._metrics = metrics

    def fuse(self, ctx: QueryContext) -> None:
        """Fuse KB + web evidence and populate ctx.evidence / ctx.fused_pack."""
        tracer = get_tracer()

        with tracer.start_as_current_span("orchestrator.evidence_fusion") as fuse_span:
            fused_pack = self._ranker.fuse_evidence(
                kb_documents=ctx.kb_evidence,
                web_documents=ctx.web_evidence,
                query=ctx.query,
            )
            ctx.evidence = list(fused_pack.documents)
            ctx.fused_pack = fused_pack
            fuse_span.set_attribute("kb_raw", len(ctx.kb_evidence))
            fuse_span.set_attribute("web_raw", len(ctx.web_evidence))
            fuse_span.set_attribute("fused", len(ctx.evidence))
            fuse_span.set_attribute("strategy", ctx.strategy)

        if not ctx.evidence:
            logger.warning(
                "ZERO EVIDENCE after retrieval — "
                "kb_raw=%d, web_raw=%d, fused=%d, strategy=%s, "
                "agent_ok=%s, query=%.100s",
                len(ctx.kb_evidence),
                len(ctx.web_evidence),
                len(fused_pack.documents),
                ctx.strategy,
                ctx.agent_result is not None,
                ctx.query,
            )
        else:
            logger.info(
                "Evidence collected: kb=%d, web=%d, fused=%d, strategy=%s",
                len(ctx.kb_evidence),
                len(ctx.web_evidence),
                len(ctx.evidence),
                ctx.strategy,
            )

        if self._metrics:
            self._metrics.record_retrieval(
                ctx.elapsed_ms, len(ctx.evidence), ctx.strategy
            )
