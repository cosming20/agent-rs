"""Custom metrics for the Agentic RAG system."""

from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from agentic_ra.core.config import Settings


def setup_metrics(settings: Settings) -> MeterProvider:
    """Initialize OpenTelemetry metrics with OTLP export.

    Args:
        settings: Application settings with OTel endpoint configuration.

    Returns:
        Configured MeterProvider.
    """
    resource = Resource.create(
        {
            "service.name": "agentic-ra",
            "service.version": "0.1.0",
        }
    )

    exporter = OTLPMetricExporter(endpoint=settings.otel_endpoint)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=30000)

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    return provider


class RAGMetrics:
    """Custom metrics for RAG pipeline observability.

    Tracks retrieval latency, hit rates, confidence distributions,
    and other domain-specific metrics.

    Args:
        meter_name: Name for the OpenTelemetry meter.
    """

    def __init__(self, meter_name: str = "agentic_ra") -> None:
        meter = metrics.get_meter(meter_name)

        self.query_count = meter.create_counter(
            name="rag.query.count",
            description="Total number of RAG queries processed",
            unit="1",
        )

        self.retrieval_latency = meter.create_histogram(
            name="rag.retrieval.latency",
            description="Retrieval latency in milliseconds",
            unit="ms",
        )

        self.retrieval_doc_count = meter.create_histogram(
            name="rag.retrieval.document_count",
            description="Number of documents retrieved per query",
            unit="1",
        )

        self.verification_confidence = meter.create_histogram(
            name="rag.verification.confidence",
            description="Grounding verification confidence scores",
            unit="1",
        )

        self.verification_pass_count = meter.create_counter(
            name="rag.verification.pass_count",
            description="Number of queries that passed grounding verification",
            unit="1",
        )

        self.verification_fail_count = meter.create_counter(
            name="rag.verification.fail_count",
            description="Number of queries that failed grounding verification",
            unit="1",
        )

        self.stage2_retry_count = meter.create_counter(
            name="rag.retrieval.stage2_retry_count",
            description="Number of Stage-2 retrieval retries",
            unit="1",
        )

        # --- Feedback (Loop 1) metrics ---
        self.feedback_applied = meter.create_counter(
            name="rag.feedback.applied",
            description="Graph weight updates applied via memify",
            unit="1",
        )

        self.feedback_score = meter.create_histogram(
            name="rag.feedback.score",
            description="Feedback scores sent to Cognee (1-5)",
            unit="1",
        )

        self.confidence_rolling = meter.create_histogram(
            name="rag.confidence.rolling",
            description="Confidence scores for rolling average tracking",
            unit="1",
        )

        # --- Strategy (Loop 2) metrics ---
        self.strategy_prediction = meter.create_counter(
            name="rag.strategy.prediction",
            description="Strategy predictions made by learner",
            unit="1",
        )

        self.strategy_override = meter.create_counter(
            name="rag.strategy.override",
            description="Times learned strategy overrode default",
            unit="1",
        )

        self.failure_evaluation = meter.create_counter(
            name="rag.strategy.failure_evaluation",
            description="PIKE-RAG failure evaluations triggered",
            unit="1",
        )

        # --- Cost metrics ---
        self.query_cost = meter.create_histogram(
            name="rag.cost.per_query",
            description="LLM cost per query in USD",
            unit="USD",
        )

    def record_query(self, strategy: str) -> None:
        """Record a query with its strategy."""
        self.query_count.add(1, {"strategy": strategy})

    def record_retrieval(
        self, latency_ms: float, doc_count: int, strategy: str
    ) -> None:
        """Record retrieval performance metrics."""
        self.retrieval_latency.record(latency_ms, {"strategy": strategy})
        self.retrieval_doc_count.record(doc_count, {"strategy": strategy})

    def record_verification(self, confidence: float, is_grounded: bool) -> None:
        """Record verification result metrics."""
        self.verification_confidence.record(confidence)
        if is_grounded:
            self.verification_pass_count.add(1)
        else:
            self.verification_fail_count.add(1)

    def record_stage2_retry(self) -> None:
        """Record a Stage-2 retrieval retry."""
        self.stage2_retry_count.add(1)

    def record_feedback(self, score: int) -> None:
        """Record a feedback score submission."""
        self.feedback_applied.add(1, {"score": str(score)})
        self.feedback_score.record(score)

    def record_strategy_prediction(
        self, predicted: str, original: str, was_override: bool
    ) -> None:
        """Record a strategy prediction event."""
        self.strategy_prediction.add(1, {"strategy": predicted})
        if was_override:
            self.strategy_override.add(
                1, {"from": original, "to": predicted}
            )

    def record_query_cost(self, cost_usd: float, model: str) -> None:
        """Record per-query LLM cost."""
        self.query_cost.record(cost_usd, {"model": model})
