"""OpenTelemetry tracing setup and instrumentation."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentic_ra.core.config import Settings


def setup_tracing(settings: Settings) -> TracerProvider:
    """Initialize OpenTelemetry tracing with OTLP export to the collector.

    Configures a TracerProvider with a BatchSpanProcessor that exports
    spans to the OTel Collector via gRPC.

    Args:
        settings: Application settings with OTel endpoint configuration.

    Returns:
        Configured TracerProvider.
    """
    resource = Resource.create(
        {
            "service.name": "agentic-ra",
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    return provider


def instrument_fastapi(app: object) -> None:
    """Instrument a FastAPI application with OpenTelemetry tracing.

    Args:
        app: FastAPI application instance.
    """
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
