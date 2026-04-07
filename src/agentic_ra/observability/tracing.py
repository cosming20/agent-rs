"""OpenTelemetry tracing setup and instrumentation.

Configures OTel TracerProvider, enables Haystack's built-in OTel tracing,
and provides a module-level tracer for custom spans in the pipeline.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentic_ra.core.config import Settings

logger = logging.getLogger(__name__)

# Module-level tracer — available after setup_tracing() is called.
_tracer: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    """Get the application tracer. Returns a no-op tracer if not yet initialized."""
    if _tracer is not None:
        return _tracer
    return trace.get_tracer("agentic-ra")


def setup_tracing(settings: Settings) -> TracerProvider:
    """Initialize OpenTelemetry tracing with OTLP export to the collector.

    Also enables Haystack's built-in OTel integration so that Agent.run(),
    tool calls, and all pipeline components emit spans automatically.

    Args:
        settings: Application settings with OTel endpoint configuration.

    Returns:
        Configured TracerProvider.
    """
    global _tracer  # noqa: PLW0603

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

    # Create the application tracer
    _tracer = provider.get_tracer("agentic-ra", "0.1.0")

    # Enable Haystack's built-in OTel tracing — this makes Agent.run(),
    # tool invocations, and all Haystack components emit spans automatically.
    try:
        import haystack.tracing as ht
        from haystack.tracing import OpenTelemetryTracer

        haystack_tracer = OpenTelemetryTracer(_tracer)
        ht.enable_tracing(haystack_tracer)
        logger.info("Haystack OTel tracing enabled")
    except ImportError:
        logger.info("Haystack tracing not available — skipping auto-instrumentation")
    except Exception:
        logger.warning("Failed to enable Haystack OTel tracing", exc_info=True)

    logger.info("OTel tracing initialized — endpoint=%s", settings.otel_endpoint)

    return provider


def instrument_fastapi(app: object) -> None:
    """Instrument a FastAPI application with OpenTelemetry tracing.

    Args:
        app: FastAPI application instance.
    """
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
