//! OpenTelemetry tracing initialization.
//!
//! Sets up a `tracing-subscriber` registry composed of three layers:
//!
//! 1. [`EnvFilter`] — honours `RUST_LOG`; defaults to `info`.
//! 2. A `fmt` layer — JSON in production, pretty-printed in development.
//! 3. An [`OpenTelemetryLayer`] — bridges every `tracing` span to the OTel
//!    tracer provider, which exports via OTLP/gRPC to the collector.
//!
//! Call [`init_tracing`] once at application startup and hold the returned
//! [`TracingGuard`] until the process is ready to exit. Dropping the guard
//! triggers a graceful shutdown of both the tracer and meter providers,
//! flushing any buffered spans/metrics.

use std::time::Duration;

use agent_rs_core::config::OtelConfig;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{KeyValue, global};
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{MeterProviderBuilder, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{RandomIdGenerator, Sampler, SdkTracerProvider};
use opentelemetry_semantic_conventions::SCHEMA_URL;
use opentelemetry_semantic_conventions::attribute::SERVICE_VERSION;

/// Deployment environment attribute key (from semconv experimental, inlined to avoid feature flag).
const DEPLOYMENT_ENVIRONMENT_NAME: &str = "deployment.environment.name";
use thiserror::Error;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, fmt, util::SubscriberInitExt};

/// Errors that can occur during tracing initialisation.
#[derive(Debug, Error)]
pub enum TracingInitError {
    #[error("failed to build OTLP span exporter: {0}")]
    SpanExporter(String),
    #[error("failed to build OTLP metric exporter: {0}")]
    MetricExporter(String),
    #[error("failed to set global tracing subscriber: {0}")]
    SubscriberInit(String),
}

/// Drop guard that shuts down both the tracer and meter providers on drop,
/// flushing all buffered telemetry before the process exits.
pub struct TracingGuard {
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Err(err) = self.tracer_provider.shutdown() {
            // Use eprintln here because tracing may already be torn down.
            eprintln!("error shutting down tracer provider: {err:?}");
        }
        if let Err(err) = self.meter_provider.shutdown() {
            eprintln!("error shutting down meter provider: {err:?}");
        }
    }
}

/// Build the OTel [`Resource`] for this service instance.
///
/// Includes the service name, crate version, and a development environment
/// tag. The schema URL ties attribute names to the semantic conventions
/// version in use.
fn build_resource(service_name: &str) -> Resource {
    Resource::builder()
        .with_service_name(service_name.to_owned())
        .with_schema_url(
            [
                KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
                KeyValue::new(DEPLOYMENT_ENVIRONMENT_NAME, "production"),
            ],
            SCHEMA_URL,
        )
        .build()
}

/// Construct the [`SdkTracerProvider`] with a batching OTLP/gRPC exporter.
///
/// Batch export is used for production throughput. A parent-based sampler
/// with 100 % trace-id-ratio sampling is configured so that all traces are
/// collected; adjust at the call site or via env vars for sampling.
fn build_tracer_provider(
    endpoint: &str,
    resource: Resource,
) -> Result<SdkTracerProvider, TracingInitError> {
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| TracingInitError::SpanExporter(e.to_string()))?;

    let provider = SdkTracerProvider::builder()
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            1.0,
        ))))
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build();

    Ok(provider)
}

/// Construct the [`SdkMeterProvider`] with a periodic OTLP/gRPC exporter.
///
/// Metrics are exported every 30 seconds by default.
fn build_meter_provider(
    endpoint: &str,
    resource: Resource,
) -> Result<SdkMeterProvider, TracingInitError> {
    let exporter = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| TracingInitError::MetricExporter(e.to_string()))?;

    let reader = PeriodicReader::builder(exporter)
        .with_interval(Duration::from_secs(30))
        .build();

    let provider = MeterProviderBuilder::default()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    Ok(provider)
}

/// Initialise the global `tracing` subscriber with OpenTelemetry integration.
///
/// This function must be called exactly once at application startup, before
/// any `tracing` macros are invoked.
///
/// # Arguments
///
/// * `config` — OTel collector configuration (host + port). The OTLP/gRPC
///   endpoint is derived via [`OtelConfig::endpoint`].
/// * `service_name` — Logical service name embedded in every span and metric.
/// * `json` — When `true`, emit structured JSON logs (production). When
///   `false`, emit human-readable pretty logs (development). Defaults to
///   `false`. Pass `true` in production.
///
/// # Returns
///
/// A [`TracingGuard`] whose [`Drop`] impl flushes and shuts down both
/// providers. Keep it alive for the entire process lifetime.
///
/// # Errors
///
/// Returns [`TracingInitError`] if the OTLP exporters cannot be built or
/// the global subscriber is already set.
pub fn init_tracing(
    config: &OtelConfig,
    service_name: &str,
) -> Result<TracingGuard, TracingInitError> {
    init_tracing_with_options(config, service_name, false)
}

/// Like [`init_tracing`] but lets the caller choose JSON vs pretty logging.
///
/// # Arguments
///
/// * `json` — `true` for JSON log output (production), `false` for pretty
///   (development / local).
pub fn init_tracing_with_options(
    config: &OtelConfig,
    service_name: &str,
    json: bool,
) -> Result<TracingGuard, TracingInitError> {
    let endpoint = config.endpoint();
    let resource = build_resource(service_name);

    // Build tracer + meter providers.
    let tracer_provider = build_tracer_provider(&endpoint, resource.clone())?;
    let meter_provider = build_meter_provider(&endpoint, resource)?;

    // Register globals so other crates can obtain a tracer/meter without
    // holding a reference to the provider directly.
    global::set_tracer_provider(tracer_provider.clone());
    global::set_meter_provider(meter_provider.clone());

    // Obtain a tracer from the provider for the tracing-opentelemetry layer.
    let tracer = tracer_provider.tracer(service_name.to_owned());
    let otel_layer = OpenTelemetryLayer::new(tracer)
        .with_location(true)
        .with_tracked_inactivity(true)
        .with_error_fields_to_exceptions(true)
        .with_error_events_to_exceptions(true)
        .with_error_events_to_status(true);

    // EnvFilter: read RUST_LOG; fall back to "info".
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Compose the subscriber registry.
    //
    // The OTel layer is added directly on top of the registry so that its
    // generic subscriber parameter `S` is bound to `Layered<EnvFilter, Registry>`
    // in both branches — avoiding a type mismatch between the pretty and JSON
    // fmt layers. The fmt layer is added last as it does not affect the OTel
    // span pipeline.
    if json {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .with(fmt::layer().json().with_target(true).with_thread_ids(true))
            .try_init()
            .map_err(|e| TracingInitError::SubscriberInit(e.to_string()))?;
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .with(fmt::layer().pretty().with_target(true))
            .try_init()
            .map_err(|e| TracingInitError::SubscriberInit(e.to_string()))?;
    }

    tracing::info!(
        service = service_name,
        endpoint = %config.endpoint(),
        json,
        "tracing initialised with OpenTelemetry"
    );

    Ok(TracingGuard {
        tracer_provider,
        meter_provider,
    })
}
