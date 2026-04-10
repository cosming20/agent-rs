//! Custom metric instruments for the agent-rs retrieval pipeline.
//!
//! All instruments are registered against the global [`opentelemetry::global`]
//! meter provider, so [`crate::tracing::init_tracing`] (which sets that
//! global) must be called before constructing a [`RetrievalMetrics`].
//!
//! # Usage
//!
//! ```no_run
//! use agent_rs_observability::metrics::RetrievalMetrics;
//! use agent_rs_core::types::RetrievalStrategy;
//!
//! let m = RetrievalMetrics::new("agent-rs");
//!
//! m.record_retrieval_latency(RetrievalStrategy::Hybrid, 0.142);
//! m.record_retrieval_results(RetrievalStrategy::Hybrid, 8);
//! m.record_grounding_pass();
//! m.record_verification_confidence(0.87);
//! ```

use agent_rs_core::types::RetrievalStrategy;
use opentelemetry::{InstrumentationScope, KeyValue, global, metrics};

/// Scope name used for all metric instruments in this crate.
const METER_SCOPE: &str = "agent_rs.retrieval";

/// Metric name constants — defined as named constants so that typos are
/// compile errors, not silent mismatches in dashboards.
const RETRIEVAL_LATENCY_SECONDS: &str = "retrieval.latency_seconds";
const RETRIEVAL_RESULTS_TOTAL: &str = "retrieval.results_total";
const RETRIEVAL_STRATEGY_USAGE: &str = "retrieval.strategy_usage";
const VERIFICATION_CONFIDENCE: &str = "verification.confidence";
const GROUNDING_PASS_TOTAL: &str = "grounding.pass_total";
const GROUNDING_FAIL_TOTAL: &str = "grounding.fail_total";

/// Histogram bucket boundaries for retrieval latency (seconds).
///
/// Chosen to give fine granularity for sub-100 ms fast paths and coarser
/// resolution for slower responses, up to 30 s for timeouts.
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Histogram bucket boundaries for confidence scores (dimensionless, 0–1).
const CONFIDENCE_BUCKETS: &[f64] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// All custom metric instruments for the retrieval pipeline.
///
/// Cheaply cloneable — the underlying OTel instruments are `Arc`-backed.
#[derive(Clone)]
pub struct RetrievalMetrics {
    /// End-to-end retrieval latency in seconds, labelled by strategy.
    retrieval_latency: metrics::Histogram<f64>,
    /// Number of results returned per retrieval call, labelled by strategy.
    retrieval_results: metrics::Counter<u64>,
    /// Invocation count per retrieval strategy.
    strategy_usage: metrics::Counter<u64>,
    /// Distribution of verifier confidence scores.
    verification_confidence: metrics::Histogram<f64>,
    /// Count of queries that passed the grounding gate.
    grounding_pass: metrics::Counter<u64>,
    /// Count of queries that failed the grounding gate.
    grounding_fail: metrics::Counter<u64>,
}

impl RetrievalMetrics {
    /// Create a new [`RetrievalMetrics`] instance, registering all instruments
    /// against the global OTel meter provider under `scope`.
    ///
    /// Typically called once at startup with the service name as `scope`.
    pub fn new(scope: &str) -> Self {
        let instrumentation_scope = InstrumentationScope::builder(METER_SCOPE)
            .with_version(env!("CARGO_PKG_VERSION"))
            .with_schema_url(SCHEMA_URL_STR)
            .with_attributes([KeyValue::new("service", scope.to_owned())])
            .build();
        let meter = global::meter_with_scope(instrumentation_scope);

        let retrieval_latency = meter
            .f64_histogram(RETRIEVAL_LATENCY_SECONDS)
            .with_description(
                "End-to-end retrieval latency in seconds, from query receipt to ranked results.",
            )
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
            .build();

        let retrieval_results = meter
            .u64_counter(RETRIEVAL_RESULTS_TOTAL)
            .with_description("Total number of search results returned across all retrieval calls.")
            .with_unit("results")
            .build();

        let strategy_usage = meter
            .u64_counter(RETRIEVAL_STRATEGY_USAGE)
            .with_description("Invocation count per retrieval strategy.")
            .with_unit("calls")
            .build();

        let verification_confidence = meter
            .f64_histogram(VERIFICATION_CONFIDENCE)
            .with_description("Distribution of verifier confidence scores for grounded responses.")
            .with_boundaries(CONFIDENCE_BUCKETS.to_vec())
            .build();

        let grounding_pass = meter
            .u64_counter(GROUNDING_PASS_TOTAL)
            .with_description("Number of queries that passed the faithfulness grounding check.")
            .with_unit("queries")
            .build();

        let grounding_fail = meter
            .u64_counter(GROUNDING_FAIL_TOTAL)
            .with_description("Number of queries that failed the faithfulness grounding check.")
            .with_unit("queries")
            .build();

        Self {
            retrieval_latency,
            retrieval_results,
            strategy_usage,
            verification_confidence,
            grounding_pass,
            grounding_fail,
        }
    }

    /// Record the end-to-end retrieval latency for one query.
    ///
    /// # Arguments
    ///
    /// * `strategy` — The [`RetrievalStrategy`] that was dispatched.
    /// * `latency_secs` — Wall-clock seconds elapsed.
    pub fn record_retrieval_latency(&self, strategy: RetrievalStrategy, latency_secs: f64) {
        self.retrieval_latency
            .record(latency_secs, &[strategy_attr(strategy)]);
    }

    /// Record the number of results returned by one retrieval call.
    ///
    /// # Arguments
    ///
    /// * `strategy` — The [`RetrievalStrategy`] that produced the results.
    /// * `count` — Number of results in the ranked result set.
    pub fn record_retrieval_results(&self, strategy: RetrievalStrategy, count: u64) {
        self.retrieval_results
            .add(count, &[strategy_attr(strategy)]);
        self.strategy_usage.add(1, &[strategy_attr(strategy)]);
    }

    /// Record a verifier confidence score for a completed response.
    ///
    /// `confidence` must be in `[0.0, 1.0]`. Values outside this range are
    /// clamped silently to avoid corrupting histogram state.
    pub fn record_verification_confidence(&self, confidence: f64) {
        let clamped = confidence.clamp(0.0, 1.0);
        self.verification_confidence.record(clamped, &[]);
    }

    /// Increment the counter for queries that passed the grounding gate.
    pub fn record_grounding_pass(&self) {
        self.grounding_pass.add(1, &[]);
    }

    /// Increment the counter for queries that failed the grounding gate.
    pub fn record_grounding_fail(&self) {
        self.grounding_fail.add(1, &[]);
    }
}

// ---- helpers ----

/// Schema URL constant (reuse the same version as the tracing setup).
const SCHEMA_URL_STR: &str = opentelemetry_semantic_conventions::SCHEMA_URL;

/// Convert a [`RetrievalStrategy`] into a stable OTel [`KeyValue`] attribute.
///
/// Using `as_db_str` keeps the attribute value consistent with Postgres
/// storage, dashboards, and log fields throughout the system.
fn strategy_attr(strategy: RetrievalStrategy) -> KeyValue {
    KeyValue::new("retrieval.strategy", strategy.as_db_str())
}
