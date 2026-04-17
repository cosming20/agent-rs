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
const ZERO_EVIDENCE_TOTAL: &str = "pipeline.zero_evidence_total";
const WEB_BUDGET_EXHAUSTED_TOTAL: &str = "tools.web_budget_exhausted_total";
const GROUNDING_RETRY_TOTAL: &str = "grounding.retry_total";
const GROUNDING_RETRIES_EXHAUSTED_TOTAL: &str = "grounding.retries_exhausted_total";
const QUERY_DURATION_MS: &str = "pipeline.query_duration_ms";
const VERIFIER_METHOD_TOTAL: &str = "verification.method_total";
const RERANKER_FALLBACK_TOTAL: &str = "retrieval.reranker_fallback_total";
const GRAPH_EXTRACTION_FAILURE_TOTAL: &str = "ingestion.graph_extraction_failure_total";

/// Histogram bucket boundaries for retrieval latency (seconds).
///
/// Chosen to give fine granularity for sub-100 ms fast paths and coarser
/// resolution for slower responses, up to 30 s for timeouts.
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Histogram bucket boundaries for confidence scores (dimensionless, 0–1).
const CONFIDENCE_BUCKETS: &[f64] = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// Histogram bucket boundaries for query duration in milliseconds.
const DURATION_MS_BUCKETS: &[f64] = &[
    10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0,
];

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
    /// Count of pipeline runs that produced zero evidence after retrieval.
    zero_evidence: metrics::Counter<u64>,
    /// Count of web budget exhaustion events by resource kind (search/news/fetch).
    web_budget_exhausted: metrics::Counter<u64>,
    /// Count of grounding-gate retry attempts, labelled by strategy.
    grounding_retry: metrics::Counter<u64>,
    /// Count of queries that exhausted all retries without passing grounding.
    grounding_retries_exhausted: metrics::Counter<u64>,
    /// End-to-end query pipeline duration in milliseconds.
    query_duration_ms: metrics::Histogram<f64>,
    /// Count of verifier invocations by method (llm/heuristic/heuristic_fallback/none).
    verifier_method: metrics::Counter<u64>,
    /// Count of cross-encoder reranker fallbacks, labelled by strategy.
    reranker_fallback: metrics::Counter<u64>,
    /// Count of graph extraction failures (both primary and fallback failed).
    graph_extraction_failure: metrics::Counter<u64>,
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

        let zero_evidence = meter
            .u64_counter(ZERO_EVIDENCE_TOTAL)
            .with_description(
                "Number of pipeline runs that produced zero evidence after retrieval and fusion.",
            )
            .with_unit("queries")
            .build();

        let web_budget_exhausted = meter
            .u64_counter(WEB_BUDGET_EXHAUSTED_TOTAL)
            .with_description(
                "Number of web tool invocations rejected due to budget exhaustion. \
                 Labelled by resource kind: search, news, fetch.",
            )
            .with_unit("events")
            .build();

        let grounding_retry = meter
            .u64_counter(GROUNDING_RETRY_TOTAL)
            .with_description(
                "Number of grounding-gate retry attempts, labelled by retrieval strategy.",
            )
            .with_unit("retries")
            .build();

        let grounding_retries_exhausted = meter
            .u64_counter(GROUNDING_RETRIES_EXHAUSTED_TOTAL)
            .with_description(
                "Number of queries that exhausted all grounding retries without passing \
                 the faithfulness check.",
            )
            .with_unit("queries")
            .build();

        let query_duration_ms = meter
            .f64_histogram(QUERY_DURATION_MS)
            .with_description("End-to-end query pipeline wall-clock duration in milliseconds.")
            .with_unit("ms")
            .with_boundaries(DURATION_MS_BUCKETS.to_vec())
            .build();

        let verifier_method = meter
            .u64_counter(VERIFIER_METHOD_TOTAL)
            .with_description(
                "Number of verifier invocations by method: llm, heuristic, heuristic_fallback, \
                 or none.",
            )
            .with_unit("calls")
            .build();

        let reranker_fallback = meter
            .u64_counter(RERANKER_FALLBACK_TOTAL)
            .with_description(
                "Number of cross-encoder reranker failures that fell back to RRF-ranked results, \
                 labelled by retrieval strategy.",
            )
            .with_unit("events")
            .build();

        let graph_extraction_failure = meter
            .u64_counter(GRAPH_EXTRACTION_FAILURE_TOTAL)
            .with_description(
                "Number of graph extraction failures where both the primary and sanitized \
                 fallback extraction attempts produced an empty graph.",
            )
            .with_unit("chunks")
            .build();

        Self {
            retrieval_latency,
            retrieval_results,
            strategy_usage,
            verification_confidence,
            grounding_pass,
            grounding_fail,
            zero_evidence,
            web_budget_exhausted,
            grounding_retry,
            grounding_retries_exhausted,
            query_duration_ms,
            verifier_method,
            reranker_fallback,
            graph_extraction_failure,
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

    /// Increment the zero-evidence counter, labelled by strategy.
    ///
    /// Call when the evidence fuser produces an empty result set after
    /// retrieval and RRF fusion.
    pub fn record_zero_evidence(&self, strategy: RetrievalStrategy) {
        self.zero_evidence.add(1, &[strategy_attr(strategy)]);
    }

    /// Increment the web budget exhaustion counter, labelled by resource kind.
    ///
    /// `kind` should be one of `"search"`, `"news"`, or `"fetch"`.
    pub fn record_web_budget_exhausted(&self, kind: &str) {
        self.web_budget_exhausted
            .add(1, &[KeyValue::new("resource.kind", kind.to_owned())]);
    }

    /// Increment the grounding retry counter, labelled by strategy.
    ///
    /// Call each time the grounding gate triggers a retry attempt.
    pub fn record_grounding_retry(&self, strategy: RetrievalStrategy) {
        self.grounding_retry.add(1, &[strategy_attr(strategy)]);
    }

    /// Increment the retries-exhausted counter, labelled by strategy.
    ///
    /// Call when the grounding gate loop exits because all retries were
    /// consumed without the answer passing the faithfulness check.
    pub fn record_grounding_retries_exhausted(&self, strategy: RetrievalStrategy) {
        self.grounding_retries_exhausted
            .add(1, &[strategy_attr(strategy)]);
    }

    /// Record the end-to-end query pipeline duration in milliseconds.
    ///
    /// `elapsed_ms` is the wall-clock milliseconds from pipeline start to
    /// the final synthesized response, labelled by strategy.
    pub fn record_query_duration_ms(&self, elapsed_ms: f64, strategy: RetrievalStrategy) {
        self.query_duration_ms
            .record(elapsed_ms, &[strategy_attr(strategy)]);
    }

    /// Increment the verifier method counter.
    ///
    /// `method` matches the `VerificationResult::method` field:
    /// `"llm"`, `"heuristic"`, `"heuristic_fallback"`, or `"none"`.
    pub fn record_verifier_method(&self, method: &str, is_grounded: bool) {
        self.verifier_method.add(
            1,
            &[
                KeyValue::new("verifier.method", method.to_owned()),
                KeyValue::new("verifier.grounded", is_grounded),
            ],
        );
    }

    /// Increment the reranker fallback counter, labelled by strategy.
    ///
    /// Call when the cross-encoder reranker fails and the pipeline falls
    /// back to RRF-ranked results.
    pub fn record_reranker_fallback(&self, strategy: RetrievalStrategy) {
        self.reranker_fallback.add(1, &[strategy_attr(strategy)]);
    }

    /// Increment the graph extraction failure counter.
    ///
    /// Call when both the primary extraction and the sanitized fallback
    /// attempt for a chunk fail, leaving it with an empty knowledge graph.
    pub fn record_graph_extraction_failure(&self) {
        self.graph_extraction_failure.add(1, &[]);
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
