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
const GROUNDING_SUBJECT_ABSENT_TOTAL: &str = "grounding.subject_absent_short_circuit_total";
const GROUNDING_CLAIM_DIRECTED_RETRY_TOTAL: &str = "grounding.claim_directed_retry_total";
const GROUNDING_CLAIM_DIRECTED_NEW_EVIDENCE_COUNT: &str =
    "grounding.claim_directed_new_evidence_count";
const QUERY_DURATION_MS: &str = "pipeline.query_duration_ms";
const VERIFIER_METHOD_TOTAL: &str = "verification.method_total";
const RERANKER_FALLBACK_TOTAL: &str = "retrieval.reranker_fallback_total";
const GRAPH_EXTRACTION_FAILURE_TOTAL: &str = "ingestion.graph_extraction_failure_total";
const COVERAGE_PLANNER_UNMATCHED_TOTAL: &str = "agent.coverage_planner_unmatched_total";
const COVERAGE_VERIFIED_FETCHED_TOTAL: &str = "agent.coverage_verified_fetched_total";
const COVERAGE_VERIFIED_EXPECTED_TOTAL: &str = "agent.coverage_verified_expected_total";

// ---- Tier 1 — cost + per-stage time ----
const LLM_CALLS_TOTAL: &str = "llm.calls_total";
const LLM_TOKENS_PROMPT_TOTAL: &str = "llm.tokens_prompt_total";
const LLM_TOKENS_COMPLETION_TOTAL: &str = "llm.tokens_completion_total";
const LLM_COST_USD: &str = "llm.cost_usd";
const LLM_LATENCY_MS: &str = "llm.latency_ms";
const LLM_FAILURES_TOTAL: &str = "llm.failures_total";
const INGEST_STAGE_MS: &str = "ingestion.stage_ms";

// ---- Tier 2 — chunk quality ----
const CHUNKS_PER_DOC: &str = "ingestion.chunks_per_doc";
const CHUNK_TOKENS: &str = "ingestion.chunk_tokens";
const CHUNK_CHARS: &str = "ingestion.chunk_chars";
const CHUNK_HEADING_DEPTH: &str = "ingestion.chunk_heading_depth";
const CHUNK_FLAG_TOTAL: &str = "ingestion.chunk_flag_total";
const CHUNK_EMPTY_EMBED_TOTAL: &str = "ingestion.chunk_empty_embed_total";

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

/// Histogram buckets for LLM cost in USD (wide range: cheap-call to costly
/// long-context responses).
const COST_USD_BUCKETS: &[f64] = &[
    0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
];

/// Histogram buckets for ingest stage latency in milliseconds (docling
/// parse can be minutes; extend to 10 min).
const INGEST_STAGE_MS_BUCKETS: &[f64] = &[
    50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0, 30_000.0, 60_000.0, 120_000.0,
    300_000.0, 600_000.0,
];

/// Histogram buckets for chunks-per-document count.
const CHUNKS_PER_DOC_BUCKETS: &[f64] = &[
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0,
];

/// Histogram buckets for per-chunk token count. Upper bound matches the
/// HybridChunker's max_tokens=1024 ceiling + a bit of headroom.
const CHUNK_TOKENS_BUCKETS: &[f64] = &[
    8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 1536.0, 2048.0,
];

/// Histogram buckets for per-chunk character count.
const CHUNK_CHARS_BUCKETS: &[f64] = &[
    32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16_384.0,
];

/// Histogram buckets for heading-path depth. Depth 0 = chunk has no
/// surrounding heading context (typically a preamble or page header).
const CHUNK_HEADING_DEPTH_BUCKETS: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];

/// Histogram buckets for the count of new evidence items surfaced by a
/// single claim-directed search. The grounding gate caps per-claim
/// retrieval at the configured `claim_search_top_k` (default 3);
/// buckets up to 16 leave headroom for operators that tune higher.
const CLAIM_DIRECTED_NEW_EVIDENCE_BUCKETS: &[f64] = &[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0];

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
    /// Count of queries where the grounding gate short-circuited further
    /// retries because the query's subject tokens were absent from the
    /// first-pass evidence — strong signal the topic is out-of-corpus.
    grounding_subject_absent: metrics::Counter<u64>,
    /// Count of claim-directed retry iterations performed by the
    /// grounding gate. Each iteration may dispatch one search per
    /// flagged claim before re-verifying.
    grounding_claim_directed_retry: metrics::Counter<u64>,
    /// Distribution of "new evidence items surfaced" per claim-directed
    /// search. A value of `0` means the search produced no novel
    /// evidence and the gate fell back to a full agent rerun.
    grounding_claim_directed_new_evidence: metrics::Histogram<u64>,
    /// End-to-end query pipeline duration in milliseconds.
    query_duration_ms: metrics::Histogram<f64>,
    /// Count of verifier invocations by method (llm/heuristic/heuristic_fallback/none).
    verifier_method: metrics::Counter<u64>,
    /// Count of cross-encoder reranker fallbacks, labelled by strategy.
    reranker_fallback: metrics::Counter<u64>,
    /// Count of graph extraction failures (both primary and fallback failed).
    graph_extraction_failure: metrics::Counter<u64>,

    // ---- Tier 1 — cost + per-stage ingest time ----
    /// Count of LLM invocations per component (labelled).
    llm_calls: metrics::Counter<u64>,
    /// Prompt tokens consumed, labelled by component.
    llm_tokens_prompt: metrics::Counter<u64>,
    /// Completion tokens consumed, labelled by component.
    llm_tokens_completion: metrics::Counter<u64>,
    /// Derived USD cost per LLM call. `0.0` for models absent from the
    /// price table (with a one-shot WARN log the first time it happens).
    llm_cost_usd: metrics::Histogram<f64>,
    /// Per-call LLM latency in ms, labelled by component.
    llm_latency_ms: metrics::Histogram<f64>,
    /// Count of LLM failures, labelled by component + error kind.
    llm_failures: metrics::Counter<u64>,
    /// Per-stage ingestion wall-clock time in ms, labelled by stage name.
    /// Stages: docling_parse, embed, kg_extraction, summarize, total.
    ingest_stage_ms: metrics::Histogram<f64>,

    // ---- Tier 2 — chunk quality ----
    /// Number of chunks produced per document.
    chunks_per_doc: metrics::Histogram<u64>,
    /// Per-chunk token count (cl100k_base).
    chunk_tokens: metrics::Histogram<u64>,
    /// Per-chunk character count.
    chunk_chars: metrics::Histogram<u64>,
    /// Per-chunk heading-path depth.
    chunk_heading_depth: metrics::Histogram<u64>,
    /// Per-chunk flag counters, labelled by flag name (has_table, has_list,
    /// has_picture, has_code, has_formula).
    chunk_flag: metrics::Counter<u64>,
    /// Chunks whose embed_text is empty or whitespace-only — a silent
    /// ingestion regression symptom if the rate climbs above baseline.
    chunk_empty_embed: metrics::Counter<u64>,
    /// Count of CoveragePlanner soft-heuristic firings where the strict
    /// hard-token rules did not promote the query to exhaustive/partial mode.
    /// Labelled by `reason` (heuristic name). Used to tune the rule set.
    coverage_planner_unmatched: metrics::Counter<u64>,
    /// Total section count that was fetched (present in evidence) across all
    /// CoverageVerifier runs. Divide by `coverage_verified_expected` to get
    /// the coverage ratio in a Grafana expression.
    coverage_verified_fetched: metrics::Counter<u64>,
    /// Total section count expected (plan target) across all CoverageVerifier
    /// runs.
    coverage_verified_expected: metrics::Counter<u64>,
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

        let grounding_subject_absent = meter
            .u64_counter(GROUNDING_SUBJECT_ABSENT_TOTAL)
            .with_description(
                "Number of queries where grounding-gate retries were skipped because \
                 the subject of the query was absent from the first-pass evidence.",
            )
            .with_unit("queries")
            .build();

        let grounding_claim_directed_retry = meter
            .u64_counter(GROUNDING_CLAIM_DIRECTED_RETRY_TOTAL)
            .with_description(
                "Number of claim-directed retry iterations performed by the grounding gate. \
                 Each iteration runs one focused retrieval per ungrounded claim before \
                 re-verifying the answer.",
            )
            .with_unit("iterations")
            .build();

        let grounding_claim_directed_new_evidence = meter
            .u64_histogram(GROUNDING_CLAIM_DIRECTED_NEW_EVIDENCE_COUNT)
            .with_description(
                "Number of new evidence items surfaced by a single claim-directed search. \
                 A value of zero means no novel evidence was found and the gate fell back \
                 to a full agent rerun.",
            )
            .with_unit("documents")
            .with_boundaries(CLAIM_DIRECTED_NEW_EVIDENCE_BUCKETS.to_vec())
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

        // ---- Tier 1 instruments ----
        let llm_calls = meter
            .u64_counter(LLM_CALLS_TOTAL)
            .with_description("Number of LLM invocations, labelled by component.")
            .with_unit("calls")
            .build();

        let llm_tokens_prompt = meter
            .u64_counter(LLM_TOKENS_PROMPT_TOTAL)
            .with_description("Prompt tokens consumed, labelled by component + model.")
            .with_unit("tokens")
            .build();

        let llm_tokens_completion = meter
            .u64_counter(LLM_TOKENS_COMPLETION_TOTAL)
            .with_description("Completion tokens consumed, labelled by component + model.")
            .with_unit("tokens")
            .build();

        let llm_cost_usd = meter
            .f64_histogram(LLM_COST_USD)
            .with_description("Derived USD cost per LLM call, labelled by component + model.")
            .with_unit("USD")
            .with_boundaries(COST_USD_BUCKETS.to_vec())
            .build();

        let llm_latency_ms = meter
            .f64_histogram(LLM_LATENCY_MS)
            .with_description("Per-call LLM latency in milliseconds, labelled by component.")
            .with_unit("ms")
            .with_boundaries(DURATION_MS_BUCKETS.to_vec())
            .build();

        let llm_failures = meter
            .u64_counter(LLM_FAILURES_TOTAL)
            .with_description(
                "Number of LLM call failures, labelled by component + error kind \
                 (http_5xx, http_4xx, timeout, rate_limit, malformed_json, schema_violation).",
            )
            .with_unit("events")
            .build();

        let ingest_stage_ms = meter
            .f64_histogram(INGEST_STAGE_MS)
            .with_description(
                "Wall-clock time in ms for each ingestion phase: docling_parse, embed, \
                 kg_extraction, summarize, total.",
            )
            .with_unit("ms")
            .with_boundaries(INGEST_STAGE_MS_BUCKETS.to_vec())
            .build();

        // ---- Tier 2 instruments ----
        let chunks_per_doc = meter
            .u64_histogram(CHUNKS_PER_DOC)
            .with_description("Number of chunks produced per ingested document.")
            .with_unit("chunks")
            .with_boundaries(CHUNKS_PER_DOC_BUCKETS.to_vec())
            .build();

        let chunk_tokens = meter
            .u64_histogram(CHUNK_TOKENS)
            .with_description("Per-chunk token count (cl100k_base).")
            .with_unit("tokens")
            .with_boundaries(CHUNK_TOKENS_BUCKETS.to_vec())
            .build();

        let chunk_chars = meter
            .u64_histogram(CHUNK_CHARS)
            .with_description("Per-chunk character count.")
            .with_unit("chars")
            .with_boundaries(CHUNK_CHARS_BUCKETS.to_vec())
            .build();

        let chunk_heading_depth = meter
            .u64_histogram(CHUNK_HEADING_DEPTH)
            .with_description(
                "Per-chunk heading-path depth. Depth 0 = chunk has no surrounding \
                 heading context.",
            )
            .with_unit("headings")
            .with_boundaries(CHUNK_HEADING_DEPTH_BUCKETS.to_vec())
            .build();

        let chunk_flag = meter
            .u64_counter(CHUNK_FLAG_TOTAL)
            .with_description(
                "Per-chunk structural-flag counts, labelled by flag name \
                 (has_table, has_list, has_picture, has_code, has_formula).",
            )
            .with_unit("chunks")
            .build();

        let chunk_empty_embed = meter
            .u64_counter(CHUNK_EMPTY_EMBED_TOTAL)
            .with_description(
                "Chunks whose embed_text is empty or whitespace-only — signals a \
                 chunker regression if the rate climbs above baseline.",
            )
            .with_unit("chunks")
            .build();

        let coverage_planner_unmatched = meter
            .u64_counter(COVERAGE_PLANNER_UNMATCHED_TOTAL)
            .with_description(
                "CoveragePlanner soft-heuristic fired but strict rules did not match. \
                 Labelled by reason (heuristic name). Use to tune the bilingual token set.",
            )
            .with_unit("events")
            .build();

        let coverage_verified_fetched = meter
            .u64_counter(COVERAGE_VERIFIED_FETCHED_TOTAL)
            .with_description(
                "Total number of sections present in evidence across all CoverageVerifier \
                 runs. Divide by coverage_verified_expected to compute the coverage ratio.",
            )
            .with_unit("sections")
            .build();

        let coverage_verified_expected = meter
            .u64_counter(COVERAGE_VERIFIED_EXPECTED_TOTAL)
            .with_description(
                "Total number of target sections across all CoverageVerifier runs \
                 (plan size summed over all exhaustive queries).",
            )
            .with_unit("sections")
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
            grounding_subject_absent,
            grounding_claim_directed_retry,
            grounding_claim_directed_new_evidence,
            query_duration_ms,
            verifier_method,
            reranker_fallback,
            graph_extraction_failure,
            llm_calls,
            llm_tokens_prompt,
            llm_tokens_completion,
            llm_cost_usd,
            llm_latency_ms,
            llm_failures,
            ingest_stage_ms,
            chunks_per_doc,
            chunk_tokens,
            chunk_chars,
            chunk_heading_depth,
            chunk_flag,
            chunk_empty_embed,
            coverage_planner_unmatched,
            coverage_verified_fetched,
            coverage_verified_expected,
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

    /// Increment the subject-absent short-circuit counter, labelled by strategy.
    ///
    /// Call when the grounding gate skipped retries because the query's
    /// subject tokens were absent from the first-pass evidence — a strong
    /// signal that additional retries will not produce grounded output and
    /// would only waste latency and LLM budget.
    pub fn record_grounding_subject_absent(&self, strategy: RetrievalStrategy) {
        self.grounding_subject_absent
            .add(1, &[strategy_attr(strategy)]);
    }

    /// Increment the claim-directed retry iteration counter, labelled
    /// by strategy.
    ///
    /// Call once per claim-directed retry iteration — typically before
    /// the gate dispatches per-claim searches and re-verifies. Distinct
    /// from [`Self::record_grounding_retry`] (which counts every retry
    /// iteration regardless of mode); claim-directed iterations are a
    /// strict subset.
    pub fn record_grounding_claim_directed_retry(&self, strategy: RetrievalStrategy) {
        self.grounding_claim_directed_retry
            .add(1, &[strategy_attr(strategy)]);
    }

    /// Record the count of new evidence items surfaced by one
    /// claim-directed search batch.
    ///
    /// `count` is the number of documents that were not already present
    /// in the evidence pack. A value of `0` indicates the
    /// claim-directed search produced no novel support and the gate
    /// fell through to a full agent rerun.
    pub fn record_grounding_claim_directed_new_evidence(&self, count: u64) {
        self.grounding_claim_directed_new_evidence
            .record(count, &[]);
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

    // ---- Tier 1 helpers ----

    /// Record one LLM call: increments call counter, token counters,
    /// emits a USD cost derived from the pricing table, and records
    /// latency. Pass-through for every LLM-using component.
    ///
    /// # Arguments
    ///
    /// * `component` — logical name (e.g. `"query_rewriter"`,
    ///   `"graph_extractor"`, `"summarizer"`, `"weighted_keywords"`).
    /// * `model` — the model identifier as sent to OpenAI; used to look
    ///   up per-token pricing and stored as a metric label.
    /// * `prompt_tokens`, `completion_tokens` — from the OpenAI response's
    ///   `usage` field. Pass `0` when unavailable.
    /// * `latency_ms` — wall-clock milliseconds for the LLM round-trip.
    pub fn record_llm_call(
        &self,
        component: &'static str,
        model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        latency_ms: f64,
    ) {
        let attrs_component = &[KeyValue::new("llm.component", component)];
        let attrs_full = &[
            KeyValue::new("llm.component", component),
            KeyValue::new("llm.model", model.to_owned()),
        ];

        self.llm_calls.add(1, attrs_component);
        self.llm_tokens_prompt.add(prompt_tokens, attrs_full);
        self.llm_tokens_completion
            .add(completion_tokens, attrs_full);
        self.llm_latency_ms.record(latency_ms, attrs_component);

        let cost = llm_cost_usd(model, prompt_tokens, completion_tokens);
        self.llm_cost_usd.record(cost, attrs_full);
    }

    /// Increment the LLM failure counter, labelled by component + kind.
    ///
    /// `kind` should be one of: `http_5xx`, `http_4xx`, `timeout`,
    /// `rate_limit`, `malformed_json`, `schema_violation`, `cancelled`,
    /// `other`.
    pub fn record_llm_failure(&self, component: &'static str, kind: &'static str) {
        self.llm_failures.add(
            1,
            &[
                KeyValue::new("llm.component", component),
                KeyValue::new("llm.failure_kind", kind),
            ],
        );
    }

    /// Record an ingest stage's wall-clock duration in milliseconds.
    ///
    /// `stage` is one of `docling_parse`, `embed`, `kg_extraction`,
    /// `summarize`, `total`.
    pub fn record_ingest_stage_ms(&self, stage: &'static str, elapsed_ms: f64) {
        self.ingest_stage_ms
            .record(elapsed_ms, &[KeyValue::new("ingest.stage", stage)]);
    }

    // ---- Tier 2 helpers — chunk quality ----

    /// Record the number of chunks produced by one document ingest.
    pub fn record_chunks_per_doc(&self, count: u64) {
        self.chunks_per_doc.record(count, &[]);
    }

    /// Record a single chunk's token count.
    pub fn record_chunk_tokens(&self, tokens: u64) {
        self.chunk_tokens.record(tokens, &[]);
    }

    /// Record a single chunk's character count.
    pub fn record_chunk_chars(&self, chars: u64) {
        self.chunk_chars.record(chars, &[]);
    }

    /// Record a single chunk's heading-path depth.
    pub fn record_chunk_heading_depth(&self, depth: u64) {
        self.chunk_heading_depth.record(depth, &[]);
    }

    /// Increment one of the structural-flag counters.
    ///
    /// `flag` is one of `has_table`, `has_list`, `has_picture`, `has_code`,
    /// `has_formula`.
    pub fn record_chunk_flag(&self, flag: &'static str) {
        self.chunk_flag.add(1, &[KeyValue::new("chunk.flag", flag)]);
    }

    /// Increment the empty-embed-text chunk counter — signals a silent
    /// chunker regression when the rate climbs above baseline.
    pub fn record_chunk_empty_embed(&self) {
        self.chunk_empty_embed.add(1, &[]);
    }

    /// Increment the coverage-planner-unmatched counter.
    ///
    /// Call when the `CoveragePlanner` soft heuristic fires but the strict
    /// hard-token rules did not promote the query to exhaustive/partial mode.
    ///
    /// `reason` is the stable heuristic name used as the `reason` label —
    /// e.g. `"soft_quantifier_single"`.
    pub fn coverage_planner_unmatched(&self, reason: &'static str) {
        self.coverage_planner_unmatched
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Record the outcome of one `CoverageVerifier` run.
    ///
    /// Increments two cumulative counters so a Grafana expression can derive
    /// the coverage ratio as `fetched_total / expected_total`:
    ///
    /// * `fetched` — number of target sections present in evidence.
    /// * `expected` — number of target sections in the plan.
    pub fn coverage_verified(&self, fetched: usize, expected: usize) {
        self.coverage_verified_fetched.add(fetched as u64, &[]);
        self.coverage_verified_expected.add(expected as u64, &[]);
    }
}

// ---- LLM pricing table (USD per 1M tokens) ----
//
// Rates are approximate and move over time — keep the table small, the
// unknown-model path emits a zero cost with a one-shot WARN so missing
// entries surface without silently corrupting dashboards. Values are a
// close-enough snapshot from the model catalogs as of 2026.

struct ModelPrice {
    prompt_per_1m: f64,
    completion_per_1m: f64,
}

fn model_price(model: &str) -> Option<ModelPrice> {
    // Match by prefix so date-stamped model IDs route to the right base
    // price without a combinatorial explosion of table entries.
    if model.starts_with("gpt-5.4-mini") {
        return Some(ModelPrice {
            prompt_per_1m: 0.25,
            completion_per_1m: 1.25,
        });
    }
    if model.starts_with("gpt-5.4") {
        return Some(ModelPrice {
            prompt_per_1m: 2.50,
            completion_per_1m: 10.00,
        });
    }
    if model.starts_with("gpt-4o-mini") {
        return Some(ModelPrice {
            prompt_per_1m: 0.15,
            completion_per_1m: 0.60,
        });
    }
    if model.starts_with("gpt-4o") {
        return Some(ModelPrice {
            prompt_per_1m: 2.50,
            completion_per_1m: 10.00,
        });
    }
    if model.starts_with("text-embedding-3-large") {
        return Some(ModelPrice {
            prompt_per_1m: 0.13,
            completion_per_1m: 0.0,
        });
    }
    if model.starts_with("text-embedding-3-small") {
        return Some(ModelPrice {
            prompt_per_1m: 0.02,
            completion_per_1m: 0.0,
        });
    }
    None
}

fn llm_cost_usd(model: &str, prompt_tokens: u64, completion_tokens: u64) -> f64 {
    match model_price(model) {
        Some(p) => {
            (prompt_tokens as f64 / 1_000_000.0) * p.prompt_per_1m
                + (completion_tokens as f64 / 1_000_000.0) * p.completion_per_1m
        }
        None => {
            // Unknown model — emit 0 and warn exactly once.
            warn_unknown_model_once(model);
            0.0
        }
    }
}

fn warn_unknown_model_once(model: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = match seen.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.insert(model.to_owned()) {
        tracing::warn!(
            model,
            "llm_cost_usd: model not in pricing table; cost metric will report 0 for this model. \
             Add it to `model_price()` in crates/observability/src/metrics.rs to fix."
        );
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
