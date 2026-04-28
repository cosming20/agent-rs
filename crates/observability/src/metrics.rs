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
//! m.record_subject_check_short_circuit(RetrievalStrategy::Hybrid);
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
const ZERO_EVIDENCE_TOTAL: &str = "pipeline.zero_evidence_total";
const WEB_BUDGET_EXHAUSTED_TOTAL: &str = "tools.web_budget_exhausted_total";
/// Pre-synth subject-absent short-circuit counter.
///
/// Renamed from `grounding.subject_absent_short_circuit_total` when
/// the LLM-based grounding gate was removed: the underlying check is
/// the same deterministic substring-matching heuristic, but it now
/// runs once before synthesis (rather than as a retry-loop pre-filter)
/// and lives under [`crate::metrics::RetrievalMetrics::record_subject_check_short_circuit`].
const SUBJECT_CHECK_SHORT_CIRCUIT_TOTAL: &str = "subject_check.short_circuit_total";
const QUERY_DURATION_MS: &str = "pipeline.query_duration_ms";
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

// ---- Per-call + per-request budget histograms ----
//
// `LLM_CALL_TOKENS` and `LLM_CALL_COST_USD` complement the existing
// `LLM_TOKENS_*` counters with a histogram view of the per-call
// distribution — useful for spotting fat-tail prompts that drive the
// cost curve. `LLM_REQUEST_TOTAL_*` are emitted once per request from
// the orchestrator, summarising the aggregated [`RequestBudget`].
const LLM_CALL_TOKENS: &str = "llm.call_tokens";
const LLM_CALL_COST_USD: &str = "llm.call_cost_usd";
const LLM_REQUEST_TOTAL_TOKENS: &str = "llm.request_total_tokens";
const LLM_REQUEST_TOTAL_COST_USD: &str = "llm.request_total_cost_usd";

// ---- Tier 2 — chunk quality ----
const CHUNKS_PER_DOC: &str = "ingestion.chunks_per_doc";
const CHUNK_TOKENS: &str = "ingestion.chunk_tokens";
const CHUNK_CHARS: &str = "ingestion.chunk_chars";
const CHUNK_HEADING_DEPTH: &str = "ingestion.chunk_heading_depth";
const CHUNK_FLAG_TOTAL: &str = "ingestion.chunk_flag_total";
const CHUNK_EMPTY_EMBED_TOTAL: &str = "ingestion.chunk_empty_embed_total";

// ---- Phase 4 — per-backend latency histograms ----
const QDRANT_SEARCH_DURATION_MS: &str = "qdrant.search_duration_ms";
const NEO4J_TRAVERSAL_DURATION_MS: &str = "neo4j.traversal_duration_ms";
const POSTGRES_FTS_DURATION_MS: &str = "postgres.fts_duration_ms";
const EMBEDDER_DURATION_MS: &str = "embedder.duration_ms";
const RRF_FUSION_DURATION_MS: &str = "rrf.fusion_duration_ms";
const RERANKER_DURATION_MS: &str = "reranker.duration_ms";
const QDRANT_CIRCUIT_BREAKER_FALLBACK_TOTAL: &str = "qdrant.circuit_breaker_fallback_total";

/// Histogram bucket boundaries for retrieval latency (seconds).
///
/// Chosen to give fine granularity for sub-100 ms fast paths and coarser
/// resolution for slower responses, up to 30 s for timeouts.
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

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

/// Per-backend latency buckets in milliseconds.
///
/// Shared across Qdrant search, Neo4j traversal, Postgres FTS, embedder,
/// reranker. Tuned for backends whose fast-path is single-digit ms and
/// whose slow-path can stretch to ~10 s (Postgres GIN scan, embedder
/// cold start, reranker rate-limit retry).
const BACKEND_LATENCY_MS_BUCKETS: &[f64] = &[
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];

/// Latency buckets for the RRF fusion stage.
///
/// RRF is in-process CPU work and dominated by hashmap merges; fast path
/// is sub-millisecond, slow path (very large pools or contended rayon
/// thread-pool) tops out around 250 ms.
const RRF_FUSION_MS_BUCKETS: &[f64] = &[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0];

/// Histogram buckets for the per-call total-token count. Tuned to
/// span small router-internal calls (a few hundred tokens) up through
/// long-context summarisation (tens of thousands).
const LLM_CALL_TOKENS_BUCKETS: &[f64] = &[
    100.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0, 25_000.0, 50_000.0, 100_000.0,
];

/// Histogram buckets for per-call USD cost. Coarser than
/// [`COST_USD_BUCKETS`] — these are aimed at request-size triage,
/// not fine-grained pricing analysis.
const LLM_CALL_COST_USD_BUCKETS: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];

/// Histogram buckets for the per-request aggregated total tokens.
/// Wider than the per-call bucket set — a single request may chain
/// dozens of LLM calls and accumulate high six-figure totals on
/// exhaustive coverage flows.
const LLM_REQUEST_TOTAL_TOKENS_BUCKETS: &[f64] = &[
    1_000.0,
    5_000.0,
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
    500_000.0,
    1_000_000.0,
];

/// Histogram buckets for the per-request aggregated USD cost.
const LLM_REQUEST_TOTAL_COST_USD_BUCKETS: &[f64] =
    &[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

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
    /// Count of pipeline runs that produced zero evidence after retrieval.
    zero_evidence: metrics::Counter<u64>,
    /// Count of web budget exhaustion events by resource kind (search/news/fetch).
    web_budget_exhausted: metrics::Counter<u64>,
    /// Count of pre-synth subject-absent short-circuit events,
    /// labelled by retrieval strategy. Increments when none of the
    /// agent's rewritten queries share any content tokens with the
    /// fused evidence — strong signal the topic is out-of-corpus and
    /// synthesis is being skipped to save the LLM round-trip.
    subject_check_short_circuit: metrics::Counter<u64>,
    /// End-to-end query pipeline duration in milliseconds.
    query_duration_ms: metrics::Histogram<f64>,
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
    /// Per-call total-token distribution, labelled by component +
    /// model. Complements [`Self::llm_tokens_prompt`] /
    /// [`Self::llm_tokens_completion`] (which are running totals) with
    /// a distribution view useful for spotting fat-tail outliers.
    llm_call_tokens: metrics::Histogram<u64>,
    /// Per-call USD cost distribution, labelled by component + model.
    /// Coarser buckets than [`Self::llm_cost_usd`]; aimed at
    /// request-size triage rather than fine-grained pricing analysis.
    llm_call_cost_usd: metrics::Histogram<f64>,
    /// Aggregated per-request total tokens. Emitted once at end of
    /// `process_query` from the [`RequestBudget`] snapshot.
    llm_request_total_tokens: metrics::Histogram<u64>,
    /// Aggregated per-request USD cost. Emitted once at end of
    /// `process_query` from the [`RequestBudget`] snapshot.
    llm_request_total_cost_usd: metrics::Histogram<f64>,

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

    // ---- Phase 4 — per-backend latency histograms ----
    /// Latency of a single Qdrant vector search call, in milliseconds.
    /// Wired in [`crate::metrics::RetrievalMetrics::record_qdrant_search_duration`]
    /// callers — the router wraps `VectorStore::search_for_user` with
    /// an `Instant::now()` guard so this measures gRPC round-trip + the
    /// store's local marshalling.
    qdrant_search_duration_ms: metrics::Histogram<f64>,
    /// Latency of a Neo4j graph traversal, in milliseconds. Measures
    /// the wall-clock duration of `graph_search` from the router's
    /// dispatch-time perspective (Cypher query + post-fetch payload
    /// rehydration).
    neo4j_traversal_duration_ms: metrics::Histogram<f64>,
    /// Latency of a Postgres BM25 / FTS query, in milliseconds. Wired
    /// at the keyword fallback path — when the Qdrant breaker is open
    /// and the router degrades to BM25-only retrieval, this measures
    /// the latency of the lexical search that filled in.
    postgres_fts_duration_ms: metrics::Histogram<f64>,
    /// Latency of dense embedding generation, in milliseconds.
    /// Recorded around `Embedder::embed_text` calls; covers OpenAI HTTP
    /// round-trip or local fastembed ONNX inference, depending on the
    /// configured backend.
    embedder_duration_ms: metrics::Histogram<f64>,
    /// Latency of a single Reciprocal Rank Fusion merge, in
    /// milliseconds. Recorded inside `ResultRanker::rank` so the
    /// measurement bounds the in-process CPU work, excluding I/O.
    rrf_fusion_duration_ms: metrics::Histogram<f64>,
    /// Latency of a single reranker call, in milliseconds. Recorded at
    /// the router-level wrap of `CrossEncoderReranker::rerank` so the
    /// measurement covers HTTP round-trip and back-off-driven retries.
    reranker_duration_ms: metrics::Histogram<f64>,

    /// Count of Qdrant→Postgres-FTS fallback events triggered by the
    /// circuit breaker, labelled by retrieval strategy. Increments when
    /// the breaker is Open and the router degrades the request to a
    /// BM25-only path. Used to detect prolonged Qdrant outages without
    /// scraping logs.
    qdrant_circuit_breaker_fallback: metrics::Counter<u64>,
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

        let subject_check_short_circuit = meter
            .u64_counter(SUBJECT_CHECK_SHORT_CIRCUIT_TOTAL)
            .with_description(
                "Number of queries where the pre-synth subject-presence gate fired and \
                 synthesis was short-circuited because no rewritten retrieval query shared \
                 any content tokens with the fused evidence.",
            )
            .with_unit("queries")
            .build();

        let query_duration_ms = meter
            .f64_histogram(QUERY_DURATION_MS)
            .with_description("End-to-end query pipeline wall-clock duration in milliseconds.")
            .with_unit("ms")
            .with_boundaries(DURATION_MS_BUCKETS.to_vec())
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

        let llm_call_tokens = meter
            .u64_histogram(LLM_CALL_TOKENS)
            .with_description(
                "Per-call total-token distribution (prompt + completion), labelled by \
                 component + model.",
            )
            .with_unit("tokens")
            .with_boundaries(LLM_CALL_TOKENS_BUCKETS.to_vec())
            .build();

        let llm_call_cost_usd = meter
            .f64_histogram(LLM_CALL_COST_USD)
            .with_description(
                "Per-call USD cost distribution, labelled by component + model. \
                 Coarser buckets than llm.cost_usd; aimed at request-size triage.",
            )
            .with_unit("USD")
            .with_boundaries(LLM_CALL_COST_USD_BUCKETS.to_vec())
            .build();

        let llm_request_total_tokens = meter
            .u64_histogram(LLM_REQUEST_TOTAL_TOKENS)
            .with_description(
                "Aggregated per-request total tokens across every LLM call in a single \
                 process_query, emitted from the RequestBudget snapshot.",
            )
            .with_unit("tokens")
            .with_boundaries(LLM_REQUEST_TOTAL_TOKENS_BUCKETS.to_vec())
            .build();

        let llm_request_total_cost_usd = meter
            .f64_histogram(LLM_REQUEST_TOTAL_COST_USD)
            .with_description(
                "Aggregated per-request USD cost across every LLM call in a single \
                 process_query, emitted from the RequestBudget snapshot.",
            )
            .with_unit("USD")
            .with_boundaries(LLM_REQUEST_TOTAL_COST_USD_BUCKETS.to_vec())
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

        // ---- Phase 4 — per-backend latency histograms ----
        let qdrant_search_duration_ms = meter
            .f64_histogram(QDRANT_SEARCH_DURATION_MS)
            .with_description(
                "Latency of a single Qdrant vector search call (gRPC round-trip plus \
                 local marshalling).",
            )
            .with_unit("ms")
            .with_boundaries(BACKEND_LATENCY_MS_BUCKETS.to_vec())
            .build();

        let neo4j_traversal_duration_ms = meter
            .f64_histogram(NEO4J_TRAVERSAL_DURATION_MS)
            .with_description(
                "Latency of a Neo4j graph traversal — Cypher query plus payload \
                 rehydration from the vector store.",
            )
            .with_unit("ms")
            .with_boundaries(BACKEND_LATENCY_MS_BUCKETS.to_vec())
            .build();

        let postgres_fts_duration_ms = meter
            .f64_histogram(POSTGRES_FTS_DURATION_MS)
            .with_description(
                "Latency of a Postgres BM25 / FTS query, recorded at the keyword \
                 fallback path triggered by the Qdrant circuit breaker.",
            )
            .with_unit("ms")
            .with_boundaries(BACKEND_LATENCY_MS_BUCKETS.to_vec())
            .build();

        let embedder_duration_ms = meter
            .f64_histogram(EMBEDDER_DURATION_MS)
            .with_description(
                "Latency of dense embedding generation — OpenAI HTTP round-trip or \
                 local fastembed ONNX inference, per call.",
            )
            .with_unit("ms")
            .with_boundaries(BACKEND_LATENCY_MS_BUCKETS.to_vec())
            .build();

        let rrf_fusion_duration_ms = meter
            .f64_histogram(RRF_FUSION_DURATION_MS)
            .with_description(
                "Latency of a single Reciprocal Rank Fusion merge, in milliseconds. \
                 Bounds the in-process CPU work and excludes any back-end I/O.",
            )
            .with_unit("ms")
            .with_boundaries(RRF_FUSION_MS_BUCKETS.to_vec())
            .build();

        let reranker_duration_ms = meter
            .f64_histogram(RERANKER_DURATION_MS)
            .with_description(
                "Latency of a single reranker call (HTTP round-trip plus any \
                 back-off-driven retries) in milliseconds.",
            )
            .with_unit("ms")
            .with_boundaries(BACKEND_LATENCY_MS_BUCKETS.to_vec())
            .build();

        let qdrant_circuit_breaker_fallback = meter
            .u64_counter(QDRANT_CIRCUIT_BREAKER_FALLBACK_TOTAL)
            .with_description(
                "Number of Qdrant→Postgres-FTS fallback events triggered by the \
                 circuit breaker, labelled by retrieval strategy.",
            )
            .with_unit("events")
            .build();

        Self {
            retrieval_latency,
            retrieval_results,
            strategy_usage,
            zero_evidence,
            web_budget_exhausted,
            subject_check_short_circuit,
            query_duration_ms,
            reranker_fallback,
            graph_extraction_failure,
            llm_calls,
            llm_tokens_prompt,
            llm_tokens_completion,
            llm_cost_usd,
            llm_latency_ms,
            llm_failures,
            ingest_stage_ms,
            llm_call_tokens,
            llm_call_cost_usd,
            llm_request_total_tokens,
            llm_request_total_cost_usd,
            chunks_per_doc,
            chunk_tokens,
            chunk_chars,
            chunk_heading_depth,
            chunk_flag,
            chunk_empty_embed,
            coverage_planner_unmatched,
            coverage_verified_fetched,
            coverage_verified_expected,
            qdrant_search_duration_ms,
            neo4j_traversal_duration_ms,
            postgres_fts_duration_ms,
            embedder_duration_ms,
            rrf_fusion_duration_ms,
            reranker_duration_ms,
            qdrant_circuit_breaker_fallback,
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

    /// Increment the pre-synth subject-absent short-circuit counter,
    /// labelled by retrieval strategy.
    ///
    /// Call when none of the agent's rewritten retrieval queries share
    /// any content tokens with the fused evidence pack — strong
    /// signal the topic is out-of-corpus, and synthesis is being
    /// skipped to save the wasted LLM round-trip.
    pub fn record_subject_check_short_circuit(&self, strategy: RetrievalStrategy) {
        self.subject_check_short_circuit
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

        // Tier-1.5 distribution metrics — see LLM_CALL_TOKENS /
        // LLM_CALL_COST_USD docs for rationale.
        self.llm_call_tokens
            .record(prompt_tokens.saturating_add(completion_tokens), attrs_full);
        self.llm_call_cost_usd.record(cost, attrs_full);
    }

    /// Record the per-request total token count, emitted once at the
    /// end of `process_query` from the [`crate::budget::RequestBudget`]
    /// snapshot.
    pub fn record_request_total_tokens(&self, total: u64) {
        self.llm_request_total_tokens.record(total, &[]);
    }

    /// Record the per-request total USD cost, emitted once at the end
    /// of `process_query` from the [`crate::budget::RequestBudget`]
    /// snapshot.
    pub fn record_request_total_cost_usd(&self, cost_usd: f64) {
        // Defensive clamp: budget never produces negatives, but a NaN
        // would corrupt the histogram so coerce to zero.
        let v = if cost_usd.is_finite() && cost_usd >= 0.0 {
            cost_usd
        } else {
            0.0
        };
        self.llm_request_total_cost_usd.record(v, &[]);
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

    // ---- Phase 4 helpers — per-backend latency ----

    /// Record the latency of a single Qdrant vector search call.
    ///
    /// `elapsed_ms` is the wall-clock milliseconds elapsed around the
    /// `VectorStore::search_for_user` invocation, including the gRPC
    /// round-trip and the store's local result marshalling.
    pub fn record_qdrant_search_duration(&self, elapsed_ms: f64) {
        self.qdrant_search_duration_ms.record(elapsed_ms, &[]);
    }

    /// Record the latency of a Neo4j graph traversal.
    ///
    /// Wraps the `graph_search` dispatch inside the router, so the
    /// measurement covers the Cypher round-trip plus payload
    /// rehydration from the vector store.
    pub fn record_neo4j_traversal_duration(&self, elapsed_ms: f64) {
        self.neo4j_traversal_duration_ms.record(elapsed_ms, &[]);
    }

    /// Record the latency of a Postgres BM25 / FTS query.
    ///
    /// Recorded at the keyword fallback path triggered when the Qdrant
    /// circuit breaker is Open and the router degrades to BM25-only
    /// retrieval.
    pub fn record_postgres_fts_duration(&self, elapsed_ms: f64) {
        self.postgres_fts_duration_ms.record(elapsed_ms, &[]);
    }

    /// Record the latency of a single embedder call.
    ///
    /// Covers `Embedder::embed_text` end-to-end — OpenAI HTTP
    /// round-trip plus retries, or local fastembed ONNX inference,
    /// depending on the configured backend.
    pub fn record_embedder_duration(&self, elapsed_ms: f64) {
        self.embedder_duration_ms.record(elapsed_ms, &[]);
    }

    /// Record the latency of a Reciprocal Rank Fusion merge.
    ///
    /// Bounds the in-process CPU work in `ResultRanker::rank` and
    /// excludes back-end I/O.
    pub fn record_rrf_fusion_duration(&self, elapsed_ms: f64) {
        self.rrf_fusion_duration_ms.record(elapsed_ms, &[]);
    }

    /// Record the latency of a single reranker call.
    ///
    /// Recorded by the router around `CrossEncoderReranker::rerank`,
    /// so the measurement covers the HTTP round-trip and any
    /// back-off-driven retries.
    pub fn record_reranker_duration(&self, elapsed_ms: f64) {
        self.reranker_duration_ms.record(elapsed_ms, &[]);
    }

    /// Record one Qdrant→Postgres-FTS circuit-breaker fallback event,
    /// labelled by the retrieval strategy that triggered it.
    ///
    /// Increments when the breaker is Open and the router degrades a
    /// dense / hybrid request to BM25-only retrieval.
    pub fn record_qdrant_circuit_breaker_fallback(&self, strategy: RetrievalStrategy) {
        self.qdrant_circuit_breaker_fallback
            .add(1, &[strategy_attr(strategy)]);
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
