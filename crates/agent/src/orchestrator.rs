//! Agent orchestrator -- slim coordinator over pipeline step structs.
//!
//! Wires together context building, coverage planning, agent execution,
//! evidence fusion, deterministic coverage verification, an optional
//! pre-synth subject-presence gate, and result persistence into the
//! full query processing pipeline.
//!
//! The previous LLM-based grounding verifier and its retry loop were
//! deleted as part of the synthesizer-streams refactor. Implicit
//! grounding is enforced by the synthesis prompt ("only claim what's
//! in evidence"); the only deterministic safety net that survives is
//! [`crate::subject_check::subject_absent_v2`], which fires once
//! before synthesis on out-of-corpus queries to skip the wasted LLM
//! call.

use std::sync::Arc;

use agent_rs_core::config::{BudgetConfig, Settings};
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{RetrievalStrategy, UserId};
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_retrieval::{ResultRanker, RetrievalRouter};
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::ChatTurn;
use tracing::{info, instrument, warn};
use uuid::Uuid;

use crate::context::QueryContext;
use crate::lang::detect_lang;
use crate::steps::agent_runner::{AgentRunner, AgentRunnerConfig};
use crate::steps::context_builder::ContextBuilder;
use crate::steps::coverage_planner::CoveragePlanner;
use crate::steps::coverage_verifier::CoverageVerifier;
use crate::steps::evidence_fuser::EvidenceFuser;
use crate::steps::result_persister::ResultPersister;
use crate::steps::tool_factory::{ToolFactory, ToolFactoryDeps};
use crate::subject_check::subject_absent_v2;
use crate::synthesis::{ResponseSynthesizer, SynthesizedResponse, out_of_corpus_response};

/// Shared dependencies required to build an [`AgentOrchestrator`].
///
/// Bundling these into a single struct keeps the constructor signature
/// compact and call sites read as "give me an orchestrator with these
/// stores + the retrieval router + metrics" rather than a long positional
/// list of `Arc`s and collection names.
///
/// Construct once at startup; the orchestrator clones the shared `Arc`
/// handles internally, so callers may continue to use the originals after
/// passing this struct to [`AgentOrchestrator::new`].
pub struct OrchestratorDeps {
    /// Shared Neo4j graph store, used by the [`CoveragePlanner`] and the
    /// coverage-aware tools (`outline_corpus`, `fetch_section_chunks`,
    /// `search_sections`).
    pub neo4j: Arc<Neo4jStore>,
    /// Shared Postgres store for history, memory, runs, and ingested-
    /// document listing.
    pub pg_store: Arc<PgStore>,
    /// Shared Qdrant vector store for coverage-aware tools.
    pub qdrant: Arc<QdrantStore>,
    /// Name of the primary Qdrant chunk collection.
    pub default_collection: String,
    /// Name of the per-section summaries collection.
    pub summaries_collection: String,
    /// Shared retrieval router for knowledge search and claim-directed
    /// retries.
    pub router: Arc<RetrievalRouter>,
    /// Shared metric instruments for the retrieval pipeline.
    pub metrics: Arc<RetrievalMetrics>,
}

/// Inputs for a single [`AgentOrchestrator::process_query`] invocation.
///
/// Bundled into a named struct so the entrypoint stays at one positional
/// argument. Per-request — never reused across calls. Clone-on-construct
/// patterns (`String`, `Vec<String>`) are preserved verbatim from the
/// previous positional API to avoid surprising upstream gRPC handlers
/// that build these from owned protobuf payloads.
pub struct QueryRequest {
    /// Raw user question or instruction.
    pub query: String,
    /// Identifier of the requesting user.
    pub user_id: UserId,
    /// Prior conversation turns. Empty for a fresh thread. Replayed
    /// inline; never persisted on the agent side.
    pub history: Vec<ChatTurn>,
    /// MinIO keys of documents scoped to this request. The union
    /// `active ∪ history` determines the retrieval filter.
    pub active_document_keys: Vec<String>,
    /// Retrieval strategy to use.
    pub strategy: RetrievalStrategy,
    /// Maximum number of evidence documents to retrieve.
    pub limit: usize,
    /// Optional client correlation UUID for log linking.
    pub trace_id: Option<Uuid>,
}

/// Coordinates the full query processing pipeline.
///
/// Constructed once at startup with all shared dependencies. Each call
/// to [`process_query`] creates a [`QueryContext`] and runs it through
/// the step pipeline:
///
/// 1. **ContextBuilder** -- loads memories, materialises chat-history context
/// 2. **CoveragePlanner** -- detects exhaustive intent, builds section plan (no-op otherwise)
/// 3. **AgentRunner** -- runs the LLM agent with tools, accumulates evidence
/// 4. **EvidenceFuser** -- merges KB + web evidence via RRF
/// 5. **CoverageVerifier** -- checks plan satisfaction, populates coverage report (no-op when plan is None)
/// 6. **Pre-synth subject-absent check** -- deterministic out-of-corpus
///    short-circuit (see [`crate::subject_check::subject_absent_v2`])
///    that skips synthesis when no rewritten retrieval query shares
///    any content tokens with the fused evidence
/// 7. **ResultPersister** -- synthesizes response with citations,
///    persists run
pub struct AgentOrchestrator {
    context_builder: ContextBuilder,
    coverage_planner: CoveragePlanner,
    agent_runner: AgentRunner,
    evidence_fuser: EvidenceFuser,
    coverage_verifier: CoverageVerifier,
    result_persister: ResultPersister,
    /// Minimum character length for a query token in the pre-synth
    /// subject-absent gate. Sourced from
    /// [`agent_rs_core::config::SubjectCheckConfig::subject_min_token_len`].
    subject_min_token_len: usize,
    /// Shared metric instruments. Reused at end-of-request to emit
    /// the per-request budget histograms once the
    /// [`crate::context::QueryContext::budget`] snapshot is available.
    metrics: Arc<RetrievalMetrics>,
    /// Per-request LLM budget thresholds. Read at end of
    /// `process_query` to decide whether to emit a soft-cap warning.
    budget_config: BudgetConfig,
}

impl AgentOrchestrator {
    /// Create a new orchestrator wired with all dependencies.
    ///
    /// # Parameters
    ///
    /// - `settings` -- application configuration.
    /// - `deps` -- shared stores, retrieval router, and metric handles
    ///   bundled into [`OrchestratorDeps`].
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Orchestrator` if the LLM client cannot be created.
    pub fn new(settings: &Settings, deps: OrchestratorDeps) -> Result<Self, AgentError> {
        let OrchestratorDeps {
            neo4j,
            pg_store,
            qdrant,
            default_collection,
            summaries_collection,
            router,
            metrics,
        } = deps;
        // Erase the concrete PgStore to the StateStore trait for the
        // persister path that only needs trait-level methods.
        let store: Arc<dyn StateStore> = Arc::clone(&pg_store) as _;

        // Step 1: Context builder — now only reads user-level memories; no
        // session / history DB reads (history is replayed inline by the
        // client via `AskRequest.history`).
        let context_builder = ContextBuilder::new(Arc::clone(&pg_store));

        // Step 1.5: Coverage planner — no-op for non-exhaustive queries.
        // Clone before passing so neo4j remains available for the tool factory.
        let coverage_planner = CoveragePlanner::new(Arc::clone(&neo4j), Arc::clone(&metrics));

        // Tool factory (shared across requests; creates per-request tool sets).
        // Receives the concrete pg_store so DocumentInfoTool/OutlineCorpusTool
        // can call PgStore::list_ingested_documents(), which isn't part of the
        // StateStore trait. Also receives neo4j and qdrant for the
        // coverage-aware tools registered in Phase 4.6.
        let tool_factory = Arc::new(ToolFactory::new(ToolFactoryDeps {
            router: Arc::clone(&router),
            pg_store: Arc::clone(&pg_store),
            neo4j,
            qdrant,
            default_collection,
            summaries_collection,
            brave_config: settings.brave.clone(),
            fetch_config: settings.fetch.clone(),
            web_budget_config: settings.web_budget.clone(),
        })?);

        // Step 2: Agent runner — receives the shared metrics handle
        // because the per-request `AgentRsHook` plumbs every LLM
        // round-trip through `RetrievalMetrics::record_llm_call`.
        let agent_runner = AgentRunner::new(AgentRunnerConfig {
            api_key: &settings.llm.api_key,
            model: &settings.llm.model,
            max_turns: settings.agent.max_steps,
            system_prompt_file: &settings.agent.system_prompt_file,
            tool_factory,
            reasoning_effort: &settings.agent.reasoning_effort,
            seed: settings.agent.seed,
            metrics: Arc::clone(&metrics),
        })?;

        // Step 3: Evidence fuser
        let ranker = ResultRanker::new(settings.ranker.rrf_k);
        let evidence_fuser = EvidenceFuser::new(ranker, Arc::clone(&metrics));

        // Step 3.5: Coverage verifier — no-op when coverage_plan is None.
        let coverage_verifier = CoverageVerifier::new(Arc::clone(&metrics));

        // Step 5: Result persister — writes query_run + evidence_items for
        // telemetry. No longer writes chat_messages (that table is gone).
        let synthesizer = ResponseSynthesizer::new(
            settings.synthesis.max_snippet_length,
            settings.synthesis.fallback_citation_count,
        );
        let result_persister = ResultPersister::new(synthesizer, store, Arc::clone(&metrics));

        Ok(Self {
            context_builder,
            coverage_planner,
            agent_runner,
            evidence_fuser,
            coverage_verifier,
            result_persister,
            subject_min_token_len: settings.subject_check.subject_min_token_len,
            metrics,
            budget_config: settings.budget.clone(),
        })
    }

    /// Process a user query through the full agent pipeline.
    ///
    /// # Arguments
    ///
    /// - `request` -- per-call inputs bundled into [`QueryRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] variants on pipeline failures.
    #[instrument(skip(self, request), fields(
        user_id = %request.user_id,
        strategy = %request.strategy,
        limit = request.limit,
        history_len = request.history.len(),
        active_keys = request.active_document_keys.len(),
    ))]
    pub async fn process_query(
        &self,
        request: QueryRequest,
    ) -> Result<SynthesizedResponse, AgentError> {
        let QueryRequest {
            query,
            user_id,
            history,
            active_document_keys,
            strategy,
            limit,
            trace_id,
        } = request;
        let mut ctx = QueryContext::new(
            query,
            user_id,
            history,
            active_document_keys,
            strategy,
            limit,
            trace_id,
        );

        // Detect query language up-front so every downstream step (prompt
        // routing, lang-aware tokenization, telemetry) can branch on a
        // single typed value rather than re-running the detector.
        ctx.detected_lang = detect_lang(&ctx.query);

        info!(
            query = %ctx.query,
            strategy = %ctx.strategy,
            lang = %ctx.detected_lang,
            "pipeline started"
        );

        // Step 1: Build context (load history, inject memories, strategy hints)
        self.context_builder.build(&mut ctx).await?;

        // Step 1.5: Coverage planning — detects exhaustive intent and pre-builds
        // a section plan over Neo4j. No-op (immediate return) for non-exhaustive
        // queries; adds zero latency on definition / how-to flows.
        self.coverage_planner.plan(&mut ctx).await;

        // Step 2: Run agent with tools (populates answer + kb_evidence)
        self.agent_runner.run(&mut ctx).await?;

        // Step 3: Fuse evidence from KB and web sources
        self.evidence_fuser.fuse(&mut ctx);

        // Step 3.5: Verify coverage — compares fused evidence against the plan
        // and populates ctx.coverage_report. No-op when ctx.coverage_plan is None.
        self.coverage_verifier.verify(&mut ctx);

        // Step 4: Pre-synth subject-absent check.
        //
        // When the agent's rewritten retrieval queries share no
        // content tokens with the fused evidence, the topic is
        // effectively out-of-corpus. Skipping synthesis saves a full
        // LLM round-trip (~40 s on reasoning models) and returns a
        // polite "not in evidence" answer that downstream clients
        // already handle. The subject-absent guard is purely
        // deterministic (substring containment + tokenisation) — no
        // LLM call, no retries.
        if subject_absent_v2(
            &ctx.query,
            &ctx.tool_calls,
            &ctx.evidence,
            self.subject_min_token_len,
        ) {
            warn!(
                query = %ctx.query,
                evidence_count = ctx.evidence.len(),
                "pre-synth subject-absent gate fired — short-circuiting synthesis"
            );
            self.metrics
                .record_subject_check_short_circuit(ctx.strategy);
            let response =
                out_of_corpus_response(&ctx.query, ctx.strategy, ctx.coverage_report.as_ref())?;
            self.emit_request_budget_summary(&ctx);
            info!(
                confidence = response.confidence.value(),
                citations = response.citations.len(),
                elapsed_ms = ctx.elapsed_ms(),
                "pipeline complete (subject-absent short-circuit)"
            );
            return Ok(response);
        }

        // Step 5: Synthesize response with citations, persist run
        let response = self.result_persister.persist(&ctx).await?;

        self.emit_request_budget_summary(&ctx);

        info!(
            confidence = response.confidence.value(),
            citations = response.citations.len(),
            elapsed_ms = ctx.elapsed_ms(),
            "pipeline complete"
        );

        Ok(response)
    }

    /// Emit per-request LLM budget histograms + structured logs.
    ///
    /// Lifted out of [`Self::process_query`] so both the synthesis and
    /// pre-synth-short-circuit paths emit the same telemetry without
    /// duplicated code. Always logs at INFO so dashboards can graph
    /// cost per query; escalates to WARN when a configured soft cap
    /// is exceeded.
    fn emit_request_budget_summary(&self, ctx: &QueryContext) {
        let snapshot = ctx.budget.snapshot();
        self.metrics
            .record_request_total_tokens(snapshot.total_tokens);
        self.metrics
            .record_request_total_cost_usd(snapshot.cost_usd);
        info!(
            call_count = snapshot.call_count,
            prompt_tokens = snapshot.prompt_tokens,
            completion_tokens = snapshot.completion_tokens,
            total_tokens = snapshot.total_tokens,
            cost_usd = snapshot.cost_usd,
            "request budget snapshot"
        );
        let token_over = snapshot.total_tokens > self.budget_config.soft_token_limit;
        let cost_over = snapshot.cost_usd > self.budget_config.soft_cost_usd;
        if token_over || cost_over {
            warn!(
                call_count = snapshot.call_count,
                total_tokens = snapshot.total_tokens,
                soft_token_limit = self.budget_config.soft_token_limit,
                cost_usd = snapshot.cost_usd,
                soft_cost_usd = self.budget_config.soft_cost_usd,
                token_over,
                cost_over,
                "request exceeded soft budget threshold"
            );
        }
    }
}
