//! Agent orchestrator -- slim coordinator over pipeline step structs.
//!
//! Wires together context building, agent execution, evidence fusion,
//! grounding verification, and result persistence into the full query
//! processing pipeline.

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

use crate::claim_searcher::{ClaimSearcher, RouterClaimSearcher};
use crate::context::QueryContext;
use crate::lang::detect_lang;
use crate::steps::agent_runner::AgentRunner;
use crate::steps::context_builder::ContextBuilder;
use crate::steps::coverage_planner::CoveragePlanner;
use crate::steps::coverage_verifier::CoverageVerifier;
use crate::steps::evidence_fuser::EvidenceFuser;
use crate::steps::grounding_gate::{GroundingGate, GroundingGateConfig};
use crate::steps::result_persister::ResultPersister;
use crate::steps::tool_factory::{ToolFactory, ToolFactoryDeps};
use crate::synthesis::{ResponseSynthesizer, SynthesizedResponse};
use crate::verifier::GroundingVerifier;

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
/// 1. **ContextBuilder** -- loads history, builds conversation context
/// 2. **CoveragePlanner** -- detects exhaustive intent, builds section plan (no-op otherwise)
/// 3. **AgentRunner** -- runs the LLM agent with tools, accumulates evidence
/// 4. **EvidenceFuser** -- merges KB + web evidence via RRF
/// 5. **CoverageVerifier** -- checks plan satisfaction, populates coverage report (no-op when plan is None)
/// 6. **GroundingGate** -- verifies grounding, retries if needed
/// 7. **ResultPersister** -- synthesizes response with citations, persists run
pub struct AgentOrchestrator {
    context_builder: ContextBuilder,
    coverage_planner: CoveragePlanner,
    agent_runner: AgentRunner,
    evidence_fuser: EvidenceFuser,
    coverage_verifier: CoverageVerifier,
    grounding_gate: GroundingGate,
    result_persister: ResultPersister,
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

        // Step 2: Agent runner
        let agent_runner = AgentRunner::new(
            &settings.llm.api_key,
            &settings.llm.model,
            settings.agent.max_steps,
            &settings.agent.system_prompt_file,
            tool_factory,
            &settings.agent.reasoning_effort,
        )?;

        // Step 3: Evidence fuser
        let ranker = ResultRanker::new(settings.ranker.rrf_k);
        let evidence_fuser = EvidenceFuser::new(ranker, Arc::clone(&metrics));

        // Step 3.5: Coverage verifier — no-op when coverage_plan is None.
        let coverage_verifier = CoverageVerifier::new(Arc::clone(&metrics));

        // Step 4: Grounding gate
        //
        // The verifier can be run on a different (typically stronger)
        // model than the primary retrieval LLM — set
        // `verifier.model = "gpt-5.4-mini"` (or similar) in settings to
        // override. When unset, falls back to the main `llm.model`.
        let verifier_model = settings
            .verifier
            .model
            .as_deref()
            .unwrap_or(settings.llm.model.as_str());
        let verifier = Arc::new(
            GroundingVerifier::new(
                settings.verifier.confidence_threshold,
                &settings.llm.api_key,
                verifier_model,
            )
            .with_metrics(Arc::clone(&metrics)),
        );
        // Claim-directed retry handle. Wraps the same shared retrieval
        // router the agent's `search_knowledge` tool uses, so a
        // claim-directed search shares cache, embedder, and
        // cross-encoder rerank state with the in-turn retrieval.
        let claim_searcher: Arc<dyn ClaimSearcher> =
            Arc::new(RouterClaimSearcher::new(Arc::clone(&router)));
        let grounding_gate = GroundingGate::new(
            verifier,
            claim_searcher,
            Arc::clone(&metrics),
            GroundingGateConfig {
                max_retries: settings.agent.max_retries,
                subject_min_token_len: settings.grounding.subject_min_token_len,
                retry_strategy: settings.grounding.retry_strategy,
                claim_search_top_k: settings.grounding.claim_search_top_k,
            },
        );

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
            grounding_gate,
            result_persister,
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

        // Step 4: Verify grounding, retry if needed
        self.grounding_gate
            .verify_and_retry(&mut ctx, &self.agent_runner, &self.evidence_fuser)
            .await?;

        // Step 4.5: Re-run CoverageVerifier when the grounding gate performed
        // at least one retry. Each retry drains a fresh coverage_collector into
        // ctx.coverage_evidence (accumulated), so the stale pre-retry report
        // must be refreshed to reflect the full post-retry evidence set.
        if ctx.retries > 0 {
            self.coverage_verifier.verify(&mut ctx);
        }

        // Step 5: Synthesize response with citations, persist run
        let response = self.result_persister.persist(&ctx).await?;

        // Per-request LLM budget summary. Always logged so dashboards
        // can graph cost per query; warns on top when a soft cap is
        // exceeded. Histograms are emitted once per request so SLO
        // calculators can pull P99 cost / tokens directly from OTel.
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

        info!(
            confidence = response.confidence.value(),
            is_grounded = response.is_grounded,
            citations = response.citations.len(),
            elapsed_ms = ctx.elapsed_ms(),
            "pipeline complete"
        );

        Ok(response)
    }
}
