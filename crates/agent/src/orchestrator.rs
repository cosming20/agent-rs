//! Agent orchestrator -- slim coordinator over pipeline step structs.
//!
//! Wires together context building, agent execution, evidence fusion,
//! grounding verification, and result persistence into the full query
//! processing pipeline.

use std::sync::Arc;

use agent_rs_core::config::Settings;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{RetrievalStrategy, UserId};
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_retrieval::{ResultRanker, RetrievalRouter};
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::ChatTurn;
use tracing::{info, instrument};
use uuid::Uuid;

use crate::context::QueryContext;
use crate::steps::agent_runner::AgentRunner;
use crate::steps::context_builder::ContextBuilder;
use crate::steps::coverage_planner::CoveragePlanner;
use crate::steps::coverage_verifier::CoverageVerifier;
use crate::steps::evidence_fuser::EvidenceFuser;
use crate::steps::grounding_gate::GroundingGate;
use crate::steps::result_persister::ResultPersister;
use crate::steps::tool_factory::ToolFactory;
use crate::synthesis::{ResponseSynthesizer, SynthesizedResponse};
use crate::verifier::GroundingVerifier;

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
}

impl AgentOrchestrator {
    /// Create a new orchestrator wired with all dependencies.
    ///
    /// # Parameters
    ///
    /// - `settings` -- application configuration.
    /// - `neo4j` -- shared Neo4j graph store, used by [`CoveragePlanner`] and
    ///   the coverage-aware tools (`outline_corpus`, `fetch_section_chunks`,
    ///   `search_sections`).
    /// - `pg_store` -- shared Postgres store for history, memory, runs, and
    ///   ingested-document listing.
    /// - `qdrant` -- shared Qdrant vector store for coverage-aware tools.
    /// - `default_collection` -- name of the primary Qdrant chunk collection.
    /// - `summaries_collection` -- name of the per-section summaries collection.
    /// - `router` -- shared retrieval router for knowledge search.
    /// - `metrics` -- shared metric instruments for the retrieval pipeline.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Orchestrator` if the LLM client cannot be created.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        settings: &Settings,
        neo4j: Arc<Neo4jStore>,
        pg_store: Arc<PgStore>,
        qdrant: Arc<QdrantStore>,
        default_collection: String,
        summaries_collection: String,
        router: Arc<RetrievalRouter>,
        metrics: Arc<RetrievalMetrics>,
    ) -> Result<Self, AgentError> {
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
        let tool_factory = Arc::new(ToolFactory::new(
            Arc::clone(&router),
            Arc::clone(&pg_store),
            neo4j,
            qdrant,
            default_collection,
            summaries_collection,
            settings.brave.clone(),
            settings.fetch.clone(),
            settings.web_budget.clone(),
        )?);

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
        let verifier = Arc::new(GroundingVerifier::new(
            settings.verifier.confidence_threshold,
            &settings.llm.api_key,
            verifier_model,
        ));
        let grounding_gate = GroundingGate::new(
            verifier,
            settings.agent.max_retries,
            settings.grounding.subject_min_token_len,
            Arc::clone(&metrics),
        );

        // Step 5: Result persister — writes query_run + evidence_items for
        // telemetry. No longer writes chat_messages (that table is gone).
        let synthesizer = ResponseSynthesizer::new(
            settings.synthesis.max_snippet_length,
            settings.synthesis.fallback_citation_count,
        );
        let result_persister = ResultPersister::new(synthesizer, store, metrics);

        Ok(Self {
            context_builder,
            coverage_planner,
            agent_runner,
            evidence_fuser,
            coverage_verifier,
            grounding_gate,
            result_persister,
        })
    }

    /// Process a user query through the full agent pipeline.
    ///
    /// # Arguments
    ///
    /// - `query` -- the raw user question or instruction.
    /// - `user_id` -- identifier of the requesting user.
    /// - `history` -- prior conversation turns (empty for a fresh thread).
    ///   Replayed inline; never persisted on the agent side.
    /// - `active_document_keys` -- MinIO keys of documents scoped to this
    ///   request (union `active ∪ history` determines retrieval filter).
    /// - `strategy` -- retrieval strategy to use.
    /// - `limit` -- maximum number of evidence documents to retrieve.
    /// - `trace_id` -- optional client correlation UUID for log linking.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] variants on pipeline failures.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, query, history, active_document_keys), fields(
        %user_id,
        strategy = %strategy,
        limit,
        history_len = history.len(),
        active_keys = active_document_keys.len(),
    ))]
    pub async fn process_query(
        &self,
        query: String,
        user_id: UserId,
        history: Vec<ChatTurn>,
        active_document_keys: Vec<String>,
        strategy: RetrievalStrategy,
        limit: usize,
        trace_id: Option<Uuid>,
    ) -> Result<SynthesizedResponse, AgentError> {
        let mut ctx = QueryContext::new(
            query,
            user_id,
            history,
            active_document_keys,
            strategy,
            limit,
            trace_id,
        );

        info!(
            query = %ctx.query,
            strategy = %ctx.strategy,
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
