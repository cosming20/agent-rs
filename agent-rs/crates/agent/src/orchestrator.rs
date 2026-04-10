//! Agent orchestrator -- slim coordinator over pipeline step structs.
//!
//! Wires together context building, agent execution, evidence fusion,
//! grounding verification, and result persistence into the full query
//! processing pipeline.

use std::sync::Arc;

use agent_rs_core::config::Settings;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{MessageRole, RetrievalStrategy, SessionId, UserId};
use agent_rs_retrieval::{ResultRanker, RetrievalRouter};
use agent_rs_stores::traits::StateStore;
use tracing::{info, instrument};

use crate::context::QueryContext;
use crate::memory::MemoryManager;
use crate::steps::agent_runner::AgentRunner;
use crate::steps::context_builder::ContextBuilder;
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
/// 2. **AgentRunner** -- runs the LLM agent with tools, accumulates evidence
/// 3. **EvidenceFuser** -- merges KB + web evidence via RRF
/// 4. **GroundingGate** -- verifies grounding, retries if needed
/// 5. **ResultPersister** -- synthesizes response with citations, persists run
pub struct AgentOrchestrator {
    context_builder: ContextBuilder,
    agent_runner: AgentRunner,
    evidence_fuser: EvidenceFuser,
    grounding_gate: GroundingGate,
    result_persister: ResultPersister,
}

impl AgentOrchestrator {
    /// Create a new orchestrator wired with all dependencies.
    ///
    /// # Parameters
    ///
    /// - `settings` -- application configuration.
    /// - `store` -- shared state store (Postgres) for history, memory, runs.
    /// - `router` -- shared retrieval router for knowledge search.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Orchestrator` if the LLM client cannot be created.
    pub fn new(
        settings: &Settings,
        store: Arc<dyn StateStore>,
        router: Arc<RetrievalRouter>,
    ) -> Result<Self, AgentError> {
        // Shared memory manager over the state store.
        let memory = Arc::new(MemoryManager::new(Arc::clone(&store)));

        // Step 1: Context builder
        let context_builder = ContextBuilder::new(Arc::clone(&memory), &settings.context);

        // Tool factory (shared across requests; creates per-request tool sets)
        let tool_factory = Arc::new(ToolFactory::new(
            Arc::clone(&router),
            Arc::clone(&store),
            settings.brave.clone(),
            settings.fetch.clone(),
            settings.web_budget.clone(),
        ));

        // Step 2: Agent runner
        let agent_runner = AgentRunner::new(
            &settings.llm.api_key,
            &settings.llm.model,
            settings.agent.max_steps,
            &settings.agent.system_prompt_file,
            tool_factory,
        )?;

        // Step 3: Evidence fuser
        let ranker = ResultRanker::new(settings.ranker.rrf_k);
        let evidence_fuser = EvidenceFuser::new(ranker);

        // Step 4: Grounding gate
        let verifier = Arc::new(GroundingVerifier::new(
            settings.verifier.confidence_threshold,
            &settings.llm.api_key,
            &settings.llm.model,
        ));
        let grounding_gate = GroundingGate::new(verifier, settings.agent.max_retries);

        // Step 5: Result persister
        let synthesizer = ResponseSynthesizer::new(settings.synthesis.max_snippet_length);
        let result_persister = ResultPersister::new(synthesizer, memory, store);

        Ok(Self {
            context_builder,
            agent_runner,
            evidence_fuser,
            grounding_gate,
            result_persister,
        })
    }

    /// Process a user query through the full agent pipeline.
    ///
    /// # Arguments
    ///
    /// - `query` -- the raw user question or instruction.
    /// - `session_id` -- conversation session for history loading.
    /// - `user_id` -- identifier of the requesting user.
    /// - `strategy` -- retrieval strategy to use.
    /// - `limit` -- maximum number of evidence documents to retrieve.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] variants on pipeline failures (orchestration,
    /// verification, synthesis, retrieval, or store errors).
    #[instrument(skip(self, query), fields(
        %session_id,
        %user_id,
        strategy = %strategy,
        limit,
    ))]
    pub async fn process_query(
        &self,
        query: String,
        session_id: SessionId,
        user_id: UserId,
        strategy: RetrievalStrategy,
        limit: usize,
    ) -> Result<SynthesizedResponse, AgentError> {
        let mut ctx = QueryContext::new(
            query,
            session_id,
            user_id,
            strategy,
            limit,
            MessageRole::User,
        );

        info!(
            query = %ctx.query,
            strategy = %ctx.strategy,
            "pipeline started"
        );

        // Step 1: Build context (load history, inject memories, strategy hints)
        self.context_builder.build(&mut ctx).await?;

        // Step 2: Run agent with tools (populates answer + kb_evidence)
        self.agent_runner.run(&mut ctx).await?;

        // Step 3: Fuse evidence from KB and web sources
        self.evidence_fuser.fuse(&mut ctx);

        // Step 4: Verify grounding, retry if needed
        self.grounding_gate
            .verify_and_retry(&mut ctx, &self.agent_runner, &self.evidence_fuser)
            .await?;

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
