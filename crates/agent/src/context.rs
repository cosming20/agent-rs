//! QueryContext -- mutable state flowing through the orchestrator pipeline.
//!
//! Created once per `process_query()` call. Each pipeline step reads
//! what it needs and writes its outputs to this context.

use std::time::Instant;

use agent_rs_core::types::{MessageRole, RetrievalStrategy, SessionId, UserId};
use agent_rs_retrieval::EvidencePack;
use agent_rs_stores::types::SearchResult;

use crate::verifier::VerificationResult;

/// Mutable state accumulating through the orchestrator pipeline.
///
/// Immutable request fields are set at construction. Mutable fields
/// are `Option` and populated by successive pipeline steps.
pub struct QueryContext {
    // ---- Immutable request parameters ----
    /// The user's query text.
    pub query: String,
    /// Session identifier for conversation continuity.
    pub session_id: SessionId,
    /// User identifier for memory and authorization.
    pub user_id: UserId,
    /// Retrieval strategy to use (may be overridden by strategy prediction).
    pub strategy: RetrievalStrategy,
    /// Maximum number of evidence documents to retrieve.
    pub limit: usize,
    /// Role of the message sender.
    pub role: MessageRole,
    /// Monotonic timestamp when the query started.
    pub start_time: Instant,

    // ---- Accumulated by ContextBuilder ----
    /// Chat history messages for the Rig agent.
    pub(crate) chat_messages: Vec<String>,
    /// Conversation context string for search augmentation.
    pub(crate) conversation_context: String,

    // ---- Accumulated by AgentRunner ----
    /// Knowledge-base evidence accumulated by tool calls.
    pub(crate) kb_evidence: Vec<SearchResult>,
    /// Web evidence accumulated by tool calls.
    pub(crate) web_evidence: Vec<SearchResult>,
    /// Raw answer text from the Rig agent.
    pub(crate) answer: String,
    /// Latency of the agent execution in milliseconds.
    pub(crate) agent_latency_ms: u64,

    // ---- Accumulated by EvidenceFuser ----
    /// Fused and ranked evidence documents.
    pub(crate) evidence: Vec<SearchResult>,
    /// The full evidence pack with provenance metadata.
    pub(crate) fused_pack: Option<EvidencePack>,

    // ---- Accumulated by GroundingGate ----
    /// Verification result from the grounding check.
    pub(crate) verification: Option<VerificationResult>,
    /// Number of retry attempts performed.
    pub(crate) retries: usize,

    // ---- Coverage-aware RAG fields (Phase 4) ----
    /// Plan emitted by `CoveragePlanner` (Task 4.1).
    /// `None` for non-exhaustive flows (current default).
    pub coverage_plan: Option<crate::coverage::CoveragePlan>,

    /// Untruncated evidence merged in by exhaustive-mode tool calls.
    /// `EvidenceFuser` (Task 4.3) will preserve this set verbatim alongside
    /// the reranked tier in `evidence`.
    pub coverage_evidence: Vec<agent_rs_stores::types::SearchResult>,

    /// Verifier output. Populated by `CoverageVerifier` (Task 4.2). `None` when
    /// the query was non-exhaustive or no plan existed.
    pub coverage_report: Option<crate::coverage::CoverageReport>,

    /// Captured tool invocations (args + tool name). Used by the grounding
    /// gate (Task 5.1) to extract subject tokens from the rewritten query
    /// rather than from the raw user prompt.
    pub tool_calls: Vec<crate::coverage::ToolCallRecord>,
}

impl QueryContext {
    /// Create a new QueryContext with immutable request parameters.
    ///
    /// All mutable fields are initialized to empty/default values.
    pub fn new(
        query: String,
        session_id: SessionId,
        user_id: UserId,
        strategy: RetrievalStrategy,
        limit: usize,
        role: MessageRole,
    ) -> Self {
        Self {
            query,
            session_id,
            user_id,
            strategy,
            limit,
            role,
            start_time: Instant::now(),
            chat_messages: Vec::new(),
            conversation_context: String::new(),
            kb_evidence: Vec::new(),
            web_evidence: Vec::new(),
            answer: String::new(),
            agent_latency_ms: 0,
            evidence: Vec::new(),
            fused_pack: None,
            verification: None,
            retries: 0,
            coverage_plan: None,
            coverage_evidence: Vec::new(),
            coverage_report: None,
            tool_calls: Vec::new(),
        }
    }

    /// Milliseconds elapsed since the query started.
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}
