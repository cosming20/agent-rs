//! QueryContext -- mutable state flowing through the orchestrator pipeline.
//!
//! Created once per `process_query()` call. Each pipeline step reads
//! what it needs and writes its outputs to this context.
//!
//! After the 2026-04-21 stateless pivot: `session_id` was removed. The
//! agent no longer owns any session state — conversation history is
//! supplied by the client inline via the `history` + `active_document_keys` +
//! `history_document_keys` fields on `AskRequest` and stored directly on
//! `QueryContext` at construction time. `trace_id` is an optional
//! client-supplied correlation UUID with no authorization meaning.

use std::sync::Arc;
use std::time::Instant;

use agent_rs_core::types::{LangCode, MessageRole, RetrievalStrategy, UserId};
use agent_rs_observability::RequestBudget;
use agent_rs_retrieval::EvidencePack;
use agent_rs_stores::types::{ChatTurn, SearchResult};
use uuid::Uuid;

/// Mutable state accumulating through the orchestrator pipeline.
///
/// Immutable request fields are set at construction. Mutable fields
/// are `Option` and populated by successive pipeline steps.
pub struct QueryContext {
    // ---- Immutable request parameters ----
    /// The user's query text.
    pub query: String,
    /// Optional client-supplied correlation UUID. Used only for linking
    /// telemetry across logs and traces; has no authorization meaning.
    pub trace_id: Option<Uuid>,
    /// User identifier for memory and authorization.
    pub user_id: UserId,
    /// Retrieval strategy to use (may be overridden by strategy prediction).
    pub strategy: RetrievalStrategy,
    /// Maximum number of evidence documents to retrieve.
    pub limit: usize,
    /// Role of the message sender.
    pub role: MessageRole,
    /// Language detected from the user's query.
    ///
    /// Set by [`crate::orchestrator::AgentOrchestrator::process_query`] before
    /// any pipeline step runs, via [`crate::lang::detect_lang`]. Defaults to
    /// [`LangCode::Other`] when [`QueryContext::new`] is invoked directly
    /// (test paths) or when the detector cannot identify the language.
    pub detected_lang: LangCode,
    /// Monotonic timestamp when the query started.
    pub start_time: Instant,

    // ---- Accumulated by ContextBuilder ----
    /// Chat history messages for the Rig agent. Populated directly from
    /// the `history: Vec<ChatTurn>` argument passed at construction —
    /// NEVER read from the DB (there is no such DB anymore).
    pub(crate) chat_messages: Vec<String>,
    /// Conversation context string for search augmentation, derived from
    /// the same client-supplied history.
    pub(crate) conversation_context: String,
    /// MinIO keys of documents active in this turn (plus history). The
    /// union `active ∪ history` determines the retrieval filter; coverage
    /// planning operates over the full union.
    ///
    /// Empty `Vec` triggers the full-library fallback on CoveragePlanner.
    /// (`Option<Vec<Uuid>>` used to carry the same signal pre-pivot; now
    /// the keys are `String`s and we treat empty-vec = no filter.)
    pub active_document_keys: Vec<String>,

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

    // ---- Coverage-aware RAG fields ----
    pub coverage_plan: Option<crate::coverage::CoveragePlan>,
    pub coverage_evidence: Vec<agent_rs_stores::types::SearchResult>,
    pub coverage_report: Option<crate::coverage::CoverageReport>,
    pub tool_calls: Vec<crate::coverage::ToolCallRecord>,

    // ---- Per-request LLM budget tracking ----
    /// Aggregates token + USD cost for every LLM call dispatched during
    /// this `process_query`. Constructed fresh per request and shared
    /// via `Arc` to any pipeline step that has access to the
    /// [`QueryContext`] and emits LLM calls (verifier, agent runner,
    /// downstream pipeline-time callers). Ingestion-time callers do
    /// not have a `QueryContext` and therefore do not update this
    /// field; per-call histograms still flow through
    /// [`agent_rs_observability::metrics::RetrievalMetrics`].
    pub budget: Arc<RequestBudget>,
}

impl QueryContext {
    /// Create a new stateless QueryContext.
    ///
    /// `history` is replayed inline by the caller (the web app). The
    /// orchestrator never reads or writes chat history to any persistent
    /// store — it exists only for the lifetime of this request.
    pub fn new(
        query: String,
        user_id: UserId,
        history: Vec<ChatTurn>,
        active_document_keys: Vec<String>,
        strategy: RetrievalStrategy,
        limit: usize,
        trace_id: Option<Uuid>,
    ) -> Self {
        // Materialize chat_messages + conversation_context directly from
        // the request-supplied history so the rest of the pipeline sees
        // the same shape it used to when ContextBuilder read from PG.
        let mut chat_messages: Vec<String> = Vec::with_capacity(history.len());
        let mut conversation_context = String::new();
        for turn in &history {
            let role_str = match turn.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
            };
            let line = format!("{role_str}: {}", turn.content);
            conversation_context.push_str(&line);
            conversation_context.push('\n');
            chat_messages.push(line);
        }

        Self {
            query,
            trace_id,
            user_id,
            strategy,
            limit,
            role: MessageRole::User,
            detected_lang: LangCode::Other,
            start_time: Instant::now(),
            chat_messages,
            conversation_context,
            active_document_keys,
            kb_evidence: Vec::new(),
            web_evidence: Vec::new(),
            answer: String::new(),
            agent_latency_ms: 0,
            evidence: Vec::new(),
            fused_pack: None,
            coverage_plan: None,
            coverage_evidence: Vec::new(),
            coverage_report: None,
            tool_calls: Vec::new(),
            budget: RequestBudget::new(),
        }
    }

    /// Milliseconds elapsed since the query started.
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}
