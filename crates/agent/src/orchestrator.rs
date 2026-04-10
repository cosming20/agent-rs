//! Agent orchestrator -- slim coordinator over pipeline step structs.
//!
//! Wires together context building, agent execution, evidence fusion,
//! grounding verification, and result persistence into the full query
//! processing pipeline.

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{RetrievalStrategy, SessionId, UserId};

use crate::synthesis::SynthesizedResponse;

/// Coordinates the full query processing pipeline.
///
/// Constructed once at startup. Each call to [`process_query`] creates a
/// [`QueryContext`] and runs it through the step pipeline.
///
/// The full step implementation (context building, LLM agent, evidence fusion,
/// grounding verification, result persistence) is completed in a later task.
/// The signature is defined here so the server crate can compile against it.
pub struct AgentOrchestrator;

impl AgentOrchestrator {
    /// Process a user query through the full agent pipeline.
    ///
    /// # Arguments
    ///
    /// - `query` — the raw user question or instruction.
    /// - `session_id` — conversation session for history loading.
    /// - `user_id` — identifier of the requesting user.
    /// - `strategy` — retrieval strategy to use.
    /// - `limit` — maximum number of evidence documents to retrieve.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] variants on pipeline failures (orchestration,
    /// verification, synthesis, retrieval, or store errors).
    pub async fn process_query(
        &self,
        query: String,
        session_id: SessionId,
        user_id: UserId,
        strategy: RetrievalStrategy,
        limit: usize,
    ) -> Result<SynthesizedResponse, AgentError> {
        // Full pipeline implementation is a subsequent task.
        // Return a placeholder so the server crate can compile.
        let _ = (query, session_id, user_id, strategy, limit);
        Err(AgentError::Orchestrator(
            "process_query not yet implemented".into(),
        ))
    }
}
