//! Context builder -- stateless.
//!
//! After the 2026-04-21 pivot this step is a thin wrapper: the
//! `QueryContext::new` constructor already materialises `chat_messages`
//! and `conversation_context` from the inline `history` argument. This
//! step only:
//!
//!   * Injects long-term memories (user-level, read from agent-rs Postgres)
//!     ahead of the history.
//!   * Appends a strategy hint to the user's query line.
//!
//! It no longer reads any DB tables related to sessions/chat_messages —
//! those were removed. Memories are the sole remaining DB read.

use std::sync::Arc;

use agent_rs_core::config::ContextConfig;
use agent_rs_core::errors::AgentError;
use agent_rs_stores::postgres::PgStore;
use tracing::{info, instrument, warn};

use crate::context::QueryContext;

/// Builds the chat message context for the agent.
///
/// Minimal responsibilities post-pivot:
/// - Inject long-term memories into the chat message list (user-level).
/// - Append the user's query + strategy hint at the end.
pub struct ContextBuilder {
    pg: Arc<PgStore>,
    #[allow(dead_code)]
    max_messages: usize,
    #[allow(dead_code)]
    max_chars_per_msg: usize,
}

impl ContextBuilder {
    /// Create a new ContextBuilder.
    pub fn new(pg: Arc<PgStore>, context_config: &ContextConfig) -> Self {
        Self {
            pg,
            max_messages: context_config.max_messages,
            max_chars_per_msg: context_config.max_chars_per_msg,
        }
    }

    /// Inject long-term memories + append the user query with a strategy hint.
    ///
    /// `chat_messages` was already pre-populated from the request-supplied
    /// history during `QueryContext::new`, so we only prepend memories (if
    /// any) and append the query itself here.
    ///
    /// # Errors
    ///
    /// - [`AgentError::MemoryStore`] — DB error while reading memories.
    #[instrument(skip(self, ctx), fields(
        user_id = %ctx.user_id,
        strategy = %ctx.strategy,
        history_len = ctx.chat_messages.len(),
    ))]
    pub async fn build(&self, ctx: &mut QueryContext) -> Result<(), AgentError> {
        // 1. Inject long-term memories for this user (user-level, not
        //    session-scoped). Fetched directly — no session transaction
        //    needed since there's no longer any session state to read
        //    consistently with.
        let memories = self
            .pg
            .list_memories(&ctx.user_id)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("list memories: {e}")))?;

        if !memories.is_empty() {
            if let Some(memory_context) = format_memory_context(&memories) {
                ctx.chat_messages.insert(0, memory_context);
                info!(count = memories.len(), "injected long-term memories");
            }
        } else {
            warn!("no long-term memories found for user");
        }

        // 2. Append the new user query with a strategy hint.
        let hint = format!("\n\n[System hint: prefer '{}' search strategy]", ctx.strategy);
        ctx.chat_messages.push(format!("{}{hint}", ctx.query));

        Ok(())
    }
}

/// Format user memories as a single leading system message.
///
/// Returns `None` for empty input so the caller can skip the injection.
fn format_memory_context(memories: &[agent_rs_stores::types::Memory]) -> Option<String> {
    if memories.is_empty() {
        return None;
    }
    let mut out = String::from("System: Long-term memories about this user:\n");
    for m in memories {
        out.push_str(&format!("- {}\n", m.content));
    }
    Some(out)
}
