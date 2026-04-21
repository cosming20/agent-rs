//! Context builder -- loads history, builds conversation context, resolves strategy.
//!
//! First step in the orchestrator pipeline. Populates:
//! - `ctx.chat_messages` -- formatted message strings for the Rig agent
//! - `ctx.conversation_context` -- search augmentation context
//! - `ctx.active_document_ids` -- documents active in this session (if any)

use std::sync::Arc;

use agent_rs_core::config::ContextConfig;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::MessageRole;
use agent_rs_stores::postgres::PgStore;
use tracing::{info, instrument, warn};

use crate::context::QueryContext;
use crate::memory::MemoryManager;

/// Builds the chat message context for the agent.
///
/// Responsibilities:
/// - Ensure a session row exists and belongs to `ctx.user_id`
/// - Load history, memories, and active documents in one consistent snapshot
/// - Inject long-term memories
/// - Build conversation context string
/// - Inject strategy hints into chat messages
/// - Populate `ctx.active_document_ids` for downstream filtering
pub struct ContextBuilder {
    memory: Arc<MemoryManager>,
    pg: Arc<PgStore>,
    max_messages: usize,
    max_chars_per_msg: usize,
}

impl ContextBuilder {
    /// Create a new ContextBuilder.
    pub fn new(memory: Arc<MemoryManager>, pg: Arc<PgStore>, context_config: &ContextConfig) -> Self {
        Self {
            memory,
            pg,
            max_messages: context_config.max_messages,
            max_chars_per_msg: context_config.max_chars_per_msg,
        }
    }

    /// Build chat messages and conversation context on the QueryContext.
    ///
    /// # Steps
    ///
    /// 1. Ensure a session row exists (upsert-with-ownership check)
    /// 2. Store the user message in session history
    /// 3. Load history, memories, and active documents in one `REPEATABLE READ` snapshot
    /// 4. Inject long-term memories into the chat message list
    /// 5. Build conversation context string for search augmentation
    /// 6. Populate `ctx.active_document_ids` (`None` = full library fallback)
    /// 7. Add strategy hint to the final user message
    ///
    /// # Errors
    ///
    /// - [`AgentError::MemoryStore`] — session ownership mismatch or DB error.
    #[instrument(skip(self, ctx), fields(
        session_id = %ctx.session_id,
        user_id = %ctx.user_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn build(&self, ctx: &mut QueryContext) -> Result<(), AgentError> {
        // 1. Ensure the session row exists and belongs to this user.
        //    Returns SessionNotFound if the session_id is owned by another user.
        self.pg
            .create_or_fetch_session(&ctx.user_id, &ctx.session_id)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("session gate: {e}")))?;

        // 2. Store user message before the snapshot so the new turn is visible.
        self.memory
            .store_message(&ctx.user_id, &ctx.session_id, MessageRole::User, &ctx.query)
            .await?;

        // 3. Load history, memories, and active documents in one REPEATABLE READ
        //    transaction — all three see the same consistent DB snapshot.
        let (history_rows, memory_rows, active_doc_rows) = self
            .pg
            .read_context_snapshot(&ctx.user_id, &ctx.session_id)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("context snapshot: {e}")))?;

        // Convert raw chat-message rows into domain messages for history formatting.
        let history: Vec<agent_rs_stores::types::ChatMessage> = history_rows
            .into_iter()
            .filter_map(|r| r.into_domain().ok())
            .collect();

        let mut chat_messages: Vec<String> = Vec::new();

        // 4a. Convert history to strings (exclude the just-stored user message)
        for msg in history.iter().rev().skip(1).rev() {
            let prefix = match msg.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::System => "System",
            };
            chat_messages.push(format!("{prefix}: {}", msg.content));
        }

        // 4b. Inject long-term memories from snapshot rows.
        if !memory_rows.is_empty() {
            let memories: Vec<agent_rs_stores::types::Memory> = memory_rows
                .into_iter()
                .filter_map(|r| r.into_domain().ok())
                .collect();
            if let Some(memory_context) = MemoryManager::format_memory_context(&memories) {
                chat_messages.insert(0, memory_context);
                info!(count = memories.len(), "injected long-term memories");
            }
        } else {
            warn!("no long-term memories found for user");
        }

        // 5. Build conversation context for search augmentation (reuse history)
        ctx.conversation_context = self.memory.build_conversation_context(
            &history,
            self.max_messages,
            self.max_chars_per_msg,
        );

        // 6. Populate active_document_ids: None triggers full-library fallback.
        ctx.active_document_ids = if active_doc_rows.is_empty() {
            None
        } else {
            Some(active_doc_rows.into_iter().map(|d| d.document_id).collect())
        };

        // 7. Add strategy hint to final user message
        let hint = format!(
            "\n\n[System hint: prefer '{}' search strategy]",
            ctx.strategy
        );
        chat_messages.push(format!("{}{hint}", ctx.query));

        ctx.chat_messages = chat_messages;
        Ok(())
    }
}
