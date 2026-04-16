//! Context builder -- loads history, builds conversation context, resolves strategy.
//!
//! First step in the orchestrator pipeline. Populates:
//! - `ctx.chat_messages` -- formatted message strings for the Rig agent
//! - `ctx.conversation_context` -- search augmentation context

use std::sync::Arc;

use agent_rs_core::config::ContextConfig;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::MessageRole;
use tracing::{info, instrument, warn};

use crate::context::QueryContext;
use crate::memory::MemoryManager;

/// Builds the chat message context for the agent.
///
/// Responsibilities:
/// - Store user message in memory
/// - Load conversation history (short-term messages)
/// - Inject long-term memories
/// - Build conversation context string
/// - Inject strategy hints into chat messages
pub struct ContextBuilder {
    memory: Arc<MemoryManager>,
    max_messages: usize,
    max_chars_per_msg: usize,
}

impl ContextBuilder {
    /// Create a new ContextBuilder.
    pub fn new(memory: Arc<MemoryManager>, context_config: &ContextConfig) -> Self {
        Self {
            memory,
            max_messages: context_config.max_messages,
            max_chars_per_msg: context_config.max_chars_per_msg,
        }
    }

    /// Build chat messages and conversation context on the QueryContext.
    ///
    /// # Steps
    ///
    /// 1. Store the user message in session history
    /// 2. Load recent conversation history
    /// 3. Load and inject long-term memories
    /// 4. Build conversation context string for search augmentation
    /// 5. Add strategy hint to the final user message
    #[instrument(skip(self, ctx), fields(
        session_id = %ctx.session_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn build(&self, ctx: &mut QueryContext) -> Result<(), AgentError> {
        // 1. Store user message
        self.memory
            .store_message(&ctx.session_id, MessageRole::User, &ctx.query)
            .await?;

        // 2. Load conversation history (single query, reused for both
        //    chat formatting and conversation context).
        let history = self
            .memory
            .load_history(&ctx.session_id, self.max_messages)
            .await?;

        let mut chat_messages: Vec<String> = Vec::new();

        // Convert history to strings (exclude the just-stored user message)
        for msg in history.iter().rev().skip(1).rev() {
            let prefix = match msg.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::System => "System",
            };
            chat_messages.push(format!("{prefix}: {}", msg.content));
        }

        // 3. Inject long-term memories
        match self.memory.get_long_term_memories(&ctx.user_id).await {
            Ok(memories) => {
                if let Some(memory_context) = MemoryManager::format_memory_context(&memories) {
                    chat_messages.insert(0, memory_context);
                    info!(count = memories.len(), "injected long-term memories");
                }
            }
            Err(err) => {
                warn!(error = %err, "failed to load long-term memories, continuing without");
                ctx.conversation_context.push_str(
                    "\n[System: Long-term memory unavailable for this query.]",
                );
            }
        }

        // 4. Build conversation context for search augmentation (reuse history)
        ctx.conversation_context = self.memory.build_conversation_context(
            &history,
            self.max_messages,
            self.max_chars_per_msg,
        );

        // 5. Add strategy hint to final user message
        let hint = format!(
            "\n\n[System hint: prefer '{}' search strategy]",
            ctx.strategy
        );
        chat_messages.push(format!("{}{hint}", ctx.query));

        ctx.chat_messages = chat_messages;
        Ok(())
    }
}
