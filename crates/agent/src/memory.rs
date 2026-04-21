//! Session (short-term) and persistent (long-term) memory over StateStore.
//!
//! [`MemoryManager`] wraps the [`StateStore`] trait and provides
//! higher-level operations for the orchestrator pipeline.

use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{MessageRole, SessionId, UserId, truncate_to_char_boundary};
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::{ChatMessage, Memory};
use chrono::Utc;
use tracing::instrument;
use uuid::Uuid;

/// Higher-level memory abstraction over the StateStore trait.
///
/// Provides session (short-term) chat history and persistent (long-term)
/// user memory operations for the orchestrator pipeline.
pub struct MemoryManager {
    store: Arc<dyn StateStore>,
}

impl MemoryManager {
    /// Create a new MemoryManager wrapping a StateStore implementation.
    pub fn new(store: Arc<dyn StateStore>) -> Self {
        Self { store }
    }

    /// Load recent conversation messages for a session.
    ///
    /// Returns messages ordered oldest-first, limited to `max_messages`.
    #[instrument(skip(self), fields(%session_id, max_messages))]
    pub async fn load_history(
        &self,
        session_id: &SessionId,
        max_messages: usize,
    ) -> Result<Vec<ChatMessage>, AgentError> {
        self.store
            .get_messages(session_id, max_messages)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("load history: {e}")))
    }

    /// Build a conversation context string from chat history.
    ///
    /// Concatenates the last `max_messages` user/assistant messages,
    /// truncating each to `max_chars_per_msg` characters.
    pub fn build_conversation_context(
        &self,
        messages: &[ChatMessage],
        max_messages: usize,
        max_chars_per_msg: usize,
    ) -> String {
        let relevant: Vec<&ChatMessage> = messages
            .iter()
            .filter(|m| matches!(m.role, MessageRole::User | MessageRole::Assistant))
            .collect();

        let window: Vec<&ChatMessage> = if relevant.len() > max_messages {
            relevant[relevant.len() - max_messages..].to_vec()
        } else {
            relevant
        };

        window
            .iter()
            .filter_map(|msg| {
                if msg.content.is_empty() {
                    None
                } else {
                    Some(truncate_to_char_boundary(&msg.content, max_chars_per_msg))
                }
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Store a user or assistant message in the session history.
    ///
    /// `user_id` is persisted alongside the message so that
    /// [`PgStore::list_chat_messages`] and the snapshot query can enforce
    /// per-tenant ownership at the SQL level.
    #[instrument(skip(self, content), fields(%session_id, %user_id, ?role))]
    pub async fn store_message(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
        role: MessageRole,
        content: &str,
    ) -> Result<(), AgentError> {
        let message = ChatMessage {
            id: Uuid::new_v4(),
            session_id: session_id.clone(),
            user_id: user_id.clone(),
            role,
            content: content.to_owned(),
            created_at: Utc::now().naive_utc(),
        };
        self.store
            .store_message(&message)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("store message: {e}")))
    }

    /// Store both the user query and assistant response in one call.
    pub async fn store_interaction(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
        user_query: &str,
        assistant_response: &str,
    ) -> Result<(), AgentError> {
        self.store_message(user_id, session_id, MessageRole::User, user_query)
            .await?;
        self.store_message(user_id, session_id, MessageRole::Assistant, assistant_response)
            .await
    }

    /// Get all long-term memories for a user.
    #[instrument(skip(self), fields(%user_id))]
    pub async fn get_long_term_memories(
        &self,
        user_id: &UserId,
    ) -> Result<Vec<Memory>, AgentError> {
        self.store
            .get_memories(user_id)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("get memories: {e}")))
    }

    /// Store a long-term memory fact for a user.
    #[instrument(skip(self, content), fields(%user_id))]
    pub async fn store_memory(&self, user_id: &UserId, content: &str) -> Result<(), AgentError> {
        let memory = Memory {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            content: content.to_owned(),
            created_at: Utc::now().naive_utc(),
        };
        self.store
            .store_memory(&memory)
            .await
            .map_err(|e| AgentError::MemoryStore(format!("store memory: {e}")))
    }

    /// Build memory context string from long-term memories.
    ///
    /// Used to inject user-specific knowledge into the agent's system prompt.
    pub fn format_memory_context(memories: &[Memory]) -> Option<String> {
        if memories.is_empty() {
            return None;
        }
        let lines: Vec<String> = memories
            .iter()
            .map(|m| format!("- {}", m.content))
            .collect();
        Some(format!("[User's stored memories]\n{}", lines.join("\n")))
    }
}
