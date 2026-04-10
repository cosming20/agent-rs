//! `remember` tool -- stores facts in long-term memory.
//!
//! Persists user-specific information via the [`StateStore`] trait so
//! it can be recalled in future conversations.

use std::sync::Arc;

use agent_rs_core::{ToolError, UserId};
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::Memory;
use chrono::Utc;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

// ---- Args ----

/// Arguments for the `remember` tool.
#[derive(Debug, Deserialize)]
pub struct RememberArgs {
    /// The content to store in long-term memory.
    pub content: String,
}

// ---- Tool struct ----

/// Stores a fact or piece of information in the user's long-term memory.
///
/// The stored memory persists across sessions and can be retrieved
/// during context building to inform future conversations.
#[derive(Serialize, Deserialize)]
pub struct RememberTool {
    #[serde(skip)]
    store: Option<Arc<dyn StateStore>>,
    #[serde(skip)]
    user_id: Option<UserId>,
}

impl RememberTool {
    /// Create a new `RememberTool`.
    ///
    /// # Parameters
    ///
    /// - `store` -- the state store for persisting memories.
    /// - `user_id` -- the user whose memory this tool writes to.
    pub fn new(store: Arc<dyn StateStore>, user_id: UserId) -> Self {
        Self {
            store: Some(store),
            user_id: Some(user_id),
        }
    }
}

impl Tool for RememberTool {
    const NAME: &'static str = "remember";
    type Error = ToolError;
    type Args = RememberArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Store important information in long-term memory for future recall. \
                          Use when the user explicitly asks you to remember something, or when \
                          you identify a key fact that should persist across conversations."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or information to remember."
                    }
                },
                "required": ["content"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("remember: store not initialized".into()))?;
        let user_id = self
            .user_id
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("remember: user_id not set".into()))?;

        let memory = Memory {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            content: args.content.clone(),
            created_at: Utc::now().naive_utc(),
        };

        store
            .store_memory(&memory)
            .await
            .map_err(|e| ToolError::McpClient(format!("failed to store memory: {e}")))?;

        tracing::info!(
            user_id = %user_id,
            memory_id = %memory.id,
            "stored memory"
        );

        Ok(format!("Remembered: \"{}\"", args.content))
    }
}
