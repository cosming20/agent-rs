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
    /// Tenant scope baked in at construction time. The LLM never sees or
    /// controls this field — it is excluded from the JSON schema.
    #[serde(skip, default = "default_unset_user_id")]
    user_id: UserId,
}

/// Sentinel `UserId` for contract-test deserialization path.
fn default_unset_user_id() -> UserId {
    UserId::try_new("_unset_test_only").expect("sentinel must pass validator")
}

impl RememberTool {
    /// Create a new `RememberTool` scoped to `user_id`.
    ///
    /// # Parameters
    ///
    /// - `store` -- the state store for persisting memories.
    /// - `user_id` -- the tenant scope baked in at construction time; never
    ///   exposed in the tool JSON schema.
    pub fn new_for_user(store: Arc<dyn StateStore>, user_id: UserId) -> Self {
        Self {
            store: Some(store),
            user_id,
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

    #[tracing::instrument(skip(self), fields(tool = Self::NAME, user_id = %self.user_id))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Sentinel guard: fails fast if the tool was constructed via the
        // serde-skip default path (contract tests) rather than new_for_user().
        assert!(
            self.user_id.as_str() != "_unset_test_only",
            "remember: call() reached with sentinel user_id — use new_for_user()"
        );

        let store = self
            .store
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("remember: store not initialized".into()))?;

        let memory = Memory {
            id: Uuid::new_v4(),
            user_id: self.user_id.clone(),
            content: args.content.clone(),
            created_at: Utc::now().naive_utc(),
        };

        store
            .store_memory(&memory)
            .await
            .map_err(|e| ToolError::McpClient(format!("failed to store memory: {e}")))?;

        tracing::info!(
            user_id = %self.user_id,
            memory_id = %memory.id,
            "stored memory"
        );

        Ok(format!("Remembered: \"{}\"", args.content))
    }
}
