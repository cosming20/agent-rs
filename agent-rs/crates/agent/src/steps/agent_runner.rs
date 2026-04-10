//! Agent runner -- executes Rig agent, extracts answer, accumulates evidence.
//!
//! In Phase 4, the runner creates a rig-core agent with the system prompt
//! and chat history, runs it, and extracts the text answer. Tools are
//! wired in Phase 5.

use std::time::Instant;

use agent_rs_core::errors::AgentError;
use tracing::{error, info, instrument};

use crate::context::QueryContext;
use crate::prompts;

/// Runs the Rig agent and extracts the answer.
///
/// In Phase 4: creates a no-tool agent that answers from context only.
/// In Phase 5: tools are attached via the ToolFactory.
pub struct AgentRunner {
    /// rig-core OpenAI client for agent creation.
    llm_client: rig::providers::openai::Client,
    /// Model identifier for the agent.
    model: String,
    /// Maximum number of tool-calling turns (used in Phase 5).
    #[allow(dead_code)]
    max_turns: usize,
    /// System prompt (loaded once, shared across requests).
    system_prompt: String,
}

impl AgentRunner {
    /// Create a new AgentRunner.
    ///
    /// # Parameters
    ///
    /// - `api_key` -- OpenAI API key.
    /// - `model` -- Model identifier (e.g., "gpt-4o").
    /// - `max_turns` -- Maximum tool-calling turns.
    /// - `system_prompt_file` -- Path to custom system prompt file (empty = default).
    /// # Errors
    ///
    /// Returns `AgentError::Orchestrator` if the OpenAI client cannot be created.
    pub fn new(
        api_key: &str,
        model: &str,
        max_turns: usize,
        system_prompt_file: &str,
    ) -> Result<Self, AgentError> {
        let system_prompt = prompts::load_system_prompt(system_prompt_file);
        let llm_client = rig::providers::openai::Client::new(api_key)
            .map_err(|e| AgentError::Orchestrator(format!("failed to create OpenAI client: {e}")))?;
        Ok(Self {
            llm_client,
            model: model.to_owned(),
            max_turns,
            system_prompt,
        })
    }

    /// Execute the agent and populate `ctx.answer` and `ctx.agent_latency_ms`.
    ///
    /// Phase 4: No tools. The agent answers from chat history context only.
    /// Phase 5: Tools will be attached here.
    #[instrument(skip(self, ctx), fields(
        session_id = %ctx.session_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn run(&self, ctx: &mut QueryContext) -> Result<(), AgentError> {
        use rig::client::CompletionClient;
        use rig::completion::Prompt;

        ctx.kb_evidence.clear();
        ctx.web_evidence.clear();

        let start = Instant::now();

        // Build the prompt from chat history + current query
        let full_prompt = self.build_prompt(ctx);

        let agent = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .build();

        match agent.prompt(&full_prompt).await {
            Ok(response) => {
                ctx.answer = response;
                info!(
                    latency_ms = start.elapsed().as_millis() as u64,
                    answer_len = ctx.answer.len(),
                    "agent run complete"
                );
            }
            Err(err) => {
                error!(error = %err, "agent run failed");
                ctx.answer =
                    "I was unable to generate an answer due to a processing error.".into();
            }
        }

        ctx.agent_latency_ms = start.elapsed().as_millis() as u64;
        Ok(())
    }

    /// Re-run the agent with retry messages (used by GroundingGate).
    #[instrument(skip(self, ctx, retry_prompt), fields(
        session_id = %ctx.session_id,
        retry = ctx.retries,
    ))]
    pub async fn rerun(
        &self,
        ctx: &mut QueryContext,
        retry_prompt: &str,
    ) -> Result<(), AgentError> {
        use rig::client::CompletionClient;
        use rig::completion::Prompt;

        ctx.kb_evidence.clear();
        ctx.web_evidence.clear();

        let agent = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .build();

        match agent.prompt(retry_prompt).await {
            Ok(response) => {
                ctx.answer = response;
            }
            Err(err) => {
                return Err(AgentError::Orchestrator(format!(
                    "agent rerun failed: {err}"
                )));
            }
        }

        Ok(())
    }

    /// Build a full prompt from chat history context.
    fn build_prompt(&self, ctx: &QueryContext) -> String {
        if ctx.chat_messages.is_empty() {
            return ctx.query.clone();
        }

        let mut parts: Vec<String> = Vec::new();

        if !ctx.conversation_context.is_empty() {
            parts.push(format!(
                "[Conversation context]\n{}",
                ctx.conversation_context
            ));
        }

        // Add the current query (last item in chat_messages already has the hint)
        if let Some(last) = ctx.chat_messages.last() {
            parts.push(last.clone());
        } else {
            parts.push(ctx.query.clone());
        }

        parts.join("\n\n")
    }
}
