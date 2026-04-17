//! Agent runner -- executes Rig agent with tools, extracts answer,
//! accumulates evidence from tool calls.
//!
//! Builds a per-request Rig agent with tools from the ToolFactory,
//! runs it with multi-turn tool calling, and extracts the text answer
//! and accumulated evidence.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use agent_rs_core::errors::AgentError;
use agent_rs_stores::types::SearchResult;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use tracing::{error, info, instrument};

use crate::context::QueryContext;
use crate::prompts;
use crate::steps::tool_factory::ToolFactory;

/// Runs the Rig agent with tools and extracts the answer.
///
/// Each call to [`run`] builds a fresh agent with per-request tools
/// and evidence collectors.
pub struct AgentRunner {
    /// rig-core OpenAI client for agent creation.
    llm_client: rig::providers::openai::Client,
    /// Model identifier for the agent.
    model: String,
    /// Maximum number of tool-calling turns.
    max_turns: usize,
    /// System prompt (loaded once, shared across requests).
    system_prompt: String,
    /// Tool factory for building per-request tool sets.
    tool_factory: Arc<ToolFactory>,
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
    /// - `tool_factory` -- Shared tool factory for building per-request tools.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Orchestrator` if the OpenAI client cannot be created.
    pub fn new(
        api_key: &str,
        model: &str,
        max_turns: usize,
        system_prompt_file: &str,
        tool_factory: Arc<ToolFactory>,
    ) -> Result<Self, AgentError> {
        let system_prompt = prompts::load_system_prompt(system_prompt_file);
        let llm_client = rig::providers::openai::Client::new(api_key).map_err(|e| {
            AgentError::Orchestrator(format!("failed to create OpenAI client: {e}"))
        })?;
        Ok(Self {
            llm_client,
            model: model.to_owned(),
            max_turns,
            system_prompt,
            tool_factory,
        })
    }

    /// Execute the agent with tools and populate `ctx.answer`,
    /// `ctx.kb_evidence`, and `ctx.agent_latency_ms`.
    ///
    /// Builds a fresh Rig agent per request with tools from the factory.
    /// The `search_knowledge` tool accumulates evidence into a shared
    /// collector, which is drained into `ctx.kb_evidence` after execution.
    #[instrument(skip(self, ctx), fields(
        session_id = %ctx.session_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn run(&self, ctx: &mut QueryContext) -> Result<(), AgentError> {
        ctx.kb_evidence.clear();
        ctx.web_evidence.clear();

        let start = Instant::now();

        // Build per-request tools and evidence collector.
        let (tools, evidence_collector) = self.tool_factory.build(&ctx.user_id);

        // Build the prompt from chat history + current query
        let full_prompt = self.build_prompt(ctx);

        // Build agent with tools
        let agent = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .tools(tools)
            .build();

        let response = agent
            .prompt(&full_prompt)
            .max_turns(self.max_turns)
            .await
            .map_err(|err| {
                error!(error = %err, "agent run failed");
                AgentError::Orchestrator(format!("agent run failed: {err}"))
            })?;

        ctx.answer = response;
        info!(
            latency_ms = start.elapsed().as_millis() as u64,
            answer_len = ctx.answer.len(),
            "agent run complete"
        );

        // Drain accumulated evidence from the search_knowledge tool.
        drain_evidence(&evidence_collector, &mut ctx.kb_evidence).await;

        ctx.agent_latency_ms = start.elapsed().as_millis() as u64;
        Ok(())
    }

    /// Re-run the agent with a retry prompt (used by GroundingGate).
    ///
    /// Builds a fresh agent with tools so the retry can make new tool calls.
    #[instrument(skip(self, ctx, retry_prompt), fields(
        session_id = %ctx.session_id,
        retry = ctx.retries,
    ))]
    pub async fn rerun(
        &self,
        ctx: &mut QueryContext,
        retry_prompt: &str,
    ) -> Result<(), AgentError> {
        // NOTE: Do NOT clear ctx.kb_evidence / ctx.web_evidence here.
        // Unlike `run()`, retries must ACCUMULATE evidence from all rounds.
        // `drain_evidence` uses `append`, so new evidence merges with existing.

        // Build per-request tools and evidence collector for the retry.
        let (tools, evidence_collector) = self.tool_factory.build(&ctx.user_id);

        let agent = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .tools(tools)
            .build();

        match agent.prompt(retry_prompt).max_turns(self.max_turns).await {
            Ok(response) => {
                ctx.answer = response;
            }
            Err(err) => {
                return Err(AgentError::Orchestrator(format!(
                    "agent rerun failed: {err}"
                )));
            }
        }

        // Drain accumulated evidence from the retry.
        drain_evidence(&evidence_collector, &mut ctx.kb_evidence).await;

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

/// Drain accumulated evidence from the shared collector into the context.
///
/// Uses `tokio::sync::Mutex` which is safe to hold across `.await` points
/// and cannot be poisoned (unlike `std::sync::Mutex`).
async fn drain_evidence(collector: &Arc<Mutex<Vec<SearchResult>>>, target: &mut Vec<SearchResult>) {
    let mut collected = collector.lock().await;
    target.append(&mut collected);
}
