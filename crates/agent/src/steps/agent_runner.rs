//! Agent runner -- executes Rig agent with tools, extracts answer,
//! accumulates evidence from tool calls.
//!
//! Builds a per-request Rig agent with tools from the ToolFactory,
//! runs it with multi-turn tool calling via [`AgentRsHook`] for
//! observability, and extracts the text answer along with accumulated
//! evidence.

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use agent_rs_core::errors::AgentError;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use serde_json::json;
use tracing::{error, info, instrument};

use crate::agent_hook::{AgentRsHook, COMPONENT_AGENT_RUNNER, drain_tool_calls};
use crate::context::QueryContext;
use crate::prompts;
use crate::steps::tool_factory::ToolFactory;

// ---- AgentRunner -------------------------------------------------------

/// Construction parameters for [`AgentRunner::new`].
///
/// Bundled into a struct so the constructor signature stays compact as
/// new top-level options accrete (reasoning effort, seed, future
/// determinism / caching knobs). Per-field doc comments describe the
/// runtime contract.
pub struct AgentRunnerConfig<'a> {
    /// OpenAI API key forwarded to `rig::providers::openai::Client`.
    pub api_key: &'a str,
    /// Model identifier (e.g., `"gpt-4o"`, `"gpt-5.4-mini-2026-03-17"`).
    pub model: &'a str,
    /// Maximum number of tool-calling turns the rig agent is allowed
    /// before it must produce a final response.
    pub max_turns: usize,
    /// Optional path to a system-prompt override file. Empty string
    /// uses the embedded [`crate::prompts::DEFAULT_SYSTEM_PROMPT`].
    pub system_prompt_file: &'a str,
    /// Shared per-request tool factory. Each [`AgentRunner::run`]
    /// constructs a fresh tool set from this factory.
    pub tool_factory: Arc<ToolFactory>,
    /// Value forwarded as the `reasoning_effort` top-level field on
    /// every Chat Completions request. Applies to gpt-5.x / o-series
    /// reasoning models; non-reasoning models ignore the unknown field.
    /// Empty string disables the override entirely (the `seed` field is
    /// still sent).
    pub reasoning_effort: &'a str,
    /// Value forwarded as the `seed` top-level field on every Chat
    /// Completions request. Always sent (no "disable" sentinel — `0` is
    /// just another seed value to OpenAI). See
    /// [`agent_rs_core::config::DEFAULT_AGENT_SEED`] for rationale.
    pub seed: u64,
    /// Shared metrics surface routed into [`AgentRsHook`] so every
    /// turn's `record_llm_call` lands on the same histograms across
    /// pipeline steps.
    pub metrics: Arc<RetrievalMetrics>,
}

/// Runs the Rig agent with tools and extracts the answer.
///
/// Each call to [`Self::run`] builds a fresh agent with per-request
/// tools, attaches a per-request [`AgentRsHook`] for budget + tool-
/// call observability, and drains accumulated evidence into the
/// query context.
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
    /// Extra top-level fields merged into every Chat Completions request
    /// body (e.g. `{"reasoning_effort": "low", "seed": 42}`). `None` =
    /// no extras. Computed once at construction. Always populated when
    /// `seed` is set (currently unconditional — see [`Self::new`]).
    additional_params: Option<serde_json::Value>,
    /// Shared observability surface — fed into the per-request hook
    /// so [`RetrievalMetrics::record_llm_call`] fires on every model
    /// round-trip without manual call-site instrumentation.
    metrics: Arc<RetrievalMetrics>,
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
    /// - `cfg` -- bundled construction parameters. See
    ///   [`AgentRunnerConfig`] for per-field documentation.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Orchestrator` if the OpenAI client cannot be created.
    pub fn new(cfg: AgentRunnerConfig<'_>) -> Result<Self, AgentError> {
        let AgentRunnerConfig {
            api_key,
            model,
            max_turns,
            system_prompt_file,
            tool_factory,
            reasoning_effort,
            seed,
            metrics,
        } = cfg;
        let system_prompt = prompts::load_system_prompt(system_prompt_file);
        let llm_client = rig::providers::openai::Client::new(api_key).map_err(|e| {
            AgentError::Orchestrator(format!("failed to create OpenAI client: {e}"))
        })?;
        // We always send `seed` (operator-tunable knob for determinism,
        // not an opt-in feature), and conditionally include
        // `reasoning_effort` when the operator opted in. `seed = 0` is a
        // valid seed to OpenAI — never elide.
        let additional_params = if reasoning_effort.is_empty() {
            Some(json!({ "seed": seed }))
        } else {
            Some(json!({ "reasoning_effort": reasoning_effort, "seed": seed }))
        };
        Ok(Self {
            llm_client,
            model: model.to_owned(),
            max_turns,
            system_prompt,
            tool_factory,
            additional_params,
            metrics,
        })
    }

    /// Execute the agent with tools and populate `ctx.answer`,
    /// `ctx.kb_evidence`, and `ctx.agent_latency_ms`.
    ///
    /// Builds a fresh Rig agent per request with tools from the factory.
    /// The `search_knowledge` tool accumulates evidence into a shared
    /// collector, which is drained into `ctx.kb_evidence` after execution.
    #[instrument(skip(self, ctx), fields(
        trace_id = ?ctx.trace_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn run(&self, ctx: &mut QueryContext) -> Result<(), AgentError> {
        ctx.kb_evidence.clear();
        ctx.web_evidence.clear();
        ctx.tool_calls.clear();

        let start = Instant::now();

        // Build per-request tools, KB evidence collector, and coverage collector.
        let (tools, evidence_collector, coverage_collector) = self
            .tool_factory
            .build(&ctx.user_id, ctx.active_document_keys.clone());

        // Build the prompt from chat history + current query
        let full_prompt = self.build_prompt(ctx);

        // Build agent with tools. `additional_params` forwards
        // `reasoning_effort` (and any other configured fields) to the
        // Chat Completions body via rig's `#[serde(flatten)]` passthrough.
        let builder = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .tools(tools);
        let agent = match self.additional_params.as_ref() {
            Some(params) => builder.additional_params(params.clone()).build(),
            None => builder.build(),
        };

        // Per-request hook: budget recording + tool-call recording +
        // per-call observability metrics. Replaces the previous
        // `RecordingToolDyn` indirection and the (planned but never
        // wired) manual budget recording around `agent.prompt(...)`.
        let tool_call_sink = AgentRsHook::new_tool_call_sink();
        let hook = AgentRsHook::new(
            COMPONENT_AGENT_RUNNER,
            self.model.clone(),
            Arc::clone(&ctx.budget),
            Arc::clone(&tool_call_sink),
            Arc::clone(&self.metrics),
            start,
        );

        // `extended_details()` swaps the request's response type from
        // `String` to `PromptResponse { output, usage, messages }` —
        // we still hand the textual `output` back to the context, but
        // the hook handles `usage` accumulation per turn so we don't
        // also need the trailing `usage` here.
        let response = agent
            .prompt(&full_prompt)
            .max_turns(self.max_turns)
            .with_hook(hook)
            .extended_details()
            .await
            .map_err(|err| {
                error!(error = %err, "agent run failed");
                AgentError::Orchestrator(format!("agent run failed: {err}"))
            })?;

        ctx.answer = response.output;
        info!(
            latency_ms = start.elapsed().as_millis() as u64,
            answer_len = ctx.answer.len(),
            "agent run complete"
        );

        // Drain accumulated evidence from the search_knowledge tool.
        drain_evidence(&evidence_collector, &mut ctx.kb_evidence).await;

        // Drain coverage evidence (fetch_section_chunks + search_sections) into
        // ctx.coverage_evidence so CoverageVerifier sees what was fetched.
        drain_evidence(&coverage_collector, &mut ctx.coverage_evidence).await;

        // Drain recorded tool call args into ctx.tool_calls for use by
        // the grounding gate (Task 5.1). The hook captured them via
        // `on_tool_call` synchronously before each tool fired.
        drain_tool_calls(&tool_call_sink, &mut ctx.tool_calls);

        ctx.agent_latency_ms = start.elapsed().as_millis() as u64;
        Ok(())
    }

    /// Re-run the agent with a retry prompt (used by GroundingGate).
    ///
    /// Builds a fresh agent with tools so the retry can make new tool calls.
    #[instrument(skip(self, ctx, retry_prompt), fields(
        trace_id = ?ctx.trace_id,
        retry = ctx.retries,
    ))]
    pub async fn rerun(
        &self,
        ctx: &mut QueryContext,
        retry_prompt: &str,
    ) -> Result<(), AgentError> {
        // NOTE: Do NOT clear ctx.kb_evidence / ctx.web_evidence / ctx.tool_calls here.
        // Unlike `run()`, retries must ACCUMULATE evidence and tool-call records
        // from all rounds so the grounding gate sees the full picture.
        // `drain_evidence` / `drain_tool_calls` use `append`, so new data
        // merges with existing.

        let start = Instant::now();

        // Build per-request tools, KB evidence collector, and coverage collector
        // for the retry. Coverage evidence accumulates across retries via append.
        let (tools, evidence_collector, coverage_collector) = self
            .tool_factory
            .build(&ctx.user_id, ctx.active_document_keys.clone());

        let builder = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .tools(tools);
        let agent = match self.additional_params.as_ref() {
            Some(params) => builder.additional_params(params.clone()).build(),
            None => builder.build(),
        };

        let tool_call_sink = AgentRsHook::new_tool_call_sink();
        let hook = AgentRsHook::new(
            COMPONENT_AGENT_RUNNER,
            self.model.clone(),
            Arc::clone(&ctx.budget),
            Arc::clone(&tool_call_sink),
            Arc::clone(&self.metrics),
            start,
        );

        match agent
            .prompt(retry_prompt)
            .max_turns(self.max_turns)
            .with_hook(hook)
            .extended_details()
            .await
        {
            Ok(response) => {
                ctx.answer = response.output;
            }
            Err(err) => {
                return Err(AgentError::Orchestrator(format!(
                    "agent rerun failed: {err}"
                )));
            }
        }

        // Drain accumulated KB evidence from the retry.
        drain_evidence(&evidence_collector, &mut ctx.kb_evidence).await;

        // Drain coverage evidence from the retry into ctx.coverage_evidence
        // (accumulated across retries so CoverageVerifier sees all fetches).
        drain_evidence(&coverage_collector, &mut ctx.coverage_evidence).await;

        // Drain retry tool-call records (accumulated alongside evidence).
        drain_tool_calls(&tool_call_sink, &mut ctx.tool_calls);

        Ok(())
    }

    /// Build a full prompt from chat history context, injecting the
    /// `COVERAGE_PLAN` block when the query context carries an exhaustive or
    /// partial coverage plan.
    ///
    /// The coverage block is appended **after** the user query so the model
    /// sees the list of section ids it must call `search_sections` with
    /// immediately before it decides which tool to invoke.
    fn build_prompt(&self, ctx: &QueryContext) -> String {
        let coverage_suffix =
            prompts::build_coverage_plan_block(ctx.coverage_plan.as_ref()).unwrap_or_default();

        if ctx.chat_messages.is_empty() {
            return format!("{}{}", ctx.query, coverage_suffix);
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

        let base = parts.join("\n\n");
        format!("{}{}", base, coverage_suffix)
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

// ---- Unit tests --------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::QueryContext;
    use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};
    use agent_rs_core::types::{RetrievalStrategy, UserId};

    // ---- build_prompt + coverage plan injection tests -----

    fn make_ctx(query: &str) -> QueryContext {
        QueryContext::new(
            query.to_owned(),
            UserId::from_string("test-user".into()),
            Vec::new(),
            Vec::new(),
            RetrievalStrategy::GraphHybrid,
            10,
            None,
        )
    }

    #[test]
    fn build_prompt_no_coverage_plan_returns_query() {
        let ctx = make_ctx("what is segmentation?");
        let prompt = build_prompt_for_test(&ctx);
        assert_eq!(prompt, "what is segmentation?");
        assert!(!prompt.contains("COVERAGE_PLAN"));
    }

    #[test]
    fn build_prompt_with_coverage_plan_appends_block() {
        let mut ctx = make_ctx("explain all segmentation methods");
        ctx.coverage_plan = Some(CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef {
                    section_id: "sec-1".into(),
                    doc_id: "doc-a".into(),
                    heading: "5.1 Region Growing".into(),
                },
                SectionRef {
                    section_id: "sec-2".into(),
                    doc_id: "doc-a".into(),
                    heading: "5.2 Watershed".into(),
                },
            ],
        });
        let prompt = build_prompt_for_test(&ctx);
        assert!(prompt.starts_with("explain all segmentation methods"));
        assert!(prompt.contains("COVERAGE_PLAN"));
        assert!(prompt.contains("sec-1"));
        assert!(prompt.contains("5.1 Region Growing"));
        assert!(prompt.contains("sec-2"));
        assert!(prompt.contains("search_sections"));
    }

    #[test]
    fn build_prompt_mode_none_no_coverage_block() {
        let mut ctx = make_ctx("what is X?");
        ctx.coverage_plan = Some(CoveragePlan::default());
        let prompt = build_prompt_for_test(&ctx);
        assert_eq!(prompt, "what is X?");
        assert!(!prompt.contains("COVERAGE_PLAN"));
    }

    /// Free function mirroring `AgentRunner::build_prompt` logic for test
    /// access without constructing a live `AgentRunner`.
    fn build_prompt_for_test(ctx: &QueryContext) -> String {
        let coverage_suffix =
            prompts::build_coverage_plan_block(ctx.coverage_plan.as_ref()).unwrap_or_default();
        if ctx.chat_messages.is_empty() {
            return format!("{}{}", ctx.query, coverage_suffix);
        }
        let mut parts: Vec<String> = Vec::new();
        if !ctx.conversation_context.is_empty() {
            parts.push(format!(
                "[Conversation context]\n{}",
                ctx.conversation_context
            ));
        }
        if let Some(last) = ctx.chat_messages.last() {
            parts.push(last.clone());
        } else {
            parts.push(ctx.query.clone());
        }
        format!("{}{}", parts.join("\n\n"), coverage_suffix)
    }
}
