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
use rig::completion::{Prompt, ToolDefinition};
use rig::tool::{ToolDyn, ToolError};
use rig::wasm_compat::WasmBoxedFuture;
use serde_json::json;
use tracing::{error, info, instrument};

use crate::context::QueryContext;
use crate::coverage::ToolCallRecord;
use crate::prompts;
use crate::steps::tool_factory::ToolFactory;

// ---- Tool-call recording -----------------------------------------------

/// Shared sink that `RecordingToolDyn` writes into after each tool call.
type ToolCallSink = Arc<Mutex<Vec<ToolCallRecord>>>;

/// Object-safe wrapper around any `Box<dyn ToolDyn>` that records
/// each invocation's name and raw JSON args into a shared sink before
/// delegating to the inner tool.
///
/// Because [`ToolDyn::call`] already receives `args` as a raw JSON `String`,
/// no `Serialize` bound is required on the tool's concrete `Args` type.
struct RecordingToolDyn {
    inner: Box<dyn ToolDyn>,
    sink: ToolCallSink,
}

impl ToolDyn for RecordingToolDyn {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn definition<'a>(&'a self, prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        self.inner.definition(prompt)
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            // Capture before delegating so we record even if the tool errors.
            let record = ToolCallRecord {
                tool_name: self.inner.name(),
                args_json: args.clone(),
            };
            {
                let mut guard = self.sink.lock().await;
                guard.push(record);
            }
            self.inner.call(args).await
        })
    }
}

/// Wrap every tool in the given vector with a `RecordingToolDyn` that writes
/// into `sink`. Returns the new vec of boxed tools.
fn wrap_tools(tools: Vec<Box<dyn ToolDyn>>, sink: ToolCallSink) -> Vec<Box<dyn ToolDyn>> {
    tools
        .into_iter()
        .map(|inner| -> Box<dyn ToolDyn> {
            Box::new(RecordingToolDyn {
                inner,
                sink: Arc::clone(&sink),
            })
        })
        .collect()
}

// ---- AgentRunner -------------------------------------------------------

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
    /// Extra top-level fields merged into every Chat Completions request
    /// body (e.g. `{"reasoning_effort": "minimal"}`). `None` = no extras.
    /// Computed once at construction.
    additional_params: Option<serde_json::Value>,
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
    /// - `reasoning_effort` -- Value forwarded as the `reasoning_effort`
    ///   top-level field on every Chat Completions request. Applies to
    ///   gpt-5.x / o-series reasoning models; non-reasoning models ignore
    ///   the unknown field. Empty string disables the override entirely.
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
        reasoning_effort: &str,
    ) -> Result<Self, AgentError> {
        let system_prompt = prompts::load_system_prompt(system_prompt_file);
        let llm_client = rig::providers::openai::Client::new(api_key).map_err(|e| {
            AgentError::Orchestrator(format!("failed to create OpenAI client: {e}"))
        })?;
        let additional_params = if reasoning_effort.is_empty() {
            None
        } else {
            Some(json!({ "reasoning_effort": reasoning_effort }))
        };
        Ok(Self {
            llm_client,
            model: model.to_owned(),
            max_turns,
            system_prompt,
            tool_factory,
            additional_params,
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

        // Wrap each tool in a recording adapter that captures (tool_name,
        // args_json) into `tool_call_sink` before delegating to the real tool.
        let tool_call_sink: ToolCallSink = Arc::new(Mutex::new(Vec::new()));
        let recording_tools = wrap_tools(tools, Arc::clone(&tool_call_sink));

        // Build the prompt from chat history + current query
        let full_prompt = self.build_prompt(ctx);

        // Build agent with tools. `additional_params` forwards
        // `reasoning_effort` (and any other configured fields) to the
        // Chat Completions body via rig's `#[serde(flatten)]` passthrough.
        let builder = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .tools(recording_tools);
        let agent = match self.additional_params.as_ref() {
            Some(params) => builder.additional_params(params.clone()).build(),
            None => builder.build(),
        };

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

        // Drain coverage evidence (fetch_section_chunks + search_sections) into
        // ctx.coverage_evidence so CoverageVerifier sees what was fetched.
        drain_evidence(&coverage_collector, &mut ctx.coverage_evidence).await;

        // Drain recorded tool call args into ctx.tool_calls for use by
        // the grounding gate (Task 5.1).
        drain_tool_calls(&tool_call_sink, &mut ctx.tool_calls).await;

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

        // Build per-request tools, KB evidence collector, and coverage collector
        // for the retry. Coverage evidence accumulates across retries via append.
        let (tools, evidence_collector, coverage_collector) = self
            .tool_factory
            .build(&ctx.user_id, ctx.active_document_keys.clone());

        // Wrap tools with recording adapters for this retry round.
        let tool_call_sink: ToolCallSink = Arc::new(Mutex::new(Vec::new()));
        let recording_tools = wrap_tools(tools, Arc::clone(&tool_call_sink));

        let builder = self
            .llm_client
            .agent(&self.model)
            .preamble(&self.system_prompt)
            .tools(recording_tools);
        let agent = match self.additional_params.as_ref() {
            Some(params) => builder.additional_params(params.clone()).build(),
            None => builder.build(),
        };

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

        // Drain accumulated KB evidence from the retry.
        drain_evidence(&evidence_collector, &mut ctx.kb_evidence).await;

        // Drain coverage evidence from the retry into ctx.coverage_evidence
        // (accumulated across retries so CoverageVerifier sees all fetches).
        drain_evidence(&coverage_collector, &mut ctx.coverage_evidence).await;

        // Drain retry tool-call records (accumulated alongside evidence).
        drain_tool_calls(&tool_call_sink, &mut ctx.tool_calls).await;

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

/// Drain recorded tool-call records from the shared sink into `ctx.tool_calls`.
///
/// `append` is used so that retry rounds accumulate on top of the initial run's
/// records — the grounding gate (Task 5.1) needs the full picture across turns.
async fn drain_tool_calls(sink: &ToolCallSink, target: &mut Vec<ToolCallRecord>) {
    let mut guard = sink.lock().await;
    target.append(&mut guard);
}

// ---- Unit tests --------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::QueryContext;
    use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};
    use agent_rs_core::types::{RetrievalStrategy, UserId};
    use rig::completion::ToolDefinition;
    use serde_json::json;

    // Minimal stand-in tool that echoes its raw JSON args back.
    struct EchoTool {
        name: String,
    }

    impl ToolDyn for EchoTool {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn definition<'a>(&'a self, _prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
            let name = self.name.clone();
            Box::pin(async move {
                ToolDefinition {
                    name,
                    description: "Echo tool".into(),
                    parameters: json!({}),
                }
            })
        }

        fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
            Box::pin(async move { Ok(args) })
        }
    }

    #[tokio::test]
    async fn recording_tool_captures_args_json_into_sink() {
        let sink: ToolCallSink = Arc::new(Mutex::new(Vec::new()));
        let recording = RecordingToolDyn {
            inner: Box::new(EchoTool {
                name: "echo".to_string(),
            }),
            sink: Arc::clone(&sink),
        };

        let args_json = r#"{"query":"segmentare metode"}"#;
        let result = recording.call(args_json.to_string()).await;

        // The inner echo tool returns the args unchanged.
        assert!(result.is_ok(), "expected Ok from echo tool");
        assert_eq!(result.unwrap(), args_json);

        // The sink must contain exactly one record with the correct fields.
        let guard = sink.lock().await;
        assert_eq!(guard.len(), 1, "expected one recorded tool call");
        assert_eq!(guard[0].tool_name, "echo");
        assert_eq!(guard[0].args_json, args_json);
    }

    #[tokio::test]
    async fn recording_tool_captures_multiple_calls() {
        let sink: ToolCallSink = Arc::new(Mutex::new(Vec::new()));
        let recording = RecordingToolDyn {
            inner: Box::new(EchoTool {
                name: "search_knowledge".to_string(),
            }),
            sink: Arc::clone(&sink),
        };

        recording
            .call(r#"{"query":"first"}"#.to_string())
            .await
            .unwrap();
        recording
            .call(r#"{"query":"second"}"#.to_string())
            .await
            .unwrap();

        let guard = sink.lock().await;
        assert_eq!(guard.len(), 2);
        assert_eq!(guard[0].args_json, r#"{"query":"first"}"#);
        assert_eq!(guard[1].args_json, r#"{"query":"second"}"#);
    }

    #[tokio::test]
    async fn drain_tool_calls_appends_to_existing() {
        let sink: ToolCallSink = Arc::new(Mutex::new(vec![ToolCallRecord {
            tool_name: "search_knowledge".into(),
            args_json: r#"{"query":"new"}"#.into(),
        }]));

        let mut target = vec![ToolCallRecord {
            tool_name: "search_web".into(),
            args_json: r#"{"q":"existing"}"#.into(),
        }];

        drain_tool_calls(&sink, &mut target).await;

        assert_eq!(target.len(), 2);
        assert_eq!(target[0].tool_name, "search_web");
        assert_eq!(target[1].tool_name, "search_knowledge");

        // Sink must be empty after drain.
        let guard = sink.lock().await;
        assert!(guard.is_empty());
    }

    #[tokio::test]
    async fn wrap_tools_records_into_shared_sink() {
        let sink: ToolCallSink = Arc::new(Mutex::new(Vec::new()));
        let tools: Vec<Box<dyn ToolDyn>> = vec![
            Box::new(EchoTool {
                name: "tool_a".to_string(),
            }),
            Box::new(EchoTool {
                name: "tool_b".to_string(),
            }),
        ];
        let wrapped = wrap_tools(tools, Arc::clone(&sink));

        // Call each wrapped tool once.
        wrapped[0].call(r#"{"a":1}"#.to_string()).await.unwrap();
        wrapped[1].call(r#"{"b":2}"#.to_string()).await.unwrap();

        let guard = sink.lock().await;
        assert_eq!(guard.len(), 2);
        assert_eq!(guard[0].tool_name, "tool_a");
        assert_eq!(guard[1].tool_name, "tool_b");
    }

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
        // Create a minimal AgentRunner-like struct to call build_prompt.
        // We can't construct AgentRunner without a real API key, so we test
        // the helper function directly via the free function below.
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
