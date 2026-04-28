//! `PromptHook` impl that wires agent-rs cross-cutting concerns into
//! rig's per-request lifecycle.
//!
//! [`AgentRsHook`] bundles three things that previously had to be
//! plumbed through bespoke wrappers around every LLM call:
//!
//! 1. **Token budget recording.** [`PromptHook::on_completion_response`]
//!    fires after every model round-trip with the exact `Usage` field
//!    rig extracted from the provider response. We map it to
//!    [`TokenUsage`] (including `cached_input_tokens`) and call
//!    [`RequestBudget::record_call`] — which is the one place query-
//!    scoped token totals accumulate across pipeline steps.
//! 2. **Per-call observability metrics.** The same callback feeds
//!    [`RetrievalMetrics::record_llm_call`] so the per-call histograms
//!    (latency, prompt/completion tokens, cost-per-call) stay accurate
//!    without per-call manual `info!` boilerplate at the call site.
//!    Latency is approximated from the [`AgentRsHook`]'s own clock
//!    because the hook does not have access to rig's internal span
//!    timing — see [`Self::with_started_at`] for how the runner threads
//!    the start-of-call instant in.
//! 3. **Tool-call recording.** [`PromptHook::on_tool_call`] receives
//!    the tool name and raw JSON args _before_ the tool executes, so
//!    we can capture them into [`crate::context::QueryContext::tool_calls`]
//!    without wrapping every `Box<dyn ToolDyn>` in a recorder adapter.
//!    This deletes the previous `RecordingToolDyn` indirection
//!    entirely.
//!
//! The hook is designed to be cheap to clone — every interior field is
//! either an [`Arc`] or a [`String`] — because rig clones the hook
//! into the per-call closure. Push semantics into shared state are
//! safe across await points: the [`std::sync::Mutex`] guard is taken
//! and dropped synchronously around a single `Vec::push`, never held
//! across an `.await`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use rig::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig::completion::{CompletionModel, CompletionResponse, Usage as RigUsage};
use rig::message::Message;

use agent_rs_observability::budget::{RequestBudget, TokenUsage};
use agent_rs_observability::metrics::RetrievalMetrics;

use crate::coverage::ToolCallRecord;

/// Map a [`rig::completion::Usage`] into the local [`TokenUsage`].
///
/// rig surfaces `input_tokens` / `output_tokens` / `cached_input_tokens`
/// as `u64`. Our local [`TokenUsage`] uses `u32` because per-call
/// values never approach `u32::MAX` (a single OpenAI request is
/// hard-capped to the model's context window — caps live in the
/// hundreds of thousands of tokens, well below 4 billion).
/// [`u32_saturate`] defensively pins each field to `u32::MAX` if a
/// provider ever returned an absurd number, instead of panicking.
///
/// Defined as a free function rather than a `From` impl because
/// Rust's orphan rule forbids `impl From<rig::Usage> for
/// agent_rs_observability::TokenUsage` from this crate (neither type
/// is local). Adding rig as a dep on the observability crate would
/// pull a heavy dependency tree there, which the Tier-1 spec rules
/// out.
pub fn token_usage_from_rig(u: RigUsage) -> TokenUsage {
    TokenUsage::with_cache(
        u32_saturate(u.input_tokens),
        u32_saturate(u.output_tokens),
        u32_saturate(u.cached_input_tokens),
    )
}

/// Saturate `u64 -> u32` so a malformed provider response cannot
/// crash the recorder. Values realistically stay well under 1M per
/// call; the saturation only fires on garbage input.
fn u32_saturate(value: u64) -> u32 {
    if value > u64::from(u32::MAX) {
        u32::MAX
    } else {
        value as u32
    }
}

/// LLM component label used for [`RetrievalMetrics::record_llm_call`]
/// when the agent runner dispatches a `prompt(...)` call. Stable
/// string so dashboards can filter by `llm.component=agent_runner`.
const LLM_COMPONENT_AGENT_RUNNER: &str = "agent_runner";

/// LLM component label used for the verifier's `prompt_typed` call.
const LLM_COMPONENT_VERIFIER: &str = "verifier";

/// Shared sink that the hook writes into after each tool call. The
/// agent runner drains this sink into
/// [`crate::context::QueryContext::tool_calls`] once the rig agent
/// returns. `std::sync::Mutex` is intentional — we never hold the
/// guard across an `.await`, so the simpler poison semantics of the
/// std mutex give us free early-failure detection.
pub type ToolCallSink = Arc<Mutex<Vec<ToolCallRecord>>>;

/// Hook bundling budget recording, tool-call recording, and per-call
/// observability metrics for one LLM request.
///
/// `Clone` is cheap (every field is `Arc` or `String`). One hook
/// instance is constructed per `agent.prompt(...)` call site; rig
/// clones it internally as needed.
#[derive(Clone)]
pub struct AgentRsHook {
    /// Component label routed into
    /// [`RetrievalMetrics::record_llm_call`]. Constant per call site.
    component: &'static str,
    /// Model identifier — drives both the metric label and the
    /// pricing lookup inside [`RequestBudget::record_call`].
    model: String,
    /// Per-request token + cost aggregator (created once at the entry
    /// of `process_query`, shared across every pipeline step that
    /// dispatches LLM calls).
    budget: Arc<RequestBudget>,
    /// Sink the hook pushes [`ToolCallRecord`]s into. The agent
    /// runner drains it after the rig agent returns.
    tool_call_sink: ToolCallSink,
    /// Per-call histogram surface (rate / errors / duration / cost).
    metrics: Arc<RetrievalMetrics>,
    /// Start-of-call wall clock, captured by the runner just before
    /// it awaits `agent.prompt(...)`. Used to compute the
    /// `latency_ms` argument to [`RetrievalMetrics::record_llm_call`]
    /// inside [`PromptHook::on_completion_response`]. We measure from
    /// the runner's perspective rather than from inside the hook
    /// because the hook fires once per turn — using a per-hook clock
    /// would only measure the model round-trip on the most recent
    /// turn, while `started_at` covers the whole request.
    started_at: Instant,
}

impl AgentRsHook {
    /// Construct a hook for a single `agent.prompt(...)` invocation.
    ///
    /// `started_at` should be the `Instant` captured immediately
    /// before the runner dispatches the prompt. See the doc on
    /// [`Self::started_at`] for why latency is measured from the
    /// caller's clock.
    pub fn new(
        component: &'static str,
        model: String,
        budget: Arc<RequestBudget>,
        tool_call_sink: ToolCallSink,
        metrics: Arc<RetrievalMetrics>,
        started_at: Instant,
    ) -> Self {
        Self {
            component,
            model,
            budget,
            tool_call_sink,
            metrics,
            started_at,
        }
    }

    /// Construct an empty [`ToolCallSink`] suitable for passing into
    /// [`Self::new`]. The runner calls this once per LLM dispatch.
    pub fn new_tool_call_sink() -> ToolCallSink {
        Arc::new(Mutex::new(Vec::new()))
    }
}

/// Component label suitable for the agent runner's `with_hook(...)`
/// call. Re-exported so call sites pass a typed constant rather than
/// a stringly-typed `"agent_runner"` argument.
pub const COMPONENT_AGENT_RUNNER: &str = LLM_COMPONENT_AGENT_RUNNER;

/// Component label suitable for the verifier's `with_hook(...)` call.
pub const COMPONENT_VERIFIER: &str = LLM_COMPONENT_VERIFIER;

impl<M> PromptHook<M> for AgentRsHook
where
    M: CompletionModel,
{
    /// Record token usage + per-call metrics after every model
    /// round-trip. Fires once per turn in a multi-turn (tool-calling)
    /// agent loop, so a request with 3 tool turns produces 3 calls
    /// here — exactly the granularity dashboards want.
    fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> impl std::future::Future<Output = HookAction> + Send {
        let component = self.component;
        let model = self.model.clone();
        let budget = Arc::clone(&self.budget);
        let metrics = Arc::clone(&self.metrics);
        let usage = token_usage_from_rig(response.usage);
        let prompt_tokens = u64::from(usage.prompt_tokens);
        let completion_tokens = u64::from(usage.completion_tokens);
        let started_at = self.started_at;

        async move {
            // Budget aggregation (accumulates across every turn of
            // every LLM-dispatching pipeline step).
            budget.record_call(&model, usage);
            // Per-call histograms. Latency uses the runner's
            // `started_at` clock — this slightly over-counts the
            // first turn for multi-turn tool loops, which is
            // acceptable: the histograms are advisory and the
            // accurate per-call timing lives in rig's own span data.
            let latency_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            metrics.record_llm_call(
                component,
                &model,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            );
            HookAction::cont()
        }
    }

    /// Record the tool name + raw JSON args into the shared sink so
    /// the runner can drain it into
    /// [`crate::context::QueryContext::tool_calls`]. Replaces the
    /// previous `RecordingToolDyn` wrapper.
    fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> impl std::future::Future<Output = ToolCallHookAction> + Send {
        let record = ToolCallRecord {
            tool_name: tool_name.to_owned(),
            args_json: args.to_owned(),
        };
        let sink = Arc::clone(&self.tool_call_sink);
        async move {
            // Block scope: `std::sync::Mutex` guard never crosses an
            // await point.
            if let Ok(mut guard) = sink.lock() {
                guard.push(record);
            } else {
                // Mutex was poisoned by a panic in another thread.
                // Tool-call records are diagnostic; recovering by
                // skipping the push is preferable to crashing the
                // request.
                tracing::warn!(
                    tool_name = %tool_name,
                    "AgentRsHook: tool_call_sink mutex poisoned; skipping record"
                );
            }
            ToolCallHookAction::cont()
        }
    }
}

/// Drain recorded tool-call records from the shared sink.
///
/// `append` is used so retry rounds accumulate on top of the initial
/// run's records — the grounding gate (Task 5.1) needs the full
/// picture across turns.
pub fn drain_tool_calls(sink: &ToolCallSink, target: &mut Vec<ToolCallRecord>) {
    if let Ok(mut guard) = sink.lock() {
        target.append(&mut guard);
    } else {
        tracing::warn!("AgentRsHook: tool_call_sink mutex poisoned at drain time");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_from_rig_maps_all_three_fields() {
        let rig_usage = RigUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cached_input_tokens: 30,
            cache_creation_input_tokens: 0,
        };
        let mapped = token_usage_from_rig(rig_usage);
        assert_eq!(mapped.prompt_tokens, 100);
        assert_eq!(mapped.completion_tokens, 50);
        assert_eq!(mapped.cached_input_tokens, 30);
    }

    #[test]
    fn token_usage_from_rig_saturates_oversized_u64() {
        // Defensive: provider returning u64::MAX must not panic on
        // u32 conversion.
        let rig_usage = RigUsage {
            input_tokens: u64::MAX,
            output_tokens: u64::MAX,
            total_tokens: u64::MAX,
            cached_input_tokens: u64::MAX,
            cache_creation_input_tokens: 0,
        };
        let mapped = token_usage_from_rig(rig_usage);
        assert_eq!(mapped.prompt_tokens, u32::MAX);
        assert_eq!(mapped.completion_tokens, u32::MAX);
        assert_eq!(mapped.cached_input_tokens, u32::MAX);
    }

    #[test]
    fn token_usage_from_rig_zero_yields_default() {
        let rig_usage = RigUsage::new();
        let mapped = token_usage_from_rig(rig_usage);
        assert_eq!(mapped, TokenUsage::default());
    }

    #[tokio::test]
    async fn drains_tool_calls_into_target_vec() {
        // Smoke test on the sink-and-drain plumbing alone. Driving
        // `PromptHook::on_tool_call` end-to-end requires a real
        // `CompletionModel` (rig's trait carries seven supertraits
        // and seven associated types) — we cover the actual hook
        // path via the integration call sites in the runner +
        // verifier instead. Here we exercise the sink contract that
        // both the hook callback and the runner depend on.
        let sink = AgentRsHook::new_tool_call_sink();
        {
            let mut guard = sink.lock().expect("fresh mutex must not be poisoned");
            guard.push(ToolCallRecord {
                tool_name: "search_knowledge".into(),
                args_json: r#"{"query":"alpha"}"#.into(),
            });
            guard.push(ToolCallRecord {
                tool_name: "outline_corpus".into(),
                args_json: "{}".into(),
            });
        }

        let mut target: Vec<ToolCallRecord> = Vec::new();
        drain_tool_calls(&sink, &mut target);
        assert_eq!(target.len(), 2);
        assert_eq!(target[0].tool_name, "search_knowledge");
        assert_eq!(target[1].tool_name, "outline_corpus");
        assert!(
            sink.lock().expect("post-drain lock").is_empty(),
            "sink must be empty after drain"
        );
    }

    #[test]
    fn budget_records_via_token_usage_from_rig_usage() {
        // End-to-end coverage of the budget-recording path used by
        // `on_completion_response` — without instantiating the rig
        // `CompletionResponse<M::Response>`, which would require a
        // fully-fledged provider model. We assert that the
        // `token_usage_from_rig` conversion plus
        // `RequestBudget::record_call` produce a non-zero
        // cached_input_tokens snapshot.
        let budget = RequestBudget::new();
        let rig_usage = RigUsage {
            input_tokens: 1000,
            output_tokens: 500,
            total_tokens: 1500,
            cached_input_tokens: 600,
            cache_creation_input_tokens: 0,
        };
        budget.record_call("gpt-4o", token_usage_from_rig(rig_usage));

        let snap = budget.snapshot();
        assert_eq!(snap.call_count, 1);
        assert_eq!(snap.prompt_tokens, 1000);
        assert_eq!(snap.completion_tokens, 500);
        assert_eq!(snap.cached_input_tokens, 600);
        // Cost should reflect the cached discount (lower than the
        // no-cache baseline — see budget unit tests for exact math).
        let baseline =
            agent_rs_observability::budget::estimate_cost_usd("gpt-4o", TokenUsage::new(1000, 500));
        assert!(
            snap.cost_usd < baseline,
            "cached_input_tokens must reduce cost (snap={}, baseline={})",
            snap.cost_usd,
            baseline
        );
    }
}
