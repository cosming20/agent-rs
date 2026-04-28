//! Streaming synthesis agent — the user-facing LLM step.
//!
//! After the agent has done its multi-turn tool calling and the
//! evidence pack has settled, [`SynthesisAgent`] runs ONE final LLM
//! call whose sole job is to produce a polished, grounded,
//! lang-coherent final answer for the user. It streams text deltas.
//!
//! # Why a separate agent
//!
//! - The exploratory agent ([`crate::steps::agent_runner::AgentRunner`])
//!   is optimized for tool use, not prose quality. Its drafts can be
//!   fragmented or include meta-commentary about the search process.
//! - Streaming requires tool-call-free generation. [`SynthesisAgent`]
//!   has no tools; the multi-turn budget is `0`.
//! - A dedicated synth prompt enforces grounding implicitly: "only
//!   claim what's in the evidence pack; cite via `[cite:<chunk_id>]`".
//!   This replaces the deleted LLM verifier with a pre-emptive
//!   constraint instead of a post-hoc check.

use std::sync::Arc;
use std::time::Instant;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{LangCode, truncate_to_char_boundary};
use agent_rs_observability::budget::RequestBudget;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use futures::Stream;
use rig::client::CompletionClient;
use rig::completion::Usage as RigUsage;
use rig::message::Text;
use rig::providers::openai::Client;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use serde_json::json;
use tracing::{debug, error, info, instrument, warn};

use crate::agent_hook::{AgentRsHook, ToolCallSink, token_usage_from_rig};
use crate::coverage::CoverageReport;

/// Component label routed into [`RetrievalMetrics::record_llm_call`]
/// when the synthesis stage dispatches its dedicated LLM call.
/// Stable string so dashboards can filter by
/// `llm.component=synthesis_agent`.
pub const COMPONENT_SYNTHESIS_AGENT: &str = "synthesis_agent";

/// Maximum characters per evidence-block snippet emitted into the
/// synthesis prompt body. Protects the prompt from context blow-up
/// when the evidence pack carries oversize chunks (tables, code
/// blocks). Chosen to match the synthesis config's
/// `max_snippet_length` default `5000` — the prompt-side cap is
/// independent of the citation-side cap so neither truncation
/// silently amplifies the other.
const PROMPT_EVIDENCE_SNIPPET_CHARS: usize = 5_000;

/// Maximum characters of the agent draft echoed into the synthesis
/// prompt. The draft is reference-only (the synth model verifies
/// every claim against the evidence pack itself), so an overlong
/// draft pollutes the prompt without buying anything. Generous
/// enough to keep the typical 800-1500 token agent answer intact;
/// truncates only on degenerate runs.
const PROMPT_DRAFT_CHARS: usize = 6_000;

/// Inputs to [`SynthesisAgent::synthesize_stream`].
///
/// Borrowed (rather than owned) so the orchestrator does not clone
/// the evidence pack on the synthesis hand-off — the stream future
/// holds the borrow until completion.
#[derive(Debug)]
pub struct SynthesisInputs<'a> {
    /// Original user query, mirrored back into the synthesis prompt.
    pub query: &'a str,
    /// The agent runner's draft answer. Reference-only: the synth
    /// prompt instructs the model to verify every claim against the
    /// evidence pack before including it.
    pub agent_draft: &'a str,
    /// Fused evidence pack (KB ∪ web, after RRF).
    pub evidence: &'a [SearchResult],
    /// Detected query language. Threaded into the prompt so the synth
    /// stage cannot accidentally translate / drift across scripts.
    pub detected_lang: LangCode,
    /// Optional coverage report — when present with non-zero
    /// `expected`, the synthesis prompt gains an "exhaustive: cover
    /// every section" directive.
    pub coverage_report: Option<&'a CoverageReport>,
}

/// Event emitted by [`SynthesisAgent::synthesize_stream`].
///
/// The stream completes after [`Self::Done`] or [`Self::Error`];
/// callers must not consume any further events past those terminal
/// variants.
#[derive(Debug, Clone)]
pub enum SynthEvent {
    /// Streaming text delta — append directly to the user-visible
    /// buffer.
    TextDelta(String),
    /// Terminal success event with the full aggregated text and the
    /// final token usage (already recorded against the per-request
    /// [`RequestBudget`]).
    Done {
        /// All text deltas concatenated.
        full_text: String,
        /// Aggregated token usage from the synthesis call.
        usage: RigUsage,
    },
    /// Terminal failure event. Caller should not consume any further
    /// events.
    Error(String),
}

/// Configuration knobs for [`SynthesisAgent::new`].
pub struct SynthesisAgentConfig<'a> {
    /// OpenAI API key forwarded to `rig::providers::openai::Client`.
    pub api_key: &'a str,
    /// Model identifier (e.g. `"gpt-4o"`,
    /// `"gpt-5.4-mini-2026-03-17"`). Independent from the agent
    /// runner's model so operators can route the user-facing answer
    /// through a different (typically smaller, faster) model.
    pub model: &'a str,
    /// Determinism seed forwarded to OpenAI Chat Completions. See
    /// [`agent_rs_core::config::DEFAULT_AGENT_SEED`] for rationale.
    pub seed: u64,
    /// Shared metrics handle. Always wired in production so per-call
    /// histograms accumulate alongside the agent-runner's metrics.
    pub metrics: Arc<RetrievalMetrics>,
}

/// Streaming synthesis agent: runs ONE tool-free LLM call that
/// produces the user-facing answer.
///
/// Constructed once at orchestrator startup; cheap to clone (every
/// field is `Arc`/`String`). One [`SynthesisAgent::synthesize_stream`]
/// call per `process_query` invocation.
pub struct SynthesisAgent {
    /// rig-core OpenAI client, shared across requests.
    llm_client: Client,
    /// Model identifier.
    model: String,
    /// Pre-built `additional_params` blob (carries `seed` always; no
    /// `reasoning_effort` because the synthesis stage is intentionally
    /// not a reasoning step — it composes already-retrieved evidence).
    additional_params: serde_json::Value,
    /// Shared metrics handle, fed into the per-call [`AgentRsHook`].
    metrics: Arc<RetrievalMetrics>,
}

impl SynthesisAgent {
    /// Construct a synthesis agent.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Orchestrator`] if the OpenAI client
    /// cannot be created (typically a malformed API key).
    pub fn new(cfg: SynthesisAgentConfig<'_>) -> Result<Self, AgentError> {
        let SynthesisAgentConfig {
            api_key,
            model,
            seed,
            metrics,
        } = cfg;
        let llm_client = Client::new(api_key).map_err(|e| {
            AgentError::Orchestrator(format!("failed to create OpenAI client for synthesis: {e}"))
        })?;
        Ok(Self {
            llm_client,
            model: model.to_owned(),
            additional_params: json!({ "seed": seed }),
            metrics,
        })
    }

    /// Build the synthesis prompt body.
    ///
    /// Lean, language-mirroring, citation-strict. Hard rules first;
    /// inputs follow. Coverage block appended only when the report
    /// flags exhaustive intent (i.e. the planner produced a non-empty
    /// expected-section list).
    pub(crate) fn build_prompt(&self, inputs: &SynthesisInputs<'_>) -> String {
        let evidence_blocks = render_evidence_blocks(inputs.evidence);
        let agent_draft = truncate_to_char_boundary(inputs.agent_draft, PROMPT_DRAFT_CHARS);
        let coverage_block = render_coverage_block(inputs.coverage_report);

        format!(
            "You are the final synthesis stage of a retrieval-augmented agent.\n\
             Your job: produce ONE polished, accurate, complete answer for the user.\n\
             \n\
             HARD RULES:\n\
             1. Mirror the user's language exactly (script, diacritics, register). Never translate.\n\
             2. Make every factual claim traceable to the EVIDENCE pack below. If a claim\n   \
                isn't supported, do not make it.\n\
             3. Cite using `[cite:<chunk_id>]` markers from the evidence pack — NEVER invent\n   \
                chunk_ids. Place markers immediately after the claims they support.\n\
             4. If the evidence does not answer the question, say so directly in the user's\n   \
                language. Do not pad. Do not speculate.\n\
             5. No meta-commentary about your process (\"I searched...\", \"Based on the retrieved\n   \
                chunks...\"). Speak as if you authored the answer from knowledge.\n\
             \n\
             FORMAT:\n\
             - Concise prose. Use short paragraphs and lists when natural.\n\
             - Numbers / formulas: copy verbatim, never reformulate.\n\
             \n\
             INPUTS:\n\
             - USER QUERY: {query}\n\
             - DETECTED LANGUAGE: {lang}\n\
             - EVIDENCE (each block prefixed with `[cite:<id>]`):\n{evidence}\n\
             - AGENT DRAFT (reference only — verify each claim against EVIDENCE before\n  \
               including it):\n{draft}\n\
             {coverage}\n\
             Now produce the final answer.\n",
            query = inputs.query,
            lang = inputs.detected_lang,
            evidence = evidence_blocks,
            draft = agent_draft,
            coverage = coverage_block,
        )
    }

    /// Run synthesis as an async stream of [`SynthEvent`]s.
    ///
    /// The stream yields [`SynthEvent::TextDelta`] for each text
    /// chunk emitted by the model, then exactly one terminal
    /// [`SynthEvent::Done`] (or [`SynthEvent::Error`]) before
    /// completing. Callers drive the stream and accumulate `full_text`
    /// from the deltas; the final `Done` event echoes the full text
    /// for callers that opted out of incremental rendering.
    ///
    /// The per-call [`AgentRsHook`] handles budget recording and
    /// per-call metrics. Some streaming providers do not invoke the
    /// completion hook for streamed responses; this method falls
    /// back to recording the `FinalResponse.aggregated_usage` against
    /// the request budget so the per-request totals stay accurate
    /// regardless of provider behaviour.
    ///
    /// # Cancellation
    ///
    /// Dropping the returned stream cancels the in-flight LLM
    /// request at the next yield point — rig's
    /// `StreamingCompletionResponse` is built on `Abortable`. Budget
    /// under-counting on cancelled calls is acceptable (the request
    /// never returned an answer).
    #[instrument(skip(self, inputs, budget), fields(
        evidence_count = inputs.evidence.len(),
        lang = %inputs.detected_lang,
    ))]
    pub fn synthesize_stream<'a>(
        &'a self,
        inputs: SynthesisInputs<'a>,
        budget: Arc<RequestBudget>,
    ) -> impl Stream<Item = SynthEvent> + Send + 'a {
        let prompt = self.build_prompt(&inputs);
        let model = self.model.clone();
        let llm_client = self.llm_client.clone();
        let additional_params = self.additional_params.clone();
        let metrics = Arc::clone(&self.metrics);
        let started_at = Instant::now();

        async_stream::stream! {
            let agent = llm_client
                .agent(&model)
                .additional_params(additional_params)
                .build();

            // The hook needs a tool-call sink even though synthesis
            // runs without tools — `AgentRsHook::on_tool_call` never
            // fires here. Empty sink is the cheapest no-op.
            let tool_call_sink: ToolCallSink = AgentRsHook::new_tool_call_sink();
            let hook = AgentRsHook::new(
                COMPONENT_SYNTHESIS_AGENT,
                model.clone(),
                Arc::clone(&budget),
                tool_call_sink,
                Arc::clone(&metrics),
                started_at,
            );

            let mut stream = agent
                .stream_prompt(prompt)
                .with_hook(hook)
                .multi_turn(0)
                .await;

            let mut full_text = String::new();
            let mut final_usage: Option<RigUsage> = None;
            let mut errored = false;

            use futures::StreamExt;
            use rig::agent::MultiTurnStreamItem;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::Text(Text { text }),
                    )) => {
                        full_text.push_str(&text);
                        yield SynthEvent::TextDelta(text);
                    }
                    Ok(MultiTurnStreamItem::FinalResponse(fin)) => {
                        // Aggregated usage from FinalResponse — the
                        // streaming hook may not fire on usage chunks
                        // for some providers, so this is the
                        // authoritative snapshot for budget recording.
                        final_usage = Some(fin.usage());
                        // Some non-streaming providers populate the
                        // FinalResponse text without ever yielding
                        // text deltas. Fall back to it so the caller
                        // still receives the full answer.
                        if full_text.is_empty() && !fin.response().is_empty() {
                            full_text.push_str(fin.response());
                        }
                    }
                    Ok(_) => {
                        // Reasoning blocks, tool-call deltas, etc. —
                        // synthesis runs without tools, so any tool
                        // call here is a provider artefact we ignore.
                    }
                    Err(e) => {
                        error!(error = %e, "synthesis stream error");
                        errored = true;
                        yield SynthEvent::Error(format!("synthesis stream error: {e}"));
                        break;
                    }
                }
            }

            if errored {
                // Already yielded an Error; do not also yield Done.
                return;
            }

            // Defence-in-depth budget recording: if rig's hook did
            // not fire on usage (streaming providers sometimes skip
            // it), record from the aggregated FinalResponse usage so
            // the request budget reflects the synthesis cost.
            let usage = final_usage.unwrap_or_default();
            if usage.input_tokens > 0 || usage.output_tokens > 0 {
                budget.record_call(&model, token_usage_from_rig(usage));
                metrics.record_llm_call(
                    COMPONENT_SYNTHESIS_AGENT,
                    &model,
                    usage.input_tokens,
                    usage.output_tokens,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }

            info!(
                latency_ms = started_at.elapsed().as_millis() as u64,
                output_chars = full_text.len(),
                "synthesis stream complete"
            );

            yield SynthEvent::Done { full_text, usage };
        }
    }

    /// Convenience non-streaming wrapper: drain the stream into the
    /// final aggregated text + token usage.
    ///
    /// Used by the unary path so callers that don't want streaming
    /// keep the same `(text, usage)` shape.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Synthesis`] when the underlying
    /// stream yielded a [`SynthEvent::Error`] before completing.
    pub async fn synthesize(
        &self,
        inputs: SynthesisInputs<'_>,
        budget: Arc<RequestBudget>,
    ) -> Result<(String, RigUsage), AgentError> {
        use futures::StreamExt;
        let mut stream = Box::pin(self.synthesize_stream(inputs, budget));
        let mut accumulated = String::new();
        let mut final_usage = RigUsage::new();
        while let Some(ev) = stream.next().await {
            match ev {
                SynthEvent::TextDelta(delta) => {
                    accumulated.push_str(&delta);
                }
                SynthEvent::Done { full_text, usage } => {
                    // Prefer the aggregated full_text over the
                    // accumulator (they should be identical, but
                    // the Done event is the authoritative final).
                    if !full_text.is_empty() {
                        accumulated = full_text;
                    }
                    final_usage = usage;
                }
                SynthEvent::Error(msg) => {
                    return Err(AgentError::Synthesis(msg));
                }
            }
        }
        debug!(
            output_chars = accumulated.len(),
            "non-streaming synthesize() drain complete"
        );
        Ok((accumulated, final_usage))
    }
}

// ---- prompt-rendering helpers --------------------------------------------

/// Render the evidence pack as a sequence of `[cite:<id>] {snippet}`
/// blocks suitable for the synthesis prompt body.
///
/// Each chunk's `chunk_id` (preferred) or fallback `id` is emitted as
/// the cite marker, followed by a length-bounded snippet of the
/// chunk text. `section_path` is appended on a separate line when
/// present so the model can ground formatting hints (e.g. "section
/// 5.1") without re-deriving them.
fn render_evidence_blocks(evidence: &[SearchResult]) -> String {
    if evidence.is_empty() {
        return "  (no evidence retrieved)\n".to_owned();
    }
    let mut out = String::new();
    for ev in evidence {
        let chunk_id = ev
            .payload
            .get("chunk_id")
            .and_then(|v| v.as_str())
            .unwrap_or(ev.id.as_str());
        let chunk_text = ev
            .payload
            .get("chunk_text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let snippet = if chunk_text.chars().count() > PROMPT_EVIDENCE_SNIPPET_CHARS {
            format!(
                "{}...",
                truncate_to_char_boundary(chunk_text, PROMPT_EVIDENCE_SNIPPET_CHARS)
            )
        } else {
            chunk_text.to_owned()
        };
        out.push_str(&format!("\n  [cite:{chunk_id}] {snippet}\n"));
        if let Some(section_path) = ev.payload.get("section_path").and_then(|v| v.as_str())
            && !section_path.is_empty()
        {
            out.push_str(&format!("    (section: {section_path})\n"));
        }
    }
    out
}

/// Render the coverage block appended to the synthesis prompt when
/// the report flags exhaustive intent.
///
/// The report itself does not carry a mode enum; it is only
/// populated when `coverage_plan` was set, which the planner does
/// for exhaustive / partial flows. We treat any non-zero `expected`
/// as a signal to inject the exhaustive directive — non-exhaustive
/// flows always have `coverage_report = None` upstream.
fn render_coverage_block(report: Option<&CoverageReport>) -> String {
    let Some(report) = report else {
        return String::new();
    };
    if report.expected == 0 {
        return String::new();
    }
    let missing_hint = if report.missing_section_ids.is_empty() {
        String::new()
    } else {
        format!(
            "\n  (Coverage planner notes {} missing section(s); call them out explicitly when \
             the evidence pack does not cover them.)",
            report.missing_section_ids.len()
        )
    };
    warn!(
        expected = report.expected,
        fetched = report.fetched,
        missing = report.missing_section_ids.len(),
        "synthesis prompt: appending exhaustive-coverage directive"
    );
    format!(
        "\nCOVERAGE NOTE: This is an exhaustive query — the user asked for a complete\n\
         enumeration. Cover every section in the evidence pack; do not skip.{missing_hint}\n",
    )
}

#[cfg(test)]
mod tests_build_prompt {
    use super::*;
    use agent_rs_core::types::Score;
    use agent_rs_stores::types::SearchResult;

    fn ev(id: &str, text: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score: Score::new(0.5),
            payload: json!({
                "chunk_id": id,
                "chunk_text": text,
                "section_path": "5.1 Test",
            }),
        }
    }

    fn agent_for_tests() -> SynthesisAgent {
        SynthesisAgent {
            llm_client: Client::new("dummy-key").expect("rig client must construct"),
            model: "gpt-4o".to_owned(),
            additional_params: json!({ "seed": 42 }),
            metrics: Arc::new(RetrievalMetrics::new("synthesis-test")),
        }
    }

    #[test]
    fn build_prompt_contains_query_and_evidence_blocks() {
        let agent = agent_for_tests();
        let evidence = vec![ev("c-1", "Otsu separates classes.")];
        let inputs = SynthesisInputs {
            query: "What is Otsu?",
            agent_draft: "Otsu is a thresholding method.",
            evidence: &evidence,
            detected_lang: LangCode::En,
            coverage_report: None,
        };
        let prompt = agent.build_prompt(&inputs);
        assert!(prompt.contains("What is Otsu?"));
        assert!(prompt.contains("[cite:c-1]"));
        assert!(prompt.contains("Otsu separates classes."));
        assert!(prompt.contains("DETECTED LANGUAGE: en"));
        // Coverage block must NOT appear when the report is None.
        assert!(
            !prompt.contains("COVERAGE NOTE"),
            "coverage block should be absent in non-exhaustive flows"
        );
    }

    #[test]
    fn build_prompt_appends_coverage_directive_for_exhaustive_report() {
        let agent = agent_for_tests();
        let evidence = vec![ev("c-1", "Region growing iterates.")];
        let report = CoverageReport {
            expected: 4,
            fetched: 3,
            missing_section_ids: vec!["s-9".into()],
        };
        let inputs = SynthesisInputs {
            query: "Enumerate all segmentation methods",
            agent_draft: "Region growing, watershed, ...",
            evidence: &evidence,
            detected_lang: LangCode::En,
            coverage_report: Some(&report),
        };
        let prompt = agent.build_prompt(&inputs);
        assert!(
            prompt.contains("COVERAGE NOTE"),
            "exhaustive report must inject COVERAGE NOTE block"
        );
        assert!(prompt.contains("missing section"));
    }

    #[test]
    fn build_prompt_skips_coverage_block_when_expected_zero() {
        let agent = agent_for_tests();
        let report = CoverageReport {
            expected: 0,
            fetched: 0,
            missing_section_ids: vec![],
        };
        let inputs = SynthesisInputs {
            query: "q",
            agent_draft: "",
            evidence: &[],
            detected_lang: LangCode::Other,
            coverage_report: Some(&report),
        };
        let prompt = agent.build_prompt(&inputs);
        assert!(!prompt.contains("COVERAGE NOTE"));
    }

    #[test]
    fn build_prompt_emits_no_evidence_marker_when_pack_empty() {
        let agent = agent_for_tests();
        let inputs = SynthesisInputs {
            query: "What does the corpus say?",
            agent_draft: "",
            evidence: &[],
            detected_lang: LangCode::En,
            coverage_report: None,
        };
        let prompt = agent.build_prompt(&inputs);
        assert!(prompt.contains("(no evidence retrieved)"));
    }

    #[test]
    fn build_prompt_truncates_oversize_chunks() {
        let agent = agent_for_tests();
        // Build a chunk well over the snippet cap to force the
        // truncation suffix.
        let big = "a".repeat(PROMPT_EVIDENCE_SNIPPET_CHARS + 1000);
        let evidence = vec![ev("c-1", &big)];
        let inputs = SynthesisInputs {
            query: "q",
            agent_draft: "",
            evidence: &evidence,
            detected_lang: LangCode::Other,
            coverage_report: None,
        };
        let prompt = agent.build_prompt(&inputs);
        // Truncation suffix is present.
        assert!(prompt.contains("...\n    (section: 5.1 Test)"));
        // Full original chunk is NOT present verbatim.
        assert!(!prompt.contains(&big));
    }

    #[test]
    fn build_prompt_omits_section_path_when_blank() {
        let agent = agent_for_tests();
        let mut e = ev("c-2", "x");
        e.payload = json!({ "chunk_id": "c-2", "chunk_text": "x", "section_path": "" });
        let inputs = SynthesisInputs {
            query: "q",
            agent_draft: "",
            evidence: &[e],
            detected_lang: LangCode::Other,
            coverage_report: None,
        };
        let prompt = agent.build_prompt(&inputs);
        assert!(!prompt.contains("(section:"));
    }
}

#[cfg(test)]
mod tests_synth_event {
    //! Stream-shape sanity checks that don't require a live LLM.
    use super::*;

    #[test]
    fn synth_event_is_clonable_for_fanout() {
        // The streaming server may need to clone events when fanning
        // them out to multiple subscribers (e.g., observability +
        // user-facing). Pin the bound here so a future refactor does
        // not silently drop `Clone`.
        let ev = SynthEvent::TextDelta("hello".into());
        let _cloned = ev.clone();
    }

    #[test]
    fn synth_event_done_carries_full_text_and_usage() {
        let usage = RigUsage::new();
        let ev = SynthEvent::Done {
            full_text: "answer".into(),
            usage,
        };
        match ev {
            SynthEvent::Done { full_text, .. } => assert_eq!(full_text, "answer"),
            _ => panic!("expected Done"),
        }
    }
}
