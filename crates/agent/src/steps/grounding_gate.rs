//! Grounding gate -- verify answer grounding, retry if needed.
//!
//! Runs the [`GroundingVerifier`] on the agent's answer and evidence.
//! If the answer is not grounded and retries remain, rebuilds the prompt
//! with feedback, re-runs the agent, re-fuses evidence, and re-verifies.

use std::cmp::Reverse;
use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::Confidence;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use tracing::{error, info, instrument, warn};

use crate::context::QueryContext;
use crate::coverage::CoverageReport;
use crate::steps::agent_runner::AgentRunner;
use crate::steps::evidence_fuser::EvidenceFuser;
use crate::verifier::{GroundingVerifier, VerificationResult};

// ---- Subject-absent short-circuit ----

/// Extract the rewritten subject phrase from a tool-call's args JSON.
///
/// All retrieval tools that carry semantic content pass the phrase under the
/// `query` key (`search_knowledge`, `search_sections`). The `fetch_section_chunks`
/// tool uses a `section_id` key that encodes a structural identifier rather than
/// a semantic phrase, so those calls are skipped (returning `None`).
fn extract_subject_phrase(args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    v.get("query")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string())
}

/// Collect all `≥min_len`-char alphabetic tokens from a query as subject
/// noun candidates.
///
/// Returns tokens sorted longest-first so the most specific term is checked
/// first when iterating. A query like `"segmentare metode algoritmice"` with
/// `min_len = 4` yields `["algoritmice", "segmentare", "metode"]`.
///
/// `min_len` is sourced from
/// [`agent_rs_core::config::GroundingConfig::subject_min_token_len`] so the
/// threshold can be tuned per deployment without a recompile.
fn subject_nouns(q: &str, min_len: usize) -> Vec<String> {
    let mut tokens: Vec<&str> = q
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| t.chars().count() >= min_len)
        .collect();
    tokens.sort_by_key(|t| Reverse(t.len()));
    tokens.iter().map(|s| s.to_lowercase()).collect()
}

/// Concatenate all chunk bodies into a single lowercased string for
/// substring lookups. Chunks lacking a `chunk_text` key are skipped.
fn build_haystack_lc(ev: &[SearchResult]) -> String {
    let mut s = String::new();
    for e in ev {
        if let Some(t) = e.payload.get("chunk_text").and_then(|v| v.as_str()) {
            s.push_str(&t.to_lowercase());
            s.push(' ');
        }
    }
    s
}

/// Returns `true` when all `≥min_len`-char alphabetic tokens from `query`
/// are absent from `haystack` (case-folded exact-substring match).
///
/// An empty token set (query too short or all tokens < `min_len` chars)
/// returns `false` so ambiguous short queries do not trigger the
/// short-circuit.
fn all_tokens_absent(query: &str, haystack: &str, min_len: usize) -> bool {
    let nouns = subject_nouns(query, min_len);
    if nouns.is_empty() {
        return false;
    }
    nouns.iter().all(|noun| !haystack.contains(noun.as_str()))
}

/// Subject-presence check on the rewritten retrieval queries.
///
/// Reads the queries the agent issued to the retrieval tools (captured in
/// `ctx.tool_calls` by `RecordingToolDyn` in `agent_runner`) rather than
/// the raw user prompt — the raw prompt contains instruction-meta tokens
/// (`cheatsheet`, `summary`, `organize`, …) that never appear in real
/// corpora and would produce false positives.
///
/// A query's subject is considered **present in evidence** when at least
/// one of its `≥min_len`-char content tokens appears as a case-folded
/// substring in the evidence haystack. The overall verdict is "absent"
/// only when **every** rewritten query's subject is absent — i.e. no
/// retrieved query's vocabulary overlaps with the evidence corpus at all.
///
/// # Safe fallback
///
/// Returns `false` ("do not short-circuit") when:
/// - `evidence` is empty — the existing zero-evidence handling takes over;
/// - `tool_calls` is empty — no recorded queries to derive a subject from;
/// - no `query` field could be extracted from any tool call.
pub(crate) fn subject_absent_v2(
    _raw_query: &str,
    tool_calls: &[crate::coverage::ToolCallRecord],
    evidence: &[SearchResult],
    min_len: usize,
) -> bool {
    if evidence.is_empty() || tool_calls.is_empty() {
        return false;
    }
    let queries: Vec<String> = tool_calls
        .iter()
        .filter_map(|t| extract_subject_phrase(&t.args_json))
        .collect();
    if queries.is_empty() {
        return false;
    }

    let haystack = build_haystack_lc(evidence);

    // Subject is "absent" only if EVERY rewritten query has ALL its tokens
    // missing from the evidence. If even one token from any tool call hits
    // the haystack, we trust that the subject was retrieved.
    queries
        .iter()
        .all(|q| all_tokens_absent(q, &haystack, min_len))
}

/// Returns `true` when the coverage report provides deterministic proof that
/// every planner-targeted section was fetched and no retry is needed.
///
/// Coverage is the truth signal for exhaustive queries: when `expected > 0`
/// and `fetched == expected`, the breadth of retrieval is guaranteed regardless
/// of what the LLM-side grounding verifier says. The verifier is structurally
/// unreliable on long aggregate answers (false negatives are common), so when
/// coverage is complete we short-circuit the retry loop and promote the answer
/// to grounded.
pub(crate) fn should_promote_via_coverage(report: Option<&CoverageReport>) -> bool {
    report.is_some_and(|r| r.expected > 0 && r.is_complete())
}

/// Verify answer grounding and retry if needed.
pub struct GroundingGate {
    verifier: Arc<GroundingVerifier>,
    max_retries: usize,
    /// Minimum character length for a query token in the subject-absence
    /// pre-filter. Sourced from
    /// [`agent_rs_core::config::GroundingConfig::subject_min_token_len`].
    subject_min_token_len: usize,
    metrics: Arc<RetrievalMetrics>,
}

impl GroundingGate {
    /// Create a new GroundingGate.
    pub fn new(
        verifier: Arc<GroundingVerifier>,
        max_retries: usize,
        subject_min_token_len: usize,
        metrics: Arc<RetrievalMetrics>,
    ) -> Self {
        Self {
            verifier,
            max_retries,
            subject_min_token_len,
            metrics,
        }
    }

    /// Verify grounding and retry if needed.
    ///
    /// Populates `ctx.verification` and `ctx.retries`.
    #[instrument(skip(self, ctx, agent_runner, evidence_fuser), fields(
        trace_id = ?ctx.trace_id,
        max_retries = self.max_retries,
    ))]
    pub async fn verify_and_retry(
        &self,
        ctx: &mut QueryContext,
        agent_runner: &AgentRunner,
        evidence_fuser: &EvidenceFuser,
    ) -> Result<(), AgentError> {
        // Initial verification
        ctx.verification = Some(self.verifier.verify(&ctx.answer, &ctx.evidence).await?);

        ctx.retries = 0;

        // Coverage short-circuit: if the planner-targeted sections were all
        // fetched (deterministic proof of breadth), skip the LLM-side grounding
        // retry. The verifier is unreliable on long aggregate answers; coverage
        // is the truth signal for exhaustive queries.
        if should_promote_via_coverage(ctx.coverage_report.as_ref()) {
            if let Some(ref mut v) = ctx.verification {
                v.is_grounded = true;
                tracing::info!(
                    expected = ctx
                        .coverage_report
                        .as_ref()
                        .map(|r| r.expected)
                        .unwrap_or(0),
                    fetched = ctx.coverage_report.as_ref().map(|r| r.fetched).unwrap_or(0),
                    confidence = v.confidence.value(),
                    "grounding gate: coverage_report complete — promoting to grounded, skipping retries"
                );
            }
            return Ok(());
        }

        // Subject-absent short-circuit: if the first-pass retrieval does not
        // contain any of the query's content tokens, further retries will not
        // surface evidence (the corpus does not cover the topic). Skip them
        // to save ~40-80 s per out-of-corpus query and preserve the agent's
        // honest "not covered" answer instead of running more searches that
        // will fail the same way.
        if let Some(ref verification) = ctx.verification {
            if self.verifier.needs_retry(verification)
                && subject_absent_v2(
                    &ctx.query,
                    &ctx.tool_calls,
                    &ctx.evidence,
                    self.subject_min_token_len,
                )
            {
                warn!(
                    query = %ctx.query,
                    evidence_count = ctx.evidence.len(),
                    "grounding gate: subject absent from first-pass evidence — skipping retries"
                );
                self.metrics.record_grounding_subject_absent(ctx.strategy);
                return Ok(());
            }
        }

        while let Some(ref verification) = ctx.verification {
            if !self.verifier.needs_retry(verification) || ctx.retries >= self.max_retries {
                // Emit retries_exhausted only when we hit the cap while still ungrounded.
                if ctx.retries >= self.max_retries && self.verifier.needs_retry(verification) {
                    self.metrics
                        .record_grounding_retries_exhausted(ctx.strategy);
                }
                break;
            }

            ctx.retries += 1;
            self.metrics.record_grounding_retry(ctx.strategy);
            warn!(
                confidence = verification.confidence.value(),
                is_grounded = verification.is_grounded,
                evidence_count = ctx.evidence.len(),
                retry = ctx.retries,
                max_retries = self.max_retries,
                query = %ctx.query,
                method = %verification.method,
                reasoning = %verification.reasoning,
                ungrounded_claims = ?verification.ungrounded_claims,
                "grounding failed, retrying"
            );

            // Spec A2: build a DIAGNOSTIC retry prompt that targets the
            // specific evidence gaps rather than a generic "try harder".
            // `ungrounded_claims` is populated by the verifier but was
            // previously ignored — now it drives targeted re-search.
            let gap_analysis = if !verification.ungrounded_claims.is_empty() {
                let claims = verification
                    .ungrounded_claims
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  {}. {}", i + 1, c))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "The following claims in your previous answer were NOT supported \
                     by the retrieved evidence:\n{}\n\n\
                     Search SPECIFICALLY for information to support or refute these \
                     claims. Use targeted queries for each unsupported claim. \
                     If you cannot find evidence for a claim, REMOVE it from your \
                     answer rather than keeping it ungrounded.",
                    claims
                )
            } else {
                "Your previous answer was not well-grounded in the evidence. \
                 The confidence score was too low. Search with broader terms \
                 and provide an answer that directly cites retrieved evidence."
                    .to_owned()
            };

            let retry_prompt = format!(
                "{}\n\n{}\n\nPrevious answer (revise based on the above): {}",
                ctx.query, gap_analysis, ctx.answer
            );

            match agent_runner.rerun(ctx, &retry_prompt).await {
                Ok(()) => {
                    // Re-fuse evidence after retry
                    evidence_fuser.fuse(ctx);

                    // Re-verify
                    ctx.verification =
                        Some(self.verifier.verify(&ctx.answer, &ctx.evidence).await?);

                    if let Some(ref v) = ctx.verification {
                        info!(
                            retry = ctx.retries,
                            is_grounded = v.is_grounded,
                            confidence = v.confidence.value(),
                            evidence_count = ctx.evidence.len(),
                            method = %v.method,
                            reasoning = %v.reasoning,
                            ungrounded_claims = ?v.ungrounded_claims,
                            "retry verification complete"
                        );
                    }
                }
                Err(err) => {
                    error!(
                        retry = ctx.retries,
                        error = %err,
                        "retry failed"
                    );

                    // Force ungrounded state
                    ctx.verification = Some(VerificationResult {
                        is_grounded: false,
                        confidence: Confidence::new(0.0).map_err(|e| {
                            AgentError::Verification(format!("confidence error: {e}"))
                        })?,
                        reasoning: format!("Retry {} failed: {err}", ctx.retries),
                        ungrounded_claims: Vec::new(),
                        claim_confidences: Vec::new(),
                        method: "retry_failed".to_string(),
                    });
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests_v2 {
    use super::*;
    use crate::coverage::ToolCallRecord;
    use agent_rs_core::types::Score;
    use agent_rs_stores::types::SearchResult;

    fn dummy_evidence(text: &str) -> SearchResult {
        SearchResult {
            id: String::new(),
            score: Score::new(0.0),
            payload: serde_json::json!({"chunk_text": text}),
        }
    }

    fn dummy_tool_call(query: &str) -> ToolCallRecord {
        ToolCallRecord {
            tool_name: "search_knowledge".into(),
            args_json: format!(r#"{{"query":"{}"}}"#, query),
        }
    }

    // ---- should_promote_via_coverage ----

    #[test]
    fn coverage_complete_promotes_to_grounded() {
        use crate::coverage::CoverageReport;

        // All sections fetched: should promote.
        let report = CoverageReport {
            expected: 20,
            fetched: 20,
            missing_section_ids: vec![],
        };
        assert!(should_promote_via_coverage(Some(&report)));
    }

    #[test]
    fn coverage_partial_does_not_promote() {
        use crate::coverage::CoverageReport;

        // Partial: retries may still fill more sections.
        let report = CoverageReport {
            expected: 20,
            fetched: 15,
            missing_section_ids: vec!["s1".into()],
        };
        assert!(!should_promote_via_coverage(Some(&report)));
    }

    #[test]
    fn coverage_none_does_not_promote() {
        // No coverage report: non-exhaustive flow, must not promote.
        assert!(!should_promote_via_coverage(None));
    }

    #[test]
    fn coverage_zero_expected_does_not_promote() {
        use crate::coverage::CoverageReport;

        // expected == 0 means the planner ran but targeted nothing useful;
        // this is not "complete" in the meaningful sense.
        let report = CoverageReport {
            expected: 0,
            fetched: 0,
            missing_section_ids: vec![],
        };
        assert!(!should_promote_via_coverage(Some(&report)));
    }

    #[test]
    fn instruction_heavy_query_is_no_longer_false_positive() {
        let raw = "vreau cheatsheet cu TOATE metodele de segmentare punctaj scriu organizat";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă.",
        )];
        let calls = vec![dummy_tool_call("segmentare metode algoritmice")];
        // v2 must extract "segmentare" subject from the rewritten query
        // (in tool_calls) and find it in evidence -> NOT absent.
        assert!(!subject_absent_v2(raw, &calls, &evidence, 4));
    }

    #[test]
    fn out_of_corpus_still_short_circuits() {
        let raw = "ce este React?";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă.",
        )];
        let calls = vec![dummy_tool_call("react javascript framework")];
        assert!(subject_absent_v2(raw, &calls, &evidence, 4));
    }

    #[test]
    fn falls_back_safely_when_no_tool_calls_recorded() {
        let raw = "anything";
        let evidence = vec![dummy_evidence("...")];
        // No tool calls -> v2 returns false (do not short-circuit).
        assert!(!subject_absent_v2(raw, &[], &evidence, 4));
    }

    #[test]
    fn falls_back_safely_when_evidence_empty() {
        let raw = "anything";
        let calls = vec![dummy_tool_call("anything")];
        // No evidence -> v2 returns false (let the existing zero-evidence
        // path in synthesis handle it).
        assert!(!subject_absent_v2(raw, &calls, &[], 4));
    }
}
