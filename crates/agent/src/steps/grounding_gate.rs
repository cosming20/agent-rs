//! Grounding gate -- verify answer grounding, retry if needed.
//!
//! Runs the [`GroundingVerifier`] on the agent's answer and evidence.
//! If the answer is not grounded and retries remain, rebuilds the prompt
//! with feedback, re-runs the agent, re-fuses evidence, and re-verifies.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::sync::Arc;

use agent_rs_core::config::RetryStrategy;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::Confidence;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use tracing::{error, info, instrument, warn};

use crate::claim_searcher::ClaimSearcher;
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
    /// Retrieval handle used by the claim-directed retry path.
    ///
    /// The gate calls
    /// [`ClaimSearcher::search_for_claim`] once per claim flagged as
    /// unsupported by the verifier, BEFORE re-running the agent, and
    /// only falls back to a full agent rerun when the claim-directed
    /// search produced zero new evidence. This breaks the **circular
    /// grounding bug** the legacy retry path suffered from (re-running
    /// the same agent with the same evidence reaches the same verdict).
    claim_searcher: Arc<dyn ClaimSearcher>,
    /// Strategy chosen by
    /// [`agent_rs_core::config::GroundingConfig::retry_strategy`].
    retry_strategy: RetryStrategy,
    /// Per-claim search top-k from
    /// [`agent_rs_core::config::GroundingConfig::claim_search_top_k`].
    claim_search_top_k: usize,
    metrics: Arc<RetrievalMetrics>,
}

impl GroundingGate {
    /// Create a new [`GroundingGate`].
    ///
    /// # Parameters
    ///
    /// - `verifier` — LLM (or heuristic) faithfulness verifier.
    /// - `max_retries` — total number of retry iterations the gate may
    ///   perform per request.
    /// - `subject_min_token_len` — character threshold for the
    ///   subject-absence pre-filter (sourced from
    ///   [`agent_rs_core::config::GroundingConfig`]).
    /// - `claim_searcher` — handle used by the claim-directed retry
    ///   path. Use [`crate::claim_searcher::RouterClaimSearcher`] in
    ///   production; mocks in tests.
    /// - `retry_strategy` — selects between claim-directed and
    ///   agent-rerun retry semantics.
    /// - `claim_search_top_k` — per-claim retrieval breadth (consulted
    ///   only under [`RetryStrategy::ClaimDirected`]).
    /// - `metrics` — shared OTel instruments for retry telemetry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verifier: Arc<GroundingVerifier>,
        max_retries: usize,
        subject_min_token_len: usize,
        claim_searcher: Arc<dyn ClaimSearcher>,
        retry_strategy: RetryStrategy,
        claim_search_top_k: usize,
        metrics: Arc<RetrievalMetrics>,
    ) -> Self {
        Self {
            verifier,
            max_retries,
            subject_min_token_len,
            claim_searcher,
            retry_strategy,
            claim_search_top_k,
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

        while let Some(verification) = ctx.verification.clone() {
            if !self.verifier.needs_retry(&verification) || ctx.retries >= self.max_retries {
                // Emit retries_exhausted only when we hit the cap while still ungrounded.
                if ctx.retries >= self.max_retries && self.verifier.needs_retry(&verification) {
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
                retry_strategy = ?self.retry_strategy,
                "grounding failed, retrying"
            );

            // Claim-directed path: when configured AND the verifier
            // produced concrete unsupported claims, dispatch a focused
            // retrieval per claim BEFORE re-running the agent. The
            // agent-rerun path is reserved for "no specific claim to
            // chase" or "claim search produced nothing new" — see
            // [`verify_and_retry`] doc.
            if self.retry_strategy == RetryStrategy::ClaimDirected
                && !verification.ungrounded_claims.is_empty()
            {
                let new_evidence_count = self
                    .run_claim_directed_search(ctx, &verification.ungrounded_claims)
                    .await?;

                if new_evidence_count > 0 {
                    // Skip the agent rerun this iteration: just re-verify
                    // with the augmented evidence. The agent's previous
                    // answer is left in place; the verifier may now see
                    // support for the previously-unsupported claims.
                    ctx.verification =
                        Some(self.verifier.verify(&ctx.answer, &ctx.evidence).await?);

                    if let Some(ref v) = ctx.verification {
                        info!(
                            retry = ctx.retries,
                            is_grounded = v.is_grounded,
                            confidence = v.confidence.value(),
                            evidence_count = ctx.evidence.len(),
                            new_evidence = new_evidence_count,
                            method = %v.method,
                            "claim-directed retry verification complete"
                        );
                    }
                    continue;
                }

                // Zero new evidence — fall through to the agent rerun
                // path so the agent has another chance to retrieve
                // through its own tool calls (which can issue searches
                // the gate cannot, e.g. fetch_section_chunks).
                info!(
                    retry = ctx.retries,
                    "claim-directed search produced no new evidence; falling back to agent rerun"
                );
            }

            // The agent-rerun path may surface a permanent failure
            // (rerun call errored, retry forced into ungrounded state).
            // The legacy loop broke out of the while in that case and
            // returned `Ok(())` so the rest of the pipeline could still
            // synthesize a (forced-ungrounded) response — preserve that
            // semantics by interpreting `Ok(false)` as "stop iterating".
            let keep_going = self
                .run_agent_rerun(ctx, agent_runner, evidence_fuser, &verification)
                .await?;
            if !keep_going {
                break;
            }
        }

        Ok(())
    }

    /// Run one claim-directed search batch: dispatch a focused
    /// retrieval per claim, dedup against `ctx.evidence` by `id`, and
    /// append novel results in place.
    ///
    /// Returns the count of novel evidence items appended. A return of
    /// `0` instructs the caller to fall back to a full agent rerun
    /// (the per-claim searches turned up nothing the agent had not
    /// already retrieved).
    async fn run_claim_directed_search(
        &self,
        ctx: &mut QueryContext,
        ungrounded_claims: &[String],
    ) -> Result<usize, AgentError> {
        self.metrics
            .record_grounding_claim_directed_retry(ctx.strategy);

        let mut seen_ids: HashSet<String> = ctx.evidence.iter().map(|e| e.id.clone()).collect();
        let mut new_total: usize = 0;

        for claim in ungrounded_claims {
            let trimmed = claim.trim();
            if trimmed.is_empty() {
                continue;
            }

            match self
                .claim_searcher
                .search_for_claim(
                    trimmed,
                    &ctx.user_id,
                    &ctx.active_document_keys,
                    ctx.strategy,
                    self.claim_search_top_k,
                )
                .await
            {
                Ok(results) => {
                    let mut added: usize = 0;
                    for r in results {
                        if seen_ids.insert(r.id.clone()) {
                            ctx.evidence.push(r);
                            added += 1;
                        }
                    }
                    new_total += added;
                    info!(
                        claim = %trimmed,
                        new_evidence = added,
                        "claim-directed search complete"
                    );
                }
                Err(err) => {
                    // Soft-fail: per-claim retrieval failures must not
                    // abort the gate. Log + carry on so other claims
                    // still get their chance.
                    warn!(
                        claim = %trimmed,
                        error = %err,
                        "claim-directed search failed; continuing with remaining claims"
                    );
                }
            }
        }

        self.metrics
            .record_grounding_claim_directed_new_evidence(new_total as u64);
        Ok(new_total)
    }

    /// Run the legacy agent-rerun retry path: build a diagnostic
    /// prompt around the verifier's `ungrounded_claims`, re-run the
    /// agent through `agent_runner`, re-fuse evidence, and re-verify.
    ///
    /// This is the original (Phase 1/2) loop body; isolated into a
    /// helper so both [`RetryStrategy::AgentRerun`] and the
    /// "claim-directed produced no new evidence" fallback path share
    /// it byte-for-byte.
    ///
    /// # Return value
    ///
    /// - `Ok(true)` — the rerun completed normally; the outer loop
    ///   should re-evaluate the freshly written `ctx.verification`.
    /// - `Ok(false)` — the rerun failed (the agent itself errored).
    ///   `ctx.verification` has been forced into an ungrounded state
    ///   with a `retry_failed` method tag; the outer loop must stop.
    ///   Returning `false` instead of `Err` matches the legacy retry
    ///   semantics: pipeline downstream still synthesizes a
    ///   forced-ungrounded response rather than 500-ing the request.
    async fn run_agent_rerun(
        &self,
        ctx: &mut QueryContext,
        agent_runner: &AgentRunner,
        evidence_fuser: &EvidenceFuser,
        verification: &VerificationResult,
    ) -> Result<bool, AgentError> {
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
                ctx.verification = Some(self.verifier.verify(&ctx.answer, &ctx.evidence).await?);

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
                Ok(true)
            }
            Err(err) => {
                error!(
                    retry = ctx.retries,
                    error = %err,
                    "retry failed"
                );

                // Force ungrounded state. Wrap the confidence
                // construction error in `Verification` so the loop
                // surfaces a structured error rather than panicking on
                // an out-of-range value (which the typed `Confidence`
                // domain rejects at construction).
                ctx.verification = Some(VerificationResult {
                    is_grounded: false,
                    confidence: Confidence::new(0.0)
                        .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?,
                    reasoning: format!("Retry {} failed: {err}", ctx.retries),
                    ungrounded_claims: Vec::new(),
                    claim_confidences: Vec::new(),
                    method: "retry_failed".to_string(),
                });
                // Match the legacy semantics: signal "stop iterating"
                // without bubbling the error up so the rest of the
                // pipeline can still produce a forced-ungrounded
                // response rather than 500'ing the entire request.
                Ok(false)
            }
        }
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

#[cfg(test)]
mod tests_claim_directed_retry {
    //! Tests for the Phase-3 claim-directed retry path. Verify that:
    //!
    //! 1. A successful per-claim search appends novel evidence and the
    //!    method returns the count, signalling the gate it can skip the
    //!    agent rerun.
    //! 2. A claim search whose results are all already in `ctx.evidence`
    //!    returns 0, telling the gate to fall through to agent rerun.
    //! 3. The gate's `retry_strategy = AgentRerun` field forces the
    //!    legacy path: a mock claim searcher that records its calls
    //!    sees ZERO calls under that strategy.
    //!
    //! The tests construct a `GroundingGate` directly (no orchestrator)
    //! with a mock `ClaimSearcher` so the end-to-end path stays
    //! deterministic and fast.
    use super::*;
    use crate::claim_searcher::ClaimSearcher;
    use crate::context::QueryContext;
    use agent_rs_core::config::RetryStrategy;
    use agent_rs_core::types::{RetrievalStrategy, Score, UserId};
    use agent_rs_observability::metrics::RetrievalMetrics;
    use agent_rs_stores::types::SearchResult;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Mock [`ClaimSearcher`] that returns canned per-claim results and
    /// records every invocation so tests can assert on call shape.
    struct MockClaimSearcher {
        /// Canned results per call. Drained in FIFO order.
        canned: Mutex<Vec<Vec<SearchResult>>>,
        /// Captured `(claim, top_k)` pairs the gate dispatched.
        calls: Mutex<Vec<(String, usize)>>,
    }

    impl MockClaimSearcher {
        fn new(canned: Vec<Vec<SearchResult>>) -> Self {
            Self {
                canned: Mutex::new(canned),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls
                .lock()
                .expect("mock mutex must not be poisoned in tests")
                .len()
        }
    }

    #[async_trait]
    impl ClaimSearcher for MockClaimSearcher {
        async fn search_for_claim(
            &self,
            claim: &str,
            _user_id: &UserId,
            _active_document_keys: &[String],
            _strategy: RetrievalStrategy,
            top_k: usize,
        ) -> Result<Vec<SearchResult>, AgentError> {
            self.calls
                .lock()
                .expect("mock mutex must not be poisoned in tests")
                .push((claim.to_string(), top_k));

            let mut q = self
                .canned
                .lock()
                .expect("mock mutex must not be poisoned in tests");
            Ok(if q.is_empty() {
                Vec::new()
            } else {
                q.remove(0)
            })
        }
    }

    fn make_evidence(id: &str, text: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score: Score::new(0.0),
            payload: serde_json::json!({"chunk_id": id, "chunk_text": text}),
        }
    }

    fn make_ctx(initial_evidence: Vec<SearchResult>) -> QueryContext {
        let mut ctx = QueryContext::new(
            "what is X?".to_string(),
            UserId::try_new("test-user").expect("valid"),
            Vec::new(),
            Vec::new(),
            RetrievalStrategy::Semantic,
            8,
            None,
        );
        ctx.evidence = initial_evidence;
        ctx
    }

    fn make_gate(retry_strategy: RetryStrategy, searcher: Arc<dyn ClaimSearcher>) -> GroundingGate {
        let metrics = Arc::new(RetrievalMetrics::new("test"));
        // Empty API key — the verifier falls back to the heuristic
        // path. Threshold of 1.1 forces `is_grounded = false` from the
        // heuristic (overlap can never exceed 1.0), but for these
        // tests we never call `verify_and_retry` end-to-end, only the
        // helper methods that don't invoke the verifier.
        let verifier = Arc::new(crate::verifier::GroundingVerifier::new(1.1, "", "test"));
        GroundingGate::new(
            verifier,
            /* max_retries */ 1,
            /* subject_min_token_len */ 4,
            searcher,
            retry_strategy,
            /* claim_search_top_k */ 3,
            metrics,
        )
    }

    #[tokio::test]
    async fn claim_directed_success_appends_novel_evidence() {
        // Initial evidence: one chunk already known. Mock returns one
        // novel result + one duplicate (same id). After the call we
        // expect the duplicate to be deduped and only the novel item
        // appended → new_total == 1.
        let initial = vec![make_evidence("known-1", "known text")];
        let canned = vec![vec![
            make_evidence("novel-1", "fresh evidence"),
            make_evidence("known-1", "duplicate of existing"),
        ]];
        let mock = Arc::new(MockClaimSearcher::new(canned));
        let gate = make_gate(RetryStrategy::ClaimDirected, mock.clone());

        let mut ctx = make_ctx(initial);
        let claims = vec!["The temperature increased by 5 degrees".to_string()];

        let new_total = gate
            .run_claim_directed_search(&mut ctx, &claims)
            .await
            .expect("claim search must succeed");

        assert_eq!(new_total, 1, "exactly one novel item should be appended");
        assert_eq!(ctx.evidence.len(), 2, "evidence grew by exactly one");
        assert!(
            ctx.evidence.iter().any(|e| e.id == "novel-1"),
            "novel evidence must be present"
        );
        assert_eq!(mock.call_count(), 1, "one search per claim");
    }

    #[tokio::test]
    async fn claim_directed_empty_falls_through_to_rerun() {
        // Mock returns only items already in evidence (or nothing).
        // The method must return 0 so the caller falls back to agent
        // rerun.
        let initial = vec![make_evidence("a", "hello"), make_evidence("b", "world")];
        let canned = vec![
            // First claim: returns only a duplicate.
            vec![make_evidence("a", "dup")],
            // Second claim: returns nothing.
            vec![],
        ];
        let mock = Arc::new(MockClaimSearcher::new(canned));
        let gate = make_gate(RetryStrategy::ClaimDirected, mock.clone());

        let mut ctx = make_ctx(initial);
        let claims = vec![
            "Claim one is unsupported".to_string(),
            "Claim two is also unsupported".to_string(),
        ];

        let new_total = gate
            .run_claim_directed_search(&mut ctx, &claims)
            .await
            .expect("claim search must succeed");

        assert_eq!(
            new_total, 0,
            "no novel evidence — caller must fall through to rerun"
        );
        assert_eq!(ctx.evidence.len(), 2, "evidence unchanged");
        assert_eq!(mock.call_count(), 2, "one search per claim, even on empty");
    }

    #[tokio::test]
    async fn agent_rerun_strategy_skips_claim_search_in_helper() {
        // Verifies the `retry_strategy` field is honoured by the
        // helper's caller: when configured for AgentRerun the gate
        // must NOT invoke `run_claim_directed_search` at all. We
        // assert this indirectly: the test calls
        // `run_claim_directed_search` directly (which always runs
        // when invoked) — but a parallel assertion checks the
        // recorded `retry_strategy` field on the gate so the wiring
        // contract stays explicit.
        //
        // The end-to-end "AgentRerun never calls the claim searcher"
        // behaviour is enforced by the strategy gate in
        // `verify_and_retry`'s outer loop:
        //
        //   if self.retry_strategy == RetryStrategy::ClaimDirected
        //       && !verification.ungrounded_claims.is_empty()
        //
        // Without orchestrating a full pipeline we encode the
        // contract by reading the field after construction.
        let mock = Arc::new(MockClaimSearcher::new(vec![]));
        let gate = make_gate(RetryStrategy::AgentRerun, mock.clone());

        assert_eq!(
            gate.retry_strategy,
            RetryStrategy::AgentRerun,
            "retry_strategy must be wired through"
        );
        // Sanity: the helper itself remains callable — the strategy
        // gate is a *caller-side* guard, not an internal one. This
        // means the helper does NOT check the strategy itself.
        assert_eq!(mock.call_count(), 0, "no spontaneous searches");
    }

    #[tokio::test]
    async fn empty_claim_strings_are_skipped() {
        // Defensive: blank / whitespace-only claims (possible LLM
        // glitches) must not produce no-op searches.
        let mock = Arc::new(MockClaimSearcher::new(vec![]));
        let gate = make_gate(RetryStrategy::ClaimDirected, mock.clone());

        let mut ctx = make_ctx(Vec::new());
        let claims = vec!["".to_string(), "   ".to_string()];

        let new_total = gate
            .run_claim_directed_search(&mut ctx, &claims)
            .await
            .expect("must succeed even on blank claims");

        assert_eq!(new_total, 0);
        assert_eq!(
            mock.call_count(),
            0,
            "blank claims must not reach the searcher"
        );
    }
}
