//! Grounding gate -- verify answer grounding, retry if needed.
//!
//! Runs the [`GroundingVerifier`] on the agent's answer and evidence.
//! If the answer is not grounded and retries remain, rebuilds the prompt
//! with feedback, re-runs the agent, re-fuses evidence, and re-verifies.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::Confidence;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use tracing::{error, info, instrument, warn};

use crate::context::QueryContext;
use crate::steps::agent_runner::AgentRunner;
use crate::steps::evidence_fuser::EvidenceFuser;
use crate::verifier::{GroundingVerifier, VerificationResult};

// ---- Subject-absent short-circuit ----

/// Minimum token length considered as a content-bearing word.
///
/// Shorter tokens are mostly function words, pronouns, and single-syllable
/// particles across EN/RO; including them would produce a very high false
/// positive rate on the subject-absence check.
const SUBJECT_ABSENT_MIN_TOKEN_LEN: usize = 4;

/// Length of the prefix used when searching for a query token inside chunk
/// text. A prefix rather than an exact match accepts inflectional endings —
/// Romanian and other morphologically rich languages routinely change the
/// trailing syllables across cases and verb conjugations, and pinning the
/// head of the word is enough to detect corpus presence.
const SUBJECT_ABSENT_PREFIX_LEN: usize = 4;

/// Stopwords dropped before the subject-absence check.
///
/// Intentionally minimal and bilingual (EN + RO) so a universal corpus can
/// adopt the gate without language-specific tuning. Additions should cover
/// tokens that carry no topical signal and whose absence from a corpus
/// would not indicate an out-of-scope query.
const SUBJECT_ABSENT_STOPWORDS: &[&str] = &[
    // EN — high-frequency function words
    "the", "and", "what", "when", "where", "this", "that", "with", "from", "have", "will", "about",
    "there", "would", "could", "should", "which", "your", "yours",
    // RO — high-frequency function words
    "care", "cum", "pentru", "acest", "aceasta", "acea", "sunt", "este", "dintre", "intre", "doar",
    "chiar", "pana", "catre", "dupa", "inainte",
];

/// Decide whether the query's subject appears in the retrieved evidence.
///
/// Returns `true` when **any** content token from the query is absent from
/// every retrieved chunk. The rationale is asymmetric: a grounded answer
/// needs all subject tokens in the evidence; missing even one is a strong
/// signal that the corpus does not cover the topic. Stemming is
/// approximated via a fixed-length prefix so inflectional endings
/// (`histograma` vs `histogramei`) do not cause spurious misses.
///
/// Returns `false` (i.e. "subject present — let the pipeline retry") when:
/// - the query has no content tokens after filtering (no signal to act on);
/// - the evidence slice is empty (retries won't help, but grounding gate
///   has its own handling for zero-evidence which we should not bypass);
/// - every content token prefix is found somewhere in the evidence.
///
/// # Deprecation
///
/// Prefer [`subject_absent_v2`] which reads the rewritten subject from
/// `ctx.tool_calls` instead of the raw user query. The raw query contains
/// instruction-meta tokens (`cheatsheet`, `punctaj`, `organizat`) that are
/// absent from every real corpus and produce false positives.
#[deprecated(note = "use subject_absent_v2 instead — see Task 5.1")]
#[allow(dead_code)]
fn subject_absent_from_evidence(query: &str, evidence: &[SearchResult]) -> bool {
    if evidence.is_empty() {
        return false;
    }

    let stopwords: HashSet<&str> = SUBJECT_ABSENT_STOPWORDS.iter().copied().collect();

    let content_prefixes: Vec<String> = query
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| t.chars().count() >= SUBJECT_ABSENT_MIN_TOKEN_LEN)
        .map(|t| t.to_lowercase())
        .filter(|t| !stopwords.contains(t.as_str()))
        .map(|t| {
            t.chars()
                .take(SUBJECT_ABSENT_PREFIX_LEN)
                .collect::<String>()
        })
        .collect();

    if content_prefixes.is_empty() {
        return false;
    }

    // Concatenate all chunk bodies into a single lowercased haystack so each
    // prefix lookup is O(|haystack|) with no repeated allocation. Payloads
    // lacking `chunk_text` are silently skipped; the gate treats them as
    // empty contributors to the haystack.
    let mut haystack = String::new();
    for doc in evidence {
        if let Some(text) = doc.payload.get("chunk_text").and_then(|v| v.as_str()) {
            haystack.push_str(&text.to_lowercase());
            haystack.push(' ');
        }
    }

    content_prefixes
        .iter()
        .any(|prefix| !haystack.contains(prefix))
}

// ---- Subject-absent v2 ----

/// Extract the rewritten subject phrase from a tool-call's args JSON.
///
/// All retrieval tools that carry semantic content pass the phrase under the
/// `query` key (`search_knowledge`, `search_sections`). The `fetch_section_chunks`
/// tool uses a `section_id` key that encodes a structural identifier rather than
/// a semantic phrase, so those calls are skipped (returning `None`).
fn extract_subject_phrase(args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    v.get("query").and_then(|q| q.as_str()).map(|s| s.to_string())
}

/// Collect all ≥4-char alphabetic tokens from a query as subject noun candidates.
///
/// Returns tokens sorted longest-first so the most specific term is checked
/// first when iterating. A query like `"segmentare metode algoritmice"` yields
/// `["algoritmice", "segmentare", "metode"]`.
fn subject_nouns(q: &str) -> Vec<String> {
    let mut tokens: Vec<&str> = q
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| t.chars().count() >= 4)
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

/// Returns `true` when all ≥4-char alphabetic tokens from `query` are absent
/// from `haystack` (case-folded exact-substring match).
///
/// An empty token set (query too short or all tokens < 4 chars) returns `false`
/// so ambiguous short queries do not trigger the short-circuit.
fn all_tokens_absent(query: &str, haystack: &str) -> bool {
    let nouns = subject_nouns(query);
    if nouns.is_empty() {
        return false;
    }
    nouns.iter().all(|noun| !haystack.contains(noun.as_str()))
}

/// Replacement for [`subject_absent_from_evidence`].
///
/// Instead of tokenising the raw user query — which contains instruction-meta
/// tokens like `cheatsheet`, `punctaj`, `organizat` that never appear in real
/// corpora and produce false positives — this function reads the **rewritten**
/// queries that the LLM issued to the retrieval tools (captured in
/// `ctx.tool_calls` by `RecordingToolDyn`, Task 4.5).
///
/// A query's subject is considered **present in evidence** when at least one
/// of its ≥4-char content tokens appears as a case-folded substring in the
/// evidence haystack.  The overall verdict is "absent" only when **every**
/// rewritten query's subject is absent — i.e. no retrieved query's vocabulary
/// overlaps with the evidence corpus at all.
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
    queries.iter().all(|q| all_tokens_absent(q, &haystack))
}

/// Verify answer grounding and retry if needed.
pub struct GroundingGate {
    verifier: Arc<GroundingVerifier>,
    max_retries: usize,
    metrics: Arc<RetrievalMetrics>,
}

impl GroundingGate {
    /// Create a new GroundingGate.
    pub fn new(
        verifier: Arc<GroundingVerifier>,
        max_retries: usize,
        metrics: Arc<RetrievalMetrics>,
    ) -> Self {
        Self {
            verifier,
            max_retries,
            metrics,
        }
    }

    /// Verify grounding and retry if needed.
    ///
    /// Populates `ctx.verification` and `ctx.retries`.
    #[instrument(skip(self, ctx, agent_runner, evidence_fuser), fields(
        session_id = %ctx.session_id,
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

        // Subject-absent short-circuit: if the first-pass retrieval does not
        // contain any of the query's content tokens, further retries will not
        // surface evidence (the corpus does not cover the topic). Skip them
        // to save ~40-80 s per out-of-corpus query and preserve the agent's
        // honest "not covered" answer instead of running more searches that
        // will fail the same way.
        if let Some(ref verification) = ctx.verification {
            if self.verifier.needs_retry(verification)
                && subject_absent_v2(&ctx.query, &ctx.tool_calls, &ctx.evidence)
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
    use agent_rs_stores::types::SearchResult;
    use agent_rs_core::types::Score;

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

    #[test]
    fn instruction_heavy_query_is_no_longer_false_positive() {
        let raw = "vreau cheatsheet cu TOATE metodele de segmentare punctaj scriu organizat";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă.",
        )];
        let calls = vec![dummy_tool_call("segmentare metode algoritmice")];
        // v2 must extract "segmentare" subject from the rewritten query
        // (in tool_calls) and find it in evidence -> NOT absent.
        assert!(!subject_absent_v2(raw, &calls, &evidence));
    }

    #[test]
    fn out_of_corpus_still_short_circuits() {
        let raw = "ce este React?";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă.",
        )];
        let calls = vec![dummy_tool_call("react javascript framework")];
        assert!(subject_absent_v2(raw, &calls, &evidence));
    }

    #[test]
    fn falls_back_safely_when_no_tool_calls_recorded() {
        let raw = "anything";
        let evidence = vec![dummy_evidence("...")];
        // No tool calls -> v2 returns false (do not short-circuit).
        assert!(!subject_absent_v2(raw, &[], &evidence));
    }

    #[test]
    fn falls_back_safely_when_evidence_empty() {
        let raw = "anything";
        let calls = vec![dummy_tool_call("anything")];
        // No evidence -> v2 returns false (let the existing zero-evidence
        // path in synthesis handle it).
        assert!(!subject_absent_v2(raw, &calls, &[]));
    }
}
