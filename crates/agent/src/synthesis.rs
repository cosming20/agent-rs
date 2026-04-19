//! Response synthesis with citations and provenance.
//!
//! [`ResponseSynthesizer`] builds structured responses with numbered
//! citations linking answer claims to source evidence documents.

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{Confidence, RetrievalStrategy, truncate_to_char_boundary};
use agent_rs_stores::types::SearchResult;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::coverage::CoverageReport;
use crate::verifier::VerificationResult;

/// A citation linking a claim to its source evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation number (1-based).
    pub index: usize,
    /// Stable chunk identifier from retrieval (UUID or content-hash).
    ///
    /// Populated from `SearchResult.id` so that citations remain stable even
    /// when RRF fusion reorders evidence between the tool call and synthesis.
    pub chunk_id: String,
    /// Relevant excerpt from the source document.
    pub content_snippet: String,
    /// Origin of the evidence (e.g., "qdrant", "brave").
    pub source: String,
    /// Relevance score from retrieval.
    pub score: f64,
}

/// Final agent response with citations and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedResponse {
    /// The synthesized answer text.
    pub answer: String,
    /// Source citations supporting the answer.
    pub citations: Vec<Citation>,
    /// Original user query.
    pub query: String,
    /// User-facing answer quality score (0.0 to 1.0).
    pub confidence: Confidence,
    /// Whether the answer passed grounding verification.
    pub is_grounded: bool,
    /// Retrieval strategy used.
    pub strategy: RetrievalStrategy,
    /// Coverage report when the query was exhaustive-mode.
    ///
    /// `None` for standard (non-exhaustive) flows. Callers use this field
    /// instead of inspecting `QueryContext` directly, so the coverage signal
    /// travels cleanly through the public API.
    pub coverage_report: Option<CoverageReport>,
}

/// Builds final responses with citations and provenance metadata.
pub struct ResponseSynthesizer {
    /// Maximum character length for citation snippets.
    max_snippet_length: usize,
}

impl ResponseSynthesizer {
    /// Create a new synthesizer.
    pub fn new(max_snippet_length: usize) -> Self {
        Self { max_snippet_length }
    }

    /// Build a response with citations from evidence documents.
    ///
    /// # Confidence semantics
    ///
    /// User-facing confidence reflects trust in the answer:
    /// - Grounded answer → verifier confidence (verifier is confident in a grounded answer)
    /// - Ungrounded answer → 0.0 (do not trust, regardless of verifier certainty)
    ///
    /// The prior "invert ungrounded verifier confidence" path was mathematically
    /// unsound: `1.0 - 0.05` flipped a low-certainty ungrounded verdict to 0.95
    /// user-facing confidence, producing confident wrong answers.
    #[instrument(skip(self, answer, evidence, verification, coverage_report), fields(
        evidence_count = evidence.len(),
        strategy = %strategy,
    ))]
    pub fn synthesize(
        &self,
        answer: &str,
        evidence: &[SearchResult],
        query: &str,
        verification: &VerificationResult,
        strategy: RetrievalStrategy,
        coverage_report: Option<&CoverageReport>,
    ) -> Result<SynthesizedResponse, AgentError> {
        // Clean the LLM's answer before further processing — system prompt
        // tries to suppress chatty tails, but a defence-in-depth strip keeps
        // the evidence pack deterministic even when the model drifts.
        let answer = strip_chatty_tail(answer);
        let answer = answer.as_str();

        // Resolve citations by stable chunk_id first ([cite:<id>]), with
        // fallback to legacy [N] positional markers during transition.
        // When the LLM cited nothing, fall back to the top-5 evidence items.
        let citations = {
            let resolved = resolve_cite_markers(answer, evidence);
            if resolved.is_empty() {
                evidence
                    .iter()
                    .enumerate()
                    .take(5)
                    .map(|(i, ev)| make_citation(i + 1, ev, self.max_snippet_length))
                    .collect()
            } else {
                resolved
            }
        };

        // User-facing confidence: trust verifier only when it says grounded.
        let answer_confidence = if verification.is_grounded {
            verification.confidence.value()
        } else {
            0.0
        };

        let confidence = Confidence::new(answer_confidence.clamp(0.0, 1.0))
            .map_err(|e| AgentError::Synthesis(format!("confidence error: {e}")))?;

        // "I don't know" boundary: when evidence is empty OR the answer is
        // ungrounded, replace the synthesized answer with an honest admission
        // rather than risking hallucination.
        let final_answer = if evidence.is_empty() {
            tracing::warn!(
                query,
                strategy = %strategy,
                "zero evidence reached synthesis — returning 'I don't know'"
            );
            "I don't have enough information in my knowledge base to answer \
             this question reliably. The retrieved evidence was insufficient \
             to support a grounded answer."
                .to_owned()
        } else if !verification.is_grounded {
            if coverage_report.is_some() {
                // Coverage path: the report itself carries the truth signal
                // (expected vs fetched sections, missing IDs). A prose hedge
                // here would double-communicate and confuse the user.
                // Per spec §14 Q4 — never append wrapper when coverage_report
                // is present; callers read coverage_report from the response
                // struct for any gate / UX decision.
                tracing::info!(
                    query,
                    strategy = %strategy,
                    "answer ungrounded but coverage_report present — skipping wrapper"
                );
                answer.to_owned()
            } else if answer_self_admits_scope(answer) {
                // Single-hedge guard: when the LLM already admitted scope
                // ("not in the evidence" / "nu apare în evidență"), the wrapper
                // adds nothing except confusion. Let the answer stand on its
                // own honesty — the caller still sees confidence=0 and
                // is_grounded=false via the struct fields.
                tracing::info!(
                    query,
                    strategy = %strategy,
                    "answer ungrounded but self-admits scope — skipping wrapper"
                );
                answer.to_owned()
            } else {
                tracing::warn!(
                    query,
                    strategy = %strategy,
                    "answer ungrounded after retries — returning partial admission"
                );
                format!(
                    "I found some potentially relevant information, but I'm not \
                     confident it fully answers your question. Here's what I found, \
                     but please verify independently:\n\n{}",
                    answer
                )
            }
        } else if answer_confidence < 0.15 {
            // Same single-hedge logic for the very-low-confidence grounded path.
            if answer_self_admits_scope(answer) {
                tracing::info!(
                    query,
                    confidence = answer_confidence,
                    "low-confidence grounded answer self-admits scope — skipping wrapper"
                );
                answer.to_owned()
            } else {
                tracing::warn!(
                    query,
                    confidence = answer_confidence,
                    "very low confidence grounded answer — caveat added"
                );
                format!(
                    "I found some potentially relevant information, but I'm not \
                     confident it fully answers your question. Here's what I found, \
                     but please verify independently:\n\n{}",
                    answer
                )
            }
        } else {
            answer.to_owned()
        };

        Ok(SynthesizedResponse {
            answer: final_answer,
            citations,
            query: query.to_owned(),
            confidence,
            is_grounded: verification.is_grounded,
            strategy,
            coverage_report: coverage_report.cloned(),
        })
    }

}

/// Resolve citation markers in the answer text into [`Citation`]s.
///
/// Supports two forms (UUID-first, with `[N]` fallback for transition):
///
/// - `[cite:<chunk-id>]` — preferred. Looks up evidence by `chunk_id`
///   payload field (also matches `SearchResult.id` for safety).
/// - `[N]` — legacy 1-based index into evidence array. Only used when
///   **no** `[cite:...]` markers are present in the answer.
///
/// Unknown UUID markers are logged at WARN and dropped — better than
/// silently emitting a citation with bogus content.
///
/// Duplicate `[cite:<id>]` references within the same answer are
/// deduplicated; the first occurrence wins.
pub(crate) fn resolve_cite_markers(answer: &str, evidence: &[SearchResult]) -> Vec<Citation> {
    static UUID_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"\[cite:([A-Za-z0-9_-]+)\]")
            .expect("resolve_cite_markers UUID regex is statically valid")
    });
    static IDX_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"\[(\d+)\]")
            .expect("resolve_cite_markers index regex is statically valid")
    });

    // Determine the natural snippet length — use a conservative default when
    // called from outside the synthesizer (e.g., unit tests).
    const DEFAULT_SNIPPET_LEN: usize = 200;

    let mut out: Vec<Citation> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // UUID-based citations first (preferred form).
    for cap in UUID_RE.captures_iter(answer) {
        let id = &cap[1];
        if seen_ids.contains(id) {
            continue;
        }
        if let Some((idx, ev)) = evidence.iter().enumerate().find(|(_, e)| {
            e.id == id
                || e.payload
                    .get("chunk_id")
                    .and_then(|v| v.as_str())
                    == Some(id)
        }) {
            seen_ids.insert(id.to_string());
            out.push(make_citation(idx + 1, ev, DEFAULT_SNIPPET_LEN));
        } else {
            tracing::warn!(cite_id = %id, "cite marker references unknown chunk_id — dropped");
        }
    }

    // Backwards-compat: legacy [N] markers — only used if no UUID markers.
    if out.is_empty() {
        for cap in IDX_RE.captures_iter(answer) {
            if let Ok(n) = cap[1].parse::<usize>() {
                if let Some(ev) = evidence.get(n.saturating_sub(1)) {
                    out.push(make_citation(n, ev, DEFAULT_SNIPPET_LEN));
                }
            }
        }
    }

    out
}

/// Build a [`Citation`] from a single evidence entry.
///
/// Centralises construction so both the UUID and legacy paths produce
/// identical shapes.
fn make_citation(index: usize, ev: &SearchResult, max_snippet_len: usize) -> Citation {
    let content = ev
        .payload
        .get("chunk_text")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let snippet = if content.chars().count() > max_snippet_len {
        format!(
            "{}...",
            truncate_to_char_boundary(content, max_snippet_len)
        )
    } else {
        content.to_owned()
    };

    let source = ev
        .payload
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();

    Citation {
        index,
        chunk_id: ev.id.clone(),
        content_snippet: snippet,
        source,
        score: ev.score.value(),
    }
}

/// Strip trailing "chatty tails" that common LLM outputs append — follow-up
/// invitations, hedged offers, "hope this helps" closers. These degrade the
/// user experience (the answer looked done, yet the model kept talking) and
/// they make the response length non-deterministic.
///
/// Matches case-insensitively at a paragraph / sentence boundary near the
/// end of the answer; the stripped region runs through end-of-string.
///
/// Multilingual: covers Romanian + English phrasings the default prompt
/// tries (but cannot guarantee) to suppress upstream.
pub(crate) fn strip_chatty_tail(answer: &str) -> String {
    static TAIL_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(
            r"(?is)\s*(?:\n{1,}|\.\s+|-\s+)(?:dacă (?:vrei|dore[șs]ti|ai nevoie|pot|ai întrebări|vrei mai multe)|pot să[-\s][șț]i|vrei să|let me know if|would you (?:like|prefer)|if you (?:want|'?d like|need)|hope (?:this|that) helps|is there anything else|feel free to|should you have).*$",
        )
        .expect("strip_chatty_tail regex is statically valid")
    });
    let stripped = TAIL_RE.replace(answer, "");
    stripped.trim_end().to_owned()
}

/// Detect whether the LLM-written answer already contains an explicit
/// "the evidence does not cover this" admission.
///
/// When this is true, the synthesizer's "I found some potentially relevant
/// information, but I'm not confident…" wrapper is redundant and produces a
/// confusing double-hedge — the user reads the model's own scope admission
/// followed by our generic hedge saying roughly the same thing.
///
/// Patterns cover EN + RO phrasings the agent is most likely to emit under
/// the system prompt's "when evidence is insufficient" clause. New phrasings
/// should be added here rather than in the caller.
pub(crate) fn answer_self_admits_scope(answer: &str) -> bool {
    // Assembled via `concat!` so each alternative lives on its own line for
    // maintainability; `r"..."` raw strings would otherwise swallow line
    // continuations as literal backslash + whitespace characters.
    static SCOPE_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        let pattern = concat!(
            r"(?i)",
            r"(?:\bnot (?:covered|mentioned|present|found) (?:in )?(?:the )?",
            r"(?:evidence|knowledge base|corpus|retrieved material)\b",
            r"|\bthe (?:retrieved )?evidence (?:does not (?:cover|mention|contain|include)",
            r"|is (?:insufficient|silent))\b",
            r"|\bno (?:relevant )?evidence (?:was )?(?:found|returned|retrieved)\b",
            r"|\b(?:nu apare|nu este (?:prezent|menționat|acoperit|inclus)",
            r"|nu se (?:regăsește|găsește)|nu conține|lipsește) ",
            r"(?:în|din) (?:evidență|evidenț[aă]|materialul? recuperat",
            r"|baza de cunoștințe|corpus)\b",
            r"|\bmaterialul (?:nu conține|nu acoperă|nu menționează)\b",
            r")",
        );
        Regex::new(pattern).expect("answer_self_admits_scope regex is statically valid")
    });
    SCOPE_RE.is_match(answer)
}

#[cfg(test)]
mod tests_cite_v2 {
    use super::*;
    use agent_rs_core::types::Score;
    use agent_rs_stores::types::SearchResult;

    /// Build a minimal `SearchResult` whose `id` and `payload.chunk_id` both
    /// equal `id`. Used to verify UUID-based citation resolution.
    fn ev_with_chunk_id(id: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score: Score::new(0.5),
            payload: serde_json::json!({
                "chunk_id": id,
                "chunk_text": "..."
            }),
        }
    }

    #[test]
    fn cite_uuid_resolves_back_to_evidence() {
        let evidence = vec![
            ev_with_chunk_id("c-1234abcd"),
            ev_with_chunk_id("c-5678ef01"),
        ];
        let answer = "Otsu separates classes [cite:c-1234abcd]. \
                      Bhattacharya is gaussian mixture-based [cite:c-5678ef01].";
        let cites = resolve_cite_markers(answer, &evidence);
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].chunk_id, "c-1234abcd");
        assert_eq!(cites[1].chunk_id, "c-5678ef01");
    }

    #[test]
    fn legacy_bracket_index_still_resolves_during_transition() {
        let evidence = vec![ev_with_chunk_id("c-1234")];
        let answer = "Otsu separates classes [1].";
        let cites = resolve_cite_markers(answer, &evidence);
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].chunk_id, "c-1234");
    }

    #[test]
    fn unknown_cite_uuid_is_dropped_with_warn() {
        let evidence = vec![ev_with_chunk_id("c-1234")];
        let answer = "Bogus reference [cite:does-not-exist].";
        let cites = resolve_cite_markers(answer, &evidence);
        assert!(cites.is_empty());
    }

    #[test]
    fn duplicate_cite_uuid_is_deduplicated() {
        let evidence = vec![ev_with_chunk_id("c-abc")];
        let answer = "First mention [cite:c-abc] and second mention [cite:c-abc].";
        let cites = resolve_cite_markers(answer, &evidence);
        assert_eq!(cites.len(), 1, "duplicate cite markers should be collapsed");
    }

    #[test]
    fn uuid_markers_take_priority_over_legacy() {
        // When BOTH [cite:...] and [N] are present, only UUID form is used.
        let evidence = vec![
            ev_with_chunk_id("id-a"),
            ev_with_chunk_id("id-b"),
        ];
        let answer = "UUID [cite:id-a] and legacy [1].";
        let cites = resolve_cite_markers(answer, &evidence);
        // Only the UUID hit should be returned (out.is_empty() guard blocks
        // the legacy path when UUID markers resolved successfully).
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].chunk_id, "id-a");
    }

    #[test]
    fn resolve_cite_markers_regex_compiles() {
        // Force both LazyLocks to initialise — turns silent panics into CI
        // failures in the same way as the regex_compile_tests module.
        let result = resolve_cite_markers("", &[]);
        assert!(result.is_empty());
    }
}

#[cfg(test)]
mod regex_compile_tests {
    //! Regex compilation smoke tests.
    //!
    //! The production regexes in this module are built behind
    //! `LazyLock` + `.expect(...)`, which means a malformed pattern
    //! does not panic at compile time or at binary startup — it
    //! panics on the first code path that evaluates it, i.e. only
    //! when a real query runs through synthesis. That failure mode
    //! is unacceptable, so these tests force every regex to be
    //! built at `cargo test` time, which turns a production crash
    //! into a CI failure.
    use super::*;

    #[test]
    fn answer_self_admits_scope_regex_compiles() {
        // Force the `LazyLock` behind `answer_self_admits_scope` to
        // initialise; this surfaces any malformed pattern here
        // instead of on the first production query.
        assert!(!answer_self_admits_scope(""));
    }

    #[test]
    fn strip_chatty_tail_regex_compiles() {
        // Same reasoning as above for `strip_chatty_tail`.
        let _ = strip_chatty_tail("");
    }

    #[test]
    fn answer_self_admits_scope_matches_ro_evidence_missing() {
        // Positive case — the Romanian admission the default prompt
        // tends to emit on out-of-corpus queries. Guards the outer
        // non-capturing group from regressing again (last incident:
        // missing trailing `)` that the unit test would have caught).
        assert!(answer_self_admits_scope(
            "Nu apare în evidență nicio mențiune despre acest subiect."
        ));
    }

    #[test]
    fn answer_self_admits_scope_matches_en_evidence_missing() {
        assert!(answer_self_admits_scope(
            "The retrieved evidence does not cover this topic."
        ));
    }

    #[test]
    fn answer_self_admits_scope_ignores_unrelated_text() {
        assert!(!answer_self_admits_scope(
            "Metoda Otsu este o metodă de segmentare."
        ));
    }
}
