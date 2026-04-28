//! Response synthesis with citations and provenance.
//!
//! [`ResponseSynthesizer`] builds structured responses with numbered
//! citations linking answer claims to source evidence documents.
//!
//! After the deletion of the LLM-based grounding verifier (see
//! [`crate::orchestrator::AgentOrchestrator`] for the new pipeline
//! shape), the synthesizer derives `confidence` from a deterministic
//! token-overlap heuristic between the answer and the evidence pack.
//! No LLM calls happen here. Implicit grounding is enforced by the
//! synthesis-stage prompt's "only claim what's in evidence" contract,
//! not by a post-hoc verifier.

use std::collections::HashSet;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{Confidence, RetrievalStrategy, truncate_to_char_boundary};
use agent_rs_stores::types::SearchResult;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use unicode_segmentation::UnicodeSegmentation;

use crate::coverage::CoverageReport;

/// Minimum length (in Unicode scalar values) for a token to enter the
/// answer/evidence overlap calculation.
///
/// Anything shorter is dropped regardless of language. This filters
/// out mono-/bi-character function words across scripts (Romance
/// articles, English particles, CJK fillers) without a per-language
/// stopword list, keeping the heuristic universal by construction.
const HEURISTIC_MIN_TOKEN_CHARS: usize = 3;

/// Confidence below which the synthesizer wraps the answer with a
/// "please verify independently" caveat. Tuned alongside the heuristic
/// confidence range so genuinely poorly-supported answers carry an
/// explicit warning while well-supported answers stand on their own.
const LOW_CONFIDENCE_HEDGE_THRESHOLD: f64 = 0.15;

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
    /// MinIO object key of the source document (stateless pivot: replaces
    /// the prior opaque `source` field — the web app resolves keys to
    /// human-readable titles via its own `ingested_documents` table).
    pub minio_object_key: String,
    /// Human-readable section path such as `"5.1.2 / Region growing"`,
    /// populated when Neo4j knows the enclosing section.
    pub section_path: String,
    /// Relevance score from retrieval.
    pub score: f64,
}

/// Final agent response with citations and provenance.
///
/// The previous verifier-driven `is_grounded` / `claim_confidences` /
/// `ungrounded_claims` fields were dropped when the LLM grounding gate
/// was removed. Coverage callers continue to read [`Self::coverage_report`]
/// for exhaustive-flow signals; non-exhaustive consumers rely on
/// [`Self::confidence`] (now a deterministic answer-vs-evidence token
/// overlap) plus the answer prose itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedResponse {
    /// The synthesized answer text.
    pub answer: String,
    /// Source citations supporting the answer.
    pub citations: Vec<Citation>,
    /// Original user query.
    pub query: String,
    /// Heuristic confidence in `[0.0, 1.0]`, computed as the
    /// case-folded Unicode-word token overlap between the answer and
    /// the evidence pack.
    ///
    /// **Semantics**: `confidence = |answer_tokens ∩ evidence_tokens| /
    /// |answer_tokens|`, where tokens are
    /// [`unicode_segmentation::UnicodeSegmentation::unicode_words`]
    /// outputs of length `>= 3` Unicode scalars. Defaults to `0.0`
    /// when either the evidence pack or the answer-token set is
    /// empty. Cheap (sub-millisecond), deterministic, and language-
    /// agnostic by construction. **Not** an LLM-derived faithfulness
    /// score — the synthesis prompt enforces grounding implicitly;
    /// this number is a side-channel quality signal for dashboards
    /// and out-of-corpus detection.
    pub confidence: Confidence,
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
    /// Number of top-evidence items used as fallback citations when the
    /// LLM produced no inline `[cite:<id>]` markers.
    fallback_citation_count: usize,
}

impl ResponseSynthesizer {
    /// Create a new synthesizer.
    pub fn new(max_snippet_length: usize, fallback_citation_count: usize) -> Self {
        Self {
            max_snippet_length,
            fallback_citation_count,
        }
    }

    /// Build a response with citations from evidence documents.
    ///
    /// # Confidence semantics
    ///
    /// `confidence` is computed inline as the case-folded
    /// Unicode-word token overlap between `answer` and `evidence`:
    ///
    /// ```text
    /// confidence = |answer_tokens ∩ evidence_tokens| / |answer_tokens|
    /// ```
    ///
    /// Tokens are
    /// [`unicode_segmentation::UnicodeSegmentation::unicode_words`]
    /// outputs filtered to length ≥
    /// [`HEURISTIC_MIN_TOKEN_CHARS`] Unicode scalars. The metric is
    /// deterministic, language-agnostic, and runs in microseconds —
    /// it replaces the previous LLM-verifier-driven score.
    ///
    /// Returns `0.0` whenever either token set is empty (zero
    /// evidence, refusal-only answer). Capped at `1.0`.
    #[instrument(skip(self, answer, evidence, coverage_report), fields(
        evidence_count = evidence.len(),
        strategy = %strategy,
    ))]
    pub fn synthesize(
        &self,
        answer: &str,
        evidence: &[SearchResult],
        query: &str,
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
        // When the LLM cited nothing, fall back to the top-N evidence items
        // (N = `fallback_citation_count`).
        let citations = {
            let resolved = resolve_cite_markers(answer, evidence);
            if resolved.is_empty() {
                evidence
                    .iter()
                    .enumerate()
                    .take(self.fallback_citation_count)
                    .map(|(i, ev)| make_citation(i + 1, ev, self.max_snippet_length))
                    .collect()
            } else {
                resolved
            }
        };

        let answer_confidence = heuristic_confidence(answer, evidence);
        let confidence = Confidence::new(answer_confidence)
            .map_err(|e| AgentError::Synthesis(format!("confidence error: {e}")))?;

        // "I don't know" boundary: when evidence is empty, replace the
        // synthesized answer with an honest admission rather than
        // risking hallucination. The synthesis prompt itself is supposed
        // to refuse on zero-evidence input — this is a defence in depth.
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
        } else if answer_confidence < LOW_CONFIDENCE_HEDGE_THRESHOLD {
            // Low-overlap path: the answer barely mentions vocabulary
            // from the evidence, suggesting either an out-of-corpus
            // hedge or a tangential answer. Skip the wrapper when:
            //   - coverage_report is present (callers read the report
            //     directly; doubling up would confuse the UX), or
            //   - the model already self-admitted scope ("not in the
            //     evidence" / "nu apare în evidență").
            if coverage_report.is_some() {
                tracing::info!(
                    query,
                    confidence = answer_confidence,
                    strategy = %strategy,
                    "low-confidence answer with coverage_report present — skipping wrapper"
                );
                answer.to_owned()
            } else if answer_self_admits_scope(answer) {
                tracing::info!(
                    query,
                    confidence = answer_confidence,
                    strategy = %strategy,
                    "low-confidence answer self-admits scope — skipping wrapper"
                );
                answer.to_owned()
            } else {
                tracing::warn!(
                    query,
                    confidence = answer_confidence,
                    strategy = %strategy,
                    "very low confidence answer — caveat added"
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
            strategy,
            coverage_report: coverage_report.cloned(),
        })
    }
}

/// Compute the answer-vs-evidence token-overlap confidence.
///
/// Both `answer` and `evidence` chunk bodies are tokenised with
/// [`UnicodeSegmentation::unicode_words`] (universal across scripts —
/// no per-language stopword list), case-folded, and filtered to a
/// minimum of [`HEURISTIC_MIN_TOKEN_CHARS`] Unicode scalars to drop
/// function words / particles.
///
/// Returns `0.0` when either token set is empty. Otherwise:
///
/// ```text
/// |answer_tokens ∩ evidence_tokens| / |answer_tokens|
/// ```
///
/// The result is clamped to `[0.0, 1.0]`. The function is `pub(crate)`
/// for unit testability — callers use [`ResponseSynthesizer::synthesize`]
/// rather than calling this directly.
pub(crate) fn heuristic_confidence(answer: &str, evidence: &[SearchResult]) -> f64 {
    let answer_tokens: HashSet<String> = answer
        .to_lowercase()
        .unicode_words()
        .filter(|t| t.chars().count() >= HEURISTIC_MIN_TOKEN_CHARS)
        .map(|s| s.to_owned())
        .collect();

    if answer_tokens.is_empty() || evidence.is_empty() {
        return 0.0;
    }

    let mut evidence_tokens: HashSet<String> = HashSet::new();
    for doc in evidence {
        let text = doc
            .payload
            .get("chunk_text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let lowered = text.to_lowercase();
        for token in lowered.unicode_words() {
            if token.chars().count() >= HEURISTIC_MIN_TOKEN_CHARS {
                evidence_tokens.insert(token.to_owned());
            }
        }
    }

    let overlap = answer_tokens.intersection(&evidence_tokens).count();
    (overlap as f64 / answer_tokens.len() as f64).clamp(0.0, 1.0)
}

/// Build a [`SynthesizedResponse`] for the pre-synth subject-absent
/// short-circuit path.
///
/// Returns a polite "topic out of corpus" answer with `confidence =
/// 0.0`, no citations, and the same `coverage_report` the caller would
/// have observed otherwise. Used by the orchestrator when
/// [`crate::subject_check::subject_absent_v2`] flags the query as
/// out-of-corpus before the synthesis LLM call — saving the wasted
/// round-trip while preserving the response shape downstream
/// consumers (gRPC server, web app) expect.
///
/// # Errors
///
/// Returns [`AgentError::Synthesis`] only if the (statically zero)
/// confidence value somehow fails [`Confidence::new`] — defensive
/// path; the typed domain accepts `0.0`.
pub fn out_of_corpus_response(
    query: &str,
    strategy: RetrievalStrategy,
    coverage_report: Option<&CoverageReport>,
) -> Result<SynthesizedResponse, AgentError> {
    let confidence = Confidence::new(0.0)
        .map_err(|e| AgentError::Synthesis(format!("confidence error: {e}")))?;
    Ok(SynthesizedResponse {
        answer: "I don't have enough information in my knowledge base to answer \
                 this question reliably. The retrieved evidence does not cover \
                 this topic."
            .to_owned(),
        citations: Vec::new(),
        query: query.to_owned(),
        confidence,
        strategy,
        coverage_report: coverage_report.cloned(),
    })
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
        Regex::new(r"\[(\d+)\]").expect("resolve_cite_markers index regex is statically valid")
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
            e.id == id || e.payload.get("chunk_id").and_then(|v| v.as_str()) == Some(id)
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
        format!("{}...", truncate_to_char_boundary(content, max_snippet_len))
    } else {
        content.to_owned()
    };

    let minio_object_key = ev
        .payload
        .get("minio_object_key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    let section_path = ev
        .payload
        .get("section_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    Citation {
        index,
        chunk_id: ev.id.clone(),
        content_snippet: snippet,
        minio_object_key,
        section_path,
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
        let evidence = vec![ev_with_chunk_id("id-a"), ev_with_chunk_id("id-b")];
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

#[cfg(test)]
mod tests_heuristic_confidence {
    //! Tests for the deterministic answer-vs-evidence token overlap
    //! that powers [`SynthesizedResponse::confidence`]. The heuristic
    //! replaces the deleted LLM grounding verifier — these tests pin
    //! the boundary cases (empty answer, empty evidence, partial /
    //! full overlap) so a future tokenizer change cannot silently
    //! regress confidence semantics.
    use super::*;
    use agent_rs_core::types::Score;

    fn ev(text: &str) -> SearchResult {
        SearchResult {
            id: String::new(),
            score: Score::new(0.0),
            payload: serde_json::json!({"chunk_text": text}),
        }
    }

    #[test]
    fn empty_evidence_yields_zero_confidence() {
        assert_eq!(heuristic_confidence("anything goes here", &[]), 0.0);
    }

    #[test]
    fn empty_answer_yields_zero_confidence() {
        assert_eq!(heuristic_confidence("", &[ev("anything")]), 0.0);
    }

    #[test]
    fn full_overlap_yields_one() {
        let evidence = vec![ev("metoda otsu segmentare histogramă")];
        // Answer uses every meaningful token from evidence — should max out.
        let c = heuristic_confidence("metoda otsu segmentare histogramă", &evidence);
        assert!((c - 1.0).abs() < f64::EPSILON, "got {c}");
    }

    #[test]
    fn partial_overlap_returns_proportional_value() {
        let evidence = vec![ev("metoda otsu segmentare")];
        // Answer has 4 ≥3-char tokens; 3 of them ("metoda", "otsu",
        // "segmentare") overlap with evidence. Confidence ≈ 0.75.
        let c = heuristic_confidence("metoda otsu segmentare oranj", &evidence);
        assert!((c - 0.75).abs() < 1e-9, "expected 0.75, got {c}");
    }

    #[test]
    fn short_tokens_below_threshold_are_dropped() {
        // The answer "no" has zero tokens ≥3 chars, so the overlap
        // calculation falls into the empty-answer branch → 0.0.
        let evidence = vec![ev("no")];
        assert_eq!(heuristic_confidence("no", &evidence), 0.0);
    }

    #[test]
    fn confidence_is_clamped_into_unit_range() {
        // Sanity: ratio is bounded by construction; no input shape
        // can make it exceed 1.0.
        let evidence = vec![ev("alpha alpha alpha alpha")];
        let c = heuristic_confidence("alpha", &evidence);
        assert!((0.0..=1.0).contains(&c));
    }
}

#[cfg(test)]
mod tests_synthesize {
    //! Integration tests on the new (verifier-less) synthesize path.
    use super::*;
    use agent_rs_core::types::{RetrievalStrategy, Score};

    fn ev(id: &str, text: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score: Score::new(0.5),
            payload: serde_json::json!({
                "chunk_id": id,
                "chunk_text": text,
            }),
        }
    }

    #[test]
    fn empty_evidence_emits_idk_answer() {
        let synth = ResponseSynthesizer::new(200, 5);
        let resp = synth
            .synthesize("anything", &[], "q", RetrievalStrategy::Semantic, None)
            .expect("must build");
        assert!(resp.answer.contains("don't have enough information"));
        assert_eq!(resp.confidence.value(), 0.0);
        assert!(resp.citations.is_empty());
    }

    #[test]
    fn high_overlap_answer_passes_through_unchanged() {
        let synth = ResponseSynthesizer::new(200, 5);
        let evidence = vec![ev("c-1", "metoda otsu segmentare histogramă")];
        let answer = "Metoda Otsu segmentare histogramă [cite:c-1].";
        let resp = synth
            .synthesize(answer, &evidence, "q", RetrievalStrategy::Semantic, None)
            .expect("must build");
        // No "please verify independently" wrapper for high-overlap.
        assert!(!resp.answer.contains("please verify independently"));
        assert!(resp.confidence.value() > LOW_CONFIDENCE_HEDGE_THRESHOLD);
    }

    #[test]
    fn low_overlap_answer_with_self_admit_skips_wrapper() {
        let synth = ResponseSynthesizer::new(200, 5);
        let evidence = vec![ev("c-1", "totally unrelated content")];
        let answer = "The retrieved evidence does not cover this topic.";
        let resp = synth
            .synthesize(answer, &evidence, "q", RetrievalStrategy::Semantic, None)
            .expect("must build");
        // Self-admit guard: no double-hedge.
        assert!(!resp.answer.contains("please verify independently"));
    }

    #[test]
    fn out_of_corpus_response_has_zero_confidence_and_no_citations() {
        let resp =
            out_of_corpus_response("q", RetrievalStrategy::Semantic, None).expect("must build");
        assert_eq!(resp.confidence.value(), 0.0);
        assert!(resp.citations.is_empty());
        assert!(resp.answer.contains("does not cover"));
    }
}
