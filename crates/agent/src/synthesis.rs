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

use crate::verifier::VerificationResult;

/// A citation linking a claim to its source evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation number (1-based).
    pub index: usize,
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
    #[instrument(skip(self, answer, evidence, verification), fields(
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
    ) -> Result<SynthesizedResponse, AgentError> {
        // Clean the LLM's answer before further processing — system prompt
        // tries to suppress chatty tails, but a defence-in-depth strip keeps
        // the evidence pack deterministic even when the model drifts.
        let answer = strip_chatty_tail(answer);
        let answer = answer.as_str();

        let all_citations = self.build_citations(evidence);

        // Only include citations the LLM actually referenced in the answer
        let referenced = find_referenced_indices(answer);
        let citations = if referenced.is_empty() {
            // If no explicit references found, keep top 5 for context
            all_citations.into_iter().take(5).collect()
        } else {
            all_citations
                .into_iter()
                .filter(|c| referenced.contains(&c.index))
                .collect()
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
            // Single-hedge guard: when the LLM already admitted scope
            // ("not in the evidence" / "nu apare în evidență"), the wrapper
            // adds nothing except confusion. Let the answer stand on its
            // own honesty — the caller still sees confidence=0 and
            // is_grounded=false via the struct fields.
            if answer_self_admits_scope(answer) {
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
        })
    }

    /// Extract citations from evidence documents.
    fn build_citations(&self, evidence: &[SearchResult]) -> Vec<Citation> {
        evidence
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let content = doc
                    .payload
                    .get("chunk_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let snippet = if content.chars().count() > self.max_snippet_length {
                    format!(
                        "{}...",
                        truncate_to_char_boundary(content, self.max_snippet_length)
                    )
                } else {
                    content.to_owned()
                };

                let source = doc
                    .payload
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_owned();

                Citation {
                    index: idx + 1,
                    content_snippet: snippet,
                    source,
                    score: doc.score.value(),
                }
            })
            .collect()
    }
}

/// Extract citation indices like [1], [4] from the answer text.
fn find_referenced_indices(answer: &str) -> Vec<usize> {
    static CITATION_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\[(\d+)\]").unwrap());
    CITATION_RE
        .captures_iter(answer)
        .filter_map(|cap| cap[1].parse::<usize>().ok())
        .collect()
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
        );
        Regex::new(pattern).expect("answer_self_admits_scope regex is statically valid")
    });
    SCOPE_RE.is_match(answer)
}
