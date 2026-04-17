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
        } else if answer_confidence < 0.15 {
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
