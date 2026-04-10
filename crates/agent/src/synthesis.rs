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
    /// The verifier's confidence is meta-confidence ("how certain am I about
    /// this grounding assessment"), not answer quality. When the answer is
    /// ungrounded, high verifier confidence means "I'm very sure this is bad"
    /// -- we invert for user-facing semantics. Special case: confidence 0.0
    /// means "no evidence to assess" -- keep at 0.0, don't invert to 1.0.
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

        // Compute user-facing confidence from verifier meta-confidence
        let answer_confidence = if verification.is_grounded {
            verification.confidence.value()
        } else if verification.confidence.value() < 1e-9 {
            0.0 // No evidence to assess -- unknown quality
        } else {
            1.0 - verification.confidence.value() // Invert: high verifier confidence = bad answer
        };

        let confidence = Confidence::new(answer_confidence.clamp(0.0, 1.0))
            .map_err(|e| AgentError::Synthesis(format!("confidence error: {e}")))?;

        Ok(SynthesizedResponse {
            answer: answer.to_owned(),
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
                    score: doc.score.0,
                }
            })
            .collect()
    }
}

/// Extract citation indices like [1], [4] from the answer text.
fn find_referenced_indices(answer: &str) -> Vec<usize> {
    let re = Regex::new(r"\[(\d+)\]").expect("citation regex is valid");
    re.captures_iter(answer)
        .filter_map(|cap| cap[1].parse::<usize>().ok())
        .collect()
}
