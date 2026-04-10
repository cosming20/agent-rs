//! Faithfulness and grounding verification for agent responses.
//!
//! [`GroundingVerifier`] checks whether an agent's answer is faithfully
//! grounded in retrieved evidence. Supports LLM-based verification
//! (via rig-core) with a heuristic token-overlap fallback.

use std::collections::HashSet;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::Confidence;
use agent_rs_stores::types::SearchResult;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::prompts::VERIFICATION_PROMPT;

/// Result of a faithfulness/grounding check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the answer is grounded in the evidence.
    pub is_grounded: bool,
    /// Confidence score between 0.0 and 1.0.
    pub confidence: Confidence,
    /// Explanation of the verification decision.
    pub reasoning: String,
    /// Claims in the answer not supported by evidence.
    pub ungrounded_claims: Vec<String>,
}

/// Raw JSON structure expected from the LLM verifier response.
#[derive(Debug, Deserialize)]
struct VerifierLlmResponse {
    is_grounded: bool,
    confidence: f64,
    reasoning: String,
    #[serde(default)]
    ungrounded_claims: Vec<String>,
}

/// Stop words excluded from heuristic token overlap scoring.
///
/// Common English function words that carry no semantic weight for
/// grounding assessment.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "while", "about", "up", "its", "it", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom",
];

/// Verifies that an answer is faithfully grounded in retrieved evidence.
///
/// Uses LLM-based evaluation via rig-core when a provider client is
/// available, falling back to heuristic token overlap otherwise.
pub struct GroundingVerifier {
    /// Minimum confidence to consider the answer grounded.
    confidence_threshold: f64,
    /// Optional rig-core client for LLM-based verification.
    /// When `None`, uses heuristic fallback only.
    llm_client: Option<rig::providers::openai::Client>,
    /// Model to use for LLM verification.
    model: String,
}

impl GroundingVerifier {
    /// Create a new verifier.
    ///
    /// # Parameters
    ///
    /// - `confidence_threshold` -- Minimum confidence for grounded verdict.
    /// - `api_key` -- OpenAI API key. If empty, uses heuristic only.
    /// - `model` -- Model for verification (e.g., "gpt-4o-mini").
    pub fn new(confidence_threshold: f64, api_key: &str, model: &str) -> Self {
        let llm_client = if api_key.is_empty() {
            None
        } else {
            match rig::providers::openai::Client::new(api_key) {
                Ok(client) => Some(client),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "failed to create OpenAI client for verifier, using heuristic only"
                    );
                    None
                }
            }
        };

        Self {
            confidence_threshold,
            llm_client,
            model: model.to_owned(),
        }
    }

    /// Check if the answer is grounded in the provided evidence.
    ///
    /// Uses LLM-based evaluation if configured, otherwise heuristic.
    #[instrument(skip(self, answer, evidence), fields(evidence_count = evidence.len()))]
    pub async fn verify(
        &self,
        answer: &str,
        evidence: &[SearchResult],
    ) -> Result<VerificationResult, AgentError> {
        if answer.is_empty() {
            return Ok(VerificationResult {
                is_grounded: false,
                confidence: Confidence::new(0.0)
                    .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?,
                reasoning: "Empty answer cannot be verified.".into(),
                ungrounded_claims: Vec::new(),
            });
        }

        if evidence.is_empty() {
            return Ok(VerificationResult {
                is_grounded: false,
                confidence: Confidence::new(0.0)
                    .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?,
                reasoning: "No evidence provided for grounding check.".into(),
                ungrounded_claims: Vec::new(),
            });
        }

        if let Some(client) = &self.llm_client {
            match self.verify_with_llm(client, answer, evidence).await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    warn!(
                        error = %err,
                        "LLM verification failed, falling back to heuristic"
                    );
                }
            }
        }

        self.verify_heuristic(answer, evidence)
    }

    /// Determine if the answer should trigger a retry.
    pub fn needs_retry(&self, result: &VerificationResult) -> bool {
        !result.is_grounded
    }

    /// LLM-based faithfulness verification via rig-core.
    async fn verify_with_llm(
        &self,
        client: &rig::providers::openai::Client,
        answer: &str,
        evidence: &[SearchResult],
    ) -> Result<VerificationResult, AgentError> {
        use rig::client::CompletionClient;
        use rig::completion::Prompt;

        let evidence_text: String = evidence
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let content = doc
                    .payload
                    .get("chunk_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                format!("[{}] {}", i + 1, content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let user_message =
            format!("<evidence>\n{evidence_text}\n</evidence>\n\n<answer>\n{answer}\n</answer>");

        let agent = client
            .agent(&self.model)
            .preamble(VERIFICATION_PROMPT)
            .build();

        let response_text = agent
            .prompt(&user_message)
            .await
            .map_err(|e| AgentError::Verification(format!("LLM verifier call failed: {e}")))?;

        let parsed: VerifierLlmResponse = serde_json::from_str(&response_text).map_err(|e| {
            AgentError::Verification(format!(
                "LLM verifier JSON parse failed: {e} -- raw: {response_text}"
            ))
        })?;

        let confidence_val = parsed.confidence.clamp(0.0, 1.0);
        let confidence = Confidence::new(confidence_val)
            .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?;

        Ok(VerificationResult {
            is_grounded: parsed.is_grounded,
            confidence,
            reasoning: parsed.reasoning,
            ungrounded_claims: parsed.ungrounded_claims,
        })
    }

    /// Heuristic token overlap verification (fallback).
    fn verify_heuristic(
        &self,
        answer: &str,
        evidence: &[SearchResult],
    ) -> Result<VerificationResult, AgentError> {
        let stop_words: HashSet<&str> = STOP_WORDS.iter().copied().collect();

        let answer_tokens: HashSet<String> = answer
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_owned())
            .filter(|t| !stop_words.contains(t.as_str()))
            .collect();

        if answer_tokens.is_empty() {
            return Ok(VerificationResult {
                is_grounded: false,
                confidence: Confidence::new(0.0)
                    .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?,
                reasoning: "Answer contains no meaningful tokens.".into(),
                ungrounded_claims: Vec::new(),
            });
        }

        let mut evidence_tokens: HashSet<String> = HashSet::new();
        for doc in evidence {
            let text = doc
                .payload
                .get("chunk_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            for token in text.to_lowercase().split_whitespace() {
                if !stop_words.contains(token) {
                    evidence_tokens.insert(token.to_owned());
                }
            }
        }

        let overlap = answer_tokens.intersection(&evidence_tokens).count();
        let confidence_val = overlap as f64 / answer_tokens.len() as f64;
        let is_grounded = confidence_val >= self.confidence_threshold;

        let confidence = Confidence::new(confidence_val.clamp(0.0, 1.0))
            .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?;

        info!(
            overlap = overlap,
            total = answer_tokens.len(),
            confidence = confidence_val,
            threshold = self.confidence_threshold,
            is_grounded = is_grounded,
            "heuristic verification complete"
        );

        Ok(VerificationResult {
            is_grounded,
            confidence,
            reasoning: format!(
                "Heuristic: token overlap {overlap}/{} ({:.1}%). Threshold: {:.1}%.",
                answer_tokens.len(),
                confidence_val * 100.0,
                self.confidence_threshold * 100.0,
            ),
            ungrounded_claims: Vec::new(),
        })
    }
}
