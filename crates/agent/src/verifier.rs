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
use tracing::{info, instrument};

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
    /// Per-claim confidence breakdown (spec C3).
    ///
    /// When populated, each individual claim in the answer has its own
    /// confidence score and source citation list.  Clients can display
    /// this to users so they see exactly which parts to trust.
    #[serde(default)]
    pub claim_confidences: Vec<ClaimConfidence>,
    /// Verification method used: "llm", "heuristic", or "heuristic_fallback".
    ///
    /// "heuristic_fallback" indicates the LLM verifier was configured but
    /// failed, so results are from the lower-quality heuristic path.
    #[serde(default = "default_method")]
    pub method: String,
}

fn default_method() -> String {
    "heuristic".to_string()
}

/// Per-claim confidence tracking (spec C3).
///
/// Each claim extracted from the answer is scored independently against
/// the evidence, letting users see exactly which parts are trustworthy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimConfidence {
    /// The claim text (a single statement from the answer).
    pub claim: String,
    /// Confidence score for this specific claim (0.0 – 1.0).
    pub confidence: f64,
    /// Whether this claim is grounded in evidence.
    pub grounded: bool,
    /// Citation indices (1-based) from the evidence pack that support this claim.
    pub source_indices: Vec<usize>,
}

/// Kind of a single claim extracted by the verifier.
///
/// The LLM classifies each atomic statement in the answer into one of
/// these categories; the Rust code then applies the grounding rule
/// **deterministically** based on the label, never on language-specific
/// lexical patterns. This keeps the verifier corpus-agnostic (works for
/// Romanian, English, Spanish, Turkish, …) because the semantic work of
/// deciding "is this sentence a factual claim about the document or is
/// it the model talking about itself" is done once by a capable model,
/// and consumed by tiny switch statements in typed code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimType {
    /// A concrete assertion about the subject matter (a formula, a
    /// definition, a quoted passage, a numeric value from the document).
    /// Only `Factual` claims affect `is_grounded`.
    Factual,
    /// A description of the answering process itself:
    /// "I verified the sources", "Am căutat în material", "I reviewed
    /// the retrieved chunks". Not a claim about the document — ignored.
    Process,
    /// A reference to a prior draft / iteration of the answer:
    /// "Compared to my earlier response", "Ce am eliminat din răspunsul
    /// anterior", "Correction: replace X with Y". Ignored.
    SelfReference,
    /// A remark about citation conventions / formatting rather than
    /// about the document's content: "The citation IDs do not match",
    /// "I used `[cite:<id>]` markers". Ignored.
    CitationMeta,
    /// An explicit hedge about completeness / exhaustiveness:
    /// "Cannot confirm this list is exhaustive", "Nu pot confirma
    /// exhaustivitatea". Ignored.
    Hedge,
}

/// One classified claim produced by the verifier.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct VerifierClaim {
    /// The claim text, copied verbatim from the answer.
    text: String,
    /// Category the LLM assigned to this claim (see [`ClaimType`]).
    #[serde(rename = "type")]
    claim_type: ClaimType,
    /// Whether the claim is supported by the evidence. Must be set for
    /// `Factual` claims; `None` is acceptable for every other type (they
    /// do not affect grounding and the LLM may leave this field out).
    #[serde(default)]
    supported: Option<bool>,
}

/// Raw JSON structure expected from the LLM verifier response.
///
/// `is_grounded` is derived in Rust from the `claims` array (any
/// `Factual` claim with `supported=false` renders the answer ungrounded),
/// so we do NOT ask the LLM for a pre-computed verdict — the LLM only
/// classifies; the code decides. This moves the grounding rule out of
/// prompt text (opaque, model-dependent) into typed Rust (testable,
/// corpus-agnostic).
#[derive(Debug, Deserialize)]
struct VerifierLlmResponse {
    /// Verifier's overall confidence in its classification (0.0–1.0).
    /// Used as `VerificationResult.confidence` when grounded; when
    /// ungrounded the user-facing confidence is clamped to 0 by the
    /// synthesizer anyway, so this value's influence is limited.
    confidence: f64,
    /// Short prose explanation — surfaced in logs only.
    reasoning: String,
    /// Classified atomic claims. Empty list means "no factual claims in
    /// the answer" (e.g. a pure refusal), which is trivially grounded.
    #[serde(default)]
    claims: Vec<VerifierClaim>,
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
                    tracing::error!(
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
                claim_confidences: Vec::new(),
                method: "none".to_string(),
            });
        }

        if evidence.is_empty() {
            return Ok(VerificationResult {
                is_grounded: false,
                confidence: Confidence::new(0.0)
                    .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?,
                reasoning: "No evidence provided for grounding check.".into(),
                ungrounded_claims: Vec::new(),
                claim_confidences: Vec::new(),
                method: "none".to_string(),
            });
        }

        if let Some(client) = &self.llm_client {
            match self.verify_with_llm(client, answer, evidence).await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    tracing::error!(
                        error = %err,
                        "LLM verification failed, falling back to heuristic — quality degraded"
                    );
                    // Fall through to heuristic with "heuristic_fallback" method tag
                    let mut result = self.verify_heuristic(answer, evidence)?;
                    result.method = "heuristic_fallback".to_string();
                    return Ok(result);
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

        // Emit each evidence block prefixed with the exact `[cite:<chunk_id>]`
        // marker the answer is expected to use. Using a plain `[N]` index
        // here (the previous behaviour) desynchronised the verifier from
        // the prompt/synthesis contract — UUID citations in the answer
        // looked "ungrounded" to the verifier even though the content was
        // present, because the verifier never saw the UUIDs at all.
        let evidence_text: String = evidence
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let content = doc
                    .payload
                    .get("chunk_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let chunk_id = doc
                    .payload
                    .get("chunk_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(doc.id.as_str());
                format!("[{}] [cite:{}] {}", i + 1, chunk_id, content)
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

        // Guard against NaN/non-finite from LLM output before clamp (f64::clamp panics on NaN).
        let confidence_val = if parsed.confidence.is_finite() {
            parsed.confidence.clamp(0.0, 1.0)
        } else {
            tracing::warn!(
                raw = parsed.confidence,
                "LLM verifier returned non-finite confidence, coercing to 0.0"
            );
            0.0
        };
        let confidence = Confidence::new(confidence_val)
            .map_err(|e| AgentError::Verification(format!("confidence error: {e}")))?;

        // Derive the grounding verdict from the typed claims array.
        // Rule: `is_grounded = true` iff every `Factual` claim is
        // `supported == Some(true)`. Every other claim type (Process,
        // SelfReference, CitationMeta, Hedge) is definitionally not a
        // statement about the document and cannot render the answer
        // ungrounded — its `supported` field is ignored.
        //
        // This moves the grounding rule out of prompt text into typed
        // Rust code so it's language-agnostic (the LLM does classification,
        // we do logic) and unit-testable.
        let is_grounded = claims_are_grounded(&parsed.claims);

        // Extract the subset of factual, unsupported claims to keep
        // filling the back-compat `ungrounded_claims: Vec<String>`
        // field on `VerificationResult` — downstream code (the retry
        // prompt in `grounding_gate`) reads that list to produce
        // targeted feedback. Only genuine factual failures end up here.
        let ungrounded_claims: Vec<String> = parsed
            .claims
            .iter()
            .filter(|c| c.claim_type == ClaimType::Factual && c.supported == Some(false))
            .map(|c| c.text.clone())
            .collect();

        Ok(VerificationResult {
            is_grounded,
            confidence,
            reasoning: parsed.reasoning,
            ungrounded_claims,
            claim_confidences: Vec::new(),
            method: "llm".to_string(),
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
                claim_confidences: Vec::new(),
                method: "heuristic".to_string(),
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
        // answer_tokens.is_empty() handled above, so division is safe.
        let confidence_val = (overlap as f64 / answer_tokens.len() as f64).clamp(0.0, 1.0);
        let is_grounded = confidence_val >= self.confidence_threshold;

        let confidence = Confidence::new(confidence_val)
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
            claim_confidences: Vec::new(),
            method: "heuristic".to_string(),
        })
    }
}

/// Decide whether the answer is grounded based on classified claims.
///
/// Rule: the answer is grounded iff every [`ClaimType::Factual`] claim
/// has `supported == Some(true)`. Claims of any other type never block
/// grounding — that's the whole point of classification: meta / process /
/// self-reference statements are not propositions about the document and
/// therefore cannot be "ungrounded with respect to the document".
///
/// An empty `claims` array is trivially grounded (no factual assertions
/// to verify — a refusal or pure-meta answer).
fn claims_are_grounded(claims: &[VerifierClaim]) -> bool {
    claims
        .iter()
        .filter(|c| c.claim_type == ClaimType::Factual)
        .all(|c| c.supported == Some(true))
}

#[cfg(test)]
mod tests_typed_claims {
    use super::*;

    fn claim(text: &str, claim_type: ClaimType, supported: Option<bool>) -> VerifierClaim {
        VerifierClaim {
            text: text.to_string(),
            claim_type,
            supported,
        }
    }

    #[test]
    fn empty_claims_are_grounded() {
        assert!(claims_are_grounded(&[]));
    }

    #[test]
    fn all_factual_supported_is_grounded() {
        let claims = vec![
            claim(
                "ε = P₀σ₀² + P₁σ₁² appears in (5.14)",
                ClaimType::Factual,
                Some(true),
            ),
            claim(
                "σ_k = √(1/|m_k|) appears in (5.9)",
                ClaimType::Factual,
                Some(true),
            ),
        ];
        assert!(claims_are_grounded(&claims));
    }

    #[test]
    fn one_factual_unsupported_breaks_grounding() {
        let claims = vec![
            claim("ε = P₀σ₀² + P₁σ₁²", ClaimType::Factual, Some(true)),
            claim(
                "FCM folosește explicit varianța",
                ClaimType::Factual,
                Some(false),
            ),
        ];
        assert!(!claims_are_grounded(&claims));
    }

    #[test]
    fn meta_claims_never_break_grounding() {
        // All five non-Factual types present; even if "supported" is
        // None or false, the answer stays grounded because none of
        // these are propositions about the document.
        let claims = vec![
            claim("ε = P₀σ₀² + P₁σ₁²", ClaimType::Factual, Some(true)),
            claim("I verified the sources", ClaimType::Process, None),
            claim(
                "Ce am eliminat din răspunsul anterior…",
                ClaimType::SelfReference,
                None,
            ),
            claim(
                "Citation IDs don't match",
                ClaimType::CitationMeta,
                Some(false),
            ),
            claim("Cannot confirm exhaustive", ClaimType::Hedge, None),
        ];
        assert!(claims_are_grounded(&claims));
    }

    #[test]
    fn factual_with_missing_supported_is_ungrounded() {
        // Defensive: if the LLM forgets to set `supported` on a factual
        // claim, we treat it as ungrounded (safer than the opposite —
        // an unreviewed claim is not a verified claim).
        let claims = vec![claim("Some factual statement", ClaimType::Factual, None)];
        assert!(!claims_are_grounded(&claims));
    }

    #[test]
    fn deserializes_typed_claim_json() {
        // End-to-end serde test: the JSON the LLM emits must round-trip
        // into VerifierClaim + ClaimType enum via serde(rename_all).
        let json = r#"{
            "text": "T = (μ0+μ1)/2 - (σ²/(μ0-μ1)) ln(P0/P1)",
            "type": "factual",
            "supported": true
        }"#;
        let parsed: VerifierClaim = serde_json::from_str(json).expect("must parse");
        assert_eq!(parsed.claim_type, ClaimType::Factual);
        assert_eq!(parsed.supported, Some(true));
    }

    #[test]
    fn deserializes_snake_case_claim_types() {
        for (label, ty) in [
            ("factual", ClaimType::Factual),
            ("process", ClaimType::Process),
            ("self_reference", ClaimType::SelfReference),
            ("citation_meta", ClaimType::CitationMeta),
            ("hedge", ClaimType::Hedge),
        ] {
            let json = format!(r#"{{"text":"x","type":"{label}"}}"#);
            let parsed: VerifierClaim =
                serde_json::from_str(&json).expect("every variant must parse");
            assert_eq!(parsed.claim_type, ty);
            assert_eq!(parsed.supported, None, "`supported` is optional");
        }
    }
}
