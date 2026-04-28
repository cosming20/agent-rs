//! Faithfulness and grounding verification for agent responses.
//!
//! [`GroundingVerifier`] checks whether an agent's answer is faithfully
//! grounded in retrieved evidence. Supports LLM-based verification
//! (via rig-core) with a heuristic token-overlap fallback.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::Confidence;
use agent_rs_observability::budget::RequestBudget;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};
use unicode_segmentation::UnicodeSegmentation;

use crate::agent_hook::{AgentRsHook, COMPONENT_VERIFIER, token_usage_from_rig};
use crate::prompts::VERIFICATION_PROMPT;

/// Minimum length (in Unicode scalar values) for a token to enter the
/// heuristic verifier's overlap calculation.
///
/// Anything shorter is dropped regardless of language. This filters out
/// mono-/bi-character function words across scripts (Romance articles,
/// English particles, CJK fillers) without a per-language stopword list,
/// keeping the heuristic universal by construction.
const HEURISTIC_MIN_TOKEN_CHARS: usize = 3;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
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
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
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

/// Typed structure the verifier asks the model to emit.
///
/// `is_grounded` is derived in Rust from the `claims` array (any
/// `Factual` claim with `supported=false` renders the answer ungrounded),
/// so we do NOT ask the LLM for a pre-computed verdict — the LLM only
/// classifies; the code decides. This moves the grounding rule out of
/// prompt text (opaque, model-dependent) into typed Rust (testable,
/// corpus-agnostic).
///
/// The `JsonSchema` derive lets us call `agent.prompt_typed::<T>(...)`,
/// which routes the type's schema into the provider's structured-
/// output mode (OpenAI returns the JSON directly conformant to the
/// schema). This replaces the previous flow that asked the model for
/// JSON in the prompt and then `serde_json::from_str` at the call
/// site — that path was brittle: stray prose, code-fence wrappers,
/// and trailing commentary all broke parsing. The typed path either
/// deserialises cleanly or surfaces a [`StructuredOutputError`] from
/// rig.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
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
    /// Optional shared metrics handle.
    ///
    /// When `Some`, the LLM verifier:
    ///
    /// 1. Records the wall-clock duration of every `verify_with_llm`
    ///    call into `llm.verifier_duration_ms`.
    /// 2. Attaches an [`AgentRsHook`] to the `prompt_typed` call so
    ///    every model round-trip flows through
    ///    [`RetrievalMetrics::record_llm_call`] (per-call latency +
    ///    tokens) and [`RequestBudget::record_call`] (cumulative
    ///    request totals + cost). The heuristic path is intentionally
    ///    not instrumented — it runs in microseconds and emits no
    ///    LLM tokens.
    metrics: Option<Arc<RetrievalMetrics>>,
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
            metrics: None,
        }
    }

    /// Attach a shared metrics handle for LLM verifier latency emission.
    ///
    /// When set, [`Self::verify_with_llm`] records the wall-clock
    /// duration of every call into `llm.verifier_duration_ms`. The
    /// heuristic path stays untimed — it runs in microseconds and
    /// would only add noise to the histogram.
    ///
    /// Returns the verifier by value to support builder-style
    /// construction.
    pub fn with_metrics(mut self, metrics: Arc<RetrievalMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Check if the answer is grounded in the provided evidence.
    ///
    /// Uses LLM-based evaluation if configured, otherwise heuristic.
    ///
    /// `budget` is the per-request token + cost aggregator. The LLM
    /// path attaches an [`AgentRsHook`] that calls
    /// [`RequestBudget::record_call`] for every model round-trip; the
    /// heuristic path leaves the budget untouched (no LLM tokens
    /// consumed). Pass `&ctx.budget` from the grounding gate.
    #[instrument(skip(self, answer, evidence, budget), fields(evidence_count = evidence.len()))]
    pub async fn verify(
        &self,
        answer: &str,
        evidence: &[SearchResult],
        budget: &Arc<RequestBudget>,
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
            match self.verify_with_llm(client, answer, evidence, budget).await {
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
    ///
    /// Uses [`rig::completion::TypedPrompt::prompt_typed`] so the
    /// provider returns a [`VerifierLlmResponse`] that has already
    /// been deserialised from a schema-conforming JSON document.
    /// This replaces the legacy path that asked the model for JSON in
    /// the prompt and called `serde_json::from_str` here — that path
    /// was brittle to stray prose, code-fence wrappers, and trailing
    /// commentary.
    async fn verify_with_llm(
        &self,
        client: &rig::providers::openai::Client,
        answer: &str,
        evidence: &[SearchResult],
        budget: &Arc<RequestBudget>,
    ) -> Result<VerificationResult, AgentError> {
        use rig::client::CompletionClient;
        use rig::completion::TypedPrompt;

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

        let started = Instant::now();

        // Per-call hook: budget recording + per-call observability
        // metrics. The verifier never invokes tools, so the tool-call
        // sink is never written to — we still pass it because the
        // hook constructor needs the type. `with_hook` is supported
        // on `TypedPromptRequest` exactly as on `PromptRequest`.
        let parsed = if let Some(metrics) = self.metrics.as_ref() {
            let sink = AgentRsHook::new_tool_call_sink();
            let hook = AgentRsHook::new(
                COMPONENT_VERIFIER,
                self.model.clone(),
                Arc::clone(budget),
                sink,
                Arc::clone(metrics),
                started,
            );
            agent
                .prompt_typed::<VerifierLlmResponse>(user_message)
                .with_hook(hook)
                .await
                .map_err(|e| AgentError::Verification(format!("LLM verifier call failed: {e}")))?
        } else {
            // No metrics handle — skip the hook so we don't fabricate
            // a no-op `RetrievalMetrics`. Budget is still updated via
            // a manual record below using the typed-response usage.
            let response = agent
                .prompt_typed::<VerifierLlmResponse>(user_message)
                .extended_details()
                .await
                .map_err(|e| AgentError::Verification(format!("LLM verifier call failed: {e}")))?;
            budget.record_call(&self.model, token_usage_from_rig(response.usage));
            response.output
        };

        if let Some(m) = &self.metrics {
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            m.record_llm_verifier_duration(elapsed_ms);
        }

        // The typed path either returns `VerifierLlmResponse` or
        // surfaces a structured error — no `serde_json::from_str` of
        // free-form text. The JSON spec forbids NaN / Infinity for
        // numbers, so `serde_json` rejects them at deserialisation.
        // `f64::clamp` panics on NaN, so we still defensively check
        // `is_finite()` here in case a future provider relaxes the
        // spec; coerce non-finite to 0.0 (safer floor than 1.0).
        let confidence_val = if parsed.confidence.is_finite() {
            parsed.confidence.clamp(0.0, 1.0)
        } else {
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

    /// Heuristic token overlap verification (lang-agnostic fallback).
    ///
    /// This path runs only when the LLM verifier client is unavailable
    /// (no API key, transport failure). The high-quality verifier path
    /// is [`Self::verify_with_llm`] — it is lang-aware by construction
    /// because the LLM does the semantic work.
    ///
    /// To stay corpus-language-neutral the heuristic tokenizes via
    /// [`UnicodeSegmentation::unicode_words`] and filters by minimum
    /// character count rather than against a per-language stopword
    /// list. This loses precision (mono-character function words like
    /// English `a` or Spanish `o` are not stripped, while shorter words
    /// in CJK / abugida scripts are over-aggressively dropped) but
    /// trades it for a single implementation that does not silently
    /// degrade on non-English corpora. The heuristic only exists to
    /// keep the pipeline well-formed when the LLM client is missing,
    /// so the precision floor is acceptable.
    fn verify_heuristic(
        &self,
        answer: &str,
        evidence: &[SearchResult],
    ) -> Result<VerificationResult, AgentError> {
        let answer_tokens: HashSet<String> = answer
            .to_lowercase()
            .unicode_words()
            .filter(|t| t.chars().count() >= HEURISTIC_MIN_TOKEN_CHARS)
            .map(|s| s.to_owned())
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
            let lowered = text.to_lowercase();
            for token in lowered.unicode_words() {
                if token.chars().count() >= HEURISTIC_MIN_TOKEN_CHARS {
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
