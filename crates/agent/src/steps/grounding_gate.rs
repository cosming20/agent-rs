//! Grounding gate -- verify answer grounding, retry if needed.
//!
//! Runs the [`GroundingVerifier`] on the agent's answer and evidence.
//! If the answer is not grounded and retries remain, rebuilds the prompt
//! with feedback, re-runs the agent, re-fuses evidence, and re-verifies.

use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::Confidence;
use tracing::{error, info, instrument, warn};

use crate::context::QueryContext;
use crate::steps::agent_runner::AgentRunner;
use crate::steps::evidence_fuser::EvidenceFuser;
use crate::verifier::{GroundingVerifier, VerificationResult};

/// Verify answer grounding and retry if needed.
pub struct GroundingGate {
    verifier: Arc<GroundingVerifier>,
    max_retries: usize,
}

impl GroundingGate {
    /// Create a new GroundingGate.
    pub fn new(verifier: Arc<GroundingVerifier>, max_retries: usize) -> Self {
        Self {
            verifier,
            max_retries,
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

        while let Some(ref verification) = ctx.verification {
            if !self.verifier.needs_retry(verification) || ctx.retries >= self.max_retries {
                break;
            }

            ctx.retries += 1;
            warn!(
                confidence = verification.confidence.value(),
                is_grounded = verification.is_grounded,
                evidence_count = ctx.evidence.len(),
                retry = ctx.retries,
                max_retries = self.max_retries,
                query = %ctx.query,
                "grounding failed, retrying"
            );

            let retry_prompt = format!(
                "{}\n\n\
                 Your previous answer was not well-grounded in the evidence. \
                 Please search again with a broader strategy (try 'graph_hybrid') \
                 and provide a more evidence-based answer.\n\n\
                 Previous answer: {}",
                ctx.query, ctx.answer
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
                    });
                    break;
                }
            }
        }

        Ok(())
    }
}
