//! Result persister -- synthesize response, persist telemetry.
//!
//! Final step in the orchestrator pipeline. Produces the
//! [`SynthesizedResponse`] and persists analytics (query_run + evidence_items).
//!
//! After the 2026-04-21 stateless pivot this no longer stores assistant
//! chat messages — that's the web app's job now.
//!
//! The previous LLM-based grounding verifier was deleted in the
//! synthesizer-streams refactor. Persisted `query_run` rows still
//! carry an `is_grounded` column for backwards-compatibility with
//! existing dashboards / migrations; we derive it deterministically
//! from the heuristic confidence:
//!
//! ```text
//! is_grounded = response.confidence >= GROUNDED_CONFIDENCE_THRESHOLD
//! ```
//!
//! No LLM call, no retries, no [`crate::context::QueryContext::verification`]
//! field on the context.

use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::QueryRun;
use chrono::Utc;
use tracing::{error, info, instrument};
use uuid::Uuid;

use crate::context::QueryContext;
use crate::synthesis::{ResponseSynthesizer, SynthesizedResponse};

/// Heuristic-confidence threshold above which a persisted `query_run`
/// row is flagged as `is_grounded = true`.
///
/// The DB column predates the verifier deletion and is consumed by
/// existing dashboards / replay tooling, so the persister keeps
/// emitting it. The value mirrors the previous `synthesis::
/// LOW_CONFIDENCE_HEDGE_THRESHOLD` (0.15) — answers that already
/// trigger the synthesizer's "please verify independently" wrapper
/// are simultaneously marked ungrounded in the audit log, which keeps
/// the two surfaces consistent.
const GROUNDED_CONFIDENCE_THRESHOLD: f64 = 0.15;

/// Synthesizes response, persists telemetry rows.
pub struct ResultPersister {
    synthesizer: ResponseSynthesizer,
    store: Arc<dyn StateStore>,
    metrics: Arc<RetrievalMetrics>,
}

impl ResultPersister {
    /// Create a new ResultPersister.
    pub fn new(
        synthesizer: ResponseSynthesizer,
        store: Arc<dyn StateStore>,
        metrics: Arc<RetrievalMetrics>,
    ) -> Self {
        Self {
            synthesizer,
            store,
            metrics,
        }
    }

    /// Synthesize, persist, and return the final response.
    #[instrument(skip(self, ctx), fields(
        trace_id = ?ctx.trace_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn persist(&self, ctx: &QueryContext) -> Result<SynthesizedResponse, AgentError> {
        self.metrics
            .record_query_duration_ms(ctx.elapsed_ms() as f64, ctx.strategy);

        let response = self.synthesizer.synthesize(
            &ctx.answer,
            &ctx.evidence,
            &ctx.query,
            ctx.strategy,
            ctx.coverage_report.as_ref(),
        )?;

        let is_grounded = response.confidence.value() >= GROUNDED_CONFIDENCE_THRESHOLD;

        info!(
            citations = response.citations.len(),
            confidence = response.confidence.value(),
            is_grounded,
            "response synthesized"
        );

        // Persist pipeline run — `trace_id` replaces the old session_id;
        // it's an optional correlation UUID with no ownership semantics.
        let run = QueryRun {
            id: Uuid::new_v4(),
            trace_id: ctx.trace_id,
            user_id: ctx.user_id.clone(),
            query: ctx.query.clone(),
            answer: response.answer.clone(),
            strategy: ctx.strategy,
            confidence: response.confidence,
            is_grounded,
            created_at: Utc::now().naive_utc(),
        };

        self.store.persist_run(&run).await.map_err(|err| {
            error!(error = %err, "failed to persist query run — audit trail lost");
            AgentError::Orchestrator(format!("failed to persist query run: {err}"))
        })?;

        // Persist evidence items.
        let evidence_items: Vec<agent_rs_stores::types::EvidenceItem> = ctx
            .evidence
            .iter()
            .map(|doc| agent_rs_stores::types::EvidenceItem {
                id: Uuid::new_v4(),
                run_id: run.id,
                content: doc
                    .payload
                    .get("chunk_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned(),
                source: doc
                    .payload
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_owned(),
                score: doc.score.value(),
                created_at: Utc::now().naive_utc(),
            })
            .collect();

        if !evidence_items.is_empty() {
            if let Err(err) = self.store.store_evidence_batch(&evidence_items).await {
                error!(error = %err, "failed to persist evidence items");
            }
        }

        Ok(response)
    }
}
