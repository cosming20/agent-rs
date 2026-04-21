//! Result persister -- synthesize response, persist telemetry.
//!
//! Final step in the orchestrator pipeline. Produces the
//! [`SynthesizedResponse`] and persists analytics (query_run + evidence_items).
//!
//! After the 2026-04-21 stateless pivot this no longer stores assistant
//! chat messages — that's the web app's job now.

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
        let verification = ctx.verification.as_ref().ok_or_else(|| {
            AgentError::Orchestrator("verification result missing before persist".into())
        })?;

        self.metrics
            .record_verifier_method(&verification.method, verification.is_grounded);
        self.metrics
            .record_query_duration_ms(ctx.elapsed_ms() as f64, ctx.strategy);

        let response = self.synthesizer.synthesize(
            &ctx.answer,
            &ctx.evidence,
            &ctx.query,
            verification,
            ctx.strategy,
            ctx.coverage_report.as_ref(),
        )?;

        info!(
            citations = response.citations.len(),
            confidence = response.confidence.value(),
            is_grounded = response.is_grounded,
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
            is_grounded: response.is_grounded,
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
                score: doc.score.value() as f64,
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
