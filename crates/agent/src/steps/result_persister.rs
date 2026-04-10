//! Result persister -- synthesize response, persist run, store messages.
//!
//! Final step in the orchestrator pipeline. Produces the
//! [`SynthesizedResponse`] and persists all state.

use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::MessageRole;
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::QueryRun;
use chrono::Utc;
use tracing::{info, instrument, warn};
use uuid::Uuid;

use crate::context::QueryContext;
use crate::memory::MemoryManager;
use crate::synthesis::{ResponseSynthesizer, SynthesizedResponse};

/// Synthesizes response, persists run data, stores messages.
pub struct ResultPersister {
    synthesizer: ResponseSynthesizer,
    memory: Arc<MemoryManager>,
    store: Arc<dyn StateStore>,
}

impl ResultPersister {
    /// Create a new ResultPersister.
    pub fn new(
        synthesizer: ResponseSynthesizer,
        memory: Arc<MemoryManager>,
        store: Arc<dyn StateStore>,
    ) -> Self {
        Self {
            synthesizer,
            memory,
            store,
        }
    }

    /// Synthesize, persist, and return the final response.
    #[instrument(skip(self, ctx), fields(
        session_id = %ctx.session_id,
        strategy = %ctx.strategy,
    ))]
    pub async fn persist(&self, ctx: &QueryContext) -> Result<SynthesizedResponse, AgentError> {
        let verification = ctx.verification.as_ref().ok_or_else(|| {
            AgentError::Orchestrator(
                "verification result missing before persist".into(),
            )
        })?;

        // Synthesize citations
        let response = self.synthesizer.synthesize(
            &ctx.answer,
            &ctx.evidence,
            &ctx.query,
            verification,
            ctx.strategy,
        )?;

        info!(
            citations = response.citations.len(),
            confidence = response.confidence.value(),
            is_grounded = response.is_grounded,
            "response synthesized"
        );

        // Persist pipeline run
        let run = QueryRun {
            id: Uuid::new_v4(),
            session_id: ctx.session_id.clone(),
            user_id: ctx.user_id.clone(),
            query: ctx.query.clone(),
            answer: response.answer.clone(),
            strategy: ctx.strategy,
            confidence: response.confidence,
            is_grounded: response.is_grounded,
            created_at: Utc::now().naive_utc(),
        };

        if let Err(err) = self.store.persist_run(&run).await {
            warn!(error = %err, "failed to persist query run");
        }

        // Persist evidence items
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
                score: doc.score.0,
                created_at: Utc::now().naive_utc(),
            })
            .collect();

        if !evidence_items.is_empty() {
            if let Err(err) = self.store.store_evidence_batch(&evidence_items).await {
                warn!(error = %err, "failed to persist evidence items");
            }
        }

        // Store assistant response in memory
        if let Err(err) = self
            .memory
            .store_message(&ctx.session_id, MessageRole::Assistant, &response.answer)
            .await
        {
            warn!(error = %err, "failed to store assistant message");
        }

        Ok(response)
    }
}
