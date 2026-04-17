//! Evidence fuser -- merges KB and web evidence via ResultRanker.
//!
//! Produces a single ranked, deduplicated [`EvidencePack`] from
//! heterogeneous evidence sources.

use std::sync::Arc;

use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_retrieval::ResultRanker;
use tracing::{info, instrument, warn};

use crate::context::QueryContext;

/// Fuses knowledge-base and web evidence into a single ranked evidence pack.
pub struct EvidenceFuser {
    ranker: ResultRanker,
    metrics: Arc<RetrievalMetrics>,
}

impl EvidenceFuser {
    /// Create a new EvidenceFuser with the given ranker and metrics handle.
    pub fn new(ranker: ResultRanker, metrics: Arc<RetrievalMetrics>) -> Self {
        Self { ranker, metrics }
    }

    /// Fuse KB + web evidence and populate `ctx.evidence` / `ctx.fused_pack`.
    #[instrument(skip(self, ctx), fields(
        kb_count = ctx.kb_evidence.len(),
        web_count = ctx.web_evidence.len(),
        strategy = %ctx.strategy,
    ))]
    pub fn fuse(&self, ctx: &mut QueryContext) {
        let fused_pack = self.ranker.fuse_evidence(
            ctx.kb_evidence.clone(),
            ctx.web_evidence.clone(),
            &ctx.query,
            ctx.strategy,
            ctx.limit,
        );

        let fused_count = fused_pack.documents.len();
        ctx.evidence = fused_pack.documents.clone();
        ctx.fused_pack = Some(fused_pack);

        if ctx.evidence.is_empty() {
            // Do NOT short-circuit here: ctx.answer has already been produced
            // by the agent and may contain a legitimate "I don't know" response
            // that still needs verification + synthesis. The synthesis layer
            // produces an honest admission when evidence is empty (see
            // synthesis.rs), so the pipeline stays well-formed.
            warn!(
                kb_raw = ctx.kb_evidence.len(),
                web_raw = ctx.web_evidence.len(),
                fused = fused_count,
                strategy = %ctx.strategy,
                query = %ctx.query,
                event = "zero_evidence",
                "ZERO EVIDENCE after retrieval — synthesis will return 'I don't know'"
            );
            self.metrics.record_zero_evidence(ctx.strategy);
        } else {
            info!(
                kb = ctx.kb_evidence.len(),
                web = ctx.web_evidence.len(),
                fused = fused_count,
                strategy = %ctx.strategy,
                "evidence collected"
            );
        }
    }
}
