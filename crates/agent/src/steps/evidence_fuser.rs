//! Evidence fuser -- merges KB and web evidence via ResultRanker.
//!
//! Produces a single ranked, deduplicated [`EvidencePack`] from
//! heterogeneous evidence sources.

use agent_rs_retrieval::ResultRanker;
use tracing::{info, instrument, warn};

use crate::context::QueryContext;

/// Fuses knowledge-base and web evidence into a single ranked evidence pack.
pub struct EvidenceFuser {
    ranker: ResultRanker,
}

impl EvidenceFuser {
    /// Create a new EvidenceFuser with the given ranker.
    pub fn new(ranker: ResultRanker) -> Self {
        Self { ranker }
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
            warn!(
                kb_raw = ctx.kb_evidence.len(),
                web_raw = ctx.web_evidence.len(),
                fused = fused_count,
                strategy = %ctx.strategy,
                query = %ctx.query,
                "ZERO EVIDENCE after retrieval"
            );
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
