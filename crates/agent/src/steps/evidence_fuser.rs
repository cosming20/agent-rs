//! Evidence fuser -- merges KB and web evidence via ResultRanker.
//!
//! Produces a single ranked, deduplicated [`EvidencePack`] from
//! heterogeneous evidence sources.
//!
//! # Coverage tier
//!
//! When `ctx.coverage_evidence` is non-empty (populated by exhaustive-mode
//! tool calls such as `SearchSectionsTool` / `FetchSectionChunksTool`), those
//! documents are appended to `ctx.evidence` **after** the RRF-ranked tier.
//! This preserves rank order for the LLM while guaranteeing that every
//! planner-chosen chunk is visible to the `CoverageVerifier` regardless of
//! the `ctx.limit` cap applied during RRF fusion.

use std::collections::HashSet;
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
    ///
    /// After RRF fusion and truncation to `ctx.limit`, any documents in
    /// `ctx.coverage_evidence` that are not already present (by `id`) are
    /// appended verbatim so that the `CoverageVerifier` can inspect them.
    #[instrument(skip(self, ctx), fields(
        kb_count = ctx.kb_evidence.len(),
        web_count = ctx.web_evidence.len(),
        coverage_count = ctx.coverage_evidence.len(),
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

        // Coverage tier: appended AFTER the ranked tier so the LLM still sees
        // the highest-scoring chunks first. Dedup by `id` prevents double-
        // counting when a coverage chunk also appears in kb_evidence OR when
        // coverage_evidence itself contains duplicates (e.g. from retry rounds
        // that re-fetch the same sections).
        if !ctx.coverage_evidence.is_empty() {
            // Build the initial set from the ranked tier, then grow it as we
            // push new coverage chunks so self-duplicates in coverage_evidence
            // are also filtered out.
            let mut existing_ids: HashSet<String> =
                ctx.evidence.iter().map(|e| e.id.clone()).collect();

            let before = ctx.evidence.len();
            for cov in &ctx.coverage_evidence {
                let id = &cov.id;
                if !existing_ids.contains(id) {
                    existing_ids.insert(id.clone());
                    ctx.evidence.push(cov.clone());
                }
            }
            let coverage_added = ctx.evidence.len() - before;
            info!(
                coverage_added,
                evidence_total = ctx.evidence.len(),
                "evidence_fuser: merged coverage tier"
            );
        }

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use agent_rs_core::types::{RetrievalStrategy, Score, UserId};
    use agent_rs_observability::metrics::RetrievalMetrics;
    use agent_rs_retrieval::ResultRanker;
    use agent_rs_stores::types::SearchResult;
    use serde_json::json;

    use super::EvidenceFuser;
    use crate::context::QueryContext;

    fn dummy_result(id: &str) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            score: Score::new(0.5),
            payload: json!({"chunk_id": id, "chunk_text": "..."}),
        }
    }

    fn ctx_with_kb_and_coverage() -> QueryContext {
        let mut ctx = QueryContext::new(
            "test query".to_string(),
            UserId::from_string("user".to_string()),
            Vec::new(),
            Vec::new(),
            RetrievalStrategy::Semantic,
            8,
            None,
        );
        ctx.kb_evidence = vec![dummy_result("kb-a"), dummy_result("kb-b")];
        ctx.coverage_evidence = vec![dummy_result("cov-1"), dummy_result("cov-2")];
        ctx
    }

    fn make_fuser() -> EvidenceFuser {
        EvidenceFuser::new(
            ResultRanker::default(),
            Arc::new(RetrievalMetrics::new("test")),
        )
    }

    #[test]
    fn fuser_preserves_every_coverage_evidence_chunk() {
        let mut ctx = ctx_with_kb_and_coverage();
        let fuser = make_fuser();
        fuser.fuse(&mut ctx);
        let ids: Vec<&str> = ctx
            .evidence
            .iter()
            .filter_map(|e| e.payload.get("chunk_id").and_then(|v| v.as_str()))
            .collect();
        assert!(
            ids.contains(&"cov-1"),
            "coverage chunk cov-1 dropped: ids = {:?}",
            ids
        );
        assert!(
            ids.contains(&"cov-2"),
            "coverage chunk cov-2 dropped: ids = {:?}",
            ids
        );
    }

    #[test]
    fn fuser_dedupes_same_id_across_kb_and_coverage() {
        let mut ctx = ctx_with_kb_and_coverage();
        // Push a kb chunk whose id matches a coverage chunk.
        ctx.kb_evidence.push(dummy_result("cov-1"));
        let fuser = make_fuser();
        fuser.fuse(&mut ctx);
        let count_cov1 = ctx.evidence.iter().filter(|e| e.id == "cov-1").count();
        assert_eq!(
            count_cov1, 1,
            "cov-1 should appear exactly once after dedup"
        );
    }

    #[test]
    fn fuser_dedupes_duplicates_within_coverage_evidence() {
        // coverage_evidence contains the same id twice (e.g. fetched in two
        // retry rounds). The fuser must emit it only once.
        let mut ctx = ctx_with_kb_and_coverage();
        // Append a duplicate of cov-1 into coverage_evidence.
        ctx.coverage_evidence.push(dummy_result("cov-1"));
        let fuser = make_fuser();
        fuser.fuse(&mut ctx);
        let count_cov1 = ctx.evidence.iter().filter(|e| e.id == "cov-1").count();
        assert_eq!(
            count_cov1, 1,
            "cov-1 should appear exactly once even when duplicated in coverage_evidence"
        );
    }

    #[test]
    fn fuser_ranked_tier_precedes_coverage_tier() {
        let mut ctx = ctx_with_kb_and_coverage();
        let fuser = make_fuser();
        fuser.fuse(&mut ctx);
        // kb-a and kb-b should come before coverage chunks.
        let ids: Vec<&str> = ctx.evidence.iter().map(|e| e.id.as_str()).collect();
        let pos_kb_a = ids.iter().position(|&id| id == "kb-a");
        let pos_cov_1 = ids.iter().position(|&id| id == "cov-1");
        assert!(
            pos_kb_a.is_some() && pos_cov_1.is_some(),
            "expected both kb-a and cov-1 in evidence: {:?}",
            ids
        );
        assert!(
            pos_kb_a.unwrap() < pos_cov_1.unwrap(),
            "ranked chunk kb-a should precede coverage chunk cov-1: {:?}",
            ids
        );
    }
}
