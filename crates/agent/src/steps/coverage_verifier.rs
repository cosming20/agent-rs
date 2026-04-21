//! CoverageVerifier — runs after EvidenceFuser, builds the CoverageReport
//! that records how much of the CoveragePlan was actually satisfied.
//!
//! The report is exposed via `QueryContext::coverage_report` (consumed by
//! synthesis to drop the hedging wrapper) and via OpenTelemetry metrics.
//! It is **never** appended to the user-facing answer body (per spec
//! §6.2 / §14 Q4).

use std::collections::HashSet;
use std::sync::Arc;

use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::types::SearchResult;
use tracing::{info, instrument};

use crate::context::QueryContext;
use crate::coverage::{CoveragePlan, CoverageReport};

/// Verifies how much of a [`CoveragePlan`] was satisfied by the evidence
/// accumulated during the agent's tool calls.
///
/// After construction with [`CoverageVerifier::new`], call [`CoverageVerifier::verify`]
/// with a mutable [`QueryContext`]. The step is a no-op when
/// `ctx.coverage_plan` is `None` (non-exhaustive flow).
pub struct CoverageVerifier {
    metrics: Arc<RetrievalMetrics>,
}

impl CoverageVerifier {
    /// Create a new [`CoverageVerifier`].
    ///
    /// # Arguments
    ///
    /// * `metrics` — shared [`RetrievalMetrics`] for emitting coverage counters.
    pub fn new(metrics: Arc<RetrievalMetrics>) -> Self {
        Self { metrics }
    }

    /// Run the verification step, stamping a [`CoverageReport`] onto `ctx`.
    ///
    /// Combines `ctx.evidence` (reranked tier) and `ctx.coverage_evidence`
    /// (untruncated exhaustive tier) into a single section-id set, then
    /// compares against the plan. The result is written to
    /// `ctx.coverage_report`.
    ///
    /// This is a no-op when `ctx.coverage_plan` is `None`.
    #[instrument(skip(self, ctx), fields(trace_id = ?ctx.trace_id))]
    pub fn verify(&self, ctx: &mut QueryContext) {
        let Some(plan) = ctx.coverage_plan.as_ref() else {
            return;
        };

        // Combine ranked evidence + coverage_evidence (the untruncated tier)
        // into a single section-id set. We only need section_id strings, so
        // no cloning of full SearchResult payloads beyond what's needed.
        let combined: Vec<&SearchResult> = ctx
            .evidence
            .iter()
            .chain(ctx.coverage_evidence.iter())
            .collect();

        let report = Self::build_report(plan, &combined);

        info!(
            expected = report.expected,
            fetched = report.fetched,
            missing = report.missing_section_ids.len(),
            "coverage verifier complete"
        );
        self.metrics
            .coverage_verified(report.fetched, report.expected);
        ctx.coverage_report = Some(report);
    }

    /// Pure helper — extracted so unit tests can target it without
    /// constructing a full [`CoverageVerifier`] with metrics.
    ///
    /// Collects all `section_id` values from the `evidence` slice and
    /// delegates to [`CoverageReport::from_plan_and_evidence`].
    pub(crate) fn build_report(
        plan: &CoveragePlan,
        evidence: &[&SearchResult],
    ) -> CoverageReport {
        let ids: HashSet<String> = evidence
            .iter()
            .filter_map(|e| {
                e.payload
                    .get("section_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();
        CoverageReport::from_plan_and_evidence(plan, &ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};
    use agent_rs_core::types::Score;
    use serde_json::json;

    fn ev(section_id: &str) -> SearchResult {
        // Build a SearchResult whose payload carries "section_id".
        // `SearchResult.payload` is `serde_json::Value`, not a Map.
        SearchResult {
            id: section_id.to_owned(),
            score: Score::new(1.0),
            payload: json!({"section_id": section_id, "chunk_text": "..."}),
        }
    }

    fn plan_for(ids: &[&str]) -> CoveragePlan {
        CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: ids
                .iter()
                .map(|id| SectionRef {
                    section_id: (*id).into(),
                    doc_id: "d".into(),
                    heading: format!("h-{id}"),
                })
                .collect(),
        }
    }

    #[test]
    fn verifier_marks_complete_when_every_target_present() {
        let plan = plan_for(&["s1", "s2"]);
        let evidence = [ev("s1"), ev("s2")];
        let refs: Vec<&SearchResult> = evidence.iter().collect();
        let report = CoverageVerifier::build_report(&plan, &refs);
        assert!(report.is_complete());
        assert_eq!(report.fetched, 2);
        assert_eq!(report.expected, 2);
        assert!(report.missing_section_ids.is_empty());
    }

    #[test]
    fn verifier_lists_missing_sections() {
        let plan = plan_for(&["s1", "s2", "s3"]);
        let evidence = [ev("s1")];
        let refs: Vec<&SearchResult> = evidence.iter().collect();
        let report = CoverageVerifier::build_report(&plan, &refs);
        assert!(!report.is_complete());
        assert_eq!(report.fetched, 1);
        assert_eq!(report.expected, 3);
        assert_eq!(report.missing_section_ids, vec!["s2", "s3"]);
    }
}
