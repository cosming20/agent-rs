//! Coverage planning & verification types.
//!
//! `CoveragePlan` is built by [`crate::steps::coverage_planner::CoveragePlanner`]
//! when an exhaustive-intent query is detected. It enumerates the sections the
//! pipeline must cover. After tool calls complete, [`crate::steps::coverage_verifier::CoverageVerifier`]
//! produces a [`CoverageReport`] showing how much of the plan was actually
//! satisfied by retrieved evidence.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Reference to one section the [`CoveragePlan`] targets.
///
/// `section_id` is the Neo4j section UUID (or the synthetic `preamble:<doc_id>`
/// id for unsectioned chunks). `heading` is denormalised for log/debug clarity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SectionRef {
    pub section_id: String,
    pub doc_id: String,
    pub heading: String,
}

/// Coverage mode for the query.
///
/// `None` = the standard non-exhaustive flow runs (definition / how-to / etc.).
/// `Exhaustive` = the agent is expected to hit every section in `target_sections`.
/// `Partial` = a relaxed exhaustive (e.g., user asked "main methods" not "ALL").
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CoverageMode {
    None,
    Partial,
    Exhaustive,
}

impl Default for CoverageMode {
    fn default() -> Self {
        CoverageMode::None
    }
}

/// Plan emitted by the planner; consumed by the agent and the verifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoveragePlan {
    pub mode: CoverageMode,
    pub target_sections: Vec<SectionRef>,
}

/// Result of comparing the plan against actually-retrieved evidence.
///
/// Exposed via the response struct + OpenTelemetry metrics. **Never** appended
/// to the user-facing answer body (per spec §6.2 / §14 Q4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub expected: usize,
    pub fetched: usize,
    pub missing_section_ids: Vec<String>,
}

impl CoverageReport {
    /// Build a report from a plan and the set of section ids present in evidence.
    pub fn from_plan_and_evidence(
        plan: &CoveragePlan,
        evidence_section_ids: &HashSet<String>,
    ) -> Self {
        let expected = plan.target_sections.len();
        let mut missing: Vec<String> = plan
            .target_sections
            .iter()
            .filter(|s| !evidence_section_ids.contains(&s.section_id))
            .map(|s| s.section_id.clone())
            .collect();
        missing.sort();
        let fetched = expected.saturating_sub(missing.len());
        Self {
            expected,
            fetched,
            missing_section_ids: missing,
        }
    }

    /// True iff every targeted section produced at least one piece of evidence.
    pub fn is_complete(&self) -> bool {
        self.missing_section_ids.is_empty()
    }
}

/// Captured tool invocation (args + which tool) used by the grounding gate to
/// extract the rewritten subject query rather than the raw user prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub args_json: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_plan_default_is_none_mode() {
        let plan = CoveragePlan::default();
        assert_eq!(plan.mode, CoverageMode::None);
        assert!(plan.target_sections.is_empty());
    }

    #[test]
    fn coverage_report_full_when_all_sections_have_evidence() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef { section_id: "s1".into(), doc_id: "d".into(), heading: "5.1".into() },
                SectionRef { section_id: "s2".into(), doc_id: "d".into(), heading: "5.2".into() },
            ],
        };
        let evidence_section_ids = ["s1", "s2"].into_iter().map(String::from).collect();
        let report = CoverageReport::from_plan_and_evidence(&plan, &evidence_section_ids);
        assert_eq!(report.expected, 2);
        assert_eq!(report.fetched, 2);
        assert!(report.missing_section_ids.is_empty());
        assert!(report.is_complete());
    }

    #[test]
    fn coverage_report_partial_lists_missing_section_ids() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef { section_id: "s1".into(), doc_id: "d".into(), heading: "5.1".into() },
                SectionRef { section_id: "s2".into(), doc_id: "d".into(), heading: "5.2".into() },
                SectionRef { section_id: "s3".into(), doc_id: "d".into(), heading: "5.3".into() },
            ],
        };
        let evidence_section_ids = ["s1"].into_iter().map(String::from).collect();
        let report = CoverageReport::from_plan_and_evidence(&plan, &evidence_section_ids);
        assert_eq!(report.expected, 3);
        assert_eq!(report.fetched, 1);
        assert_eq!(report.missing_section_ids, vec!["s2", "s3"]);
        assert!(!report.is_complete());
    }
}
