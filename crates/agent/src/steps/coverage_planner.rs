//! CoveragePlanner — pre-agent step that detects exhaustive intent and
//! pre-builds a [`CoveragePlan`] over Neo4j sections.
//!
//! Runs between ContextBuilder and AgentRunner. No-op for non-exhaustive
//! queries (zero added latency for definition / how-to flows).
//!
//! Post-stateless-pivot: agent-rs no longer owns an ingested-documents
//! catalogue — the web layer pins the active document set per turn and
//! passes MinIO object keys inline via [`QueryContext::active_document_keys`].
//! The planner walks those keys directly; it does not consult Postgres.

use std::sync::Arc;

use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::neo4j::Neo4jStore;
use tracing::{instrument, warn};

use crate::context::QueryContext;
use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};

/// Hard tokens that unambiguously mean "exhaustive coverage".
///
/// Bilingual (RO / EN). Matched case-insensitively on word boundaries.
/// "complete" is intentionally absent — it is too broad ("complete picture"
/// does not mean exhaustive enumeration) and lives in the soft set instead.
const HARD_EXHAUSTIVE_TOKENS: &[&str] = &[
    // RO
    "toate",
    "intreaga",
    "exhaustiv",
    "complet",
    "enumera",
    "enumeraza",
    "lista",
    "listeaza",
    "cheatsheet",
    // EN
    "all",
    "every",
    "exhaustive",
    "full list",
    "enumerate",
    "list of all",
    "comprehensive",
];

/// Soft quantifiers — alone they do not trigger exhaustive mode but they
/// log a signal so the rule set can be tuned when users complain.
///
/// "complete" is deliberately here (not in the hard set): "complete X"
/// often means thorough but not necessarily list-all — e.g. "the complete
/// picture". Two soft tokens together promote to [`CoverageMode::Partial`].
const SOFT_QUANTIFIER_TOKENS: &[&str] = &[
    // RO
    "majoritatea",
    "fiecare",
    "principalele",
    // EN
    "most",
    "main",
    "primary",
    "complete",
];

/// Pre-agent step: detect exhaustive intent and build a [`CoveragePlan`].
///
/// Constructed once and reused across requests (cheaply cloneable via
/// internal [`Arc`]s). The [`plan`](CoveragePlanner::plan) method is the
/// single entry point.
pub struct CoveragePlanner {
    neo4j: Arc<Neo4jStore>,
    metrics: Arc<RetrievalMetrics>,
}

impl CoveragePlanner {
    /// Construct a new planner.
    ///
    /// Both dependencies must outlive the planner (typically managed by the
    /// orchestrator's [`Arc`]s).
    pub fn new(neo4j: Arc<Neo4jStore>, metrics: Arc<RetrievalMetrics>) -> Self {
        Self { neo4j, metrics }
    }

    /// Analyse `ctx.query` and, when exhaustive intent is detected, populate
    /// `ctx.coverage_plan` with every Neo4j section of every pinned document.
    ///
    /// For non-exhaustive queries this method returns immediately with no
    /// side effects — zero added latency on definition / how-to flows.
    ///
    /// # Errors (soft)
    ///
    /// Store failures are logged at WARN and the plan is left as `None`
    /// (degraded mode — the pipeline continues without coverage tracking).
    #[instrument(skip(self, ctx), fields(trace_id = ?ctx.trace_id))]
    pub async fn plan(&self, ctx: &mut QueryContext) {
        let intent = detect_intent_with_signal(&ctx.query);

        if intent.soft_heuristic_fired && intent.mode == CoverageMode::None {
            warn!(
                query = %ctx.query,
                heuristic = %intent.heuristic_name,
                "coverage_planner.likely_exhaustive_unmatched"
            );
            self.metrics
                .coverage_planner_unmatched(intent.heuristic_name);
        }

        if intent.mode == CoverageMode::None {
            return;
        }

        if ctx.active_document_keys.is_empty() {
            warn!(
                query = %ctx.query,
                "coverage_planner: exhaustive intent but no active documents; skipping plan"
            );
            return;
        }

        let mut targets: Vec<SectionRef> = Vec::new();
        for key in &ctx.active_document_keys {
            match self.neo4j.list_sections_for_doc(&ctx.user_id, key).await {
                Ok(rows) => {
                    for r in rows {
                        targets.push(SectionRef {
                            section_id: r.section_id,
                            doc_id: r.doc_id,
                            heading: r.heading,
                        });
                    }
                }
                Err(e) => {
                    warn!(
                        minio_object_key = %key,
                        error = %e,
                        "coverage_planner: skipping doc"
                    );
                }
            }
        }

        ctx.coverage_plan = Some(CoveragePlan {
            mode: intent.mode,
            target_sections: targets,
        });
    }
}

// ---- Intent detection ----

/// Result of intent detection.
///
/// Internal type — exposed only for `pub(crate)` tests.
#[derive(Debug, PartialEq)]
pub(crate) struct IntentSignal {
    pub mode: CoverageMode,
    /// True when a soft quantifier fired but the strict hard-token rules
    /// did not promote it to exhaustive/partial mode.
    pub soft_heuristic_fired: bool,
    /// Stable string label used as the Prometheus `reason` label on the
    /// `coverage_planner_unmatched_total` counter.
    pub heuristic_name: &'static str,
}

/// Return `true` when `text` contains `token` as a whole word (delimited by
/// non-alphanumeric characters on both sides).
///
/// Multi-word tokens (containing spaces, e.g. `"full list"`) are matched
/// via a simple substring check because the spaces already act as word
/// boundaries in the surrounding text.
fn contains_word(text: &str, token: &str) -> bool {
    if token.contains(' ') {
        return text.contains(token);
    }
    let mut start = 0;
    while let Some(pos) = text[start..].find(token) {
        let abs = start + pos;
        let before_ok = abs == 0
            || !text
                .as_bytes()
                .get(abs - 1)
                .copied()
                .map(|b| b.is_ascii_alphanumeric())
                .unwrap_or(false);
        let after_idx = abs + token.len();
        let after_ok = after_idx >= text.len()
            || !text
                .as_bytes()
                .get(after_idx)
                .copied()
                .map(|b| b.is_ascii_alphanumeric())
                .unwrap_or(false);
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
        if start >= text.len() {
            break;
        }
    }
    false
}

/// Return the [`CoverageMode`] for `q` using the bilingual rule table.
///
/// Convenience wrapper around [`detect_intent_with_signal`] for callers
/// that only need the mode.
pub fn detect_intent(q: &str) -> CoverageMode {
    detect_intent_with_signal(q).mode
}

/// Classify `q` and return a full [`IntentSignal`] (mode + soft-heuristic
/// metadata for observability).
///
/// # Rules (in priority order)
///
/// 1. Any hard-exhaustive token (case-insensitive) → `Exhaustive`.
/// 2. Two or more soft-quantifier tokens → `Partial` with soft heuristic.
/// 3. Exactly one soft-quantifier token → `None` with soft heuristic.
/// 4. No tokens matched → `None`, no heuristic.
pub(crate) fn detect_intent_with_signal(q: &str) -> IntentSignal {
    let lc = q.to_lowercase();

    if HARD_EXHAUSTIVE_TOKENS.iter().any(|t| contains_word(&lc, t)) {
        return IntentSignal {
            mode: CoverageMode::Exhaustive,
            soft_heuristic_fired: false,
            heuristic_name: "hard_token",
        };
    }

    let soft_count = SOFT_QUANTIFIER_TOKENS
        .iter()
        .filter(|t| contains_word(&lc, t))
        .count();
    if soft_count >= 2 {
        return IntentSignal {
            mode: CoverageMode::Partial,
            soft_heuristic_fired: true,
            heuristic_name: "soft_quantifier_pair",
        };
    }
    if soft_count == 1 {
        return IntentSignal {
            mode: CoverageMode::None,
            soft_heuristic_fired: true,
            heuristic_name: "soft_quantifier_single",
        };
    }

    IntentSignal {
        mode: CoverageMode::None,
        soft_heuristic_fired: false,
        heuristic_name: "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_exhaustive_intent_via_toate() {
        assert_eq!(
            detect_intent("vreau toate metodele de segmentare"),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn detects_exhaustive_intent_via_cheatsheet() {
        assert_eq!(
            detect_intent("write a cheatsheet of all methods"),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn does_not_detect_exhaustive_for_definition_query() {
        assert_eq!(detect_intent("ce este metoda Otsu"), CoverageMode::None);
    }

    #[test]
    fn soft_heuristic_logs_when_partial_signal_without_hard_match() {
        let r = detect_intent_with_signal("give me the complete picture");
        assert_eq!(r.mode, CoverageMode::None);
        assert!(r.soft_heuristic_fired);
    }
}
