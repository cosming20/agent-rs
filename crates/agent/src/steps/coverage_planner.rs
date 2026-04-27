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

use agent_rs_core::types::LangCode;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::neo4j::Neo4jStore;
use tracing::{instrument, warn};

use crate::context::QueryContext;
use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};

// ---- Per-language token tables ----
//
// These lists are best-effort starter sets — extend per language as user
// feedback identifies gaps. Tokens are matched case-insensitively on word
// boundaries (multi-word entries via substring); see [`contains_word`].
//
// "complete" is intentionally absent from the hard sets — it is too broad
// ("complete picture" does not mean exhaustive enumeration) and lives in
// the soft set instead.

/// Hard exhaustive tokens — English.
const HARD_EN: &[&str] = &[
    "all",
    "every",
    "exhaustive",
    "full list",
    "enumerate",
    "list of all",
    "comprehensive",
];

/// Hard exhaustive tokens — Romanian.
const HARD_RO: &[&str] = &[
    "toate",
    "intreaga",
    "exhaustiv",
    "complet",
    "enumera",
    "enumeraza",
    "lista",
    "listeaza",
    "cheatsheet",
];

/// Hard exhaustive tokens — Spanish.
const HARD_ES: &[&str] = &[
    "todos",
    "todas",
    "cada",
    "completo",
    "exhaustivo",
    "enumera",
    "lista completa",
];

/// Hard exhaustive tokens — German.
const HARD_DE: &[&str] = &[
    "alle",
    "jede",
    "jeder",
    "jedes",
    "vollständig",
    "vollstandig",
    "erschöpfend",
    "erschoepfend",
];

/// Hard exhaustive tokens — French.
const HARD_FR: &[&str] = &[
    "tous",
    "toutes",
    "chaque",
    "complet",
    "complète",
    "exhaustif",
    "exhaustive",
    "liste complète",
];

/// Hard exhaustive tokens — Italian.
const HARD_IT: &[&str] = &[
    "tutti",
    "tutte",
    "ogni",
    "completo",
    "esaustivo",
    "elenco completo",
];

/// Hard exhaustive tokens — Portuguese.
const HARD_PT: &[&str] = &[
    "todos",
    "todas",
    "cada",
    "completo",
    "exaustivo",
    "lista completa",
];

/// Soft quantifier tokens — English.
///
/// Alone they do not trigger exhaustive mode but they log a signal so the
/// rule set can be tuned when users complain. Two soft tokens together
/// promote to [`CoverageMode::Partial`].
const SOFT_EN: &[&str] = &["most", "main", "primary", "complete"];

/// Soft quantifier tokens — Romanian.
const SOFT_RO: &[&str] = &["majoritatea", "fiecare", "principalele"];

/// Soft quantifier tokens — Spanish.
const SOFT_ES: &[&str] = &["mayoría", "mayoria", "principales", "principal"];

/// Soft quantifier tokens — German.
const SOFT_DE: &[&str] = &["meisten", "wichtigsten", "haupt", "primär", "primaer"];

/// Soft quantifier tokens — French.
const SOFT_FR: &[&str] = &["plupart", "principaux", "principales", "principal"];

/// Soft quantifier tokens — Italian.
const SOFT_IT: &[&str] = &["maggior", "principali", "principale"];

/// Soft quantifier tokens — Portuguese.
const SOFT_PT: &[&str] = &["maioria", "principais", "principal"];

/// Return the hard-exhaustive token list for `lang`.
///
/// `LangCode::Other` falls back to the English list — a safe default that
/// keeps coverage detection working for the most common cross-language
/// case (English-loanword queries) while avoiding domain-specific bias.
fn hard_tokens(lang: LangCode) -> &'static [&'static str] {
    match lang {
        LangCode::En => HARD_EN,
        LangCode::Ro => HARD_RO,
        LangCode::Es => HARD_ES,
        LangCode::De => HARD_DE,
        LangCode::Fr => HARD_FR,
        LangCode::It => HARD_IT,
        LangCode::Pt => HARD_PT,
        LangCode::Other => HARD_EN,
    }
}

/// Return the soft-quantifier token list for `lang`.
///
/// Mirrors [`hard_tokens`]: `LangCode::Other` falls back to English.
fn soft_tokens(lang: LangCode) -> &'static [&'static str] {
    match lang {
        LangCode::En => SOFT_EN,
        LangCode::Ro => SOFT_RO,
        LangCode::Es => SOFT_ES,
        LangCode::De => SOFT_DE,
        LangCode::Fr => SOFT_FR,
        LangCode::It => SOFT_IT,
        LangCode::Pt => SOFT_PT,
        LangCode::Other => SOFT_EN,
    }
}

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
        let intent = detect_intent_with_signal(&ctx.query, ctx.detected_lang);

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

/// Return the [`CoverageMode`] for `q` using the rule table for `lang`.
///
/// Convenience wrapper around [`detect_intent_with_signal`] for callers
/// that only need the mode.
pub fn detect_intent(q: &str, lang: LangCode) -> CoverageMode {
    detect_intent_with_signal(q, lang).mode
}

/// Classify `q` and return a full [`IntentSignal`] (mode + soft-heuristic
/// metadata for observability), driven by the language-specific token table
/// selected via `lang`.
///
/// # Rules (in priority order)
///
/// 1. Any hard-exhaustive token (case-insensitive) → `Exhaustive`.
/// 2. Two or more soft-quantifier tokens → `Partial` with soft heuristic.
/// 3. Exactly one soft-quantifier token → `None` with soft heuristic.
/// 4. No tokens matched → `None`, no heuristic.
///
/// `lang` selects which per-language list to consult; `LangCode::Other`
/// falls back to English. Mis-detected language therefore degrades to the
/// English list, not to silence — but tokens specific to the actual query
/// language will be missed in that case.
pub(crate) fn detect_intent_with_signal(q: &str, lang: LangCode) -> IntentSignal {
    let lc = q.to_lowercase();

    if hard_tokens(lang).iter().any(|t| contains_word(&lc, t)) {
        return IntentSignal {
            mode: CoverageMode::Exhaustive,
            soft_heuristic_fired: false,
            heuristic_name: "hard_token",
        };
    }

    let soft_count = soft_tokens(lang)
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
            detect_intent("vreau toate metodele de segmentare", LangCode::Ro),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn detects_exhaustive_intent_via_cheatsheet() {
        // "cheatsheet" is in the RO list; an English-tagged query still
        // matches because EN's "all" fires first.
        assert_eq!(
            detect_intent("write a cheatsheet of all methods", LangCode::En),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn does_not_detect_exhaustive_for_definition_query() {
        assert_eq!(
            detect_intent("ce este metoda Otsu", LangCode::Ro),
            CoverageMode::None
        );
    }

    #[test]
    fn soft_heuristic_logs_when_partial_signal_without_hard_match() {
        let r = detect_intent_with_signal("give me the complete picture", LangCode::En);
        assert_eq!(r.mode, CoverageMode::None);
        assert!(r.soft_heuristic_fired);
    }

    #[test]
    fn german_query_detects_alle_as_hard_exhaustive() {
        assert_eq!(
            detect_intent("Erkläre alle Methoden", LangCode::De),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn spanish_query_detects_todos_as_hard_exhaustive() {
        assert_eq!(
            detect_intent("Explica todos los métodos", LangCode::Es),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn french_query_detects_tous_as_hard_exhaustive() {
        assert_eq!(
            detect_intent("Explique tous les méthodes", LangCode::Fr),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn italian_query_detects_tutti_as_hard_exhaustive() {
        assert_eq!(
            detect_intent("Elenca tutti i metodi", LangCode::It),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn portuguese_query_detects_todos_as_hard_exhaustive() {
        assert_eq!(
            detect_intent("Explique todos os métodos", LangCode::Pt),
            CoverageMode::Exhaustive
        );
    }

    #[test]
    fn english_word_in_french_lang_misses_french_list() {
        // NOTE: language misdetection makes intent detection miss. Here a
        // plainly English query is mistagged as French — the FR table has
        // no "all", so the hard rule does not fire. This test documents
        // the known limitation: lang-detect quality directly bounds intent
        // detection quality. Mitigation: keep LangCode::Other (and English
        // as the fallback) as the default for low-confidence detections.
        assert_eq!(
            detect_intent("explain all methods", LangCode::Fr),
            CoverageMode::None
        );
    }

    #[test]
    fn lang_code_other_falls_back_to_english() {
        // `Other` selects the EN list, so EN tokens still fire.
        assert_eq!(
            detect_intent("list all the methods", LangCode::Other),
            CoverageMode::Exhaustive
        );
    }
}
