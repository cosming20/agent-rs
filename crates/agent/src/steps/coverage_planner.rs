//! CoveragePlanner — pre-agent step that detects exhaustive intent and
//! pre-builds a CoveragePlan over Neo4j sections.
//!
//! Runs between ContextBuilder and AgentRunner. No-op for non-exhaustive
//! queries (zero added latency for definition / how-to flows).

use std::sync::Arc;

use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use tracing::{instrument, warn};

use crate::context::QueryContext;
use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};

/// Hard tokens that unambiguously mean "exhaustive coverage".
///
/// Bilingual (RO / EN). Matched case-insensitively via `str::contains`.
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
/// [`Arc`] internals). The [`plan`](CoveragePlanner::plan) method is the
/// single entry point.
pub struct CoveragePlanner {
    neo4j: Arc<Neo4jStore>,
    pg: Arc<PgStore>,
    metrics: Arc<RetrievalMetrics>,
}

impl CoveragePlanner {
    /// Construct a new planner.
    ///
    /// All three dependencies must outlive the planner (typically managed by
    /// the orchestrator's `Arc`s).
    pub fn new(neo4j: Arc<Neo4jStore>, pg: Arc<PgStore>, metrics: Arc<RetrievalMetrics>) -> Self {
        Self { neo4j, pg, metrics }
    }

    /// Analyse `ctx.query` and, when exhaustive intent is detected, populate
    /// `ctx.coverage_plan` with all Neo4j sections for every ingested document.
    ///
    /// For non-exhaustive queries this method returns immediately with no
    /// side-effects — zero added latency on the definition / how-to flows.
    ///
    /// # Errors (soft)
    ///
    /// Store failures are logged at WARN and the plan is left as `None`
    /// (degraded mode — the pipeline continues without coverage tracking).
    #[instrument(skip(self, ctx), fields(session_id = %ctx.session_id))]
    pub async fn plan(&self, ctx: &mut QueryContext) {
        let intent = detect_intent_with_signal(&ctx.query);

        if intent.soft_heuristic_fired && intent.mode == CoverageMode::None {
            warn!(
                query = %ctx.query,
                heuristic = %intent.heuristic_name,
                "coverage_planner.likely_exhaustive_unmatched"
            );
            self.metrics.coverage_planner_unmatched(intent.heuristic_name);
        }

        if intent.mode == CoverageMode::None {
            return;
        }

        // Resolve target docs. If `ctx.active_document_ids` is set, restrict
        // to that pinned set. Otherwise fall back to the user's full document
        // library — preserves the previous single-corpus exhaustive behaviour.
        let all_docs = match self.pg.list_ingested_documents(&ctx.user_id).await {
            Ok(d) => d,
            Err(e) => {
                warn!(error = %e, "coverage_planner: failed to list docs; skipping plan");
                return;
            }
        };

        // When the session has active documents, restrict to that subset;
        // otherwise use the full library (None = no filter applied).
        let library_docs: Vec<agent_rs_stores::postgres::IngestedDocumentRow> =
            match ctx.active_document_ids.as_ref() {
                Some(ids) => all_docs
                    .into_iter()
                    .filter(|d| ids.contains(&d.document_id))
                    .collect(),
                None => all_docs,
            };

        let docs = filter_docs_by_query(&ctx.query, &library_docs);
        if docs.len() < library_docs.len() {
            tracing::info!(
                matched = docs.len(),
                total = library_docs.len(),
                "coverage_planner: query references specific documents"
            );
        }

        let mut targets: Vec<SectionRef> = Vec::new();
        for doc in docs {
            let rows =
                match self.neo4j.list_sections_for_doc(&ctx.user_id, &doc.document_id.to_string()).await {
                    Ok(r) => r,
                    Err(e) => {
                        warn!(
                            doc_id = %doc.document_id,
                            error = %e,
                            "coverage_planner: skipping doc"
                        );
                        continue;
                    }
                };
            for r in rows {
                targets.push(SectionRef {
                    section_id: r.section_id,
                    doc_id: r.doc_id,
                    heading: r.heading,
                });
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
/// non-alphanumeric / non-diacritic characters on both sides).
///
/// Multi-word tokens (containing spaces, e.g. `"full list"`) are matched
/// via a simple substring check because the spaces already act as word
/// boundaries in the surrounding text.
fn contains_word(text: &str, token: &str) -> bool {
    if token.contains(' ') {
        return text.contains(token);
    }
    // Single-word token: require non-alphanumeric boundary on each side.
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

    let soft_count = SOFT_QUANTIFIER_TOKENS.iter().filter(|t| contains_word(&lc, *t)).count();
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

// ---- Multi-document scoping ----

/// Minimum token length considered as a document-name hint.
///
/// Shorter substrings ("a", "pe", "the") match everything and would
/// cause every ingested document to look "named" in the query.
const DOC_NAME_MIN_LEN: usize = 4;

/// Filter `all_docs` to only those whose `source` filename stem or
/// `title` substring appears in the query (case-insensitive, normalised
/// on non-alphanumeric boundaries). If the query names no ingested
/// document, returns the full slice — preserves the previous
/// "all-corpus exhaustive" behaviour so single-doc deployments keep
/// working unchanged.
pub(crate) fn filter_docs_by_query<'a>(
    query: &str,
    all_docs: &'a [agent_rs_stores::postgres::IngestedDocumentRow],
) -> Vec<&'a agent_rs_stores::postgres::IngestedDocumentRow> {
    let lc_query = query.to_lowercase();

    let matched: Vec<&agent_rs_stores::postgres::IngestedDocumentRow> = all_docs
        .iter()
        .filter(|d| doc_referenced_in_query(&lc_query, d))
        .collect();

    if matched.is_empty() {
        all_docs.iter().collect()
    } else {
        matched
    }
}

/// True when `lc_query` references this document by filename stem or
/// title substring. All comparisons are case-insensitive; candidate
/// strings below [`DOC_NAME_MIN_LEN`] are ignored to avoid matching
/// every doc on high-frequency short tokens.
fn doc_referenced_in_query(
    lc_query: &str,
    doc: &agent_rs_stores::postgres::IngestedDocumentRow,
) -> bool {
    // Source filename: strip extension, take the stem.
    let stem: String = doc
        .source
        .rsplit_once('.')
        .map(|(pre, _)| pre.to_lowercase())
        .unwrap_or_else(|| doc.source.to_lowercase());
    if stem.chars().count() >= DOC_NAME_MIN_LEN && lc_query.contains(&stem) {
        return true;
    }

    // Title: split on non-alphanumeric into candidate tokens, require
    // at least one ≥4-char token of the title to appear in the query.
    if let Some(title) = doc.title.as_deref() {
        let lc_title = title.to_lowercase();
        let any_hit = lc_title
            .split(|c: char| !c.is_alphanumeric())
            .any(|t| t.chars().count() >= DOC_NAME_MIN_LEN && lc_query.contains(t));
        if any_hit {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_exhaustive_intent_via_toate() {
        assert_eq!(detect_intent("vreau toate metodele de segmentare"), CoverageMode::Exhaustive);
    }

    #[test]
    fn detects_exhaustive_intent_via_cheatsheet() {
        assert_eq!(detect_intent("write a cheatsheet of all methods"), CoverageMode::Exhaustive);
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

    // ---- Multi-document scoping tests ----

    fn doc_row(source: &str, title: Option<&str>) -> agent_rs_stores::postgres::IngestedDocumentRow {
        agent_rs_stores::postgres::IngestedDocumentRow {
            document_id: uuid::Uuid::nil(),
            sha256: String::new(),
            source: source.to_string(),
            ingest_status: "complete".to_string(),
            ingest_run_id: uuid::Uuid::nil(),
            failure_reason: None,
            docling_version: None,
            docling_core_version: None,
            schema_version: None,
            n_pages: None,
            n_chunks: None,
            n_tables: None,
            n_pictures: None,
            n_formulas: None,
            language: None,
            title: title.map(String::from),
            mimetype: None,
            parse_latency_ms: None,
            total_ingest_latency_ms: None,
            started_at: chrono::NaiveDateTime::default(),
            completed_at: None,
            updated_at: chrono::NaiveDateTime::default(),
            user_id: "test-user".to_string(),
            minio_object_key: None,
        }
    }

    #[test]
    fn filter_returns_all_docs_when_query_names_none() {
        let docs = vec![
            doc_row("segmentare.pdf", Some("Image Segmentation")),
            doc_row("histograme.pdf", None),
        ];
        let filtered = filter_docs_by_query("ce este metoda Otsu", &docs);
        assert_eq!(filtered.len(), 2, "fallback must preserve all-corpus behaviour");
    }

    #[test]
    fn filter_matches_source_filename_stem() {
        let docs = vec![
            doc_row("segmentare.pdf", None),
            doc_row("histograme.pdf", None),
        ];
        let filtered = filter_docs_by_query("vreau lista din segmentare", &docs);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].source, "segmentare.pdf");
    }

    #[test]
    fn filter_matches_title_substring_case_insensitive() {
        let docs = vec![
            doc_row("a.pdf", Some("Image Segmentation handbook")),
            doc_row("b.pdf", Some("Network Protocols")),
        ];
        let filtered = filter_docs_by_query("enumerate methods from the segmentation paper", &docs);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].source, "a.pdf");
    }

    #[test]
    fn filter_ignores_tokens_shorter_than_four_chars() {
        let docs = vec![doc_row("a.pdf", Some("API Guide"))];
        // "api" is 3 chars — must NOT match alone.
        let filtered = filter_docs_by_query("explain the api", &docs);
        // Fallback kicks in (no doc matched) → returns all docs.
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn filter_returns_multiple_when_query_names_both() {
        let docs = vec![
            doc_row("segmentare.pdf", None),
            doc_row("histograme.pdf", None),
            doc_row("preprocessing.pdf", None),
        ];
        let filtered = filter_docs_by_query(
            "compară metodele din segmentare cu cele din histograme",
            &docs,
        );
        assert_eq!(filtered.len(), 2);
    }
}
