//! Retrieval evaluation harness for agent-rs.
//!
//! Measures retrieval quality across all supported `RetrievalStrategy` variants
//! using a JSON-defined test dataset. Produces per-strategy, per-difficulty-tier,
//! and per-question-type metrics (Hit@K, MRR, NDCG@K, text-overlap, latency) and
//! writes a structured JSON report to `test_results/eval_report_{timestamp}.json`.
//!
//! # Usage
//!
//! ```bash
//! # Ensure infrastructure is up (Qdrant ports 1050/1051, Neo4j 1061, Postgres 1054).
//! # Optional: EMBEDDER_TYPE=openai (uses text-embedding-3-large).
//! cargo run --example eval_harness
//!
//! # Use a different dataset (v2 expanded suite):
//! EVAL_DATASET=eval/test_questions_v2.json cargo run --example eval_harness
//!
//! # To save the current run as the baseline for future comparisons:
//! EVAL_SAVE_BASELINE=1 cargo run --example eval_harness
//! ```
//!
//! # Output
//!
//! - `test_results/eval_report_{timestamp}.json` — full structured report
//! - `test_results/baseline_eval.json` — written when EVAL_SAVE_BASELINE=1
//! - Console summary table of per-strategy metrics
//! - Per-difficulty-tier and per-question-type breakdown tables
//! - Baseline delta report (if `baseline_eval.json` exists)
//!
//! # Dataset
//!
//! Reads `eval/baseline_v1.json` by default (or `EVAL_DATASET` env var path).
//! Each question declares its `type` (single_hop, multi_hop, lexical_exact,
//! summary_only, temporal, conceptual, negation, comparison, disambiguation,
//! bridge) and `difficulty` (easy, medium, hard), plus expected `doc_ids`
//! and optional `expected_text_contains`.
//! Empty `expected_doc_ids` falls back to text-only fuzzy evaluation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

use agent_rs::embedder::build_embedder;
use agent_rs::ingest::{IngestFeatures, build_ingestion_pipeline};
use agent_rs::router::{RouterFeatures, build_router};
use agent_rs::stores::connect_stores;
use agent_rs_core::config::Settings;
use agent_rs_core::types::{CollectionName, RetrievalStrategy};
use agent_rs_stores::traits::GraphStore;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Strategies evaluated for every question.
///
/// Graph-dependent strategies that are still stubs are intentionally excluded
/// here to keep the baseline meaningful. They can be added once implemented.
const EVAL_STRATEGIES: &[RetrievalStrategy] = &[
    RetrievalStrategy::Semantic,
    RetrievalStrategy::Keyword,
    RetrievalStrategy::Hybrid,
    RetrievalStrategy::Graph,
    RetrievalStrategy::GraphHybrid,
];

/// Maximum rank cutoffs for Hit@K / NDCG@K metrics.
const NDCG_CUTOFF: usize = 5;

/// Directory containing the evaluation dataset — used only for path resolution
/// when the EVAL_DATASET env var supplies a bare filename without a directory.
const EVAL_DIR: &str = "eval";

/// Default dataset file name (baseline v1 — 15 questions).
/// Override at runtime with: `EVAL_DATASET=eval/test_questions_v2.json`.
const DATASET_FILE: &str = "baseline_v1.json";

/// Directory where reports are written.
const RESULTS_DIR: &str = "test_results";

/// Filename for the persisted baseline report.
const BASELINE_FILENAME: &str = "baseline_eval.json";

/// Small epsilon used when comparing deltas to decide if a change is material.
const DELTA_EPSILON: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Dataset types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct Dataset {
    #[serde(default)]
    #[allow(dead_code)]
    dataset_name: String,
    #[serde(default)]
    #[allow(dead_code)]
    description: String,
    /// Optional: allowed enum values for the `type` field.
    /// When present, the harness validates every question's type against this list.
    #[serde(default)]
    allowed_types: Vec<String>,
    /// Optional: allowed enum values for the `difficulty` field.
    /// When present, the harness validates every question's difficulty against this list.
    #[serde(default)]
    allowed_difficulties: Vec<String>,
    documents: Vec<DatasetDocument>,
    #[serde(default = "default_limit")]
    limit: usize,
    questions: Vec<TestQuestion>,
}

fn default_limit() -> usize {
    5
}

#[derive(Debug, Clone, Deserialize)]
struct DatasetDocument {
    /// Relative path to the source document (resolved against the dataset file location).
    path: String,
    /// Unique document identifier (used as the ingestion doc_id).
    doc_id: String,
    /// Human-readable source tag (e.g. "legal_code_ro").
    #[serde(default)]
    source: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TestQuestion {
    id: String,
    /// Difficulty tier: easy | medium | hard.
    /// Defaults to "easy" when absent (backward-compatible with v1 datasets).
    #[serde(default = "default_difficulty")]
    difficulty: String,
    #[serde(rename = "type")]
    question_type: String,
    question: String,
    #[serde(default)]
    expected_doc_ids: Vec<String>,
    #[serde(default)]
    expected_text_contains: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    notes: String,
}

fn default_difficulty() -> String {
    "easy".to_owned()
}

// ---------------------------------------------------------------------------
// Per-result metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionStrategyMetrics {
    /// Whether any expected_doc_id appeared at rank 1.
    hit_at_1: bool,
    /// Whether any expected_doc_id appeared in top 3.
    hit_at_3: bool,
    /// Whether any expected_doc_id appeared in top 5.
    hit_at_5: bool,
    /// 1-based rank of the first matching expected_doc_id, or `None` if not found.
    first_relevant_rank: Option<usize>,
    /// Reciprocal rank (1 / first_relevant_rank), 0.0 if none.
    reciprocal_rank: f64,
    /// NDCG@5.
    ndcg_at_5: f64,
    /// Fraction of `expected_text_contains` terms found across top results (0..=1).
    text_overlap: f64,
    /// Whether a fuzzy text match was observed (at least one expected term found).
    text_hit: bool,
    /// Query duration in milliseconds.
    latency_ms: u64,
    /// Number of results returned (capped by router limit).
    results_count: usize,
    /// Flag indicating the metric is fuzzy (expected_doc_ids was empty).
    fuzzy_only: bool,
    /// Optional error message if the query failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// Top document IDs (truncated to first 5) — useful for debugging.
    top_ids: Vec<String>,
}

impl QuestionStrategyMetrics {
    fn failed(error: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            hit_at_1: false,
            hit_at_3: false,
            hit_at_5: false,
            first_relevant_rank: None,
            reciprocal_rank: 0.0,
            ndcg_at_5: 0.0,
            text_overlap: 0.0,
            text_hit: false,
            latency_ms,
            results_count: 0,
            fuzzy_only: false,
            error: Some(error.into()),
            top_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerQuestionReport {
    question_id: String,
    difficulty: String,
    #[serde(rename = "type")]
    question_type: String,
    question: String,
    strategies: HashMap<String, QuestionStrategyMetrics>,
}

// ---------------------------------------------------------------------------
// Aggregate metrics per strategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct StrategyAggregate {
    #[serde(rename = "hit@1")]
    hit_at_1: f64,
    #[serde(rename = "hit@3")]
    hit_at_3: f64,
    #[serde(rename = "hit@5")]
    hit_at_5: f64,
    mrr: f64,
    #[serde(rename = "ndcg@5")]
    ndcg_at_5: f64,
    avg_text_overlap: f64,
    avg_latency_ms: f64,
    error_rate: f64,
    /// Number of questions contributing to this aggregate.
    question_count: usize,
    /// How many questions were fuzzy-only (no expected_doc_ids).
    fuzzy_only_count: usize,
}

/// Breakdown of aggregate metrics sliced by a single dimension (tier or type).
///
/// Keys in `by_label` are the distinct dimension values (e.g. "easy", "hard",
/// "single_hop", "conceptual"). Each value is itself a `HashMap<strategy_name,
/// StrategyAggregate>`, giving full per-strategy numbers for that slice.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DimensionBreakdown {
    #[serde(flatten)]
    by_label: HashMap<String, HashMap<String, StrategyAggregate>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvalReport {
    run_id: String,
    timestamp: String,
    embedder: String,
    total_questions: usize,
    per_strategy: HashMap<String, StrategyAggregate>,
    /// Metrics broken down by difficulty tier (easy / medium / hard).
    per_tier: DimensionBreakdown,
    /// Metrics broken down by question type (single_hop, conceptual, …).
    per_type: DimensionBreakdown,
    per_question: Vec<PerQuestionReport>,
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

fn load_dataset() -> Result<(Dataset, PathBuf)> {
    // EVAL_DATASET env var overrides the default dataset file.
    // Accepts either an absolute path or a path relative to the working directory.
    // Example: EVAL_DATASET=eval/test_questions_v2.json
    let dataset_path = if let Ok(env_path) = std::env::var("EVAL_DATASET") {
        let p = PathBuf::from(&env_path);
        if p.exists() {
            p
        } else {
            // Try relative to the workspace root as a convenience.
            let alt = PathBuf::from("agent-rs").join(&env_path);
            if alt.exists() {
                alt
            } else {
                anyhow::bail!(
                    "EVAL_DATASET path not found: {} (also tried {})",
                    p.display(),
                    alt.display()
                );
            }
        }
    } else {
        // Default: eval/baseline_v1.json (relative to CWD or workspace root).
        let default_path = PathBuf::from(EVAL_DIR).join(DATASET_FILE);
        if default_path.exists() {
            default_path
        } else {
            PathBuf::from("agent-rs").join(EVAL_DIR).join(DATASET_FILE)
        }
    };

    let raw = std::fs::read_to_string(&dataset_path)
        .with_context(|| format!("read dataset file {}", dataset_path.display()))?;
    let dataset: Dataset = serde_json::from_str(&raw)
        .with_context(|| format!("parse dataset file {}", dataset_path.display()))?;

    anyhow::ensure!(
        !dataset.documents.is_empty(),
        "dataset must declare at least one document"
    );
    anyhow::ensure!(
        !dataset.questions.is_empty(),
        "dataset must declare at least one question"
    );

    Ok((dataset, dataset_path))
}

/// Validate question `type` and `difficulty` against the dataset's declared
/// allowed values. Logs a warning for each violation but does not abort —
/// unknown values are still evaluated, just flagged.
fn validate_question_schema(dataset: &Dataset) {
    for q in &dataset.questions {
        if !dataset.allowed_types.is_empty() && !dataset.allowed_types.contains(&q.question_type) {
            warn!(
                question_id = %q.id,
                found = %q.question_type,
                allowed = ?dataset.allowed_types,
                "unknown question type — not in allowed_types"
            );
        }
        if !dataset.allowed_difficulties.is_empty()
            && !dataset.allowed_difficulties.contains(&q.difficulty)
        {
            warn!(
                question_id = %q.id,
                found = %q.difficulty,
                allowed = ?dataset.allowed_difficulties,
                "unknown difficulty — not in allowed_difficulties"
            );
        }
    }
}

/// Resolve a document path declared in the dataset relative to the dataset file.
fn resolve_doc_path(dataset_path: &Path, relative: &str) -> Result<PathBuf> {
    let base = dataset_path
        .parent()
        .context("dataset file has no parent directory")?;
    let candidate = base.join(relative);
    if candidate.exists() {
        return Ok(candidate);
    }
    // Fallback: try the relative path from CWD.
    let cwd_candidate = PathBuf::from(relative);
    if cwd_candidate.exists() {
        return Ok(cwd_candidate);
    }
    anyhow::bail!(
        "document not found: {} (also tried {})",
        candidate.display(),
        cwd_candidate.display()
    );
}

// ---------------------------------------------------------------------------
// Metric computation
// ---------------------------------------------------------------------------

/// Compute metrics for a single query result against the expected answers.
fn compute_metrics(
    question: &TestQuestion,
    documents: &[agent_rs_stores::types::SearchResult],
    latency_ms: u64,
) -> QuestionStrategyMetrics {
    let results_count = documents.len();
    let top_ids: Vec<String> = documents.iter().take(5).map(|d| d.id.clone()).collect();

    // Determine the rank of the first relevant document.
    // Relevance = id OR doc_id payload field matches any expected_doc_id.
    let expected: Vec<&str> = question
        .expected_doc_ids
        .iter()
        .map(String::as_str)
        .collect();
    let fuzzy_only = expected.is_empty();

    // When expected_doc_ids is empty, fall back to text-based ranking:
    // find the first document whose chunk_text contains ANY of the
    // expected_text_contains terms. This gives real Hit@k/MRR signal
    // even without manually-annotated expected_doc_ids.
    let first_relevant_rank = if fuzzy_only {
        documents
            .iter()
            .enumerate()
            .find(|(_, doc)| doc_matches_text(doc, &question.expected_text_contains))
            .map(|(idx, _)| idx + 1)
    } else {
        documents
            .iter()
            .enumerate()
            .find(|(_, doc)| is_relevant_doc(doc, &expected))
            .map(|(idx, _)| idx + 1)
    };

    let hit_at_1 = first_relevant_rank == Some(1);
    let hit_at_3 = first_relevant_rank.is_some_and(|r| r <= 3);
    let hit_at_5 = first_relevant_rank.is_some_and(|r| r <= 5);
    let reciprocal_rank = first_relevant_rank.map_or(0.0, |r| 1.0 / r as f64);

    // NDCG@5 — binary relevance (1 for hit, 0 otherwise).
    // In fuzzy-only mode, use text matching instead of expected_doc_ids.
    let ndcg_at_5 = if fuzzy_only {
        compute_ndcg_at_k_text(documents, &question.expected_text_contains, NDCG_CUTOFF)
    } else {
        compute_ndcg_at_k(documents, &expected, NDCG_CUTOFF)
    };

    // Text overlap: fraction of expected_text_contains terms found across
    // concatenated top-5 payload chunk_text (case-insensitive).
    let (text_overlap, text_hit) =
        compute_text_overlap(documents, &question.expected_text_contains);

    QuestionStrategyMetrics {
        hit_at_1,
        hit_at_3,
        hit_at_5,
        first_relevant_rank,
        reciprocal_rank,
        ndcg_at_5,
        text_overlap,
        text_hit,
        latency_ms,
        results_count,
        fuzzy_only,
        error: None,
        top_ids,
    }
}

/// Check if a document's chunk_text contains ANY of the expected terms.
/// Case-insensitive substring match. Used for fuzzy-only evaluation when
/// `expected_doc_ids` is empty.
fn doc_matches_text(doc: &agent_rs_stores::types::SearchResult, terms: &[String]) -> bool {
    if terms.is_empty() {
        return false;
    }
    let Some(text) = doc.payload.get("chunk_text").and_then(|v| v.as_str()) else {
        return false;
    };
    let haystack = text.to_ascii_lowercase();
    terms
        .iter()
        .any(|t| haystack.contains(&t.to_ascii_lowercase()))
}

/// Check if a document matches any expected doc_id.
///
/// Matches against the raw `id` field as well as common payload identifier
/// fields (`doc_id`, `article_number`, `chunk_id`).
fn is_relevant_doc(doc: &agent_rs_stores::types::SearchResult, expected: &[&str]) -> bool {
    if expected.is_empty() {
        return false;
    }
    if expected.contains(&doc.id.as_str()) {
        return true;
    }
    let payload = &doc.payload;
    for key in ["doc_id", "chunk_id", "article_number", "id"] {
        if let Some(v) = payload.get(key).and_then(|v| v.as_str())
            && expected.contains(&v)
        {
            return true;
        }
    }
    false
}

/// Compute NDCG@k with binary relevance judgements.
fn compute_ndcg_at_k(
    documents: &[agent_rs_stores::types::SearchResult],
    expected: &[&str],
    k: usize,
) -> f64 {
    if expected.is_empty() {
        return 0.0;
    }

    // DCG: sum of rel_i / log2(i+1) for i in 1..=k.
    let mut dcg = 0.0;
    for (idx, doc) in documents.iter().take(k).enumerate() {
        if is_relevant_doc(doc, expected) {
            let rank = (idx + 1) as f64; // 1-based
            dcg += 1.0 / (rank + 1.0).log2();
        }
    }

    // Ideal DCG: ideal ranking has all relevant docs at the top.
    let ideal_hits = expected.len().min(k);
    if ideal_hits == 0 {
        return 0.0;
    }
    let idcg: f64 = (1..=ideal_hits)
        .map(|i| 1.0 / ((i as f64) + 1.0).log2())
        .sum();

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// NDCG@k using text-match relevance instead of expected_doc_ids.
/// Binary relevance: 1 if doc contains any expected term, 0 otherwise.
fn compute_ndcg_at_k_text(
    documents: &[agent_rs_stores::types::SearchResult],
    expected_terms: &[String],
    k: usize,
) -> f64 {
    if expected_terms.is_empty() {
        return 0.0;
    }

    let mut dcg = 0.0;
    let mut hit_count = 0;
    for (idx, doc) in documents.iter().take(k).enumerate() {
        if doc_matches_text(doc, expected_terms) {
            let rank = (idx + 1) as f64;
            dcg += 1.0 / (rank + 1.0).log2();
            hit_count += 1;
        }
    }

    if hit_count == 0 {
        return 0.0;
    }

    // Ideal DCG: all hits at the top.
    let ideal_hits = hit_count.min(k);
    let idcg: f64 = (1..=ideal_hits)
        .map(|i| 1.0 / ((i as f64) + 1.0).log2())
        .sum();

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Compute the fraction of expected_text_contains terms present anywhere in
/// the concatenated top-5 chunk texts (case-insensitive).
///
/// Returns `(overlap, text_hit)`. `text_hit` is true if at least one expected
/// term was found.
fn compute_text_overlap(
    documents: &[agent_rs_stores::types::SearchResult],
    expected_terms: &[String],
) -> (f64, bool) {
    if expected_terms.is_empty() {
        return (0.0, false);
    }
    let haystack: String = documents
        .iter()
        .take(5)
        .filter_map(|doc| {
            doc.payload
                .get("chunk_text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_ascii_lowercase())
        })
        .collect::<Vec<_>>()
        .join(" ");

    let matched = expected_terms
        .iter()
        .filter(|term| haystack.contains(&term.to_ascii_lowercase()))
        .count();

    let overlap = matched as f64 / expected_terms.len() as f64;
    (overlap, matched > 0)
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

fn aggregate_per_strategy(reports: &[PerQuestionReport]) -> HashMap<String, StrategyAggregate> {
    let mut out: HashMap<String, StrategyAggregate> = HashMap::new();

    let total = reports.len();
    if total == 0 {
        return out;
    }

    // Collect unique strategy names.
    let strategy_names: Vec<String> = reports
        .iter()
        .flat_map(|r| r.strategies.keys().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for strategy in strategy_names {
        let mut hit1 = 0.0_f64;
        let mut hit3 = 0.0_f64;
        let mut hit5 = 0.0_f64;
        let mut mrr_sum = 0.0_f64;
        let mut ndcg_sum = 0.0_f64;
        let mut overlap_sum = 0.0_f64;
        let mut latency_sum = 0.0_f64;
        let mut error_count = 0_usize;
        let mut fuzzy_count = 0_usize;
        let mut counted = 0_usize;

        for report in reports {
            let Some(m) = report.strategies.get(&strategy) else {
                continue;
            };
            counted += 1;
            if m.error.is_some() {
                error_count += 1;
                latency_sum += m.latency_ms as f64;
                continue;
            }
            if m.fuzzy_only {
                fuzzy_count += 1;
            }
            if m.hit_at_1 {
                hit1 += 1.0;
            }
            if m.hit_at_3 {
                hit3 += 1.0;
            }
            if m.hit_at_5 {
                hit5 += 1.0;
            }
            mrr_sum += m.reciprocal_rank;
            ndcg_sum += m.ndcg_at_5;
            overlap_sum += m.text_overlap;
            latency_sum += m.latency_ms as f64;
        }

        let divisor = counted.max(1) as f64;
        out.insert(
            strategy,
            StrategyAggregate {
                hit_at_1: hit1 / divisor,
                hit_at_3: hit3 / divisor,
                hit_at_5: hit5 / divisor,
                mrr: mrr_sum / divisor,
                ndcg_at_5: ndcg_sum / divisor,
                avg_text_overlap: overlap_sum / divisor,
                avg_latency_ms: latency_sum / divisor,
                error_rate: error_count as f64 / divisor,
                question_count: counted,
                fuzzy_only_count: fuzzy_count,
            },
        );
    }

    out
}

/// Aggregate metrics broken down by a single dimension (difficulty tier or
/// question type). The `label_fn` extracts the dimension value from a report row.
fn aggregate_by_dimension(
    reports: &[PerQuestionReport],
    label_fn: impl Fn(&PerQuestionReport) -> &str,
) -> DimensionBreakdown {
    // Group report rows by label, cloning into owned vecs so we can call
    // aggregate_per_strategy which expects &[PerQuestionReport].
    let mut groups: HashMap<String, Vec<PerQuestionReport>> = HashMap::new();
    for r in reports {
        groups
            .entry(label_fn(r).to_owned())
            .or_default()
            .push(r.clone());
    }

    let mut by_label: HashMap<String, HashMap<String, StrategyAggregate>> = HashMap::new();
    for (label, group) in groups {
        by_label.insert(label, aggregate_per_strategy(&group));
    }

    DimensionBreakdown { by_label }
}

// ---------------------------------------------------------------------------
// Console reporting
// ---------------------------------------------------------------------------

fn print_summary_table(report: &EvalReport) {
    println!();
    println!("=== Eval Results ===");
    println!("Run ID:    {}", report.run_id);
    println!("Embedder:  {}", report.embedder);
    println!("Questions: {}", report.total_questions);
    println!();
    println!(
        "{:<18} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10}",
        "Strategy", "Hit@1", "Hit@3", "Hit@5", "MRR", "NDCG@5", "TextOvr", "Latency"
    );
    println!("{}", "-".repeat(94));

    // Keep a stable, deterministic order for the console output.
    let mut strategy_names: Vec<&String> = report.per_strategy.keys().collect();
    strategy_names.sort();

    for name in strategy_names {
        let agg = &report.per_strategy[name];
        println!(
            "{:<18} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10}",
            name,
            format_pct(agg.hit_at_1),
            format_pct(agg.hit_at_3),
            format_pct(agg.hit_at_5),
            format!("{:.3}", agg.mrr),
            format!("{:.3}", agg.ndcg_at_5),
            format_pct(agg.avg_text_overlap),
            format!("{:.0}ms", agg.avg_latency_ms),
        );
    }
    println!();
}

fn format_pct(v: f64) -> String {
    format!("{:>5.1}%", v * 100.0)
}

/// Print a breakdown table (per-tier or per-type) for a single strategy.
///
/// `strategy_name` is used to look up which aggregate to display for each label.
/// If the strategy is absent for a given label (e.g. the label has no questions
/// that ran that strategy) the row is skipped.
fn print_dimension_table(title: &str, breakdown: &DimensionBreakdown, strategy_name: &str) {
    if breakdown.by_label.is_empty() {
        return;
    }

    // Only show the table if at least one label has data for this strategy.
    let has_data = breakdown
        .by_label
        .values()
        .any(|m| m.contains_key(strategy_name));
    if !has_data {
        return;
    }

    println!("  {} — strategy: {}", title, strategy_name);
    println!(
        "  {:<20} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}",
        "Label", "Hit@1", "Hit@3", "Hit@5", "MRR", "NDCG@5"
    );
    println!("  {}", "-".repeat(74));

    let mut labels: Vec<&String> = breakdown.by_label.keys().collect();
    labels.sort();

    for label in labels {
        let Some(agg) = breakdown.by_label[label].get(strategy_name) else {
            continue;
        };
        println!(
            "  {:<20} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}",
            label,
            format_pct(agg.hit_at_1),
            format_pct(agg.hit_at_3),
            format_pct(agg.hit_at_5),
            format!("{:.3}", agg.mrr),
            format!("{:.3}", agg.ndcg_at_5),
        );
    }
    println!();
}

/// Print per-tier and per-type breakdowns for the best strategy (highest MRR).
fn print_dimension_breakdowns(report: &EvalReport) {
    // Pick the strategy with the highest overall MRR to use as the reference
    // column in the breakdown tables.
    let best_strategy = report
        .per_strategy
        .iter()
        .max_by(|a, b| {
            a.1.mrr
                .partial_cmp(&b.1.mrr)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, _)| name.as_str())
        .unwrap_or("hybrid");

    println!("=== Per-Difficulty-Tier Breakdown ===");
    print_dimension_table("Difficulty tiers", &report.per_tier, best_strategy);

    println!("=== Per-Question-Type Breakdown ===");
    print_dimension_table("Question types", &report.per_type, best_strategy);
}

fn print_baseline_delta(current: &EvalReport, baseline: &EvalReport) {
    println!("=== vs baseline ({}) ===", baseline.run_id);

    let mut names: Vec<&String> = current.per_strategy.keys().collect();
    names.sort();

    for name in names {
        let Some(curr) = current.per_strategy.get(name) else {
            continue;
        };
        let Some(base) = baseline.per_strategy.get(name) else {
            println!("  {name}: (new — no baseline)");
            continue;
        };
        let d_hit1 = curr.hit_at_1 - base.hit_at_1;
        let d_hit3 = curr.hit_at_3 - base.hit_at_3;
        let d_mrr = curr.mrr - base.mrr;
        let d_ndcg = curr.ndcg_at_5 - base.ndcg_at_5;

        println!(
            "  {:<16} Hit@1 {}  Hit@3 {}  MRR {}  NDCG@5 {}",
            name,
            format_delta_pct(d_hit1),
            format_delta_pct(d_hit3),
            format_delta(d_mrr),
            format_delta(d_ndcg),
        );
    }
    println!();
}

fn format_delta_pct(d: f64) -> String {
    if d.abs() < DELTA_EPSILON {
        format!("{:>+6.1}%  =", 0.0)
    } else if d > 0.0 {
        format!("{:>+6.1}%  UP", d * 100.0)
    } else {
        format!("{:>+6.1}%  DN", d * 100.0)
    }
}

fn format_delta(d: f64) -> String {
    if d.abs() < DELTA_EPSILON {
        format!("{:>+.3} =", 0.0)
    } else if d > 0.0 {
        format!("{:>+.3} UP", d)
    } else {
        format!("{:>+.3} DN", d)
    }
}

// ---------------------------------------------------------------------------
// Cleanup (fresh state per run)
// ---------------------------------------------------------------------------

async fn reset_collection(collection: &CollectionName, settings: &Settings) -> Result<()> {
    let mut builder = qdrant_client::Qdrant::from_url(&settings.qdrant.grpc_url());
    if !settings.qdrant.api_key.is_empty() {
        builder = builder.api_key(settings.qdrant.api_key.clone());
    }
    let raw_client = builder.build().context("build raw Qdrant client")?;

    if raw_client
        .collection_exists(collection.as_str())
        .await
        .unwrap_or(false)
    {
        info!(collection = %collection, "deleting existing collection (clean state)");
        raw_client
            .delete_collection(collection.as_str())
            .await
            .context("delete existing Qdrant collection")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::from_filename("../.env");
    let _ = dotenvy::dotenv();

    let settings = Settings::load().context("failed to load configuration")?;
    agent_rs_core::logging::init_logging(false);

    let run_id = Uuid::new_v4().to_string();
    info!(run_id = %run_id, "eval harness starting");

    // 1. Load dataset.
    let (dataset, dataset_path) = load_dataset().context("load test dataset")?;
    info!(
        questions = dataset.questions.len(),
        docs = dataset.documents.len(),
        dataset = %dataset_path.display(),
        "dataset loaded"
    );
    validate_question_schema(&dataset);
    for q in &dataset.questions {
        if q.expected_doc_ids.is_empty() {
            warn!(
                question_id = %q.id,
                "expected_doc_ids is empty — fuzzy text-only evaluation will be used"
            );
        }
    }

    // 2. Connect stores + build embedder.
    let stores = connect_stores(&settings).await?;
    let (embedder, embedder_label) = build_embedder().await?;
    info!(embedder = %embedder_label, "embedder ready");

    // 3. Collection names + pre-run reset (eval-specific: each run starts clean).
    let collection = CollectionName::from_string("eval_harness".into());
    let triplet_collection = CollectionName::from_string("eval_harness_triplets".into());
    let summaries_collection = CollectionName::from_string("eval_harness_summaries".into());
    reset_collection(&collection, &settings).await?;
    reset_collection(&triplet_collection, &settings).await?;
    reset_collection(&summaries_collection, &settings).await?;
    for doc in &dataset.documents {
        if let Err(e) = stores.neo4j.delete_node(&doc.doc_id).await {
            warn!(doc_id = %doc.doc_id, error = %e, "pre-run graph node cleanup failed (ignored)");
        }
    }

    // 4. Build ingestion pipeline + ingest all dataset documents.
    let pipeline = build_ingestion_pipeline(
        &settings,
        &stores,
        embedder.clone(),
        &collection,
        &triplet_collection,
        &summaries_collection,
        &IngestFeatures::from_env(),
        None, // metrics: eval harness does not instrument graph extraction
    )
    .await?;

    for doc in &dataset.documents {
        let doc_path = resolve_doc_path(&dataset_path, &doc.path)
            .with_context(|| format!("resolve document path {}", doc.path))?;
        info!(path = %doc_path.display(), doc_id = %doc.doc_id, "ingesting document");

        let is_pdf = doc_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("pdf"));

        let result = if is_pdf {
            let bytes = std::fs::read(&doc_path)
                .with_context(|| format!("read PDF {}", doc_path.display()))?;
            pipeline
                .ingest_pdf(&bytes, &doc.doc_id, &doc.source)
                .await
                .context("ingest_pdf")?
        } else {
            let text = std::fs::read_to_string(&doc_path)
                .with_context(|| format!("read text document {}", doc_path.display()))?;
            pipeline
                .ingest_text(&text, &doc.doc_id, &doc.source)
                .await
                .context("ingest_text")?
        };

        info!(
            doc_id = %result.doc_id,
            chunks = result.chunks_created,
            vectors = result.vectors_upserted,
            graph_nodes = result.graph_nodes_created,
            graph_edges = result.graph_edges_created,
            triplets = result.triplets_upserted,
            summaries = result.summaries_created,
            duration_ms = result.duration_ms,
            "ingestion complete"
        );
    }

    // 5. Build retrieval router (features gated by EVAL_SKIP_* env vars).
    let router = build_router(
        &settings,
        &stores,
        embedder,
        &collection,
        &triplet_collection,
        &summaries_collection,
        &RouterFeatures::from_env(),
        // Eval harness does not use a live OTel provider, so no metrics are wired.
        Arc::new(agent_rs_observability::metrics::RetrievalMetrics::new(
            "eval",
        )),
    )
    .await?;

    // 6. Run each question through every strategy.
    info!("--- EVAL PHASE ---");
    let mut per_question_reports: Vec<PerQuestionReport> =
        Vec::with_capacity(dataset.questions.len());

    for question in &dataset.questions {
        info!(id = %question.id, q_type = %question.question_type, "evaluating question");

        let mut strategies_map: HashMap<String, QuestionStrategyMetrics> = HashMap::new();

        for &strategy in EVAL_STRATEGIES {
            let started = Instant::now();
            let metrics = match router
                .retrieve(&question.question, strategy, dataset.limit)
                .await
            {
                Ok(pack) => {
                    let latency = started.elapsed().as_millis() as u64;
                    compute_metrics(question, &pack.documents, latency)
                }
                Err(e) => {
                    let latency = started.elapsed().as_millis() as u64;
                    warn!(
                        id = %question.id, strategy = %strategy, error = %e,
                        "strategy failed"
                    );
                    QuestionStrategyMetrics::failed(e.to_string(), latency)
                }
            };
            strategies_map.insert(strategy.as_db_str().to_owned(), metrics);
        }

        per_question_reports.push(PerQuestionReport {
            question_id: question.id.clone(),
            difficulty: question.difficulty.clone(),
            question_type: question.question_type.clone(),
            question: question.question.clone(),
            strategies: strategies_map,
        });
    }

    // 7. Aggregate + build report.
    let per_strategy = aggregate_per_strategy(&per_question_reports);
    let per_tier = aggregate_by_dimension(&per_question_reports, |r| r.difficulty.as_str());
    let per_type = aggregate_by_dimension(&per_question_reports, |r| r.question_type.as_str());
    let report = EvalReport {
        run_id: run_id.clone(),
        timestamp: Utc::now().to_rfc3339(),
        embedder: embedder_label,
        total_questions: dataset.questions.len(),
        per_strategy,
        per_tier,
        per_type,
        per_question: per_question_reports,
    };

    print_summary_table(&report);
    print_dimension_breakdowns(&report);

    // 8. Compare against baseline if present.
    let results_dir = PathBuf::from(RESULTS_DIR);
    std::fs::create_dir_all(&results_dir).context("create test_results directory")?;

    let baseline_path = results_dir.join(BASELINE_FILENAME);
    if baseline_path.exists() {
        match std::fs::read_to_string(&baseline_path)
            .map_err(anyhow::Error::from)
            .and_then(|s| serde_json::from_str::<EvalReport>(&s).map_err(anyhow::Error::from))
        {
            Ok(baseline) => print_baseline_delta(&report, &baseline),
            Err(e) => warn!(error = %e, path = %baseline_path.display(), "failed to load baseline"),
        }
    } else {
        info!(
            path = %baseline_path.display(),
            "no baseline found — run with EVAL_SAVE_BASELINE=1 to create one"
        );
    }

    // 9. Write timestamped report.
    let ts = Utc::now().format("%Y%m%dT%H%M%S");
    let report_path = results_dir.join(format!("eval_report_{ts}.json"));
    let json = serde_json::to_string_pretty(&report).context("serialize report")?;
    std::fs::write(&report_path, &json)
        .with_context(|| format!("write {}", report_path.display()))?;
    info!(path = %report_path.display(), "eval report written");
    println!("Report written to: {}", report_path.display());

    // 10. Optionally save as baseline.
    let save_baseline = std::env::var("EVAL_SAVE_BASELINE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false);
    if save_baseline {
        std::fs::write(&baseline_path, &json)
            .with_context(|| format!("write baseline {}", baseline_path.display()))?;
        info!(path = %baseline_path.display(), "baseline saved");
        println!("Baseline saved to: {}", baseline_path.display());
    }

    // 11. Post-run cleanup so repeated runs stay clean.
    if let Err(e) = reset_collection(&collection, &settings).await {
        warn!(error = %e, "post-run collection cleanup failed (non-fatal)");
    }
    for doc in &dataset.documents {
        if let Err(e) = stores.neo4j.delete_node(&doc.doc_id).await {
            warn!(doc_id = %doc.doc_id, error = %e, "post-run graph node cleanup failed");
        }
    }

    info!(run_id = %run_id, "eval harness complete");
    Ok(())
}
