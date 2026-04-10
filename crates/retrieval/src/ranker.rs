//! Reciprocal Rank Fusion (RRF) ranker and evidence pack assembly.
//!
//! # Overview
//!
//! [`ResultRanker`] accepts multiple ranked lists of [`SearchResult`]s (from
//! different retrieval strategies or sources), deduplicates them by ID, fuses
//! their scores with RRF, and returns an [`EvidencePack`] ready for the agent.
//!
//! # RRF Formula
//!
//! For a document `d` appearing at 1-based rank `r` in a list:
//!
//! ```text
//! score(d) += 1.0 / (k + r)
//! ```
//!
//! `k` defaults to 60 (the standard empirical value).  The per-list ranks are
//! derived from `enumerate()` (0-based) and shifted to 1-based internally.
//!
//! # Source Weighting
//!
//! [`ResultRanker::fuse_evidence`] supports heterogeneous source types with
//! configurable per-source weights. Knowledge-base results are weighted at 1.0
//! and web results at 0.8. Before RRF, each document's score is scaled by the
//! list weight so that KB results rank higher for equal positions.

use std::collections::HashMap;

use agent_rs_core::{RetrievalStrategy, Score};
use agent_rs_stores::types::SearchResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::instrument;

// ---- Constants ----

/// Default RRF `k` parameter (the standard empirical value).
const DEFAULT_RRF_K: usize = 60;

/// Threshold above which parallel scoring via rayon is used instead of serial.
const PARALLEL_THRESHOLD: usize = 100;

/// Weight applied to knowledge-base results in [`ResultRanker::fuse_evidence`].
const KB_SOURCE_WEIGHT: f64 = 1.0;

/// Weight applied to web results in [`ResultRanker::fuse_evidence`].
const WEB_SOURCE_WEIGHT: f64 = 0.8;

// ---- EvidencePack ----

/// The output of ranking: a ranked, deduplicated set of documents ready for
/// the agent together with provenance metadata.
///
/// `source_counts` maps a source identifier (e.g. `"qdrant"`, `"brave"`) to
/// the number of documents originating from that source in this pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePack {
    /// Ranked, deduplicated documents ordered by fused score descending.
    pub documents: Vec<SearchResult>,
    /// The original user query that produced this evidence.
    pub query: String,
    /// The retrieval strategy used to produce this pack.
    pub strategy: RetrievalStrategy,
    /// Number of documents contributed by each source identifier.
    pub source_counts: HashMap<String, usize>,
}

impl EvidencePack {
    /// Return `true` if the pack contains no documents.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Return the number of documents in the pack.
    pub fn len(&self) -> usize {
        self.documents.len()
    }
}

// ---- ResultRanker ----

/// Reciprocal Rank Fusion ranker.
///
/// # Example
///
/// ```rust,no_run
/// use agent_rs_retrieval::ranker::ResultRanker;
/// use agent_rs_core::RetrievalStrategy;
///
/// let ranker = ResultRanker::default();
/// // Pass multiple ranked lists; receive a fused EvidencePack.
/// let _pack = ranker.rank(vec![vec![], vec![]], "my query", RetrievalStrategy::Hybrid, 10);
/// ```
#[derive(Debug, Clone)]
pub struct ResultRanker {
    /// The `k` smoothing constant used in the RRF formula.
    ///
    /// Lower values give a stronger preference to top-ranked documents.
    /// The canonical default is 60.
    pub rrf_k: usize,
}

impl Default for ResultRanker {
    fn default() -> Self {
        Self {
            rrf_k: DEFAULT_RRF_K,
        }
    }
}

impl ResultRanker {
    /// Create a new ranker with a custom `k` parameter.
    pub fn new(rrf_k: usize) -> Self {
        Self { rrf_k }
    }

    /// Fuse multiple ranked result lists into a single [`EvidencePack`].
    ///
    /// # Algorithm
    ///
    /// 1. Collect every unique document ID across all lists.
    /// 2. For each document, sum `1.0 / (k + rank)` contributions from every
    ///    list in which it appears (ranks are 1-based).
    /// 3. Sort by fused score descending and truncate to `limit`.
    ///
    /// Rayon parallel iterators are used when the total document count across
    /// all lists exceeds [`PARALLEL_THRESHOLD`].
    ///
    /// # Parameters
    ///
    /// - `result_lists`: Multiple ranked lists from different search strategies.
    /// - `query`: The original user query (stored in the returned pack).
    /// - `strategy`: The retrieval strategy that produced these lists.
    /// - `limit`: Maximum number of documents to include in the pack.
    #[instrument(skip(self, result_lists), fields(
        num_lists = result_lists.len(),
        strategy = %strategy,
        limit
    ))]
    pub fn rank(
        &self,
        result_lists: Vec<Vec<SearchResult>>,
        query: &str,
        strategy: RetrievalStrategy,
        limit: usize,
    ) -> EvidencePack {
        let total_docs: usize = result_lists.iter().map(|l| l.len()).sum();

        // Build a map from document ID to the best (lowest-score-already-set)
        // SearchResult payload so we can reconstruct results after scoring.
        let mut doc_index: HashMap<String, SearchResult> = HashMap::with_capacity(total_docs);

        // Accumulate per-list weighted RRF contributions.
        // Each entry is (id, accumulated_rrf_score).
        let mut rrf_scores: HashMap<String, f64> = HashMap::with_capacity(total_docs);

        for list in &result_lists {
            for (rank_0based, doc) in list.iter().enumerate() {
                // Shift to 1-based rank to avoid division by (k+0).
                let rank_1based = rank_0based + 1;
                let contribution = 1.0 / (self.rrf_k as f64 + rank_1based as f64);

                *rrf_scores.entry(doc.id.clone()).or_insert(0.0) += contribution;

                // Keep the first-seen payload (highest-ranked in first list).
                doc_index.entry(doc.id.clone()).or_insert_with(|| doc.clone());
            }
        }

        let source_counts = build_source_counts(&doc_index);

        let mut fused: Vec<(String, f64)> = if rrf_scores.len() > PARALLEL_THRESHOLD {
            rrf_scores.into_par_iter().collect()
        } else {
            rrf_scores.into_iter().collect()
        };

        // Sort descending by fused score; use id as tiebreaker for determinism.
        fused.sort_unstable_by(|(id_a, score_a), (id_b, score_b)| {
            score_b
                .partial_cmp(score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| id_a.cmp(id_b))
        });

        let documents: Vec<SearchResult> = fused
            .into_iter()
            .take(limit)
            .filter_map(|(id, fused_score)| {
                doc_index.remove(&id).map(|mut doc| {
                    doc.score = Score(fused_score);
                    doc
                })
            })
            .collect();

        EvidencePack {
            documents,
            query: query.to_owned(),
            strategy,
            source_counts,
        }
    }

    /// Fuse knowledge-base and web result lists into a single [`EvidencePack`].
    ///
    /// KB results are weighted at [`KB_SOURCE_WEIGHT`] (1.0) and web results
    /// at [`WEB_SOURCE_WEIGHT`] (0.8). The weight is applied by scaling each
    /// document's initial score before RRF so that high-rank KB documents
    /// remain preferred when KB and web results tie positionally.
    ///
    /// Internally this delegates to [`ResultRanker::rank`] after tagging
    /// each list with its weight via score pre-scaling.
    ///
    /// # Parameters
    ///
    /// - `kb_results`: Ranked results from the knowledge-base retrieval path.
    /// - `web_results`: Ranked results from web search tools.
    /// - `query`: The original user query.
    /// - `strategy`: The retrieval strategy used.
    #[instrument(skip(self, kb_results, web_results), fields(
        kb_count = kb_results.len(),
        web_count = web_results.len(),
        strategy = %strategy,
    ))]
    pub fn fuse_evidence(
        &self,
        kb_results: Vec<SearchResult>,
        web_results: Vec<SearchResult>,
        query: &str,
        strategy: RetrievalStrategy,
    ) -> EvidencePack {
        let limit = kb_results.len().max(web_results.len()).max(1);

        let weighted_kb = apply_source_weight(kb_results, KB_SOURCE_WEIGHT);
        let weighted_web = apply_source_weight(web_results, WEB_SOURCE_WEIGHT);

        self.rank(
            vec![weighted_kb, weighted_web],
            query,
            strategy,
            limit,
        )
    }
}

// ---- Helpers ----

/// Scale each result's score by `weight`.
///
/// Used to give different source types a relative priority boost before
/// the lists are merged and RRF-scored.
fn apply_source_weight(results: Vec<SearchResult>, weight: f64) -> Vec<SearchResult> {
    results
        .into_iter()
        .map(|mut r| {
            r.score = Score(r.score.0 * weight);
            r
        })
        .collect()
}

/// Build a map counting how many documents originate from each source.
///
/// The source is extracted from the `"source"` field in the document payload,
/// falling back to `"unknown"` when the field is absent or not a string.
fn build_source_counts(doc_index: &HashMap<String, SearchResult>) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for doc in doc_index.values() {
        let source = doc
            .payload
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_owned();
        *counts.entry(source).or_insert(0) += 1;
    }
    counts
}
