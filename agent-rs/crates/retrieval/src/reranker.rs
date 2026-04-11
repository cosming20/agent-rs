//! Cross-encoder reranking stage for the retrieval pipeline.
//!
//! # Overview
//!
//! [`CrossEncoderReranker`] takes the top-N results from RRF fusion and
//! reranks them against the original query using a local ONNX cross-encoder
//! model via `fastembed`. Unlike bi-encoder vector similarity (used in the
//! primary retrieval stage), a cross-encoder scores query+document pairs
//! jointly, producing more accurate relevance estimates at the cost of
//! higher latency.
//!
//! # Thread-safety contract
//!
//! `fastembed`'s [`TextRerank`] is **not** `Send + Sync` because the
//! underlying ONNX session holds non-`Send` state. It is wrapped in
//! `Arc<Mutex<TextRerank>>` and every scoring call is offloaded via
//! [`tokio::task::spawn_blocking`] so the async runtime is never blocked.
//!
//! # Model
//!
//! Uses `RerankerModel::BGERerankerBase` — a multilingual cross-encoder
//! (approx. 278 M params) with strong cross-lingual transfer, making it
//! appropriate for the multilingual retrieval workloads in agent-rs.
//!
//! Model weights are downloaded on first use to the fastembed cache directory
//! (`~/.cache/fastembed` by default) and reused on subsequent starts.
//!
//! # Integration
//!
//! The reranker is optional on [`crate::router::RetrievalRouter`]. When
//! configured, `retrieve` passes the top-50 RRF results through
//! [`CrossEncoderReranker::rerank`] before truncating to the caller's `limit`.

use std::sync::{Arc, Mutex};

use agent_rs_core::Score;
use agent_rs_stores::types::SearchResult;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use tracing::instrument;

use crate::error::RetrievalError;

// ---- Constants ----

/// Number of top RRF results forwarded to the cross-encoder.
///
/// Taking 50 gives the reranker enough candidates to improve recall while
/// keeping per-request latency bounded. Values above ~100 provide
/// diminishing returns for most cross-encoders.
pub const RERANK_CANDIDATE_COUNT: usize = 50;

/// Payload key that carries the chunk text used as the reranker document body.
const CHUNK_TEXT_KEY: &str = "chunk_text";

// ---- CrossEncoderReranker ----

/// Cross-encoder reranker backed by a local fastembed ONNX model.
///
/// # Construction
///
/// ```rust,no_run
/// # async fn example() -> anyhow::Result<()> {
/// use agent_rs_retrieval::reranker::CrossEncoderReranker;
/// let reranker = CrossEncoderReranker::try_new().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct CrossEncoderReranker {
    /// Thread-safe wrapper around the cross-encoder ONNX model.
    model: Arc<Mutex<TextRerank>>,
}

impl std::fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("model", &"TextRerank(BGERerankerBase)")
            .finish()
    }
}

impl CrossEncoderReranker {
    /// Initialise the cross-encoder model and return a ready reranker.
    ///
    /// Model weights are downloaded on first use and cached locally.
    /// Subsequent calls reuse the on-disk cache and require no network access.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when the ONNX model fails to
    /// initialise (e.g. unsupported platform, cache write error).
    #[instrument]
    pub async fn try_new() -> Result<Self, RetrievalError> {
        let model = tokio::task::spawn_blocking(|| {
            let opts = RerankInitOptions::new(RerankerModel::BGERerankerBase)
                .with_show_download_progress(false);
            TextRerank::try_new(opts).map_err(|e| RetrievalError::Embedding {
                reason: format!("cross-encoder model init failed: {e}"),
            })
        })
        .await
        .map_err(|join_err| RetrievalError::Embedding {
            reason: format!("cross-encoder model init task panicked: {join_err}"),
        })??;

        tracing::info!("CrossEncoderReranker initialised (BGERerankerBase)");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }

    /// Rerank documents by semantic relevance to `query`.
    ///
    /// Extracts the `chunk_text` payload field from each [`SearchResult`] as
    /// the document body. Documents missing `chunk_text` fall back to an
    /// empty string, which the cross-encoder will score near-zero relative to
    /// well-populated documents.
    ///
    /// Returns up to `top_k` results sorted by descending cross-encoder score.
    /// The [`SearchResult::score`] field is replaced with the cross-encoder
    /// relevance score on each returned result.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] when the ONNX session fails or
    /// the Mutex is poisoned.
    ///
    /// # Panics
    ///
    /// Does not panic; index out-of-range from the cross-encoder is treated
    /// as a retrieval error.
    #[instrument(skip(self, documents), fields(
        query_len = query.len(),
        candidates = documents.len(),
        top_k,
    ))]
    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, RetrievalError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let query_owned = query.to_owned();
        let model = Arc::clone(&self.model);

        // Extract text bodies for the cross-encoder. We keep the original
        // documents around so we can map scores back by index.
        let texts: Vec<String> = documents
            .iter()
            .map(|doc| {
                doc.payload
                    .get(CHUNK_TEXT_KEY)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned()
            })
            .collect();

        let rerank_results = tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().map_err(|_| RetrievalError::Embedding {
                reason: "cross-encoder model mutex poisoned".to_owned(),
            })?;
            // Convert to Vec<&str> within the blocking closure so all string
            // data lives in the same stack frame — no lifetime issues since
            // `texts` is owned by this closure.
            // `return_documents = false` — we keep our original SearchResult
            // objects and only use the scores; avoids copying text back.
            // fastembed's `rerank<S>` uses the same `S` for both `query` and
            // `documents`. Passing `query_owned.as_str()` pins S = &str, which
            // allows the documents slice to be Vec<&str>.
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            guard
                .rerank(query_owned.as_str(), text_refs, false, None)
                .map_err(|e| RetrievalError::Embedding {
                    reason: format!("cross-encoder rerank failed: {e}"),
                })
        })
        .await
        .map_err(|join_err| RetrievalError::Embedding {
            reason: format!("cross-encoder rerank task panicked: {join_err}"),
        })??;

        // Map cross-encoder scores back onto the original SearchResult objects.
        let n = documents.len();
        let mut scored: Vec<(f32, usize)> = rerank_results
            .into_iter()
            .map(|r| {
                if r.index >= n {
                    Err(RetrievalError::Embedding {
                        reason: format!(
                            "cross-encoder returned out-of-range index {} (candidates: {n})",
                            r.index
                        ),
                    })
                } else {
                    Ok((r.score, r.index))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Sort descending by cross-encoder score; use original index as
        // tiebreaker for determinism.
        scored.sort_unstable_by(|(score_a, idx_a), (score_b, idx_b)| {
            score_b
                .partial_cmp(score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| idx_a.cmp(idx_b))
        });

        // Reconstruct the result list in reranked order, clamped to top_k.
        // We need indexed access into `documents`; collect into a Vec<Option<SearchResult>>
        // so we can move out by index without cloning.
        let mut doc_slots: Vec<Option<SearchResult>> = documents.into_iter().map(Some).collect();

        let reranked: Vec<SearchResult> = scored
            .into_iter()
            .take(top_k)
            .filter_map(|(score, idx)| {
                doc_slots[idx].take().map(|mut doc| {
                    doc.score = Score(score as f64);
                    doc
                })
            })
            .collect();

        tracing::debug!(
            input_count = n,
            output_count = reranked.len(),
            "cross-encoder reranking complete"
        );

        Ok(reranked)
    }
}
