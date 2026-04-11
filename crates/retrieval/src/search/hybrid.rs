//! Hybrid multi-backend search dispatcher.
//!
//! Fans out to multiple search back-ends in parallel via [`tokio::join!`] and
//! returns an unfused [`HybridResults`] struct. Fusion is intentionally
//! deferred to the caller (the ranker/router layer) so that weighting and
//! truncation decisions live in one place.
//!
//! # Functions
//!
//! - [`hybrid_search`] — vector + BM25 keyword in parallel (HYBRID strategy).
//! - [`graph_hybrid_search`] — vector + BM25 keyword + graph in parallel
//!   (GRAPH_HYBRID strategy).
//!
//! Server-side BM25 inference is used for the keyword leg: the raw query text
//! is forwarded to Qdrant as `Document { model: "qdrant/bm25" }` so no local
//! SPLADE model is required.

use agent_rs_core::{CollectionName, Embedding};
use agent_rs_stores::{
    traits::{GraphStore, VectorStore},
    types::SearchResult,
};
use tracing::instrument;

use super::{graph::graph_search, keyword::keyword_search, vector::vector_search};
use crate::error::RetrievalError;

// ---- HybridResults ----

/// Unfused results from a multi-backend hybrid search.
///
/// Contains the raw result lists from each participating back-end.
/// The caller (typically [`crate::ranker::ResultRanker`]) is responsible
/// for deduplication and RRF fusion.
#[derive(Debug, Clone)]
pub struct HybridResults {
    /// Results from the dense vector ANN search.
    pub vector_results: Vec<SearchResult>,
    /// Results from the sparse / BM25 lexical search.
    pub keyword_results: Vec<SearchResult>,
    /// Results from the graph traversal search.
    ///
    /// `None` for [`hybrid_search`]; `Some` for [`graph_hybrid_search`].
    pub graph_results: Option<Vec<SearchResult>>,
}

// ---- hybrid_search ----

/// Dispatch vector and BM25 keyword searches in parallel.
///
/// Both searches target the same `collection`. Results are returned as
/// an unfused [`HybridResults`] (graph_results is `None`).
///
/// # Parameters
///
/// - `vector_store` — the vector backend.
/// - `collection` — the target collection (must have both dense and sparse
///   named vectors; sparse must be configured with `Modifier::Idf`).
/// - `query_embedding` — the dense query vector.
/// - `query_text` — the raw query string for server-side BM25 inference.
/// - `limit` — maximum results per back-end.
///
/// # Errors
///
/// Returns the first [`RetrievalError`] encountered across either search arm.
#[instrument(skip(vector_store, query_embedding), fields(
    collection = %collection,
    limit,
))]
pub async fn hybrid_search(
    vector_store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    query_embedding: &Embedding,
    query_text: &str,
    limit: usize,
) -> Result<HybridResults, RetrievalError> {
    let (vector_result, keyword_result) = tokio::join!(
        vector_search(vector_store, collection, query_embedding, limit),
        keyword_search(vector_store, collection, query_embedding, query_text, limit),
    );

    Ok(HybridResults {
        vector_results: vector_result?,
        keyword_results: keyword_result?,
        graph_results: None,
    })
}

// ---- GraphHybridParams ----

/// Parameters for a graph-hybrid search, bundling all inputs into one struct.
///
/// Used by [`graph_hybrid_search`] to stay within clippy's argument-count
/// limit while keeping the call site readable.
pub struct GraphHybridParams<'a> {
    /// Dense query vector for the vector search arm.
    pub query_embedding: &'a Embedding,
    /// Raw query string for the BM25 keyword search arm (server-side inference).
    pub query_text: &'a str,
    /// Original query string forwarded to graph search for provenance metadata.
    pub query: &'a str,
    /// Starting node IDs for the graph traversal arm.
    pub seed_node_ids: &'a [String],
    /// Maximum graph traversal depth from each seed.
    pub max_depth: usize,
    /// Maximum results per back-end.
    pub limit: usize,
}

// ---- graph_hybrid_search ----

/// Dispatch vector, BM25 keyword, and graph searches in parallel.
///
/// All three searches run concurrently. Vector and keyword searches target
/// `collection`; graph search expands from `params.seed_node_ids`. Results are
/// returned as an unfused [`HybridResults`] with all three result sets
/// populated.
///
/// # Parameters
///
/// - `vector_store` — the vector backend (used for both vector and keyword).
/// - `graph_store` — the graph backend.
/// - `collection` — the target vector/keyword collection.
/// - `params` — all remaining search parameters; see [`GraphHybridParams`].
///
/// # Errors
///
/// Returns the first [`RetrievalError`] encountered across any search arm.
#[instrument(skip(vector_store, graph_store, params), fields(
    collection = %collection,
    num_seeds = params.seed_node_ids.len(),
    max_depth = params.max_depth,
    limit = params.limit,
))]
pub async fn graph_hybrid_search(
    vector_store: &(dyn VectorStore + Send + Sync),
    graph_store: &(dyn GraphStore + Send + Sync),
    collection: &CollectionName,
    params: GraphHybridParams<'_>,
) -> Result<HybridResults, RetrievalError> {
    let (vector_result, keyword_result, graph_result) = tokio::join!(
        vector_search(
            vector_store,
            collection,
            params.query_embedding,
            params.limit
        ),
        keyword_search(
            vector_store,
            collection,
            params.query_embedding,
            params.query_text,
            params.limit
        ),
        graph_search(
            graph_store,
            params.query,
            params.seed_node_ids,
            params.max_depth,
            params.limit
        ),
    );

    Ok(HybridResults {
        vector_results: vector_result?,
        keyword_results: keyword_result?,
        graph_results: Some(graph_result?),
    })
}
