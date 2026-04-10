//! BM25 / lexical (sparse vector) search back-end.
//!
//! Qdrant supports sparse vector search via BM42 / SPLADE models. This module
//! wraps [`VectorStore::search_hybrid`] with a lexical-first weighting:
//! by passing a zero-length dense embedding the caller signals "sparse only",
//! but in practice callers pass the real dense vector too — Qdrant's prefetch
//! mechanism handles weighting at the collection level.
//!
//! The [`keyword_search`] function is intentionally thin. All collection-level
//! tuning (named vector weights, HNSW ef) lives in the store layer.

use agent_rs_core::{CollectionName, Embedding};
use agent_rs_stores::{traits::VectorStore, types::SearchResult, types::SparseVector};
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a BM25-style lexical search via Qdrant sparse vectors.
///
/// Delegates to [`VectorStore::search_hybrid`] which internally issues a
/// Qdrant query with a named sparse vector (`"sparse"`) prefetched at higher
/// weight than the dense vector (`"dense"`). The result set is ranked by
/// sparse-dominated score, giving keyword-recall semantics.
///
/// # Parameters
///
/// - `store` — the vector backend to query.
/// - `collection` — the target collection (must have a sparse named vector).
/// - `query_embedding` — the dense query vector (used for fallback coverage).
/// - `sparse_query` — the sparse (BM42 / SPLADE) query vector.
/// - `limit` — maximum number of results to return.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying store call fails.
#[instrument(skip(store, query_embedding, sparse_query), fields(
    collection = %collection,
    sparse_nnz = sparse_query.indices.len(),
    limit,
))]
pub async fn keyword_search(
    store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    query_embedding: &Embedding,
    sparse_query: &SparseVector,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search_hybrid(collection, query_embedding, sparse_query, limit)
        .await
        .map_err(RetrievalError::Store)
}
