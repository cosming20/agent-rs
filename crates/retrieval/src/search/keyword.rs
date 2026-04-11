//! BM25 / lexical (sparse vector) search back-end.
//!
//! Qdrant 1.17+ supports server-side BM25 inference via the `qdrant/bm25`
//! model. This module delegates to [`VectorStore::search_hybrid`] with the
//! raw query text; the store layer sends a `Document { model: "qdrant/bm25" }`
//! prefetch query and lets Qdrant handle tokenisation and IDF weighting.
//!
//! No local SPLADE or fastembed sparse model is required.
//!
//! The [`keyword_search`] function is intentionally thin. All collection-level
//! tuning (named vector weights, HNSW ef) lives in the store layer.

use agent_rs_core::{CollectionName, Embedding};
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::SearchResult;
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a BM25-style lexical search via Qdrant server-side sparse inference.
///
/// Delegates to [`VectorStore::search_hybrid`] which internally issues a
/// Qdrant prefetch query with `Document { text: query_text, model: "qdrant/bm25" }`
/// on the `"sparse"` named vector, fused with a dense prefetch via RRF.
///
/// # Parameters
///
/// - `store` — the vector backend to query.
/// - `collection` — the target collection (must have a sparse named vector
///   configured with `Modifier::Idf`).
/// - `query_embedding` — the dense query vector (used for the RRF dense leg).
/// - `query_text` — the raw query string sent to Qdrant for BM25 inference.
/// - `limit` — maximum number of results to return.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying store call fails.
#[instrument(skip(store, query_embedding), fields(
    collection = %collection,
    limit,
))]
pub async fn keyword_search(
    store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    query_embedding: &Embedding,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search_hybrid(collection, query_embedding, query_text, limit)
        .await
        .map_err(RetrievalError::Store)
}
