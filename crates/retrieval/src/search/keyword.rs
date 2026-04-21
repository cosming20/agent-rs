//! BM25 / lexical (sparse vector) search back-end.
//!
//! Qdrant 1.17+ supports server-side BM25 inference via the `qdrant/bm25`
//! model. This module wraps [`VectorStore::search_for_user`] with a text-match
//! extra filter that replicates the sparse keyword leg while enforcing the
//! mandatory `user_id` scope.
//!
//! No local SPLADE or fastembed sparse model is required.
//!
//! The [`keyword_search`] function is intentionally thin. All collection-level
//! tuning (named vector weights, HNSW ef) lives in the store layer.

use agent_rs_core::{CollectionName, Embedding, UserId};
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::SearchResult;
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a user-scoped BM25-style lexical search.
///
/// Builds a text-match extra filter for `query_text` and delegates to
/// [`VectorStore::search_for_user`] so the mandatory `user_id` scope is
/// always applied. The underlying store implementation handles the
/// BM25 / hybrid fusion.
///
/// # Parameters
///
/// - `store` — the vector backend to query.
/// - `collection` — the target collection (must have a sparse named vector
///   configured with `Modifier::Idf`).
/// - `user_id` — tenant scope; empty value is a programming error.
/// - `query_embedding` — the dense query vector (used for the RRF dense leg).
/// - `query_text` — the raw query string sent for BM25 inference.
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
    user_id: &UserId,
    query_embedding: &Embedding,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    // Build a text-match extra filter so the store can push BM25 scoring
    // server-side in addition to the mandatory user_id scope filter.
    let text_filter = serde_json::json!({
        "must": [
            { "key": "chunk_text", "match": { "text": query_text } }
        ]
    });
    store
        .search_for_user(
            collection,
            user_id,
            query_embedding,
            limit,
            Some(text_filter),
        )
        .await
        .map_err(RetrievalError::Store)
}
