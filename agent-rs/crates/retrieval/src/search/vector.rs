//! Dense vector (ANN) search back-end.
//!
//! Provides thin async wrappers over [`VectorStore::search`] with consistent
//! error mapping into [`RetrievalError`]. The functions are standalone (not
//! methods on a struct) so callers can compose them freely without carrying
//! any extra state.
//!
//! # Variants
//!
//! - [`vector_search`] — standard dense ANN search.
//! - [`temporal_search`] — time-aware search (Phase 3: delegates to dense
//!   search; Phase 4 will add timestamp filter pushdown).
//! - [`summaries_search`] — searches a dedicated summaries collection.

use agent_rs_core::{CollectionName, Embedding};
use agent_rs_stores::{traits::VectorStore, types::SearchResult};
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a dense vector ANN search against `collection`.
///
/// Delegates directly to [`VectorStore::search`] and maps any store error
/// into [`RetrievalError::Store`].
///
/// # Parameters
///
/// - `store` — the vector backend to query.
/// - `collection` — the target collection name.
/// - `query_embedding` — the dense query vector.
/// - `limit` — maximum number of results to return.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying store call fails.
#[instrument(skip(store, query_embedding), fields(
    collection = %collection,
    limit,
))]
pub async fn vector_search(
    store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search(collection, query_embedding, limit)
        .await
        .map_err(RetrievalError::Store)
}

/// Perform a time-aware dense vector search against `collection`.
///
/// **Phase 3 stub:** delegates to [`vector_search`]. Phase 4 will add
/// timestamp filter pushdown via Qdrant payload filtering.
///
/// # Parameters
///
/// Same as [`vector_search`]. A `reference_time` parameter will be added
/// in Phase 4 when timestamp filtering is wired in.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying store call fails.
#[instrument(skip(store, query_embedding), fields(
    collection = %collection,
    limit,
))]
pub async fn temporal_search(
    store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    // Phase 3: straight dense search. Phase 4 will inject a payload filter
    // for `created_at` / `timestamp` fields via Qdrant's filter API.
    tracing::debug!(
        "temporal_search: timestamp filtering not yet implemented; using vector_search"
    );
    vector_search(store, collection, query_embedding, limit).await
}

/// Perform a dense vector search against a dedicated summaries collection.
///
/// The summaries collection typically contains document-level embeddings
/// rather than chunk-level embeddings, yielding coarser but broader results.
///
/// # Parameters
///
/// - `store` — the vector backend to query.
/// - `summaries_collection` — the name of the summaries collection to query.
/// - `query_embedding` — the dense query vector.
/// - `limit` — maximum number of results to return.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying store call fails.
#[instrument(skip(store, query_embedding), fields(
    collection = %summaries_collection,
    limit,
))]
pub async fn summaries_search(
    store: &(dyn VectorStore + Send + Sync),
    summaries_collection: &CollectionName,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search(summaries_collection, query_embedding, limit)
        .await
        .map_err(RetrievalError::Store)
}
