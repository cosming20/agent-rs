//! Dense vector (ANN) search back-end.
//!
//! Provides thin async wrappers over [`VectorStore::search_for_user`] with
//! consistent error mapping into [`RetrievalError`]. Every search function
//! carries a mandatory `user_id` so cross-tenant reads are structurally
//! impossible.
//!
//! # Variants
//!
//! - [`vector_search`] — standard dense ANN search, user-scoped.
//! - [`temporal_search`] — time-aware search (delegates to dense search;
//!   timestamp filter pushdown is a future enhancement).
//! - [`summaries_search`] — searches a dedicated summaries collection.

use agent_rs_core::{CollectionName, Embedding, UserId};
use agent_rs_stores::{traits::VectorStore, types::SearchResult};
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a user-scoped dense vector ANN search against `collection`.
///
/// Delegates to [`VectorStore::search_for_user`] so the mandatory `user_id`
/// filter is always applied. Returns [`RetrievalError::Store`] on failure.
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
    user_id: &UserId,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search_for_user(collection, user_id, query_embedding, limit, None)
        .await
        .map_err(RetrievalError::Store)
}

/// Perform a user-scoped time-aware dense vector search against `collection`.
///
/// **Phase 3 stub:** delegates to [`vector_search`]. A future phase will add
/// timestamp filter pushdown via Qdrant payload filtering.
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
    user_id: &UserId,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    tracing::debug!(
        "temporal_search: timestamp filtering not yet implemented; using vector_search"
    );
    vector_search(store, collection, user_id, query_embedding, limit).await
}

/// Perform a user-scoped dense vector search against a dedicated summaries collection.
///
/// The summaries collection typically contains document-level embeddings
/// rather than chunk-level embeddings, yielding coarser but broader results.
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
    user_id: &UserId,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search_for_user(summaries_collection, user_id, query_embedding, limit, None)
        .await
        .map_err(RetrievalError::Store)
}
