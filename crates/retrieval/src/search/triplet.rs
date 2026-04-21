//! Triplet vector search back-end.
//!
//! Scores complete `(source) -[relation]-> (target)` triplets against the
//! query embedding, mirroring Cognee's
//! [`brute_force_triplet_search`](https://github.com/topoteretes/cognee/blob/dev/cognee/modules/retrieval/utils/brute_force_triplet_search.py).
//!
//! Unlike [`vector_search`](super::vector::vector_search), which scores
//! individual chunks, `triplet_search` queries a dedicated collection whose
//! points are linearised triplets (see
//! [`Triplet`](crate::ingestion::Triplet)). The returned [`SearchResult`]s
//! carry the original triplet payload so the ranker can reassemble the
//! subject / predicate / object for evidence fusion.

use agent_rs_core::{CollectionName, Embedding, UserId};
use agent_rs_stores::{traits::VectorStore, types::SearchResult};
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a user-scoped dense vector ANN search over the dedicated **triplets**
/// collection.
///
/// Delegates to [`VectorStore::search_for_user`] so the mandatory `user_id`
/// filter is always applied. Returns [`RetrievalError::Store`] on failure.
///
/// # Parameters
///
/// - `store` — the vector backend holding the triplets collection.
/// - `triplet_collection` — name of the collection that stores triplet
///   embeddings. Typically `"{base_collection}_triplets"`.
/// - `user_id` — tenant scope; empty value is a programming error.
/// - `query_embedding` — the dense query vector (same model as ingestion).
/// - `limit` — maximum number of triplet hits to return.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying store call fails.
#[instrument(skip(store, query_embedding), fields(
    collection = %triplet_collection,
    limit,
))]
pub async fn triplet_search(
    store: &(dyn VectorStore + Send + Sync),
    triplet_collection: &CollectionName,
    user_id: &UserId,
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search_for_user(triplet_collection, user_id, query_embedding, limit, None)
        .await
        .map_err(RetrievalError::Store)
}
