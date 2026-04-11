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

use agent_rs_core::{CollectionName, Embedding};
use agent_rs_stores::{traits::VectorStore, types::SearchResult};
use tracing::instrument;

use crate::error::RetrievalError;

/// Perform a dense vector ANN search over the dedicated **triplets**
/// collection.
///
/// This is a thin wrapper over [`VectorStore::search`] that maps store
/// errors into [`RetrievalError::Store`]. It is intentionally separate from
/// [`vector_search`](super::vector::vector_search) so that callers can fuse
/// chunk hits and triplet hits as independent ranked lists in the ranker.
///
/// # Parameters
///
/// - `store` — the vector backend holding the triplets collection.
/// - `triplet_collection` — name of the collection that stores triplet
///   embeddings. Typically `"{base_collection}_triplets"`.
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
    query_embedding: &Embedding,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    store
        .search(triplet_collection, query_embedding, limit)
        .await
        .map_err(RetrievalError::Store)
}
