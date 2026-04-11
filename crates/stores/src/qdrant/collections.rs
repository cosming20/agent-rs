//! Collection management helpers for Qdrant.
//!
//! Encapsulates collection creation logic, including named-vector
//! configuration for both dense-only and hybrid (dense + sparse) modes.

use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, Modifier, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, VectorParamsBuilder, VectorsConfigBuilder,
};
use tracing::instrument;

use crate::types::CollectionConfig;

/// Build and issue a create-collection request against the given client.
///
/// Named vectors are used so that dense ("dense") and optional sparse
/// ("sparse") embeddings coexist in the same collection.
///
/// When `enable_sparse` is true, the sparse vector is configured with
/// [`Modifier::Idf`] so that Qdrant performs server-side BM25 inference
/// (via the `qdrant/bm25` model) during upsert and query operations.
///
/// # Errors
///
/// Returns `qdrant_client::QdrantError` on any transport or protocol error.
#[instrument(skip(client), fields(collection = %name))]
pub async fn create_named_vector_collection(
    client: &qdrant_client::Qdrant,
    name: &str,
    config: &CollectionConfig,
) -> Result<(), qdrant_client::QdrantError> {
    let mut vectors_config = VectorsConfigBuilder::default();

    // Dense named vector (always present).
    vectors_config.add_named_vector_params(
        &config.dense_vector_name,
        VectorParamsBuilder::new(config.dense_vector_size as u64, Distance::Cosine),
    );

    let mut builder = CreateCollectionBuilder::new(name).vectors_config(vectors_config);

    // Sparse named vector (optional, for hybrid BM25 search).
    //
    // `Modifier::Idf` is required for server-side BM25 inference via the
    // `qdrant/bm25` model — Qdrant uses IDF weighting at query time to
    // combine term frequencies with corpus statistics.
    if config.enable_sparse {
        let mut sparse_config = SparseVectorsConfigBuilder::default();
        sparse_config.add_named_vector_params(
            &config.sparse_vector_name,
            SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
        );
        builder = builder.sparse_vectors_config(sparse_config);
    }

    client.create_collection(builder).await?;
    Ok(())
}
