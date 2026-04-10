//! Collection management helpers for Qdrant.
//!
//! Encapsulates collection creation logic, including named-vector
//! configuration for both dense-only and hybrid (dense + sparse) modes.

use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, SparseIndexConfig, SparseVectorConfig, SparseVectorParams,
    VectorParamsBuilder, VectorsConfigBuilder,
};
use tracing::instrument;

use crate::types::CollectionConfig;

/// Build and issue a create-collection request against the given client.
///
/// Named vectors are used so that dense ("dense") and optional sparse
/// ("sparse") embeddings coexist in the same collection.
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

    // Sparse named vector (optional, for hybrid search).
    if config.enable_sparse {
        let mut sparse_map = std::collections::HashMap::new();
        sparse_map.insert(
            config.sparse_vector_name.clone(),
            SparseVectorParams {
                index: Some(SparseIndexConfig {
                    on_disk: Some(false),
                    ..Default::default()
                }),
                modifier: None,
            },
        );
        builder = builder.sparse_vectors_config(SparseVectorConfig { map: sparse_map });
    }

    client.create_collection(builder).await?;
    Ok(())
}
