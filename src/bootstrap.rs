//! Runtime bootstrap helpers — idempotent `create-if-missing` calls
//! that the `agent-service` / `indexer-worker` binaries run at startup.
//!
//! Keeping them here means every deployment gets the invariants
//! automatically, instead of relying on a separate `just bootstrap`
//! step that operators can forget to run.

use std::sync::Arc;

use agent_rs_core::types::CollectionName;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::CollectionConfig;
use anyhow::{Context, Result};
use tracing::info;

/// Ensure the three Qdrant collections used by the pipeline exist.
///
/// Writes are idempotent via `collection_exists` + `create_collection`:
/// a cold cluster gets freshly-created collections; a warm one is a
/// no-op. All three collections are configured as **hybrid** (dense +
/// server-side BM25 sparse) so `VectorStore::upsert` can write the
/// `text` field without the collection rejecting the sparse named
/// vector.
///
/// Collection names:
/// - `main`      — primary chunk collection (1 point per chunk)
/// - `summaries` — per-section LLM summaries (1 point per summarised chunk)
/// - `triplets`  — flattened KG relations (`src -> rel -> dst`)
///
/// `dense_dims` must match `Embedder::dimensions()` of the instance that
/// will write into the collection later — a mismatch at upsert time is
/// a hard Qdrant error.
pub async fn ensure_collections(
    qdrant: &Arc<QdrantStore>,
    dense_dims: usize,
    main: &CollectionName,
    summaries: &CollectionName,
    triplets: &CollectionName,
) -> Result<()> {
    for name in [main, summaries, triplets] {
        if qdrant
            .collection_exists(name)
            .await
            .with_context(|| format!("check collection {name}"))?
        {
            info!(collection = %name, "collection already exists");
            continue;
        }

        let config = CollectionConfig::hybrid(dense_dims);
        qdrant
            .create_collection(name, &config)
            .await
            .with_context(|| format!("create collection {name}"))?;
        info!(
            collection = %name,
            dense_dims,
            "collection created (dense + bm25 sparse)"
        );
    }
    Ok(())
}
