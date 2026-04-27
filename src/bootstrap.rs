//! Runtime bootstrap helpers — idempotent `create-if-missing` calls
//! that the `agent-service` / `indexer-worker` binaries run at startup.
//!
//! Keeping them here means every deployment gets the invariants
//! automatically, instead of relying on a separate `just bootstrap`
//! step that operators can forget to run.

use std::sync::Arc;

use agent_rs_core::types::CollectionName;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::CollectionConfig;
use anyhow::{Context, Result};
use tracing::{info, warn};

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

/// Boot-time probe that logs the current population of each retrieval
/// collection.
///
/// Empty `summaries` / `triplets` collections silently downgrade the
/// `GraphHybrid` strategy (which declares `uses_triplets = true`) to a
/// dense-only search: operators deserve a high-visibility signal at
/// startup rather than a debug session weeks later. This probe emits:
///
/// - one `INFO` with populated collections and their point counts;
/// - one `WARN` per empty collection, enumerating the retrieval
///   capabilities that will be degraded.
///
/// The probe is **best-effort**: count failures log and continue
/// rather than aborting startup, because a transient Qdrant blip
/// during a rolling deploy must not block the service from coming up.
///
/// # Arguments
///
/// - `qdrant`     — shared vector store handle.
/// - `chunks`     — primary chunks collection (`default`).
/// - `summaries`  — per-section summaries collection.
/// - `triplets`   — knowledge-graph triplets collection.
pub async fn probe_retrieval_coverage(
    qdrant: &Arc<QdrantStore>,
    chunks: &CollectionName,
    summaries: &CollectionName,
    triplets: &CollectionName,
) {
    // Each probe error logs WARN with the collection name so operators
    // can distinguish "collection empty" (INFO / WARN below) from
    // "probe itself is broken" (e.g. expired Qdrant TLS, auth failure).
    // The former is a retrieval-capability gap; the latter is an
    // operational issue that would otherwise look identical ("unknown").
    let chunks_count = points_count_or_warn(qdrant, chunks).await;
    let summaries_count = points_count_or_warn(qdrant, summaries).await;
    let triplets_count = points_count_or_warn(qdrant, triplets).await;

    info!(
        chunks = %display_count(chunks_count),
        summaries = %display_count(summaries_count),
        triplets = %display_count(triplets_count),
        "retrieval_backend_coverage"
    );

    if summaries_count == Some(0) {
        warn!(
            collection = %summaries,
            strategy = "Summaries",
            remediation = "re-ingest with AGENT_RS__INDEXER__DISABLE_SUMMARIES unset and an LLM key present",
            "summaries collection is empty — `RetrievalStrategy::Summaries` and `outline_corpus` will return no results"
        );
    }
    if triplets_count == Some(0) {
        warn!(
            collection = %triplets,
            strategy = "GraphHybrid/GraphCoT",
            remediation = "re-ingest with AGENT_RS__INDEXER__DISABLE_KG unset and an LLM key present",
            "triplets collection is empty — graph-hybrid retrieval will run with reduced signal coverage"
        );
    }
    if chunks_count == Some(0) {
        // An empty primary collection is the expected state on a cold
        // deployment right after `ensure_collections` runs. A WARN here
        // would falsely alarm on every fresh cluster, so we emit INFO
        // instead — operators still see the signal in structured logs,
        // but it doesn't page anyone on a clean boot. The `summaries` /
        // `triplets` warnings above keep their severity because those
        // collections being empty *with* a populated `chunks` collection
        // is a real retrieval-capability gap.
        info!(
            collection = %chunks,
            "primary chunks collection is empty — knowledge base has no documents yet (expected on cold start)"
        );
    }
}

fn display_count(count: Option<u64>) -> String {
    match count {
        Some(n) => n.to_string(),
        None => "unknown".to_owned(),
    }
}

/// Wrap [`QdrantStore::points_count`] so a transport failure surfaces as
/// a structured WARN before being collapsed to `None`. Without this, a
/// broken probe (TLS, auth, DNS) is indistinguishable in the retrieval
/// coverage INFO from a collection that simply does not exist yet.
async fn points_count_or_warn(
    qdrant: &Arc<QdrantStore>,
    collection: &CollectionName,
) -> Option<u64> {
    match qdrant.points_count(collection).await {
        Ok(c) => c,
        Err(e) => {
            warn!(
                collection = %collection,
                error = %e,
                "points_count probe failed; coverage will display as 'unknown'"
            );
            None
        }
    }
}

/// Ensure the Neo4j structural-layer constraints and the
/// `entity_mention_fts` fulltext index exist.
///
/// Counterpart of [`ensure_collections`] for the graph store: every
/// binary that opens a `Neo4jStore` must call this at startup so a
/// cold cluster gets the index the entity linker relies on. Without
/// it, `db.index.fulltext.queryNodes("entity_mention_fts", ...)` fails
/// with `ProcedureCallFailed` on every query and `EntityLinker::link`
/// silently returns zero matches, degrading `GraphHybrid` retrieval
/// to dense-only.
///
/// Idempotent — the underlying Cypher uses `IF NOT EXISTS` for every
/// constraint and the fulltext index, so re-running against a warm
/// cluster is a no-op.
///
/// # Errors
///
/// Hard-fails on any Neo4j error (auth, connectivity, privilege). A
/// graph-less startup is strictly worse than a refused startup: the
/// service would run but silently lose the graph-hybrid branch on
/// every query. Operators need to see the failure during `docker
/// compose up`, not week two of production.
pub async fn ensure_graph_indexes(neo4j: &Arc<Neo4jStore>) -> Result<()> {
    neo4j
        .ensure_structural_constraints()
        .await
        .context("ensure Neo4j structural constraints + entity_mention_fts index")?;
    info!("Neo4j structural constraints + entity_mention_fts fulltext index ready");
    Ok(())
}
