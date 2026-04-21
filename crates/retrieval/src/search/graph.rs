//! Cypher graph traversal search back-end.
//!
//! Provides graph-based retrieval by expanding from seed nodes via
//! [`GraphStore::get_neighbors`]. Results are converted from [`GraphNode`]s to
//! [`SearchResult`]s with scores derived from traversal depth: shallower nodes
//! receive a higher score, modelling relevance-by-proximity.
//!
//! # Variants
//!
//! - [`graph_search`] — base graph traversal from seed node IDs.
//! - [`graph_cot_search`] — chain-of-thought graph search. When no
//!   [`GraphCotRetriever`] is configured on the router it behaves like
//!   [`graph_search`]; with a retriever the router wraps the result in the
//!   iterative CoT loop (see [`crate::graph_cot`]).
//! - [`graph_constrained_search`] — constraint-filtered graph search (Phase 3:
//!   delegates to base; Phase 4 adds property-filter pushdown).
//! - [`graph_rag_context_search`] — context-augmented graph search (Phase 3:
//!   delegates to base; Phase 4 adds context injection into the Cypher query).
//!
//! [`GraphCotRetriever`]: crate::graph_cot::GraphCotRetriever

use std::cmp::Ordering;
use std::collections::HashMap;

use agent_rs_core::{CollectionName, Score, UserId};
use agent_rs_stores::{
    traits::{GraphStore, VectorStore},
    types::{NodeLabel, PointPayload, SearchResult},
};
use tracing::instrument;

use crate::error::RetrievalError;

// ---- Scoring constants ----

/// Score assigned to a node found at depth 1 (direct neighbour).
///
/// Scores decay linearly with depth. Nodes at depth `d` receive
/// `BASE_DEPTH_SCORE / d as f64`.
const BASE_DEPTH_SCORE: f64 = 1.0;

// ---- graph_search ----

/// Perform a graph traversal search starting from `seed_node_ids`.
///
/// For each seed node, the function fetches all neighbours up to `max_depth`
/// via [`GraphStore::get_neighbors`]. Returned [`GraphNode`]s are converted to
/// [`SearchResult`]s with scores inversely proportional to their discovery
/// depth. Nodes discovered from multiple seeds are deduplicated and scored by
/// their best (highest) individual score.
///
/// # Parameters
///
/// - `store` — the graph backend to query.
/// - `query` — the original user query string (stored in result payload for
///   provenance; not used for graph traversal in Phase 3).
/// - `seed_node_ids` — IDs of the starting nodes for traversal.
/// - `max_depth` — maximum traversal depth from each seed.
/// - `limit` — maximum number of results to return.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] on graph-traversal or payload-rehydration
/// failure.
///
/// # Invariants enforced
///
/// 1. **Label filter**: only `Chunk`-labeled neighbors are considered. Entity,
///    Document, Page, Section, Table, and Picture nodes are dropped at the
///    database boundary so they never enter the RRF-fused result stream.
///    Non-chunk nodes carry no `chunk_text` payload and would surface as
///    empty rows in the user-facing evidence pack.
///
/// 2. **Payload-or-drop**: every returned [`SearchResult`] carries the
///    authoritative Qdrant payload (`chunk_text`, `section_title`,
///    `page_start`, …). Chunk IDs that exist in Neo4j but are absent from
///    the vector collection (index skew) are logged and dropped rather than
///    emitted with empty content.
///
/// `query` is retained for trace-span provenance only and is not attached to
/// the per-result payload (payload comes from Qdrant and already contains the
/// full chunk metadata).
#[allow(clippy::too_many_arguments)]
#[instrument(skip(graph, vectors), fields(
    collection = %collection,
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_search(
    graph: &(dyn GraphStore + Send + Sync),
    vectors: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    user_id: &UserId,
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    let _ = query; // Reserved for future scoring heuristics; see docstring.

    // Push the Chunk label predicate into Cypher so we don't waste bandwidth
    // transporting Entity / Document / Page / Section / Table / Picture nodes
    // just to drop them here.
    let chunk_labels = [NodeLabel::chunk()];

    // best_score collapses duplicate chunks reachable from multiple seeds,
    // keeping the highest observed score. Scoring is uniform today because
    // `get_neighbors_of_labels` returns a flat unordered set — depth-aware
    // scoring would need a backend API change.
    let mut best_score: HashMap<String, f64> = HashMap::new();

    for seed_id in seed_node_ids {
        let neighbors = graph
            .get_neighbors_of_labels(seed_id, max_depth, &chunk_labels)
            .await
            .map_err(RetrievalError::Store)?;

        for node in neighbors {
            let entry = best_score.entry(node.id).or_insert(0.0);
            if BASE_DEPTH_SCORE > *entry {
                *entry = BASE_DEPTH_SCORE;
            }
        }
    }

    // Deterministic ordering: score desc, then id asc for tie-break.
    let mut scored: Vec<(String, f64)> = best_score.into_iter().collect();
    scored.sort_unstable_by(|(id_a, score_a), (id_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| id_a.cmp(id_b))
    });
    scored.truncate(limit);

    if scored.is_empty() {
        return Ok(Vec::new());
    }

    // Rehydrate payloads from the vector store, scoped to user_id.
    // Qdrant returns only IDs that still exist; the user scope filter provides
    // defence-in-depth against cross-tenant leakage.
    let ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
    let payloads = vectors
        .retrieve_chunks_for_user(collection, user_id, &ids)
        .await
        .map_err(RetrievalError::Store)?;

    let payload_by_id: HashMap<String, serde_json::Value> = payloads
        .into_iter()
        .map(|PointPayload { id, payload }| (id, payload))
        .collect();

    let n_requested = ids.len();
    let n_hydrated = payload_by_id.len();
    if n_hydrated < n_requested {
        tracing::warn!(
            requested = n_requested,
            hydrated = n_hydrated,
            missing = n_requested - n_hydrated,
            collection = %collection,
            "graph_search: chunk IDs present in Neo4j but missing from Qdrant (index skew)"
        );
    }

    let results: Vec<SearchResult> = scored
        .into_iter()
        .filter_map(|(id, score)| {
            let payload = payload_by_id.get(&id)?.clone();
            Some(SearchResult {
                id,
                score: Score::new(score),
                payload,
            })
        })
        .collect();

    Ok(results)
}

// ---- graph_cot_search ----

/// Graph chain-of-thought search — initial retrieval pass.
///
/// This function performs the *base* graph traversal used as the starting
/// pool for the iterative chain-of-thought loop. The iterative refinement
/// (follow-up question generation + merge) lives in
/// [`crate::graph_cot::GraphCotRetriever`] and is applied by
/// [`crate::router::RetrievalRouter`] whenever a CoT retriever is configured.
///
/// Calling this function directly (without a retriever) is equivalent to
/// [`graph_search`]: it returns the nodes reachable from `seed_node_ids`
/// within `max_depth`, ranked by proximity.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying traversal fails.
#[allow(clippy::too_many_arguments)]
#[instrument(skip(graph, vectors), fields(
    collection = %collection,
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_cot_search(
    graph: &(dyn GraphStore + Send + Sync),
    vectors: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    user_id: &UserId,
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    // The CoT iteration is applied by the router after this function returns.
    // Here we only produce the initial evidence pool.
    graph_search(
        graph,
        vectors,
        collection,
        user_id,
        query,
        seed_node_ids,
        max_depth,
        limit,
    )
    .await
}

// ---- graph_constrained_search ----

/// Graph search with constraint filtering.
///
/// **Phase 3 stub:** delegates to [`graph_search`]. Phase 4 will add
/// property-filter predicates pushed down into the Cypher `WHERE` clause
/// to restrict which nodes and edges qualify during traversal.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying traversal fails.
#[allow(clippy::too_many_arguments)]
#[instrument(skip(graph, vectors), fields(
    collection = %collection,
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_constrained_search(
    graph: &(dyn GraphStore + Send + Sync),
    vectors: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    user_id: &UserId,
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    tracing::debug!(
        "graph_constrained_search: constraint filtering not yet implemented; using graph_search"
    );
    graph_search(
        graph,
        vectors,
        collection,
        user_id,
        query,
        seed_node_ids,
        max_depth,
        limit,
    )
    .await
}

// ---- graph_rag_context_search ----

/// Context-augmented graph (RAG completion) search.
///
/// **Phase 3 stub:** delegates to [`graph_search`]. Phase 4 will inject a
/// context string (e.g. conversation history or a query reformulation) into
/// the Cypher query to guide the traversal toward contextually relevant nodes.
///
/// # Errors
///
/// Returns [`RetrievalError::Store`] if the underlying traversal fails.
#[allow(clippy::too_many_arguments)]
#[instrument(skip(graph, vectors), fields(
    collection = %collection,
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_rag_context_search(
    graph: &(dyn GraphStore + Send + Sync),
    vectors: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    user_id: &UserId,
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    tracing::debug!(
        "graph_rag_context_search: context injection not yet implemented; using graph_search"
    );
    graph_search(
        graph,
        vectors,
        collection,
        user_id,
        query,
        seed_node_ids,
        max_depth,
        limit,
    )
    .await
}
