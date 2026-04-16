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

use std::collections::HashMap;

use agent_rs_core::Score;
use agent_rs_stores::{traits::GraphStore, types::SearchResult};
use serde_json::json;
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
/// Returns [`RetrievalError::Store`] if any `get_neighbors` call fails.
#[instrument(skip(store), fields(
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_search(
    store: &(dyn GraphStore + Send + Sync),
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    // best_score tracks the highest score seen for each unique node ID so that
    // nodes reachable from multiple seeds keep only their best result.
    let mut best_score: HashMap<String, f64> = HashMap::new();
    // node_payload retains the payload of the first (highest-score) encounter.
    let mut node_payload: HashMap<String, serde_json::Value> = HashMap::new();

    for seed_id in seed_node_ids {
        let neighbors = store
            .get_neighbors(seed_id, max_depth)
            .await
            .map_err(RetrievalError::Store)?;

        for node in neighbors.iter() {
            // Uniform scoring: `get_neighbors` returns a flat unordered list
            // without depth information. The enumerate index is NOT traversal
            // depth. Depth-aware scoring requires a GraphStore API change to
            // return `(GraphNode, usize)`. RRF fusion handles relative ranking.
            let score = BASE_DEPTH_SCORE;

            let entry = best_score.entry(node.id.clone()).or_insert(0.0);
            if score > *entry {
                *entry = score;
                node_payload.insert(
                    node.id.clone(),
                    json!({
                        "node_id": node.id,
                        "label": node.label.as_str(),
                        "properties": node.properties,
                        "source": "neo4j",
                        "query": query,
                        "depth": max_depth,
                    }),
                );
            }
        }
    }

    // Sort by descending score then by ID for determinism.
    let mut scored: Vec<(String, f64)> = best_score.into_iter().collect();
    scored.sort_unstable_by(|(id_a, score_a), (id_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| id_a.cmp(id_b))
    });

    let results: Vec<SearchResult> = scored
        .into_iter()
        .take(limit)
        .filter_map(|(id, score)| {
            node_payload.remove(&id).map(|payload| SearchResult {
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
#[instrument(skip(store), fields(
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_cot_search(
    store: &(dyn GraphStore + Send + Sync),
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    // The CoT iteration is applied by the router after this function returns.
    // Here we only produce the initial evidence pool.
    graph_search(store, query, seed_node_ids, max_depth, limit).await
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
#[instrument(skip(store), fields(
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_constrained_search(
    store: &(dyn GraphStore + Send + Sync),
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    tracing::debug!(
        "graph_constrained_search: constraint filtering not yet implemented; using graph_search"
    );
    graph_search(store, query, seed_node_ids, max_depth, limit).await
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
#[instrument(skip(store), fields(
    num_seeds = seed_node_ids.len(),
    max_depth,
    limit,
))]
pub async fn graph_rag_context_search(
    store: &(dyn GraphStore + Send + Sync),
    query: &str,
    seed_node_ids: &[String],
    max_depth: usize,
    limit: usize,
) -> Result<Vec<SearchResult>, RetrievalError> {
    tracing::debug!(
        "graph_rag_context_search: context injection not yet implemented; using graph_search"
    );
    graph_search(store, query, seed_node_ids, max_depth, limit).await
}
