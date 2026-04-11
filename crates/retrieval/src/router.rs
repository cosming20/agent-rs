//! Retrieval router — embeds, dispatches, and ranks.
//!
//! # Overview
//!
//! [`RetrievalRouter`] is the main entry point for all retrieval operations.
//! It:
//!
//! 1. Looks up the [`StrategyDescriptor`] via [`get_descriptor`].
//! 2. Embeds the query (dense always; sparse when keyword back-end is needed).
//! 3. Dispatches to the correct search functions from [`crate::search`].
//! 4. Passes raw result lists through [`ResultRanker`] for RRF fusion.
//! 5. Returns a fully assembled [`EvidencePack`].
//!
//! # Thread safety
//!
//! All fields are wrapped in `Arc` so the router can be cloned cheaply and
//! shared across concurrent request handlers.
//!
//! # Graph search defaults
//!
//! Graph-based strategies require seed node IDs and a traversal depth.
//! Phase 3 uses an empty seed list (the graph back-end falls back to a
//! full-graph walk when no seeds are provided) and [`DEFAULT_GRAPH_DEPTH`].
//! Phase 4 will extract seed IDs from the query embedding nearest-neighbours.

use std::sync::Arc;

use agent_rs_core::{CollectionName, RetrievalStrategy};
use tracing::instrument;

use crate::{
    Embedder, EvidencePack, ResultRanker, RetrievalError,
    reranker::{CrossEncoderReranker, RERANK_CANDIDATE_COUNT},
    search::{
        graph::{
            graph_constrained_search, graph_cot_search, graph_rag_context_search, graph_search,
        },
        hybrid::{GraphHybridParams, graph_hybrid_search, hybrid_search},
        keyword::keyword_search,
        triplet::triplet_search,
        vector::{summaries_search, temporal_search, vector_search},
    },
    strategy::get_descriptor,
};
use agent_rs_stores::{
    traits::{GraphStore, VectorStore},
    types::SparseVector as StoreSparseVector,
};

// ---- Helpers ----

/// Convert an [`crate::embedder::SparseVector`] (returned by the embedder) into
/// the [`StoreSparseVector`] expected by the search back-ends.
///
/// Both types are structurally identical; this is a newtype-bridge conversion.
fn to_store_sparse(sv: crate::SparseVector) -> StoreSparseVector {
    StoreSparseVector {
        indices: sv.indices,
        values: sv.values,
    }
}

// ---- Constants ----

/// Default graph traversal depth when no explicit depth is configured.
///
/// Shallower traversals are faster and typically sufficient for Phase 3.
/// Phase 4 will make this configurable per-query.
const DEFAULT_GRAPH_DEPTH: usize = 2;
// ---- RetrievalRouter ----

/// Routes queries to the appropriate search back-ends and returns fused results.
///
/// # Construction
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use agent_rs_retrieval::router::RetrievalRouter;
/// use agent_rs_retrieval::ranker::ResultRanker;
/// // embedder, vector_store, graph_store created elsewhere
/// // let router = RetrievalRouter::new(embedder, vector_store, graph_store, ranker, collection);
/// ```
pub struct RetrievalRouter {
    /// Embedding back-end shared across requests.
    embedder: Arc<dyn Embedder + Send + Sync>,
    /// Vector similarity search back-end.
    vector_store: Arc<dyn VectorStore + Send + Sync>,
    /// Graph traversal back-end.
    graph_store: Arc<dyn GraphStore + Send + Sync>,
    /// RRF ranker that fuses multi-list results.
    ranker: ResultRanker,
    /// Default collection to query for vector and keyword searches.
    collection: CollectionName,
    /// Optional dedicated collection holding linearised triplet embeddings.
    ///
    /// When `Some`, strategies whose descriptor sets
    /// [`StrategyDescriptor::uses_triplets`](crate::strategy::StrategyDescriptor::uses_triplets)
    /// run [`triplet_search`] in addition to their primary back-ends and
    /// fuse the triplet hits into the RRF ranking.
    triplet_collection: Option<CollectionName>,
    /// Optional dedicated collection holding chunk summary embeddings.
    ///
    /// When `Some`, [`RetrievalStrategy::Summaries`] searches this collection.
    /// When `None`, `Summaries` falls back to the main chunk collection, which
    /// returns empty results if summaries were never ingested there.
    summaries_collection: Option<CollectionName>,

    /// Optional cross-encoder reranker applied after RRF fusion.
    ///
    /// When `Some`, the top-[`RERANK_CANDIDATE_COUNT`] RRF results are passed
    /// through the cross-encoder before truncating to the caller's `limit`.
    reranker: Option<Arc<CrossEncoderReranker>>,
}

impl RetrievalRouter {
    /// Create a new router without a dedicated triplet collection.
    ///
    /// # Parameters
    ///
    /// - `embedder` — embedding back-end (dense + sparse).
    /// - `vector_store` — Qdrant or compatible vector store.
    /// - `graph_store` — Neo4j or compatible graph store.
    /// - `ranker` — RRF ranker; use [`ResultRanker::default()`] for standard
    ///   configuration.
    /// - `collection` — default collection name for vector/keyword searches.
    pub fn new(
        embedder: Arc<dyn Embedder + Send + Sync>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        ranker: ResultRanker,
        collection: CollectionName,
    ) -> Self {
        Self {
            embedder,
            vector_store,
            graph_store,
            ranker,
            collection,
            triplet_collection: None,
            summaries_collection: None,
            reranker: None,
        }
    }

    /// Attach an optional triplet collection to this router.
    ///
    /// When set, graph-oriented strategies (`Graph`, `GraphHybrid`, `GraphCoT`,
    /// `GraphConstrained`) will additionally dispatch a triplet vector search
    /// against `triplet_collection` and fuse the results via RRF.
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_triplet_collection(mut self, triplet_collection: CollectionName) -> Self {
        self.triplet_collection = Some(triplet_collection);
        self
    }

    /// Attach an optional summaries collection to this router.
    ///
    /// When set, [`RetrievalStrategy::Summaries`] searches this collection
    /// instead of the main chunk collection. Must match the collection name
    /// used in [`IngestionPipelineConfig::with_summaries_collection`].
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_summaries_collection(mut self, summaries_collection: CollectionName) -> Self {
        self.summaries_collection = Some(summaries_collection);
        self
    }

    /// Attach an optional cross-encoder reranker to this router.
    ///
    /// When set, the top-[`RERANK_CANDIDATE_COUNT`] results from RRF fusion
    /// are passed through the cross-encoder before the result set is truncated
    /// to the caller's `limit`. This improves precision at the cost of added
    /// ONNX inference latency per request.
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_reranker(mut self, reranker: Arc<CrossEncoderReranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Rewrite a raw user query for improved retrieval quality.
    ///
    /// Returns the (possibly modified) query string. Phase 3: no-op returning
    /// the input unchanged. Phase 4 will add LLM-based HyDE / query expansion.
    #[instrument(skip(self), fields(strategy = %strategy))]
    pub fn rewrite_query(&self, query: &str, strategy: RetrievalStrategy) -> String {
        // Phase 4: HyDE / expansion rewriting.
        tracing::debug!(
            strategy = %strategy,
            "query rewriting not yet implemented; using raw query"
        );
        query.to_owned()
    }

    /// Hydrate triplet search results by looking up the chunks that generated
    /// them. Each triplet carries a `chunk_id` in its payload; this method
    /// runs a filtered search on the main collection to retrieve the actual
    /// chunk documents with their `chunk_text` for proper ranking.
    async fn hydrate_chunks_from_triplets(
        &self,
        chunk_ids: &[String],
        dense: &agent_rs_core::Embedding,
        limit: usize,
    ) -> Vec<agent_rs_stores::types::SearchResult> {
        use std::collections::HashSet;
        let unique_ids: HashSet<&String> = chunk_ids.iter().collect();
        let mut hydrated = Vec::new();

        for chunk_id in unique_ids {
            let filter = serde_json::json!({
                "key": "chunk_id",
                "match": {"value": chunk_id}
            });
            match self
                .vector_store
                .search_with_filter(&self.collection, dense, 1, Some(filter))
                .await
            {
                Ok(mut results) => {
                    if let Some(chunk) = results.pop() {
                        hydrated.push(chunk);
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        chunk_id = %chunk_id,
                        error = %e,
                        "failed to hydrate triplet chunk"
                    );
                }
            }
            if hydrated.len() >= limit {
                break;
            }
        }
        hydrated
    }

    /// Derive seed node IDs for graph traversal by running a quick vector search
    /// and extracting document IDs from the top results.
    ///
    /// This bridges vector → graph: find relevant documents first, then traverse
    /// their entity graph neighborhood.
    async fn derive_seed_ids(
        &self,
        _query: &str,
        dense: &agent_rs_core::Embedding,
        limit: usize,
    ) -> Result<Vec<String>, RetrievalError> {
        let top_docs = vector_search(
            &*self.vector_store,
            &self.collection,
            dense,
            limit.min(10), // cap seed count
        )
        .await?;

        let seeds: Vec<String> = top_docs.into_iter().map(|r| r.id).collect();
        tracing::debug!(
            seed_count = seeds.len(),
            "derived graph seed IDs from vector search"
        );
        Ok(seeds)
    }

    /// Embed, dispatch, rank, and return an [`EvidencePack`] for `query`.
    ///
    /// # Algorithm
    ///
    /// 1. Rewrite query (Phase 3: no-op).
    /// 2. Embed dense (always required).
    /// 3. Embed sparse if the strategy touches the keyword back-end.
    /// 4. Dispatch to the correct search function(s) based on strategy.
    /// 5. Fuse result lists via RRF and return an [`EvidencePack`].
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] on embedding failure, or
    /// [`RetrievalError::Store`] on any search back-end failure.
    #[instrument(skip(self), fields(strategy = %strategy, limit))]
    pub async fn retrieve(
        &self,
        query: &str,
        strategy: RetrievalStrategy,
        limit: usize,
    ) -> Result<EvidencePack, RetrievalError> {
        let descriptor = get_descriptor(strategy);

        let rewritten = self.rewrite_query(query, strategy);
        let query_str: &str = &rewritten;

        // Always embed dense — every strategy needs at minimum a dense vector.
        let dense = self.embedder.embed_text(query_str).await?;

        // Embed sparse only when a keyword back-end will be called.
        let needs_sparse = descriptor
            .search_types
            .contains(&crate::strategy::SearchType::Keyword);
        let sparse_opt: Option<StoreSparseVector> = if needs_sparse {
            let sv = self.embedder.embed_sparse(query_str).await?;
            Some(to_store_sparse(sv))
        } else {
            None
        };

        // Dispatch and collect result lists for the ranker.
        let mut result_lists: Vec<Vec<agent_rs_stores::types::SearchResult>> = match strategy {
            // ---- Single-backend: dense vector ----
            RetrievalStrategy::Semantic => {
                let results =
                    vector_search(&*self.vector_store, &self.collection, &dense, limit).await?;
                vec![results]
            }

            // ---- Single-backend: summaries vector ----
            RetrievalStrategy::Summaries => {
                // Use the dedicated summaries collection when available;
                // fall back to the main collection otherwise.
                let target = self
                    .summaries_collection
                    .as_ref()
                    .unwrap_or(&self.collection);
                let results =
                    summaries_search(&*self.vector_store, target, &dense, limit).await?;
                vec![results]
            }

            // ---- Single-backend: temporal vector ----
            RetrievalStrategy::Temporal => {
                let results =
                    temporal_search(&*self.vector_store, &self.collection, &dense, limit).await?;
                vec![results]
            }

            // ---- Single-backend: keyword ----
            RetrievalStrategy::Keyword => {
                let sparse = sparse_opt.as_ref().ok_or_else(|| RetrievalError::Routing {
                    strategy: strategy.to_string(),
                    reason: "sparse embedding required but not computed".into(),
                })?;
                let results =
                    keyword_search(&*self.vector_store, &self.collection, &dense, sparse, limit)
                        .await?;
                vec![results]
            }

            // ---- Single-backend: graph (base traversal) ----
            // Seed graph traversal from vector search nearest neighbors
            RetrievalStrategy::Graph => {
                let seeds = self.derive_seed_ids(query_str, &dense, limit).await?;
                let results = graph_search(
                    &*self.graph_store,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;
                vec![results]
            }

            // ---- Single-backend: graph RAG completion ----
            RetrievalStrategy::RagCompletion => {
                let seeds = self.derive_seed_ids(query_str, &dense, limit).await?;
                let results = graph_rag_context_search(
                    &*self.graph_store,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;
                vec![results]
            }

            // ---- Single-backend: graph chain-of-thought ----
            RetrievalStrategy::GraphCoT => {
                let seeds = self.derive_seed_ids(query_str, &dense, limit).await?;
                let results = graph_cot_search(
                    &*self.graph_store,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;
                vec![results]
            }

            // ---- Parallel: vector + keyword (HYBRID) ----
            RetrievalStrategy::Hybrid => {
                let sparse = sparse_opt.as_ref().ok_or_else(|| RetrievalError::Routing {
                    strategy: strategy.to_string(),
                    reason: "sparse embedding required but not computed".into(),
                })?;
                let hybrid_results =
                    hybrid_search(&*self.vector_store, &self.collection, &dense, sparse, limit)
                        .await?;
                vec![
                    hybrid_results.vector_results,
                    hybrid_results.keyword_results,
                ]
            }

            // ---- Parallel: vector + keyword + graph (GRAPH_HYBRID) ----
            RetrievalStrategy::GraphHybrid => {
                let sparse = sparse_opt.as_ref().ok_or_else(|| RetrievalError::Routing {
                    strategy: strategy.to_string(),
                    reason: "sparse embedding required but not computed".into(),
                })?;
                let seeds = self.derive_seed_ids(query_str, &dense, limit).await?;
                let params = GraphHybridParams {
                    query_embedding: &dense,
                    sparse_query: sparse,
                    query: query_str,
                    seed_node_ids: &seeds,
                    max_depth: DEFAULT_GRAPH_DEPTH,
                    limit,
                };
                let hybrid_results = graph_hybrid_search(
                    &*self.vector_store,
                    &*self.graph_store,
                    &self.collection,
                    params,
                )
                .await?;
                let mut lists = vec![
                    hybrid_results.vector_results,
                    hybrid_results.keyword_results,
                ];
                if let Some(graph_res) = hybrid_results.graph_results {
                    lists.push(graph_res);
                }
                lists
            }

            // ---- Parallel: graph + vector (GRAPH_CONSTRAINED) ----
            RetrievalStrategy::GraphConstrained => {
                let seeds = self.derive_seed_ids(query_str, &dense, limit).await?;
                let (graph_result, vector_result) = tokio::join!(
                    graph_constrained_search(
                        &*self.graph_store,
                        query_str,
                        &seeds,
                        DEFAULT_GRAPH_DEPTH,
                        limit,
                    ),
                    vector_search(&*self.vector_store, &self.collection, &dense, limit),
                );
                vec![graph_result?, vector_result?]
            }
        };

        // Additional triplet search for graph-oriented strategies.
        //
        // This mirrors Cognee's brute_force_triplet_search: for strategies
        // that walk the knowledge graph, we also score complete
        // (source) -[relation]-> (target) triplets against the query embedding
        // and fuse them as a separate ranked list.
        if descriptor.uses_triplets {
            if let Some(triplet_collection) = self.triplet_collection.as_ref() {
                match triplet_search(&*self.vector_store, triplet_collection, &dense, limit).await {
                    Ok(triplet_results) => {
                        // Hydrate triplet results with their source chunks.
                        // Each triplet payload carries a chunk_id pointing to
                        // the chunk that generated it. We fetch those chunks
                        // from the main collection so the fused result list
                        // contains real chunk_text for ranking and evidence.
                        let chunk_ids: Vec<String> = triplet_results
                            .iter()
                            .filter_map(|r| {
                                r.payload
                                    .get("chunk_id")
                                    .and_then(|v| v.as_str())
                                    .map(String::from)
                            })
                            .collect();

                        if !chunk_ids.is_empty() {
                            // Retrieve the actual chunk documents via a filtered
                            // scroll on the main collection. For each chunk_id,
                            // run a filtered search with a dummy vector and take
                            // the first match. This is a simple approach; a
                            // proper scroll-by-id would be faster.
                            let hydrated = self
                                .hydrate_chunks_from_triplets(&chunk_ids, &dense, limit)
                                .await;
                            if !hydrated.is_empty() {
                                result_lists.push(hydrated);
                            }
                        }

                        tracing::debug!(
                            strategy = %strategy,
                            count = triplet_results.len(),
                            "triplet search contributed to RRF fusion"
                        );
                        result_lists.push(triplet_results);
                    }
                    Err(e) => {
                        // Triplet search is best-effort: a missing or empty
                        // collection should not fail the overall query. Log
                        // and continue with the primary back-end results.
                        tracing::warn!(
                            strategy = %strategy,
                            error = %e,
                            "triplet search failed; continuing without triplet evidence"
                        );
                    }
                }
            } else {
                tracing::debug!(
                    strategy = %strategy,
                    "triplet search requested but no triplet collection configured"
                );
            }
        }

        // RRF fusion. When reranking is enabled we request more candidates
        // (RERANK_CANDIDATE_COUNT) so the cross-encoder has a richer pool to
        // score; the final truncation to `limit` happens after reranking.
        let rrf_limit = if self.reranker.is_some() {
            RERANK_CANDIDATE_COUNT.max(limit)
        } else {
            limit
        };

        let mut evidence = self
            .ranker
            .rank(result_lists, query_str, strategy, rrf_limit);

        // Optional cross-encoder reranking stage.
        if let Some(reranker) = self.reranker.as_ref() {
            // Clone the candidates so we can fall back to the RRF-ranked
            // results if reranking fails. The clone is bounded by RRF_limit
            // (≤50 documents) so the cost is negligible compared to the
            // network round-trip to the reranker.
            let candidates: Vec<_> = evidence.documents.to_vec();
            tracing::debug!(
                strategy = %strategy,
                candidates = candidates.len(),
                top_k = limit,
                "cross-encoder reranking"
            );
            match reranker.rerank(query_str, candidates, limit).await {
                Ok(reranked) => {
                    evidence.documents = reranked;
                }
                Err(e) => {
                    // Reranking is best-effort: on failure (rate limit,
                    // network error, etc.) keep the original RRF-ranked
                    // results truncated to `limit`. This preserves retrieval
                    // availability and quality under transient API failures.
                    tracing::warn!(
                        strategy = %strategy,
                        error = %e,
                        "cross-encoder reranking failed; falling back to RRF-ranked results"
                    );
                    evidence.documents.truncate(limit);
                }
            }
        }

        Ok(evidence)
    }
}
