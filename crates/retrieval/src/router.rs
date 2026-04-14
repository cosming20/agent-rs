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

use agent_rs_core::{CollectionName, Embedding, RetrievalStrategy};
use tracing::instrument;

use crate::{
    ContextExpander, Embedder, EntityLinker, EvidencePack, MetadataFilterExtractor, QueryAnalyzer,
    QueryRewriter, ResultRanker, RetrievalError,
    graph_cot::GraphCotRetriever,
    metadata_filter::FilterCondition,
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
use agent_rs_stores::traits::{GraphStore, VectorStore};

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

    /// Optional iterative chain-of-thought retriever used only for the
    /// [`RetrievalStrategy::GraphCoT`] strategy.
    ///
    /// When `Some`, after the initial graph traversal the router invokes
    /// [`GraphCotRetriever::retrieve`] to iteratively generate follow-up
    /// questions and merge additional evidence. When `None`, `GraphCoT`
    /// behaves identically to a plain [`RetrievalStrategy::Graph`] search.
    graph_cot_retriever: Option<Arc<GraphCotRetriever>>,

    /// Optional LLM-based query decomposition engine.
    ///
    /// When `Some`, the router calls [`QueryRewriter::rewrite`] before
    /// dispatching to search back-ends. Complex queries are decomposed into
    /// up to 3 focused sub-queries; each is retrieved independently and the
    /// results are fused by RRF. Simple queries pass through unchanged.
    query_rewriter: Option<Arc<QueryRewriter>>,

    /// Optional LLM-based metadata hint extractor for Qdrant filter narrowing.
    ///
    /// When `Some`, [`MetadataFilterExtractor::extract`] is called on the
    /// original query before dispatch. Extracted hints (page range, article
    /// number, content-type flags) are translated to Qdrant `must` conditions
    /// and pushed down to the vector/hybrid search calls.
    metadata_filter: Option<Arc<MetadataFilterExtractor>>,

    /// Optional post-retrieval context expander using Neo4j structural edges.
    ///
    /// When `Some`, after retrieval and reranking the router calls
    /// [`ContextExpander::expand`] on the top hits to fetch their FOLLOWS
    /// neighbors and same-section siblings. The expanded context text is
    /// appended to each document's content in the returned [`EvidencePack`].
    context_expander: Option<Arc<ContextExpander>>,

    /// Optional entity → KG linker (spec B3).
    entity_linker: Option<Arc<EntityLinker>>,

    /// Optional query analyzer (spec B1).
    ///
    /// Intent classification + augmented query generation. For conceptual,
    /// negation, and disambiguation intents, generates additional search
    /// queries that target the specific retrieval need (synonym expansion,
    /// positive+exception split, multiple interpretations).
    query_analyzer: Option<Arc<QueryAnalyzer>>,
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
            graph_cot_retriever: None,
            query_rewriter: None,
            metadata_filter: None,
            context_expander: None,
            entity_linker: None,
            query_analyzer: None,
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

    /// Attach an optional iterative chain-of-thought retriever.
    ///
    /// When set, [`RetrievalStrategy::GraphCoT`] runs the initial graph
    /// traversal followed by up to `max_iterations` follow-up passes driven
    /// by an LLM. See [`crate::graph_cot::GraphCotRetriever`] for details.
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_graph_cot_retriever(mut self, retriever: Arc<GraphCotRetriever>) -> Self {
        self.graph_cot_retriever = Some(retriever);
        self
    }

    /// Attach an optional LLM-based query rewriter.
    ///
    /// When set, complex queries are decomposed into up to 3 focused sub-queries
    /// before dispatch. Each sub-query is retrieved independently and the results
    /// are fused by RRF. Simple queries pass through unchanged.
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_query_rewriter(mut self, rewriter: Arc<QueryRewriter>) -> Self {
        self.query_rewriter = Some(rewriter);
        self
    }

    /// Attach an optional LLM-based metadata filter extractor.
    ///
    /// When set, structural hints (page range, article number, content-type
    /// flags) are extracted from the query and translated to Qdrant payload
    /// filter conditions applied to vector and hybrid search calls.
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_metadata_filter(mut self, extractor: Arc<MetadataFilterExtractor>) -> Self {
        self.metadata_filter = Some(extractor);
        self
    }

    /// Attach an optional post-retrieval context expander.
    ///
    /// When set, after retrieval and reranking the expander fetches FOLLOWS
    /// neighbors and same-section siblings from Neo4j for each top hit and
    /// appends their text to the document content in the returned
    /// [`EvidencePack`].
    ///
    /// Returns the router by value to support builder-style construction.
    /// Attach an entity → KG linker (spec B3).
    pub fn with_entity_linker(mut self, linker: Arc<EntityLinker>) -> Self {
        self.entity_linker = Some(linker);
        self
    }

    /// Attach a query analyzer (spec B1) for intent classification +
    /// augmented query generation.
    pub fn with_query_analyzer(mut self, analyzer: Arc<QueryAnalyzer>) -> Self {
        self.query_analyzer = Some(analyzer);
        self
    }

    pub fn with_context_expander(mut self, expander: Arc<ContextExpander>) -> Self {
        self.context_expander = Some(expander);
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
    /// 1. LLM query rewriting (if [`QueryRewriter`] configured): decompose into sub-queries.
    /// 2. Extract metadata filter hints (if [`MetadataFilterExtractor`] configured).
    /// 3. Embed dense (always required).
    /// 4. For each sub-query (1 when not decomposed, 2-3 when decomposed):
    ///    a. Dispatch to the correct search function(s) based on strategy.
    /// 5. Fuse all result lists (across sub-queries) via RRF.
    /// 6. Optional cross-encoder reranking.
    /// 7. Optional post-retrieval context expansion (if [`ContextExpander`] configured).
    /// 8. Return the assembled [`EvidencePack`].
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

        // ---- Steps 1 + 1b + 2: PARALLEL pre-dispatch LLM calls ----
        //
        // Query rewriting, entity linking, and metadata extraction are
        // INDEPENDENT — none depends on the other's output. Running them
        // in parallel via tokio::join! cuts ~900ms → ~300ms per query
        // (latency = max of the three, not their sum).
        let word_count = query.split_whitespace().count();

        // Build futures for each optional LLM call.
        let rewrite_fut = async {
            if let Some(rewriter) = self.query_rewriter.as_ref().filter(|_| word_count >= 12) {
                let rewritten = rewriter.rewrite(query).await?;
                tracing::debug!(
                    sub_query_count = rewritten.sub_queries.len(),
                    was_decomposed = rewritten.was_decomposed,
                    word_count,
                    "query rewriting complete"
                );
                Ok::<Vec<String>, RetrievalError>(
                    rewritten
                        .sub_queries
                        .into_iter()
                        .map(|sq| sq.text)
                        .collect(),
                )
            } else {
                if word_count < 12 && self.query_rewriter.is_some() {
                    tracing::debug!(word_count, "short query — skipping LLM decomposition");
                }
                Ok(vec![self.rewrite_query(query, strategy)])
            }
        };

        let entity_link_fut = async {
            if let Some(linker) = self.entity_linker.as_ref() {
                let link_result = linker.link(query).await;
                if link_result.prefer_graph {
                    tracing::info!(
                        linked = link_result.linked_count,
                        entities = ?link_result.entities.iter().filter(|e| e.is_linked).map(|e| &e.mention).collect::<Vec<_>>(),
                        "entity linker suggests graph-first strategy"
                    );
                }
                // TODO: use link_result.prefer_graph to auto-switch strategy
            }
        };

        let metadata_filter_fut = async {
            if let Some(extractor) = self.metadata_filter.as_ref() {
                let hints = extractor.extract(query).await;
                if let Some(conditions) = MetadataFilterExtractor::to_qdrant_filter(&hints) {
                    tracing::debug!(
                        num_conditions = conditions.len(),
                        "metadata filter conditions will be applied to search"
                    );
                    return Some(filter_conditions_to_json(&conditions));
                }
            }
            None
        };

        // Query analyzer: intent classification + augmented queries.
        let analyze_fut = async {
            if let Some(analyzer) = self.query_analyzer.as_ref() {
                Some(analyzer.analyze(query).await)
            } else {
                None
            }
        };

        // Run all four in parallel.
        let (sub_queries_result, _entity_link, qdrant_filter_json, analysis_opt) = tokio::join!(
            rewrite_fut,
            entity_link_fut,
            metadata_filter_fut,
            analyze_fut
        );
        let mut sub_queries = sub_queries_result?;

        // Merge augmented queries from analyzer into sub_queries.
        // Analyzer generates intent-specific queries (synonym expansions for
        // conceptual, positive+exception for negation, interpretations for
        // disambiguation) that complement the rewriter's decomposition.
        if let Some(analysis) = analysis_opt {
            tracing::info!(
                intent = ?analysis.intent,
                augmented = analysis.augmented_queries.len(),
                "query analyzer: intent classified"
            );
            // Add augmented queries that aren't already present.
            for aug in &analysis.augmented_queries {
                let trimmed = aug.trim();
                if !trimmed.is_empty() && !sub_queries.iter().any(|sq| sq.trim() == trimmed) {
                    sub_queries.push(trimmed.to_owned());
                }
            }
            // Cap total sub-queries to prevent explosion.
            sub_queries.truncate(5);
        }

        // ---- Step 3–4: Embed + dispatch per sub-query ----
        //
        // Each sub-query is embedded independently and dispatched to the search
        // back-ends. Result lists from all sub-queries are pooled together for
        // RRF fusion in step 5.
        let mut all_result_lists: Vec<Vec<agent_rs_stores::types::SearchResult>> = Vec::new();

        for query_str in &sub_queries {
            // HyDE DISABLED — caused -48pp regression on semantic search.
            // The hypothetical answer is too specific for Romanian legal text
            // and mismatches with the actual chunk embeddings. Raw query
            // embedding + Docling heading-prefixed chunks works better.
            // TODO: re-enable selectively for Conceptual queries only.
            let embed_text = query_str.to_string();

            // Always embed dense — every strategy needs at minimum a dense vector.
            // Sparse embeddings are no longer computed client-side: the store layer
            // sends the raw query text as `Document { model: "qdrant/bm25" }` and
            // Qdrant performs BM25 inference server-side.
            let dense = self.embedder.embed_text(&embed_text).await?;

            let per_sub_lists = self
                .dispatch_single(
                    query_str,
                    strategy,
                    &dense,
                    limit,
                    qdrant_filter_json.as_ref(),
                )
                .await?;

            all_result_lists.extend(per_sub_lists);

            // Triplet search for graph-oriented strategies (same logic as before,
            // run once per sub-query and merged into all_result_lists).
            if descriptor.uses_triplets {
                if let Some(triplet_collection) = self.triplet_collection.as_ref() {
                    match triplet_search(&*self.vector_store, triplet_collection, &dense, limit)
                        .await
                    {
                        Ok(triplet_results) => {
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
                                let hydrated = self
                                    .hydrate_chunks_from_triplets(&chunk_ids, &dense, limit)
                                    .await;
                                if !hydrated.is_empty() {
                                    all_result_lists.push(hydrated);
                                }
                            }

                            tracing::debug!(
                                strategy = %strategy,
                                count = triplet_results.len(),
                                "triplet search contributed to RRF fusion"
                            );
                            all_result_lists.push(triplet_results);
                        }
                        Err(e) => {
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
        }

        // Use the original query string for RRF ranking / reranking labels.
        // When the query was not decomposed, sub_queries[0] == original query.
        let primary_query = sub_queries.first().map(String::as_str).unwrap_or(query);

        // ---- Step 5: RRF fusion ----
        // When reranking is enabled we request more candidates
        // (RERANK_CANDIDATE_COUNT) so the cross-encoder has a richer pool to
        // score; the final truncation to `limit` happens after reranking.
        let rrf_limit = if self.reranker.is_some() {
            RERANK_CANDIDATE_COUNT.max(limit)
        } else {
            limit
        };

        let mut evidence = self
            .ranker
            .rank(all_result_lists, primary_query, strategy, rrf_limit);

        // ---- Step 6: Optional cross-encoder reranking ----
        if let Some(reranker) = self.reranker.as_ref() {
            // Clone the candidates so we can fall back to the RRF-ranked
            // results if reranking fails. The clone is bounded by rrf_limit
            // (≤50 documents) so the cost is negligible compared to the
            // network round-trip to the reranker.
            let candidates: Vec<_> = evidence.documents.to_vec();
            tracing::debug!(
                strategy = %strategy,
                candidates = candidates.len(),
                top_k = limit,
                "cross-encoder reranking"
            );
            match reranker.rerank(primary_query, candidates, limit).await {
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

        // ---- Step 7: Optional post-retrieval context expansion ----
        //
        // For each top hit, fetch FOLLOWS neighbors and same-section siblings
        // from Neo4j. The expanded context text is appended to each document's
        // `chunk_text` payload field so the caller's synthesis step has richer
        // surrounding text available.
        if let Some(expander) = self.context_expander.as_ref() {
            let before_count = evidence.documents.len();

            // Build (chunk_id, original_text) pairs from the evidence documents.
            let hit_chunks: Vec<(String, String)> = evidence
                .documents
                .iter()
                .map(|doc| {
                    let text = doc
                        .payload
                        .get("chunk_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                    (doc.id.clone(), text)
                })
                .collect();

            match expander.expand(&hit_chunks).await {
                Ok(expanded) => {
                    // Merge expanded context into the evidence document payloads.
                    // For each hit, collect context_before + context_after + siblings
                    // and append their text to chunk_text with a separator.
                    for (doc, expanded_chunk) in evidence.documents.iter_mut().zip(expanded.iter())
                    {
                        let mut extra_parts: Vec<&str> = Vec::new();

                        for ctx in &expanded_chunk.context_before {
                            if !ctx.text.is_empty() {
                                extra_parts.push(ctx.text.as_str());
                            }
                        }
                        for ctx in &expanded_chunk.context_after {
                            if !ctx.text.is_empty() {
                                extra_parts.push(ctx.text.as_str());
                            }
                        }
                        for sibling in &expanded_chunk.section_siblings {
                            if !sibling.text.is_empty() {
                                extra_parts.push(sibling.text.as_str());
                            }
                        }

                        if !extra_parts.is_empty() {
                            let original = doc
                                .payload
                                .get("chunk_text")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_owned();
                            let appended = format!("{}\n\n{}", original, extra_parts.join("\n\n"));
                            if let Some(payload_obj) = doc.payload.as_object_mut() {
                                payload_obj.insert(
                                    "chunk_text".to_owned(),
                                    serde_json::Value::String(appended),
                                );
                            }
                        }
                    }

                    let after_count: usize = expanded
                        .iter()
                        .map(|e| {
                            1 + e.context_before.len()
                                + e.context_after.len()
                                + e.section_siblings.len()
                        })
                        .sum();
                    tracing::info!(
                        before = before_count,
                        after = after_count,
                        "context expansion merged into evidence pack"
                    );
                }
                Err(e) => {
                    // Context expansion is best-effort: on failure keep the
                    // original evidence unchanged.
                    tracing::warn!(
                        strategy = %strategy,
                        error = %e,
                        "context expansion failed; returning unexpanded evidence"
                    );
                }
            }
        }

        Ok(evidence)
    }

    /// Dispatch a single (sub-)query to the appropriate search back-ends for
    /// the given `strategy`.
    ///
    /// Returns the raw per-backend result lists to be pooled and fused by the
    /// caller. The `filter` JSON is pushed down to Qdrant vector/hybrid calls
    /// when provided.
    async fn dispatch_single(
        &self,
        query_str: &str,
        strategy: RetrievalStrategy,
        dense: &Embedding,
        limit: usize,
        filter: Option<&serde_json::Value>,
    ) -> Result<Vec<Vec<agent_rs_stores::types::SearchResult>>, RetrievalError> {
        match strategy {
            // ---- Single-backend: dense vector ----
            //
            // Semantic search relies PURELY on vector similarity — metadata
            // payload filters are intentionally NOT applied here. Tokenized
            // text matching on payload fields (like section_title) is a keyword
            // operation that reduces recall without semantic justification.
            // The metadata filter is applied to Hybrid/GraphHybrid where
            // keyword and structural signals complement vector similarity.
            RetrievalStrategy::Semantic => {
                let results =
                    vector_search(&*self.vector_store, &self.collection, dense, limit).await?;
                Ok(vec![results])
            }

            // ---- Single-backend: summaries vector ----
            RetrievalStrategy::Summaries => {
                // Use the dedicated summaries collection when available;
                // fall back to the main collection otherwise.
                // Metadata filters are not applied to summaries (they may not
                // carry the same payload schema as chunk collections).
                let target = self
                    .summaries_collection
                    .as_ref()
                    .unwrap_or(&self.collection);
                let results = summaries_search(&*self.vector_store, target, dense, limit).await?;
                Ok(vec![results])
            }

            // ---- Single-backend: temporal vector ----
            RetrievalStrategy::Temporal => {
                let results =
                    temporal_search(&*self.vector_store, &self.collection, dense, limit).await?;
                Ok(vec![results])
            }

            // ---- Single-backend: keyword ----
            // Query text is forwarded to Qdrant as a `Document { model: "qdrant/bm25" }`
            // for server-side BM25 sparse inference — no local SPLADE model needed.
            RetrievalStrategy::Keyword => {
                let results = keyword_search(
                    &*self.vector_store,
                    &self.collection,
                    dense,
                    query_str,
                    limit,
                )
                .await?;
                Ok(vec![results])
            }

            // ---- Single-backend: graph (base traversal) ----
            // Seed graph traversal from vector search nearest neighbors.
            // Metadata filter is applied to the seed derivation vector search.
            RetrievalStrategy::Graph => {
                let seeds = self.derive_seed_ids(query_str, dense, limit).await?;
                let results = graph_search(
                    &*self.graph_store,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;
                Ok(vec![results])
            }

            // ---- Single-backend: graph RAG completion ----
            RetrievalStrategy::RagCompletion => {
                let seeds = self.derive_seed_ids(query_str, dense, limit).await?;
                let results = graph_rag_context_search(
                    &*self.graph_store,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;
                Ok(vec![results])
            }

            // ---- Single-backend: graph chain-of-thought ----
            //
            // Two-phase flow:
            //   1. Initial retrieval: graph_cot_search (== graph_search on the
            //      derived seeds) produces the starting evidence pool.
            //   2. If a `GraphCotRetriever` is configured, that pool is passed
            //      through the iterative CoT loop. Each follow-up question is
            //      embedded with this router's embedder, seeds are re-derived
            //      from a quick vector search, and graph_search is invoked to
            //      produce new candidates which are merged and deduplicated by
            //      the retriever.
            //   3. If no retriever is configured, the initial pool is returned
            //      unchanged (equivalent to plain graph search).
            RetrievalStrategy::GraphCoT => {
                let seeds = self.derive_seed_ids(query_str, dense, limit).await?;
                let initial = graph_cot_search(
                    &*self.graph_store,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;

                let final_results = if let Some(cot) = self.graph_cot_retriever.as_ref() {
                    // Closure that re-runs a graph retrieval for a follow-up
                    // question. Captures `&self` by reference (the enclosing
                    // `retrieve` future outlives the CoT loop, so the borrow
                    // is valid). Each call:
                    //   a) embeds the follow-up question (dense only),
                    //   b) derives fresh seeds from the main collection, and
                    //   c) runs graph_search to the same depth/limit as the
                    //      initial pass.
                    let this = self;
                    let depth = DEFAULT_GRAPH_DEPTH;
                    let cot_limit = limit;
                    let retrieve_fn = |follow_up: String| async move {
                        let f_dense = this.embedder.embed_text(&follow_up).await?;
                        let f_seeds = this
                            .derive_seed_ids(&follow_up, &f_dense, cot_limit)
                            .await?;
                        graph_search(&*this.graph_store, &follow_up, &f_seeds, depth, cot_limit)
                            .await
                    };

                    cot.retrieve(query_str, initial, retrieve_fn).await
                } else {
                    tracing::debug!(
                        "GraphCoT: no retriever configured; returning initial graph pool"
                    );
                    initial
                };

                Ok(vec![final_results])
            }

            // ---- Parallel: vector + keyword (HYBRID) ----
            RetrievalStrategy::Hybrid => {
                // When a metadata filter is present, use separate filtered vector
                // and keyword searches rather than the combined hybrid path (which
                // does not yet accept a filter parameter).
                if let Some(f) = filter {
                    let (vector_result, keyword_result) = tokio::join!(
                        self.vector_store.search_with_filter(
                            &self.collection,
                            dense,
                            limit,
                            Some(f.clone())
                        ),
                        keyword_search(
                            &*self.vector_store,
                            &self.collection,
                            dense,
                            query_str,
                            limit
                        ),
                    );
                    Ok(vec![
                        vector_result.map_err(RetrievalError::Store)?,
                        keyword_result?,
                    ])
                } else {
                    let hybrid_results = hybrid_search(
                        &*self.vector_store,
                        &self.collection,
                        dense,
                        query_str,
                        limit,
                    )
                    .await?;
                    Ok(vec![
                        hybrid_results.vector_results,
                        hybrid_results.keyword_results,
                    ])
                }
            }

            // ---- Parallel: vector + keyword + graph (GRAPH_HYBRID) ----
            RetrievalStrategy::GraphHybrid => {
                let seeds = self.derive_seed_ids(query_str, dense, limit).await?;
                if let Some(f) = filter {
                    let (vector_result, keyword_result, graph_result) = tokio::join!(
                        self.vector_store.search_with_filter(
                            &self.collection,
                            dense,
                            limit,
                            Some(f.clone())
                        ),
                        keyword_search(
                            &*self.vector_store,
                            &self.collection,
                            dense,
                            query_str,
                            limit
                        ),
                        graph_search(
                            &*self.graph_store,
                            query_str,
                            &seeds,
                            DEFAULT_GRAPH_DEPTH,
                            limit
                        ),
                    );
                    Ok(vec![
                        vector_result.map_err(RetrievalError::Store)?,
                        keyword_result?,
                        graph_result?,
                    ])
                } else {
                    let params = GraphHybridParams {
                        query_embedding: dense,
                        query_text: query_str,
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
                    Ok(lists)
                }
            }

            // ---- Parallel: graph + vector (GRAPH_CONSTRAINED) ----
            RetrievalStrategy::GraphConstrained => {
                let seeds = self.derive_seed_ids(query_str, dense, limit).await?;
                if let Some(f) = filter {
                    let (graph_result, vector_result) = tokio::join!(
                        graph_constrained_search(
                            &*self.graph_store,
                            query_str,
                            &seeds,
                            DEFAULT_GRAPH_DEPTH,
                            limit,
                        ),
                        self.vector_store.search_with_filter(
                            &self.collection,
                            dense,
                            limit,
                            Some(f.clone())
                        ),
                    );
                    Ok(vec![
                        graph_result?,
                        vector_result.map_err(RetrievalError::Store)?,
                    ])
                } else {
                    let (graph_result, vector_result) = tokio::join!(
                        graph_constrained_search(
                            &*self.graph_store,
                            query_str,
                            &seeds,
                            DEFAULT_GRAPH_DEPTH,
                            limit,
                        ),
                        vector_search(&*self.vector_store, &self.collection, dense, limit),
                    );
                    Ok(vec![graph_result?, vector_result?])
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert a slice of [`FilterCondition`] into the `serde_json::Value` format
/// that the Qdrant store's `search_with_filter` understands.
///
/// Produces a `{"must": [...]}` object where each element encodes one condition.
/// The `json_to_qdrant_filter` function in the stores crate handles this format.
fn filter_conditions_to_json(conditions: &[FilterCondition]) -> serde_json::Value {
    let must: Vec<serde_json::Value> = conditions
        .iter()
        .map(|c| match c {
            FilterCondition::Match { field, value } => {
                serde_json::json!({
                    "key": field,
                    "match": { "value": value }
                })
            }
            FilterCondition::Range { field, gte, lte } => {
                let mut range = serde_json::Map::new();
                if let Some(v) = gte {
                    range.insert("gte".to_owned(), serde_json::json!(v));
                }
                if let Some(v) = lte {
                    range.insert("lte".to_owned(), serde_json::json!(v));
                }
                serde_json::json!({ "key": field, "range": range })
            }
            FilterCondition::Contains { field, text } => {
                // Use Qdrant's full-text `match.text` (tokenized substring
                // search within the field value) rather than exact `match.value`.
                // "188" will match "Articolul 188 - Omorul" via token overlap.
                serde_json::json!({
                    "key": field,
                    "match": { "text": text }
                })
            }
        })
        .collect();

    serde_json::json!({ "must": must })
}
