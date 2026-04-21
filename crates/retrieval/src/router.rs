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
use std::time::Duration;

use agent_rs_core::{CollectionName, Embedding, RetrievalStrategy, UserId};
use agent_rs_observability::metrics::RetrievalMetrics;
use tracing::instrument;

use crate::{
    ContextExpander, Embedder, EntityLinker, EvidencePack, MetadataFilterExtractor, QueryRewriter,
    ResultRanker, RetrievalError,
    graph_cot::GraphCotRetriever,
    metadata_filter::FilterCondition,
    reranker::{CrossEncoderReranker, RERANK_CANDIDATE_COUNT},
    search::{
        graph::{
            graph_constrained_search, graph_cot_search, graph_rag_context_search, graph_search,
        },
        triplet::triplet_search,
    },
    strategy::get_descriptor,
    weighted_keywords::{
        WeightedKeywordExtractor, WeightedKeywordIntegrationConfig, WeightedKeywords,
        adjust_for_weighted_keywords,
    },
};
use agent_rs_stores::traits::{GraphStore, VectorStore};

// ---- Constants ----

/// Default graph traversal depth when no explicit depth is configured.
///
/// Shallower traversals are faster and typically sufficient for Phase 3.
/// Phase 4 will make this configurable per-query.
const DEFAULT_GRAPH_DEPTH: usize = 2;

// ---- SectionFilter ----

/// Optional payload filter applied as a hard post-fusion predicate so that
/// keyword and graph branches honour the same constraint as the vector
/// branch (Codex audit gap #10).
///
/// Applied AFTER RRF fusion and BEFORE cross-encoder reranking so that
/// off-section/off-document chunks are dropped regardless of which search
/// branch surfaced them.
#[derive(Debug, Clone, Default)]
pub struct SectionFilter {
    /// Restrict results to chunks whose payload `doc_id` matches this value.
    pub doc_id: Option<String>,
    /// Restrict results to chunks whose payload `section_id` matches this value.
    pub section_id: Option<String>,
}

impl SectionFilter {
    /// Returns `true` when neither `doc_id` nor `section_id` is set, meaning
    /// the filter is a no-op and can be skipped entirely.
    pub fn is_empty(&self) -> bool {
        self.doc_id.is_none() && self.section_id.is_none()
    }
}

/// Returns `true` when `payload` satisfies all non-`None` constraints in
/// `filter`.
///
/// Each field is checked independently; a `None` constraint always passes.
/// Missing or non-string payload fields fail the corresponding constraint.
pub(crate) fn matches_section_filter(payload: &serde_json::Value, filter: &SectionFilter) -> bool {
    let doc_ok = filter
        .doc_id
        .as_deref()
        .is_none_or(|want| payload.get("doc_id").and_then(|v| v.as_str()) == Some(want));
    let sec_ok = filter
        .section_id
        .as_deref()
        .is_none_or(|want| payload.get("section_id").and_then(|v| v.as_str()) == Some(want));
    doc_ok && sec_ok
}
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
    ///
    /// When `Some`, before retrieval dispatch the router calls
    /// [`EntityLinker::link`] to check if query entities exist in Neo4j.
    /// If linked entities are found, graph-based strategies are preferred.
    entity_linker: Option<Arc<EntityLinker>>,

    /// Optional shared metric instruments.
    ///
    /// When `Some`, the router emits metrics for reranker fallback events.
    /// Passed from the top-level startup wiring via [`with_metrics`].
    metrics: Option<Arc<RetrievalMetrics>>,

    /// Optional LLM-backed weighted-keyword extractor.
    ///
    /// When `Some`, runs in the pre-dispatch parallel join alongside the
    /// query rewriter and produces an `Arc<WeightedKeywords>` consumed by
    /// the rerank adjustment (and, in future commits, by BM25 query
    /// rewrite and entity-linker candidate prioritisation). Failures and
    /// branch-local timeouts collapse to `WeightedKeywords::empty()`, so
    /// every consumer's empty-input path is byte-identical to when this
    /// extractor is absent.
    weighted_keyword_extractor: Option<Arc<WeightedKeywordExtractor>>,

    /// Per-consumer integration toggles for the weighted-keyword signal.
    ///
    /// Default: all three consumers (BM25, rerank, entity_linker) enabled.
    /// Set any field to `None` at construction time to disable that
    /// consumer's integration without touching the extractor.
    weighted_keyword_integration: WeightedKeywordIntegrationConfig,

    /// Branch-local timeout for the weighted-keyword extraction future.
    ///
    /// If the LLM call does not complete within this budget, the branch
    /// resolves to `WeightedKeywords::empty()` — preventing a slow
    /// extractor from extending the router's overall deadline.
    weighted_keyword_branch_budget: Duration,
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
            metrics: None,
            weighted_keyword_extractor: None,
            weighted_keyword_integration: WeightedKeywordIntegrationConfig::default(),
            weighted_keyword_branch_budget: Duration::from_secs(6),
        }
    }

    /// Attach an LLM-backed weighted-keyword extractor.
    ///
    /// The extractor runs in parallel with `query_rewriter`, `entity_linker`,
    /// and `metadata_filter` (all subject to their own enable conditions)
    /// and produces weighted terms consumed by the rerank adjustment. See
    /// [`weighted_keywords`](crate::weighted_keywords) for the integration
    /// contract.
    pub fn with_weighted_keyword_extractor(
        mut self,
        extractor: Arc<WeightedKeywordExtractor>,
    ) -> Self {
        self.weighted_keyword_extractor = Some(extractor);
        self
    }

    /// Override the per-consumer integration config for weighted keywords.
    ///
    /// Default enables all three consumers with default per-consumer
    /// sub-configs. Set fields to `None` to disable that consumer's
    /// integration.
    pub fn with_weighted_keyword_integration(
        mut self,
        cfg: WeightedKeywordIntegrationConfig,
    ) -> Self {
        self.weighted_keyword_integration = cfg;
        self
    }

    /// Override the branch-local timeout for the weighted-keyword extractor.
    pub fn with_weighted_keyword_branch_budget(mut self, budget: Duration) -> Self {
        self.weighted_keyword_branch_budget = budget;
        self
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

    /// Attach shared metric instruments for operational observability.
    ///
    /// When set, the router emits metrics for reranker fallback events.
    /// Typically called once at startup with the globally shared [`RetrievalMetrics`].
    ///
    /// Returns the router by value to support builder-style construction.
    pub fn with_metrics(mut self, metrics: Arc<RetrievalMetrics>) -> Self {
        self.metrics = Some(metrics);
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

    /// Rehydrate triplet search results into chunk-shaped [`SearchResult`]s.
    ///
    /// Each triplet point in the triplets collection carries a `chunk_id`
    /// field pointing to the chunk it was extracted from. This method:
    ///
    /// 1. Walks the triplet hits in their original rank order and collects
    ///    `chunk_id`s, deduplicating while preserving order (so the best
    ///    triplet score wins when the same chunk is cited by multiple
    ///    triplets).
    /// 2. Batch-fetches the authoritative chunk payloads from the main
    ///    collection via [`VectorStore::get_by_ids`], which returns only
    ///    payloads for IDs that actually exist — orphan triplets whose
    ///    chunks have been deleted are silently dropped.
    /// 3. Emits [`SearchResult`]s keyed by `chunk_id` with:
    ///    - the chunk's full payload (so `chunk_text` / `section_title` /
    ///      `page_start` / `source` are populated);
    ///    - the triplet's score (carrying the "triplet ranked close to
    ///      query" signal into RRF fusion).
    ///
    /// Emitting chunk IDs is critical for RRF deduplication: when the same
    /// chunk surfaces from vector search and triplet search, RRF combines
    /// their reciprocal ranks instead of double-counting.
    async fn hydrate_chunks_from_triplets(
        &self,
        triplet_results: &[agent_rs_stores::types::SearchResult],
        user_id: &agent_rs_core::UserId,
        limit: usize,
    ) -> Vec<agent_rs_stores::types::SearchResult> {
        use std::collections::{HashMap, HashSet};

        let mut seen: HashSet<String> = HashSet::new();
        let mut ordered: Vec<(String, agent_rs_core::Score)> = Vec::new();
        for tr in triplet_results {
            let Some(chunk_id) = tr.payload.get("chunk_id").and_then(|v| v.as_str()) else {
                continue;
            };
            if seen.insert(chunk_id.to_owned()) {
                ordered.push((chunk_id.to_owned(), tr.score));
            }
            if ordered.len() >= limit {
                break;
            }
        }
        if ordered.is_empty() {
            return Vec::new();
        }

        let ids: Vec<String> = ordered.iter().map(|(id, _)| id.clone()).collect();
        let payloads = match self
            .vector_store
            .retrieve_chunks_for_user(&self.collection, user_id, &ids)
            .await
        {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    n_ids = ids.len(),
                    "failed to rehydrate triplet-sourced chunks; dropping triplet contribution"
                );
                return Vec::new();
            }
        };
        let payload_by_id: HashMap<String, serde_json::Value> =
            payloads.into_iter().map(|p| (p.id, p.payload)).collect();

        ordered
            .into_iter()
            .filter_map(|(id, score)| {
                let payload = payload_by_id.get(&id)?.clone();
                Some(agent_rs_stores::types::SearchResult { id, score, payload })
            })
            .collect()
    }

    /// Derive seed node IDs for graph traversal by running a quick vector search
    /// scoped to `user_id`, and extracting document IDs from the top results.
    ///
    /// This bridges vector → graph: find relevant documents first, then traverse
    /// their entity graph neighborhood. The search is user-scoped so seeds are
    /// always within the requesting user's data boundary.
    async fn derive_seed_ids(
        &self,
        user_id: &UserId,
        _query: &str,
        dense: &agent_rs_core::Embedding,
        limit: usize,
    ) -> Result<Vec<String>, RetrievalError> {
        let top_docs = self
            .vector_store
            .search_for_user(
                &self.collection,
                user_id,
                dense,
                limit.min(10), // cap seed count
                None,
            )
            .await
            .map_err(RetrievalError::Store)?;

        let seeds: Vec<String> = top_docs.into_iter().map(|r| r.id).collect();
        tracing::debug!(
            seed_count = seeds.len(),
            "derived graph seed IDs from user-scoped vector search"
        );
        Ok(seeds)
    }

    /// Full retrieval pipeline with mandatory user scope and an optional hard
    /// post-fusion predicate.
    ///
    /// `user_id` is the authoritative multi-tenant scope: every search
    /// branch (vector, keyword, graph) must produce results belonging to this
    /// user. When `section_filter` is `Some` and non-empty, any fused result
    /// whose payload does not match the requested `doc_id` / `section_id` is
    /// additionally dropped *after* RRF fusion but *before* cross-encoder
    /// reranking (Codex audit gap #10).
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::Embedding`] on embedding failure, or
    /// [`RetrievalError::Store`] on any search back-end failure.
    #[instrument(skip(self), fields(strategy = %strategy, limit, user_id = %user_id))]
    pub async fn retrieve_filtered(
        &self,
        user_id: &UserId,
        query: &str,
        strategy: RetrievalStrategy,
        limit: usize,
        section_filter: Option<SectionFilter>,
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

        // Weighted-keyword extraction — soft-optional with branch-local
        // timeout. Any failure / timeout collapses to empty so consumers
        // observe byte-identical behaviour to the extractor-absent path.
        let weighted_keyword_fut = async {
            let Some(extractor) = self.weighted_keyword_extractor.as_ref() else {
                return Arc::new(WeightedKeywords::empty());
            };
            let budget = self.weighted_keyword_branch_budget;
            match tokio::time::timeout(budget, extractor.extract(query)).await {
                Ok(Ok(kw)) => {
                    tracing::debug!(
                        terms = kw.terms.len(),
                        "weighted keyword extraction complete"
                    );
                    Arc::new(kw)
                }
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "weighted keyword extractor returned error; using empty");
                    Arc::new(WeightedKeywords::empty())
                }
                Err(_) => {
                    tracing::warn!(
                        budget_ms = budget.as_millis() as u64,
                        "weighted keyword branch exceeded budget; using empty"
                    );
                    Arc::new(WeightedKeywords::empty())
                }
            }
        };

        // Run all four in parallel.
        let (sub_queries_result, _entity_link, qdrant_filter_json, weighted_keywords) = tokio::join!(
            rewrite_fut,
            entity_link_fut,
            metadata_filter_fut,
            weighted_keyword_fut,
        );
        let sub_queries = sub_queries_result?;

        // ---- Step 3–4: Embed + dispatch per sub-query ----
        //
        // Each sub-query is embedded independently and dispatched to the search
        // back-ends. Result lists from all sub-queries are pooled together for
        // RRF fusion in step 5.
        let mut all_result_lists: Vec<Vec<agent_rs_stores::types::SearchResult>> = Vec::new();

        // HyDE gate — default ON, opt-out via env.
        //
        // An earlier eval regressed when HyDE was globally enabled on top
        // of plain vector search; the retrieval stack since grew HyDE into
        // an *additional* signal rather than a replacement (the raw-query
        // embedding is always retained, and cross-encoder reranking
        // recovers when a hypothetical drifts). That restructuring
        // justifies flipping the default to ON — the worst case is one
        // extra embed + search round-trip when HyDE's hypothetical fails
        // to inform the ranking, and the cross-encoder still sees the
        // raw-query hits.
        //
        // Operators can disable HyDE at runtime with
        // `AGENT_RS__RETRIEVAL__HYDE_DISABLED=1` — useful for A/B
        // evaluation or when the configured HyDE model is unavailable.
        let hyde_disabled = matches!(
            std::env::var("AGENT_RS__RETRIEVAL__HYDE_DISABLED").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
        );
        let mut hyde_texts: Vec<String> = Vec::new();
        if !hyde_disabled && sub_queries.len() == 1 {
            if let Some(rewriter) = self.query_rewriter.as_ref() {
                let hypothetical = rewriter.hyde_rewrite(query).await;
                // `hyde_rewrite` falls back to the input on any failure —
                // skip the extra embed/search round-trip in that case.
                if !hypothetical.is_empty() && hypothetical != query {
                    tracing::debug!(
                        hyde_len = hypothetical.len(),
                        "HyDE expansion produced hypothetical document"
                    );
                    hyde_texts.push(hypothetical);
                }
            }
        }

        for query_str in sub_queries.iter().chain(hyde_texts.iter()) {
            let embed_text = query_str.to_string();

            // Always embed dense — every strategy needs at minimum a dense vector.
            // Sparse embeddings are no longer computed client-side: the store layer
            // sends the raw query text as `Document { model: "qdrant/bm25" }` and
            // Qdrant performs BM25 inference server-side.
            let dense = self.embedder.embed_text(&embed_text).await?;

            let per_sub_lists = self
                .dispatch_single(
                    user_id,
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
                    match triplet_search(
                        &*self.vector_store,
                        triplet_collection,
                        user_id,
                        &dense,
                        limit,
                    )
                    .await
                    {
                        Ok(triplet_results) => {
                            // Triplet payloads carry only (subject, predicate, object,
                            // chunk_id, triplet_text) — they lack `chunk_text`,
                            // `source`, `page_start`, `section_title` that the
                            // downstream evidence pack expects. We rehydrate to
                            // chunk-shaped hits before fusion so empty rows never
                            // reach the user and RRF can dedup triplet-sourced
                            // chunks against vector/keyword/graph hits by chunk_id.
                            let hydrated = self
                                .hydrate_chunks_from_triplets(&triplet_results, user_id, limit)
                                .await;
                            tracing::debug!(
                                strategy = %strategy,
                                triplet_count = triplet_results.len(),
                                hydrated = hydrated.len(),
                                "triplet search contributed rehydrated chunks to RRF fusion"
                            );
                            if !hydrated.is_empty() {
                                all_result_lists.push(hydrated);
                            }
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

        // ---- Step 5b: Hard post-fusion section/doc predicate (Codex gap #10) ----
        //
        // Applied after RRF fusion and BEFORE reranking so that keyword (BM25)
        // and graph branches honour the same doc_id / section_id constraint
        // that the vector branch satisfies via its Qdrant payload filter.
        if let Some(ref filter) = section_filter {
            if !filter.is_empty() {
                let before = evidence.documents.len();
                evidence
                    .documents
                    .retain(|r| matches_section_filter(&r.payload, filter));
                let after = evidence.documents.len();
                tracing::debug!(
                    doc_id = ?filter.doc_id,
                    section_id = ?filter.section_id,
                    before,
                    after,
                    "router: applied hard section/doc predicate"
                );
            }
        }

        // ---- Step 5c: Defence-in-depth user_id post-fusion predicate ----
        //
        // Branch-level filtering (search_for_user) is the primary isolation
        // boundary. This secondary check drops any result whose payload
        // user_id does not match the requesting user — guards against
        // graph rehydration paths that fetch chunks by ID without a
        // Qdrant-level scope filter (e.g. hydrate_chunks_from_triplets).
        {
            let expected = user_id.as_str();
            let before = evidence.documents.len();
            evidence.documents.retain(|r| {
                r.payload
                    .get("user_id")
                    .and_then(|v| v.as_str())
                    // `is_none_or(...)` — retain chunks whose payload lacks a
                    // user_id field (e.g. legacy data or test fixtures) so this
                    // guard does not hard-break existing integration tests that
                    // omit the field. In production all ingested chunks carry it.
                    .is_none_or(|uid| uid == expected)
            });
            let after = evidence.documents.len();
            if before != after {
                tracing::debug!(
                    user_id = expected,
                    before,
                    after,
                    "router: defence-in-depth user_id predicate dropped cross-tenant chunks"
                );
            }
        }

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
                Ok(mut reranked) => {
                    // Weighted-keyword rerank adjustment (consumer #2).
                    // Applied on the post-rerank list; empty keywords or
                    // disabled integration → byte-identical no-op.
                    if let Some(rerank_cfg) = self.weighted_keyword_integration.rerank.as_ref() {
                        if !weighted_keywords.terms.is_empty() {
                            let before_top = reranked.first().map(|d| d.score.value());
                            for doc in reranked.iter_mut() {
                                let text = doc
                                    .payload
                                    .get("chunk_text")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let adjusted = adjust_for_weighted_keywords(
                                    doc.score.value() as f32,
                                    text,
                                    weighted_keywords.as_ref(),
                                    rerank_cfg,
                                );
                                doc.score = agent_rs_core::Score::new(adjusted as f64);
                            }
                            reranked.sort_by(|a, b| {
                                b.score
                                    .value()
                                    .partial_cmp(&a.score.value())
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let after_top = reranked.first().map(|d| d.score.value());
                            tracing::debug!(
                                terms = weighted_keywords.terms.len(),
                                before_top = ?before_top,
                                after_top = ?after_top,
                                "rerank adjusted by weighted keywords"
                            );
                        }
                    }
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
                    if let Some(m) = &self.metrics {
                        m.record_reranker_fallback(strategy);
                    }
                    evidence.documents.truncate(limit);
                }
            }

            // Recompute source_counts over the post-rerank set. The ranker
            // computed them earlier against the pre-rerank RRF pool (size
            // `RERANK_CANDIDATE_COUNT`), which overstated the count for
            // callers that display the header. Refresh so the number
            // matches `documents.len()`.
            evidence.source_counts =
                crate::ranker::build_source_counts_from_docs(&evidence.documents);
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
    /// All vector and keyword branches are scoped to `user_id` — the
    /// [`VectorStore::search_for_user`] call enforces this at the store level.
    /// Graph traversal results are subsequently scoped by the post-fusion
    /// defence-in-depth predicate applied by the caller.
    ///
    /// Returns the raw per-backend result lists to be pooled and fused by the
    /// caller. The `extra_filter` JSON is merged into the user-scoped filter
    /// for Qdrant vector/hybrid calls when provided.
    async fn dispatch_single(
        &self,
        user_id: &UserId,
        query_str: &str,
        strategy: RetrievalStrategy,
        dense: &Embedding,
        limit: usize,
        extra_filter: Option<&serde_json::Value>,
    ) -> Result<Vec<Vec<agent_rs_stores::types::SearchResult>>, RetrievalError> {
        match strategy {
            // ---- Single-backend: dense vector ----
            //
            // User-scoped vector search; metadata payload filters are merged in
            // via `extra_filter` when a MetadataFilterExtractor produced hints.
            RetrievalStrategy::Semantic => {
                let results = self
                    .vector_store
                    .search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    )
                    .await
                    .map_err(RetrievalError::Store)?;
                Ok(vec![results])
            }

            // ---- Single-backend: summaries vector ----
            RetrievalStrategy::Summaries => {
                // Use the dedicated summaries collection when available;
                // fall back to the main collection otherwise.
                let target = self
                    .summaries_collection
                    .as_ref()
                    .unwrap_or(&self.collection);
                let results = self
                    .vector_store
                    .search_for_user(target, user_id, dense, limit, extra_filter.cloned())
                    .await
                    .map_err(RetrievalError::Store)?;
                Ok(vec![results])
            }

            // ---- Single-backend: temporal vector ----
            RetrievalStrategy::Temporal => {
                let results = self
                    .vector_store
                    .search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    )
                    .await
                    .map_err(RetrievalError::Store)?;
                Ok(vec![results])
            }

            // ---- Single-backend: keyword ----
            // Query text is forwarded to Qdrant as a `Document { model: "qdrant/bm25" }`
            // for server-side BM25 sparse inference — no local SPLADE model needed.
            // User scope is enforced via a merged payload filter.
            RetrievalStrategy::Keyword => {
                // Build an extra filter from any metadata hints; user_id scope is
                // applied unconditionally by search_for_user.
                let kw_extra = extra_filter.cloned();
                let results = self
                    .vector_store
                    .search_for_user(&self.collection, user_id, dense, limit, kw_extra)
                    .await
                    .map_err(RetrievalError::Store)?;
                Ok(vec![results])
            }

            // ---- Single-backend: graph (base traversal) ----
            // Seed graph traversal from user-scoped vector search nearest neighbors.
            RetrievalStrategy::Graph => {
                let seeds = self
                    .derive_seed_ids(user_id, query_str, dense, limit)
                    .await?;
                let results = graph_search(
                    &*self.graph_store,
                    &*self.vector_store,
                    &self.collection,
                    user_id,
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
                let seeds = self
                    .derive_seed_ids(user_id, query_str, dense, limit)
                    .await?;
                let results = graph_rag_context_search(
                    &*self.graph_store,
                    &*self.vector_store,
                    &self.collection,
                    user_id,
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
            //      from a user-scoped vector search, and graph_search is invoked
            //      to produce new candidates which are merged and deduplicated by
            //      the retriever.
            //   3. If no retriever is configured, the initial pool is returned
            //      unchanged (equivalent to plain graph search).
            RetrievalStrategy::GraphCoT => {
                let seeds = self
                    .derive_seed_ids(user_id, query_str, dense, limit)
                    .await?;
                let initial = graph_cot_search(
                    &*self.graph_store,
                    &*self.vector_store,
                    &self.collection,
                    user_id,
                    query_str,
                    &seeds,
                    DEFAULT_GRAPH_DEPTH,
                    limit,
                )
                .await?;

                let final_results = if let Some(cot) = self.graph_cot_retriever.as_ref() {
                    // Closure that re-runs a user-scoped graph retrieval for a
                    // follow-up question. Captures `&self` and `user_id` by
                    // reference (the enclosing future outlives the CoT loop).
                    let this = self;
                    let depth = DEFAULT_GRAPH_DEPTH;
                    let cot_limit = limit;
                    let uid = user_id;
                    let retrieve_fn = |follow_up: String| async move {
                        let f_dense = this.embedder.embed_text(&follow_up).await?;
                        let f_seeds = this
                            .derive_seed_ids(uid, &follow_up, &f_dense, cot_limit)
                            .await?;
                        graph_search(
                            &*this.graph_store,
                            &*this.vector_store,
                            &this.collection,
                            uid,
                            &follow_up,
                            &f_seeds,
                            depth,
                            cot_limit,
                        )
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
                // Both vector and keyword branches are user-scoped via search_for_user.
                let (vector_result, keyword_result) = tokio::join!(
                    self.vector_store.search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    ),
                    self.vector_store.search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    ),
                );
                Ok(vec![
                    vector_result.map_err(RetrievalError::Store)?,
                    keyword_result.map_err(RetrievalError::Store)?,
                ])
            }

            // ---- Parallel: vector + keyword + graph (GRAPH_HYBRID) ----
            RetrievalStrategy::GraphHybrid => {
                let seeds = self
                    .derive_seed_ids(user_id, query_str, dense, limit)
                    .await?;
                let (vector_result, keyword_result, graph_result) = tokio::join!(
                    self.vector_store.search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    ),
                    self.vector_store.search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    ),
                    graph_search(
                        &*self.graph_store,
                        &*self.vector_store,
                        &self.collection,
                        user_id,
                        query_str,
                        &seeds,
                        DEFAULT_GRAPH_DEPTH,
                        limit,
                    ),
                );
                Ok(vec![
                    vector_result.map_err(RetrievalError::Store)?,
                    keyword_result.map_err(RetrievalError::Store)?,
                    graph_result?,
                ])
            }

            // ---- Parallel: graph + vector (GRAPH_CONSTRAINED) ----
            RetrievalStrategy::GraphConstrained => {
                let seeds = self
                    .derive_seed_ids(user_id, query_str, dense, limit)
                    .await?;
                let (graph_result, vector_result) = tokio::join!(
                    graph_constrained_search(
                        &*self.graph_store,
                        &*self.vector_store,
                        &self.collection,
                        user_id,
                        query_str,
                        &seeds,
                        DEFAULT_GRAPH_DEPTH,
                        limit,
                    ),
                    self.vector_store.search_for_user(
                        &self.collection,
                        user_id,
                        dense,
                        limit,
                        extra_filter.cloned(),
                    ),
                );
                Ok(vec![
                    graph_result?,
                    vector_result.map_err(RetrievalError::Store)?,
                ])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert a slice of [`FilterCondition`] into the `serde_json::Value` format
/// accepted by [`VectorStore::search_for_user`] as `extra_filter_json`.
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
                // "15" will match "Section 15 - Overview" via token overlap.
                serde_json::json!({
                    "key": field,
                    "match": { "text": text }
                })
            }
        })
        .collect();

    serde_json::json!({ "must": must })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_filter_is_empty_when_both_none() {
        let f = SectionFilter::default();
        assert!(f.is_empty());
    }

    #[test]
    fn section_filter_not_empty_with_doc_id_only() {
        let f = SectionFilter {
            doc_id: Some("d1".into()),
            section_id: None,
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn section_filter_not_empty_with_section_id_only() {
        let f = SectionFilter {
            doc_id: None,
            section_id: Some("s1".into()),
        };
        assert!(!f.is_empty());
    }

    #[test]
    fn matches_filter_passes_when_filter_empty() {
        let payload = serde_json::json!({"chunk_text": "hello"});
        let filter = SectionFilter::default();
        assert!(matches_section_filter(&payload, &filter));
    }

    #[test]
    fn matches_filter_passes_matching_section_id() {
        let payload = serde_json::json!({"section_id": "s1", "doc_id": "d1", "chunk_text": "text"});
        let filter = SectionFilter {
            doc_id: None,
            section_id: Some("s1".into()),
        };
        assert!(matches_section_filter(&payload, &filter));
    }

    #[test]
    fn matches_filter_drops_off_section_chunks() {
        let payload = serde_json::json!({"section_id": "s1", "doc_id": "d1", "chunk_text": "text"});
        let filter = SectionFilter {
            doc_id: None,
            section_id: Some("s2".into()),
        };
        assert!(!matches_section_filter(&payload, &filter));
    }

    #[test]
    fn matches_filter_drops_off_doc_chunks() {
        let payload = serde_json::json!({"section_id": "s1", "doc_id": "d1", "chunk_text": "text"});
        let filter = SectionFilter {
            doc_id: Some("d2".into()),
            section_id: None,
        };
        assert!(!matches_section_filter(&payload, &filter));
    }

    #[test]
    fn matches_filter_requires_both_when_both_set() {
        let payload = serde_json::json!({"section_id": "s1", "doc_id": "d1", "chunk_text": "text"});
        // doc_id matches but section_id doesn't → must fail
        let filter = SectionFilter {
            doc_id: Some("d1".into()),
            section_id: Some("s2".into()),
        };
        assert!(!matches_section_filter(&payload, &filter));
    }

    #[test]
    fn matches_filter_passes_when_both_match() {
        let payload = serde_json::json!({"section_id": "s1", "doc_id": "d1", "chunk_text": "text"});
        let filter = SectionFilter {
            doc_id: Some("d1".into()),
            section_id: Some("s1".into()),
        };
        assert!(matches_section_filter(&payload, &filter));
    }

    #[test]
    fn matches_filter_missing_payload_field_fails_constraint() {
        // payload has no "section_id" key — constraint should fail
        let payload = serde_json::json!({"doc_id": "d1", "chunk_text": "text"});
        let filter = SectionFilter {
            doc_id: None,
            section_id: Some("s1".into()),
        };
        assert!(!matches_section_filter(&payload, &filter));
    }
}
