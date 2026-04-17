//! Retrieval router assembly with all cognitive features.

use std::sync::Arc;
use std::time::Duration;

use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_retrieval::context_expander::{ContextExpander, ContextExpanderConfig};
use agent_rs_retrieval::embedder::Embedder;
use agent_rs_retrieval::entity_linker::{EntityLinker, EntityLinkerConfig};
use agent_rs_retrieval::graph_cot::{GraphCotConfig, GraphCotRetriever};
use agent_rs_retrieval::http::build_http_client;
use agent_rs_retrieval::metadata_filter::{MetadataFilterConfig, MetadataFilterExtractor};
use agent_rs_retrieval::query_rewriter::{QueryRewriter, QueryRewriterConfig};
use agent_rs_retrieval::ranker::ResultRanker;
use agent_rs_retrieval::reranker::CrossEncoderReranker;
use agent_rs_retrieval::router::RetrievalRouter;
use agent_rs_retrieval::weighted_keywords::{
    BM25WeightingConfig, EntityLinkerPrioritizationConfig, RerankAdjustmentConfig,
    WeightedKeywordExtractor, WeightedKeywordExtractorConfig, WeightedKeywordIntegrationConfig,
};
use agent_rs_stores::traits::{GraphStore, VectorStore};
use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::stores::Stores;

// ---------------------------------------------------------------------------
// Feature toggle config
// ---------------------------------------------------------------------------

/// Controls which cognitive retrieval features are enabled.
///
/// `from_env()` reads `EVAL_SKIP_*` env vars (for eval/debug).
/// `production()` enables everything (for the server).
pub struct RouterFeatures {
    pub skip_query_rewriter: bool,
    pub skip_metadata_filter: bool,
    pub skip_context_expander: bool,
    pub skip_entity_linker: bool,
    pub skip_reranking: bool,
    pub skip_graph_cot: bool,
    /// Skip the entire weighted-keyword extractor. With this true the
    /// router sees `None` for the extractor and all three consumer
    /// integrations collapse to byte-identical no-ops. Used to capture
    /// a feature-off baseline.
    pub skip_weighted_keywords: bool,
}

impl RouterFeatures {
    /// All features enabled (production default).
    pub fn production() -> Self {
        Self {
            skip_query_rewriter: false,
            skip_metadata_filter: false,
            skip_context_expander: false,
            skip_entity_linker: false,
            skip_reranking: false,
            skip_graph_cot: false,
            skip_weighted_keywords: false,
        }
    }

    /// Read skip flags from `EVAL_SKIP_*` environment variables.
    pub fn from_env() -> Self {
        let skip = |var: &str| {
            std::env::var(var)
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        };
        Self {
            skip_query_rewriter: skip("EVAL_SKIP_QUERY_REWRITER"),
            skip_metadata_filter: skip("EVAL_SKIP_METADATA_FILTER"),
            skip_context_expander: skip("EVAL_SKIP_CONTEXT_EXPANDER"),
            skip_entity_linker: skip("EVAL_SKIP_ENTITY_LINKER"),
            skip_reranking: skip("EVAL_SKIP_RERANKING"),
            skip_graph_cot: skip("EVAL_SKIP_GRAPH_COT"),
            skip_weighted_keywords: skip("EVAL_SKIP_WEIGHTED_KEYWORDS"),
        }
    }
}

/// Helper: read OpenAI/LLM API key from environment.
fn api_key() -> Option<String> {
    std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .ok()
}

// ---------------------------------------------------------------------------
// Router builder
// ---------------------------------------------------------------------------

/// Build a fully-wired [`RetrievalRouter`] with cognitive retrieval features.
///
/// Features (query rewriting, metadata filtering, context expansion,
/// entity linking, reranking, GraphCoT) are each gated by the
/// corresponding flag in `features` AND by the presence of an API key.
// All arguments are mandatory wiring inputs; a config struct would add
// indirection without clarity gains at these call sites.
#[allow(clippy::too_many_arguments)]
pub async fn build_router(
    settings: &Settings,
    stores: &Stores,
    embedder: Arc<dyn Embedder + Send + Sync>,
    collection: &CollectionName,
    triplet_collection: &CollectionName,
    summaries_collection: &CollectionName,
    features: &RouterFeatures,
    metrics: Arc<RetrievalMetrics>,
) -> Result<Arc<RetrievalRouter>> {
    // Build one shared HTTP client for all LLM-calling components.
    // 60-second request timeout; connection timeout is 10 s (set in build_http_client).
    let http_client = build_http_client(Duration::from_secs(60))
        .context("build shared HTTP client for router")?;

    let ranker = ResultRanker::new(settings.ranker.rrf_k);

    let mut router = RetrievalRouter::new(
        embedder,
        stores.qdrant.clone() as Arc<dyn VectorStore + Send + Sync>,
        stores.neo4j.clone() as Arc<dyn GraphStore + Send + Sync>,
        ranker,
        collection.clone(),
    )
    .with_triplet_collection(triplet_collection.clone())
    .with_summaries_collection(summaries_collection.clone());

    // GraphCoT iterative retriever.
    if !features.skip_graph_cot {
        if let Some(key) = api_key() {
            let model =
                std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-5.4-mini-2026-03-17".to_owned());
            let config = GraphCotConfig::new(key).with_model(model);
            match GraphCotRetriever::new(config, Arc::clone(&http_client)) {
                Ok(cot) => {
                    router = router.with_graph_cot_retriever(Arc::new(cot));
                    info!("GraphCoT retriever enabled");
                }
                Err(e) => {
                    warn!(error = %e, "GraphCoT init failed");
                }
            }
        } else {
            info!("no API key — GraphCoT disabled");
        }
    }

    // Query rewriter (LLM decomposition).
    if !features.skip_query_rewriter {
        if let Some(key) = api_key() {
            let model = std::env::var("QUERY_REWRITER_MODEL")
                .unwrap_or_else(|_| "gpt-5.4-mini-2026-03-17".to_owned());
            let config = QueryRewriterConfig {
                api_key: key,
                model: model.clone(),
                max_sub_queries: 3,
                temperature: 0.1,
            };
            let mut rewriter = QueryRewriter::new(config, Arc::clone(&http_client));
            rewriter = rewriter.with_metrics(Arc::clone(&metrics));
            router = router.with_query_rewriter(Arc::new(rewriter));
            info!(model = %model, "QueryRewriter enabled");
        }
    }

    // Metadata filter extractor.
    if !features.skip_metadata_filter {
        if let Some(key) = api_key() {
            let model =
                std::env::var("METADATA_FILTER_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned());
            let config = MetadataFilterConfig {
                api_key: key,
                model: model.clone(),
                temperature: 0.0,
            };
            match MetadataFilterExtractor::new(config, Arc::clone(&http_client)) {
                Ok(extractor) => {
                    router = router.with_metadata_filter(Arc::new(extractor));
                    info!(model = %model, "MetadataFilterExtractor enabled");
                }
                Err(e) => {
                    warn!(error = %e, "MetadataFilterExtractor init failed");
                }
            }
        }
    }

    // Context expander (FOLLOWS / IN_SECTION edges).
    if !features.skip_context_expander {
        let expander = Arc::new(ContextExpander::new(
            stores.neo4j.clone() as Arc<dyn GraphStore + Send + Sync>,
            ContextExpanderConfig::default(),
        ));
        router = router.with_context_expander(expander);
        info!("ContextExpander enabled");
    }

    // Entity → KG linker.
    if !features.skip_entity_linker {
        if let Some(key) = api_key() {
            let config = EntityLinkerConfig {
                api_key: key,
                model: "gpt-5.4-mini-2026-03-17".to_owned(),
                max_entities: 5,
            };
            let linker = Arc::new(EntityLinker::new(
                stores.neo4j.clone() as Arc<dyn GraphStore + Send + Sync>,
                config,
                Arc::clone(&http_client),
            ));
            router = router.with_entity_linker(linker);
            info!("EntityLinker enabled");
        }
    }

    // Cross-encoder reranker (Jina or ZeroEntropy per config).
    //
    // When the config provides a provider + api_key, use it directly.
    // Otherwise fall back to env-var detection (prefers ZEROENTROPY_API_KEY
    // over JINA_API_KEY so ops can flip providers by swapping a secret
    // without touching config files).
    if !features.skip_reranking && !settings.reranker.disabled {
        let has_config_key = !settings.reranker.api_key.is_empty();
        let reranker_result = if has_config_key {
            Some(CrossEncoderReranker::from_config(&settings.reranker))
        } else {
            CrossEncoderReranker::try_from_env()
        };

        if let Some(result) = reranker_result {
            match result {
                Ok(reranker) => {
                    router = router.with_reranker(Arc::new(reranker));
                    info!(
                        provider = ?settings.reranker.provider,
                        "cross-encoder reranker enabled"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "reranker init failed");
                }
            }
        }
    }

    // LLM-backed weighted-keyword extractor. Runs in parallel with the
    // other pre-dispatch LLM calls; feeds the rerank adjustment (BM25
    // rewrite and entity-linker candidate tiering wiring land in
    // follow-up commits). Failures / branch-local timeouts collapse to
    // empty, keeping the pipeline byte-identical to the extractor-absent
    // path.
    if !features.skip_weighted_keywords {
        if let Some(key) = api_key() {
            let model = std::env::var("WEIGHTED_KEYWORD_MODEL")
                .unwrap_or_else(|_| "gpt-5.4-mini-2026-03-17".to_owned());
            let mut wk_cfg = WeightedKeywordExtractorConfig::new(key);
            wk_cfg.model = model.clone();
            let extractor = Arc::new(
                WeightedKeywordExtractor::new(wk_cfg, Arc::clone(&http_client))
                    .with_metrics(Arc::clone(&metrics)),
            );
            // Default integration: all three consumers enabled. Today
            // only rerank is wired at call-site (see crates/retrieval/
            // src/router.rs); BM25 and entity_linker integrations land
            // as follow-ups. The integration config stays complete so
            // we don't need to touch this block when they land.
            let integration = WeightedKeywordIntegrationConfig {
                bm25: Some(BM25WeightingConfig::default()),
                rerank: Some(RerankAdjustmentConfig::default()),
                entity_linker: Some(EntityLinkerPrioritizationConfig::default()),
            };
            router = router
                .with_weighted_keyword_extractor(extractor)
                .with_weighted_keyword_integration(integration);
            info!(model = %model, "WeightedKeywordExtractor enabled");
        } else {
            info!("no API key — WeightedKeywordExtractor disabled");
        }
    }

    // Wire shared metric instruments for reranker-fallback events.
    router = router.with_metrics(metrics);

    Ok(Arc::new(router))
}
