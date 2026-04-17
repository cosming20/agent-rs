//! Ingestion pipeline assembly.

use std::sync::Arc;
use std::time::Duration;

use agent_rs_core::config::{IngestBackend, Settings};
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_retrieval::embedder::Embedder;
use agent_rs_retrieval::http::build_http_client;
use agent_rs_retrieval::ingestion::docling_client::DoclingSidecarClient;
use agent_rs_retrieval::ingestion::{
    ChunkSummarizer, ChunkerConfig, IngestionPipeline, IngestionPipelineConfig, LlmGraphExtractor,
};
use agent_rs_stores::traits::{GraphStore, VectorStore};
use anyhow::{Context, Result};
use tracing::info;

use crate::stores::Stores;

// ---------------------------------------------------------------------------
// Ingest feature toggles
// ---------------------------------------------------------------------------

/// Controls which ingestion features are enabled.
pub struct IngestFeatures {
    pub skip_kg_extraction: bool,
    pub skip_summaries: bool,
}

impl IngestFeatures {
    /// All features enabled (production default).
    pub fn production() -> Self {
        Self {
            skip_kg_extraction: false,
            skip_summaries: false,
        }
    }

    /// Read from `EVAL_SKIP_*` env vars.
    pub fn from_env() -> Self {
        let skip = |var: &str| {
            std::env::var(var)
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        };
        Self {
            skip_kg_extraction: skip("EVAL_SKIP_KG_EXTRACTION"),
            skip_summaries: skip("EVAL_SKIP_SUMMARIES"),
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
// Pipeline builder
// ---------------------------------------------------------------------------

/// Build a fully-wired [`IngestionPipeline`] with Docling sidecar, KG
/// extraction, summarization, and Postgres state machine.
// All arguments are mandatory wiring inputs; a config struct would add
// indirection without clarity gains at these call sites.
#[allow(clippy::too_many_arguments)]
pub async fn build_ingestion_pipeline(
    settings: &Settings,
    stores: &Stores,
    embedder: Arc<dyn Embedder + Send + Sync>,
    collection: &CollectionName,
    triplet_collection: &CollectionName,
    summaries_collection: &CollectionName,
    features: &IngestFeatures,
    metrics: Option<Arc<RetrievalMetrics>>,
) -> Result<IngestionPipeline> {
    // Build one shared HTTP client for all LLM-calling ingestion components.
    // 120-second request timeout to accommodate slow graph extraction on large chunks.
    let http_client = build_http_client(Duration::from_secs(120))
        .context("build shared HTTP client for ingestion pipeline")?;

    let mut config = IngestionPipelineConfig::new(ChunkerConfig::default());

    // LLM KG extraction.
    if !features.skip_kg_extraction {
        if let Some(key) = api_key() {
            let model = std::env::var("LLM_GRAPH_MODEL")
                .or_else(|_| std::env::var("LLM_MODEL"))
                .unwrap_or_else(|_| "gpt-5.4-mini-2026-03-17".to_owned());
            info!(model = %model, "LLM graph extractor enabled");
            let mut extractor = LlmGraphExtractor::new(key, model, Arc::clone(&http_client));
            if let Some(m) = metrics.clone() {
                extractor = extractor.with_metrics(m);
            }
            config = config.with_llm_extractor(Arc::new(extractor));
        } else {
            info!("no API key — heuristic entity extraction");
        }
    } else {
        info!("KG extraction skipped");
    }

    // Chunk summarization.
    if !features.skip_summaries {
        if let Some(key) = api_key() {
            let model = std::env::var("LLM_SUMMARY_MODEL")
                .or_else(|_| std::env::var("LLM_MODEL"))
                .unwrap_or_else(|_| "gpt-5.4-mini-2026-03-17".to_owned());
            info!(model = %model, "ChunkSummarizer enabled");
            let mut summarizer = ChunkSummarizer::new(key, model, Arc::clone(&http_client));
            if let Some(m) = metrics.clone() {
                summarizer = summarizer.with_metrics(m);
            }
            config = config
                .with_summarizer(Arc::new(summarizer))
                .with_summaries_collection(summaries_collection.clone());
        }
    } else {
        info!("chunk summarization skipped");
    }

    // Triplet indexing.
    config = config.with_triplet_collection(triplet_collection.clone());

    // Docling sidecar + ingest backend routing.
    config = config.with_ingest_backend(settings.ingest_backend);
    if matches!(
        settings.ingest_backend,
        IngestBackend::Docling | IngestBackend::DoclingWithLegacyFallback
    ) {
        info!(endpoint = %settings.docling.endpoint, "connecting to Docling sidecar");
        let sidecar = DoclingSidecarClient::connect(
            settings.docling.endpoint.clone(),
            Duration::from_secs(settings.docling.connect_timeout_s),
            Duration::from_secs(settings.docling.stream_idle_timeout_s),
        )
        .await
        .context("connect to Docling sidecar")?;
        config = config.with_docling_client(sidecar);
        info!("Docling sidecar ready");

        // PgStore for ingest state machine.
        config = config.with_pg_store(stores.postgres.clone());
        info!("PgStore wired into ingestion pipeline");
    }

    let mut pipeline = IngestionPipeline::with_config(
        embedder,
        stores.qdrant.clone() as Arc<dyn VectorStore + Send + Sync>,
        stores.neo4j.clone() as Arc<dyn GraphStore + Send + Sync>,
        collection.clone(),
        config,
    );
    if let Some(m) = metrics {
        pipeline = pipeline.with_metrics(m);
    }
    Ok(pipeline)
}
