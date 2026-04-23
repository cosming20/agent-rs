//! Indexer worker binary — long-running RabbitMQ consumer that runs the
//! document indexing pipeline.
//!
//! Spin one instance per replica in docker-compose / Kubernetes — the
//! broker fans out jobs, `with_prefetch(N)` caps in-flight work per
//! replica, and `AGENT_RS__INDEXER__PREFETCH` lets operators tune without
//! redeploying.
//!
//! All wiring is identical to `agent-service::main` up to the
//! orchestrator: same stores, same embedder, same metrics. We deliberately
//! skip the tonic server + the retrieval router (the worker never answers
//! gRPC and never runs agent tools) to keep process memory low.

use std::sync::Arc;
use std::time::Duration;

use agent_rs::bootstrap::ensure_collections;
use agent_rs::embedder::build_embedder;
use agent_rs::stores::connect_stores;
use agent_rs_agent_service::indexer::{AgentIndexHandler, AgentIndexHandlerDeps};
use agent_rs_core::config::{IngestBackend, Settings};
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_queue::run_indexer;
use agent_rs_retrieval::ingestion::docling_client::DoclingSidecarClient;
use agent_rs_retrieval::ingestion::summarizer::ChunkSummarizer;
use agent_rs_retrieval::ingestion::{ChunkerConfig, LlmGraphExtractor};
use anyhow::{Context, Result};
use tracing::{info, warn};

/// Default number of in-flight jobs per worker process. Tuned for the
/// MVP pipeline (~200 MiB peak per job). Override via
/// `AGENT_RS__INDEXER__PREFETCH` without rebuilding.
const DEFAULT_PREFETCH: u16 = 5;

/// Parse a boolean-shaped env var. Accepts `1`, `true` (case-insensitive);
/// everything else, including unset, is `false`.
fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("load settings")?;

    let _otel_guard = init_tracing(&settings.otel, "agent-rs-indexer-worker")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let stores = connect_stores(&settings).await.context("connect_stores")?;
    let (embedder, embedder_label) = build_embedder().await.context("build_embedder")?;
    info!(embedder = %embedder_label, "embedder ready");

    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));

    // Idempotent bootstrap — create collections on cold clusters, no-op
    // on warm ones. Without this the first upsert after a fresh
    // `docker compose up` would fail with "collection not found".
    ensure_collections(
        &stores.qdrant,
        embedder.dimensions(),
        &collection,
        &summaries_collection,
        &triplet_collection,
    )
    .await
    .context("ensure Qdrant collections")?;
    // Metrics handle kept for parity with the gRPC service and to keep
    // the retrieval-metrics registration path hot for future Qdrant /
    // Neo4j instrumentation of the indexing hot loop.
    let _metrics = Arc::new(RetrievalMetrics::new("agent-rs-indexer-worker"));

    // Docling sidecar — wire only when the configured ingest backend
    // wants it. With `DoclingWithLegacyFallback` a connect failure is
    // non-fatal: we log it and let the pdf-extract path carry the load
    // until the sidecar comes back.
    let docling: Option<Arc<DoclingSidecarClient>> = match settings.ingest_backend {
        IngestBackend::Docling | IngestBackend::DoclingWithLegacyFallback => {
            match DoclingSidecarClient::connect(
                settings.docling.endpoint.clone(),
                Duration::from_secs(settings.docling.connect_timeout_s),
                Duration::from_secs(settings.docling.stream_idle_timeout_s),
            )
            .await
            {
                Ok(client) => {
                    info!(
                        endpoint = %settings.docling.endpoint,
                        "Docling sidecar client ready"
                    );
                    Some(Arc::new(client))
                }
                Err(e) if matches!(
                    settings.ingest_backend,
                    IngestBackend::DoclingWithLegacyFallback
                ) =>
                {
                    warn!(error = %e, "Docling sidecar unreachable; falling back to pdf-extract");
                    None
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Docling backend required but sidecar unavailable: {e}"
                    ));
                }
            }
        }
        IngestBackend::Legacy => None,
    };

    // Optional LLM stages (per-chunk summaries + KG triplets).
    //
    // Controlled by:
    //   - OPENAI_API_KEY / LLM_API_KEY env — turned into a shared key.
    //   - AGENT_RS__INDEXER__ENABLE_SUMMARIES (default false)
    //   - AGENT_RS__INDEXER__ENABLE_KG        (default false)
    //
    // Both default off so a fresh deployment does not accidentally spend
    // LLM budget on every ingested document. Flip the flags on when the
    // deployment has observability + rate-limit protection in place.
    let llm_api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .ok();
    let llm_model =
        std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-5.4-mini-2026-03-17".into());
    let http_client = Arc::new(reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("build shared HTTP client")?);

    let summarizer = if env_flag("AGENT_RS__INDEXER__ENABLE_SUMMARIES") {
        match llm_api_key.as_deref() {
            Some(key) => {
                info!(model = %llm_model, "summarizer enabled");
                Some(Arc::new(ChunkSummarizer::new(
                    key.to_owned(),
                    llm_model.clone(),
                    Arc::clone(&http_client),
                )))
            }
            None => {
                warn!("AGENT_RS__INDEXER__ENABLE_SUMMARIES=true but no OPENAI_API_KEY set; disabling");
                None
            }
        }
    } else {
        None
    };

    let extractor = if env_flag("AGENT_RS__INDEXER__ENABLE_KG") {
        match llm_api_key.as_deref() {
            Some(key) => {
                info!(model = %llm_model, "KG extractor enabled");
                Some(Arc::new(LlmGraphExtractor::new(
                    key.to_owned(),
                    llm_model.clone(),
                    Arc::clone(&http_client),
                )))
            }
            None => {
                warn!("AGENT_RS__INDEXER__ENABLE_KG=true but no OPENAI_API_KEY set; disabling");
                None
            }
        }
    } else {
        None
    };

    let handler = Arc::new(AgentIndexHandler::new(AgentIndexHandlerDeps {
        minio: Arc::new(
            agent_rs_stores::minio::MinioStore::new(settings.minio.clone())
                .await
                .context("connect MinIO")?,
        ),
        embedder,
        qdrant: stores.qdrant.clone(),
        neo4j: stores.neo4j.clone(),
        pg: stores.postgres.clone(),
        chunk_collection: collection,
        summaries_collection,
        triplet_collection,
        chunker_config: ChunkerConfig::default(),
        docling,
        ingest_backend: settings.ingest_backend,
        summarizer,
        extractor,
    }));

    let prefetch: u16 = std::env::var("AGENT_RS__INDEXER__PREFETCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PREFETCH);

    info!(
        endpoint = %settings.rabbitmq.endpoint,
        prefetch,
        "indexer worker starting"
    );

    // `run_indexer` blocks until the consumer loop ends (broker drop or
    // an unrecoverable error); any exit here means the supervisor
    // (docker-compose / k8s) should restart us.
    run_indexer(&settings.rabbitmq.endpoint, prefetch, handler)
        .await
        .context("run_indexer")?;

    Ok(())
}
