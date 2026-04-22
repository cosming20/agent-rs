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

use agent_rs::embedder::build_embedder;
use agent_rs::stores::connect_stores;
use agent_rs_agent_service::indexer::AgentIndexHandler;
use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_queue::run_indexer;
use agent_rs_retrieval::ingestion::ChunkerConfig;
use anyhow::{Context, Result};
use tracing::info;

/// Default number of in-flight jobs per worker process. Tuned for the
/// MVP pipeline (~200 MiB peak per job). Override via
/// `AGENT_RS__INDEXER__PREFETCH` without rebuilding.
const DEFAULT_PREFETCH: u16 = 5;

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("load settings")?;

    let _otel_guard = init_tracing(&settings.otel, "agent-rs-indexer-worker")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let stores = connect_stores(&settings).await.context("connect_stores")?;
    let (embedder, embedder_label) = build_embedder().await.context("build_embedder")?;
    info!(embedder = %embedder_label, "embedder ready");

    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    // Metrics handle kept for parity with the gRPC service and to keep
    // the retrieval-metrics registration path hot for future Qdrant /
    // Neo4j instrumentation of the indexing hot loop.
    let _metrics = Arc::new(RetrievalMetrics::new("agent-rs-indexer-worker"));

    let handler = Arc::new(AgentIndexHandler::new(
        Arc::new(
            agent_rs_stores::minio::MinioStore::new(settings.minio.clone())
                .await
                .context("connect MinIO")?,
        ),
        embedder,
        stores.qdrant.clone(),
        stores.postgres.clone(),
        collection,
        ChunkerConfig::default(),
    ));

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
