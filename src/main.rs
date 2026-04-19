//! agent-rs — production gRPC server for agentic RAG.
//!
//! Thin entry point: loads config, connects stores, builds the retrieval
//! router with all cognitive features, creates the agent orchestrator,
//! and starts the gRPC server.
//!
//! All wiring logic lives in `agent_rs::{stores, embedder, router, ingest}`.

use std::sync::Arc;

use agent_rs::router::RouterFeatures;
use agent_rs::stores::connect_stores;
use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use anyhow::{Context, Result};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("failed to load configuration")?;
    let _guard =
        init_tracing(&settings.otel, "agent-rs").context("failed to initialise tracing")?;

    info!("agent-rs starting");

    // 1. Connect to all storage backends.
    let stores = connect_stores(&settings).await?;

    // 2. Build embedder (OpenAI or local).
    let (embedder, label) = agent_rs::embedder::build_embedder().await?;
    info!(embedder = %label, "embedder ready");

    // 3. Instantiate shared metric instruments (registered against the global
    //    OTel meter provider set up by init_tracing above).
    let metrics = Arc::new(RetrievalMetrics::new("agent-rs"));

    // 4. Build retrieval router with all cognitive features enabled.
    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));

    let router = agent_rs::router::build_router(
        &settings,
        &stores,
        embedder,
        &collection,
        &triplet_collection,
        &summaries_collection,
        &RouterFeatures::production(),
        Arc::clone(&metrics),
    )
    .await?;

    info!("retrieval router ready");

    // 5. Create agent orchestrator, passing the shared metrics handle.
    let orchestrator = AgentOrchestrator::new(
        &settings,
        stores.neo4j.clone(),
        stores.postgres.clone(),
        stores.qdrant.clone(),
        collection.as_str().to_owned(),
        summaries_collection.as_str().to_owned(),
        Arc::clone(&router),
        Arc::clone(&metrics),
    )
    .context("failed to create agent orchestrator")?;
    let orchestrator = Arc::new(orchestrator);

    info!("orchestrator ready — starting gRPC server");

    // 6. Start gRPC server (blocks until shutdown).
    agent_rs_server::run(&settings, orchestrator)
        .await
        .context("gRPC server failed")?;

    Ok(())
}
