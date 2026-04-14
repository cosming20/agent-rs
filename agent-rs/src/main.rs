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

    // 3. Build retrieval router with all cognitive features enabled.
    let collection = CollectionName(settings.retrieval.dataset.clone());
    let triplet_collection = CollectionName(format!("{}_triplets", collection.0));
    let summaries_collection = CollectionName(format!("{}_summaries", collection.0));

    let router = agent_rs::router::build_router(
        &settings,
        &stores,
        embedder,
        &collection,
        &triplet_collection,
        &summaries_collection,
        &RouterFeatures::production(),
    )
    .await?;

    info!("retrieval router ready");

    // 4. Create agent orchestrator (wires pipeline steps).
    let orchestrator =
        AgentOrchestrator::new(&settings, stores.postgres.clone(), Arc::clone(&router))
            .context("failed to create agent orchestrator")?;
    let orchestrator = Arc::new(orchestrator);

    info!("orchestrator ready — starting gRPC server");

    // 5. Start gRPC server (blocks until shutdown).
    agent_rs_server::run(&settings, orchestrator)
        .await
        .context("gRPC server failed")?;

    Ok(())
}
