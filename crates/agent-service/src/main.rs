//! Entry point for the agent gRPC service.
//!
//! Boots the full pipeline stack (stores, embedder, ingestion pipeline,
//! retrieval router, agent orchestrator, MinIO) and serves the
//! `agent.v1.Agent` tonic service on the configured bind address.
//!
//! The boot sequence is intentionally identical to the composition used by
//! `examples/ingest.rs` and `examples/ask.rs` so that the service and the
//! dev-loop CLIs share a single source of truth for wiring decisions.

use std::net::SocketAddr;
use std::sync::Arc;

use agent_rs::embedder::build_embedder;
use agent_rs::ingest::{IngestFeatures, build_ingestion_pipeline};
use agent_rs::router::{RouterFeatures, build_router};
use agent_rs::stores::connect_stores;
use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_stores::minio::MinioStore;
use anyhow::{Context, Result};
use tonic::transport::Server;
use tracing::info;

use agent_rs_agent_service::pb::agent_server::AgentServer as AgentGrpcServer;
use agent_rs_agent_service::server::{AgentServer, AgentServerDeps};

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("load settings")?;

    let _otel_guard =
        init_tracing(&settings.otel, "agent-rs-agent-service")
            .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    // ---- Connect everything once at boot (same as ask.rs + ingest.rs) -------
    let stores = connect_stores(&settings)
        .await
        .context("connect_stores")?;
    let (embedder, embedder_label) = build_embedder()
        .await
        .context("build_embedder")?;

    let collection =
        CollectionName::from_string(settings.retrieval.dataset.clone());
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));

    let metrics = Arc::new(RetrievalMetrics::new("agent-rs-agent-service"));

    // ---- Ingestion pipeline (identical wiring to ingest.rs) -----------------
    let pipeline = build_ingestion_pipeline(
        &settings,
        &stores,
        Arc::clone(&embedder),
        &collection,
        &triplet_collection,
        &summaries_collection,
        &IngestFeatures::from_env(),
        Some(Arc::clone(&metrics)),
    )
    .await
    .context("build_ingestion_pipeline")?;

    // ---- Retrieval router (identical wiring to ask.rs) ----------------------
    let router = build_router(
        &settings,
        &stores,
        embedder,
        &collection,
        &triplet_collection,
        &summaries_collection,
        &RouterFeatures::from_env(),
        Arc::clone(&metrics),
    )
    .await
    .context("build_router")?;

    // ---- Agent orchestrator -------------------------------------------------
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
    .context("build AgentOrchestrator")?;

    // ---- MinIO --------------------------------------------------------------
    let minio = Arc::new(
        MinioStore::new(settings.minio.clone())
            .await
            .context("connect MinioStore")?,
    );

    // ---- Assemble server ----------------------------------------------------
    let server = AgentServer::new(AgentServerDeps {
        stores,
        pipeline: Arc::new(pipeline),
        orchestrator: Arc::new(orchestrator),
        minio,
        collection,
        triplet_collection,
        summaries_collection,
        embedder_label,
    });

    let port: u16 = std::env::var("AGENT_RS__AGENT_SERVICE__PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1072);
    let addr: SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .context("parse bind address")?;

    info!(%addr, "agent-rs-agent-service listening");

    Server::builder()
        .add_service(AgentGrpcServer::new(server))
        .serve_with_shutdown(addr, shutdown_signal())
        .await
        .context("tonic server")?;

    info!("agent-rs-agent-service shutting down cleanly");
    Ok(())
}

/// Waits for SIGINT or SIGTERM, then returns so tonic can drain
/// in-flight RPCs before the process exits.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = sigint.recv() => info!("SIGINT received"),
        _ = sigterm.recv() => info!("SIGTERM received"),
    }
}
