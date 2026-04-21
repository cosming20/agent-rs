//! Entry point for the stateless agent gRPC service (`agent.v1.Agent`).
//!
//! Boots the retrieval + orchestrator stack and exposes the four RPCs
//! defined in `agent-rs-proto/proto/agent.proto`: `Ask`, `EnqueueIndex`,
//! `GetDocumentStatus`, `DeleteDocument`.
//!
//! Ingestion (PDF → chunks → embed → upsert) is handled out-of-band by the
//! RabbitMQ-driven `IndexerWorker` (crate `agent-rs-queue`); this service
//! only exposes the control-plane RPCs and blocks `Ask` until indexing for
//! the referenced keys completes.

use std::net::SocketAddr;
use std::sync::Arc;

use agent_rs::embedder::build_embedder;
use agent_rs::router::{RouterFeatures, build_router};
use agent_rs::stores::connect_stores;
use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_queue::IndexPublisher;
use anyhow::{Context, Result};
use tonic::transport::Server;
use tracing::info;

use agent_rs_agent_service::pb::agent_server::AgentServer as AgentGrpcServer;
use agent_rs_agent_service::server::{AgentServer, AgentServerDeps};

/// Default bind port (overridable via `AGENT_RS__AGENT_SERVICE__PORT`).
const DEFAULT_BIND_PORT: u16 = 1072;

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("load settings")?;

    let _otel_guard = init_tracing(&settings.otel, "agent-rs-agent-service")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let stores = connect_stores(&settings).await.context("connect_stores")?;
    let (embedder, embedder_label) = build_embedder().await.context("build_embedder")?;
    info!(embedder = %embedder_label, "embedder ready");

    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));

    let metrics = Arc::new(RetrievalMetrics::new("agent-rs-agent-service"));

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

    let amqp_endpoint = std::env::var("AGENT_RS__RABBITMQ__ENDPOINT")
        .unwrap_or_else(|_| "amqp://agentrs:agentrs-local@localhost:1075".to_owned());
    let index_publisher = Arc::new(
        IndexPublisher::connect(&amqp_endpoint)
            .await
            .context("connect to RabbitMQ indexing queue")?,
    );

    let server = AgentServer::new(AgentServerDeps {
        stores,
        orchestrator: Arc::new(orchestrator),
        index_publisher,
    });

    let port: u16 = std::env::var("AGENT_RS__AGENT_SERVICE__PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BIND_PORT);
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
