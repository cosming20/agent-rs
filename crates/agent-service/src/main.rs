//! Entry point for the agent gRPC service.
//!
//! Loads `Settings`, initialises tracing, connects to Postgres, builds
//! an [`agent_rs_agent_service::server::AgentServer`], and serves the
//! `agent.v1.Agent` tonic service on the configured bind address.

use std::net::SocketAddr;
use std::sync::Arc;

use agent_rs_core::config::Settings;
use agent_rs_stores::postgres::PgStore;
use anyhow::{Context, Result};
use tonic::transport::Server;
use tracing::info;

use agent_rs_agent_service::pb::agent_server::AgentServer as AgentGrpcServer;
use agent_rs_agent_service::server::AgentServer;

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("load settings")?;

    // Shared tracing stack (OTel + fmt subscriber). Matches the pattern
    // used by src/main.rs and examples/ask.rs so Jaeger / Prometheus
    // dashboards work out of the box on the same dev compose.
    let _otel_guard =
        agent_rs_observability::tracing::init_tracing(&settings.otel, "agent-rs-agent-service")
            .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let pg = PgStore::new(&settings.postgres.dsn())
        .await
        .context("pg connect")?;
    let server = AgentServer::new(Arc::new(pg));

    // Port 1072 chosen to sit next to docling (1068), MinIO (1069/1070),
    // leaving 1071 free for future expansion. Override via
    // AGENT_RS__AGENT_SERVICE__PORT when needed.
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
/// in-flight RPCs.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = sigint.recv() => info!("SIGINT received"),
        _ = sigterm.recv() => info!("SIGTERM received"),
    }
}
