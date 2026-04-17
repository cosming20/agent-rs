//! gRPC server crate for agent-rs.
//!
//! Exposes two services over HTTP/2:
//! - [`QueryService`] — processes user queries through the agent pipeline.
//! - [`HealthService`] — liveness / readiness health check endpoint.
//!
//! # Startup
//!
//! Call [`run`] with a [`Settings`] reference and a shared [`AgentOrchestrator`]
//! to start the tonic server:
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use agent_rs_core::config::Settings;
//! use agent_rs_agent::orchestrator::AgentOrchestrator;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let settings = Settings::load()?;
//!     let orchestrator = Arc::new(AgentOrchestrator);
//!     agent_rs_server::run(&settings, orchestrator).await?;
//!     Ok(())
//! }
//! ```
//!
//! # Port
//!
//! Default bind address: `[::]:1059` (configurable via `AGENT_RS__SERVER__PORT`).

mod convert;
mod error;
pub mod service;

/// Generated protobuf types and service traits.
pub mod proto {
    /// Package `agentic_rag.v1` — all generated types live here.
    pub mod agentic_rag {
        pub mod v1 {
            tonic::include_proto!("agentic_rag.v1");
        }
    }
}

use std::sync::Arc;
use std::time::Duration;

use tonic::transport::Server;
use tracing::{info, instrument};

use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::config::Settings;

use error::ServerError;
use proto::agentic_rag::v1::{
    health_service_server::HealthServiceServer, query_service_server::QueryServiceServer,
};
use service::{HealthServiceImpl, QueryServiceImpl};

/// Start the gRPC server and block until it terminates.
///
/// Binds to the address from `settings.server` (default `[::]:1059`).
///
/// # Errors
///
/// Returns [`ServerError::InvalidAddress`] if the configured bind address
/// cannot be parsed, or [`ServerError::Transport`] if tonic fails to serve.
#[instrument(skip(settings, orchestrator), fields(bind_addr = %settings.server.bind_addr()))]
pub async fn run(
    settings: &Settings,
    orchestrator: Arc<AgentOrchestrator>,
) -> Result<(), ServerError> {
    let addr: std::net::SocketAddr =
        settings
            .server
            .bind_addr()
            .parse()
            .map_err(|e: std::net::AddrParseError| ServerError::InvalidAddress {
                addr: settings.server.bind_addr(),
                reason: e.to_string(),
            })?;

    let query_svc = QueryServiceServer::new(QueryServiceImpl::new(
        Arc::clone(&orchestrator),
        settings.server.max_limit_per_query,
    ));
    let health_svc = HealthServiceServer::new(HealthServiceImpl);

    let srv_cfg = &settings.server;
    let request_timeout = Duration::from_secs(srv_cfg.request_timeout_s);
    let tcp_keepalive = Duration::from_secs(30);

    info!(
        bind_addr = %addr,
        request_timeout_s = srv_cfg.request_timeout_s,
        max_concurrent_streams = srv_cfg.max_concurrent_streams,
        concurrency_per_connection = srv_cfg.concurrency_per_connection,
        "gRPC server starting",
    );

    Server::builder()
        // Per-request wall-time cap — cancels runaway agent pipelines.
        .timeout(request_timeout)
        // Detect dead peers and reclaim file descriptors.
        .tcp_keepalive(Some(tcp_keepalive))
        // Per-connection in-flight call cap (tower layer).
        .concurrency_limit_per_connection(srv_cfg.concurrency_per_connection as usize)
        // HTTP/2 stream-level cap.
        .max_concurrent_streams(Some(srv_cfg.max_concurrent_streams))
        // TODO: wire TLS when `settings.server.tls_cert_path` / `tls_key_path`
        // config fields are added.  For now the server binds in plaintext.
        .add_service(query_svc)
        .add_service(health_svc)
        .serve(addr)
        .await
        .map_err(|e| ServerError::Transport(e.to_string()))?;

    Ok(())
}
