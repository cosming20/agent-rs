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

    let query_svc = QueryServiceServer::new(QueryServiceImpl::new(Arc::clone(&orchestrator)));
    let health_svc = HealthServiceServer::new(HealthServiceImpl);

    info!(bind_addr = %addr, "gRPC server starting");

    Server::builder()
        .add_service(query_svc)
        .add_service(health_svc)
        .serve(addr)
        .await
        .map_err(|e| ServerError::Transport(e.to_string()))?;

    Ok(())
}
