//! gRPC service implementations.
//!
//! [`QueryServiceImpl`] delegates to [`AgentOrchestrator`] for query processing.
//! [`HealthServiceImpl`] answers liveness/readiness probes.
//!
//! Both structs hold `Arc` references to shared state and are `Send + Sync`,
//! satisfying tonic's service constraints.

use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{instrument, warn};
use uuid::Uuid;

use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::types::{SessionId, UserId};

use crate::convert::{proto_strategy_to_domain, synthesized_to_query_response};
use crate::error::agent_error_to_status;
use crate::proto::agentic_rag::v1::health_check_response::ServingStatus;
use crate::proto::agentic_rag::v1::{
    HealthCheckRequest, HealthCheckResponse, QueryRequest, QueryResponse,
    health_service_server::HealthService, query_service_server::QueryService,
};

/// Default number of results when the caller supplies `limit == 0`.
const DEFAULT_LIMIT: usize = 10;

// ---- QueryServiceImpl ----

/// gRPC service that routes queries through the agent pipeline.
pub struct QueryServiceImpl {
    orchestrator: Arc<AgentOrchestrator>,
    /// Hard cap on caller-supplied `limit` values. Sourced from
    /// [`ServerConfig::max_limit_per_query`].
    max_limit: usize,
}

impl QueryServiceImpl {
    /// Construct a new query service wrapping the given orchestrator.
    ///
    /// `max_limit` is the maximum number of evidence results the caller is
    /// allowed to request in a single [`QueryRequest`]. Any higher value is
    /// clamped and a `WARN` log is emitted so operators can detect abuse.
    pub fn new(orchestrator: Arc<AgentOrchestrator>, max_limit: usize) -> Self {
        Self {
            orchestrator,
            max_limit,
        }
    }
}

#[tonic::async_trait]
impl QueryService for QueryServiceImpl {
    /// Process a user query through the agent pipeline and return citations.
    ///
    /// # Errors
    ///
    /// Returns gRPC status errors:
    /// - `INVALID_ARGUMENT` if `session_id` is not a valid UUID.
    /// - `RESOURCE_EXHAUSTED` if the pipeline exceeded max retries.
    /// - `UNAVAILABLE` for transient retrieval failures.
    /// - `INTERNAL` for unexpected server-side failures.
    #[instrument(
        skip(self, request),
        fields(
            session_id = %request.get_ref().session_id,
            user_id = %request.get_ref().user_id,
            strategy = request.get_ref().strategy,
        )
    )]
    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();

        let session_uuid = Uuid::parse_str(&req.session_id).map_err(|e| {
            Status::invalid_argument(format!(
                "session_id '{}' is not a valid UUID: {e}",
                req.session_id
            ))
        })?;

        let session_id = SessionId::from_uuid(session_uuid);
        let user_id = UserId::from_string(req.user_id.clone());
        let strategy = proto_strategy_to_domain(req.strategy);

        let requested_limit = if req.limit > 0 {
            req.limit as usize
        } else {
            DEFAULT_LIMIT
        };
        let limit = if requested_limit > self.max_limit {
            warn!(
                requested = requested_limit,
                clamped = self.max_limit,
                session_id = %req.session_id,
                user_id = %req.user_id,
                "caller limit exceeded cap; clamping to max_limit_per_query",
            );
            self.max_limit
        } else {
            requested_limit
        };

        let response = self
            .orchestrator
            .process_query(req.query.clone(), session_id, user_id, strategy, limit)
            .await
            .map_err(agent_error_to_status)?;

        Ok(Response::new(synthesized_to_query_response(&response)))
    }
}

// ---- HealthServiceImpl ----

/// gRPC health check service.
///
/// Returns `SERVING` unconditionally. A future enhancement can probe
/// downstream dependencies and return `NOT_SERVING` during degraded states.
#[derive(Debug)]
pub struct HealthServiceImpl;

#[tonic::async_trait]
impl HealthService for HealthServiceImpl {
    #[instrument(skip(self, _request))]
    async fn check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            status: ServingStatus::Serving as i32,
        }))
    }
}
