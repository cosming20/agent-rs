//! Tonic service impl — `agent.v1.Agent`.
//!
//! First milestone: `list_documents` is end-to-end (PgStore read). Every
//! other RPC returns `Status::unimplemented` so the client can type-check
//! the full surface while we iterate. Each RPC gets filled in task by
//! task until the service is feature-parity with the CLI.

use std::pin::Pin;
use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::UserId;
use agent_rs_stores::postgres::PgStore;
use futures::Stream;
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument};

use crate::pb;
use pb::agent_server::Agent;

/// Concrete agent service. Holds shared-store handles; RPC methods take
/// `&self`, so cloning an `Arc<AgentServer>` across tasks is cheap.
pub struct AgentServer {
    pg: Arc<PgStore>,
    // Additional stores (Qdrant, Neo4j, MinIO) and the AgentOrchestrator
    // are added here as their respective RPCs land.
}

impl AgentServer {
    pub fn new(pg: Arc<PgStore>) -> Self {
        Self { pg }
    }
}

// ---- helpers -------------------------------------------------------------

/// Convert a client-supplied user_id string into a validated [`UserId`].
/// All malformed inputs map to `INVALID_ARGUMENT` with a generic message
/// — verbose details stay in tracing so audit logs keep the exact input
/// for investigation while API surface does not leak validator internals.
fn validate_user_id(raw: &str) -> Result<UserId, Status> {
    UserId::try_new(raw).map_err(|e: AgentError| {
        error!(raw = %raw, error = %e, "invalid user_id rejected at gRPC boundary");
        Status::invalid_argument("invalid user_id")
    })
}

/// Map a store error to a tonic status. Keeps user-facing messages
/// generic (never leak internal details) while preserving structured
/// tracing context for operators.
fn map_store_err(e: agent_rs_core::errors::StoreError) -> Status {
    error!(error = %e, "store error in gRPC handler");
    match e {
        agent_rs_core::errors::StoreError::SessionNotFound { .. } => {
            Status::not_found("session not found")
        }
        agent_rs_core::errors::StoreError::DocumentNotOwnedByUser { .. } => {
            Status::permission_denied("document not owned by user")
        }
        _ => Status::internal("store error"),
    }
}

// ---- Streaming type aliases ---------------------------------------------

type AskEventStream =
    Pin<Box<dyn Stream<Item = Result<pb::AskEvent, Status>> + Send + 'static>>;

// ---- Agent trait impl ----------------------------------------------------

#[tonic::async_trait]
impl Agent for AgentServer {
    type AskStream = AskEventStream;

    #[instrument(skip(self, request), fields(user_id))]
    async fn list_documents(
        &self,
        request: Request<pb::UserRef>,
    ) -> Result<Response<pb::DocumentList>, Status> {
        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let rows = self
            .pg
            .list_ingested_documents(&user_id)
            .await
            .map_err(map_store_err)?;

        let documents: Vec<pb::Document> = rows.into_iter().map(row_to_pb_document).collect();
        info!(count = documents.len(), "list_documents ok");

        Ok(Response::new(pb::DocumentList { documents }))
    }

    async fn ask(
        &self,
        _request: Request<pb::AskRequest>,
    ) -> Result<Response<Self::AskStream>, Status> {
        Err(Status::unimplemented("Ask not yet wired"))
    }

    async fn ingest(
        &self,
        _request: Request<tonic::Streaming<pb::IngestChunk>>,
    ) -> Result<Response<pb::IngestReply>, Status> {
        Err(Status::unimplemented("Ingest not yet wired"))
    }

    async fn list_sessions(
        &self,
        _request: Request<pb::UserRef>,
    ) -> Result<Response<pb::SessionList>, Status> {
        Err(Status::unimplemented("ListSessions not yet wired"))
    }

    async fn session_history(
        &self,
        _request: Request<pb::SessionRef>,
    ) -> Result<Response<pb::MessageList>, Status> {
        Err(Status::unimplemented("SessionHistory not yet wired"))
    }

    async fn activate_docs(
        &self,
        _request: Request<pb::ActivateDocsRequest>,
    ) -> Result<Response<pb::ActivateDocsResponse>, Status> {
        Err(Status::unimplemented("ActivateDocs not yet wired"))
    }

    async fn deactivate_docs(
        &self,
        _request: Request<pb::DeactivateDocsRequest>,
    ) -> Result<Response<pb::DeactivateDocsResponse>, Status> {
        Err(Status::unimplemented("DeactivateDocs not yet wired"))
    }

    async fn delete_document(
        &self,
        _request: Request<pb::DocRef>,
    ) -> Result<Response<pb::DeleteDocumentResponse>, Status> {
        Err(Status::unimplemented("DeleteDocument not yet wired"))
    }
}

fn row_to_pb_document(row: agent_rs_stores::postgres::IngestedDocumentRow) -> pb::Document {
    pb::Document {
        document_id: row.document_id.to_string(),
        user_id: row.user_id,
        source: row.source,
        title: row.title.unwrap_or_default(),
        ingest_status: row.ingest_status,
        n_pages: row.n_pages.unwrap_or(0) as u32,
        n_chunks: row.n_chunks.unwrap_or(0) as u32,
        minio_object_key: row.minio_object_key.unwrap_or_default(),
        started_at: row.started_at.and_utc().to_rfc3339(),
        completed_at: row
            .completed_at
            .map(|t| t.and_utc().to_rfc3339())
            .unwrap_or_default(),
    }
}
