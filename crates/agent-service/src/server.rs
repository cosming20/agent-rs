//! Tonic service impl — `agent.v1.Agent`.
//!
//! Wires `Ingest` and `Ask` RPCs directly to the same pipeline functions
//! used by `examples/ingest.rs` and `examples/ask.rs`:
//!
//! - `Ingest` → `MinioStore::put_object` + `IngestionPipeline::ingest_pdf` /
//!   `ingest_text` + `PgStore::set_minio_object_key`
//! - `Ask` → `AgentOrchestrator::process_query` wrapped in a server-streaming
//!   response (session_bound event first, final_answer event last).

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use agent_rs::stores::Stores;
use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{CollectionName, UserId};
use agent_rs_retrieval::ingestion::IngestionPipeline;
use agent_rs_stores::minio::MinioStore;
use futures::Stream;
use sha2::{Digest, Sha256};
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument};
use uuid::Uuid;

use crate::pb;
use pb::agent_server::Agent;

// ---- dependency bundle -----------------------------------------------------

/// All wired dependencies for the agent gRPC server.
///
/// Constructed once at startup via [`AgentServer::new`]; cheaply cloneable
/// across tasks because every field is already an `Arc`.
pub struct AgentServerDeps {
    pub stores: Stores,
    pub pipeline: Arc<IngestionPipeline>,
    pub orchestrator: Arc<AgentOrchestrator>,
    pub minio: Arc<MinioStore>,
    pub collection: CollectionName,
    pub triplet_collection: CollectionName,
    pub summaries_collection: CollectionName,
    /// Human-readable label for the active embedder model (logged at startup).
    pub embedder_label: String,
}

// ---- server struct ----------------------------------------------------------

/// Concrete agent service.
///
/// All fields are `Arc`-wrapped so `&self` handlers can spawn tasks without
/// lifetime coupling.
pub struct AgentServer {
    stores: Stores,
    pipeline: Arc<IngestionPipeline>,
    orchestrator: Arc<AgentOrchestrator>,
    minio: Arc<MinioStore>,
    collection: CollectionName,
    triplet_collection: CollectionName,
    summaries_collection: CollectionName,
    embedder_label: String,
}

impl AgentServer {
    /// Construct from a pre-wired dependency bundle.
    pub fn new(deps: AgentServerDeps) -> Self {
        let server = Self {
            stores: deps.stores,
            pipeline: deps.pipeline,
            orchestrator: deps.orchestrator,
            minio: deps.minio,
            collection: deps.collection,
            triplet_collection: deps.triplet_collection,
            summaries_collection: deps.summaries_collection,
            embedder_label: deps.embedder_label,
        };
        info!(
            collection = %server.collection,
            triplet_collection = %server.triplet_collection,
            summaries_collection = %server.summaries_collection,
            embedder = %server.embedder_label,
            "AgentServer initialised"
        );
        server
    }
}

// ---- helpers ----------------------------------------------------------------

/// Convert a client-supplied `user_id` string into a validated [`UserId`].
///
/// All malformed inputs map to `INVALID_ARGUMENT` with a generic message;
/// verbose details stay in tracing so audit logs keep the exact input
/// for investigation while the API surface does not leak validator internals.
// `tonic::Status` is inherently large (176 bytes); boxing it at this private
// helper boundary would add indirection without benefit — the caller always
// uses `?` and immediately propagates. The lint is suppressed locally.
#[allow(clippy::result_large_err)]
fn validate_user_id(raw: &str) -> Result<UserId, Status> {
    UserId::try_new(raw).map_err(|e: AgentError| {
        error!(raw = %raw, error = %e, "invalid user_id rejected at gRPC boundary");
        Status::invalid_argument("invalid user_id")
    })
}

/// Map a [`agent_rs_core::errors::StoreError`] to a tonic [`Status`].
///
/// Keeps user-facing messages generic while preserving structured tracing
/// context for operators.
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

// ---- streaming type alias ---------------------------------------------------

type AskEventStream =
    Pin<Box<dyn Stream<Item = Result<pb::AskEvent, Status>> + Send + 'static>>;

// ---- Agent trait impl -------------------------------------------------------

#[tonic::async_trait]
impl Agent for AgentServer {
    type AskStream = AskEventStream;

    // ---- list_documents -----------------------------------------------------

    #[instrument(skip(self, request), fields(user_id))]
    async fn list_documents(
        &self,
        request: Request<pb::UserRef>,
    ) -> Result<Response<pb::DocumentList>, Status> {
        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let rows = self
            .stores
            .postgres
            .list_ingested_documents(&user_id)
            .await
            .map_err(map_store_err)?;

        let documents: Vec<pb::Document> = rows.into_iter().map(row_to_pb_document).collect();
        info!(count = documents.len(), "list_documents ok");

        Ok(Response::new(pb::DocumentList { documents }))
    }

    // ---- ask ----------------------------------------------------------------

    /// Server-streaming Ask RPC.
    ///
    /// MVP behaviour: emits a `session_bound` event first (so the client can
    /// bind UI state before the answer lands), then runs
    /// [`AgentOrchestrator::process_query`] inline and emits a single
    /// `final_answer` event. Progress (`tool_call` / `partial_answer`) events
    /// are a follow-up PR.
    #[instrument(skip(self, request), fields(user_id, session_id, strategy))]
    async fn ask(
        &self,
        request: Request<pb::AskRequest>,
    ) -> Result<Response<Self::AskStream>, Status> {
        use agent_rs_core::types::{RetrievalStrategy, SessionId};

        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let session_id = if req.session_id.is_empty() {
            SessionId::from_uuid(Uuid::new_v4())
        } else {
            let uuid = Uuid::parse_str(&req.session_id)
                .map_err(|_| Status::invalid_argument("session_id is not a valid UUID"))?;
            SessionId::from_uuid(uuid)
        };
        tracing::Span::current().record("session_id", session_id.inner().to_string().as_str());

        let strategy = if req.strategy.is_empty() {
            RetrievalStrategy::GraphHybrid
        } else {
            RetrievalStrategy::from_db_str(&req.strategy)
                .unwrap_or(RetrievalStrategy::GraphHybrid)
        };
        tracing::Span::current().record("strategy", format!("{strategy:?}").as_str());

        let limit = if req.limit == 0 {
            8
        } else {
            req.limit as usize
        };
        let question = req.question;

        // Echo the session_id back immediately so the client can bind UI state
        // before the orchestrator run completes.
        let session_bound_event = pb::AskEvent {
            payload: Some(pb::ask_event::Payload::SessionBound(pb::AskSessionBound {
                session_id: session_id.inner().to_string(),
            })),
        };

        // Clone Arc handles to move into the response stream (no lifetime coupling).
        let orchestrator = Arc::clone(&self.orchestrator);

        let t = Instant::now();
        let response = orchestrator
            .process_query(question, session_id, user_id, strategy, limit)
            .await
            .map_err(|e| {
                error!(error = %e, "process_query failed");
                Status::internal(format!("process_query: {e}"))
            })?;
        let elapsed_ms = t.elapsed().as_millis() as u32;

        info!(
            is_grounded = response.is_grounded,
            confidence = response.confidence.value(),
            citations = response.citations.len(),
            elapsed_ms,
            "ask ok"
        );

        let citations: Vec<pb::Citation> = response
            .citations
            .iter()
            .map(|c| pb::Citation {
                index: c.index as u32,
                chunk_id: c.chunk_id.clone(),
                content_snippet: c.content_snippet.clone(),
                source: c.source.clone(),
                score: c.score,
            })
            .collect();

        let coverage = response.coverage_report.as_ref().map(|cr| pb::CoverageReport {
            expected: cr.expected as u32,
            fetched: cr.fetched as u32,
            missing_section_ids: cr.missing_section_ids.clone(),
        });

        let final_event = pb::AskEvent {
            payload: Some(pb::ask_event::Payload::FinalAnswer(pb::AskFinal {
                answer: response.answer,
                citations,
                confidence: response.confidence.value(),
                is_grounded: response.is_grounded,
                elapsed_ms,
                strategy: format!("{:?}", response.strategy),
                coverage,
            })),
        };

        let stream =
            futures::stream::iter(vec![Ok(session_bound_event), Ok(final_event)]);
        Ok(Response::new(Box::pin(stream)))
    }

    // ---- ingest -------------------------------------------------------------

    /// Client-streaming Ingest RPC.
    ///
    /// Protocol:
    /// 1. First [`pb::IngestChunk`] **must** be a `header` variant carrying
    ///    `user_id`, `source`, and `content_type`.
    /// 2. Subsequent chunks carry raw bytes (`data` variant).
    ///
    /// Ordering mirrors `examples/ingest.rs`:
    /// 1. MinIO upload (before the pipeline, so the object survives pipeline failure).
    /// 2. `ingest_pdf` / `ingest_text` depending on content-type.
    /// 3. `set_minio_object_key` in Postgres.
    #[instrument(skip(self, request), fields(user_id))]
    async fn ingest(
        &self,
        request: Request<tonic::Streaming<pb::IngestChunk>>,
    ) -> Result<Response<pb::IngestReply>, Status> {
        use pb::ingest_chunk::Payload;

        let mut stream = request.into_inner();

        // First message MUST be an IngestHeader.
        let first = stream
            .message()
            .await
            .map_err(|e| Status::internal(format!("recv first chunk: {e}")))?
            .ok_or_else(|| Status::invalid_argument("empty ingest stream"))?;

        let header = match first.payload {
            Some(Payload::Header(h)) => h,
            _ => return Err(Status::invalid_argument("first ingest chunk must be header")),
        };
        let user_id = validate_user_id(&header.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        // Accumulate bytes from subsequent messages.
        // 200 MiB ceiling prevents OOM via hostile clients.
        const MAX_BYTES: usize = 200 * 1024 * 1024;
        let mut bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = stream
            .message()
            .await
            .map_err(|e| Status::internal(format!("recv chunk: {e}")))?
        {
            match chunk.payload {
                Some(Payload::Data(data)) => {
                    if bytes.len() + data.len() > MAX_BYTES {
                        return Err(Status::resource_exhausted(
                            "upload exceeds 200 MiB limit",
                        ));
                    }
                    bytes.extend_from_slice(&data);
                }
                Some(Payload::Header(_)) => {
                    return Err(Status::invalid_argument(
                        "unexpected header mid-stream",
                    ));
                }
                None => {
                    return Err(Status::invalid_argument("chunk payload missing"));
                }
            }
        }

        if bytes.is_empty() {
            return Err(Status::invalid_argument("upload has zero bytes"));
        }

        let t_total = Instant::now();

        // SHA-256 → deterministic document_id (same derivation as ingest.rs).
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let sha256 = format!("{:x}", hasher.finalize());
        let document_id = agent_rs_retrieval::ingestion::derive_document_id(&user_id, &sha256);

        info!(
            user_id = %user_id,
            document_id = %document_id,
            sha256 = %sha256,
            "derived document_id from sha256"
        );

        // MinIO upload BEFORE the pipeline — object survives if Docling fails.
        let content_type = header.content_type.clone();
        let minio_key = self
            .minio
            .put_object(&user_id, document_id, &bytes, &content_type)
            .await
            .map_err(map_store_err)?;

        info!(
            user_id = %user_id,
            document_id = %document_id,
            object_key = %minio_key,
            "uploaded to MinIO"
        );

        // Branch on content-type, same as ingest.rs.
        let source = if header.source.is_empty() {
            format!("upload-{}", &sha256[..8])
        } else {
            header.source.clone()
        };

        let doc_id_str = document_id.to_string();
        let is_text_like = matches!(
            content_type.as_str(),
            "text/plain" | "text/markdown" | "text/html"
        );

        let result = if is_text_like {
            let text = std::str::from_utf8(&bytes).map_err(|e| {
                Status::invalid_argument(format!("text upload is not valid UTF-8: {e}"))
            })?;
            self.pipeline
                .ingest_text(text, &doc_id_str, &source, user_id.as_str())
                .await
                .map_err(|e| {
                    error!(error = %e, "ingest_text failed");
                    Status::internal(format!("ingest_text: {e}"))
                })?
        } else {
            self.pipeline
                .ingest_pdf(&bytes, &doc_id_str, &source, user_id.as_str())
                .await
                .map_err(|e| {
                    error!(error = %e, "ingest_pdf failed");
                    Status::internal(format!("ingest_pdf: {e}"))
                })?
        };

        // Persist the MinIO object key (same as ingest.rs step 3).
        self.stores
            .postgres
            .set_minio_object_key(&user_id, document_id, &minio_key)
            .await
            .map_err(map_store_err)?;

        let latency_ms = t_total.elapsed().as_millis() as u32;

        info!(
            user_id = %user_id,
            document_id = %doc_id_str,
            chunks = result.chunks_created,
            latency_ms,
            "ingest ok"
        );

        Ok(Response::new(pb::IngestReply {
            document_id: doc_id_str,
            sha256,
            ingest_status: "complete".into(),
            minio_object_key: minio_key,
            n_pages: 0, // pipeline result doesn't carry page count today
            n_chunks: result.chunks_created as u32,
            latency_ms,
        }))
    }

    // ---- stubs (subsequent PRs) ---------------------------------------------

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

// ---- row mapping ------------------------------------------------------------

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
