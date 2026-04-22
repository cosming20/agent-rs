//! Tonic service impl — `agent.v1.Agent`.
//!
//! Four RPCs, per the 2026-04-21 stateless contract:
//!
//! | RPC                 | Shape             | Backend wiring                                                  |
//! |---------------------|-------------------|-----------------------------------------------------------------|
//! | `Ask`               | server-streaming  | Emits `AskIndexingWait` until every `active_document_keys` entry reaches `complete` state, then delegates to [`AgentOrchestrator::process_query`] and emits a single `AskFinal`. |
//! | `EnqueueIndex`      | unary             | Inserts a row into `document_index_state` with `state=pending` (CAS via `ON CONFLICT DO NOTHING`) and publishes a job to the RabbitMQ indexing queue. |
//! | `GetDocumentStatus` | unary             | Reads the `document_index_state` row for `(user_id, minio_object_key)`.                                                                                |
//! | `DeleteDocument`    | unary             | Purges Qdrant points, Neo4j subgraph, and the `document_index_state` row. The MinIO object itself is removed by the web app (it owns the blob lifecycle). |

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_rs::stores::Stores;
use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{CollectionName, MessageRole, RetrievalStrategy, UserId};
use agent_rs_queue::{IndexJob, IndexPublisher};
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::{ChatTurn, FilterSpec};
use futures::Stream;
use tokio::time::sleep;
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument};
use uuid::Uuid;

use crate::pb;
use pb::agent_server::Agent;

/// Polling period while an active document is still indexing.
///
/// Kept at 1 second so a single connected client generates at most 1 poll/s
/// on Postgres per document — a 1000-client burst produces ≤1000 SELECTs/s
/// rather than the 2000/s the 500 ms cadence would. The client's UI spinner
/// still animates; the server just emits coarser `AskIndexingWait` ticks.
const INDEXING_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Safety cap on how long `Ask` will wait for all active documents to finish
/// indexing before returning an error.
///
/// A short cap is deliberate: a long deadline encourages web clients to hold
/// the gRPC stream open while the IndexerWorker chews through a PDF, which
/// would keep a tower concurrency slot occupied. Web polls
/// `GetDocumentStatus` separately — once the poll goes `complete`, it
/// retries `Ask` and we burn zero wait time.
const INDEXING_MAX_WAIT: Duration = Duration::from_secs(30);

// ---- dependency bundle -----------------------------------------------------

/// All wired dependencies for the agent gRPC server.
pub struct AgentServerDeps {
    pub stores: Stores,
    pub orchestrator: Arc<AgentOrchestrator>,
    /// RabbitMQ publisher used by `EnqueueIndex` to hand jobs off to the
    /// `IndexerWorker`.
    pub index_publisher: Arc<IndexPublisher>,
    /// Primary chunks collection (e.g. `agent-rs`).
    pub chunk_collection: CollectionName,
    /// Per-section summaries collection (e.g. `agent-rs_summaries`).
    pub summaries_collection: CollectionName,
    /// Knowledge-graph triplets collection (e.g. `agent-rs_triplets`).
    pub triplet_collection: CollectionName,
}

// ---- server struct ----------------------------------------------------------

/// Concrete agent service.
pub struct AgentServer {
    stores: Stores,
    orchestrator: Arc<AgentOrchestrator>,
    index_publisher: Arc<IndexPublisher>,
    chunk_collection: CollectionName,
    summaries_collection: CollectionName,
    triplet_collection: CollectionName,
}

impl AgentServer {
    /// Construct from a pre-wired dependency bundle.
    pub fn new(deps: AgentServerDeps) -> Self {
        info!(
            chunk_collection = %deps.chunk_collection,
            summaries_collection = %deps.summaries_collection,
            triplet_collection = %deps.triplet_collection,
            "AgentServer initialised"
        );
        Self {
            stores: deps.stores,
            orchestrator: deps.orchestrator,
            index_publisher: deps.index_publisher,
            chunk_collection: deps.chunk_collection,
            summaries_collection: deps.summaries_collection,
            triplet_collection: deps.triplet_collection,
        }
    }
}

// ---- helpers ----------------------------------------------------------------

/// Convert a client-supplied `user_id` string into a validated [`UserId`].
///
/// All malformed inputs map to `INVALID_ARGUMENT` with a generic message;
/// verbose details stay in tracing so audit logs keep the exact input.
// `tonic::Status` is inherently large (176 bytes); boxing it at a private
// helper boundary would add indirection without benefit — callers always
// propagate with `?`.
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
        agent_rs_core::errors::StoreError::DocumentNotOwnedByUser { .. } => {
            Status::permission_denied("document not owned by user")
        }
        _ => Status::internal("store error"),
    }
}

/// Parse a `history.role` string into a [`MessageRole`]. Unknown values map
/// to `User` because chat clients sometimes send non-canonical casings
/// ("USER", "human", ...) and dropping history would silently lose context.
fn parse_role(raw: &str) -> MessageRole {
    match raw.to_ascii_lowercase().as_str() {
        "assistant" => MessageRole::Assistant,
        "system" => MessageRole::System,
        _ => MessageRole::User,
    }
}

/// Parse an optional `strategy` string. Empty or unknown → `GraphHybrid`.
fn parse_strategy(raw: &str) -> RetrievalStrategy {
    if raw.is_empty() {
        return RetrievalStrategy::GraphHybrid;
    }
    RetrievalStrategy::from_db_str(raw).unwrap_or(RetrievalStrategy::GraphHybrid)
}

// ---- streaming type alias ---------------------------------------------------

type AskEventStream =
    Pin<Box<dyn Stream<Item = Result<pb::AskEvent, Status>> + Send + 'static>>;

// ---- Agent trait impl -------------------------------------------------------

#[tonic::async_trait]
impl Agent for AgentServer {
    type AskStream = AskEventStream;

    // ---- Ask ----------------------------------------------------------------

    /// Server-streaming `Ask` RPC.
    ///
    /// 1. Polls `document_index_state` for every `active_document_keys`
    ///    entry. While any entry is `pending`/`indexing`, emits an
    ///    `AskIndexingWait` event every [`INDEXING_POLL_INTERVAL`].
    /// 2. Once all active keys are `complete`, delegates to
    ///    [`AgentOrchestrator::process_query`] with inline history and
    ///    active/history key union.
    /// 3. Emits a single terminal `AskFinal` (or `AskError`).
    ///
    /// # Cancellation
    ///
    /// Two safety nets cap the lifetime of this RPC:
    ///
    /// - The tower `TimeoutLayer` wired in `main.rs` enforces
    ///   `settings.server.request_timeout_s` on the whole call.
    /// - The body is built with `async_stream::try_stream!`; when the
    ///   client drops the tonic stream, the returned `Stream` is
    ///   dropped and every `.await` inside is cancelled at its next
    ///   yield point. The `AgentOrchestrator::process_query` pipeline
    ///   uses no `tokio::spawn`, so dropping the future drops every
    ///   sub-future transitively — no orphan tool calls survive the
    ///   disconnect.
    ///
    /// If a long-running tool (LLM request, Brave fetch) is between yield
    /// points, the in-flight HTTP call completes before cancellation
    /// surfaces. That's an accepted latency on the disconnection path;
    /// `TimeoutLayer` bounds the damage.
    #[instrument(skip(self, request), fields(user_id, strategy, trace_id))]
    async fn ask(
        &self,
        request: Request<pb::AskRequest>,
    ) -> Result<Response<Self::AskStream>, Status> {
        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let strategy = parse_strategy(&req.strategy);
        tracing::Span::current().record("strategy", format!("{strategy:?}").as_str());

        let trace_id = if req.trace_id.is_empty() {
            None
        } else {
            Uuid::parse_str(&req.trace_id).ok()
        };
        if let Some(tid) = trace_id {
            tracing::Span::current().record("trace_id", tid.to_string().as_str());
        }

        let limit = if req.limit == 0 { 8 } else { req.limit as usize };

        let history: Vec<ChatTurn> = req
            .history
            .into_iter()
            .map(|t| ChatTurn {
                role: parse_role(&t.role),
                content: t.content,
            })
            .collect();

        let active_keys = req.active_document_keys;
        let history_keys = req.history_document_keys;

        // Retrieval operates over the union active ∪ history; we still need
        // to know the active set separately so only those wait for indexing.
        let mut retrieval_keys: Vec<String> =
            Vec::with_capacity(active_keys.len() + history_keys.len());
        retrieval_keys.extend(active_keys.iter().cloned());
        for k in &history_keys {
            if !retrieval_keys.contains(k) {
                retrieval_keys.push(k.clone());
            }
        }

        let pg = self.stores.postgres.clone();
        let orchestrator = Arc::clone(&self.orchestrator);
        let query = req.query;

        let stream = async_stream::try_stream! {
            // ---- indexing wait loop --------------------------------------
            let deadline = Instant::now() + INDEXING_MAX_WAIT;
            loop {
                let mut all_ready = true;
                for key in &active_keys {
                    match pg.get_index_state(&user_id, key).await {
                        Ok(Some(row)) => {
                            match row.state.as_str() {
                                "complete" => {}
                                "failed" => {
                                    Err(Status::failed_precondition(format!(
                                        "indexing failed for {key}"
                                    )))?;
                                }
                                other => {
                                    all_ready = false;
                                    yield pb::AskEvent {
                                        payload: Some(pb::ask_event::Payload::IndexingWait(
                                            pb::AskIndexingWait {
                                                minio_object_key: key.clone(),
                                                state: other.to_string(),
                                                message: String::new(),
                                            },
                                        )),
                                    };
                                }
                            }
                        }
                        Ok(None) => {
                            Err(Status::failed_precondition(format!(
                                "document {key} has not been enqueued for indexing"
                            )))?;
                        }
                        Err(e) => {
                            Err(map_store_err(e))?;
                        }
                    }
                }
                if all_ready {
                    break;
                }
                if Instant::now() >= deadline {
                    Err(Status::deadline_exceeded(
                        "active documents did not finish indexing within budget",
                    ))?;
                }
                sleep(INDEXING_POLL_INTERVAL).await;
            }

            // ---- dispatch to orchestrator --------------------------------
            let t = Instant::now();
            let response = orchestrator
                .process_query(
                    query,
                    user_id.clone(),
                    history,
                    retrieval_keys,
                    strategy,
                    limit,
                    trace_id,
                )
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
                    // Synthesizer already assigns 1-based indices — preserve
                    // them so that textual `[n]` markers in `answer` continue
                    // to line up with `citations[n-1]` even if synthesis
                    // adds a filter/dedupe pass later.
                    index: c.index as u32,
                    chunk_id: c.chunk_id.clone(),
                    content_snippet: c.content_snippet.clone(),
                    minio_object_key: c.minio_object_key.clone(),
                    section_path: c.section_path.clone(),
                    score: c.score,
                })
                .collect();

            let coverage = response.coverage_report.as_ref().map(|cr| pb::CoverageReport {
                expected: cr.expected as u32,
                fetched: cr.fetched as u32,
                missing_section_ids: cr.missing_section_ids.clone(),
            });

            yield pb::AskEvent {
                payload: Some(pb::ask_event::Payload::Final(pb::AskFinal {
                    answer: response.answer,
                    citations,
                    confidence: response.confidence.value(),
                    is_grounded: response.is_grounded,
                    elapsed_ms,
                    strategy: strategy.as_db_str().to_owned(),
                    coverage,
                })),
            };
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // ---- EnqueueIndex -------------------------------------------------------

    /// Unary `EnqueueIndex` RPC.
    ///
    /// Idempotently inserts a `document_index_state` row with `state=pending`
    /// (CAS via `ON CONFLICT DO NOTHING`), then publishes an [`IndexJob`] to
    /// the RabbitMQ queue. If the publish fails, the state row is rolled
    /// back so the next `EnqueueIndex` attempt can re-publish.
    ///
    /// ## Rollback race (documented trade-off)
    ///
    /// A pathological timeline exists where client A inserts the row and
    /// starts publishing, client B observes `state=pending` and returns
    /// `already_queued`, then A's publish fails and wipes the row. B's
    /// caller still sees `already_queued` but `GetDocumentStatus` now
    /// answers `not_found`. Web treats `not_found` as the authoritative
    /// signal and simply re-enqueues — the inconsistency converges within
    /// one poll cycle. We accept this over adding per-job-id optimistic
    /// concurrency on the state row.
    #[instrument(skip(self, request), fields(user_id))]
    async fn enqueue_index(
        &self,
        request: Request<pb::EnqueueIndexRequest>,
    ) -> Result<Response<pb::EnqueueIndexResponse>, Status> {
        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let key = req.minio_object_key;
        if key.is_empty() {
            return Err(Status::invalid_argument("minio_object_key is empty"));
        }

        let existing = self
            .stores
            .postgres
            .get_index_state(&user_id, &key)
            .await
            .map_err(map_store_err)?;

        let job_id = Uuid::new_v4().to_string();

        let state = match existing.as_ref().map(|r| r.state.as_str()) {
            Some("complete") => "already_indexed",
            Some("pending") | Some("indexing") => "already_queued",
            _ => {
                let inserted = self
                    .stores
                    .postgres
                    .enqueue_index_state(&user_id, &key)
                    .await
                    .map_err(map_store_err)?;
                if inserted {
                    // Only publish to RabbitMQ when we successfully inserted
                    // a fresh row — redelivery after a crash is the broker's
                    // job, not ours.
                    let job = IndexJob {
                        user_id: user_id.as_str().to_owned(),
                        minio_object_key: key.clone(),
                        job_id: job_id.clone(),
                    };
                    if let Err(e) = self.index_publisher.publish(&job).await {
                        // Roll back the state row so the next EnqueueIndex
                        // retry can re-publish instead of seeing a stale
                        // `pending` with no corresponding queue message.
                        let _ = self
                            .stores
                            .postgres
                            .delete_index_state(&user_id, &key)
                            .await;
                        error!(error = %e, "queue publish failed; rolled back state row");
                        return Err(Status::unavailable("indexing queue unavailable"));
                    }
                    "queued"
                } else {
                    "already_queued"
                }
            }
        };

        Ok(Response::new(pb::EnqueueIndexResponse {
            state: state.to_owned(),
            job_id,
        }))
    }

    // ---- GetDocumentStatus --------------------------------------------------

    #[instrument(skip(self, request), fields(user_id))]
    async fn get_document_status(
        &self,
        request: Request<pb::GetDocumentStatusRequest>,
    ) -> Result<Response<pb::GetDocumentStatusResponse>, Status> {
        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let key = req.minio_object_key;
        if key.is_empty() {
            return Err(Status::invalid_argument("minio_object_key is empty"));
        }

        let row = self
            .stores
            .postgres
            .get_index_state(&user_id, &key)
            .await
            .map_err(map_store_err)?;

        let response = match row {
            Some(r) => pb::GetDocumentStatusResponse {
                state: r.state,
                sha256: r.sha256.unwrap_or_default(),
                n_pages: r.n_pages.unwrap_or(0) as u32,
                n_chunks: r.n_chunks.unwrap_or(0) as u32,
                error_message: r.error_message.unwrap_or_default(),
            },
            None => pb::GetDocumentStatusResponse {
                state: "not_found".into(),
                sha256: String::new(),
                n_pages: 0,
                n_chunks: 0,
                error_message: String::new(),
            },
        };

        Ok(Response::new(response))
    }

    // ---- DeleteDocument -----------------------------------------------------

    /// Purge every trace of a document owned by `user_id`:
    ///
    /// 1. Qdrant: `delete_by_filter` on the chunks + summaries + triplets
    ///    collections, scoped to `(user_id, minio_object_key)`.
    /// 2. Neo4j: detach-delete the `Document` node (matched on `id` +
    ///    `user_id`) and every descendant (`Section`, `Chunk`, `Page`,
    ///    `Table`, `Picture`) reachable through the structural relations.
    /// 3. Postgres: drop the `document_index_state` row so a future
    ///    `EnqueueIndex` for the same key starts fresh.
    ///
    /// The web app separately removes the MinIO object — it wrote the
    /// bytes; it owns the blob lifecycle.
    #[instrument(skip(self, request), fields(user_id))]
    async fn delete_document(
        &self,
        request: Request<pb::DeleteDocumentRequest>,
    ) -> Result<Response<pb::DeleteDocumentResponse>, Status> {
        let req = request.into_inner();
        let user_id = validate_user_id(&req.user_id)?;
        tracing::Span::current().record("user_id", user_id.as_str());

        let key = req.minio_object_key;
        if key.is_empty() {
            return Err(Status::invalid_argument("minio_object_key is empty"));
        }

        // ---- Qdrant: purge all three collections -----------------------
        let build_filter = || {
            FilterSpec::And(vec![
                FilterSpec::eq("user_id", user_id.as_str().to_owned()),
                FilterSpec::eq("minio_object_key", key.clone()),
            ])
        };

        let removed_chunks = self
            .stores
            .qdrant
            .delete_by_filter(&self.chunk_collection, build_filter())
            .await
            .map_err(map_store_err)?;
        let _ = self
            .stores
            .qdrant
            .delete_by_filter(&self.summaries_collection, build_filter())
            .await
            .map_err(map_store_err)?;
        let _ = self
            .stores
            .qdrant
            .delete_by_filter(&self.triplet_collection, build_filter())
            .await
            .map_err(map_store_err)?;

        // ---- Neo4j: detach-delete the Document + descendants -----------
        //
        // The Cypher interpolates only the label constants; the caller
        // input flows through `$minio_key` and `$user_id` parameters.
        let removed_sections = self
            .stores
            .neo4j
            .delete_document_by_minio_key(&user_id, &key)
            .await
            .map_err(map_store_err)?;

        // ---- Postgres: drop the state row ------------------------------
        self.stores
            .postgres
            .delete_index_state(&user_id, &key)
            .await
            .map_err(map_store_err)?;

        info!(
            removed_chunks,
            removed_sections,
            minio_object_key = %key,
            "document deleted"
        );

        Ok(Response::new(pb::DeleteDocumentResponse {
            removed_chunks: removed_chunks as u32,
            removed_sections: removed_sections as u32,
        }))
    }
}
