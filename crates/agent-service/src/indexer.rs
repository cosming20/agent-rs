//! Production [`IndexHandler`] implementation for the RabbitMQ
//! `IndexerWorker` consumer.
//!
//! One handler instance is shared across the worker fleet (every replica
//! holds an `Arc<AgentIndexHandler>` and passes it into
//! [`agent_rs_queue::run_indexer`]). Each call to [`Self::handle`] owns a
//! single document from MinIO-read to "state = complete".
//!
//! ## MVP pipeline (no Docling)
//!
//! 1. Validate the `user_id` newtype and claim the state row via
//!    `claim_pending_for_indexing` (CAS `pending → indexing`). If the CAS
//!    fails another worker already took the job — we return `Ok(())` so
//!    the message is acked.
//! 2. Pull bytes from MinIO via [`MinioStore::get_bytes_by_key`]; the
//!    ownership guard on the key prefix prevents cross-user fetches even
//!    with a hand-crafted queue message.
//! 3. Extract plain text: `pdf-extract` for PDFs, UTF-8 for `.txt` / `.md`.
//! 4. Chunk via `agent_rs_retrieval::ingestion::chunker::chunk_document`
//!    (`RecursiveCharacter` default) keyed by MinIO object key.
//! 5. Embed chunks through the shared `Arc<dyn Embedder>`.
//! 6. Upsert to the primary chunks collection in Qdrant, with a payload
//!    carrying `user_id`, `minio_object_key`, `chunk_id`, `chunk_text`,
//!    and `chunk_index`.
//! 7. Mark the state row `complete` with `sha256`, `n_pages`, `n_chunks`.
//!
//! Any failure along the way transitions the state row to `failed` with
//! the error message. The queue message is nacked so lapin redelivers
//! (rust-rabbit's default retry policy decides the backoff).
//!
//! Neo4j structural writes + KG extraction + per-section summaries are
//! intentionally out of scope for this MVP — they require the Docling
//! sidecar + LLM API credentials and will land in a follow-up PR.

use std::sync::Arc;

use agent_rs_core::config::IngestBackend;
use agent_rs_core::types::{CollectionName, UserId};
use agent_rs_queue::{IndexHandler, IndexJob};
use agent_rs_retrieval::embedder::Embedder;
use agent_rs_retrieval::ingestion::docling_client::{DoclingResponse, DoclingSidecarClient};
use agent_rs_retrieval::ingestion::marker_client::MarkerSidecarClient;
use agent_rs_retrieval::ingestion::metadata::{ChunkExtras, ChunkMetadata, extract_metadata};
use agent_rs_retrieval::ingestion::payload::{
    ChunkPayloadInput, SummaryPayloadInput, TripletPayloadInput, build_chunk_payload,
    build_summary_payload, build_triplet_payload,
};
use agent_rs_retrieval::ingestion::summarizer::{ChunkSummarizer, SummarizeItem, SummaryStatus};
use agent_rs_retrieval::ingestion::{ChunkerConfig, LlmGraphExtractor, chunk_document};
use agent_rs_stores::MinimalChunkRow;
use agent_rs_stores::minio::MinioStore;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::VectorPoint;
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tracing::{error, info, instrument, warn};

/// Default in-flight LLM summarisation requests per indexer job.
///
/// Bumped from 4 → 32 to match what the library's tuning comment
/// (`DEFAULT_SUMMARIZER_CONCURRENCY = 20` in
/// [`agent_rs_retrieval::ingestion::summarizer`]) already validated, with
/// extra headroom for bursty throughput. At 32 concurrent calls on
/// `gpt-5.4-mini` a single worker stays well under OpenAI's per-key
/// TPM/RPM ceiling on the standard paid tier; re-tune downward if
/// scaling to N workers per API key causes 429 storms.
const DEFAULT_SUMMARIZER_CONCURRENCY: usize = 32;
/// Default in-flight LLM KG-extraction requests per indexer job.
///
/// Same rationale as [`DEFAULT_SUMMARIZER_CONCURRENCY`]: 4 → 32 to
/// saturate the long per-call LLM latency (5–16 s on gpt-5.x reasoning
/// models) so the KG phase finishes in ~`ceil(n_chunks / 32)` waves
/// rather than `ceil(n_chunks / 4)`.
const DEFAULT_KG_CONCURRENCY: usize = 32;

/// Dependency bundle passed to [`AgentIndexHandler::new`].
///
/// Replaces the prior positional constructor to keep the argument count
/// manageable as optional LLM stages land.
pub struct AgentIndexHandlerDeps {
    pub minio: Arc<MinioStore>,
    pub embedder: Arc<dyn Embedder + Send + Sync>,
    pub qdrant: Arc<QdrantStore>,
    pub neo4j: Arc<Neo4jStore>,
    pub pg: Arc<PgStore>,
    /// Primary chunks collection (same as `server::AgentServer.chunk_collection`).
    pub chunk_collection: CollectionName,
    /// Per-section summaries collection for the optional summarizer stage.
    pub summaries_collection: CollectionName,
    /// Knowledge-graph triplets collection for the optional KG stage.
    pub triplet_collection: CollectionName,
    pub chunker_config: ChunkerConfig,
    /// Optional Docling sidecar. When `Some` and
    /// `ingest_backend` allows Docling, PDF ingestion uses the sidecar's
    /// structural chunking; otherwise the `pdf-extract` fallback runs.
    pub docling: Option<Arc<DoclingSidecarClient>>,
    /// Optional Marker sidecar. When `Some` and `ingest_backend` is
    /// `Marker` / `MarkerWithLegacyFallback`, PDF ingestion uses
    /// Marker's MPS-native pipeline. Ignored otherwise.
    pub marker: Option<Arc<MarkerSidecarClient>>,
    pub ingest_backend: IngestBackend,
    /// Optional per-chunk summariser. Best-effort; failures log + skip.
    pub summarizer: Option<Arc<ChunkSummarizer>>,
    /// Optional LLM knowledge-graph extractor. Best-effort; failures log + skip.
    pub extractor: Option<Arc<LlmGraphExtractor>>,
}

/// Concrete [`IndexHandler`] wiring the agent-rs storage stack.
pub struct AgentIndexHandler {
    minio: Arc<MinioStore>,
    embedder: Arc<dyn Embedder + Send + Sync>,
    qdrant: Arc<QdrantStore>,
    neo4j: Arc<Neo4jStore>,
    pg: Arc<PgStore>,
    chunk_collection: CollectionName,
    summaries_collection: CollectionName,
    triplet_collection: CollectionName,
    chunker_config: ChunkerConfig,
    docling: Option<Arc<DoclingSidecarClient>>,
    marker: Option<Arc<MarkerSidecarClient>>,
    ingest_backend: IngestBackend,
    summarizer: Option<Arc<ChunkSummarizer>>,
    extractor: Option<Arc<LlmGraphExtractor>>,
}

impl AgentIndexHandler {
    /// Build a new handler from a [`AgentIndexHandlerDeps`] bundle.
    pub fn new(deps: AgentIndexHandlerDeps) -> Self {
        Self {
            minio: deps.minio,
            embedder: deps.embedder,
            qdrant: deps.qdrant,
            neo4j: deps.neo4j,
            pg: deps.pg,
            chunk_collection: deps.chunk_collection,
            summaries_collection: deps.summaries_collection,
            triplet_collection: deps.triplet_collection,
            chunker_config: deps.chunker_config,
            docling: deps.docling,
            marker: deps.marker,
            ingest_backend: deps.ingest_backend,
            summarizer: deps.summarizer,
            extractor: deps.extractor,
        }
    }

    /// Run the full indexing pipeline, surfacing the structured error so
    /// the outer `handle` impl can mark the state row `failed` before
    /// propagating the error to the queue consumer loop.
    #[instrument(
        skip(self, job),
        fields(
            user_id = %job.user_id,
            minio_object_key = %job.minio_object_key,
            job_id = %job.job_id,
            bytes = tracing::field::Empty,
            sha256 = tracing::field::Empty,
            n_pages = tracing::field::Empty,
            n_chunks = tracing::field::Empty,
            n_embeddings = tracing::field::Empty,
            n_qdrant_points = tracing::field::Empty,
            n_neo4j_sections = tracing::field::Empty,
        )
    )]
    async fn index(&self, job: &IndexJob) -> Result<IndexSummary, IndexError> {
        let user_id =
            UserId::try_new(&job.user_id).map_err(|e| IndexError::InvalidUser(e.to_string()))?;

        // CAS pending → indexing. If another worker already claimed it
        // (`Ok(false)`), acknowledge and skip.
        let claimed = self
            .pg
            .claim_pending_for_indexing(&job.minio_object_key)
            .await
            .map_err(|e| IndexError::StateMachine(e.to_string()))?;
        if !claimed {
            warn!("job not in pending state; another worker claimed it or it is already done");
            return Ok(IndexSummary::already_claimed());
        }

        // ---- 1. read bytes -----------------------------------------------
        let bytes = self
            .minio
            .get_bytes_by_key(&user_id, &job.minio_object_key)
            .await
            .map_err(|e| IndexError::Fetch(e.to_string()))?;

        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        tracing::Span::current().record("bytes", bytes.len());
        tracing::Span::current().record("sha256", &sha256);
        info!(bytes = bytes.len(), sha256 = %sha256, "document fetched from MinIO");

        // ---- 2. produce structured chunks -------------------------------
        let (chunks, n_pages) = self.chunk_pipeline(&job.minio_object_key, &bytes).await?;
        if chunks.is_empty() {
            return Err(IndexError::Parse(
                "chunker produced zero chunks — document below minimum size".into(),
            ));
        }
        let n_chunks = chunks.len() as i32;
        tracing::Span::current().record("n_pages", n_pages);
        tracing::Span::current().record("n_chunks", n_chunks);

        // ---- 3. embed ----------------------------------------------------
        let texts: Vec<&str> = chunks.iter().map(|c| c.embed_text.as_str()).collect();
        let embeddings = self
            .embedder
            .embed_documents(&texts)
            .await
            .map_err(|e| IndexError::Embed(e.to_string()))?;
        tracing::Span::current().record("n_embeddings", embeddings.len());
        if embeddings.len() != chunks.len() {
            return Err(IndexError::Embed(format!(
                "embedder returned {} vectors for {} chunks",
                embeddings.len(),
                chunks.len()
            )));
        }

        // ---- 4. build VectorPoints --------------------------------------
        //
        // Payload construction is delegated to
        // [`agent_rs_retrieval::ingestion::payload::build_chunk_payload`]
        // so every persisted field name comes from the single
        // `agent_rs_core::payload` schema module — preventing the
        // indexer-vs-retriever key-name drift that produces silent
        // zero-result searches.
        //
        // `doc_id` is the MinIO key in the stateless architecture (the
        // original upload filename lives on the web side); `source` is
        // the ranker-facing identifier, same string in this pipeline.
        let points: Vec<VectorPoint> = chunks
            .iter()
            .zip(embeddings.into_iter())
            .map(|(c, emb)| {
                let payload = build_chunk_payload(
                    ChunkPayloadInput {
                        user_id: user_id.as_str(),
                        doc_id: &job.minio_object_key,
                        minio_object_key: &job.minio_object_key,
                        source: &job.minio_object_key,
                        chunk_id: &c.chunk_id,
                        chunk_text: &c.raw_text,
                        chunk_index: c.chunk_index,
                        total_chunks: c.total_chunks,
                        section_id: c.section_id.as_deref(),
                        section_path: c.section_path.as_deref(),
                        page_start: c.page_start,
                        page_end: c.page_end,
                    },
                    Some(&c.metadata),
                );
                VectorPoint {
                    id: c.chunk_id.clone(),
                    embedding: emb,
                    text: c.raw_text.clone(),
                    payload,
                }
            })
            .collect();

        // ---- 5. Qdrant upsert -------------------------------------------
        let n_points = points.len();
        self.qdrant
            .upsert(&self.chunk_collection, points)
            .await
            .map_err(|e| IndexError::Upsert(e.to_string()))?;
        tracing::Span::current().record("n_qdrant_points", n_points);
        info!(n_points, collection = %self.chunk_collection, "qdrant chunks upserted");

        // ---- 6a. Optional LLM stages (best-effort) ----------------------
        //
        // Both stages are opt-in (Option<Arc<..>>). Failures here must NOT
        // fail the core indexing operation: the chunks + structure are
        // already in Qdrant/Neo4j, so the document is retrievable. We
        // log and continue.
        self.run_summaries(&user_id, &job.minio_object_key, &chunks)
            .await;
        self.run_kg_extraction(&user_id, &job.minio_object_key, &chunks)
            .await;

        // ---- 6. Neo4j minimal structural upsert --------------------------
        //
        // Docling-sourced chunks carry proper heading paths; pdf-extract
        // chunks fall back to a preview of the chunk content. Either way
        // we emit one `Section` per chunk so `coverage_planner` and
        // `outline_corpus` have a flat TOC to walk.
        //
        // Chunk ids come from the same `IndexedChunk.chunk_id` values we
        // just upserted to Qdrant, so `search_sections` /
        // `fetch_section_chunks` can follow `IN_SECTION` from Neo4j back
        // into the payloads on the vector side.
        let minimal_rows: Vec<MinimalChunkRow> = chunks
            .iter()
            .map(|c| MinimalChunkRow {
                chunk_id: c.chunk_id.clone(),
                heading: c.section_path.clone().unwrap_or_else(|| c.raw_text.clone()),
                chunk_index: c.chunk_index as i64,
                page_start: c.page_start.map(|p| p as i64),
                page_end: c.page_end.map(|p| p as i64),
            })
            .collect();
        let n_sections = minimal_rows.len();
        self.neo4j
            .upsert_minimal_document(
                &user_id,
                &job.minio_object_key,
                &sha256,
                &job.minio_object_key,
                &minimal_rows,
            )
            .await
            .map_err(|e| IndexError::Upsert(e.to_string()))?;
        tracing::Span::current().record("n_neo4j_sections", n_sections);
        info!(n_sections, "neo4j minimal structure upserted");

        // ---- 7. mark complete -------------------------------------------
        self.pg
            .mark_index_complete(&job.minio_object_key, &sha256, n_pages, n_chunks)
            .await
            .map_err(|e| IndexError::StateMachine(e.to_string()))?;

        let n_pages_estimate = n_pages;

        info!(
            sha256 = %sha256,
            n_pages = n_pages_estimate,
            n_chunks,
            "indexing complete"
        );
        Ok(IndexSummary {
            claimed: true,
            sha256,
            n_pages: n_pages_estimate,
            n_chunks,
        })
    }

    /// Route a document to the structural pipeline appropriate for its
    /// MIME / extension + the configured [`IngestBackend`].
    ///
    /// Returns the chunk list plus an estimated page count.
    #[instrument(
        skip(self, bytes),
        fields(
            minio_object_key = %minio_key,
            bytes = bytes.len(),
            ext = tracing::field::Empty,
            backend = tracing::field::Empty,
            n_chunks = tracing::field::Empty,
            n_pages = tracing::field::Empty,
        )
    )]
    async fn chunk_pipeline(
        &self,
        minio_key: &str,
        bytes: &[u8],
    ) -> Result<(Vec<IndexedChunk>, i32), IndexError> {
        let ext = minio_key
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        tracing::Span::current().record("ext", &ext);

        let is_pdf = ext == "pdf";

        // Primary/fallback dispatch for PDF inputs.
        //
        // - Docling / Marker: sidecar required; error propagates on failure.
        // - DoclingWithLegacyFallback / MarkerWithLegacyFallback: try the
        //   sidecar, fall through to pdf-extract on error.
        // - Legacy: skip both sidecars entirely.
        if is_pdf {
            match self.ingest_backend {
                IngestBackend::Docling | IngestBackend::DoclingWithLegacyFallback => {
                    if let Some(docling) = self.docling.as_ref() {
                        match self.chunk_via_docling(docling, minio_key, bytes).await {
                            Ok(result) => {
                                tracing::Span::current().record("backend", "docling");
                                tracing::Span::current().record("n_chunks", result.0.len());
                                tracing::Span::current().record("n_pages", result.1);
                                return Ok(result);
                            }
                            Err(e) => {
                                if matches!(
                                    self.ingest_backend,
                                    IngestBackend::DoclingWithLegacyFallback
                                ) {
                                    warn!(
                                        error = %e,
                                        fallback = "pdf-extract",
                                        "docling failed; falling back to pdf-extract"
                                    );
                                } else {
                                    return Err(e);
                                }
                            }
                        }
                    } else if matches!(self.ingest_backend, IngestBackend::Docling) {
                        return Err(IndexError::Parse(
                            "Docling backend required but no sidecar client configured".into(),
                        ));
                    }
                }
                IngestBackend::Marker | IngestBackend::MarkerWithLegacyFallback => {
                    if let Some(marker) = self.marker.as_ref() {
                        match self.chunk_via_marker(marker, minio_key, bytes).await {
                            Ok(result) => {
                                tracing::Span::current().record("backend", "marker");
                                tracing::Span::current().record("n_chunks", result.0.len());
                                tracing::Span::current().record("n_pages", result.1);
                                return Ok(result);
                            }
                            Err(e) => {
                                if matches!(
                                    self.ingest_backend,
                                    IngestBackend::MarkerWithLegacyFallback
                                ) {
                                    warn!(
                                        error = %e,
                                        fallback = "pdf-extract",
                                        "marker failed; falling back to pdf-extract"
                                    );
                                } else {
                                    return Err(e);
                                }
                            }
                        }
                    } else if matches!(self.ingest_backend, IngestBackend::Marker) {
                        return Err(IndexError::Parse(
                            "Marker backend required but no sidecar client configured".into(),
                        ));
                    }
                }
                IngestBackend::Legacy => {}
            }
        }

        let (chunks, pages) = self.chunk_via_text_fallback(minio_key, bytes, &ext)?;
        tracing::Span::current().record("backend", "pdf_extract");
        tracing::Span::current().record("n_chunks", chunks.len());
        tracing::Span::current().record("n_pages", pages);
        tracing::Span::current().record("n_chunks", chunks.len());
        Ok((chunks, pages))
    }

    async fn chunk_via_docling(
        &self,
        docling: &DoclingSidecarClient,
        minio_key: &str,
        bytes: &[u8],
    ) -> Result<(Vec<IndexedChunk>, i32), IndexError> {
        let response = docling
            .chunk_document(bytes.to_vec(), minio_key, minio_key, "application/pdf")
            .await
            .map_err(|e| IndexError::Parse(format!("docling: {e}")))?;
        Ok(Self::sidecar_response_to_chunks(response, minio_key))
    }

    async fn chunk_via_marker(
        &self,
        marker: &MarkerSidecarClient,
        minio_key: &str,
        bytes: &[u8],
    ) -> Result<(Vec<IndexedChunk>, i32), IndexError> {
        let response = marker
            .chunk_document(bytes.to_vec(), minio_key, minio_key, "application/pdf")
            .await
            .map_err(|e| IndexError::Parse(format!("marker: {e}")))?;
        Ok(Self::sidecar_response_to_chunks(response, minio_key))
    }

    /// Common mapping from a sidecar [`DoclingResponse`] (also used by
    /// Marker — same proto) to the crate's [`IndexedChunk`] shape.
    ///
    /// Page count is derived from the maximum `page_end` across chunks
    /// because neither sidecar footer carries a page-count field. Falls
    /// back to 1 when every chunk is page-less.
    fn sidecar_response_to_chunks(
        response: DoclingResponse,
        minio_key: &str,
    ) -> (Vec<IndexedChunk>, i32) {
        let total = response.chunks.len() as u32;
        let chunks: Vec<IndexedChunk> = response
            .chunks
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let section_path = if c.heading_path.is_empty() {
                    None
                } else {
                    Some(c.heading_path.join(" / "))
                };
                let metadata = metadata_for_chunk(&c.raw_text, minio_key);
                IndexedChunk {
                    chunk_id: c.chunk_id,
                    chunk_index: i,
                    total_chunks: total as usize,
                    embed_text: c.embed_text,
                    raw_text: c.raw_text,
                    section_id: if c.section_id.is_empty() {
                        None
                    } else {
                        Some(c.section_id)
                    },
                    section_path,
                    page_start: if c.page_start == 0 {
                        None
                    } else {
                        Some(c.page_start)
                    },
                    page_end: if c.page_end == 0 {
                        None
                    } else {
                        Some(c.page_end)
                    },
                    metadata,
                }
            })
            .collect();

        let n_pages = chunks.iter().filter_map(|c| c.page_end).max().unwrap_or(1) as i32;
        (chunks, n_pages)
    }

    /// Best-effort per-chunk LLM summarisation.
    ///
    /// Emits `summary_text` points into the summaries collection so the
    /// `outline_corpus` tool can render one-line section blurbs and the
    /// `Summaries` retrieval strategy can re-rank at summary granularity.
    /// When the summarizer is absent (no API key / disabled), this is a
    /// no-op. Any downstream failure logs at WARN and does not fail
    /// indexing.
    #[instrument(
        skip(self, chunks),
        fields(
            user_id = %user_id,
            minio_object_key = %minio_key,
            enabled = self.summarizer.is_some(),
            n_input = chunks.len(),
            n_usable = tracing::field::Empty,
            n_upserted = tracing::field::Empty,
        )
    )]
    async fn run_summaries(&self, user_id: &UserId, minio_key: &str, chunks: &[IndexedChunk]) {
        let Some(summarizer) = self.summarizer.as_ref() else {
            return;
        };
        let items: Vec<SummarizeItem<'_>> = chunks
            .iter()
            .map(|c| SummarizeItem {
                chunk_id: c.chunk_id.as_str(),
                chunk_text: c.raw_text.as_str(),
                doc_id: Some(minio_key),
                section_id: c.section_id.as_deref(),
                ingest_run_id: None,
            })
            .collect();

        let summaries = summarizer
            .summarize_batch(&items, DEFAULT_SUMMARIZER_CONCURRENCY)
            .await;

        // Keep only the ones with a non-empty summary text so we never
        // embed an empty string.
        let usable: Vec<_> = summaries
            .into_iter()
            .filter(|s| {
                matches!(s.summary_status, SummaryStatus::Ok) && !s.summary_text.trim().is_empty()
            })
            .collect();
        tracing::Span::current().record("n_usable", usable.len());
        if usable.is_empty() {
            info!("summariser produced no usable summaries; skipping upsert");
            return;
        }

        let texts: Vec<&str> = usable.iter().map(|s| s.summary_text.as_str()).collect();
        let embeddings = match self.embedder.embed_documents(&texts).await {
            Ok(v) => v,
            Err(e) => {
                // Drops ALL summaries for this document silently from the
                // retrieval index. Logged at ERROR so Sentry / PagerDuty
                // alerting catches it — a paying tenant's `Summaries`
                // strategy will silently return nothing otherwise.
                tracing::error!(
                    error = %e,
                    user_id = %user_id,
                    minio_key,
                    n_summaries = usable.len(),
                    "summary embedding failed; dropping all summaries for this document"
                );
                return;
            }
        };

        if embeddings.len() != usable.len() {
            tracing::error!(
                got = embeddings.len(),
                expected = usable.len(),
                user_id = %user_id,
                minio_key,
                "summary embedding count mismatch — embedder contract violation; dropping summaries"
            );
            return;
        }

        let points: Vec<VectorPoint> = usable
            .iter()
            .zip(embeddings.into_iter())
            .map(|(s, emb)| {
                let payload = build_summary_payload(SummaryPayloadInput {
                    user_id: user_id.as_str(),
                    minio_object_key: minio_key,
                    chunk_id: &s.chunk_id,
                    section_id: s.section_id.as_deref(),
                    summary_id: &s.summary_id,
                    summary_text: &s.summary_text,
                });
                VectorPoint {
                    id: s.summary_id.clone(),
                    embedding: emb,
                    text: s.summary_text.clone(),
                    payload,
                }
            })
            .collect();

        let n_points = points.len();
        if let Err(e) = self.qdrant.upsert(&self.summaries_collection, points).await {
            tracing::error!(
                error = %e,
                user_id = %user_id,
                minio_key,
                n_dropped = n_points,
                collection = %self.summaries_collection,
                "summary Qdrant upsert failed; tenant summaries lost for this document"
            );
        } else {
            tracing::Span::current().record("n_upserted", n_points);
            info!(
                count = n_points,
                collection = %self.summaries_collection,
                "summaries upserted"
            );
        }
    }

    /// Best-effort per-chunk LLM knowledge-graph extraction.
    ///
    /// Flattens each [`KnowledgeGraph`] into [`Triplet`]-like vector
    /// points (`"{src} -> {rel} -> {dst}"` as embed text) and writes them
    /// to the triplets Qdrant collection, where the graph-CoT retriever
    /// scores whole relations against a query embedding.
    ///
    /// Failures log + skip. The structural graph already on Neo4j is
    /// untouched.
    #[instrument(
        skip(self, chunks),
        fields(
            user_id = %user_id,
            minio_object_key = %minio_key,
            enabled = self.extractor.is_some(),
            n_input = chunks.len(),
            n_triplets = tracing::field::Empty,
            n_upserted = tracing::field::Empty,
        )
    )]
    async fn run_kg_extraction(&self, user_id: &UserId, minio_key: &str, chunks: &[IndexedChunk]) {
        let Some(extractor) = self.extractor.as_ref() else {
            return;
        };

        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.raw_text.as_str()).collect();
        let graphs = extractor
            .extract_batch(&chunk_texts, DEFAULT_KG_CONCURRENCY)
            .await;

        let mut triplets: Vec<TripletRecord> = Vec::new();
        for (chunk, graph) in chunks.iter().zip(graphs.iter()) {
            if graph.is_empty() {
                continue;
            }
            // Build a name lookup so edge endpoints carry their
            // human-readable names rather than the bare snake_case id.
            let name_of: std::collections::HashMap<&str, &str> = graph
                .nodes
                .iter()
                .map(|n| (n.id.as_str(), n.name.as_str()))
                .collect();
            for edge in &graph.edges {
                let src_name = name_of
                    .get(edge.source_node_id.as_str())
                    .copied()
                    .unwrap_or(edge.source_node_id.as_str())
                    .to_owned();
                let dst_name = name_of
                    .get(edge.target_node_id.as_str())
                    .copied()
                    .unwrap_or(edge.target_node_id.as_str())
                    .to_owned();
                let key = format!(
                    "{}|{}|{}",
                    edge.source_node_id, edge.relationship_name, edge.target_node_id
                );
                let id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, key.as_bytes()).to_string();
                let text = format!("{src_name} -> {} -> {dst_name}", edge.relationship_name);
                triplets.push(TripletRecord {
                    id,
                    chunk_id: chunk.chunk_id.clone(),
                    src_node_id: edge.source_node_id.clone(),
                    src_name,
                    rel_name: edge.relationship_name.clone(),
                    dst_node_id: edge.target_node_id.clone(),
                    dst_name,
                    text,
                });
            }
        }

        tracing::Span::current().record("n_triplets", triplets.len());
        if triplets.is_empty() {
            info!("KG extractor produced zero triplets; skipping upsert");
            return;
        }

        let texts: Vec<&str> = triplets.iter().map(|t| t.text.as_str()).collect();
        let embeddings = match self.embedder.embed_documents(&texts).await {
            Ok(v) => v,
            Err(e) => {
                // Drops ALL triplets for this document silently — every
                // future GraphHybrid / GraphCoT query on this tenant
                // will operate with reduced graph coverage. Bump to
                // ERROR so alerting catches it.
                tracing::error!(
                    error = %e,
                    user_id = %user_id,
                    minio_key,
                    n_triplets = triplets.len(),
                    "triplet embedding failed; dropping all KG triplets for this document"
                );
                return;
            }
        };
        if embeddings.len() != triplets.len() {
            tracing::error!(
                got = embeddings.len(),
                expected = triplets.len(),
                user_id = %user_id,
                minio_key,
                "triplet embedding count mismatch — embedder contract violation; dropping triplets"
            );
            return;
        }

        let points: Vec<VectorPoint> = triplets
            .iter()
            .zip(embeddings.into_iter())
            .map(|(t, emb)| {
                let payload = build_triplet_payload(TripletPayloadInput {
                    user_id: user_id.as_str(),
                    chunk_id: &t.chunk_id,
                    source_node_id: &t.src_node_id,
                    source_node_name: &t.src_name,
                    relationship_name: &t.rel_name,
                    target_node_id: &t.dst_node_id,
                    target_node_name: &t.dst_name,
                    triplet_text: &t.text,
                });
                VectorPoint {
                    id: t.id.clone(),
                    embedding: emb,
                    text: t.text.clone(),
                    payload,
                }
            })
            .collect();

        let n_points = points.len();
        if let Err(e) = self.qdrant.upsert(&self.triplet_collection, points).await {
            tracing::error!(
                error = %e,
                user_id = %user_id,
                minio_key,
                n_dropped = n_points,
                collection = %self.triplet_collection,
                "triplets Qdrant upsert failed; tenant KG triplets lost for this document"
            );
        } else {
            tracing::Span::current().record("n_upserted", n_points);
            info!(
                count = n_points,
                collection = %self.triplet_collection,
                "triplets upserted"
            );
        }
    }

    fn chunk_via_text_fallback(
        &self,
        minio_key: &str,
        bytes: &[u8],
        ext: &str,
    ) -> Result<(Vec<IndexedChunk>, i32), IndexError> {
        let (text, n_pages) = extract_text(minio_key, bytes, ext)?;
        if text.trim().is_empty() {
            return Err(IndexError::Parse(
                "document produced no extractable text".into(),
            ));
        }
        let doc_chunks = chunk_document(&text, minio_key, minio_key, &self.chunker_config);
        let total = doc_chunks.len();
        let chunks: Vec<IndexedChunk> = doc_chunks
            .into_iter()
            .map(|c| {
                let metadata = metadata_for_chunk(&c.content, minio_key);
                IndexedChunk {
                    chunk_id: c.chunk_id,
                    chunk_index: c.chunk_index,
                    total_chunks: total,
                    // pdf-extract has no context-enriched form — reuse raw.
                    embed_text: c.content.clone(),
                    raw_text: c.content,
                    section_id: None,
                    section_path: None,
                    page_start: None,
                    page_end: None,
                    metadata,
                }
            })
            .collect();
        Ok((chunks, n_pages))
    }
}

#[async_trait]
impl IndexHandler for AgentIndexHandler {
    async fn handle(&self, job: IndexJob) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self.index(&job).await {
            Ok(summary) => {
                if summary.claimed {
                    info!(
                        minio_object_key = %job.minio_object_key,
                        sha256 = %summary.sha256,
                        n_pages = summary.n_pages,
                        n_chunks = summary.n_chunks,
                        "job completed"
                    );
                }
                Ok(())
            }
            Err(e) => {
                // Two-tier nack policy:
                //
                // 1. `mark_index_failed` succeeds → ack the message: the
                //    failure is now durably recorded in Postgres; web will
                //    see it via `GetDocumentStatus` and decide whether to
                //    re-enqueue. Redelivering would only loop back into
                //    `claim_pending_for_indexing`, fail the CAS (state is
                //    now 'failed'), and no-op — pure broker overhead.
                //
                // 2. `mark_index_failed` itself fails → the state is still
                //    `indexing`, so we must ask the broker to redeliver so
                //    another attempt can record the outcome. Return Err
                //    so lapin nacks with requeue.
                error!(
                    minio_object_key = %job.minio_object_key,
                    job_id = %job.job_id,
                    error = %e,
                    "indexing failed; persisting state"
                );
                match self
                    .pg
                    .mark_index_failed(&job.minio_object_key, &e.to_string())
                    .await
                {
                    Ok(()) => {
                        info!(
                            minio_object_key = %job.minio_object_key,
                            "failure persisted; acking message"
                        );
                        Ok(())
                    }
                    Err(persist_err) => {
                        error!(
                            minio_object_key = %job.minio_object_key,
                            persist_error = %persist_err,
                            "could not persist failure state; nacking for redelivery"
                        );
                        Err(Box::new(e))
                    }
                }
            }
        }
    }
}

// ---- helpers ----------------------------------------------------------------

/// Single canonical shape for a chunk regardless of whether it came from
/// the Docling sidecar or the `pdf-extract` fallback. Keeps the embed +
/// upsert code paths fully backend-agnostic.
struct IndexedChunk {
    chunk_id: String,
    chunk_index: usize,
    total_chunks: usize,
    /// Text used as input to the embedder. For Docling chunks this is
    /// the heading-contextualised `embed_text`; for the fallback it is
    /// the same as `raw_text`.
    embed_text: String,
    /// Text used for Qdrant's server-side BM25 and stored verbatim in
    /// the point's payload as `chunk_text`.
    raw_text: String,
    section_id: Option<String>,
    section_path: Option<String>,
    page_start: Option<u32>,
    page_end: Option<u32>,
    /// Content-type signals + `section_title` / `language` heuristics
    /// derived from `raw_text`. Populated unconditionally at chunk
    /// construction so every `IndexedChunk` downstream sees the same
    /// schema — the payload builder relies on these fields being present
    /// so the retrieval-side `MetadataFilterExtractor` has a non-empty
    /// target for `has_table` / `has_formula` / `has_code` predicates.
    metadata: ChunkMetadata,
}

/// Flattened `(graph × chunk)` triplet row produced during KG extraction.
///
/// Carries every field `build_triplet_payload` needs, by name. The
/// previous implementation used a seven-element tuple which silently
/// lost `dst_name` — keeping `target_node_name` as a distinct named
/// field makes it a compile error to forget the display name again.
struct TripletRecord {
    /// Deterministic UUIDv5 derived from `src|rel|dst` so re-ingesting
    /// the same edge from a different chunk is idempotent.
    id: String,
    /// Chunk that sourced this triplet (for lineage / citation).
    chunk_id: String,
    src_node_id: String,
    src_name: String,
    rel_name: String,
    dst_node_id: String,
    /// Human-readable display name for the destination node. Used as
    /// `target_node_name` in the Qdrant payload so citations can render
    /// the triplet without a separate Neo4j lookup.
    dst_name: String,
    /// `src_name -> rel_name -> dst_name` — also the text embedded.
    text: String,
}

/// Compute [`ChunkMetadata`] for the given chunk text.
///
/// Centralises the enrichment call so both the Docling and pdf-extract
/// paths produce byte-identical metadata for identical input.
fn metadata_for_chunk(raw_text: &str, source: &str) -> ChunkMetadata {
    extract_metadata(raw_text, source, &[], ChunkExtras::default())
}

/// Determine text content from the raw bytes using the object key's
/// extension as the routing hint. Returns `(text, estimated_n_pages)` so
/// the state row carries an approximate page count even when the backend
/// (plain text / markdown) has no native notion of pages.
fn extract_text(key: &str, bytes: &[u8], ext: &str) -> Result<(String, i32), IndexError> {
    match ext {
        "pdf" => {
            let text = pdf_extract::extract_text_from_mem(bytes)
                .map_err(|e| IndexError::Parse(format!("pdf_extract: {e}")))?;
            // pdf-extract does not report page count; approximate from `\x0c`
            // form-feed separators the crate inserts between pages.
            let n_pages = (text.matches('\x0c').count() as i32).max(1);
            Ok((text, n_pages))
        }
        "txt" | "md" | "markdown" | "html" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| IndexError::Parse(format!("utf-8: {e}")))?
                .to_owned();
            Ok((text, 1))
        }
        other => Err(IndexError::Parse(format!(
            "unsupported extension '{other}' on key '{key}'"
        ))),
    }
}

/// What we've accomplished on a successful pass — used for debug logs and
/// potentially tests.
#[derive(Debug)]
struct IndexSummary {
    /// `false` when the CAS claim failed (another worker took the job);
    /// `true` when this invocation did the work.
    claimed: bool,
    sha256: String,
    n_pages: i32,
    n_chunks: i32,
}

impl IndexSummary {
    fn already_claimed() -> Self {
        Self {
            claimed: false,
            sha256: String::new(),
            n_pages: 0,
            n_chunks: 0,
        }
    }
}

/// Typed error enum so `handle` can decide retry semantics per variant
/// when we stop erasing it to `Box<dyn Error>` (future work).
#[derive(Debug, thiserror::Error)]
enum IndexError {
    #[error("invalid user_id: {0}")]
    InvalidUser(String),
    #[error("state machine: {0}")]
    StateMachine(String),
    #[error("fetch from MinIO: {0}")]
    Fetch(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("embed: {0}")]
    Embed(String),
    #[error("upsert to Qdrant: {0}")]
    Upsert(String),
}
