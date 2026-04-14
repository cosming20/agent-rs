//! End-to-end ingestion pipeline.
//!
//! Orchestrates the full ingestion flow for both the Docling sidecar path
//! and the legacy `pdf-extract` path:
//!
//! **Docling path** (`ingest_with_docling`):
//! - sha256 dedup via Postgres
//! - Docling sidecar gRPC streaming
//! - Embed + Qdrant upsert (27-field rich payload)
//! - Neo4j structural upsert
//! - LLM KG extraction, triplet indexing, chunk summarization
//! - Postgres `mark_complete` with 3-retry exponential backoff
//!
//! **Legacy path** (`ingest_pdf_legacy`, feature-gated):
//! - `pdf-extract` sync extraction
//! - Same `ingest_text` pipeline as before

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use agent_rs_core::{CollectionName, IngestBackend, NAMESPACE_OID};
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::postgres::documents::HeaderMetadata;
use agent_rs_stores::postgres::models::{BeginIngestionOutcome, NewIngestedDocument};
use agent_rs_stores::traits::{GraphStore, VectorStore};
use agent_rs_stores::types::{GraphEdge, GraphNode, NodeLabel, RelationType, VectorPoint};
use chrono::Utc;
use regex::Regex;
use serde::Serialize;
use serde_json::json;
use uuid::Uuid;

use crate::embedder::Embedder;
use crate::error::RetrievalError;
use crate::proto::agentic_rag::v1::{
    Chunk as ProtoChunk, DocItemLabel, DocumentFooter as ProtoDocumentFooter,
    DocumentHeader as ProtoDocumentHeader,
};
use agent_rs_stores::neo4j::structural::{
    DocumentHeaderSnapshot, PageInfo as StructuralPageInfo, StructuralChunk, StructuralPicture,
    StructuralTable,
};

use super::chunker::{self, ChunkerConfig, DocumentChunk};
use super::docling_client::DoclingSidecarClient;
use super::graph_extractor::{
    DEFAULT_BATCH_CONCURRENCY, ExtractedNode, KnowledgeGraph, LlmGraphExtractor, Triplet,
    sanitize_relationship,
};
use super::summarizer::{ChunkSummarizer, DEFAULT_SUMMARIZER_CONCURRENCY};

// ---- Result type ----

/// Summary of a completed ingestion operation.
#[derive(Debug, Clone, Serialize)]
pub struct IngestionResult {
    /// Identifier of the ingested document.
    pub doc_id: String,
    /// Number of chunks created from the document.
    pub chunks_created: usize,
    /// Number of vectors upserted to the vector store.
    pub vectors_upserted: usize,
    /// Number of graph nodes created.
    pub graph_nodes_created: usize,
    /// Number of graph edges created.
    pub graph_edges_created: usize,
    /// Number of linearised triplet vectors upserted to the triplet
    /// collection. Always zero when no triplet collection is configured.
    pub triplets_upserted: usize,
    /// Number of chunk summary vectors upserted to the summaries collection.
    /// Always zero when no summarizer or summaries collection is configured.
    pub summaries_created: usize,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

// ---- Pipeline ----

/// Optional extensions for the ingestion pipeline.
///
/// Use this to supply components that are not required for a minimal ingestion
/// run (e.g. an [`LlmGraphExtractor`] for LLM-based entity extraction or a
/// [`DoclingSidecarClient`] for the Docling ingest path).
///
/// Defaults leave all optional extensions disabled; the pipeline then falls
/// back to its heuristic entity extractor.
#[derive(Default)]
pub struct IngestionPipelineConfig {
    /// Chunking configuration used when splitting documents.
    pub chunker: ChunkerConfig,
    /// Optional LLM-based knowledge graph extractor. When `Some`, the pipeline
    /// calls the extractor after chunking and uses the typed entities +
    /// relations it returns instead of the heuristic fallback.
    pub llm_extractor: Option<Arc<LlmGraphExtractor>>,
    /// Concurrency for batch LLM extraction. Defaults to
    /// [`DEFAULT_BATCH_CONCURRENCY`].
    pub llm_concurrency: usize,
    /// Optional dedicated collection for storing linearised triplet
    /// embeddings.
    ///
    /// When `Some`, the pipeline embeds every
    /// `(source) -[relation]-> (target)` triple produced by the LLM
    /// extractor and upserts the embeddings into this collection so that
    /// retrieval can later score triplets directly against the query.
    /// Has no effect when [`Self::llm_extractor`] is `None`.
    pub triplet_collection: Option<CollectionName>,
    /// Optional LLM-based chunk summarizer.
    ///
    /// When `Some` (and [`Self::summaries_collection`] is also `Some`), the
    /// pipeline generates a 1-2 sentence summary for every chunk after the
    /// main chunk upsert and stores the summary embeddings in the summaries
    /// collection. This powers the [`RetrievalStrategy::Summaries`] search path.
    pub summarizer: Option<Arc<ChunkSummarizer>>,
    /// Concurrency for batch summarization. Defaults to
    /// [`DEFAULT_SUMMARIZER_CONCURRENCY`].
    pub summarizer_concurrency: usize,
    /// Optional dedicated collection for storing chunk summary embeddings.
    ///
    /// When `Some` (and [`Self::summarizer`] is also `Some`), summarized
    /// chunk embeddings are upserted here.
    pub summaries_collection: Option<CollectionName>,
    /// Ingest backend selector.  Default: [`IngestBackend::Docling`].
    pub ingest_backend: IngestBackend,
    /// Docling sidecar streaming client.
    ///
    /// When `Some`, `ingest_pdf` routes Docling-path calls to this client.
    /// When `None` and the backend is `Docling` or `DoclingWithLegacyFallback`,
    /// `ingest_with_docling` returns an error.
    pub docling_client: Option<DoclingSidecarClient>,
    /// Postgres store for ingest state machine tracking.
    pub pg_store: Option<Arc<PgStore>>,
}

impl IngestionPipelineConfig {
    /// Create a new config with the given chunker settings. All optional
    /// extensions are left disabled.
    pub fn new(chunker: ChunkerConfig) -> Self {
        Self {
            chunker,
            llm_extractor: None,
            llm_concurrency: DEFAULT_BATCH_CONCURRENCY,
            triplet_collection: None,
            summarizer: None,
            summarizer_concurrency: DEFAULT_SUMMARIZER_CONCURRENCY,
            summaries_collection: None,
            ingest_backend: IngestBackend::default(),
            docling_client: None,
            pg_store: None,
        }
    }

    /// Set the ingest backend.
    pub fn with_ingest_backend(mut self, backend: IngestBackend) -> Self {
        self.ingest_backend = backend;
        self
    }

    /// Attach a Docling sidecar client.
    pub fn with_docling_client(mut self, client: DoclingSidecarClient) -> Self {
        self.docling_client = Some(client);
        self
    }

    /// Attach a Postgres store for ingest state machine tracking.
    pub fn with_pg_store(mut self, pg_store: Arc<PgStore>) -> Self {
        self.pg_store = Some(pg_store);
        self
    }

    /// Attach an LLM-based graph extractor.
    pub fn with_llm_extractor(mut self, extractor: Arc<LlmGraphExtractor>) -> Self {
        self.llm_extractor = Some(extractor);
        self
    }

    /// Override the batch extraction concurrency.
    pub fn with_llm_concurrency(mut self, concurrency: usize) -> Self {
        self.llm_concurrency = concurrency.max(1);
        self
    }

    /// Enable triplet indexing by setting the destination collection.
    ///
    /// Triplet indexing only runs when an [`LlmGraphExtractor`] is also
    /// attached — the heuristic extractor does not emit typed relations.
    pub fn with_triplet_collection(mut self, triplet_collection: CollectionName) -> Self {
        self.triplet_collection = Some(triplet_collection);
        self
    }

    /// Attach a [`ChunkSummarizer`] for LLM-based chunk summarization.
    ///
    /// Summarization only runs when a [`Self::summaries_collection`] is also
    /// configured.
    pub fn with_summarizer(mut self, summarizer: Arc<ChunkSummarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Override the batch summarization concurrency.
    pub fn with_summarizer_concurrency(mut self, concurrency: usize) -> Self {
        self.summarizer_concurrency = concurrency.max(1);
        self
    }

    /// Enable summary indexing by setting the destination collection.
    ///
    /// Summary indexing only runs when a [`Self::summarizer`] is also attached.
    pub fn with_summaries_collection(mut self, summaries_collection: CollectionName) -> Self {
        self.summaries_collection = Some(summaries_collection);
        self
    }
}

/// Universal document ingestion pipeline.
///
/// Wires together chunking, embedding, vector storage, entity extraction,
/// and graph storage into a single async operation.
///
/// # Construction
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use agent_rs_retrieval::ingestion::pipeline::IngestionPipeline;
/// # use agent_rs_retrieval::ingestion::chunker::ChunkerConfig;
/// # use agent_rs_core::CollectionName;
/// // let pipeline = IngestionPipeline::new(
/// //     embedder,
/// //     vector_store,
/// //     graph_store,
/// //     ChunkerConfig::default(),
/// //     CollectionName("documents".into()),
/// // );
/// ```
pub struct IngestionPipeline {
    embedder: Arc<dyn Embedder + Send + Sync>,
    vector_store: Arc<dyn VectorStore + Send + Sync>,
    graph_store: Arc<dyn GraphStore + Send + Sync>,
    chunker_config: ChunkerConfig,
    collection: CollectionName,
    llm_extractor: Option<Arc<LlmGraphExtractor>>,
    llm_concurrency: usize,
    /// Optional collection for linearised triplet embeddings. See
    /// [`IngestionPipelineConfig::triplet_collection`].
    triplet_collection: Option<CollectionName>,
    /// Optional LLM-based chunk summarizer. See
    /// [`IngestionPipelineConfig::summarizer`].
    summarizer: Option<Arc<ChunkSummarizer>>,
    /// Concurrency for batch summarization calls.
    summarizer_concurrency: usize,
    /// Optional collection for chunk summary embeddings. See
    /// [`IngestionPipelineConfig::summaries_collection`].
    summaries_collection: Option<CollectionName>,
    /// Selects which ingest backend `ingest_pdf` uses.
    ///
    /// Default: [`IngestBackend::Docling`].  Configurable via
    /// `AGENT_RS__INGEST_BACKEND` and surfaced by [`IngestionPipelineConfig`].
    ingest_backend: IngestBackend,
    /// Docling sidecar streaming client.
    ///
    /// `None` when the Docling backend is not configured; calling
    /// `ingest_with_docling` returns an error in that case.
    docling_client: Option<tokio::sync::Mutex<DoclingSidecarClient>>,
    /// Postgres store for ingest state machine tracking.
    ///
    /// `None` when not configured; the ingest state machine is skipped
    /// gracefully (dedup and partial-failure tracking are disabled).
    pg_store: Option<Arc<PgStore>>,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline with only the mandatory dependencies.
    ///
    /// Uses heuristic entity extraction. Use [`Self::with_config`] to supply
    /// an [`LlmGraphExtractor`] or other optional extensions.
    pub fn new(
        embedder: Arc<dyn Embedder + Send + Sync>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        chunker_config: ChunkerConfig,
        collection: CollectionName,
    ) -> Self {
        Self::with_config(
            embedder,
            vector_store,
            graph_store,
            collection,
            IngestionPipelineConfig::new(chunker_config),
        )
    }

    /// Create a new ingestion pipeline with full optional configuration.
    pub fn with_config(
        embedder: Arc<dyn Embedder + Send + Sync>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        collection: CollectionName,
        config: IngestionPipelineConfig,
    ) -> Self {
        let llm_concurrency = if config.llm_concurrency == 0 {
            DEFAULT_BATCH_CONCURRENCY
        } else {
            config.llm_concurrency
        };
        let summarizer_concurrency = if config.summarizer_concurrency == 0 {
            DEFAULT_SUMMARIZER_CONCURRENCY
        } else {
            config.summarizer_concurrency
        };
        Self {
            embedder,
            vector_store,
            graph_store,
            chunker_config: config.chunker,
            collection,
            llm_extractor: config.llm_extractor,
            llm_concurrency,
            triplet_collection: config.triplet_collection,
            summarizer: config.summarizer,
            summarizer_concurrency,
            summaries_collection: config.summaries_collection,
            ingest_backend: config.ingest_backend,
            docling_client: config.docling_client.map(tokio::sync::Mutex::new),
            pg_store: config.pg_store,
        }
    }

    /// Attach a Docling sidecar client after construction (builder-style).
    pub fn with_docling_client(mut self, client: DoclingSidecarClient) -> Self {
        self.docling_client = Some(tokio::sync::Mutex::new(client));
        self
    }

    /// Attach a Postgres store after construction (builder-style).
    pub fn with_pg_store(mut self, pg_store: Arc<PgStore>) -> Self {
        self.pg_store = Some(pg_store);
        self
    }

    /// Ingest a plain-text document.
    ///
    /// # Arguments
    ///
    /// * `text` -- full document text
    /// * `doc_id` -- unique document identifier
    /// * `source` -- filename or URL
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError`] on embedding, storage, or graph failures.
    #[tracing::instrument(skip(self, text), fields(doc_id = doc_id, source = source, text_len = text.len()))]
    pub async fn ingest_text(
        &self,
        text: &str,
        doc_id: &str,
        source: &str,
    ) -> Result<IngestionResult, RetrievalError> {
        let start = Instant::now();

        // 1. Chunk the document.
        let chunks = chunker::chunk_document(text, doc_id, source, &self.chunker_config);
        if chunks.is_empty() {
            tracing::warn!(doc_id, "document produced zero chunks after chunking");
            return Ok(IngestionResult {
                doc_id: doc_id.to_owned(),
                chunks_created: 0,
                vectors_upserted: 0,
                graph_nodes_created: 0,
                graph_edges_created: 0,
                triplets_upserted: 0,
                summaries_created: 0,
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        let chunks_created = chunks.len();
        tracing::info!(doc_id, chunks_created, "chunking complete");

        // 2. Embed all chunks (batch).
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embedder.embed_documents(&texts).await?;

        if embeddings.len() != chunks.len() {
            return Err(RetrievalError::Ingestion {
                reason: format!(
                    "embedding count mismatch: got {} embeddings for {} chunks",
                    embeddings.len(),
                    chunks.len()
                ),
            });
        }

        // 2b. Server-side BM25 sparse inference via Qdrant `qdrant/bm25` model.
        //
        // No local SPLADE/fastembed sparse model is needed. Each VectorPoint
        // carries its raw `text`; the store layer sends a `Document { model:
        // "qdrant/bm25" }` named vector at upsert time and Qdrant handles
        // tokenisation and IDF weighting. The collection is always created with
        // `enable_sparse: true` so the sparse named vector exists.
        tracing::debug!(
            count = chunks.len(),
            "using server-side BM25 for sparse vectors"
        );

        // 3. Build VectorPoints with rich payloads.
        let vector_points: Vec<VectorPoint> = chunks
            .iter()
            .zip(embeddings.into_iter())
            .map(|(chunk, embedding)| {
                let payload = build_chunk_payload(chunk);
                VectorPoint {
                    id: chunk.chunk_id.clone(),
                    embedding,
                    text: chunk.content.clone(),
                    payload,
                }
            })
            .collect();

        // 4. Ensure collection exists, then upsert to vector store.
        // Always enable sparse (BM25) — Qdrant infers it server-side.
        let dims = self.embedder.dimensions();
        let exists = self
            .vector_store
            .collection_exists(&self.collection)
            .await
            .map_err(RetrievalError::Store)?;
        if !exists {
            let config = agent_rs_stores::types::CollectionConfig {
                dense_vector_size: dims,
                enable_sparse: true,
                dense_vector_name: "dense".to_owned(),
                sparse_vector_name: "sparse".to_owned(),
            };
            self.vector_store
                .create_collection(&self.collection, &config)
                .await
                .map_err(RetrievalError::Store)?;
            tracing::info!(
                collection = %self.collection,
                dims,
                enable_sparse = true,
                "created collection for ingestion with server-side BM25"
            );
        }

        let vectors_upserted = vector_points.len();
        self.vector_store
            .upsert(&self.collection, vector_points)
            .await
            .map_err(RetrievalError::Store)?;
        tracing::info!(doc_id, vectors_upserted, "vector upsert complete");

        // 5. Extract entities from chunks.
        //
        // If an `LlmGraphExtractor` is configured, run it in parallel across
        // all chunks and deduplicate entities by `node_id` (cross-document
        // entity linking). Otherwise fall back to the heuristic extractor.
        let (graph_nodes, graph_edges, triplets) = if let Some(extractor) =
            self.llm_extractor.as_ref()
        {
            tracing::debug!(
                chunks = chunks.len(),
                concurrency = self.llm_concurrency,
                "running LLM graph extraction"
            );
            let text_refs: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
            let graphs = extractor
                .extract_batch(&text_refs, self.llm_concurrency)
                .await;

            let (nodes, edges) = build_llm_graph_structures(doc_id, source, &chunks, &graphs)?;
            let triplets = build_triplets(&chunks, &graphs);
            (nodes, edges, triplets)
        } else {
            let all_entities = extract_entities_from_chunks(&chunks);
            let (nodes, edges) = build_graph_structures(doc_id, source, &chunks, &all_entities)?;
            (nodes, edges, Vec::new())
        };

        let graph_nodes_created = graph_nodes.len();
        let graph_edges_created = graph_edges.len();

        if !graph_nodes.is_empty() {
            self.graph_store
                .add_nodes(&graph_nodes)
                .await
                .map_err(RetrievalError::Store)?;
        }

        if !graph_edges.is_empty() {
            self.graph_store
                .add_edges(&graph_edges)
                .await
                .map_err(RetrievalError::Store)?;
        }

        tracing::info!(
            doc_id,
            graph_nodes_created,
            graph_edges_created,
            "graph storage complete"
        );

        // 6. Triplet indexing (optional).
        //
        // When a triplet collection is configured AND triplets were
        // generated (LLM path only), embed each linearised
        // `(source) -> relation -> target` string and upsert them into the
        // dedicated triplet collection. This enables
        // [`crate::search::triplet::triplet_search`] to score complete
        // relations against queries, replicating Cognee's
        // brute_force_triplet_search behaviour.
        let triplets_upserted = self.upsert_triplets_if_enabled(doc_id, &triplets).await?;

        // 7. Chunk summarization (optional).
        //
        // When a `ChunkSummarizer` and a `summaries_collection` are both
        // configured, generate a 1-2 sentence LLM summary for every chunk,
        // embed the summaries, and upsert them into the summaries collection.
        // This powers the `RetrievalStrategy::Summaries` search path.
        let summaries_created = self.upsert_summaries_if_enabled(doc_id, &chunks).await?;

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::info!(doc_id, duration_ms, summaries_created, "ingestion complete");

        Ok(IngestionResult {
            doc_id: doc_id.to_owned(),
            chunks_created,
            vectors_upserted,
            graph_nodes_created,
            graph_edges_created,
            triplets_upserted,
            summaries_created,
            duration_ms,
        })
    }

    /// Embed triplets and upsert them into the configured triplet collection.
    ///
    /// Returns `0` when triplet indexing is disabled (no collection) or when
    /// no triplets were generated. Any storage error is surfaced; embedding
    /// failures are also propagated so the caller knows the collection may
    /// be partially populated.
    #[tracing::instrument(
        skip(self, triplets),
        fields(doc_id, triplet_count = triplets.len())
    )]
    async fn upsert_triplets_if_enabled(
        &self,
        doc_id: &str,
        triplets: &[Triplet],
    ) -> Result<usize, RetrievalError> {
        let Some(triplet_collection) = self.triplet_collection.as_ref() else {
            if !triplets.is_empty() {
                tracing::debug!(
                    doc_id,
                    triplet_count = triplets.len(),
                    "triplets generated but no triplet_collection configured; skipping upsert"
                );
            }
            return Ok(0);
        };
        if triplets.is_empty() {
            return Ok(0);
        }

        // Embed every triplet text via the passage path (same as chunks).
        let triplet_texts: Vec<&str> = triplets.iter().map(|t| t.triplet_text.as_str()).collect();
        let triplet_embeddings = self.embedder.embed_documents(&triplet_texts).await?;

        if triplet_embeddings.len() != triplets.len() {
            return Err(RetrievalError::Ingestion {
                reason: format!(
                    "triplet embedding count mismatch: got {} embeddings for {} triplets",
                    triplet_embeddings.len(),
                    triplets.len()
                ),
            });
        }

        // Ensure the triplet collection exists before upsert. Triplet
        // collections are always dense-only (sparse fusion adds little over
        // already-linearised relation text) so we deliberately disable sparse.
        let dims = self.embedder.dimensions();
        let exists = self
            .vector_store
            .collection_exists(triplet_collection)
            .await
            .map_err(RetrievalError::Store)?;
        if !exists {
            let config = agent_rs_stores::types::CollectionConfig {
                dense_vector_size: dims,
                enable_sparse: false,
                dense_vector_name: "dense".to_owned(),
                sparse_vector_name: "sparse".to_owned(),
            };
            self.vector_store
                .create_collection(triplet_collection, &config)
                .await
                .map_err(RetrievalError::Store)?;
            tracing::info!(
                collection = %triplet_collection,
                dims,
                "created dedicated triplet collection"
            );
        }

        let points: Vec<VectorPoint> = triplets
            .iter()
            .zip(triplet_embeddings.into_iter())
            .map(|(triplet, embedding)| VectorPoint {
                id: triplet.id.clone(),
                embedding,
                // Triplet collections use dense-only search; no BM25 needed.
                text: String::new(),
                payload: build_triplet_payload(triplet),
            })
            .collect();

        let count = points.len();
        self.vector_store
            .upsert(triplet_collection, points)
            .await
            .map_err(RetrievalError::Store)?;
        tracing::info!(
            doc_id,
            collection = %triplet_collection,
            triplets_upserted = count,
            "triplet upsert complete"
        );
        Ok(count)
    }

    /// Generate chunk summaries and upsert them into the configured summaries
    /// collection.
    ///
    /// Returns `0` when summarization is disabled (no summarizer or no
    /// collection) or when there are no chunks. LLM failures for individual
    /// chunks are logged at WARN and silently skipped — the operation is
    /// best-effort; a partial failure never aborts ingestion.
    #[tracing::instrument(
        skip(self, chunks),
        fields(doc_id, chunk_count = chunks.len())
    )]
    async fn upsert_summaries_if_enabled(
        &self,
        doc_id: &str,
        chunks: &[DocumentChunk],
    ) -> Result<usize, RetrievalError> {
        let (Some(summarizer), Some(summaries_collection)) =
            (self.summarizer.as_ref(), self.summaries_collection.as_ref())
        else {
            if !chunks.is_empty() && self.summarizer.is_some() {
                tracing::debug!(
                    doc_id,
                    "summarizer configured but no summaries_collection; skipping"
                );
            }
            return Ok(0);
        };

        if chunks.is_empty() {
            return Ok(0);
        }

        // Build (chunk_id, chunk_text) pairs for the batch call.
        let pairs: Vec<(&str, &str)> = chunks
            .iter()
            .map(|c| (c.chunk_id.as_str(), c.content.as_str()))
            .collect();

        let results = summarizer
            .summarize_batch(&pairs, self.summarizer_concurrency)
            .await;

        // Collect successful summaries, skip empty / failed ones.
        let mut valid_summaries = Vec::with_capacity(results.len());
        for (idx, result) in results.into_iter().enumerate() {
            match result {
                Ok(summary) if !summary.summary_text.is_empty() => {
                    valid_summaries.push(summary);
                }
                Ok(_) => {
                    tracing::debug!(
                        doc_id,
                        chunk_index = idx,
                        "summarizer returned empty text for chunk; skipping"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        doc_id,
                        chunk_index = idx,
                        error = %e,
                        "chunk summarization failed; skipping"
                    );
                }
            }
        }

        if valid_summaries.is_empty() {
            tracing::debug!(doc_id, "no valid summaries produced; skipping upsert");
            return Ok(0);
        }

        // Embed the summary texts via the passage path.
        let summary_texts: Vec<&str> = valid_summaries
            .iter()
            .map(|s| s.summary_text.as_str())
            .collect();
        let summary_embeddings = self.embedder.embed_documents(&summary_texts).await?;

        if summary_embeddings.len() != valid_summaries.len() {
            return Err(RetrievalError::Ingestion {
                reason: format!(
                    "summary embedding count mismatch: got {} embeddings for {} summaries",
                    summary_embeddings.len(),
                    valid_summaries.len()
                ),
            });
        }

        // Ensure the summaries collection exists.
        let dims = self.embedder.dimensions();
        let exists = self
            .vector_store
            .collection_exists(summaries_collection)
            .await
            .map_err(RetrievalError::Store)?;
        if !exists {
            let config = agent_rs_stores::types::CollectionConfig {
                dense_vector_size: dims,
                enable_sparse: false,
                dense_vector_name: "dense".to_owned(),
                sparse_vector_name: "sparse".to_owned(),
            };
            self.vector_store
                .create_collection(summaries_collection, &config)
                .await
                .map_err(RetrievalError::Store)?;
            tracing::info!(
                collection = %summaries_collection,
                dims,
                "created dedicated summaries collection"
            );
        }

        let points: Vec<VectorPoint> = valid_summaries
            .iter()
            .zip(summary_embeddings.into_iter())
            .map(|(summary, embedding)| VectorPoint {
                id: summary.summary_id.clone(),
                embedding,
                // Summaries collections use dense-only search; no BM25 needed.
                text: String::new(),
                payload: build_summary_payload(summary),
            })
            .collect();

        let count = points.len();
        self.vector_store
            .upsert(summaries_collection, points)
            .await
            .map_err(RetrievalError::Store)?;
        tracing::info!(
            doc_id,
            collection = %summaries_collection,
            summaries_upserted = count,
            "summary upsert complete"
        );
        Ok(count)
    }

    /// Ingest a PDF document from raw bytes.
    ///
    /// Routes to the configured [`IngestBackend`]:
    /// - [`IngestBackend::Docling`] — Docling sidecar gRPC path.
    /// - [`IngestBackend::DoclingWithLegacyFallback`] — Docling first; falls back
    ///   to `pdf-extract` when the sidecar reports a retryable or
    ///   `triggers_legacy_fallback` error.
    /// - [`IngestBackend::Legacy`] — `pdf-extract` sync extraction, same as the
    ///   pre-Docling behaviour.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError`] from whichever path is taken.
    #[tracing::instrument(
        skip(self, pdf_bytes),
        fields(doc_id = doc_id, source = source, byte_len = pdf_bytes.len())
    )]
    pub async fn ingest_pdf(
        &self,
        pdf_bytes: &[u8],
        doc_id: &str,
        source: &str,
    ) -> Result<IngestionResult, RetrievalError> {
        match self.ingest_backend {
            IngestBackend::Docling => self.ingest_with_docling(pdf_bytes, doc_id, source).await,
            IngestBackend::DoclingWithLegacyFallback => {
                match self.ingest_with_docling(pdf_bytes, doc_id, source).await {
                    Ok(result) => Ok(result),
                    Err(RetrievalError::DoclingIngestFailed {
                        triggers_legacy_fallback: true,
                        ref message,
                        ..
                    }) => {
                        tracing::warn!(
                            doc_id,
                            reason = %message,
                            "Docling ingest failed with legacy-fallback flag; switching to pdf-extract"
                        );
                        self.ingest_pdf_legacy(pdf_bytes, doc_id, source).await
                    }
                    Err(other) => Err(other),
                }
            }
            IngestBackend::Legacy => self.ingest_pdf_legacy(pdf_bytes, doc_id, source).await,
        }
    }

    /// Ingest a document via the Docling sidecar gRPC streaming path.
    ///
    /// Full 8-step state machine:
    ///
    /// 1. Compute sha256 of the raw bytes.
    /// 2. Dedup check via Postgres (`begin_ingestion`).
    /// 3. Call `docling_client.chunk_document()`.
    /// 4. Embed chunks, build 27-field Qdrant payload tagged with `ingest_run_id`, upsert.
    /// 5. Neo4j structural upsert (doc node, chunk nodes, entity edges) tagged with `ingest_run_id`.
    /// 6. LLM KG extraction + triplet indexing (if configured).
    /// 7. Chunk summarization (if configured).
    /// 8. Mark Postgres row `complete` (3-retry exponential backoff at 1s, 2s, 4s).
    ///
    /// On failure after step 2, the Postgres row is marked `partial_failure` and
    /// the caller may invoke `IngestCoordinator::cleanup_partial` to roll back
    /// Qdrant and Neo4j by `ingest_run_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::DoclingIngestFailed`] when the sidecar rejects
    /// the document, or various [`RetrievalError`] variants on embedding /
    /// store failures.
    #[tracing::instrument(
        skip(self, pdf_bytes),
        fields(doc_id = doc_id, source = source, byte_len = pdf_bytes.len())
    )]
    async fn ingest_with_docling(
        &self,
        pdf_bytes: &[u8],
        doc_id: &str,
        source: &str,
    ) -> Result<IngestionResult, RetrievalError> {
        use backon::{ExponentialBuilder, Retryable};
        use sha2::{Digest, Sha256};
        use std::time::Duration;

        let start = Instant::now();

        // ---- Step 1: sha256 ----
        let sha256 = {
            let mut hasher = Sha256::new();
            hasher.update(pdf_bytes);
            format!("{:x}", hasher.finalize())
        };

        // ---- Precondition: PgStore required for Docling path ----
        //
        // Without PgStore, orphan ingest_run_id tags accumulate in Qdrant/Neo4j
        // with no way to clean them up, and the dedup / partial-failure state
        // machine is silently skipped.
        if self.pg_store.is_none() {
            return Err(RetrievalError::Ingestion {
                reason: "Docling ingest path requires PgStore for state machine — \
                         wire with_pg_store() on IngestionPipelineConfig"
                    .to_owned(),
            });
        }

        // ---- Step 2: Dedup via Postgres ----
        let maybe_run_id: Option<Uuid> = if let Some(pg) = &self.pg_store {
            let now = Utc::now().naive_utc();
            let new_doc = NewIngestedDocument {
                document_id: Uuid::new_v5(&NAMESPACE_OID, sha256.as_bytes()),
                sha256: sha256.clone(),
                source: source.to_owned(),
                ingest_status: agent_rs_core::IngestStatus::InProgress.as_str().to_owned(),
                ingest_run_id: Uuid::new_v4(),
                started_at: now,
                updated_at: now,
            };

            match pg
                .begin_ingestion(new_doc)
                .await
                .map_err(RetrievalError::Store)?
            {
                BeginIngestionOutcome::AlreadyComplete => {
                    tracing::info!(doc_id, sha256 = %sha256, "document already complete; skipping");
                    return Ok(IngestionResult {
                        doc_id: doc_id.to_owned(),
                        chunks_created: 0,
                        vectors_upserted: 0,
                        graph_nodes_created: 0,
                        graph_edges_created: 0,
                        triplets_upserted: 0,
                        summaries_created: 0,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                BeginIngestionOutcome::AlreadyBeingIngested => {
                    return Err(RetrievalError::IngestInProgress { sha256 });
                }
                BeginIngestionOutcome::Started { ingest_run_id }
                | BeginIngestionOutcome::Resumed { ingest_run_id } => {
                    tracing::info!(doc_id, %ingest_run_id, "ingestion started/resumed");
                    Some(ingest_run_id)
                }
            }
        } else {
            None
        };

        let ingest_run_id = maybe_run_id.unwrap_or_else(Uuid::new_v4);

        // ---- Step 3: Call Docling sidecar ----
        let docling_response = {
            let client_mutex =
                self.docling_client
                    .as_ref()
                    .ok_or_else(|| RetrievalError::Ingestion {
                        reason: "ingest_backend is Docling but no DoclingSidecarClient configured"
                            .to_owned(),
                    })?;
            let mut client = client_mutex.lock().await;
            client
                .chunk_document(pdf_bytes.to_vec(), doc_id, source, "application/pdf")
                .await
        };

        let docling_response = match docling_response {
            Ok(r) => r,
            Err(e) => {
                // Best-effort: mark partial failure in Postgres.
                if let Some(pg) = &self.pg_store {
                    if let Err(pg_err) =
                        pg.mark_partial_failure(ingest_run_id, &e.to_string()).await
                    {
                        tracing::warn!(
                            doc_id,
                            error = %pg_err,
                            "failed to mark partial failure in Postgres after Docling error"
                        );
                    }
                }
                return Err(e);
            }
        };

        let header = &docling_response.header;
        tracing::info!(
            doc_id,
            chunks = docling_response.chunks.len(),
            n_pages = header.n_pages,
            "Docling sidecar response received"
        );

        // ---- Step 4: Embed + Qdrant upsert (27-field payload, tagged with ingest_run_id) ----
        let embed_texts: Vec<&str> = docling_response
            .chunks
            .iter()
            .map(|c| c.embed_text.as_str())
            .collect();

        let embeddings = match self.embedder.embed_documents(&embed_texts).await {
            Ok(e) => e,
            Err(e) => {
                if let Some(pg) = &self.pg_store {
                    if let Err(pg_err) =
                        pg.mark_partial_failure(ingest_run_id, &e.to_string()).await
                    {
                        tracing::warn!(doc_id, error = %pg_err, "mark_partial_failure failed after embedding error");
                    }
                }
                return Err(e);
            }
        };

        if embeddings.len() != docling_response.chunks.len() {
            let reason = format!(
                "embedding count mismatch: got {} for {} chunks",
                embeddings.len(),
                docling_response.chunks.len()
            );
            if let Some(pg) = &self.pg_store {
                if let Err(pg_err) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                    tracing::warn!(error = %pg_err, %ingest_run_id, "mark_partial_failure failed — row may be stuck in_progress");
                }
            }
            return Err(RetrievalError::Ingestion { reason });
        }

        let total_chunks = docling_response.chunks.len();

        let doc_n_tables = docling_response.footer.n_tables;
        let doc_n_pictures = docling_response.footer.n_pictures;

        let vector_points: Vec<VectorPoint> = docling_response
            .chunks
            .iter()
            .zip(embeddings.into_iter())
            .map(|(chunk, embedding)| {
                let payload = build_docling_chunk_payload(
                    chunk,
                    header,
                    doc_id,
                    source,
                    ingest_run_id,
                    doc_n_tables,
                    doc_n_pictures,
                );
                VectorPoint {
                    id: chunk.chunk_id.clone(),
                    embedding,
                    // raw_text is what Qdrant's server-side BM25 tokenises.
                    text: chunk.raw_text.clone(),
                    payload,
                }
            })
            .collect();

        // Ensure collection exists.
        let dims = self.embedder.dimensions();
        match self.vector_store.collection_exists(&self.collection).await {
            Ok(false) => {
                let config = agent_rs_stores::types::CollectionConfig {
                    dense_vector_size: dims,
                    enable_sparse: true,
                    dense_vector_name: "dense".to_owned(),
                    sparse_vector_name: "sparse".to_owned(),
                };
                if let Err(e) = self
                    .vector_store
                    .create_collection(&self.collection, &config)
                    .await
                {
                    let reason = e.to_string();
                    if let Some(pg) = &self.pg_store {
                        if let Err(pg_err) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                            tracing::warn!(error = %pg_err, %ingest_run_id, "mark_partial_failure failed — row may be stuck in_progress");
                        }
                    }
                    return Err(RetrievalError::Store(e));
                }
            }
            Ok(true) => {}
            Err(e) => {
                let reason = e.to_string();
                if let Some(pg) = &self.pg_store {
                    if let Err(pg_err) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                        tracing::warn!(error = %pg_err, %ingest_run_id, "mark_partial_failure failed — row may be stuck in_progress");
                    }
                }
                return Err(RetrievalError::Store(e));
            }
        }

        let vectors_upserted = vector_points.len();
        if let Err(e) = self
            .vector_store
            .upsert(&self.collection, vector_points)
            .await
        {
            let reason = e.to_string();
            if let Some(pg) = &self.pg_store {
                if let Err(pg_err) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                    tracing::warn!(error = %pg_err, %ingest_run_id, "mark_partial_failure failed — row may be stuck in_progress");
                }
            }
            return Err(RetrievalError::Store(e));
        }
        tracing::info!(doc_id, vectors_upserted, "Docling: vector upsert complete");

        // ---- Step 5: Neo4j structural upsert (tagged with ingest_run_id) ----
        //
        // Reuse the LLM path if an extractor is configured; otherwise heuristic.
        // Both paths produce `DocumentChunk`-shaped data, so we convert the proto
        // chunks first.
        let pseudo_chunks: Vec<DocumentChunk> = docling_response
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| proto_chunk_to_document_chunk(c, doc_id, source, total_chunks, i))
            .collect();

        let ingest_run_id_str = ingest_run_id.to_string();

        let (graph_nodes, graph_edges, triplets) = if let Some(extractor) =
            self.llm_extractor.as_ref()
        {
            let text_refs: Vec<&str> = pseudo_chunks.iter().map(|c| c.content.as_str()).collect();
            let graphs = extractor
                .extract_batch(&text_refs, self.llm_concurrency)
                .await;
            let (mut nodes, edges) =
                build_llm_graph_structures(doc_id, source, &pseudo_chunks, &graphs)?;
            let triplets = build_triplets(&pseudo_chunks, &graphs);
            // Tag every node with ingest_run_id for rollback.
            for node in &mut nodes {
                if let Some(obj) = node.properties.as_object_mut() {
                    obj.insert(
                        "ingest_run_id".to_owned(),
                        serde_json::Value::String(ingest_run_id_str.clone()),
                    );
                }
            }
            (nodes, edges, triplets)
        } else {
            let all_entities = extract_entities_from_chunks(&pseudo_chunks);
            let (mut nodes, edges) =
                build_graph_structures(doc_id, source, &pseudo_chunks, &all_entities)?;
            for node in &mut nodes {
                if let Some(obj) = node.properties.as_object_mut() {
                    obj.insert(
                        "ingest_run_id".to_owned(),
                        serde_json::Value::String(ingest_run_id_str.clone()),
                    );
                }
            }
            (nodes, edges, Vec::new())
        };

        let graph_nodes_created = graph_nodes.len();
        let graph_edges_created = graph_edges.len();

        let graph_result: Result<(), RetrievalError> = async {
            if !graph_nodes.is_empty() {
                self.graph_store
                    .add_nodes(&graph_nodes)
                    .await
                    .map_err(RetrievalError::Store)?;
            }
            if !graph_edges.is_empty() {
                self.graph_store
                    .add_edges(&graph_edges)
                    .await
                    .map_err(RetrievalError::Store)?;
            }
            Ok(())
        }
        .await;

        if let Err(e) = graph_result {
            let reason = e.to_string();
            // Rollback: Qdrant has data from step 4, Neo4j may have partial entity data.
            self.try_rollback_stores(doc_id, ingest_run_id).await;
            if let Some(pg) = &self.pg_store {
                if let Err(me) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                    tracing::warn!(?me, ingest_run_id = %ingest_run_id, "failed to mark partial failure");
                }
            }
            return Err(e);
        }
        tracing::info!(
            doc_id,
            graph_nodes_created,
            graph_edges_created,
            "Docling: entity graph upsert complete"
        );

        // ---- Step 5b: Neo4j STRUCTURAL upsert (Document / Page / Section / Table / Picture) ----
        //
        // Runs AFTER the entity graph step so that pre-existing Chunk nodes
        // (created by `build_llm_graph_structures`) exist when the structural
        // layer enriches them with page_start/page_end/section_id/has_* props.
        // All new nodes are tagged with `ingest_run_id` so rollback by run_id
        // can target exactly this ingestion without touching pre-existing data.
        let structural_result: Result<(), RetrievalError> = async {
            let snapshot =
                proto_header_to_snapshot(header, &docling_response.footer, ingest_run_id, source);
            let doc_uuid = snapshot.document_id;

            // Build ordered StructuralChunk list with FOLLOWS back-pointers.
            let mut structural_chunks: Vec<StructuralChunk> =
                Vec::with_capacity(docling_response.chunks.len());
            let mut prev: Option<Uuid> = None;
            for proto_chunk in &docling_response.chunks {
                let sc = proto_chunk_to_structural(proto_chunk, doc_uuid, prev);
                prev = Some(sc.chunk_id);
                structural_chunks.push(sc);
            }

            let structural_tables = extract_tables_from_chunks(doc_uuid, &docling_response.chunks);
            let structural_pictures =
                extract_pictures_from_chunks(doc_uuid, &docling_response.chunks);

            self.graph_store
                .upsert_document_structure(
                    &snapshot,
                    &structural_chunks,
                    &structural_tables,
                    &structural_pictures,
                )
                .await
                .map_err(RetrievalError::Store)
        }
        .await;

        if let Err(e) = structural_result {
            let reason = e.to_string();
            // Rollback: Qdrant has step 4 data + Neo4j has entity data from step 5.
            self.try_rollback_stores(doc_id, ingest_run_id).await;
            if let Some(pg) = &self.pg_store {
                if let Err(me) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                    tracing::warn!(?me, ingest_run_id = %ingest_run_id, "failed to mark partial failure");
                }
            }
            return Err(e);
        }
        tracing::info!(doc_id, "Docling: structural graph upsert complete");

        // ---- Step 6: Triplet indexing (optional) ----
        let triplets_upserted = match self.upsert_triplets_if_enabled(doc_id, &triplets).await {
            Ok(n) => n,
            Err(e) => {
                let reason = e.to_string();
                if let Some(pg) = &self.pg_store {
                    if let Err(pg_err) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                        tracing::warn!(error = %pg_err, %ingest_run_id, "mark_partial_failure failed — row may be stuck in_progress");
                    }
                }
                return Err(e);
            }
        };

        // ---- Step 7: Chunk summarization (optional) ----
        let summaries_created = match self
            .upsert_summaries_if_enabled(doc_id, &pseudo_chunks)
            .await
        {
            Ok(n) => n,
            Err(e) => {
                let reason = e.to_string();
                if let Some(pg) = &self.pg_store {
                    if let Err(pg_err) = pg.mark_partial_failure(ingest_run_id, &reason).await {
                        tracing::warn!(error = %pg_err, %ingest_run_id, "mark_partial_failure failed — row may be stuck in_progress");
                    }
                }
                return Err(e);
            }
        };

        // ---- Step 8: Mark Postgres row complete (3-retry exponential backoff) ----
        if let Some(pg) = &self.pg_store {
            let total_ingest_ms = start.elapsed().as_millis() as i64;
            let header_meta = HeaderMetadata {
                docling_version: header.docling_version.clone(),
                docling_core_version: header.docling_core_version.clone(),
                schema_version: header.schema_version as i32,
                n_pages: header.n_pages as i32,
                n_chunks: docling_response.footer.n_chunks as i32,
                n_tables: docling_response.footer.n_tables as i32,
                n_pictures: docling_response.footer.n_pictures as i32,
                n_formulas: docling_response.footer.n_formulas as i32,
                language: if header.language.is_empty() {
                    None
                } else {
                    Some(header.language.clone())
                },
                title: if header.title.is_empty() {
                    None
                } else {
                    Some(header.title.clone())
                },
                mimetype: if header.mimetype.is_empty() {
                    None
                } else {
                    Some(header.mimetype.clone())
                },
                parse_latency_ms: Some(header.parse_latency_ms as i64),
                total_ingest_latency_ms: Some(total_ingest_ms),
            };

            let pg_ref = Arc::clone(pg);
            let mark_complete =
                || async { pg_ref.mark_complete(ingest_run_id, &header_meta).await };

            if let Err(e) = mark_complete
                .retry(
                    ExponentialBuilder::default()
                        .with_min_delay(Duration::from_secs(1))
                        .with_max_delay(Duration::from_secs(4))
                        .with_max_times(3),
                )
                .await
            {
                return Err(RetrievalError::PostgresCommitFailed {
                    ingest_run_id: ingest_run_id.to_string(),
                    reason: e.to_string(),
                });
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            doc_id,
            duration_ms,
            vectors_upserted,
            graph_nodes_created,
            "Docling ingestion complete"
        );

        Ok(IngestionResult {
            doc_id: doc_id.to_owned(),
            chunks_created: total_chunks,
            vectors_upserted,
            graph_nodes_created,
            graph_edges_created,
            triplets_upserted,
            summaries_created,
            duration_ms,
        })
    }

    /// Ingest a PDF document via the legacy `pdf-extract` path.
    ///
    /// Extracts text using `pdf-extract`, then delegates to [`Self::ingest_text`].
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::PdfExtraction`] if the PDF cannot be parsed.
    #[cfg(feature = "legacy_pdf_extract")]
    #[tracing::instrument(
        skip(self, pdf_bytes),
        fields(doc_id = doc_id, source = source, byte_len = pdf_bytes.len())
    )]
    async fn ingest_pdf_legacy(
        &self,
        pdf_bytes: &[u8],
        doc_id: &str,
        source: &str,
    ) -> Result<IngestionResult, RetrievalError> {
        // pdf-extract is synchronous and CPU-bound; offload to blocking pool.
        let bytes = pdf_bytes.to_vec();
        let text = tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text_from_mem(&bytes).map_err(|e| RetrievalError::PdfExtraction {
                reason: e.to_string(),
            })
        })
        .await
        .map_err(|join_err| RetrievalError::PdfExtraction {
            reason: format!("PDF extraction task panicked: {join_err}"),
        })??;

        tracing::info!(
            doc_id,
            extracted_chars = text.len(),
            "PDF text extraction complete (legacy path)"
        );

        self.ingest_text(&text, doc_id, source).await
    }

    /// Stub for the legacy path when the `legacy_pdf_extract` feature is disabled.
    ///
    /// Callers that select [`IngestBackend::Legacy`] or
    /// [`IngestBackend::DoclingWithLegacyFallback`] without enabling the
    /// feature will receive a clear error rather than a silent no-op.
    #[cfg(not(feature = "legacy_pdf_extract"))]
    async fn ingest_pdf_legacy(
        &self,
        _pdf_bytes: &[u8],
        _doc_id: &str,
        _source: &str,
    ) -> Result<IngestionResult, RetrievalError> {
        Err(RetrievalError::Ingestion {
            reason: "legacy_pdf_extract feature is not enabled; recompile with \
                     --features legacy_pdf_extract to use the pdf-extract backend"
                .to_owned(),
        })
    }

    /// Best-effort rollback of Qdrant + Neo4j data tagged with `ingest_run_id`.
    ///
    /// Called in failure paths after Qdrant/Neo4j writes have occurred.  Logs
    /// warnings on individual rollback failures but never propagates errors —
    /// the caller's original error takes priority.
    async fn try_rollback_stores(&self, doc_id: &str, ingest_run_id: Uuid) {
        let run_id_str = ingest_run_id.to_string();

        // Qdrant: delete points by ingest_run_id payload filter.
        let filter = agent_rs_stores::types::FilterSpec::eq("ingest_run_id", &run_id_str);
        match self
            .vector_store
            .delete_by_filter(&self.collection, filter)
            .await
        {
            Ok(n) => {
                tracing::info!(doc_id, %ingest_run_id, qdrant_deleted = n, "rollback: Qdrant points deleted")
            }
            Err(e) => {
                tracing::warn!(doc_id, %ingest_run_id, error = %e, "rollback: Qdrant delete_by_filter failed")
            }
        }

        // Neo4j: delete structural + entity nodes by ingest_run_id.
        match self.graph_store.rollback_by_run_id(ingest_run_id).await {
            Ok(n) => {
                tracing::info!(doc_id, %ingest_run_id, neo4j_deleted = n, "rollback: Neo4j nodes deleted")
            }
            Err(e) => {
                tracing::warn!(doc_id, %ingest_run_id, error = %e, "rollback: Neo4j rollback_by_run_id failed")
            }
        }
    }
}

// ---- Private helpers ----

/// Build the JSON payload for a chunk to be stored in Qdrant.
fn build_chunk_payload(chunk: &DocumentChunk) -> serde_json::Value {
    json!({
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_text": chunk.content,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "source": chunk.metadata.source,
        "section_title": chunk.metadata.section_title,
        "headings": chunk.metadata.headings,
        "has_table": chunk.metadata.has_table,
        "has_list": chunk.metadata.has_list,
        "language": chunk.metadata.language,
        "char_count": chunk.metadata.char_count,
        "word_count": chunk.metadata.word_count,
    })
}

/// Build the 27-field JSON payload for a Docling-sourced chunk stored in Qdrant.
///
/// The payload carries all filterable/retrievable metadata so downstream
/// evidence assembly can render citations without a second round-trip.
/// Every point is also tagged with `ingest_run_id` for rollback.
///
/// `doc_n_tables` / `doc_n_pictures` come from the stream footer, which the
/// caller has already received in full before this function is called.
#[allow(clippy::too_many_arguments)]
fn build_docling_chunk_payload(
    chunk: &ProtoChunk,
    header: &ProtoDocumentHeader,
    doc_id: &str,
    source: &str,
    ingest_run_id: Uuid,
    doc_n_tables: u32,
    doc_n_pictures: u32,
) -> serde_json::Value {
    json!({
        // ---- identity ----
        "chunk_id":       chunk.chunk_id,
        "doc_id":         doc_id,
        "ingest_run_id":  ingest_run_id.to_string(),
        // ---- content ----
        "chunk_text":     chunk.embed_text,
        "raw_text":       chunk.raw_text,
        // ---- position ----
        "chunk_index":    chunk.chunk_index,
        "total_chunks":   chunk.total_chunks,
        "page_start":     chunk.page_start,
        "page_end":       chunk.page_end,
        // ---- provenance ----
        "source":         source,
        "title":          header.title,
        "language":       header.language,
        "mimetype":       header.mimetype,
        // ---- structure ----
        "section_title":  chunk.section_title,
        "section_id":     chunk.section_id,
        "heading_path":   chunk.heading_path,
        // ---- content flags ----
        "has_table":      chunk.has_table,
        "has_picture":    chunk.has_picture,
        "has_code":       chunk.has_code,
        "has_formula":    chunk.has_formula,
        // ---- counts ----
        "char_count":     chunk.char_count,
        "token_count":    chunk.token_count,
        // ---- doc-level counts (from footer — available because the full stream
        //      was received before payload construction begins) ----
        "doc_n_pages":    header.n_pages,
        "doc_n_tables":   doc_n_tables,
        "doc_n_pictures": doc_n_pictures,
        // ---- sidecar meta ----
        "docling_version": header.docling_version,
        "schema_version":  header.schema_version,
    })
}

/// Convert a proto [`ProtoChunk`] into the lightweight [`DocumentChunk`] type
/// used by the graph extraction and summarization helpers.
///
/// The `DocumentChunk` was originally designed for the legacy text path;
/// this conversion maps the proto fields onto the closest equivalents so
/// existing helpers (`build_llm_graph_structures`, `upsert_summaries_if_enabled`,
/// etc.) can be reused without modification.
fn proto_chunk_to_document_chunk(
    chunk: &ProtoChunk,
    doc_id: &str,
    source: &str,
    total_chunks: usize,
    index: usize,
) -> DocumentChunk {
    use super::metadata::ChunkMetadata;

    let section_title = if chunk.section_title.is_empty() {
        None
    } else {
        Some(chunk.section_title.clone())
    };

    // heading_path contains the full heading hierarchy root-first.
    let headings: Vec<String> = chunk.heading_path.clone();

    DocumentChunk {
        chunk_id: chunk.chunk_id.clone(),
        doc_id: doc_id.to_owned(),
        content: chunk.embed_text.clone(),
        chunk_index: index,
        total_chunks,
        metadata: ChunkMetadata {
            source: source.to_owned(),
            page_number: if chunk.page_start > 0 {
                Some(chunk.page_start as usize)
            } else {
                None
            },
            section_title,
            headings,
            has_table: chunk.has_table,
            has_list: false, // not surfaced by proto
            language: None,  // language is on DocumentHeader, not per-chunk
            char_count: chunk.char_count as usize,
            word_count: 0, // not surfaced by proto directly
            token_count: chunk.token_count as usize,
            byte_start: 0,
            byte_end: 0,
            paragraph_ids: Vec::new(),
        },
    }
}

/// Build the JSON payload for a summary point stored in the summaries collection.
fn build_summary_payload(summary: &super::summarizer::ChunkSummary) -> serde_json::Value {
    json!({
        "summary_id": summary.summary_id,
        "chunk_id": summary.chunk_id,
        "summary_text": summary.summary_text,
        "chunk_text_preview": summary.chunk_text_preview,
    })
}

// ---- Structural layer helpers (Docling metadata -> Neo4j structural nodes) ----

/// Build a `DocumentHeaderSnapshot` from a proto `DocumentHeader` + `DocumentFooter`.
///
/// The document UUID is derived deterministically from the sha256 via UUIDv5
/// so the same source bytes always produce the same document node.
fn proto_header_to_snapshot(
    header: &ProtoDocumentHeader,
    footer: &ProtoDocumentFooter,
    ingest_run_id: Uuid,
    source: &str,
) -> DocumentHeaderSnapshot {
    let document_id = Uuid::new_v5(&NAMESPACE_OID, header.sha256.as_bytes());

    let pages: Vec<StructuralPageInfo> = header
        .pages
        .iter()
        .map(|p| StructuralPageInfo {
            page_no: p.page_no as i64,
            width: p.width,
            height: p.height,
        })
        .collect();

    let none_if_empty = |s: &str| {
        if s.is_empty() {
            None
        } else {
            Some(s.to_owned())
        }
    };

    DocumentHeaderSnapshot {
        document_id,
        ingest_run_id,
        sha256: header.sha256.clone(),
        source: source.to_owned(),
        docling_version: none_if_empty(&header.docling_version),
        n_pages: header.n_pages as i64,
        n_chunks: footer.n_chunks as i64,
        n_tables: footer.n_tables as i64,
        n_pictures: footer.n_pictures as i64,
        language: none_if_empty(&header.language),
        title: none_if_empty(&header.title),
        mimetype: none_if_empty(&header.mimetype),
        pages,
    }
}

/// Convert one proto `Chunk` into a `StructuralChunk` for the Neo4j layer.
///
/// `prev_chunk_id` is the UUID of the previous chunk in the same document
/// (`None` for the first chunk) — used to seed the `FOLLOWS` edge.
fn proto_chunk_to_structural(
    chunk: &ProtoChunk,
    document_id: Uuid,
    prev_chunk_id: Option<Uuid>,
) -> StructuralChunk {
    let chunk_uuid = Uuid::parse_str(&chunk.chunk_id)
        .unwrap_or_else(|_| Uuid::new_v5(&NAMESPACE_OID, chunk.chunk_id.as_bytes()));

    let section_id = if chunk.section_id.is_empty() {
        None
    } else {
        Uuid::parse_str(&chunk.section_id).ok()
    };

    let section_title = if chunk.section_title.is_empty() {
        None
    } else {
        Some(chunk.section_title.clone())
    };

    // Heading level = depth of the heading path (0 if no path).
    let heading_level = if chunk.heading_path.is_empty() {
        None
    } else {
        Some(chunk.heading_path.len() as i64)
    };

    // Derive the presence of list / footnote / caption / picture-ref flags
    // from doc_items since the top-level Chunk proto only exposes table /
    // picture / code / formula directly.
    let mut has_list = false;
    let mut has_footnote = false;
    let mut has_caption = false;
    let mut has_picture_ref = false;
    let mut table_refs: Vec<Uuid> = Vec::new();
    let mut picture_refs: Vec<Uuid> = Vec::new();
    for item in &chunk.doc_items {
        match DocItemLabel::try_from(item.label) {
            Ok(DocItemLabel::ListItem) => has_list = true,
            Ok(DocItemLabel::Footnote) => has_footnote = true,
            Ok(DocItemLabel::Caption) => has_caption = true,
            Ok(DocItemLabel::Table) => {
                table_refs.push(doc_item_to_uuid(document_id, &item.self_ref));
            }
            Ok(DocItemLabel::Picture) => {
                has_picture_ref = true;
                picture_refs.push(doc_item_to_uuid(document_id, &item.self_ref));
            }
            _ => {}
        }
    }

    StructuralChunk {
        chunk_id: chunk_uuid,
        page_start: chunk.page_start as i64,
        page_end: chunk.page_end as i64,
        section_id,
        section_title,
        heading_level,
        heading_path: chunk.heading_path.clone(),
        has_table: chunk.has_table,
        has_list,
        has_code: chunk.has_code,
        has_formula: chunk.has_formula,
        has_picture_ref,
        has_footnote,
        has_caption,
        table_refs,
        picture_refs,
        prev_chunk_id,
    }
}

/// Extract the set of unique `StructuralTable` records from all chunks.
///
/// Tables referenced from multiple chunks produce a single record, identified
/// by the stable UUIDv5 derived from (document_id, self_ref).
fn extract_tables_from_chunks(document_id: Uuid, chunks: &[ProtoChunk]) -> Vec<StructuralTable> {
    use std::collections::BTreeMap;
    let mut by_ref: BTreeMap<String, StructuralTable> = BTreeMap::new();
    for chunk in chunks {
        for item in &chunk.doc_items {
            if DocItemLabel::try_from(item.label) != Ok(DocItemLabel::Table) {
                continue;
            }
            if by_ref.contains_key(&item.self_ref) {
                continue;
            }
            by_ref.insert(
                item.self_ref.clone(),
                StructuralTable {
                    table_id: doc_item_to_uuid(document_id, &item.self_ref),
                    doc_item_self_ref: item.self_ref.clone(),
                    page_no: item.page_no as i64,
                    n_rows: item.table_n_rows as i64,
                    n_cols: item.table_n_cols as i64,
                    caption: None, // caption ends up as a separate CAPTION doc_item
                },
            );
        }
    }
    by_ref.into_values().collect()
}

/// Extract the set of unique `StructuralPicture` records from all chunks.
fn extract_pictures_from_chunks(
    document_id: Uuid,
    chunks: &[ProtoChunk],
) -> Vec<StructuralPicture> {
    use std::collections::BTreeMap;
    let mut by_ref: BTreeMap<String, StructuralPicture> = BTreeMap::new();
    for chunk in chunks {
        for item in &chunk.doc_items {
            if DocItemLabel::try_from(item.label) != Ok(DocItemLabel::Picture) {
                continue;
            }
            if by_ref.contains_key(&item.self_ref) {
                continue;
            }
            let caption = if item.picture_caption.is_empty() {
                None
            } else {
                Some(item.picture_caption.clone())
            };
            by_ref.insert(
                item.self_ref.clone(),
                StructuralPicture {
                    picture_id: doc_item_to_uuid(document_id, &item.self_ref),
                    doc_item_self_ref: item.self_ref.clone(),
                    page_no: item.page_no as i64,
                    caption,
                },
            );
        }
    }
    by_ref.into_values().collect()
}

/// Deterministic UUIDv5 for a `DocItem` identified by its `self_ref` within a
/// document.
fn doc_item_to_uuid(document_id: Uuid, self_ref: &str) -> Uuid {
    let seed = format!("{document_id}:{self_ref}");
    Uuid::new_v5(&NAMESPACE_OID, seed.as_bytes())
}

/// Build the JSON payload for a triplet stored in the triplet collection.
///
/// The payload carries the full triplet (names, ids, chunk provenance) so
/// downstream evidence assembly can render `subject predicate object`
/// citations without a second round-trip to fetch the nodes.
fn build_triplet_payload(triplet: &Triplet) -> serde_json::Value {
    json!({
        "triplet_id": triplet.id,
        "source_node_id": triplet.source_node_id,
        "source_node_name": triplet.source_node_name,
        "relationship_name": triplet.relationship_name,
        "target_node_id": triplet.target_node_id,
        "target_node_name": triplet.target_node_name,
        "chunk_id": triplet.chunk_id,
        "triplet_text": triplet.triplet_text,
    })
}

/// Project LLM-extracted knowledge graphs into a deduplicated list of
/// [`Triplet`]s suitable for vector indexing.
///
/// The algorithm mirrors the Cognee ingestion pipeline: for every
/// `(source_node_id, relationship, target_node_id)` emitted by the LLM we
/// emit one triplet, carrying the originating chunk id as provenance.
/// Duplicates across chunks are collapsed by the deterministic
/// [`Triplet`]-id (UUID v5 over the tuple), so repeated ingestion is
/// idempotent at the vector-store level.
///
/// Edges whose source/target entity names cannot be resolved from the
/// per-chunk graph are dropped with a debug trace — we never index a
/// dangling triplet.
fn build_triplets(chunks: &[DocumentChunk], graphs: &[KnowledgeGraph]) -> Vec<Triplet> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<Triplet> = Vec::new();

    for (i, graph) in graphs.iter().enumerate() {
        let Some(chunk) = chunks.get(i) else {
            continue;
        };

        // Index nodes by id for name lookup.
        let name_by_id: HashMap<&str, &str> = graph
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), n.name.as_str()))
            .collect();

        for edge in &graph.edges {
            let Some(source_name) = name_by_id.get(edge.source_node_id.as_str()) else {
                tracing::debug!(
                    src = %edge.source_node_id,
                    dst = %edge.target_node_id,
                    "dropping triplet with unknown source node"
                );
                continue;
            };
            let Some(target_name) = name_by_id.get(edge.target_node_id.as_str()) else {
                tracing::debug!(
                    src = %edge.source_node_id,
                    dst = %edge.target_node_id,
                    "dropping triplet with unknown target node"
                );
                continue;
            };

            let rel = sanitize_relationship(&edge.relationship_name);
            if rel.is_empty() {
                continue;
            }

            let triplet = Triplet::from_parts(
                &edge.source_node_id,
                *source_name,
                &rel,
                &edge.target_node_id,
                *target_name,
                &chunk.chunk_id,
            );

            if seen.insert(triplet.id.clone()) {
                out.push(triplet);
            }
        }
    }

    out
}

/// A named entity extracted from text (Phase 1: heuristic-based).
struct ExtractedEntity {
    /// Normalised entity name.
    name: String,
    /// Entity type label (e.g., "ENTITY", "ACRONYM").
    entity_type: String,
}

/// Extract named entities from chunks using heuristics.
///
/// Phase 1 implementation: detects capitalised multi-word phrases and
/// acronyms. Phase 2 will add LLM-based extraction.
fn extract_entities_from_chunks(chunks: &[DocumentChunk]) -> Vec<Vec<ExtractedEntity>> {
    // Pre-compile regex patterns once.
    // Capitalised multi-word phrase: "United States", "Machine Learning", etc.
    let cap_phrase_re =
        Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b").expect("valid regex: cap_phrase");
    // Acronyms: 2+ uppercase letters, optionally with dots.
    let acronym_re = Regex::new(r"\b([A-Z]{2,}(?:\.[A-Z]+)*)\b").expect("valid regex: acronym");

    chunks
        .iter()
        .map(|chunk| {
            let mut entities: Vec<ExtractedEntity> = Vec::new();
            let mut seen = std::collections::HashSet::new();

            // Capitalised multi-word phrases.
            for cap in cap_phrase_re.captures_iter(&chunk.content) {
                if let Some(m) = cap.get(1) {
                    let name = m.as_str().to_owned();
                    if seen.insert(name.clone()) {
                        entities.push(ExtractedEntity {
                            name,
                            entity_type: "ENTITY".to_owned(),
                        });
                    }
                }
            }

            // Acronyms (filter common English words that look like acronyms).
            let common_words = ["THE", "AND", "FOR", "NOT", "BUT", "NOR", "YET", "SO"];
            for cap in acronym_re.captures_iter(&chunk.content) {
                if let Some(m) = cap.get(1) {
                    let name = m.as_str().to_owned();
                    if !common_words.contains(&name.as_str()) && seen.insert(name.clone()) {
                        entities.push(ExtractedEntity {
                            name,
                            entity_type: "ACRONYM".to_owned(),
                        });
                    }
                }
            }

            entities
        })
        .collect()
}

/// Build graph nodes and edges for the ingested document.
///
/// Creates:
/// - 1 Document node
/// - N Chunk nodes
/// - M Entity nodes (deduplicated across chunks)
/// - Document -[HAS_CHUNK]-> Chunk edges
/// - Chunk -[NEXT]-> Chunk edges (ordering)
/// - Chunk -[MENTIONS]-> Entity edges
fn build_graph_structures(
    doc_id: &str,
    source: &str,
    chunks: &[DocumentChunk],
    entities_per_chunk: &[Vec<ExtractedEntity>],
) -> Result<(Vec<GraphNode>, Vec<GraphEdge>), RetrievalError> {
    let map_label_err = |e| RetrievalError::Ingestion {
        reason: format!("invalid graph label: {e}"),
    };

    let doc_label = NodeLabel::new("Document").map_err(map_label_err)?;
    let chunk_label = NodeLabel::new("Chunk").map_err(map_label_err)?;
    let entity_label = NodeLabel::new("Entity").map_err(map_label_err)?;

    let has_chunk_rel = RelationType::new("HAS_CHUNK").map_err(map_label_err)?;
    let next_rel = RelationType::new("NEXT").map_err(map_label_err)?;
    let mentions_rel = RelationType::new("MENTIONS").map_err(map_label_err)?;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();

    // Document node.
    nodes.push(GraphNode {
        id: doc_id.to_owned(),
        label: doc_label,
        properties: json!({
            "source": source,
            "total_chunks": chunks.len(),
        }),
    });

    // Track entity nodes to deduplicate across chunks.
    let mut entity_node_ids = std::collections::HashMap::<String, String>::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_node_id = chunk.chunk_id.clone();

        // Chunk node.
        nodes.push(GraphNode {
            id: chunk_node_id.clone(),
            label: chunk_label.clone(),
            properties: json!({
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.metadata.section_title,
            }),
        });

        // Document -[HAS_CHUNK]-> Chunk.
        edges.push(GraphEdge {
            source_id: doc_id.to_owned(),
            target_id: chunk_node_id.clone(),
            relation_type: has_chunk_rel.clone(),
            properties: json!({"chunk_index": chunk.chunk_index}),
        });

        // Chunk -[NEXT]-> Chunk (ordering).
        if i > 0 {
            let prev_chunk_id = chunks[i - 1].chunk_id.clone();
            edges.push(GraphEdge {
                source_id: prev_chunk_id,
                target_id: chunk_node_id.clone(),
                relation_type: next_rel.clone(),
                properties: json!({}),
            });
        }

        // Entities for this chunk.
        if let Some(chunk_entities) = entities_per_chunk.get(i) {
            for entity in chunk_entities {
                // Deterministic entity ID based on name (dedup across chunks).
                let entity_id = entity_node_ids
                    .entry(entity.name.clone())
                    .or_insert_with(|| {
                        let id =
                            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, entity.name.as_bytes())
                                .to_string();

                        nodes.push(GraphNode {
                            id: id.clone(),
                            label: entity_label.clone(),
                            properties: json!({
                                "name": entity.name,
                                "entity_type": entity.entity_type,
                            }),
                        });

                        id
                    })
                    .clone();

                // Chunk -[MENTIONS]-> Entity.
                edges.push(GraphEdge {
                    source_id: chunk_node_id.clone(),
                    target_id: entity_id,
                    relation_type: mentions_rel.clone(),
                    properties: json!({}),
                });
            }
        }
    }

    Ok((nodes, edges))
}

/// Build graph structures from LLM-extracted knowledge graphs.
///
/// Given one [`KnowledgeGraph`] per chunk, this function:
/// 1. Creates the `Document` and `Chunk` nodes + their `HAS_CHUNK` / `NEXT`
///    edges (identical to the heuristic path).
/// 2. Deduplicates LLM-extracted entities across chunks by `id`. The first
///    description / type / name wins; later occurrences are merged through
///    `MENTIONS` edges only.
/// 3. Emits typed relation edges between entity nodes, using the
///    `relationship_name` (sanitized for Neo4j's `RelationType`).
/// 4. Emits `Chunk -[MENTIONS]-> Entity` edges for every entity that appeared
///    in a given chunk.
fn build_llm_graph_structures(
    doc_id: &str,
    source: &str,
    chunks: &[DocumentChunk],
    graphs: &[KnowledgeGraph],
) -> Result<(Vec<GraphNode>, Vec<GraphEdge>), RetrievalError> {
    let map_label_err = |e| RetrievalError::Ingestion {
        reason: format!("invalid graph label: {e}"),
    };

    let doc_label = NodeLabel::new("Document").map_err(map_label_err)?;
    let chunk_label = NodeLabel::new("Chunk").map_err(map_label_err)?;
    let entity_label = NodeLabel::new("Entity").map_err(map_label_err)?;

    let has_chunk_rel = RelationType::new("HAS_CHUNK").map_err(map_label_err)?;
    let next_rel = RelationType::new("NEXT").map_err(map_label_err)?;
    let mentions_rel = RelationType::new("MENTIONS").map_err(map_label_err)?;

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();

    // Document node.
    nodes.push(GraphNode {
        id: doc_id.to_owned(),
        label: doc_label,
        properties: json!({
            "source": source,
            "total_chunks": chunks.len(),
        }),
    });

    // Cross-chunk entity dedup: keep the first seen version of each entity.
    let mut entity_index: HashMap<String, ExtractedNode> = HashMap::new();

    // First pass: collect the unique entities from all chunks (first writer wins).
    for graph in graphs {
        for node in &graph.nodes {
            entity_index
                .entry(node.id.clone())
                .or_insert_with(|| node.clone());
        }
    }

    // Emit graph nodes for unique entities (sorted for determinism).
    let mut entity_ids_sorted: Vec<&String> = entity_index.keys().collect();
    entity_ids_sorted.sort();
    for entity_id in &entity_ids_sorted {
        let entity = &entity_index[*entity_id];
        nodes.push(GraphNode {
            id: (*entity_id).clone(),
            label: entity_label.clone(),
            properties: json!({
                "name": entity.name,
                "entity_type": entity.node_type,
                "description": entity.description,
            }),
        });
    }

    // Second pass: emit chunk nodes + chunk<->entity edges + document<->chunk edges.
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_node_id = chunk.chunk_id.clone();

        nodes.push(GraphNode {
            id: chunk_node_id.clone(),
            label: chunk_label.clone(),
            properties: json!({
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.metadata.section_title,
            }),
        });

        edges.push(GraphEdge {
            source_id: doc_id.to_owned(),
            target_id: chunk_node_id.clone(),
            relation_type: has_chunk_rel.clone(),
            properties: json!({"chunk_index": chunk.chunk_index}),
        });

        if i > 0 {
            edges.push(GraphEdge {
                source_id: chunks[i - 1].chunk_id.clone(),
                target_id: chunk_node_id.clone(),
                relation_type: next_rel.clone(),
                properties: json!({}),
            });
        }

        let Some(graph) = graphs.get(i) else {
            continue;
        };

        // MENTIONS edges from this chunk to every unique entity in its graph.
        let mut mentioned_in_chunk: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for node in &graph.nodes {
            if !entity_index.contains_key(&node.id) {
                continue;
            }
            if mentioned_in_chunk.insert(node.id.as_str()) {
                edges.push(GraphEdge {
                    source_id: chunk_node_id.clone(),
                    target_id: node.id.clone(),
                    relation_type: mentions_rel.clone(),
                    properties: json!({}),
                });
            }
        }

        // Typed entity->entity edges from this chunk's graph.
        for edge in &graph.edges {
            if !entity_index.contains_key(&edge.source_node_id)
                || !entity_index.contains_key(&edge.target_node_id)
            {
                tracing::debug!(
                    src = %edge.source_node_id,
                    dst = %edge.target_node_id,
                    "dropping LLM edge referencing unknown node"
                );
                continue;
            }

            let rel_str = sanitize_relationship(&edge.relationship_name);
            let relation_type = match RelationType::new(&rel_str) {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::warn!(
                        raw = %edge.relationship_name,
                        sanitized = %rel_str,
                        error = %e,
                        "dropping edge with invalid relation type"
                    );
                    continue;
                }
            };

            edges.push(GraphEdge {
                source_id: edge.source_node_id.clone(),
                target_id: edge.target_node_id.clone(),
                relation_type,
                properties: json!({
                    "description": edge.description,
                    "chunk_id": chunk.chunk_id,
                }),
            });
        }
    }

    Ok((nodes, edges))
}
