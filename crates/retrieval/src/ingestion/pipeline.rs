//! End-to-end ingestion pipeline.
//!
//! Orchestrates: chunk -> extract metadata -> embed -> upsert to vector store
//! -> extract entities -> store in graph.
//!
//! Works on any document type. PDF text extraction is supported via
//! `pdf-extract`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use agent_rs_core::CollectionName;
use agent_rs_stores::traits::{GraphStore, VectorStore};
use agent_rs_stores::types::{GraphEdge, GraphNode, NodeLabel, RelationType, VectorPoint};
use regex::Regex;
use serde::Serialize;
use serde_json::json;

use crate::embedder::Embedder;
use crate::error::RetrievalError;

use super::chunker::{self, ChunkerConfig, DocumentChunk};
use super::graph_extractor::{
    DEFAULT_BATCH_CONCURRENCY, ExtractedNode, KnowledgeGraph, LlmGraphExtractor, Triplet,
    sanitize_relationship,
};

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
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

// ---- Pipeline ----

/// Optional extensions for the ingestion pipeline.
///
/// Use this to supply components that are not required for a minimal ingestion
/// run (e.g. an [`LlmGraphExtractor`] for LLM-based entity extraction).
///
/// Defaults leave all optional extensions disabled; the pipeline then falls
/// back to its heuristic entity extractor.
#[derive(Clone, Default)]
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
        }
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
        Self {
            embedder,
            vector_store,
            graph_store,
            chunker_config: config.chunker,
            collection,
            llm_extractor: config.llm_extractor,
            llm_concurrency,
            triplet_collection: config.triplet_collection,
        }
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

        // 2b. Attempt sparse embedding for hybrid/keyword search.
        //
        // Falls back gracefully when the embedder does not support sparse
        // (e.g. a bare `OpenAIEmbedder`). In that case ingestion continues
        // with dense-only vectors and `enable_sparse: false`.
        let sparse_vectors: Vec<Option<agent_rs_stores::types::SparseVector>> =
            match self.embedder.embed_sparse_batch(&texts).await {
                Ok(svecs) => {
                    tracing::debug!(
                        count = svecs.len(),
                        "sparse vectors generated for ingestion"
                    );
                    svecs
                        .into_iter()
                        .map(|sv| {
                            Some(agent_rs_stores::types::SparseVector {
                                indices: sv.indices,
                                values: sv.values,
                            })
                        })
                        .collect()
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "embedder does not support sparse vectors; ingesting dense-only"
                    );
                    vec![None; chunks.len()]
                }
            };

        let has_sparse = sparse_vectors.iter().any(Option::is_some);

        // 3. Build VectorPoints with rich payloads.
        let vector_points: Vec<VectorPoint> = chunks
            .iter()
            .zip(embeddings.into_iter())
            .zip(sparse_vectors.into_iter())
            .map(|((chunk, embedding), sparse_vector)| {
                let payload = build_chunk_payload(chunk);
                VectorPoint {
                    id: chunk.chunk_id.clone(),
                    embedding,
                    sparse_vector,
                    payload,
                }
            })
            .collect();

        // 4. Ensure collection exists, then upsert to vector store.
        let dims = self.embedder.dimensions();
        let exists = self
            .vector_store
            .collection_exists(&self.collection)
            .await
            .map_err(RetrievalError::Store)?;
        if !exists {
            let config = agent_rs_stores::types::CollectionConfig {
                dense_vector_size: dims,
                enable_sparse: has_sparse,
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
                enable_sparse = has_sparse,
                "created collection for ingestion"
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

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::info!(doc_id, duration_ms, "ingestion complete");

        Ok(IngestionResult {
            doc_id: doc_id.to_owned(),
            chunks_created,
            vectors_upserted,
            graph_nodes_created,
            graph_edges_created,
            triplets_upserted,
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
                sparse_vector: None,
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

    /// Ingest a PDF document from raw bytes.
    ///
    /// Extracts text using `pdf-extract`, then delegates to [`ingest_text`].
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalError::PdfExtraction`] if the PDF cannot be parsed.
    #[tracing::instrument(skip(self, pdf_bytes), fields(doc_id = doc_id, source = source, byte_len = pdf_bytes.len()))]
    pub async fn ingest_pdf(
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
            "PDF text extraction complete"
        );

        self.ingest_text(&text, doc_id, source).await
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
