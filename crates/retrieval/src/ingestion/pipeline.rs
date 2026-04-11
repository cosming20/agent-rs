//! End-to-end ingestion pipeline.
//!
//! Orchestrates: chunk -> extract metadata -> embed -> upsert to vector store
//! -> extract entities -> store in graph.
//!
//! Works on any document type. PDF text extraction is supported via
//! `pdf-extract`.

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
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

// ---- Pipeline ----

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
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline.
    pub fn new(
        embedder: Arc<dyn Embedder + Send + Sync>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        chunker_config: ChunkerConfig,
        collection: CollectionName,
    ) -> Self {
        Self {
            embedder,
            vector_store,
            graph_store,
            chunker_config,
            collection,
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

        // 5. Extract entities from chunks (heuristic Phase 1).
        let all_entities = extract_entities_from_chunks(&chunks);

        // 6. Build and store graph nodes.
        let (graph_nodes, graph_edges) =
            build_graph_structures(doc_id, source, &chunks, &all_entities)?;

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

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::info!(doc_id, duration_ms, "ingestion complete");

        Ok(IngestionResult {
            doc_id: doc_id.to_owned(),
            chunks_created,
            vectors_upserted,
            graph_nodes_created,
            graph_edges_created,
            duration_ms,
        })
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
