//! E2E Legal QA test: ingest the Romanian Penal Code from `test.pdf`, chunk
//! by articles, embed + index into Qdrant and Neo4j, then run legal questions
//! through the retrieval router with HYBRID strategy.
//!
//! # Usage
//!
//! ```bash
//! # Ensure Qdrant (ports 1050/1051), Neo4j (port 1061), and Postgres (port 1054) are running.
//! cargo run --example e2e_legal_qa
//! ```
//!
//! Results are written to `agent-rs/test_results/legal_qa_trace_{timestamp}.json`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use regex::Regex;
use serde::Serialize;
use tracing::info;
use uuid::Uuid;

use agent_rs_core::config::Settings;
use agent_rs_core::types::{CollectionName, Embedding, RetrievalStrategy};
use agent_rs_retrieval::Embedder;
use agent_rs_retrieval::embedder::{FastEmbedder, FastEmbedderConfig};
use agent_rs_retrieval::ranker::ResultRanker;
use agent_rs_retrieval::router::RetrievalRouter;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use agent_rs_stores::traits::{GraphStore, VectorStore};
use agent_rs_stores::types::{
    CollectionConfig, GraphEdge, GraphNode, NodeLabel, RelationType, VectorPoint,
};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A single article extracted from the Romanian Penal Code PDF.
#[derive(Debug, Clone, Serialize)]
struct LegalChunk {
    /// Article number, e.g. "170".
    article_number: String,
    /// Article title, e.g. "Reînnoirea cererii de reabilitare judecătorească".
    title: String,
    /// Full article text including the title line.
    content: String,
}

// ---------------------------------------------------------------------------
// Trace output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct EvidenceSnippet {
    article: String,
    snippet: String,
    score: f64,
}

#[derive(Serialize)]
struct QueryResult {
    question: String,
    strategy: String,
    results_count: usize,
    top_evidence: Vec<EvidenceSnippet>,
    duration_ms: u64,
}

#[derive(Serialize)]
struct LegalQaTrace {
    test_run_id: String,
    pdf_file: String,
    total_articles_extracted: usize,
    total_chunks_ingested: usize,
    ingestion_duration_ms: u64,
    queries: Vec<QueryResult>,
}

// ---------------------------------------------------------------------------
// PDF extraction
// ---------------------------------------------------------------------------

/// Resolve the path to `test.pdf`, checking env var `LEGAL_PDF_PATH` first,
/// then falling back to `../../test.pdf` relative to the workspace root.
fn resolve_pdf_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("LEGAL_PDF_PATH") {
        let p = PathBuf::from(path);
        anyhow::ensure!(p.exists(), "LEGAL_PDF_PATH does not exist: {}", p.display());
        return Ok(p);
    }

    // Relative to the agent-rs directory (Cargo sets CWD to workspace root).
    let candidates = [
        PathBuf::from("../../test.pdf"),
        PathBuf::from("../test.pdf"),
        PathBuf::from("test.pdf"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    anyhow::bail!(
        "test.pdf not found. Set LEGAL_PDF_PATH env var or place test.pdf at the repo root."
    );
}

/// Extract raw text from a PDF file using `pdf_extract`.
fn extract_pdf_text(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read PDF: {}", path.display()))?;
    let text = pdf_extract::extract_text_from_mem(&bytes)
        .with_context(|| format!("failed to extract text from PDF: {}", path.display()))?;
    Ok(text)
}

// ---------------------------------------------------------------------------
// Article chunking
// ---------------------------------------------------------------------------

/// Split Romanian legal text by article boundaries.
///
/// Looks for patterns like:
/// - `Articolul 170` (followed by an optional title on the next line)
/// - `Art. 170` (abbreviated form)
///
/// Each chunk contains the article number, extracted title, and full content.
fn chunk_by_articles(text: &str) -> Vec<LegalChunk> {
    // Match "Articolul NNN" or "Art. NNN" at the start of a line.
    // The article number can have superscripts like "170^1" in some editions.
    let article_re =
        Regex::new(r"(?m)^[\s]*(Articolul|Art\.?)\s+(\d+(?:\^?\d+)?)[\s]*[-–—.]?[\s]*(.*)$")
            .expect("article regex must compile");

    let matches: Vec<_> = article_re.find_iter(text).collect();

    if matches.is_empty() {
        // Fallback: if no articles found, try a more lenient pattern
        tracing::warn!("no articles found with strict pattern; trying lenient match");
        return chunk_by_articles_lenient(text);
    }

    let cap_iter: Vec<_> = article_re.captures_iter(text).collect();
    let match_iter: Vec<_> = article_re.find_iter(text).collect();

    let mut chunks = Vec::with_capacity(cap_iter.len());

    for (i, caps) in cap_iter.iter().enumerate() {
        let article_number = caps[2].trim().to_string();

        // The rest of the first line after the article number may contain the title.
        let inline_title = caps[3].trim().to_string();

        // Content extends from this match start to the next match start (or end of text).
        let start = match_iter[i].start();
        let end = if i + 1 < match_iter.len() {
            match_iter[i + 1].start()
        } else {
            text.len()
        };

        let raw_content = text[start..end].trim().to_string();

        // Extract the title: first non-empty line after the article header,
        // or the inline title if present.
        let title = if !inline_title.is_empty() {
            inline_title
        } else {
            extract_title_from_content(&raw_content, &article_number)
        };

        chunks.push(LegalChunk {
            article_number,
            title,
            content: raw_content,
        });
    }

    chunks
}

/// Lenient article chunking for PDFs with different formatting.
fn chunk_by_articles_lenient(text: &str) -> Vec<LegalChunk> {
    let article_re =
        Regex::new(r"(?mi)(articolul|art\.?)\s+(\d+)").expect("lenient article regex must compile");

    let cap_iter: Vec<_> = article_re.captures_iter(text).collect();
    let match_iter: Vec<_> = article_re.find_iter(text).collect();

    let mut chunks = Vec::with_capacity(cap_iter.len());

    for (i, caps) in cap_iter.iter().enumerate() {
        let article_number = caps[2].trim().to_string();

        let start = match_iter[i].start();
        let end = if i + 1 < match_iter.len() {
            match_iter[i + 1].start()
        } else {
            text.len()
        };

        let raw_content = text[start..end].trim().to_string();
        let title = extract_title_from_content(&raw_content, &article_number);

        chunks.push(LegalChunk {
            article_number,
            title,
            content: raw_content,
        });
    }

    chunks
}

/// Extract the title from article content by looking at lines after the header.
fn extract_title_from_content(content: &str, _article_number: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    // Skip the first line (the "Articolul NNN" line) and find the first
    // non-empty line that looks like a title (usually short, not starting
    // with "(1)" which indicates paragraph content).
    for line in lines.iter().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip numbered paragraphs like "(1)", "(2)", etc.
        if trimmed.starts_with('(') && trimmed.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) {
            break;
        }
        // The title is typically a short line (under 200 chars).
        if trimmed.len() < 200 {
            return trimmed.to_string();
        }
        break;
    }
    String::new()
}

/// Deduplicate chunks by article number, keeping the first occurrence.
fn dedup_chunks(chunks: Vec<LegalChunk>) -> Vec<LegalChunk> {
    let mut seen = std::collections::HashSet::new();
    chunks
        .into_iter()
        .filter(|c| seen.insert(c.article_number.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

/// Ingest legal chunks into Qdrant (vectors) and Neo4j (graph).
async fn ingest_chunks(
    chunks: &[LegalChunk],
    embedder: &FastEmbedder,
    vector_store: &dyn VectorStore,
    graph_store: &dyn GraphStore,
    collection: &CollectionName,
    dims: usize,
) -> Result<u64> {
    let start = Instant::now();

    // 1. Delete existing collection if it exists, then create fresh.
    if vector_store
        .collection_exists(collection)
        .await
        .unwrap_or(false)
    {
        info!(collection = %collection, "deleting existing collection");
        // Use raw Qdrant client for collection deletion since the trait
        // only supports point-level deletion.
        let settings = Settings::load().context("reload settings for Qdrant cleanup")?;
        let mut builder = qdrant_client::Qdrant::from_url(&settings.qdrant.grpc_url());
        if !settings.qdrant.api_key.is_empty() {
            builder = builder.api_key(settings.qdrant.api_key.clone());
        }
        let raw_client = builder
            .build()
            .context("build raw Qdrant client for cleanup")?;
        raw_client
            .delete_collection(collection.as_str())
            .await
            .context("delete existing Qdrant collection")?;
        info!("existing collection deleted");
    }

    // 2. Create collection with hybrid (dense + sparse) config.
    let config = CollectionConfig::hybrid(dims);
    vector_store
        .create_collection(collection, &config)
        .await
        .context("create_collection")?;
    info!(collection = %collection, dims = dims, "collection created");

    // 3. Embed all chunks (batch for efficiency).
    let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

    // Process in batches to avoid OOM on large PDFs.
    let batch_size = 64;
    let mut all_dense: Vec<Embedding> = Vec::with_capacity(chunks.len());
    let mut all_sparse: Vec<agent_rs_retrieval::SparseVector> = Vec::with_capacity(chunks.len());

    for (batch_idx, text_batch) in texts.chunks(batch_size).enumerate() {
        info!(
            batch = batch_idx + 1,
            total_batches = texts.len().div_ceil(batch_size),
            batch_size = text_batch.len(),
            "embedding batch"
        );

        let dense_batch = embedder
            .embed_documents(text_batch)
            .await
            .context("embed dense batch (passage prefix)")?;
        let sparse_batch = embedder
            .embed_sparse_batch(text_batch)
            .await
            .context("embed sparse batch")?;

        all_dense.extend(dense_batch);
        all_sparse.extend(sparse_batch);
    }

    // 4. Build vector points and upsert.
    let points: Vec<VectorPoint> = chunks
        .iter()
        .zip(all_dense.iter())
        .zip(all_sparse.iter())
        .map(|((chunk, dense), sparse)| VectorPoint {
            id: format!("art_{}", chunk.article_number),
            embedding: Embedding(dense.0.clone()),
            sparse_vector: Some(agent_rs_stores::types::SparseVector {
                indices: sparse.indices.clone(),
                values: sparse.values.clone(),
            }),
            payload: serde_json::json!({
                "article_number": chunk.article_number,
                "title": chunk.title,
                "chunk_text": chunk.content,
                "source": "legal_code_ro",
                "doc_id": format!("art_{}", chunk.article_number),
            }),
        })
        .collect();

    // Upsert in batches to avoid gRPC message size limits.
    let upsert_batch_size = 100;
    for (batch_idx, point_batch) in points.chunks(upsert_batch_size).enumerate() {
        info!(
            batch = batch_idx + 1,
            total_batches = points.len().div_ceil(upsert_batch_size),
            points = point_batch.len(),
            "upserting vector batch"
        );
        vector_store
            .upsert(collection, point_batch.to_vec())
            .await
            .with_context(|| format!("upsert batch {}", batch_idx + 1))?;
    }

    info!(total_points = points.len(), "all vectors upserted");

    // 5. Create Neo4j nodes for each article.
    let article_label = NodeLabel::new("Article").expect("'Article' is a valid node label");

    for chunk in chunks {
        let node = GraphNode {
            id: format!("art_{}", chunk.article_number),
            label: article_label.clone(),
            properties: serde_json::json!({
                "id": format!("art_{}", chunk.article_number),
                "number": chunk.article_number,
                "title": chunk.title,
            }),
        };
        graph_store
            .add_node(&node)
            .await
            .with_context(|| format!("add node for article {}", chunk.article_number))?;
    }
    info!(node_count = chunks.len(), "graph nodes created");

    // 6. Create FOLLOWS edges between consecutive articles.
    let follows_type = RelationType::new("FOLLOWS").expect("'FOLLOWS' is a valid relation type");

    for window in chunks.windows(2) {
        let edge = GraphEdge {
            source_id: format!("art_{}", window[0].article_number),
            target_id: format!("art_{}", window[1].article_number),
            relation_type: follows_type.clone(),
            properties: serde_json::json!({}),
        };
        graph_store.add_edge(&edge).await.with_context(|| {
            format!(
                "add FOLLOWS edge: art_{} -> art_{}",
                window[0].article_number, window[1].article_number
            )
        })?;
    }
    info!(
        edge_count = chunks.len().saturating_sub(1),
        "FOLLOWS edges created"
    );

    Ok(start.elapsed().as_millis() as u64)
}

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

/// Run a single legal question through the retrieval router.
async fn run_legal_query(
    question: &str,
    strategy: RetrievalStrategy,
    router: &RetrievalRouter,
    limit: usize,
) -> QueryResult {
    let start = Instant::now();

    let result = router.retrieve(question, strategy, limit).await;

    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(pack) => {
            let top_evidence: Vec<EvidenceSnippet> = pack
                .documents
                .iter()
                .take(3)
                .map(|doc| {
                    let article = doc
                        .payload
                        .get("article_number")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string();

                    let snippet = doc
                        .payload
                        .get("chunk_text")
                        .and_then(|v| v.as_str())
                        .map(|s| agent_rs_core::types::truncate_to_char_boundary(s, 200).to_owned())
                        .unwrap_or_default();

                    EvidenceSnippet {
                        article,
                        snippet,
                        score: doc.score.0,
                    }
                })
                .collect();

            QueryResult {
                question: question.to_owned(),
                strategy: strategy.as_db_str().to_owned(),
                results_count: pack.len(),
                top_evidence,
                duration_ms,
            }
        }
        Err(e) => {
            tracing::error!(error = %e, question = %question, "query failed");
            QueryResult {
                question: question.to_owned(),
                strategy: strategy.as_db_str().to_owned(),
                results_count: 0,
                top_evidence: vec![],
                duration_ms,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Human-readable summary
// ---------------------------------------------------------------------------

fn print_summary(
    pdf_path: &Path,
    chunks: &[LegalChunk],
    ingestion_ms: u64,
    results: &[QueryResult],
) {
    println!();
    println!("=== Legal QA E2E Test ===");
    println!("PDF: {}", pdf_path.display());
    println!("Articles extracted: {}", chunks.len());
    println!("Chunks ingested: {}", chunks.len());
    println!("Ingestion time: {}ms", ingestion_ms);
    println!();

    for (i, qr) in results.iter().enumerate() {
        println!(
            "Q{}: \"{}\"",
            i + 1,
            agent_rs_core::types::truncate_to_char_boundary(&qr.question, 80)
        );
        if qr.top_evidence.is_empty() {
            println!("   -> No results found");
        } else {
            for ev in &qr.top_evidence {
                let snippet_preview =
                    agent_rs_core::types::truncate_to_char_boundary(&ev.snippet, 100);
                println!(
                    "   -> Found Article {}: \"{}\"",
                    ev.article, snippet_preview
                );
                println!(
                    "   -> Score: {:.4}, Duration: {}ms",
                    ev.score, qr.duration_ms
                );
            }
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

/// Delete all test data from Qdrant (collection) and Neo4j (article nodes).
async fn cleanup(
    collection: &CollectionName,
    graph_store: &dyn GraphStore,
    chunks: &[LegalChunk],
    settings: &Settings,
) {
    info!("--- CLEANUP PHASE ---");

    // Delete Qdrant collection via raw client.
    let cleanup_result: Result<()> = async {
        let mut builder = qdrant_client::Qdrant::from_url(&settings.qdrant.grpc_url());
        if !settings.qdrant.api_key.is_empty() {
            builder = builder.api_key(settings.qdrant.api_key.clone());
        }
        let raw_client = builder.build().context("build cleanup Qdrant client")?;
        if raw_client
            .collection_exists(collection.as_str())
            .await
            .unwrap_or(false)
        {
            raw_client
                .delete_collection(collection.as_str())
                .await
                .context("delete Qdrant collection")?;
            info!(collection = %collection, "Qdrant collection deleted");
        }
        Ok(())
    }
    .await;

    if let Err(e) = cleanup_result {
        tracing::warn!(error = %e, "Qdrant cleanup failed");
    }

    // Delete Neo4j article nodes.
    let mut deleted = 0usize;
    let mut errors = 0usize;
    for chunk in chunks {
        let node_id = format!("art_{}", chunk.article_number);
        match graph_store.delete_node(&node_id).await {
            Ok(()) => deleted += 1,
            Err(e) => {
                tracing::warn!(node_id = %node_id, error = %e, "failed to delete graph node");
                errors += 1;
            }
        }
    }
    info!(deleted = deleted, errors = errors, "Neo4j cleanup complete");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load configuration.
    let settings = Settings::load().context("failed to load configuration")?;
    agent_rs_core::logging::init_logging(false);

    let test_run_id = Uuid::new_v4().to_string();
    info!(test_run_id = %test_run_id, "Legal QA E2E test starting");

    // 2. Resolve and extract PDF text.
    let pdf_path = resolve_pdf_path().context("resolve PDF path")?;
    info!(path = %pdf_path.display(), "extracting text from PDF");

    let raw_text = extract_pdf_text(&pdf_path).context("extract PDF text")?;
    info!(
        text_len = raw_text.len(),
        "PDF text extracted ({} chars)",
        raw_text.len()
    );

    // 3. Chunk by articles.
    let chunks_raw = chunk_by_articles(&raw_text);
    let chunks = dedup_chunks(chunks_raw);
    info!(
        articles = chunks.len(),
        "articles extracted and deduplicated"
    );

    if chunks.is_empty() {
        anyhow::bail!("no articles extracted from PDF — check the text format");
    }

    // Print a sample of extracted articles.
    for chunk in chunks.iter().take(3) {
        info!(
            article = %chunk.article_number,
            title = %chunk.title,
            content_len = chunk.content.len(),
            "sample article"
        );
    }

    // 4. Connect to storage backends.
    info!("connecting to Qdrant...");
    let qdrant_store =
        QdrantStore::new(&settings.qdrant).context("failed to create Qdrant client")?;
    let qdrant_store = Arc::new(qdrant_store);

    info!("connecting to Neo4j...");
    let neo4j_store = Neo4jStore::connect(&settings.neo4j)
        .await
        .context("failed to connect to Neo4j")?;
    let neo4j_store = Arc::new(neo4j_store);

    info!("connecting to PostgreSQL...");
    let pg_store = PgStore::new(&settings.postgres.dsn())
        .await
        .context("failed to connect to PostgreSQL")?;
    let _pg_store = Arc::new(pg_store);

    info!("all storage backends connected");

    // 5. Initialize embedder.
    info!("initializing embedder (may download ONNX models on first run)...");
    let embedder = FastEmbedder::try_new(FastEmbedderConfig {
        show_download_progress: true,
        ..Default::default()
    })
    .await
    .context("failed to initialize embedding models")?;
    let dims = embedder.dimensions();
    let embedder = Arc::new(embedder);
    info!(dims = dims, "embedder initialized");

    // 6. Set up collection.
    let collection = CollectionName("legal_code_ro".into());

    // 7. Ingest.
    info!("--- INGESTION PHASE ---");
    let ingestion_ms = ingest_chunks(
        &chunks,
        &embedder,
        qdrant_store.as_ref(),
        neo4j_store.as_ref(),
        &collection,
        dims,
    )
    .await
    .context("ingestion failed")?;
    info!(duration_ms = ingestion_ms, "ingestion complete");

    // 8. Create retrieval router.
    let ranker = ResultRanker::new(settings.ranker.rrf_k);
    let router = Arc::new(RetrievalRouter::new(
        embedder.clone() as Arc<dyn agent_rs_retrieval::Embedder + Send + Sync>,
        qdrant_store.clone() as Arc<dyn VectorStore + Send + Sync>,
        neo4j_store.clone() as Arc<dyn GraphStore + Send + Sync>,
        ranker,
        collection.clone(),
    ));

    // 9. Run queries.
    info!("--- QUERY PHASE ---");

    let questions: Vec<(&str, RetrievalStrategy)> = vec![
        (
            "Ce prevede articolul 170 din Codul Penal?",
            RetrievalStrategy::Hybrid,
        ),
        (
            "Ce se intampla daca fur si bat o persoana?",
            RetrievalStrategy::Hybrid,
        ),
        (
            "Care sunt pedepsele pentru furt calificat?",
            RetrievalStrategy::Hybrid,
        ),
        (
            "Ce este tentativa conform Codului Penal?",
            RetrievalStrategy::Hybrid,
        ),
        (
            "Care sunt circumstantele agravante?",
            RetrievalStrategy::Hybrid,
        ),
    ];

    let limit = 5;
    let mut query_results: Vec<QueryResult> = Vec::with_capacity(questions.len());

    for (question, strategy) in &questions {
        info!(question = %question, strategy = %strategy, "running query");
        let qr = run_legal_query(question, *strategy, &router, limit).await;
        info!(
            question = %question,
            results = qr.results_count,
            duration_ms = qr.duration_ms,
            "query complete"
        );
        query_results.push(qr);
    }

    // 10. Print human-readable summary.
    print_summary(&pdf_path, &chunks, ingestion_ms, &query_results);

    // 11. Write JSON trace.
    let trace = LegalQaTrace {
        test_run_id: test_run_id.clone(),
        pdf_file: pdf_path.display().to_string(),
        total_articles_extracted: chunks.len(),
        total_chunks_ingested: chunks.len(),
        ingestion_duration_ms: ingestion_ms,
        queries: query_results,
    };

    let trace_dir = PathBuf::from("test_results");
    std::fs::create_dir_all(&trace_dir).context("create test_results directory")?;
    let timestamp = Utc::now().format("%Y%m%dT%H%M%S");
    let trace_path = trace_dir.join(format!("legal_qa_trace_{timestamp}.json"));
    let trace_json = serde_json::to_string_pretty(&trace).context("serialize trace to JSON")?;
    std::fs::write(&trace_path, &trace_json)
        .with_context(|| format!("write trace to {}", trace_path.display()))?;
    info!(path = %trace_path.display(), "trace written");

    // 12. Cleanup.
    cleanup(&collection, neo4j_store.as_ref(), &chunks, &settings).await;

    info!(test_run_id = %test_run_id, "Legal QA E2E test complete");
    println!("Trace written to: {}", trace_path.display());

    Ok(())
}
