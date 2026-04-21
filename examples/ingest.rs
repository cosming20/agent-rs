//! Dev CLI: ingest a single document and print per-stage stats.
//!
//! Usage:
//!
//! ```bash
//! # 2 positional args: user_id and document path
//! just ingest "alice" test.pdf
//!
//! # Plain text file (detected by .txt / .md extension)
//! just ingest "alice" notes.txt
//! ```
//!
//! Prerequisites: `.env` loaded, `docker compose up -d qdrant postgres neo4j
//! docling-sidecar minio`, `OPENAI_API_KEY` (or `LLM_API_KEY`) set.
//!
//! Flow:
//! 1. Parse user_id (validated) + document path.
//! 2. Read file bytes; compute sha256 to derive a deterministic document_id.
//! 3. Upload bytes to MinIO under `users/{user_id}/docs/{document_id}.{ext}`.
//! 4. Call the ingest pipeline (Docling + embedder + Qdrant + Neo4j + Postgres).
//! 5. Update `ingested_documents.minio_object_key` in Postgres.
//!
//! If MinIO upload succeeds but the pipeline fails, the MinIO object is kept
//! in place as an upload record — the user can retry and dedup will resume.
//!
//! Prints a detailed summary at the end:
//! - user_id, document_id, sha256, minio_object_key
//! - chunks created
//! - vectors upserted (dense + sparse via Qdrant server-side BM25)
//! - knowledge graph: nodes, edges, triplets
//! - summaries generated
//! - wall-clock duration per stage + total

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use agent_rs::embedder::build_embedder;
use agent_rs::ingest::{IngestFeatures, build_ingestion_pipeline};
use agent_rs::stores::connect_stores;
use agent_rs_core::config::Settings;
use agent_rs_core::types::{CollectionName, UserId};
use agent_rs_retrieval::ingestion::derive_document_id;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_stores::minio::MinioStore;
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::CollectionConfig;
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

struct Args {
    user_id: UserId,
    pdf_path: PathBuf,
    source: String,
}

fn parse_args() -> Result<Args> {
    let mut positional: Vec<String> = Vec::new();
    let mut source: Option<String> = None;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--source" => {
                source = Some(iter.next().context("--source needs a value")?);
            }
            "-h" | "--help" => {
                println!(
                    "Usage: ingest <user_id> <path/to/doc>\n\
                     Options:\n\
                       --source <label>  Override the source label (default: filename stem)"
                );
                std::process::exit(0);
            }
            flag if flag.starts_with("--") => {
                bail!("unknown flag: {flag}");
            }
            arg => positional.push(arg.to_owned()),
        }
    }
    if positional.len() != 2 {
        bail!(
            "usage: just ingest \"<user_id>\" \"<path>\" — got {} positional arg(s)",
            positional.len()
        );
    }
    let user_id = UserId::try_new(&positional[0]).context("invalid user_id")?;
    let pdf_path = PathBuf::from(&positional[1]);
    let source = source.unwrap_or_else(|| {
        pdf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("cli")
            .to_owned()
    });
    Ok(Args {
        user_id,
        pdf_path,
        source,
    })
}

/// Compute the SHA-256 hex digest of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}


async fn ensure_collection(
    store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    dense_dim: usize,
) -> Result<()> {
    let cfg = CollectionConfig {
        dense_vector_size: dense_dim,
        enable_sparse: true,
        dense_vector_name: "dense".into(),
        sparse_vector_name: "sparse".into(),
    };
    match store.create_collection(collection, &cfg).await {
        Ok(()) => {
            info!(%collection, "collection created");
            Ok(())
        }
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            if msg.contains("already exists") || msg.contains("conflict") {
                warn!(%collection, "collection already exists — reusing");
                Ok(())
            } else {
                Err(e).with_context(|| format!("create_collection {collection}"))
            }
        }
    }
}

/// Content-type string for a given file extension.
fn content_type_for_ext(ext: &str) -> &'static str {
    match ext {
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "html" | "htm" => "text/html",
        _ => "application/octet-stream",
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    if std::env::var_os("RUST_LOG").is_none() {
        // SAFETY: we only set this when the caller did not.
        unsafe { std::env::set_var("RUST_LOG", "info,agent_rs_retrieval=info") };
    }

    if !args.pdf_path.exists() {
        bail!("file not found: {}", args.pdf_path.display());
    }

    let settings = Settings::load().context("load settings")?;

    // Full OTel pipeline (traces → Jaeger, metrics → Prometheus via the
    // OTel collector). Guard must live until end of main so the background
    // exporters flush before shutdown.
    let _otel_guard = init_tracing(&settings.otel, "agent-rs-ingest")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    info!(
        user_id = %args.user_id,
        source = %args.source,
        path = %args.pdf_path.display(),
        "ingest CLI starting"
    );

    // ---- Connect to stores + build embedder ----
    let t0 = Instant::now();
    let stores = connect_stores(&settings).await?;
    let (embedder, embedder_label) = build_embedder().await?;
    info!(
        embedder = %embedder_label,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "stores + embedder ready"
    );

    // ---- Read file bytes + derive document_id from sha256 ----
    let ext = args
        .pdf_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_text_like = matches!(ext.as_str(), "txt" | "md" | "markdown");

    let bytes = std::fs::read(&args.pdf_path)
        .with_context(|| format!("read file {}", args.pdf_path.display()))?;

    let sha256 = sha256_hex(&bytes);
    let document_id = derive_document_id(&args.user_id, &sha256);

    info!(
        user_id = %args.user_id,
        document_id = %document_id,
        sha256 = %sha256,
        "derived document_id from sha256"
    );

    // ---- Step 1: Upload to MinIO ----
    // The upload is attempted before the pipeline so the object exists in
    // storage regardless of whether chunking/embedding succeeds.  This
    // preserves the file for debugging or retry without re-upload.
    let minio_store = MinioStore::new(settings.minio.clone())
        .await
        .context("connect MinioStore")?;

    let content_type = content_type_for_ext(&ext);
    let minio_object_key = minio_store
        .put_object(&args.user_id, document_id, &bytes, content_type)
        .await
        .context("MinioStore::put_object")?;

    info!(
        user_id = %args.user_id,
        document_id = %document_id,
        object_key = %minio_object_key,
        "uploaded to MinIO"
    );

    // ---- Derive collection names ----
    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));

    // ---- Ensure collections exist (idempotent) ----
    let dense_dim = embedder.dimensions();
    ensure_collection(&*stores.qdrant, &collection, dense_dim).await?;
    ensure_collection(&*stores.qdrant, &triplet_collection, dense_dim).await?;
    ensure_collection(&*stores.qdrant, &summaries_collection, dense_dim).await?;
    info!(%collection, %triplet_collection, %summaries_collection, "collections ready");

    // ---- Build ingestion pipeline ----
    let metrics = Arc::new(RetrievalMetrics::new("agent-rs-ingest"));
    let pipeline = build_ingestion_pipeline(
        &settings,
        &stores,
        Arc::clone(&embedder),
        &collection,
        &triplet_collection,
        &summaries_collection,
        &IngestFeatures::from_env(),
        Some(Arc::clone(&metrics)),
    )
    .await?;

    // ---- Step 2: Run ingest pipeline ----
    let doc_id_str = document_id.to_string();
    let t_ingest = Instant::now();
    let result = if is_text_like {
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("text file is not valid UTF-8: {}", args.pdf_path.display()))?;
        pipeline
            .ingest_text(text, &doc_id_str, &args.source, args.user_id.as_str())
            .await
            .context("ingest_text")?
    } else {
        pipeline
            .ingest_pdf(&bytes, &doc_id_str, &args.source, args.user_id.as_str())
            .await
            .context("ingest_pdf")?
    };
    let ingest_ms = t_ingest.elapsed().as_millis() as u64;

    metrics.record_ingest_stage_ms("total", ingest_ms as f64);
    metrics.record_ingest_stage_ms("pipeline", result.duration_ms as f64);

    // ---- Step 3: Persist the MinIO object key in Postgres ----
    // The pipeline's begin_ingestion inserts the row with minio_object_key = NULL;
    // we update it here now that both the upload and ingest have succeeded.
    stores
        .postgres
        .set_minio_object_key(&args.user_id, document_id, &minio_object_key)
        .await
        .context("set_minio_object_key")?;

    info!(
        user_id = %args.user_id,
        document_id = %document_id,
        object_key = %minio_object_key,
        "persisted MinIO key in Postgres"
    );

    // ---- Pretty summary ----
    println!();
    println!("================ INGEST SUMMARY ================");
    println!("  user_id             : {}", args.user_id);
    println!("  document_id         : {}", document_id);
    println!("  sha256              : {}", sha256);
    println!("  minio_object_key    : {}", minio_object_key);
    println!("  status              : complete");
    println!("  source              : {}", args.source);
    println!("  path                : {}", args.pdf_path.display());
    println!("  chunks created      : {}", result.chunks_created);
    println!("  vectors upserted    : {}", result.vectors_upserted);
    println!("  graph nodes created : {}", result.graph_nodes_created);
    println!("  graph edges created : {}", result.graph_edges_created);
    println!("  triplets upserted   : {}", result.triplets_upserted);
    println!("  summaries created   : {}", result.summaries_created);
    println!("  ingest duration_ms  : {}", result.duration_ms);
    println!("  wall-clock ms       : {}", ingest_ms);
    println!("  collection          : {}", collection);
    println!("  embedder            : {}", embedder_label);
    println!("================================================");

    Ok(())
}
