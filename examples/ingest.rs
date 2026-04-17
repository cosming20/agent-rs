//! Dev CLI: ingest a single document and print per-stage stats.
//!
//! Usage:
//!
//! ```bash
//! # PDF, default doc_id derived from filename, default source "cli"
//! cargo run --release --example ingest -- test.pdf
//!
//! # Override doc_id / source
//! cargo run --release --example ingest -- --doc-id mypdf --source manual test.pdf
//!
//! # Plain text file (detected by .txt extension; anything else goes through Docling)
//! cargo run --release --example ingest -- notes.txt
//! ```
//!
//! Prerequisites: `.env` loaded, `docker compose up -d qdrant postgres neo4j
//! docling-sidecar`, `OPENAI_API_KEY` (or `LLM_API_KEY`) set.
//!
//! Prints a detailed summary at the end:
//! - doc_id
//! - chunks created
//! - vectors upserted (dense + sparse via Qdrant server-side BM25)
//! - knowledge graph: nodes, edges, triplets
//! - summaries generated
//! - wall-clock duration per stage + total
//!
//! Stage timings come from the pipeline's per-phase `#[instrument]` spans,
//! surfaced via `RUST_LOG=info`. This example enables info-level logging
//! by default so the stages are visible live.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use agent_rs::embedder::build_embedder;
use agent_rs::ingest::{IngestFeatures, build_ingestion_pipeline};
use agent_rs::stores::connect_stores;
use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_stores::traits::VectorStore;
use agent_rs_stores::types::CollectionConfig;
use anyhow::{Context, Result, bail};
use tracing::{info, warn};

struct Args {
    pdf_path: PathBuf,
    doc_id: Option<String>,
    source: String,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    let mut doc_id: Option<String> = None;
    let mut source: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--doc-id" => {
                doc_id = Some(args.next().context("--doc-id needs a value")?);
            }
            "--source" => {
                source = Some(args.next().context("--source needs a value")?);
            }
            "-h" | "--help" => {
                println!("Usage: ingest [--doc-id <id>] [--source <label>] <path>");
                std::process::exit(0);
            }
            flag if flag.starts_with("--") => {
                bail!("unknown flag: {flag}");
            }
            positional => {
                if path.is_some() {
                    bail!("extra positional argument: {positional}");
                }
                path = Some(PathBuf::from(positional));
            }
        }
    }
    let pdf_path = path.context("missing document path; see --help")?;
    Ok(Args {
        pdf_path,
        doc_id,
        source: source.unwrap_or_else(|| "cli".to_owned()),
    })
}

async fn ensure_collection(
    store: &(dyn VectorStore + Send + Sync),
    collection: &CollectionName,
    dense_dim: usize,
) -> Result<()> {
    // Idempotent wrapper: the store trait's create_collection errors when
    // the collection already exists; treat that as success.
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
            // Qdrant responds with a gRPC 6 (ALREADY_EXISTS) that the
            // store layer surfaces as a StoreError::Backend with a
            // message mentioning "already exists". Tolerate that.
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
    // exporters flush before shutdown. If the collector is unreachable the
    // exporters fail silently per-batch; init itself only errors on
    // subscriber conflicts.
    let _otel_guard = init_tracing(&settings.otel, "agent-rs-ingest")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let derived_doc_id = args.doc_id.unwrap_or_else(|| {
        args.pdf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("doc")
            .to_owned()
    });

    info!(
        doc_id = %derived_doc_id,
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

    // ---- Derive collection names the same way main.rs does ----
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

    // ---- Build ingestion pipeline (all features on unless EVAL_SKIP_* is set) ----
    // Construct RetrievalMetrics so per-call LLM cost/tokens/latency flow
    // through graph_extractor + summarizer into Prometheus/Grafana.
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

    // ---- Run ingest ----
    let ext = args
        .pdf_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_text_like = matches!(ext.as_str(), "txt" | "md" | "markdown");

    let t_ingest = Instant::now();
    let result = if is_text_like {
        let text = std::fs::read_to_string(&args.pdf_path)
            .with_context(|| format!("read text file {}", args.pdf_path.display()))?;
        pipeline
            .ingest_text(&text, &derived_doc_id, &args.source)
            .await
            .context("ingest_text")?
    } else {
        let bytes = std::fs::read(&args.pdf_path)
            .with_context(|| format!("read PDF {}", args.pdf_path.display()))?;
        pipeline
            .ingest_pdf(&bytes, &derived_doc_id, &args.source)
            .await
            .context("ingest_pdf")?
    };
    let ingest_ms = t_ingest.elapsed().as_millis() as u64;

    // Emit total ingest time to OTel (Jaeger traces already show the
    // per-stage breakdown via #[instrument] spans on docling_client,
    // graph_extractor.extract_batch, summarizer.summarize_batch, embedder).
    metrics.record_ingest_stage_ms("total", ingest_ms as f64);
    metrics.record_ingest_stage_ms("pipeline", result.duration_ms as f64);

    // ---- Pretty summary ----
    println!();
    println!("================ INGEST SUMMARY ================");
    println!("  doc_id              : {}", result.doc_id);
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
