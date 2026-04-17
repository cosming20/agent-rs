//! Dev CLI: run a single retrieval query and print results + internals.
//!
//! Usage:
//!
//! ```bash
//! # Default strategy (graph_hybrid) and limit (10)
//! cargo run --release --example query -- "ce este histograma?"
//!
//! # Override strategy / limit
//! cargo run --release --example query -- --strategy hybrid --limit 5 "metoda bhattacharyya"
//!
//! # Valid strategies: semantic | keyword | hybrid | graph | graph_hybrid |
//! # graph_cot | graph_constrained | rag_completion | summaries | temporal |
//! # triplet
//! ```
//!
//! Prerequisites: `.env` loaded, docker services up, `OPENAI_API_KEY` set,
//! at least one document already ingested (use `examples/ingest.rs` first).
//!
//! The example configures `RUST_LOG=info,agent_rs_retrieval=debug` by
//! default so the router's internal spans surface inline:
//!
//! - `query_rewriter` — sub-queries produced (or "skipped" for short queries)
//! - `weighted_keyword_extractor` — extracted terms + weights + phrase flags
//! - `metadata_filter` — structural hints
//! - `entity_linker` — linked entities (prefer_graph flag)
//! - per-strategy search calls (vector, keyword, graph, hybrid, triplet)
//! - RRF fusion counts
//! - cross-encoder rerank + weighted-keyword adjustment before/after
//! - context expansion (if enabled)
//!
//! After the trace, a compact table of the top-K results is printed.

use std::sync::Arc;
use std::time::Instant;

use agent_rs::embedder::build_embedder;
use agent_rs::router::{build_router, RouterFeatures};
use agent_rs::stores::connect_stores;
use agent_rs_core::config::Settings;
use agent_rs_core::types::{CollectionName, RetrievalStrategy};
use agent_rs_observability::metrics::RetrievalMetrics;
use anyhow::{bail, Context, Result};

struct Args {
    query: String,
    strategy: RetrievalStrategy,
    limit: usize,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut strategy_str: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut query_parts: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--strategy" | "-s" => {
                strategy_str = Some(args.next().context("--strategy needs a value")?);
            }
            "--limit" | "-n" => {
                let v = args.next().context("--limit needs a value")?;
                limit = Some(v.parse().with_context(|| format!("invalid --limit: {v}"))?);
            }
            "-h" | "--help" => {
                println!(
                    "Usage: query [--strategy <name>] [--limit <N>] <query text...>\n\
                     Strategies: semantic keyword hybrid graph graph_hybrid graph_cot\n\
                                graph_constrained rag_completion summaries temporal triplet"
                );
                std::process::exit(0);
            }
            flag if flag.starts_with("--") => {
                bail!("unknown flag: {flag}");
            }
            positional => query_parts.push(positional.to_owned()),
        }
    }
    if query_parts.is_empty() {
        bail!("missing query text; see --help");
    }
    let strategy = match strategy_str.as_deref() {
        Some(s) => RetrievalStrategy::from_db_str(s)
            .map_err(|e| anyhow::anyhow!("unknown strategy '{s}': {e}"))?,
        None => RetrievalStrategy::GraphHybrid,
    };
    Ok(Args {
        query: query_parts.join(" "),
        strategy,
        limit: limit.unwrap_or(10),
    })
}

fn truncate_snippet(text: &str, max_chars: usize) -> String {
    let trimmed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() <= max_chars {
        trimmed
    } else {
        let mut acc = String::with_capacity(max_chars + 1);
        for (i, c) in trimmed.chars().enumerate() {
            if i >= max_chars {
                acc.push('…');
                break;
            }
            acc.push(c);
        }
        acc
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    if std::env::var_os("RUST_LOG").is_none() {
        // SAFETY: we only set this when the caller did not.
        // debug-level on agent_rs_retrieval surfaces the per-stage tracing
        // spans (sub-queries, weighted keywords, rerank adjustments).
        unsafe { std::env::set_var("RUST_LOG", "info,agent_rs_retrieval=debug") };
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let settings = Settings::load().context("load settings")?;
    let t_boot = Instant::now();
    let stores = connect_stores(&settings).await?;
    let (embedder, embedder_label) = build_embedder().await?;
    let boot_ms = t_boot.elapsed().as_millis() as u64;

    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));

    let metrics = Arc::new(RetrievalMetrics::new("query-cli"));

    let router = build_router(
        &settings,
        &stores,
        embedder,
        &collection,
        &triplet_collection,
        &summaries_collection,
        &RouterFeatures::from_env(),
        Arc::clone(&metrics),
    )
    .await?;

    println!();
    println!("================ QUERY =========================");
    println!("  query       : {}", args.query);
    println!("  strategy    : {:?}", args.strategy);
    println!("  limit       : {}", args.limit);
    println!("  collection  : {}", collection);
    println!("  embedder    : {}", embedder_label);
    println!("  boot_ms     : {}", boot_ms);
    println!("================================================");
    println!();

    let t_retrieve = Instant::now();
    let pack = router.retrieve(&args.query, args.strategy, args.limit).await?;
    let retrieve_ms = t_retrieve.elapsed().as_millis() as u64;

    println!();
    println!("================ RESULTS =======================");
    println!("  wall-clock_ms      : {}", retrieve_ms);
    println!("  documents_returned : {}", pack.documents.len());
    println!("  source_counts      : {:?}", pack.source_counts);
    println!("------------------------------------------------");
    if pack.documents.is_empty() {
        println!("  (no documents matched)");
    } else {
        for (rank, doc) in pack.documents.iter().enumerate() {
            let text = doc
                .payload
                .get("chunk_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let snippet = truncate_snippet(text, 160);
            let section = doc
                .payload
                .get("section_title")
                .and_then(|v| v.as_str())
                .unwrap_or("-");
            let page = doc
                .payload
                .get("page_start")
                .and_then(|v| v.as_u64())
                .map(|p| p.to_string())
                .unwrap_or_else(|| "?".into());
            println!(
                "#{:02} score={:.4}  page={:>3}  section={:.40}",
                rank + 1,
                doc.score.value(),
                page,
                section
            );
            println!("     id={}", doc.id);
            println!("     {}", snippet);
        }
    }
    println!("================================================");

    Ok(())
}
