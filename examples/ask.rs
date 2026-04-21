//! Dev CLI: ask a question through the full agent pipeline.
//!
//! Unlike `examples/query.rs` (which calls the retrieval router directly and
//! returns ranked chunks), this runs the complete `AgentOrchestrator`:
//!
//! 1. **Context build** — loads short conversation history + memory hints.
//! 2. **Tool-loop** — LLM agent decides which tools to call iteratively:
//!    `search_knowledge`, `search_web`, `fetch_url`, `remember`,
//!    `document_info`. Each tool call accrues evidence into the context.
//! 3. **Evidence fuse** — merge KB + web hits via RRF.
//! 4. **Grounding verification** — second LLM checks whether every claim in
//!    the draft answer is supported by the evidence. Failure triggers a
//!    retry with a different retrieval strategy.
//! 5. **Response synthesis** — final answer with `[1] [2]` citations.
//!
//! Usage:
//!
//! ```bash
//! # 3 positional args: user_id, session_id (empty string = new random session), question
//! just ask "alice" "" "what is X?"
//!
//! # with optional flags
//! just ask "alice" "" "what is X?" --strategy hybrid --limit 5
//!
//! # or directly:
//! cargo run --release --example ask -- "alice" "" "my question"
//! ```
//!
//! Prerequisites: `.env` loaded, docker services up, `AGENT_RS__LLM__API_KEY`
//! set, at least one doc already ingested via `just ingest`.

use std::sync::Arc;
use std::time::Instant;

use agent_rs::embedder::build_embedder;
use agent_rs::router::{RouterFeatures, build_router};
use agent_rs::stores::connect_stores;
use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::config::Settings;
use agent_rs_core::types::{CollectionName, RetrievalStrategy, SessionId, UserId};
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use anyhow::{Context, Result, bail};
use uuid::Uuid;

struct Args {
    user_id: UserId,
    session_id: SessionId,
    question: String,
    strategy: RetrievalStrategy,
    limit: usize,
}

fn parse_args() -> Result<Args> {
    let mut positional: Vec<String> = Vec::new();
    let mut strategy_str: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--strategy" | "-s" => {
                strategy_str = Some(iter.next().context("--strategy needs a value")?);
            }
            "--limit" | "-n" => {
                let v = iter.next().context("--limit needs a value")?;
                limit = Some(v.parse().with_context(|| format!("invalid --limit: {v}"))?);
            }
            "-h" | "--help" => {
                println!(
                    "Usage: ask <user_id> <session_id_or_empty> <question> [--strategy <name>] [--limit N]\n\
                     Strategies: semantic keyword hybrid graph graph_hybrid graph_cot\n\
                                graph_constrained rag_completion summaries temporal\n\
                     Pass empty string for session_id to generate a new random session."
                );
                std::process::exit(0);
            }
            flag if flag.starts_with("--") => bail!("unknown flag: {flag}"),
            arg => positional.push(arg.to_owned()),
        }
    }
    if positional.len() != 3 {
        bail!(
            "usage: just ask \"<user_id>\" \"<session_id_or_empty>\" \"<question>\" — got {} positional arg(s)",
            positional.len()
        );
    }
    let user_id = UserId::try_new(&positional[0]).context("invalid user_id")?;
    let raw_session = positional[1].as_str();
    let session_id = if raw_session.is_empty() {
        SessionId::from_uuid(Uuid::new_v4())
    } else {
        SessionId::from_uuid(
            Uuid::parse_str(raw_session).context("invalid session UUID")?,
        )
    };
    let question = positional[2].clone();
    let strategy = match strategy_str.as_deref() {
        Some(s) => RetrievalStrategy::from_db_str(s)
            .map_err(|e| anyhow::anyhow!("unknown strategy '{s}': {e}"))?,
        None => RetrievalStrategy::GraphHybrid,
    };
    Ok(Args {
        user_id,
        session_id,
        question,
        strategy,
        limit: limit.unwrap_or(8),
    })
}

fn print_wrapped(prefix: &str, text: &str, width: usize) {
    let indent = " ".repeat(prefix.chars().count());
    let mut first = true;
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.chars().count() + word.chars().count() + 1 > width {
            if first {
                println!("{prefix}{line}");
                first = false;
            } else {
                println!("{indent}{line}");
            }
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        if first {
            println!("{prefix}{line}");
        } else {
            println!("{indent}{line}");
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    if std::env::var_os("RUST_LOG").is_none() {
        // SAFETY: we only set this when the caller did not.
        unsafe {
            std::env::set_var(
                "RUST_LOG",
                "info,agent_rs_retrieval=debug,agent_rs_agent=debug",
            )
        };
    }

    let settings = Settings::load().context("load settings")?;
    let _otel_guard = init_tracing(&settings.otel, "agent-rs-ask")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let t_boot = Instant::now();
    let stores = connect_stores(&settings).await?;
    let (embedder, embedder_label) = build_embedder().await?;
    let boot_ms = t_boot.elapsed().as_millis() as u64;

    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));

    let metrics = Arc::new(RetrievalMetrics::new("agent-rs-ask"));

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

    let orchestrator = AgentOrchestrator::new(
        &settings,
        stores.neo4j.clone(),
        stores.postgres.clone(),
        stores.qdrant.clone(),
        collection.as_str().to_owned(),
        summaries_collection.as_str().to_owned(),
        Arc::clone(&router),
        Arc::clone(&metrics),
    )
    .context("build AgentOrchestrator")?;

    println!();
    println!("================ ASK ===========================");
    println!("  question    : {}", args.question);
    println!("  strategy    : {:?}", args.strategy);
    println!("  limit       : {}", args.limit);
    println!("  user        : {}", args.user_id);
    println!("  session     : {}", args.session_id);
    println!("  collection  : {}", collection);
    println!("  embedder    : {}", embedder_label);
    println!("  boot_ms     : {}", boot_ms);
    println!("================================================");
    println!();

    let t = Instant::now();
    let response = orchestrator
        .process_query(
            args.question,
            args.session_id,
            args.user_id,
            args.strategy,
            args.limit,
        )
        .await?;
    let elapsed_ms = t.elapsed().as_millis() as u64;

    println!();
    println!("================ ANSWER ========================");
    print_wrapped("", &response.answer, 88);
    println!();
    println!("------------------------------------------------");
    println!("  is_grounded : {}", response.is_grounded);
    println!("  confidence  : {:.3}", response.confidence.value());
    println!("  strategy    : {:?}", response.strategy);
    println!("  citations   : {}", response.citations.len());
    println!("  elapsed_ms  : {}", elapsed_ms);
    println!("------------------------------------------------");

    if response.citations.is_empty() {
        println!("  (no citations)");
    } else {
        for c in &response.citations {
            println!("[{}] {}  (score={:.4})", c.index, c.source, c.score);
            print_wrapped("    ", &c.content_snippet, 84);
            println!();
        }
    }
    println!("================================================");

    Ok(())
}
