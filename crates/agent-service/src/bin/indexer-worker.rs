//! Indexer worker binary — long-running RabbitMQ consumer that runs the
//! document indexing pipeline.
//!
//! Spin one instance per replica in docker-compose / Kubernetes — the
//! broker fans out jobs, `with_prefetch(N)` caps in-flight work per
//! replica, and `AGENT_RS__INDEXER__PREFETCH` lets operators tune without
//! redeploying.
//!
//! All wiring is identical to `agent-service::main` up to the
//! orchestrator: same stores, same embedder, same metrics. We deliberately
//! skip the tonic server + the retrieval router (the worker never answers
//! gRPC and never runs agent tools) to keep process memory low.

use std::sync::Arc;
use std::time::Duration;

use agent_rs::bootstrap::{ensure_collections, ensure_graph_indexes};
use agent_rs::embedder::build_embedder;
use agent_rs::stores::connect_stores;
use agent_rs_agent_service::indexer::{AgentIndexHandler, AgentIndexHandlerDeps};
use agent_rs_core::config::{DEFAULT_LLM_MODEL, IngestBackend, Settings};
use agent_rs_core::types::CollectionName;
use agent_rs_observability::metrics::RetrievalMetrics;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_queue::run_indexer;
use agent_rs_retrieval::ingestion::docling_client::DoclingSidecarClient;
use agent_rs_retrieval::ingestion::marker_client::MarkerSidecarClient;
use agent_rs_retrieval::ingestion::summarizer::ChunkSummarizer;
use agent_rs_retrieval::ingestion::{ChunkerConfig, LlmGraphExtractor};
use anyhow::{Context, Result};
use tracing::{info, warn};

/// Default number of in-flight jobs per worker process. Tuned for the
/// MVP pipeline (~200 MiB peak per job). Override via
/// `AGENT_RS__INDEXER__PREFETCH` without rebuilding.
const DEFAULT_PREFETCH: u16 = 5;

/// HTTP client request timeout for every LLM call issued by the
/// indexer (`ChunkSummarizer` + `LlmGraphExtractor`).
///
/// Sized for cold-start latency on GPT-class models — the embedding
/// endpoint usually answers in < 3 s but a large summarise-this-chunk
/// prompt against a model that just cold-started can run 30-60 s. The
/// 120 s budget keeps the first-request case responsive while still
/// catching hung connections before the RabbitMQ prefetch window
/// starves. Not configurable via env to keep worker tuning hermetic;
/// raise this const if the configured LLM regularly exceeds 120 s.
const LLM_HTTP_CLIENT_TIMEOUT: Duration = Duration::from_secs(120);

/// Parse a boolean-shaped env var. Accepts `1`, `true` (case-insensitive);
/// everything else, including unset, is `false`.
fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::load().context("load settings")?;

    let _otel_guard = init_tracing(&settings.otel, "agent-rs-indexer-worker")
        .context("init_tracing — is the otel-collector running? `just obs-up`")?;

    let stores = connect_stores(&settings).await.context("connect_stores")?;
    let (embedder, embedder_label) = build_embedder().await.context("build_embedder")?;
    info!(embedder = %embedder_label, "embedder ready");

    let collection = CollectionName::from_string(settings.retrieval.dataset.clone());
    let summaries_collection =
        CollectionName::from_string(format!("{}_summaries", collection.as_str()));
    let triplet_collection =
        CollectionName::from_string(format!("{}_triplets", collection.as_str()));

    // Idempotent bootstrap — create collections on cold clusters, no-op
    // on warm ones. Without this the first upsert after a fresh
    // `docker compose up` would fail with "collection not found".
    ensure_collections(
        &stores.qdrant,
        embedder.dimensions(),
        &collection,
        &summaries_collection,
        &triplet_collection,
    )
    .await
    .context("ensure Qdrant collections")?;

    // Idempotent Neo4j bootstrap — the indexer writes Document /
    // Page / Section / Chunk nodes whose uniqueness constraints live
    // alongside `entity_mention_fts`. Running the same helper here as
    // in agent-service keeps constraint creation race-free: whichever
    // binary boots first on a cold cluster creates the schema, the
    // second is a no-op.
    ensure_graph_indexes(&stores.neo4j)
        .await
        .context("ensure Neo4j graph indexes")?;
    // Metrics handle kept for parity with the gRPC service and to keep
    // the retrieval-metrics registration path hot for future Qdrant /
    // Neo4j instrumentation of the indexing hot loop.
    let _metrics = Arc::new(RetrievalMetrics::new("agent-rs-indexer-worker"));

    // Docling sidecar — wire only when the configured ingest backend
    // wants it. With `DoclingWithLegacyFallback` a connect failure is
    // non-fatal: we log it and let the pdf-extract path carry the load
    // until the sidecar comes back.
    let docling: Option<Arc<DoclingSidecarClient>> = match settings.ingest_backend {
        IngestBackend::Docling | IngestBackend::DoclingWithLegacyFallback => {
            match DoclingSidecarClient::connect(
                settings.docling.endpoint.clone(),
                Duration::from_secs(settings.docling.connect_timeout_s),
                Duration::from_secs(settings.docling.stream_idle_timeout_s),
            )
            .await
            {
                Ok(client) => {
                    info!(
                        endpoint = %settings.docling.endpoint,
                        "Docling sidecar client ready"
                    );
                    Some(Arc::new(client))
                }
                Err(e)
                    if matches!(
                        settings.ingest_backend,
                        IngestBackend::DoclingWithLegacyFallback
                    ) =>
                {
                    warn!(error = %e, "Docling sidecar unreachable; falling back to pdf-extract");
                    None
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Docling backend required but sidecar unavailable: {e}"
                    ));
                }
            }
        }
        IngestBackend::Marker | IngestBackend::MarkerWithLegacyFallback | IngestBackend::Legacy => {
            None
        }
    };

    // Marker sidecar client. Constructed only when the active backend
    // is `Marker` / `MarkerWithLegacyFallback` so we don't speculatively
    // open a gRPC channel to a sidecar that isn't running.
    let marker: Option<Arc<MarkerSidecarClient>> = match settings.ingest_backend {
        IngestBackend::Marker | IngestBackend::MarkerWithLegacyFallback => {
            match MarkerSidecarClient::connect(
                settings.marker.endpoint.clone(),
                Duration::from_secs(settings.marker.connect_timeout_s),
                Duration::from_secs(settings.marker.stream_idle_timeout_s),
            )
            .await
            {
                Ok(client) => {
                    info!(
                        endpoint = %settings.marker.endpoint,
                        "Marker sidecar client ready"
                    );
                    Some(Arc::new(client))
                }
                Err(e)
                    if matches!(
                        settings.ingest_backend,
                        IngestBackend::MarkerWithLegacyFallback
                    ) =>
                {
                    warn!(error = %e, "Marker sidecar unreachable; falling back to pdf-extract");
                    None
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Marker backend required but sidecar unavailable: {e}"
                    ));
                }
            }
        }
        IngestBackend::Docling
        | IngestBackend::DoclingWithLegacyFallback
        | IngestBackend::Legacy => None,
    };

    // Optional LLM stages (per-chunk summaries + KG triplets).
    //
    // Policy — **fail-open when an LLM key is available**. Summaries and
    // KG-triplet extraction are what make `RetrievalStrategy::GraphHybrid`
    // (the documented flagship strategy, `uses_triplets = true`) produce
    // its full signal set. Leaving them off-by-default silently degrades
    // retrieval quality on every fresh deployment. Enterprise deployments
    // typically *already* gate LLM spend at the key level — the indexer
    // does not need a second off-switch.
    //
    // Controls:
    //   - `OPENAI_API_KEY` / `LLM_API_KEY` — presence of either enables
    //     both stages. The key lives in an env var, not config, so it is
    //     never written to disk by a misconfigured operator.
    //   - `AGENT_RS__INDEXER__DISABLE_SUMMARIES=true` — explicit opt-out.
    //   - `AGENT_RS__INDEXER__DISABLE_KG=true`        — explicit opt-out.
    //
    // **Model name** — sourced from `settings.llm.model`
    // (`AGENT_RS__LLM__MODEL` env var overrides the `config.toml` value
    // via figment). Falls back to [`DEFAULT_LLM_MODEL`] only when the
    // `Settings::default()` path fires and the operator has supplied no
    // override. A legacy `LLM_MODEL` env var is no longer honoured here
    // because it would diverge from the value the retrieval path reads
    // from `settings.llm.model`.
    //
    // Missing LLM key → both stages off, with a single WARN so operators
    // see the degraded retrieval capability at startup rather than
    // discover it by an empty `default_triplets` collection weeks later.
    let llm_api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .ok();
    let llm_model = if settings.llm.model.is_empty() {
        warn!(
            default_model = DEFAULT_LLM_MODEL,
            "settings.llm.model is empty — using compile-time default; set AGENT_RS__LLM__MODEL to override"
        );
        DEFAULT_LLM_MODEL.to_owned()
    } else {
        settings.llm.model.clone()
    };
    let http_client = Arc::new(
        reqwest::Client::builder()
            .timeout(LLM_HTTP_CLIENT_TIMEOUT)
            .build()
            .context("build shared HTTP client")?,
    );

    let summaries_opt_out = env_flag("AGENT_RS__INDEXER__DISABLE_SUMMARIES");
    let kg_opt_out = env_flag("AGENT_RS__INDEXER__DISABLE_KG");

    let summarizer = match (llm_api_key.as_deref(), summaries_opt_out) {
        (Some(key), false) => {
            info!(
                model = %llm_model,
                control = "AGENT_RS__INDEXER__DISABLE_SUMMARIES",
                "summarizer enabled (opt out with DISABLE_SUMMARIES=true)"
            );
            Some(Arc::new(ChunkSummarizer::new(
                key.to_owned(),
                llm_model.clone(),
                Arc::clone(&http_client),
            )))
        }
        (Some(_), true) => {
            info!(
                control = "AGENT_RS__INDEXER__DISABLE_SUMMARIES",
                "summarizer disabled by operator — `RetrievalStrategy::Summaries` will be empty"
            );
            None
        }
        (None, _) => {
            warn!(
                reason = "no_llm_key",
                "summarizer disabled: neither OPENAI_API_KEY nor LLM_API_KEY set — \
                 `default_summaries` will remain empty and summary-aware retrieval is degraded"
            );
            None
        }
    };

    let extractor = match (llm_api_key.as_deref(), kg_opt_out) {
        (Some(key), false) => {
            info!(
                model = %llm_model,
                control = "AGENT_RS__INDEXER__DISABLE_KG",
                "KG extractor enabled (opt out with DISABLE_KG=true)"
            );
            Some(Arc::new(LlmGraphExtractor::new(
                key.to_owned(),
                llm_model.clone(),
                Arc::clone(&http_client),
            )))
        }
        (Some(_), true) => {
            info!(
                control = "AGENT_RS__INDEXER__DISABLE_KG",
                "KG extractor disabled by operator — `default_triplets` will stay empty \
                 and `RetrievalStrategy::GraphHybrid` will run with reduced signal coverage"
            );
            None
        }
        (None, _) => {
            warn!(
                reason = "no_llm_key",
                "KG extractor disabled: neither OPENAI_API_KEY nor LLM_API_KEY set — \
                 `default_triplets` will remain empty and graph-hybrid retrieval is degraded"
            );
            None
        }
    };

    let handler = Arc::new(AgentIndexHandler::new(AgentIndexHandlerDeps {
        minio: Arc::new(
            agent_rs_stores::minio::MinioStore::new(settings.minio.clone())
                .await
                .context("connect MinIO")?,
        ),
        embedder,
        qdrant: stores.qdrant.clone(),
        neo4j: stores.neo4j.clone(),
        pg: stores.postgres.clone(),
        chunk_collection: collection,
        summaries_collection,
        triplet_collection,
        chunker_config: ChunkerConfig::default(),
        docling,
        marker,
        ingest_backend: settings.ingest_backend,
        summarizer,
        extractor,
    }));

    let prefetch: u16 = std::env::var("AGENT_RS__INDEXER__PREFETCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PREFETCH);

    info!(
        endpoint = %settings.rabbitmq.endpoint,
        prefetch,
        "indexer worker starting"
    );

    // `run_indexer` blocks until the consumer loop ends (broker drop or
    // an unrecoverable error); any exit here means the supervisor
    // (docker-compose / k8s) should restart us.
    run_indexer(&settings.rabbitmq.endpoint, prefetch, handler)
        .await
        .context("run_indexer")?;

    Ok(())
}
