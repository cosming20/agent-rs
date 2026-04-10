//! agent-rs binary entry point.
//!
//! Loads configuration, initialises observability, connects to all
//! storage backends, builds the retrieval router and agent orchestrator,
//! then starts the gRPC server.

use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

use agent_rs_agent::orchestrator::AgentOrchestrator;
use agent_rs_core::config::Settings;
use agent_rs_core::types::CollectionName;
use agent_rs_observability::tracing::init_tracing;
use agent_rs_retrieval::embedder::{Embedder, FastEmbedder, FastEmbedderConfig};
use agent_rs_retrieval::ranker::ResultRanker;
use agent_rs_retrieval::router::RetrievalRouter;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load configuration (env vars, .env file, config.toml, defaults)
    let settings = Settings::load().context("failed to load configuration")?;

    // 2. Initialise observability (tracing + OTel + metrics)
    let _guard =
        init_tracing(&settings.otel, "agent-rs").context("failed to initialise tracing")?;

    info!(
        model = %settings.llm.model,
        qdrant = %settings.qdrant.grpc_url(),
        postgres = format!("{}:{}", settings.postgres.host, settings.postgres.port),
        neo4j = %settings.neo4j.endpoint(),
        server = %settings.server.bind_addr(),
        "agent-rs starting"
    );

    // 3. Connect to storage backends
    let qdrant_store =
        QdrantStore::new(&settings.qdrant).context("failed to create Qdrant client")?;
    let qdrant_store: Arc<QdrantStore> = Arc::new(qdrant_store);

    let neo4j_store = Neo4jStore::connect(&settings.neo4j)
        .await
        .context("failed to connect to Neo4j")?;
    let neo4j_store: Arc<Neo4jStore> = Arc::new(neo4j_store);

    let pg_store = PgStore::new(&settings.postgres.dsn())
        .await
        .context("failed to connect to PostgreSQL")?;
    let pg_store: Arc<PgStore> = Arc::new(pg_store);

    info!("all storage backends connected");

    // 4. Create embedder (local ONNX inference via fastembed)
    let embedder = FastEmbedder::try_new(FastEmbedderConfig::default())
        .await
        .context("failed to initialise embedding models")?;
    let embedder: Arc<FastEmbedder> = Arc::new(embedder);

    info!(dims = embedder.dimensions(), "embedder initialised");

    // 5. Create retrieval router
    let ranker = ResultRanker::new(settings.ranker.rrf_k);
    let collection = CollectionName(settings.retrieval.dataset.clone());
    let router = Arc::new(RetrievalRouter::new(
        embedder,
        qdrant_store,
        neo4j_store,
        ranker,
        collection,
    ));

    info!("retrieval router created");

    // 6. Create orchestrator (wires all pipeline steps)
    let orchestrator = AgentOrchestrator::new(&settings, pg_store, Arc::clone(&router))
        .context("failed to create agent orchestrator")?;
    let orchestrator = Arc::new(orchestrator);

    info!("orchestrator created, starting gRPC server");

    // 7. Start gRPC server (blocks until shutdown)
    agent_rs_server::run(&settings, orchestrator)
        .await
        .context("gRPC server failed")?;

    Ok(())
}
