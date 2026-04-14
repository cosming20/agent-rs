//! Storage backend initialization.

use std::sync::Arc;

use agent_rs_core::config::Settings;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::qdrant::QdrantStore;
use anyhow::{Context, Result};
use tracing::info;

/// All connected storage backends.
pub struct Stores {
    pub qdrant: Arc<QdrantStore>,
    pub neo4j: Arc<Neo4jStore>,
    pub postgres: Arc<PgStore>,
}

/// Connect to Qdrant, Neo4j, and Postgres from settings.
pub async fn connect_stores(settings: &Settings) -> Result<Stores> {
    info!("connecting to Qdrant...");
    let qdrant =
        Arc::new(QdrantStore::new(&settings.qdrant).context("failed to create Qdrant client")?);

    info!("connecting to Neo4j...");
    let neo4j = Arc::new(
        Neo4jStore::connect(&settings.neo4j)
            .await
            .context("failed to connect to Neo4j")?,
    );

    info!("connecting to Postgres...");
    let postgres = Arc::new(
        PgStore::new(&settings.postgres.dsn())
            .await
            .context("failed to connect to Postgres")?,
    );

    info!("all storage backends connected");
    Ok(Stores {
        qdrant,
        neo4j,
        postgres,
    })
}
