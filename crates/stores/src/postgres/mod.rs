//! PostgreSQL-backed implementation of [`StateStore`].
//!
//! Uses Diesel 2 with `diesel-async` and a `bb8` connection pool so that
//! every query is non-blocking. Connection management is encapsulated
//! inside [`PgStore`]; callers never interact with the pool directly.
//!
//! # Tables owned by this module
//!
//! - `memories` — user-level long-term facts
//! - `query_runs` — per-Ask telemetry rows (correlated by `trace_id`)
//! - `evidence_items` — evidence surfaced per run (FK cascade from runs)
//! - `strategy_performance` — adaptive strategy learning
//! - `document_index_state` — MinIO key → indexing state machine (see
//!   [`document_index_state`] submodule)
//!
//! Sessions / chat_messages / ingested_documents USED to live here but
//! were relocated to agent-rs-web during the 2026-04-21 stateless pivot.
//!
//! # Error handling
//!
//! All Diesel errors are mapped to [`StoreError::Postgres`] via the
//! `#[from]` impl in `agent-rs-core`. Pool checkout errors are mapped
//! to [`StoreError::ConnectionPool`].

use agent_rs_core::{StoreError, UserId};
use async_trait::async_trait;
use diesel::prelude::*;
use diesel_async::{
    AsyncPgConnection, RunQueryDsl,
    pooled_connection::{AsyncDieselConnectionManager, bb8::Pool},
};
use tracing::instrument;
use uuid::Uuid;

use crate::traits::StateStore;
use crate::types::{EvidenceItem, Memory, QueryRun};

pub mod document_index_state;
pub mod models;
pub mod schema;

// ---------------------------------------------------------------------------
// Shared raw-query helpers
// ---------------------------------------------------------------------------

/// Projection type for `SELECT COUNT(*)::bigint AS count` raw queries.
#[derive(Debug, diesel::QueryableByName)]
pub struct CountRow {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub count: i64,
}

use models::{EvidenceItemRow, MemoryRow, NewEvidenceItem, NewMemory, NewQueryRun, QueryRunRow};
use schema::{evidence_items, memories, query_runs};

// ---------------------------------------------------------------------------
// PgStore
// ---------------------------------------------------------------------------

/// Sizing parameters for the bb8 connection pool.
///
/// Defaults chosen for a single-node Postgres sized against
/// docker-compose's `max_connections = 200`: 50 conns per agent-rs
/// process leaves headroom for the web app, the indexer-worker fleet,
/// and the Postgres admin session. Bump via
/// [`PgStoreConfig::max_connections`] / `..::connection_timeout_ms` on a
/// sharded/replicated setup.
#[derive(Debug, Clone, Copy)]
pub struct PgStoreConfig {
    /// Hard cap on simultaneous pooled connections. A caller that misses
    /// the cap gets a [`StoreError::ConnectionPool`] after
    /// `connection_timeout_ms` elapses.
    pub max_connections: u32,
    /// How long [`PgStore::conn`] waits for a free slot before erroring.
    pub connection_timeout_ms: u64,
}

impl Default for PgStoreConfig {
    fn default() -> Self {
        Self {
            max_connections: 50,
            connection_timeout_ms: 5_000,
        }
    }
}

#[derive(Clone)]
pub struct PgStore {
    pool: Pool<AsyncPgConnection>,
}

impl PgStore {
    /// Create a new [`PgStore`] with the default [`PgStoreConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ConnectionPool`] if the pool fails to build or
    /// if no initial connection can be established within
    /// `connection_timeout_ms`.
    pub async fn new(database_url: &str) -> Result<Self, StoreError> {
        Self::with_config(database_url, PgStoreConfig::default()).await
    }

    /// Create a new [`PgStore`] with a caller-supplied [`PgStoreConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ConnectionPool`] if the pool fails to build
    /// or if no initial connection can be established.
    pub async fn with_config(
        database_url: &str,
        config: PgStoreConfig,
    ) -> Result<Self, StoreError> {
        let manager = AsyncDieselConnectionManager::<AsyncPgConnection>::new(database_url);
        let pool = Pool::builder()
            .max_size(config.max_connections)
            .connection_timeout(std::time::Duration::from_millis(
                config.connection_timeout_ms,
            ))
            .build(manager)
            .await
            .map_err(|e| StoreError::ConnectionPool(e.to_string()))?;
        Ok(Self { pool })
    }

    /// Obtain a pooled connection, mapping pool errors to [`StoreError`].
    pub(crate) async fn conn(
        &self,
    ) -> Result<
        diesel_async::pooled_connection::bb8::PooledConnection<'_, AsyncPgConnection>,
        StoreError,
    > {
        self.pool
            .get()
            .await
            .map_err(|e| StoreError::ConnectionPool(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// StateStore implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl StateStore for PgStore {
    #[instrument(skip(self), fields(user_id = %user_id))]
    async fn get_memories(&self, user_id: &UserId) -> Result<Vec<Memory>, StoreError> {
        let mut conn = self.conn().await?;
        // Uses idx_memories_user_created (user_id, created_at DESC).
        // We reverse client-side to return oldest-first (callers expect
        // chronological order for prompt assembly).
        let mut rows: Vec<MemoryRow> = memories::table
            .filter(memories::user_id.eq(user_id.as_str()))
            .order(memories::created_at.desc())
            .select(MemoryRow::as_select())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)?;
        rows.reverse();

        rows.into_iter()
            .map(|r| r.into_domain())
            .collect::<Result<Vec<_>, _>>()
    }

    #[instrument(skip(self), fields(memory_id = %memory.id))]
    async fn store_memory(&self, memory: &Memory) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let new_row = NewMemory::from_domain(memory);
        diesel::insert_into(memories::table)
            .values(&new_row)
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    #[instrument(skip(self), fields(run_id = %run.id))]
    async fn persist_run(&self, run: &QueryRun) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let new_row = NewQueryRun::from_domain(run);
        diesel::insert_into(query_runs::table)
            .values(&new_row)
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    #[instrument(skip(self), fields(run_id = %id))]
    async fn get_run(&self, id: &Uuid) -> Result<Option<QueryRun>, StoreError> {
        let mut conn = self.conn().await?;
        let maybe_row: Option<QueryRunRow> = query_runs::table
            .filter(query_runs::id.eq(id))
            .select(QueryRunRow::as_select())
            .first(&mut conn)
            .await
            .optional()
            .map_err(StoreError::from)?;

        maybe_row.map(|r| r.into_domain()).transpose()
    }

    #[instrument(skip(self), fields(evidence_id = %item.id, run_id = %item.run_id))]
    async fn store_evidence(&self, item: &EvidenceItem) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let new_row = NewEvidenceItem::from_domain(item);
        diesel::insert_into(evidence_items::table)
            .values(&new_row)
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    #[instrument(skip(self, items), fields(run_id = ?items.first().map(|i| i.run_id), count = items.len()))]
    async fn store_evidence_batch(&self, items: &[EvidenceItem]) -> Result<(), StoreError> {
        if items.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn().await?;
        let rows: Vec<NewEvidenceItem<'_>> =
            items.iter().map(NewEvidenceItem::from_domain).collect();
        diesel::insert_into(evidence_items::table)
            .values(&rows)
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    #[instrument(skip(self), fields(run_id = %run_id))]
    async fn get_evidence(&self, run_id: &Uuid) -> Result<Vec<EvidenceItem>, StoreError> {
        let mut conn = self.conn().await?;
        let rows: Vec<EvidenceItemRow> = evidence_items::table
            .filter(evidence_items::run_id.eq(run_id))
            .order(evidence_items::created_at.asc())
            .select(EvidenceItemRow::as_select())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)?;

        rows.into_iter()
            .map(|r| r.into_domain())
            .collect::<Result<Vec<_>, _>>()
    }
}

// ---------------------------------------------------------------------------
// User-scoped memory helpers (direct PgStore methods)
// ---------------------------------------------------------------------------
//
// These enforce `user_id` at the SQL level — an independent guard beneath
// any application-layer authorisation check.

impl PgStore {
    /// Persist a memory entry for `user_id`.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — database-level error.
    #[instrument(skip(self), fields(user_id = %user_id))]
    pub async fn save_memory(&self, user_id: &UserId, content: &str) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let new_row = NewMemory {
            id: Uuid::new_v4(),
            user_id: user_id.as_str(),
            content,
        };
        diesel::insert_into(memories::table)
            .values(&new_row)
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    /// Fetch all memories belonging to `user_id`, ordered oldest first.
    ///
    /// Cross-user reads always return an empty vec — the `WHERE user_id = ?`
    /// predicate is enforced at the database level.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — database-level error.
    #[instrument(skip(self), fields(user_id = %user_id))]
    pub async fn list_memories(&self, user_id: &UserId) -> Result<Vec<Memory>, StoreError> {
        // Reuses the StateStore impl so indexing + ordering stay consistent.
        <Self as StateStore>::get_memories(self, user_id).await
    }
}
