//! PostgreSQL-backed implementation of [`StateStore`].
//!
//! Uses Diesel 2 with `diesel-async` and a `bb8` connection pool so that
//! every query is non-blocking. Connection management is encapsulated
//! inside [`PgStore`]; callers never interact with the pool directly.
//!
//! # Construction
//!
//! ```rust,ignore
//! let store = PgStore::new("postgres://user:pass@localhost/mydb").await?;
//! ```
//!
//! # Error handling
//!
//! All Diesel errors are mapped to [`StoreError::Postgres`] via the
//! `#[from]` impl in `agent-rs-core`. Pool checkout errors are mapped
//! to [`StoreError::ConnectionPool`].

use agent_rs_core::{SessionId, StoreError, UserId};
use async_trait::async_trait;
use diesel::prelude::*;
use diesel_async::{
    pooled_connection::{bb8::Pool, AsyncDieselConnectionManager},
    AsyncPgConnection, RunQueryDsl,
};
use tracing::instrument;
use uuid::Uuid;

use crate::traits::StateStore;
use crate::types::{ChatMessage, EvidenceItem, Memory, QueryRun};

pub mod models;
pub mod schema;

use models::{
    ChatMessageRow, EvidenceItemRow, MemoryRow, NewChatMessage, NewEvidenceItem, NewMemory,
    NewQueryRun, QueryRunRow,
};
use schema::{chat_messages, evidence_items, memories, query_runs};

// ---------------------------------------------------------------------------
// PgStore
// ---------------------------------------------------------------------------

/// A [`StateStore`] backed by PostgreSQL via Diesel + diesel-async + bb8.
///
/// All public methods are instrumented with `tracing::instrument` so that
/// spans appear in distributed traces.
#[derive(Clone)]
pub struct PgStore {
    pool: Pool<AsyncPgConnection>,
}

impl PgStore {
    /// Create a new [`PgStore`], connecting to `database_url` and building
    /// a `bb8` connection pool with the default configuration.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ConnectionPool`] if the pool fails to build or
    /// if no initial connection can be established.
    pub async fn new(database_url: &str) -> Result<Self, StoreError> {
        let manager =
            AsyncDieselConnectionManager::<AsyncPgConnection>::new(database_url);
        let pool = Pool::builder()
            .build(manager)
            .await
            .map_err(|e| StoreError::ConnectionPool(e.to_string()))?;
        Ok(Self { pool })
    }

    /// Obtain a pooled connection, mapping pool errors to [`StoreError`].
    async fn conn(
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
    /// Fetch the last `limit` messages for `session_id`, ordered oldest first.
    #[instrument(skip(self), fields(session_id = %session_id, limit = limit))]
    async fn get_messages(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, StoreError> {
        let mut conn = self.conn().await?;
        let raw_limit = i64::try_from(limit).unwrap_or(i64::MAX);

        // Subquery trick: select the N most-recent rows, then reverse for ASC order.
        let rows: Vec<ChatMessageRow> = chat_messages::table
            .filter(chat_messages::session_id.eq(session_id.0))
            .order(chat_messages::created_at.desc())
            .limit(raw_limit)
            .select(ChatMessageRow::as_select())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)?;

        // Reverse so callers receive messages in chronological order.
        let mut rows = rows;
        rows.reverse();

        rows.into_iter()
            .map(|r| r.into_domain())
            .collect::<Result<Vec<_>, _>>()
    }

    /// Persist a single [`ChatMessage`].
    #[instrument(skip(self), fields(message_id = %message.id))]
    async fn store_message(&self, message: &ChatMessage) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let new_row = NewChatMessage::from_domain(message);
        diesel::insert_into(chat_messages::table)
            .values(&new_row)
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    /// Fetch all memories for `user_id`, ordered oldest first.
    #[instrument(skip(self), fields(user_id = %user_id))]
    async fn get_memories(&self, user_id: &UserId) -> Result<Vec<Memory>, StoreError> {
        let mut conn = self.conn().await?;
        let rows: Vec<MemoryRow> = memories::table
            .filter(memories::user_id.eq(&user_id.0))
            .order(memories::created_at.asc())
            .select(MemoryRow::as_select())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)?;

        rows.into_iter()
            .map(|r| r.into_domain())
            .collect::<Result<Vec<_>, _>>()
    }

    /// Persist a single [`Memory`].
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

    /// Persist a [`QueryRun`].
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

    /// Fetch a [`QueryRun`] by ID, returning `None` if it does not exist.
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

    /// Persist a single [`EvidenceItem`] linked to a query run.
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

    /// Persist a batch of [`EvidenceItem`]s for a query run in a single INSERT.
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

    /// Fetch all evidence items for a query run, ordered oldest first.
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
