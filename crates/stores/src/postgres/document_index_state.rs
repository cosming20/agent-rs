//! CRUD over `document_index_state` — the MinIO key → indexing state
//! machine backing the `GetDocumentStatus` RPC and the IndexerWorker
//! consumer loop.
//!
//! State transitions:
//!
//! ```text
//!    pending  ─►  indexing  ─►  complete
//!       │            │             │
//!       └────────────┴─────────────┴─► failed
//! ```
//!
//! Written by:
//!   - `EnqueueIndex` RPC: INSERT new row with state='pending'
//!   - IndexerWorker on job start: UPDATE state='indexing', started_at=NOW
//!   - IndexerWorker on job success: UPDATE state='complete', completed_at=NOW, sha256/n_pages/n_chunks
//!   - IndexerWorker on job failure: UPDATE state='failed', error_message

use agent_rs_core::{StoreError, UserId};
use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use tracing::instrument;

use super::PgStore;
use super::models::{DocumentIndexStateRow, NewDocumentIndexState};
use super::schema::document_index_state;

impl PgStore {
    /// Upsert a `pending` row for a newly-enqueued key.
    ///
    /// Returns `true` if a new row was created, `false` if the key was
    /// already known (regardless of its current state — the caller can
    /// distinguish via [`Self::get_index_state`]).
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — DB-level error.
    #[instrument(skip(self), fields(user_id = %user_id, minio_object_key = %minio_object_key))]
    pub async fn enqueue_index_state(
        &self,
        user_id: &UserId,
        minio_object_key: &str,
    ) -> Result<bool, StoreError> {
        let mut conn = self.conn().await?;
        let row = NewDocumentIndexState {
            minio_object_key,
            user_id: user_id.as_str(),
            state: "pending",
        };
        // `ON CONFLICT DO NOTHING` + returning affected rows tells us whether
        // this was a first-time insert. Idempotent: the IndexerWorker may
        // observe multiple EnqueueIndex calls for the same key if a client
        // retries — we handle that gracefully by returning the existing row's
        // state to the caller.
        let inserted = diesel::insert_into(document_index_state::table)
            .values(&row)
            .on_conflict(document_index_state::minio_object_key)
            .do_nothing()
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;

        Ok(inserted == 1)
    }

    /// Fetch the index state row for a given MinIO key, scoped to `user_id`.
    ///
    /// Cross-user reads return `None` — the WHERE clause enforces ownership
    /// at the SQL level as a defence-in-depth measure.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — DB-level error.
    #[instrument(skip(self), fields(user_id = %user_id, minio_object_key = %minio_object_key))]
    pub async fn get_index_state(
        &self,
        user_id: &UserId,
        minio_object_key: &str,
    ) -> Result<Option<DocumentIndexStateRow>, StoreError> {
        let mut conn = self.conn().await?;
        document_index_state::table
            .filter(document_index_state::minio_object_key.eq(minio_object_key))
            .filter(document_index_state::user_id.eq(user_id.as_str()))
            .select(DocumentIndexStateRow::as_select())
            .first(&mut conn)
            .await
            .optional()
            .map_err(StoreError::from)
    }

    /// Transition a row from `pending` → `indexing` atomically.
    ///
    /// The `WHERE state = 'pending'` predicate makes this a compare-and-swap
    /// primitive — if two IndexerWorkers race on the same queue message
    /// (shouldn't happen with proper ACK semantics, but defence-in-depth),
    /// only one sees `rows_affected > 0` and proceeds.
    ///
    /// Returns `true` if the transition succeeded.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — DB-level error.
    #[instrument(skip(self), fields(minio_object_key = %minio_object_key))]
    pub async fn claim_pending_for_indexing(
        &self,
        minio_object_key: &str,
    ) -> Result<bool, StoreError> {
        let mut conn = self.conn().await?;
        let affected = diesel::update(document_index_state::table)
            .filter(document_index_state::minio_object_key.eq(minio_object_key))
            .filter(document_index_state::state.eq("pending"))
            .set((
                document_index_state::state.eq("indexing"),
                document_index_state::started_at.eq(diesel::dsl::now),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(affected == 1)
    }

    /// Transition to `complete` with populated metadata.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — DB-level error.
    #[instrument(skip(self), fields(minio_object_key = %minio_object_key, n_pages, n_chunks))]
    pub async fn mark_index_complete(
        &self,
        minio_object_key: &str,
        sha256: &str,
        n_pages: i32,
        n_chunks: i32,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        diesel::update(document_index_state::table)
            .filter(document_index_state::minio_object_key.eq(minio_object_key))
            .set((
                document_index_state::state.eq("complete"),
                document_index_state::sha256.eq(sha256),
                document_index_state::n_pages.eq(n_pages),
                document_index_state::n_chunks.eq(n_chunks),
                document_index_state::completed_at.eq(diesel::dsl::now),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    /// Transition to `failed` with an error message.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — DB-level error.
    #[instrument(skip(self, error_message), fields(minio_object_key = %minio_object_key))]
    pub async fn mark_index_failed(
        &self,
        minio_object_key: &str,
        error_message: &str,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        diesel::update(document_index_state::table)
            .filter(document_index_state::minio_object_key.eq(minio_object_key))
            .set((
                document_index_state::state.eq("failed"),
                document_index_state::error_message.eq(error_message),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    /// Remove an index state row — called by `DeleteDocument` after the
    /// Qdrant + Neo4j purges succeed.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — DB-level error.
    #[instrument(skip(self), fields(user_id = %user_id, minio_object_key = %minio_object_key))]
    pub async fn delete_index_state(
        &self,
        user_id: &UserId,
        minio_object_key: &str,
    ) -> Result<usize, StoreError> {
        let mut conn = self.conn().await?;
        diesel::delete(
            document_index_state::table
                .filter(document_index_state::minio_object_key.eq(minio_object_key))
                .filter(document_index_state::user_id.eq(user_id.as_str())),
        )
        .execute(&mut conn)
        .await
        .map_err(StoreError::from)
    }
}
