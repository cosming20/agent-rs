//! Sessions: authorisation gate between a user and a conversation.
//!
//! Every read on `chat_messages`, `memories`, or `session_active_documents`
//! should validate that `(user_id, session_id)` belongs together — a session
//! cannot be cross-read by a different user.
//!
//! The core primitive exposed here is [`PgStore::create_or_fetch_session`],
//! which uses `INSERT … ON CONFLICT DO NOTHING` followed by a `SELECT` and an
//! ownership check inside the same Postgres transaction so that concurrent
//! callers see serialised results.
//!
//! [`PgStore::read_context_snapshot`] loads history, memories, and active
//! documents in a single `REPEATABLE READ` transaction for a consistent view.

use agent_rs_core::{
    StoreError,
    types::{SessionId, UserId},
};
use diesel::prelude::*;
use diesel_async::{AsyncConnection, RunQueryDsl, scoped_futures::ScopedFutureExt};
use tracing::instrument;

use super::PgStore;
use super::models::{ChatMessageRow, IngestedDocumentRow, MemoryRow, NewSession, SessionRow};
use super::schema::{chat_messages, ingested_documents, memories, session_active_documents, sessions};

/// Return type of [`PgStore::read_context_snapshot`]: chat messages, memories,
/// and active ingested documents bundled together.
type ContextSnapshot = (Vec<ChatMessageRow>, Vec<MemoryRow>, Vec<IngestedDocumentRow>);

impl PgStore {
    /// Upsert-with-ownership semantics for a session row.
    ///
    /// - If no row exists for `session_id`, insert one owned by `user_id` and
    ///   return it.
    /// - If a row exists **and belongs to `user_id`**, return it without
    ///   modification (`created_at` is preserved — the caller can detect
    ///   idempotent repeat calls by comparing timestamps).
    /// - If a row exists for a **different user**, return
    ///   [`StoreError::SessionNotFound`].  This deliberately collapses "does
    ///   not exist" and "belongs to someone else" into the same error to
    ///   prevent session-ID enumeration attacks.
    ///
    /// The two database operations (`INSERT … ON CONFLICT DO NOTHING` and
    /// `SELECT`) are executed inside a single Postgres transaction, so
    /// concurrent callers for the same `session_id` always observe a
    /// consistent owner.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — could not acquire a connection.
    /// - [`StoreError::Postgres`] — any database-level error.
    /// - [`StoreError::SessionNotFound`] — row exists but belongs to a
    ///   different user, or the `SELECT` after insert returns no row.
    #[instrument(skip(self), fields(user_id = %user_id, session_id = %session_id))]
    pub async fn create_or_fetch_session(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<SessionRow, StoreError> {
        let mut conn = self.conn().await?;

        // Capture owned values before entering the async closure so that the
        // closure is `'static` and the borrow checker is satisfied across
        // `.await` points.
        let user_id_str = user_id.as_str().to_owned();
        let sid = *session_id.inner();

        conn.transaction::<SessionRow, StoreError, _>(|conn| {
            async move {
                // Attempt to insert; on conflict (duplicate session_id) silently
                // skip — the existing row's owner wins.
                diesel::insert_into(sessions::table)
                    .values(NewSession {
                        session_id: sid,
                        user_id: &user_id_str,
                    })
                    .on_conflict(sessions::session_id)
                    .do_nothing()
                    .execute(conn)
                    .await
                    .map_err(StoreError::from)?;

                // Fetch the authoritative row — whether freshly inserted or
                // pre-existing.
                let row: SessionRow = sessions::table
                    .filter(sessions::session_id.eq(sid))
                    .select(SessionRow::as_select())
                    .first(conn)
                    .await
                    .map_err(StoreError::from)?;

                // Cross-user hijack guard.  We intentionally return the same
                // error variant for "not found" and "owned by someone else" to
                // prevent enumeration attacks.
                if row.user_id != user_id_str {
                    return Err(StoreError::SessionNotFound {
                        session_id: sid.to_string(),
                        user_id: user_id_str,
                    });
                }

                Ok(row)
            }
            .scope_boxed()
        })
        .await
    }

    /// Load conversation history, long-term memories, and active documents
    /// for `(user_id, session_id)` in a single `REPEATABLE READ` transaction.
    ///
    /// All three reads observe the same consistent database snapshot, preventing
    /// torn reads when documents are activated or messages are appended concurrently.
    ///
    /// # Returns
    ///
    /// A tuple `(history, memories, active_docs)` where:
    /// - `history` — chat messages ordered oldest-first.
    /// - `memories` — long-term memory rows for the user.
    /// - `active_docs` — documents activated for this session (only `complete` rows).
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — could not acquire a connection.
    /// - [`StoreError::Postgres`] — any database-level error.
    #[instrument(skip(self), fields(user_id = %user_id, session_id = %session_id))]
    pub async fn read_context_snapshot(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<ContextSnapshot, StoreError> {
        let mut conn = self.conn().await?;
        let uid = user_id.as_str().to_owned();
        let sid = *session_id.inner();

        conn.transaction::<_, StoreError, _>(|conn| {
            async move {
                diesel::sql_query("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                    .execute(conn)
                    .await
                    .map_err(StoreError::from)?;

                let history: Vec<ChatMessageRow> = chat_messages::table
                    .filter(chat_messages::user_id.eq(&uid))
                    .filter(chat_messages::session_id.eq(sid))
                    .order(chat_messages::created_at.asc())
                    .select(ChatMessageRow::as_select())
                    .load(conn)
                    .await
                    .map_err(StoreError::from)?;

                let memory_rows: Vec<MemoryRow> = memories::table
                    .filter(memories::user_id.eq(&uid))
                    .order(memories::created_at.asc())
                    .select(MemoryRow::as_select())
                    .load(conn)
                    .await
                    .map_err(StoreError::from)?;

                let active: Vec<IngestedDocumentRow> = ingested_documents::table
                    .inner_join(
                        session_active_documents::table.on(
                            session_active_documents::document_id
                                .eq(ingested_documents::document_id),
                        ),
                    )
                    .filter(session_active_documents::session_id.eq(sid))
                    .filter(ingested_documents::user_id.eq(&uid))
                    .filter(ingested_documents::ingest_status.eq("complete"))
                    .select(IngestedDocumentRow::as_select())
                    .load(conn)
                    .await
                    .map_err(StoreError::from)?;

                Ok((history, memory_rows, active))
            }
            .scope_boxed()
        })
        .await
    }
}
