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

use agent_rs_core::{StoreError, types::{SessionId, UserId}};
use diesel::prelude::*;
use diesel_async::{AsyncConnection, RunQueryDsl, scoped_futures::ScopedFutureExt};
use tracing::instrument;

use super::PgStore;
use super::models::{NewSession, SessionRow};
use super::schema::sessions;

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
}
