//! session_active_documents — which of a user's docs participate in
//! a given conversation. Activation is ownership-verified in a single
//! Postgres round-trip (no TOCTOU between check and insert).

use agent_rs_core::errors::StoreError;
use agent_rs_core::types::{SessionId, UserId};
use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use uuid::Uuid;

use super::PgStore;
use super::models::IngestedDocumentRow;
use super::schema::{ingested_documents, session_active_documents};

impl PgStore {
    /// Activate `document_id` in `session_id` for `user_id`. The INSERT
    /// is gated by a single `INSERT ... SELECT ... WHERE` that joins
    /// `sessions` and `ingested_documents` — it completes ONLY if both
    /// the session and the document belong to `user_id`, and the doc
    /// is in `ingest_status='complete'`.
    ///
    /// Idempotent: duplicate activation (same session + same doc)
    /// collapses to a no-op via `ON CONFLICT DO NOTHING`.
    ///
    /// Returns [`StoreError::DocumentNotOwnedByUser`] when:
    /// - the session belongs to a different user;
    /// - the doc belongs to a different user;
    /// - the doc is not `ingest_status='complete'`.
    ///
    /// All three cases collapse to the same error to avoid leaking
    /// which case triggered.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — database-level error.
    /// - [`StoreError::DocumentNotOwnedByUser`] — guard rejected the activation.
    pub async fn activate_document_for_session(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
        document_id: &Uuid,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let sid = *session_id.inner();
        let did = *document_id;
        let uid = user_id.as_str().to_owned();

        // Raw SQL — the "ownership JOIN + ON CONFLICT" shape is cleaner
        // as one statement than as a DSL pipeline.
        let rows: usize = diesel::sql_query(
            "INSERT INTO session_active_documents (session_id, document_id)
             SELECT $1, $2
             FROM sessions s
             JOIN ingested_documents d ON d.user_id = s.user_id
             WHERE s.session_id = $1
               AND d.document_id = $2
               AND s.user_id = $3
               AND d.ingest_status = 'complete'
             ON CONFLICT DO NOTHING",
        )
        .bind::<diesel::sql_types::Uuid, _>(sid)
        .bind::<diesel::sql_types::Uuid, _>(did)
        .bind::<diesel::sql_types::Text, _>(&uid)
        .execute(&mut conn)
        .await
        .map_err(StoreError::from)?;

        if rows > 0 {
            return Ok(());
        }

        // Zero rows inserted — either the guard rejected (wrong owner /
        // not complete) OR it was an idempotent no-op on conflict. Re-check
        // with the same ownership JOIN so we cannot confuse another user's
        // row for an idempotent no-op belonging to this user.
        let owned_exists: i64 = diesel::sql_query(
            "SELECT COUNT(*)::bigint AS count
             FROM session_active_documents sad
             JOIN sessions s ON s.session_id = sad.session_id
             JOIN ingested_documents d
               ON d.document_id = sad.document_id
              AND d.user_id = s.user_id
             WHERE sad.session_id = $1
               AND sad.document_id = $2
               AND s.user_id = $3
               AND d.ingest_status = 'complete'",
        )
        .bind::<diesel::sql_types::Uuid, _>(sid)
        .bind::<diesel::sql_types::Uuid, _>(did)
        .bind::<diesel::sql_types::Text, _>(&uid)
        .get_result::<crate::postgres::CountRow>(&mut conn)
        .await
        .map_err(StoreError::from)?
        .count;

        if owned_exists > 0 {
            return Ok(()); // idempotent re-activation by the same owner
        }

        Err(StoreError::DocumentNotOwnedByUser {
            document_id: did.to_string(),
            user_id: uid,
        })
    }

    /// List every document activated for `session_id` owned by `user_id`.
    ///
    /// Returns empty for cross-user reads — defence in depth even if
    /// the session authz layer is bypassed.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ConnectionPool`] — pool exhausted.
    /// - [`StoreError::Postgres`] — database-level error.
    pub async fn list_active_documents(
        &self,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<Vec<IngestedDocumentRow>, StoreError> {
        let mut conn = self.conn().await?;
        let rows: Vec<IngestedDocumentRow> = ingested_documents::table
            .inner_join(
                session_active_documents::table
                    .on(session_active_documents::document_id.eq(ingested_documents::document_id)),
            )
            .filter(session_active_documents::session_id.eq(*session_id.inner()))
            .filter(ingested_documents::user_id.eq(user_id.as_str()))
            .filter(ingested_documents::ingest_status.eq("complete"))
            .select(IngestedDocumentRow::as_select())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)?;
        Ok(rows)
    }
}
