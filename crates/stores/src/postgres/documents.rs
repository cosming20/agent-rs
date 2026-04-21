//! [`PgStore`] methods for the `ingested_documents` table.
//!
//! Implements the ingest state machine:
//!
//! ```text
//! (new sha256) ──begin_ingestion──► in_progress
//!                                       │
//!                         ┌─────────────┴──────────────┐
//!                         ▼                            ▼
//!                  mark_complete                mark_partial_failure
//!                     complete                    partial_failure
//!                                                     │
//!                                         begin_ingestion (same sha256)
//!                                         ──► Resumed (new run_id, reset to in_progress)
//! ```
//!
//! All queries use Diesel's typed DSL — no raw SQL string interpolation.

use agent_rs_core::{IngestStatus, StoreError, UserId};
use chrono::Utc;
use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use tracing::instrument;
use uuid::Uuid;

use super::PgStore;
use super::models::{BeginIngestionOutcome, IngestedDocumentRow, NewIngestedDocument};
use super::schema::ingested_documents;

// ---------------------------------------------------------------------------
// HeaderMetadata
// ---------------------------------------------------------------------------

/// Metadata extracted from the sidecar's `DocumentHeader` message.
///
/// W3's `IngestCoordinator` maps from the proto `DocumentHeader` to this
/// struct at the gRPC boundary.  All fields use owned types so this struct
/// can be passed across `.await` points without lifetime complexity.
#[derive(Debug, Clone)]
pub struct HeaderMetadata {
    /// Docling library version string (e.g. `"2.7.0"`).
    pub docling_version: String,
    /// `docling-core` package version string.
    pub docling_core_version: String,
    /// Docling schema version integer.
    pub schema_version: i32,
    /// Number of pages in the document.
    pub n_pages: i32,
    /// Number of chunks emitted by the sidecar.
    pub n_chunks: i32,
    /// Number of tables found in the document.
    pub n_tables: i32,
    /// Number of pictures found in the document.
    pub n_pictures: i32,
    /// Number of formula items found in the document.
    pub n_formulas: i32,
    /// Detected primary language (ISO 639-1 or free-form, as returned by Docling).
    pub language: Option<String>,
    /// Document title if detected.
    pub title: Option<String>,
    /// MIME type of the source file (e.g. `"application/pdf"`).
    pub mimetype: Option<String>,
    /// Parse latency measured inside the sidecar (milliseconds).
    pub parse_latency_ms: Option<i64>,
    /// End-to-end ingest latency measured by the Rust caller (milliseconds).
    pub total_ingest_latency_ms: Option<i64>,
}

// ---------------------------------------------------------------------------
// PgStore methods
// ---------------------------------------------------------------------------

impl PgStore {
    /// Open a new ingestion record for `new_doc` or determine the state of an
    /// existing record for the same `(user_id, sha256)` pair.
    ///
    /// Dedup is scoped per-tenant: two different users uploading the same
    /// file each receive a distinct `document_id` and independent lifecycle.
    ///
    /// Uses `INSERT … ON CONFLICT (user_id, sha256) DO NOTHING` to serialise
    /// concurrent attempts for the same (user, document) without taking a lock.
    ///
    /// # Returns
    ///
    /// - [`BeginIngestionOutcome::Started`] — fresh insert; proceed.
    /// - [`BeginIngestionOutcome::Resumed`] — prior `partial_failure` was
    ///   reset to `in_progress` with a new run ID; proceed.
    /// - [`BeginIngestionOutcome::AlreadyBeingIngested`] — another worker
    ///   holds the row in `in_progress`; caller should surface
    ///   `RetrievalError::IngestInProgress`.
    /// - [`BeginIngestionOutcome::AlreadyComplete`] — document is already
    ///   fully ingested; caller should return an idempotent result.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error, or
    /// [`StoreError::InvalidState`] if `ingest_status` contains an
    /// unrecognised value.
    #[instrument(skip(self, new_doc), fields(sha256 = %new_doc.sha256, user_id = %user_id.as_str()))]
    pub async fn begin_ingestion(
        &self,
        user_id: &UserId,
        new_doc: NewIngestedDocument,
    ) -> Result<BeginIngestionOutcome, StoreError> {
        let mut conn = self.conn().await?;

        // Attempt insert; on conflict (duplicate (user_id, sha256)) do nothing
        // and return no row, which we detect via `.optional()`.
        let inserted: Option<IngestedDocumentRow> = diesel::insert_into(ingested_documents::table)
            .values(&new_doc)
            .on_conflict((ingested_documents::user_id, ingested_documents::sha256))
            .do_nothing()
            .returning(IngestedDocumentRow::as_returning())
            .get_result(&mut conn)
            .await
            .optional()
            .map_err(StoreError::from)?;

        match inserted {
            Some(row) => Ok(BeginIngestionOutcome::Started {
                ingest_run_id: row.ingest_run_id,
            }),
            None => {
                // Row already exists for this (user_id, sha256); fetch it to
                // inspect status.
                let existing: IngestedDocumentRow = ingested_documents::table
                    .filter(ingested_documents::user_id.eq(user_id.as_str()))
                    .filter(ingested_documents::sha256.eq(&new_doc.sha256))
                    .select(IngestedDocumentRow::as_select())
                    .first(&mut conn)
                    .await
                    .map_err(StoreError::from)?;

                match existing.parse_status()? {
                    IngestStatus::Complete => Ok(BeginIngestionOutcome::AlreadyComplete),

                    IngestStatus::InProgress => {
                        // W7 fix #7: If the row has been stuck in_progress for
                        // more than 15 minutes, treat it as abandoned by a crashed
                        // worker — reset to in_progress with a fresh run_id so
                        // the current caller can proceed.  Without this, a single
                        // crash during mark_complete leaves the document bricked.
                        let age_minutes =
                            (Utc::now().naive_utc() - existing.updated_at).num_minutes();
                        if age_minutes >= 15 {
                            tracing::warn!(
                                sha256 = %existing.sha256,
                                user_id = %user_id.as_str(),
                                age_minutes,
                                old_run_id = %existing.ingest_run_id,
                                "in_progress row older than 15 min — treating as abandoned"
                            );
                            let new_run_id = Uuid::new_v4();
                            let now = Utc::now().naive_utc();
                            diesel::update(ingested_documents::table)
                                .filter(ingested_documents::user_id.eq(user_id.as_str()))
                                .filter(ingested_documents::sha256.eq(&new_doc.sha256))
                                .set((
                                    ingested_documents::ingest_run_id.eq(new_run_id),
                                    ingested_documents::updated_at.eq(now),
                                ))
                                .execute(&mut conn)
                                .await
                                .map_err(StoreError::from)?;
                            Ok(BeginIngestionOutcome::Resumed {
                                ingest_run_id: new_run_id,
                            })
                        } else {
                            Ok(BeginIngestionOutcome::AlreadyBeingIngested)
                        }
                    }

                    IngestStatus::PartialFailure => {
                        // Reuse the row: issue a new run_id and reset to in_progress.
                        let new_run_id = Uuid::new_v4();
                        let now = Utc::now().naive_utc();

                        diesel::update(ingested_documents::table)
                            .filter(ingested_documents::user_id.eq(user_id.as_str()))
                            .filter(ingested_documents::sha256.eq(&new_doc.sha256))
                            .set((
                                ingested_documents::ingest_status
                                    .eq(IngestStatus::InProgress.as_str()),
                                ingested_documents::ingest_run_id.eq(new_run_id),
                                ingested_documents::failure_reason.eq::<Option<String>>(None),
                                ingested_documents::updated_at.eq(now),
                            ))
                            .execute(&mut conn)
                            .await
                            .map_err(StoreError::from)?;

                        Ok(BeginIngestionOutcome::Resumed {
                            ingest_run_id: new_run_id,
                        })
                    }
                }
            }
        }
    }

    /// Record a partial failure for the given `ingest_run_id`.
    ///
    /// Sets `ingest_status = 'partial_failure'` and stores `failure_reason`.
    /// The row remains in the table so operators can inspect it and callers
    /// can resume via [`Self::begin_ingestion`].
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self), fields(ingest_run_id = %ingest_run_id))]
    pub async fn mark_partial_failure(
        &self,
        ingest_run_id: Uuid,
        failure_reason: &str,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let now = Utc::now().naive_utc();

        diesel::update(ingested_documents::table)
            .filter(ingested_documents::ingest_run_id.eq(ingest_run_id))
            .set((
                ingested_documents::ingest_status.eq(IngestStatus::PartialFailure.as_str()),
                ingested_documents::failure_reason.eq(Some(failure_reason)),
                ingested_documents::updated_at.eq(now),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;

        Ok(())
    }

    /// Finalize an ingestion run with `complete` status and persist the
    /// header metadata captured from the sidecar's `DocumentHeader`.
    ///
    /// Sets `ingest_status = 'complete'`, `completed_at = NOW()`, and
    /// all header metadata columns.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self, header), fields(ingest_run_id = %ingest_run_id))]
    pub async fn mark_complete(
        &self,
        ingest_run_id: Uuid,
        header: &HeaderMetadata,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let now = Utc::now().naive_utc();

        diesel::update(ingested_documents::table)
            .filter(ingested_documents::ingest_run_id.eq(ingest_run_id))
            .set((
                ingested_documents::ingest_status.eq(IngestStatus::Complete.as_str()),
                ingested_documents::completed_at.eq(Some(now)),
                ingested_documents::updated_at.eq(now),
                ingested_documents::docling_version.eq(Some(&header.docling_version)),
                ingested_documents::docling_core_version.eq(Some(&header.docling_core_version)),
                ingested_documents::schema_version.eq(Some(header.schema_version)),
                ingested_documents::n_pages.eq(Some(header.n_pages)),
                ingested_documents::n_chunks.eq(Some(header.n_chunks)),
                ingested_documents::n_tables.eq(Some(header.n_tables)),
                ingested_documents::n_pictures.eq(Some(header.n_pictures)),
                ingested_documents::n_formulas.eq(Some(header.n_formulas)),
                ingested_documents::language.eq(header.language.as_deref()),
                ingested_documents::title.eq(header.title.as_deref()),
                ingested_documents::mimetype.eq(header.mimetype.as_deref()),
                ingested_documents::parse_latency_ms.eq(header.parse_latency_ms),
                ingested_documents::total_ingest_latency_ms.eq(header.total_ingest_latency_ms),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;

        Ok(())
    }

    /// Persist the MinIO object key for an `ingested_documents` row identified
    /// by `(user_id, document_id)`.
    ///
    /// Called by the ingest CLI after a successful `MinioStore::put_object` so
    /// the object key is recorded in Postgres alongside the pipeline metadata.
    /// The update is idempotent: re-running the CLI for a document that was
    /// already uploaded just overwrites the key with the same value.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self), fields(document_id = %document_id, user_id = %user_id.as_str()))]
    pub async fn set_minio_object_key(
        &self,
        user_id: &UserId,
        document_id: Uuid,
        object_key: &str,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;
        let now = Utc::now().naive_utc();

        diesel::update(ingested_documents::table)
            .filter(ingested_documents::user_id.eq(user_id.as_str()))
            .filter(ingested_documents::document_id.eq(document_id))
            .set((
                ingested_documents::minio_object_key.eq(Some(object_key)),
                ingested_documents::updated_at.eq(now),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;

        Ok(())
    }

    /// Return `true` if a document with `sha256` has `ingest_status = 'complete'`
    /// **and** is owned by `user_id`.
    ///
    /// Scoped to a single tenant: bob cannot probe alice's sha256 fingerprint.
    ///
    /// This is the fast-path dedup check: callers can skip ingestion entirely
    /// when this returns `true`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self), fields(sha256 = %sha256, user_id = %user_id.as_str()))]
    pub async fn is_already_ingested_complete(
        &self,
        user_id: &UserId,
        sha256: &str,
    ) -> Result<bool, StoreError> {
        use diesel::dsl::exists;
        use diesel::select;

        let mut conn = self.conn().await?;

        let found: bool = select(exists(
            ingested_documents::table
                .filter(ingested_documents::user_id.eq(user_id.as_str()))
                .filter(ingested_documents::sha256.eq(sha256))
                .filter(ingested_documents::ingest_status.eq(IngestStatus::Complete.as_str())),
        ))
        .get_result(&mut conn)
        .await
        .map_err(StoreError::from)?;

        Ok(found)
    }

    /// Fetch the `ingested_documents` row identified by `(user_id, sha256)`.
    ///
    /// Returns `None` if no row exists for the given user + sha256 (i.e., the
    /// document was never ingested by this user, or was already deleted).
    ///
    /// Scoped to a single tenant: bob cannot retrieve alice's row by sha256.
    ///
    /// Used by `IngestCoordinator::cleanup_partial` (W3) to fetch the
    /// `ingest_run_id` needed to roll back Qdrant and Neo4j before deleting
    /// the row itself.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self), fields(sha256 = %sha256, user_id = %user_id.as_str()))]
    pub async fn get_ingested_document_by_sha256(
        &self,
        user_id: &UserId,
        sha256: &str,
    ) -> Result<Option<IngestedDocumentRow>, StoreError> {
        let mut conn = self.conn().await?;

        ingested_documents::table
            .filter(ingested_documents::user_id.eq(user_id.as_str()))
            .filter(ingested_documents::sha256.eq(sha256))
            .select(IngestedDocumentRow::as_select())
            .first(&mut conn)
            .await
            .optional()
            .map_err(StoreError::from)
    }

    /// Delete the `ingested_documents` row identified by `ingest_run_id`.
    ///
    /// Used by `IngestCoordinator::cleanup_partial` (W3) to fully purge a
    /// failed ingestion attempt after Qdrant and Neo4j have been rolled back.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self), fields(ingest_run_id = %ingest_run_id))]
    pub async fn delete_row_by_run_id(&self, ingest_run_id: Uuid) -> Result<(), StoreError> {
        let mut conn = self.conn().await?;

        diesel::delete(
            ingested_documents::table.filter(ingested_documents::ingest_run_id.eq(ingest_run_id)),
        )
        .execute(&mut conn)
        .await
        .map_err(StoreError::from)?;

        Ok(())
    }

    /// List every ingested document owned by `user_id` that has
    /// `ingest_status = 'complete'`.
    ///
    /// Partial, failed, in-progress, and deleting rows are never visible to
    /// the caller — they represent transient pipeline state, not user-owned
    /// data.  Results are ordered newest-first by `started_at`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    #[instrument(skip(self), fields(user_id = %user_id.as_str()))]
    pub async fn list_ingested_documents(
        &self,
        user_id: &UserId,
    ) -> Result<Vec<IngestedDocumentRow>, StoreError> {
        let mut conn = self.conn().await?;

        ingested_documents::table
            .filter(ingested_documents::user_id.eq(user_id.as_str()))
            .filter(ingested_documents::ingest_status.eq(IngestStatus::Complete.as_str()))
            .select(IngestedDocumentRow::as_select())
            .order(ingested_documents::started_at.desc())
            .load(&mut conn)
            .await
            .map_err(StoreError::from)
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "integration-tests"))]
impl PgStore {
    /// Insert a minimal `ingest_status='complete'` row, bypassing the full
    /// ingest saga.  Intended for security / multi-tenant integration tests
    /// where the pipeline itself is out of scope.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    pub async fn test_insert_document_complete(
        &self,
        user_id: &UserId,
        source: &str,
    ) -> Result<IngestedDocumentRow, StoreError> {
        self.test_insert_document_with_status(user_id, source, "complete")
            .await
    }

    /// Insert a row with an arbitrary `ingest_status` value.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    pub async fn test_insert_document_with_status(
        &self,
        user_id: &UserId,
        source: &str,
        status: &str,
    ) -> Result<IngestedDocumentRow, StoreError> {
        use chrono::Utc;
        let mut conn = self.conn().await?;
        let new_id = Uuid::new_v4();
        let new_run = Uuid::new_v4();
        // Build a deterministic 64-char hex string from the UUID (no dashes,
        // padded/trimmed to exactly 64 chars to satisfy the `bpchar(64)` column).
        let raw = new_id.as_simple().to_string(); // 32 hex chars
        let sha = format!("{raw:0>64}"); // left-pad with '0' to 64 chars
        let sha = &sha[..64];
        let now = Utc::now().naive_utc();

        diesel::insert_into(ingested_documents::table)
            .values((
                ingested_documents::document_id.eq(new_id),
                ingested_documents::sha256.eq(sha),
                ingested_documents::source.eq(source),
                ingested_documents::ingest_status.eq(status),
                ingested_documents::ingest_run_id.eq(new_run),
                ingested_documents::user_id.eq(user_id.as_str()),
                ingested_documents::started_at.eq(now),
                ingested_documents::updated_at.eq(now),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;

        ingested_documents::table
            .filter(ingested_documents::document_id.eq(new_id))
            .select(IngestedDocumentRow::as_select())
            .first(&mut conn)
            .await
            .map_err(StoreError::from)
    }

    /// Insert a `complete` row with an explicit `sha256` value.
    ///
    /// Used for the cross-user same-SHA dedup test: callers supply the same
    /// sha for two different users and assert they get separate `document_id`s.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Postgres`] on any database error.
    pub async fn test_insert_document_with_sha(
        &self,
        user_id: &UserId,
        source: &str,
        sha256: &str,
    ) -> Result<IngestedDocumentRow, StoreError> {
        use chrono::Utc;
        let mut conn = self.conn().await?;
        let new_id = Uuid::new_v4();
        let new_run = Uuid::new_v4();
        let now = Utc::now().naive_utc();

        diesel::insert_into(ingested_documents::table)
            .values((
                ingested_documents::document_id.eq(new_id),
                ingested_documents::sha256.eq(sha256),
                ingested_documents::source.eq(source),
                ingested_documents::ingest_status.eq("complete"),
                ingested_documents::ingest_run_id.eq(new_run),
                ingested_documents::user_id.eq(user_id.as_str()),
                ingested_documents::started_at.eq(now),
                ingested_documents::updated_at.eq(now),
            ))
            .execute(&mut conn)
            .await
            .map_err(StoreError::from)?;

        ingested_documents::table
            .filter(ingested_documents::document_id.eq(new_id))
            .select(IngestedDocumentRow::as_select())
            .first(&mut conn)
            .await
            .map_err(StoreError::from)
    }
}
