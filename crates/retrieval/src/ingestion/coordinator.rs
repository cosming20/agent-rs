//! [`IngestCoordinator`] — cross-store cleanup for partial ingest failures.
//!
//! This struct owns references to all three storage backends so that operator-
//! triggered or pipeline-triggered cleanup can reach across Qdrant, Neo4j, and
//! Postgres in a single method call.
//!
//! # Why not on `PgStore`?
//!
//! `PgStore` owns only the Postgres pool and cannot clean Qdrant or Neo4j.
//! Placing cleanup here keeps each store crate focused on its own backend
//! while giving the retrieval crate a unified reconciliation point.

use std::sync::Arc;

use agent_rs_core::{CollectionName, UserId};
use agent_rs_stores::postgres::PgStore;
use agent_rs_stores::traits::{GraphStore, VectorStore};
use agent_rs_stores::types::FilterSpec;
use tracing::{instrument, warn};

use crate::error::RetrievalError;

// ---- IngestCoordinator ----

/// Coordinates cross-store cleanup for partial ingest failures.
///
/// Exposes a single [`Self::cleanup_partial`] method that rolls back
/// Qdrant points, Neo4j nodes, and the Postgres row for a given document
/// sha256 in the correct order.
///
/// # Ownership
///
/// Holds `Arc` references so it can be cheaply cloned and passed across
/// async tasks without lifetime complexity.
pub struct IngestCoordinator {
    pub(crate) vector_store: Arc<dyn VectorStore + Send + Sync>,
    pub(crate) graph_store: Arc<dyn GraphStore + Send + Sync>,
    pub(crate) pg_store: Arc<PgStore>,
}

impl IngestCoordinator {
    /// Construct a new coordinator.
    pub fn new(
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        pg_store: Arc<PgStore>,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            pg_store,
        }
    }

    /// Purge all artifacts of a partial ingest run for the document identified
    /// by `sha256`.
    ///
    /// Steps (best-effort — all three run even if one fails):
    ///
    /// 1. Query Postgres for the row.  Return `Ok` if not found (idempotent).
    /// 2. Refuse to clean a `complete` row (caller bug; surface as error).
    /// 3. Extract `ingest_run_id`.
    /// 4. Delete Qdrant points tagged `ingest_run_id = <run>`.
    /// 5. Delete Neo4j nodes tagged `ingest_run_id = <run>`.
    /// 6. Delete the Postgres row.
    ///
    /// If any of steps 4–6 fail, the error is logged at WARN and the first
    /// error is returned after all steps complete.  This ensures cleanup is
    /// maximally idempotent even when one backend is temporarily unreachable.
    ///
    /// # Errors
    ///
    /// - [`RetrievalError::Ingestion`] if the row is in `complete` status.
    /// - [`RetrievalError::Store`] if Postgres lookup fails.
    /// - The first error from steps 4–6 if any backend operation fails.
    #[instrument(skip(self), fields(sha256 = %sha256, collection = %collection, user_id = %user_id.as_str()))]
    pub async fn cleanup_partial(
        &self,
        user_id: &UserId,
        sha256: &str,
        collection: &CollectionName,
    ) -> Result<(), RetrievalError> {
        // Step 1: look up the row scoped to user_id (dedup is per-tenant).
        let row = self
            .pg_store
            .get_ingested_document_by_sha256(user_id, sha256)
            .await
            .map_err(RetrievalError::Store)?;

        let Some(row) = row else {
            // Already cleaned or never existed — idempotent success.
            return Ok(());
        };

        // Step 2: refuse to delete complete rows.
        if row.ingest_status == "complete" {
            return Err(RetrievalError::Ingestion {
                reason: format!(
                    "cleanup_partial refused: document sha256={sha256} has status 'complete'; \
                     use a separate operator command to retire completed documents"
                ),
            });
        }

        let ingest_run_id = row.ingest_run_id;
        let run_id_str = ingest_run_id.to_string();

        let mut first_error: Option<RetrievalError> = None;

        // Step 4: delete Qdrant points by run_id filter.
        match self
            .vector_store
            .delete_by_filter(collection, FilterSpec::eq("ingest_run_id", &run_id_str))
            .await
        {
            Ok(deleted) => {
                tracing::info!(
                    sha256,
                    run_id = %run_id_str,
                    qdrant_deleted = deleted,
                    "cleanup_partial: Qdrant rollback complete"
                );
            }
            Err(e) => {
                warn!(
                    sha256,
                    run_id = %run_id_str,
                    error = %e,
                    "cleanup_partial: Qdrant rollback failed (continuing)"
                );
                if first_error.is_none() {
                    first_error = Some(RetrievalError::Store(e));
                }
            }
        }

        // Step 5: delete Neo4j nodes by run_id.
        match self.graph_store.rollback_by_run_id(ingest_run_id).await {
            Ok(deleted) => {
                tracing::info!(
                    sha256,
                    run_id = %run_id_str,
                    neo4j_deleted = deleted,
                    "cleanup_partial: Neo4j rollback complete"
                );
            }
            Err(e) => {
                warn!(
                    sha256,
                    run_id = %run_id_str,
                    error = %e,
                    "cleanup_partial: Neo4j rollback failed (continuing)"
                );
                if first_error.is_none() {
                    first_error = Some(RetrievalError::Store(e));
                }
            }
        }

        // Step 6: delete the Postgres row.
        match self.pg_store.delete_row_by_run_id(ingest_run_id).await {
            Ok(()) => {
                tracing::info!(
                    sha256,
                    run_id = %run_id_str,
                    "cleanup_partial: Postgres row deleted"
                );
            }
            Err(e) => {
                warn!(
                    sha256,
                    run_id = %run_id_str,
                    error = %e,
                    "cleanup_partial: Postgres row deletion failed (continuing)"
                );
                if first_error.is_none() {
                    first_error = Some(RetrievalError::Store(e));
                }
            }
        }

        match first_error {
            None => Ok(()),
            Some(e) => Err(e),
        }
    }
}
