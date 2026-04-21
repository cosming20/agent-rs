//! Diesel row types for the five tables in the initial consolidated schema.
//!
//! Each table has a `*Row` struct (Queryable + Selectable for SELECT) and
//! a `New*` struct (Insertable for INSERT). `into_domain()` / `from_domain()`
//! conversions never silently fall back — unknown values return
//! [`StoreError::InvalidData`].
//!
//! This module was scoped down during the 2026-04-21 stateless pivot.
//! The removed tables (sessions, session_active_documents, chat_messages,
//! ingested_documents) and their row types are now owned by agent-rs-web's
//! Postgres.

use agent_rs_core::{Confidence, RetrievalStrategy, StoreError};
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use uuid::Uuid;

use crate::types::{EvidenceItem, Memory, QueryRun};

use super::schema::{document_index_state, evidence_items, memories, query_runs};

// ---------------------------------------------------------------------------
// Strategy helpers (shared across query_runs + strategy_performance writers)
// ---------------------------------------------------------------------------

/// Serialise a [`RetrievalStrategy`] to its DB text representation.
pub(super) fn strategy_to_str(strategy: RetrievalStrategy) -> &'static str {
    strategy.as_db_str()
}

/// Parse a [`RetrievalStrategy`] from its DB text representation.
///
/// Returns `Err(StoreError::InvalidData)` for unrecognised values.
pub(super) fn str_to_strategy(s: &str) -> Result<RetrievalStrategy, StoreError> {
    RetrievalStrategy::from_db_str(s).map_err(|e| StoreError::InvalidData(e.to_string()))
}

// ---------------------------------------------------------------------------
// memories
// ---------------------------------------------------------------------------

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = memories)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct MemoryRow {
    pub id: Uuid,
    pub user_id: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

impl MemoryRow {
    pub fn into_domain(self) -> Result<Memory, StoreError> {
        use agent_rs_core::UserId;
        let user_id = UserId::try_new(&self.user_id)
            .map_err(|e| StoreError::InvalidData(e.to_string()))?;
        Ok(Memory {
            id: self.id,
            user_id,
            content: self.content,
            created_at: self.created_at.naive_utc(),
        })
    }
}

#[derive(Debug, Insertable)]
#[diesel(table_name = memories)]
pub struct NewMemory<'a> {
    pub id: Uuid,
    pub user_id: &'a str,
    pub content: &'a str,
}

impl<'a> NewMemory<'a> {
    pub fn from_domain(memory: &'a Memory) -> Self {
        Self {
            id: memory.id,
            user_id: memory.user_id.as_str(),
            content: &memory.content,
        }
    }
}

// ---------------------------------------------------------------------------
// query_runs
// ---------------------------------------------------------------------------
//
// `trace_id` replaces the old `session_id` column. It's nullable and has no
// authorization meaning — just a correlation UUID carried through from the
// client for log/trace linking.

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = query_runs)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct QueryRunRow {
    pub id: Uuid,
    pub trace_id: Option<Uuid>,
    pub user_id: String,
    pub query: String,
    pub answer: String,
    pub strategy: String,
    pub confidence: f64,
    pub is_grounded: bool,
    pub created_at: DateTime<Utc>,
}

impl QueryRunRow {
    pub fn into_domain(self) -> Result<QueryRun, StoreError> {
        use agent_rs_core::UserId;
        let user_id = UserId::try_new(&self.user_id)
            .map_err(|e| StoreError::InvalidData(e.to_string()))?;
        let strategy = str_to_strategy(&self.strategy)?;
        let confidence = Confidence::new(self.confidence)
            .map_err(|e| StoreError::InvalidData(e.to_string()))?;
        Ok(QueryRun {
            id: self.id,
            trace_id: self.trace_id,
            user_id,
            query: self.query,
            answer: self.answer,
            strategy,
            confidence,
            is_grounded: self.is_grounded,
            created_at: self.created_at.naive_utc(),
        })
    }
}

#[derive(Debug, Insertable)]
#[diesel(table_name = query_runs)]
pub struct NewQueryRun<'a> {
    pub id: Uuid,
    pub trace_id: Option<Uuid>,
    pub user_id: &'a str,
    pub query: &'a str,
    pub answer: &'a str,
    pub strategy: &'a str,
    pub confidence: f64,
    pub is_grounded: bool,
}

impl<'a> NewQueryRun<'a> {
    pub fn from_domain(run: &'a QueryRun) -> Self {
        Self {
            id: run.id,
            trace_id: run.trace_id,
            user_id: run.user_id.as_str(),
            query: &run.query,
            answer: &run.answer,
            strategy: strategy_to_str(run.strategy),
            confidence: run.confidence.value(),
            is_grounded: run.is_grounded,
        }
    }
}

// ---------------------------------------------------------------------------
// evidence_items
// ---------------------------------------------------------------------------

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = evidence_items)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct EvidenceItemRow {
    pub id: Uuid,
    pub run_id: Uuid,
    pub content: String,
    pub source: String,
    pub score: f64,
    pub created_at: DateTime<Utc>,
}

impl EvidenceItemRow {
    pub fn into_domain(self) -> Result<EvidenceItem, StoreError> {
        Ok(EvidenceItem {
            id: self.id,
            run_id: self.run_id,
            content: self.content,
            source: self.source,
            score: self.score,
            created_at: self.created_at.naive_utc(),
        })
    }
}

#[derive(Debug, Insertable)]
#[diesel(table_name = evidence_items)]
pub struct NewEvidenceItem<'a> {
    pub id: Uuid,
    pub run_id: Uuid,
    pub content: &'a str,
    pub source: &'a str,
    pub score: f64,
}

impl<'a> NewEvidenceItem<'a> {
    pub fn from_domain(item: &'a EvidenceItem) -> Self {
        Self {
            id: item.id,
            run_id: item.run_id,
            content: &item.content,
            source: &item.source,
            score: item.score,
        }
    }
}

// ---------------------------------------------------------------------------
// document_index_state
// ---------------------------------------------------------------------------
//
// The MinIO key → indexing state machine. Single source of truth for the
// `GetDocumentStatus` RPC; written by the IndexerWorker consumer as it
// processes jobs off the RabbitMQ queue.
//
// Valid states: `pending` (queued, not yet picked up), `indexing` (worker
// has claimed the job and is running Docling/chunking/embedding), `complete`
// (Qdrant + Neo4j upserts finished), `failed` (terminal — `error_message`
// populated). State transitions are enforced by application code; the
// schema-level CHECK only validates the enum domain.

#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = document_index_state)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct DocumentIndexStateRow {
    pub minio_object_key: String,
    pub user_id: String,
    pub state: String,
    pub sha256: Option<String>,
    pub n_pages: Option<i32>,
    pub n_chunks: Option<i32>,
    pub error_message: Option<String>,
    pub queued_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Insertable)]
#[diesel(table_name = document_index_state)]
pub struct NewDocumentIndexState<'a> {
    pub minio_object_key: &'a str,
    pub user_id: &'a str,
    pub state: &'a str,
}
