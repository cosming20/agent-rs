//! Diesel row types for all four state-store tables.
//!
//! Each table has:
//! - A `*Row` struct implementing `Queryable` + `Selectable` for SELECT queries.
//! - A `New*` struct implementing `Insertable` for INSERT statements.
//! - An `into_domain()` method returning `Result<DomainType, StoreError>` —
//!   never a silent fallback.
//!
//! Role and strategy serialisation helpers are module-private and called
//! only from `into_domain()` / `from_domain()`.

use agent_rs_core::{Confidence, MessageRole, RetrievalStrategy, SessionId, StoreError, UserId};
use chrono::NaiveDateTime;
use diesel::prelude::*;
use uuid::Uuid;

use crate::types::{ChatMessage, EvidenceItem, Memory, QueryRun};

use super::schema::{chat_messages, evidence_items, memories, query_runs};

// ---------------------------------------------------------------------------
// Role helpers
// ---------------------------------------------------------------------------

/// Serialise a [`MessageRole`] to its DB text representation.
pub(super) fn role_to_str(role: MessageRole) -> &'static str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
    }
}

/// Parse a [`MessageRole`] from its DB text representation.
///
/// Returns `Err(StoreError::InvalidData)` for unrecognised values —
/// never silently defaults to a wrong role.
pub(super) fn str_to_role(s: &str) -> Result<MessageRole, StoreError> {
    match s {
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "system" => Ok(MessageRole::System),
        other => Err(StoreError::InvalidData(format!(
            "unknown message role in DB: '{other}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Strategy helpers
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
// chat_messages
// ---------------------------------------------------------------------------

/// A row read from the `chat_messages` table.
#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = chat_messages)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct ChatMessageRow {
    pub id: Uuid,
    pub session_id: Uuid,
    pub role: String,
    pub content: String,
    pub created_at: NaiveDateTime,
}

impl ChatMessageRow {
    /// Convert this DB row into the domain type.
    ///
    /// # Errors
    /// Returns [`StoreError::InvalidData`] if `role` is not a known variant.
    pub fn into_domain(self) -> Result<ChatMessage, StoreError> {
        let role = str_to_role(&self.role)?;
        Ok(ChatMessage {
            id: self.id,
            session_id: SessionId(self.session_id),
            role,
            content: self.content,
            created_at: self.created_at,
        })
    }
}

/// A new row to insert into `chat_messages`.
#[derive(Debug, Insertable)]
#[diesel(table_name = chat_messages)]
pub struct NewChatMessage<'a> {
    pub id: Uuid,
    pub session_id: Uuid,
    pub role: &'a str,
    pub content: &'a str,
    pub created_at: NaiveDateTime,
}

impl<'a> NewChatMessage<'a> {
    /// Build an [`NewChatMessage`] from the domain type.
    pub fn from_domain(msg: &'a ChatMessage) -> Self {
        Self {
            id: msg.id,
            session_id: msg.session_id.0,
            role: role_to_str(msg.role),
            content: &msg.content,
            created_at: msg.created_at,
        }
    }
}

// ---------------------------------------------------------------------------
// memories
// ---------------------------------------------------------------------------

/// A row read from the `memories` table.
#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = memories)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct MemoryRow {
    pub id: Uuid,
    pub user_id: String,
    pub content: String,
    pub created_at: NaiveDateTime,
}

impl MemoryRow {
    /// Convert this DB row into the domain type.
    pub fn into_domain(self) -> Result<Memory, StoreError> {
        Ok(Memory {
            id: self.id,
            user_id: UserId(self.user_id),
            content: self.content,
            created_at: self.created_at,
        })
    }
}

/// A new row to insert into `memories`.
#[derive(Debug, Insertable)]
#[diesel(table_name = memories)]
pub struct NewMemory<'a> {
    pub id: Uuid,
    pub user_id: &'a str,
    pub content: &'a str,
    pub created_at: NaiveDateTime,
}

impl<'a> NewMemory<'a> {
    /// Build a [`NewMemory`] from the domain type.
    pub fn from_domain(mem: &'a Memory) -> Self {
        Self {
            id: mem.id,
            user_id: &mem.user_id.0,
            content: &mem.content,
            created_at: mem.created_at,
        }
    }
}

// ---------------------------------------------------------------------------
// query_runs
// ---------------------------------------------------------------------------

/// A row read from the `query_runs` table.
#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = query_runs)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct QueryRunRow {
    pub id: Uuid,
    pub session_id: Uuid,
    pub user_id: String,
    pub query: String,
    pub answer: String,
    pub strategy: String,
    pub confidence: f64,
    pub is_grounded: bool,
    pub created_at: NaiveDateTime,
}

impl QueryRunRow {
    /// Convert this DB row into the domain type.
    ///
    /// # Errors
    /// - [`StoreError::InvalidData`] if `strategy` is unrecognised.
    /// - [`StoreError::InvalidData`] if `confidence` is outside `[0.0, 1.0]`.
    pub fn into_domain(self) -> Result<QueryRun, StoreError> {
        let strategy = str_to_strategy(&self.strategy)?;
        let confidence = Confidence::new(self.confidence)
            .map_err(|e| StoreError::InvalidData(format!("confidence out of range in DB: {e}")))?;
        Ok(QueryRun {
            id: self.id,
            session_id: SessionId(self.session_id),
            user_id: UserId(self.user_id),
            query: self.query,
            answer: self.answer,
            strategy,
            confidence,
            is_grounded: self.is_grounded,
            created_at: self.created_at,
        })
    }
}

/// A new row to insert into `query_runs`.
#[derive(Debug, Insertable)]
#[diesel(table_name = query_runs)]
pub struct NewQueryRun<'a> {
    pub id: Uuid,
    pub session_id: Uuid,
    pub user_id: &'a str,
    pub query: &'a str,
    pub answer: &'a str,
    pub strategy: &'a str,
    pub confidence: f64,
    pub is_grounded: bool,
    pub created_at: NaiveDateTime,
}

impl<'a> NewQueryRun<'a> {
    /// Build a [`NewQueryRun`] from the domain type.
    pub fn from_domain(run: &'a QueryRun) -> Self {
        Self {
            id: run.id,
            session_id: run.session_id.0,
            user_id: &run.user_id.0,
            query: &run.query,
            answer: &run.answer,
            strategy: strategy_to_str(run.strategy),
            confidence: run.confidence.value(),
            is_grounded: run.is_grounded,
            created_at: run.created_at,
        }
    }
}

// ---------------------------------------------------------------------------
// evidence_items
// ---------------------------------------------------------------------------

/// A row read from the `evidence_items` table.
#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = evidence_items)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct EvidenceItemRow {
    pub id: Uuid,
    pub run_id: Uuid,
    pub content: String,
    pub source: String,
    pub score: f64,
    pub created_at: NaiveDateTime,
}

impl EvidenceItemRow {
    /// Convert this DB row into the domain type.
    pub fn into_domain(self) -> Result<EvidenceItem, StoreError> {
        Ok(EvidenceItem {
            id: self.id,
            run_id: self.run_id,
            content: self.content,
            source: self.source,
            score: self.score,
            created_at: self.created_at,
        })
    }
}

/// A new row to insert into `evidence_items`.
#[derive(Debug, Insertable)]
#[diesel(table_name = evidence_items)]
pub struct NewEvidenceItem<'a> {
    pub id: Uuid,
    pub run_id: Uuid,
    pub content: &'a str,
    pub source: &'a str,
    pub score: f64,
    pub created_at: NaiveDateTime,
}

impl<'a> NewEvidenceItem<'a> {
    /// Build a [`NewEvidenceItem`] from the domain type.
    pub fn from_domain(item: &'a EvidenceItem) -> Self {
        Self {
            id: item.id,
            run_id: item.run_id,
            content: &item.content,
            source: &item.source,
            score: item.score,
            created_at: item.created_at,
        }
    }
}
