//! Validated entity identifier for LLM-extracted knowledge graph nodes.
//!
//! Raw LLM output frequently contains IDs that are structurally valid
//! strings but semantically worthless as graph node keys:
//!
//! - empty strings, whitespace-only strings;
//! - single characters (`"a"`, `"0"`, `"?"`);
//! - pure-numeric tokens (`"0"`, `"1234"`) — page numbers, table cells,
//!   list-item labels — which collide trivially across chunks and pollute
//!   the entity graph with uninformative super-nodes;
//! - common stopwords from the source language masquerading as entities
//!   (`"it"`, `"the"`, `"of"`).
//!
//! Letting these reach Neo4j makes every downstream retriever (entity
//! linker, graph CoT, triplet scoring) less precise, and — until the
//! retrieval-side label filter landed — was the exact mechanism by which
//! the `id=0` phantom row reached user-facing evidence packs.
//!
//! [`EntityId::try_new`] rejects the above classes and NFC-normalises what
//! remains, guaranteeing that every key entering the entity index is a
//! meaningful, comparable string.
//!
//! # Policy is intentionally global
//!
//! The validation thresholds live as module constants rather than per-call
//! arguments so every call site enforces the same contract. If a corpus
//! needs a looser or stricter policy, override it through the configuration
//! layer (`EntityIdPolicy` in settings), not by calling `try_new` with
//! different arguments from different places.

use std::fmt;

use unicode_normalization::UnicodeNormalization;

/// Minimum number of Unicode scalar values an entity ID must contain.
///
/// Two keeps room for legitimate single-letter symbols in compound form
/// (e.g. "x²", "ch.") while dropping the `"0"` / `"a"` noise class.
pub const MIN_LEN: usize = 2;

/// Maximum accepted length. Guards against an LLM that emits an entire
/// paragraph as a node id — rare, but corrosive when it happens because
/// the malformed key blocks every legitimate entity from deduplicating.
pub const MAX_LEN: usize = 256;

/// Reasons [`EntityId::try_new`] may reject an input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EntityIdError {
    /// Input was empty or whitespace-only.
    #[error("entity id is empty or whitespace-only")]
    Empty,

    /// Input shorter than [`MIN_LEN`].
    #[error("entity id too short: {len} chars (minimum {})", MIN_LEN)]
    TooShort { len: usize },

    /// Input longer than [`MAX_LEN`].
    #[error("entity id too long: {len} chars (cap {})", MAX_LEN)]
    TooLong { len: usize },

    /// Input contains only digits. These collide trivially across chunks
    /// (every page number yields the same node) and never carry semantic
    /// weight in an entity graph.
    #[error("entity id is purely numeric: `{raw}`")]
    NumericOnly { raw: String },
}

impl EntityIdError {
    /// Short machine-friendly label for metrics / structured logs.
    pub fn kind(&self) -> &'static str {
        match self {
            EntityIdError::Empty => "empty",
            EntityIdError::TooShort { .. } => "too_short",
            EntityIdError::TooLong { .. } => "too_long",
            EntityIdError::NumericOnly { .. } => "numeric_only",
        }
    }
}

/// Validated entity identifier.
///
/// Constructed exclusively via [`EntityId::try_new`]; the inner `String`
/// is private so the only way to obtain one is through the validator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntityId(String);

impl EntityId {
    /// Validate and normalise `raw` into an [`EntityId`].
    ///
    /// Pipeline: trim outer whitespace, NFC compose, run the rejection
    /// rules. Returns `Err` with the specific failure class so the caller
    /// can emit a structured metric or log line — production code should
    /// drop failed entities silently (possibly counting them) rather than
    /// aborting the ingest.
    ///
    /// # Errors
    ///
    /// See [`EntityIdError`].
    pub fn try_new(raw: &str) -> Result<Self, EntityIdError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(EntityIdError::Empty);
        }

        let normalised: String = trimmed.nfc().collect();

        let len = normalised.chars().count();
        if len < MIN_LEN {
            return Err(EntityIdError::TooShort { len });
        }
        if len > MAX_LEN {
            return Err(EntityIdError::TooLong { len });
        }

        // Pure-numeric guard. Uses `char::is_numeric` which covers ASCII
        // digits plus Unicode numeric characters (Arabic-Indic, Devanagari
        // digits, superscripts). We intentionally *don't* reject mixed
        // alphanumerics like "H264" — those are legitimate entity names.
        if normalised.chars().all(|c| c.is_numeric()) {
            return Err(EntityIdError::NumericOnly {
                raw: normalised.clone(),
            });
        }

        Ok(Self(normalised))
    }

    /// Borrow the validated string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the newtype and return the validated `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for EntityId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
