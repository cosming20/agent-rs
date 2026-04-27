//! Canonical Qdrant payload schema shared across ingestion and retrieval.
//!
//! Every payload key written to Qdrant (by the indexer) and every payload
//! key matched in a filter (by the retrieval router / metadata extractor)
//! is declared exactly once as a [`PayloadKey`] constant in this module.
//!
//! # Why
//!
//! Payload keys are the schema contract between the indexer and the
//! retriever. Stringly-typed names drift silently: the indexer omits a
//! key, or the retriever mis-spells one, and filters match zero points
//! without surfacing a compile-time error. Centralising the names here
//! means any divergence is caught at build time when the constant is
//! referenced from both sides.
//!
//! # Conventions
//!
//! - Keys are `snake_case` strings exactly as persisted in Qdrant.
//! - Boolean presence flags are prefixed with `has_` (e.g. [`HAS_TABLE`]).
//! - Per-section boundary fields are prefixed with `section_`.
//! - Per-chunk position fields use `chunk_` / `page_` / `byte_` prefixes.
//! - Free-form free-text fields use the domain noun (`source`, `language`).
//!
//! # Usage
//!
//! ```rust,no_run
//! use agent_rs_core::payload;
//! use serde_json::json;
//!
//! let doc = json!({
//!     payload::USER_ID.as_str():    "u-abc",
//!     payload::CHUNK_ID.as_str():   "c-123",
//!     payload::CHUNK_TEXT.as_str(): "…",
//!     payload::HAS_FORMULA.as_str(): true,
//! });
//! ```

// ---------------------------------------------------------------------------
// PayloadKey
// ---------------------------------------------------------------------------

/// A Qdrant payload field name.
///
/// Wraps a `&'static str` so that payload keys are:
/// - authoritatively sourced from the constants in this module;
/// - impossible to mis-spell at the call site (the name of the constant
///   *is* the compile-time identifier);
/// - directly usable with `serde_json::json!` and Qdrant filter builders
///   via [`PayloadKey::as_str`].
///
/// Equality and hashing are borrowed from the underlying string so a
/// `PayloadKey` can be used interchangeably with `&str` in `HashMap`
/// lookups against payload-field maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PayloadKey(&'static str);

impl PayloadKey {
    /// Construct a new key. `const fn` so every key can live as a
    /// compile-time constant.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Return the underlying field name as `&'static str`.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for PayloadKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl From<PayloadKey> for String {
    fn from(k: PayloadKey) -> String {
        k.0.to_owned()
    }
}

impl AsRef<str> for PayloadKey {
    fn as_ref(&self) -> &str {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Canonical keys — chunks collection (default)
// ---------------------------------------------------------------------------

/// Tenant scope. Mandatory on every point; filtered unconditionally by
/// `VectorStore::search_for_user` / `keyword_search_for_user`.
pub const USER_ID: PayloadKey = PayloadKey::new("user_id");

/// Opaque document identifier (typically the MinIO object key or a UUID
/// supplied by upstream). Stable across re-ingest of the same bytes.
pub const DOC_ID: PayloadKey = PayloadKey::new("doc_id");

/// Primary chunk identifier (UUID v5 from the chunk text). Globally
/// unique, stable across re-ingests.
pub const CHUNK_ID: PayloadKey = PayloadKey::new("chunk_id");

/// Raw chunk text as stored for BM25 / context rendering.
pub const CHUNK_TEXT: PayloadKey = PayloadKey::new("chunk_text");

/// Zero-based position of this chunk in the sequence produced by the
/// chunker.
pub const CHUNK_INDEX: PayloadKey = PayloadKey::new("chunk_index");

/// Total number of chunks produced by the chunker for this document.
pub const TOTAL_CHUNKS: PayloadKey = PayloadKey::new("total_chunks");

/// The chunk's logical section identifier (hierarchy UUID) when the
/// chunker can emit one; `null` otherwise.
pub const SECTION_ID: PayloadKey = PayloadKey::new("section_id");

/// Rendered heading path (e.g. `"Ch 5 / 5.2 / 5.2.1"`). Used by
/// outline-style tools and by filters that substring-match a section
/// reference.
pub const SECTION_PATH: PayloadKey = PayloadKey::new("section_path");

/// First heading-like line detected in the chunk. Populated by
/// `retrieval::ingestion::metadata::extract_metadata`.
pub const SECTION_TITLE: PayloadKey = PayloadKey::new("section_title");

/// 1-based inclusive page where the chunk starts.
pub const PAGE_START: PayloadKey = PayloadKey::new("page_start");

/// 1-based inclusive page where the chunk ends.
pub const PAGE_END: PayloadKey = PayloadKey::new("page_end");

/// MinIO object key. Persisted so the agent can resolve back to the raw
/// source object during lineage queries.
pub const MINIO_OBJECT_KEY: PayloadKey = PayloadKey::new("minio_object_key");

/// Ranker-facing source identifier. Equal to `minio_object_key` in the
/// stateless architecture.
pub const SOURCE: PayloadKey = PayloadKey::new("source");

/// Detected ISO 639-1 language code (e.g. `"ro"`, `"en"`). `null` when
/// the heuristic lacks confidence.
pub const LANGUAGE: PayloadKey = PayloadKey::new("language");

// ---------------------------------------------------------------------------
// Canonical keys — content-type presence flags
// ---------------------------------------------------------------------------
//
// These booleans are the targets of LLM metadata-hint filters
// (`requires_table`, `requires_formula`, `requires_code` in
// `retrieval::metadata_filter`). They MUST be written at ingest time so
// those filters are non-empty — otherwise the retriever narrows to a
// payload schema the indexer never wrote, producing zero results.

/// `true` when the chunk contains table-like markup.
pub const HAS_TABLE: PayloadKey = PayloadKey::new("has_table");

/// `true` when the chunk contains a formula or equation-like fragment.
pub const HAS_FORMULA: PayloadKey = PayloadKey::new("has_formula");

/// `true` when the chunk contains a source-code fragment.
pub const HAS_CODE: PayloadKey = PayloadKey::new("has_code");

/// `true` when the chunk contains bullet / numbered list items.
pub const HAS_LIST: PayloadKey = PayloadKey::new("has_list");

// ---------------------------------------------------------------------------
// Canonical keys — summaries collection (default_summaries)
// ---------------------------------------------------------------------------

/// Primary summary identifier (UUID v5 from the summary text).
pub const SUMMARY_ID: PayloadKey = PayloadKey::new("summary_id");

/// Raw summary text (one to two sentences).
pub const SUMMARY_TEXT: PayloadKey = PayloadKey::new("summary_text");

// ---------------------------------------------------------------------------
// Canonical keys — triplets collection (default_triplets)
// ---------------------------------------------------------------------------

/// Source-entity node identifier in the knowledge graph.
pub const SOURCE_NODE_ID: PayloadKey = PayloadKey::new("source_node_id");

/// Display name of the source entity.
pub const SOURCE_NODE_NAME: PayloadKey = PayloadKey::new("source_node_name");

/// Relationship type / label connecting source → target.
pub const RELATIONSHIP_NAME: PayloadKey = PayloadKey::new("relationship_name");

/// Target-entity node identifier.
pub const TARGET_NODE_ID: PayloadKey = PayloadKey::new("target_node_id");

/// Display name of the target entity.
pub const TARGET_NODE_NAME: PayloadKey = PayloadKey::new("target_node_name");

/// Linearised `"<source> <relation> <target>"` string used as the
/// sparse BM25 query text and embedder input for triplet points.
pub const TRIPLET_TEXT: PayloadKey = PayloadKey::new("triplet_text");

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique() {
        // The full catalogue — if we ever duplicate a constant or
        // introduce a collision with a raw string used in retrieval,
        // this test surfaces it.
        let all = [
            USER_ID,
            DOC_ID,
            CHUNK_ID,
            CHUNK_TEXT,
            CHUNK_INDEX,
            TOTAL_CHUNKS,
            SECTION_ID,
            SECTION_PATH,
            SECTION_TITLE,
            PAGE_START,
            PAGE_END,
            MINIO_OBJECT_KEY,
            SOURCE,
            LANGUAGE,
            HAS_TABLE,
            HAS_FORMULA,
            HAS_CODE,
            HAS_LIST,
            SUMMARY_ID,
            SUMMARY_TEXT,
            SOURCE_NODE_ID,
            SOURCE_NODE_NAME,
            RELATIONSHIP_NAME,
            TARGET_NODE_ID,
            TARGET_NODE_NAME,
            TRIPLET_TEXT,
        ];
        let unique: std::collections::HashSet<&'static str> =
            all.iter().map(PayloadKey::as_str).collect();
        assert_eq!(
            unique.len(),
            all.len(),
            "payload key constants must be unique"
        );
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(HAS_FORMULA.as_str(), "has_formula");
        assert_eq!(format!("{}", HAS_FORMULA), "has_formula");
    }
}
