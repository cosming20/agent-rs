//! Authoritative payload builders for Qdrant points.
//!
//! A single module owns the mapping from in-memory ingest records to the
//! JSON payload persisted in Qdrant. Every field name comes from
//! [`agent_rs_core::payload`] constants, so any divergence with retrieval-
//! side filters (metadata extractor, section predicate, triplet reader)
//! is a compile-time error rather than a silent zero-result at runtime.
//!
//! # Why this module exists
//!
//! Before this module existed, `crates/agent-service/src/indexer.rs` built
//! payloads inline with `serde_json::json!({...})`. That approach made it
//! easy for the indexer to omit a key (e.g. `has_formula`) that
//! [`crate::metadata_filter::MetadataFilterExtractor`] would then match
//! against — producing zero-result searches on perfectly indexed
//! documents. Centralising the builders here eliminates that drift.
//!
//! # Collections
//!
//! - [`build_chunk_payload`] — primary chunks collection (`default`).
//! - [`build_summary_payload`] — per-section summaries
//!   (`default_summaries`).
//! - [`build_triplet_payload`] — knowledge-graph triplets
//!   (`default_triplets`).

use agent_rs_core::payload::{self, PayloadKey};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::ingestion::metadata::ChunkMetadata;

/// Ergonomic wrapper around `serde_json::Map::insert` that takes a typed
/// [`PayloadKey`] rather than a stringly-typed field name. Keeps the
/// payload builders readable while guaranteeing every key comes from the
/// canonical schema module.
fn put<V: Into<Value>>(map: &mut Map<String, Value>, key: PayloadKey, value: V) {
    map.insert(key.as_str().to_owned(), value.into());
}

// ---------------------------------------------------------------------------
// ChunkPayloadInput
// ---------------------------------------------------------------------------

/// Structural data for a single chunk, carried from the chunker to the
/// Qdrant upsert.
///
/// Borrows rather than owns so callers can build the input on the fly
/// from their existing `IndexedChunk`-style types without allocation.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ChunkPayloadInput<'a> {
    pub user_id: &'a str,
    pub doc_id: &'a str,
    pub minio_object_key: &'a str,
    pub source: &'a str,
    pub chunk_id: &'a str,
    pub chunk_text: &'a str,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub section_id: Option<&'a str>,
    pub section_path: Option<&'a str>,
    pub page_start: Option<u32>,
    pub page_end: Option<u32>,
}

/// Build the Qdrant payload for a primary chunk point.
///
/// `metadata` supplies the content-type flags (`has_table`,
/// `has_formula`, `has_code`, `has_list`), the detected `section_title`,
/// and the `language` hint. Pass `None` when the caller does not have
/// a metadata record — in that case the content-type flags default to
/// `false` and optional fields are emitted as `null` so every payload
/// shares the same schema.
///
/// The returned `Value` is a JSON object with only the keys declared in
/// [`agent_rs_core::payload`]. No additional fields are added here.
pub fn build_chunk_payload(
    input: ChunkPayloadInput<'_>,
    metadata: Option<&ChunkMetadata>,
) -> Value {
    // Carry over optional fields from metadata with safe defaults. The
    // defaults are chosen so that `MetadataFilterExtractor`-generated
    // predicates behave consistently: `has_*` defaults to `false`, which
    // means "no, this chunk does not have X" (the predicate narrows
    // *away from* this chunk when the corresponding `requires_*` hint
    // fires — which is the correct behaviour when detection did not
    // run).
    let has_table = metadata.map(|m| m.has_table).unwrap_or(false);
    let has_formula = metadata.map(|m| m.has_formula).unwrap_or(false);
    let has_code = metadata.map(|m| m.has_code).unwrap_or(false);
    let has_list = metadata.map(|m| m.has_list).unwrap_or(false);
    let section_title: Option<&str> = metadata.and_then(|m| m.section_title.as_deref());
    let language: Option<&str> = metadata.and_then(|m| m.language.as_deref());

    let mut map = Map::with_capacity(18);
    put(&mut map, payload::USER_ID, input.user_id);
    put(&mut map, payload::DOC_ID, input.doc_id);
    put(&mut map, payload::MINIO_OBJECT_KEY, input.minio_object_key);
    put(&mut map, payload::SOURCE, input.source);
    put(&mut map, payload::CHUNK_ID, input.chunk_id);
    put(&mut map, payload::CHUNK_TEXT, input.chunk_text);
    put(&mut map, payload::CHUNK_INDEX, input.chunk_index);
    put(&mut map, payload::TOTAL_CHUNKS, input.total_chunks);
    put(&mut map, payload::SECTION_ID, json!(input.section_id));
    put(&mut map, payload::SECTION_PATH, json!(input.section_path));
    put(&mut map, payload::SECTION_TITLE, json!(section_title));
    put(&mut map, payload::PAGE_START, json!(input.page_start));
    put(&mut map, payload::PAGE_END, json!(input.page_end));
    put(&mut map, payload::LANGUAGE, json!(language));
    put(&mut map, payload::HAS_TABLE, has_table);
    put(&mut map, payload::HAS_FORMULA, has_formula);
    put(&mut map, payload::HAS_CODE, has_code);
    put(&mut map, payload::HAS_LIST, has_list);
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// SummaryPayloadInput
// ---------------------------------------------------------------------------

/// Structural data for a single per-section summary point.
///
/// A "summary point" is a Qdrant point in the `default_summaries`
/// collection that carries an LLM-generated synopsis of a *source
/// chunk*. The lineage fields distinguish:
///
/// - [`Self::summary_id`] — unique id of **this** summary point. Used
///   as the Qdrant point id and as the payload key when downstream
///   callers need to look the summary up directly.
/// - [`Self::chunk_id`] — id of the **source chunk** being summarised.
///   Used by the retrieval side to hop from a matched summary back to
///   the full chunk for citation and context expansion. This is *not*
///   the summary's own id — keep the distinction tight to avoid
///   cross-wiring summary lookups.
/// - [`Self::section_id`] — id of the source chunk's enclosing section,
///   when one was detected. Enables "retrieve all summaries in section"
///   style filtering.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SummaryPayloadInput<'a> {
    pub user_id: &'a str,
    pub minio_object_key: &'a str,
    /// ID of the chunk that this summary summarises (lineage — not the
    /// summary's own id, see [`Self::summary_id`]).
    pub chunk_id: &'a str,
    pub section_id: Option<&'a str>,
    /// Unique identifier for this summary point.
    pub summary_id: &'a str,
    pub summary_text: &'a str,
}

/// Build the Qdrant payload for a single summary point
/// (`default_summaries` collection).
pub fn build_summary_payload(input: SummaryPayloadInput<'_>) -> Value {
    let mut map = Map::with_capacity(6);
    put(&mut map, payload::USER_ID, input.user_id);
    put(&mut map, payload::MINIO_OBJECT_KEY, input.minio_object_key);
    put(&mut map, payload::CHUNK_ID, input.chunk_id);
    put(&mut map, payload::SECTION_ID, json!(input.section_id));
    put(&mut map, payload::SUMMARY_ID, input.summary_id);
    put(&mut map, payload::SUMMARY_TEXT, input.summary_text);
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// TripletPayloadInput
// ---------------------------------------------------------------------------

/// Structural data for a single knowledge-graph triplet point.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TripletPayloadInput<'a> {
    pub user_id: &'a str,
    pub chunk_id: &'a str,
    pub source_node_id: &'a str,
    pub source_node_name: &'a str,
    pub relationship_name: &'a str,
    pub target_node_id: &'a str,
    pub target_node_name: &'a str,
    pub triplet_text: &'a str,
}

/// Build the Qdrant payload for a single triplet point
/// (`default_triplets` collection).
pub fn build_triplet_payload(input: TripletPayloadInput<'_>) -> Value {
    let mut map = Map::with_capacity(8);
    put(&mut map, payload::USER_ID, input.user_id);
    put(&mut map, payload::CHUNK_ID, input.chunk_id);
    put(&mut map, payload::SOURCE_NODE_ID, input.source_node_id);
    put(&mut map, payload::SOURCE_NODE_NAME, input.source_node_name);
    put(
        &mut map,
        payload::RELATIONSHIP_NAME,
        input.relationship_name,
    );
    put(&mut map, payload::TARGET_NODE_ID, input.target_node_id);
    put(&mut map, payload::TARGET_NODE_NAME, input.target_node_name);
    put(&mut map, payload::TRIPLET_TEXT, input.triplet_text);
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> ChunkMetadata {
        ChunkMetadata {
            source: "doc.pdf".into(),
            page_number: None,
            section_title: Some("Section 5.2".into()),
            headings: vec![],
            has_table: false,
            has_formula: true,
            has_code: false,
            has_list: false,
            language: Some("ro".into()),
            char_count: 1234,
            word_count: 250,
            token_count: 500,
            byte_start: 0,
            byte_end: 1234,
            paragraph_ids: vec![],
        }
    }

    #[test]
    fn chunk_payload_carries_all_canonical_keys() {
        let input = ChunkPayloadInput {
            user_id: "u",
            doc_id: "d",
            minio_object_key: "k",
            source: "k",
            chunk_id: "c",
            chunk_text: "hello",
            chunk_index: 0,
            total_chunks: 1,
            section_id: Some("s"),
            section_path: Some("1 / 1.1"),
            page_start: Some(1),
            page_end: Some(1),
        };
        let meta = sample_metadata();
        let v = build_chunk_payload(input, Some(&meta));

        // Spot-check every payload key declared for the chunk schema.
        for key in [
            payload::USER_ID,
            payload::DOC_ID,
            payload::MINIO_OBJECT_KEY,
            payload::SOURCE,
            payload::CHUNK_ID,
            payload::CHUNK_TEXT,
            payload::CHUNK_INDEX,
            payload::TOTAL_CHUNKS,
            payload::SECTION_ID,
            payload::SECTION_PATH,
            payload::SECTION_TITLE,
            payload::PAGE_START,
            payload::PAGE_END,
            payload::LANGUAGE,
            payload::HAS_TABLE,
            payload::HAS_FORMULA,
            payload::HAS_CODE,
            payload::HAS_LIST,
        ] {
            assert!(v.get(key.as_str()).is_some(), "missing key: {key}");
        }
        assert_eq!(v[payload::HAS_FORMULA.as_str()], true);
        assert_eq!(v[payload::HAS_TABLE.as_str()], false);
        assert_eq!(v[payload::SECTION_TITLE.as_str()], "Section 5.2");
    }

    #[test]
    fn chunk_payload_defaults_without_metadata() {
        let input = ChunkPayloadInput {
            user_id: "u",
            doc_id: "d",
            minio_object_key: "k",
            source: "k",
            chunk_id: "c",
            chunk_text: "hello",
            chunk_index: 0,
            total_chunks: 1,
            section_id: None,
            section_path: None,
            page_start: None,
            page_end: None,
        };
        let v = build_chunk_payload(input, None);
        assert_eq!(v[payload::HAS_FORMULA.as_str()], false);
        assert_eq!(v[payload::HAS_TABLE.as_str()], false);
        assert!(v[payload::SECTION_TITLE.as_str()].is_null());
        assert!(v[payload::LANGUAGE.as_str()].is_null());
    }

    #[test]
    fn summary_and_triplet_payloads_build() {
        let s = build_summary_payload(SummaryPayloadInput {
            user_id: "u",
            minio_object_key: "k",
            chunk_id: "c",
            section_id: Some("s"),
            summary_id: "sid",
            summary_text: "…",
        });
        assert_eq!(s[payload::SUMMARY_ID.as_str()], "sid");

        let t = build_triplet_payload(TripletPayloadInput {
            user_id: "u",
            chunk_id: "c",
            source_node_id: "n1",
            source_node_name: "Alpha",
            relationship_name: "RELATES_TO",
            target_node_id: "n2",
            target_node_name: "Beta",
            triplet_text: "Alpha RELATES_TO Beta",
        });
        assert_eq!(t[payload::TRIPLET_TEXT.as_str()], "Alpha RELATES_TO Beta");
    }
}
