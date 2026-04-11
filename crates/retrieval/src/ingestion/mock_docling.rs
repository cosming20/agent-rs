//! Mock Docling sidecar stream for W3's development against the gRPC contract.
//!
//! Not feature-gated (no `#[cfg(test)]`) per project rule "no test code".
//! This is a plain `pub` module that returns a canned stream of
//! [`ChunkDocumentResponse`] messages, allowing W3 to write and verify the
//! tonic client pipeline before the Python sidecar exists.
//!
//! # Usage
//!
//! ```no_run
//! use agent_rs_retrieval::proto::agentic_rag::v1::*;
//! let _ = ChunkDocumentRequest::default();
//! ```
//!
//! # Stream contract
//!
//! Both functions return a stream that emits exactly:
//!   1. One message with `payload = header`
//!   2. N messages with `payload = chunk`
//!   3. One message with `payload = footer`
//!
//! This matches the wire contract defined in `chunker.proto`.
//!
//! # Fixture file
//!
//! [`mock_stream_from_fixture`] reads
//! `crates/retrieval/tests/fixtures/mock_stream.json` (path relative to the
//! manifest dir at runtime). The JSON schema uses lowercase snake_case field
//! names matching the proto field names. Numeric enum values are used for
//! `label` (DocItemLabel ordinal).
//!
//! [`canned_small_doc_stream`] returns the same three-chunk document as a
//! hard-coded stream — no filesystem I/O, safe to use in any context.

use std::path::Path;

use tokio_stream::Stream;
use tonic::Status;

use crate::proto::agentic_rag::v1::{
    Chunk, ChunkDocumentResponse, DocItem, DocItemLabel, DocumentFooter, DocumentHeader, PageInfo,
    chunk_document_response::Payload,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load a mock stream from a JSON fixture file.
///
/// The fixture file must contain a top-level `messages` array where each
/// object has a `"type"` field with one of `"header"`, `"chunk"`, `"footer"`.
/// Field names match the proto snake_case names; numeric enum ordinals are
/// used for `label`.
///
/// # Errors
///
/// Returns a single-element stream containing a [`Status::internal`] error if
/// the file cannot be read or parsed. This keeps the caller's stream-consuming
/// loop uniform — it does not need special error handling for fixture loading.
pub fn mock_stream_from_fixture(
    path: &Path,
) -> impl Stream<Item = Result<ChunkDocumentResponse, Status>> {
    let messages = load_fixture(path);
    tokio_stream::iter(messages)
}

/// Return a canned three-chunk document stream with no I/O.
///
/// Emits: `DocumentHeader` → `Chunk(0)` → `Chunk(1)` → `Chunk(2)` → `DocumentFooter`.
///
/// Intended as the "zero-setup" entry point for W3 to bring up the client
/// pipeline before any file system or network dependencies are available.
pub fn canned_small_doc_stream() -> impl Stream<Item = Result<ChunkDocumentResponse, Status>> {
    tokio_stream::iter(build_canned_messages())
}

// ---------------------------------------------------------------------------
// Fixture loading (serde_json::Value — no derives on generated types)
// ---------------------------------------------------------------------------

fn load_fixture(path: &Path) -> Vec<Result<ChunkDocumentResponse, Status>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return vec![Err(Status::internal(format!(
                "mock_docling: cannot read fixture {:?}: {}",
                path, e
            )))];
        }
    };

    let root: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            return vec![Err(Status::internal(format!(
                "mock_docling: fixture JSON parse error: {}",
                e
            )))];
        }
    };

    let messages = match root.get("messages").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => {
            return vec![Err(Status::internal(
                "mock_docling: fixture missing top-level `messages` array",
            ))];
        }
    };

    messages
        .iter()
        .map(|msg| {
            let msg_type = msg
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            match msg_type {
                "header" => Ok(ChunkDocumentResponse {
                    payload: Some(Payload::Header(parse_header(msg))),
                }),
                "chunk" => Ok(ChunkDocumentResponse {
                    payload: Some(Payload::Chunk(parse_chunk(msg))),
                }),
                "footer" => Ok(ChunkDocumentResponse {
                    payload: Some(Payload::Footer(parse_footer(msg))),
                }),
                other => Err(Status::internal(format!(
                    "mock_docling: unknown message type `{other}` in fixture"
                ))),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// JSON → proto conversions (no serde derives — manual extraction)
// ---------------------------------------------------------------------------

fn parse_header(v: &serde_json::Value) -> DocumentHeader {
    DocumentHeader {
        schema_version: v
            .get("schema_version")
            .and_then(|x| x.as_u64())
            .unwrap_or(1) as u32,
        document_id: str_field(v, "document_id"),
        sha256: str_field(v, "sha256"),
        filename: str_field(v, "filename"),
        mimetype: str_field(v, "mimetype"),
        n_pages: v.get("n_pages").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        language: str_field(v, "language"),
        title: str_field(v, "title"),
        docling_version: str_field(v, "docling_version"),
        docling_core_version: str_field(v, "docling_core_version"),
        parse_latency_ms: v
            .get("parse_latency_ms")
            .and_then(|x| x.as_u64())
            .unwrap_or(0),
        pages: v
            .get("pages")
            .and_then(|x| x.as_array())
            .map(|arr| arr.iter().map(parse_page_info).collect())
            .unwrap_or_default(),
    }
}

fn parse_page_info(v: &serde_json::Value) -> PageInfo {
    PageInfo {
        page_no: v.get("page_no").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        width: v.get("width").and_then(|x| x.as_f64()).unwrap_or(0.0),
        height: v.get("height").and_then(|x| x.as_f64()).unwrap_or(0.0),
    }
}

fn parse_chunk(v: &serde_json::Value) -> Chunk {
    Chunk {
        chunk_index: v.get("chunk_index").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        total_chunks: v.get("total_chunks").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        chunk_id: str_field(v, "chunk_id"),
        embed_text: str_field(v, "embed_text"),
        raw_text: str_field(v, "raw_text"),
        heading_path: v
            .get("heading_path")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        page_start: v.get("page_start").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        page_end: v.get("page_end").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        token_count: v.get("token_count").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        char_count: v.get("char_count").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        doc_items: v
            .get("doc_items")
            .and_then(|x| x.as_array())
            .map(|arr| arr.iter().map(parse_doc_item).collect())
            .unwrap_or_default(),
        has_table: v
            .get("has_table")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        has_picture: v
            .get("has_picture")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        has_code: v.get("has_code").and_then(|x| x.as_bool()).unwrap_or(false),
        has_formula: v
            .get("has_formula")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        section_title: str_field(v, "section_title"),
        section_id: str_field(v, "section_id"),
    }
}

fn parse_doc_item(v: &serde_json::Value) -> DocItem {
    let label_ordinal = v.get("label").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
    // Validate the ordinal is a known DocItemLabel variant; fall back to OTHER.
    let label = DocItemLabel::try_from(label_ordinal)
        .unwrap_or(DocItemLabel::Other)
        .into();

    DocItem {
        self_ref: str_field(v, "self_ref"),
        label,
        raw_label: str_field(v, "raw_label"),
        page_no: v.get("page_no").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        bbox: None, // fixture does not include bbox for simplicity
        text: str_field(v, "text"),
        table_markdown: str_field(v, "table_markdown"),
        table_n_rows: v.get("table_n_rows").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        table_n_cols: v.get("table_n_cols").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        picture_caption: str_field(v, "picture_caption"),
        doc_item_id: str_field(v, "doc_item_id"),
    }
}

fn parse_footer(v: &serde_json::Value) -> DocumentFooter {
    DocumentFooter {
        n_chunks: v.get("n_chunks").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        n_tables: v.get("n_tables").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        n_pictures: v.get("n_pictures").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        n_formulas: v.get("n_formulas").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        n_code: v.get("n_code").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        total_handle_latency_ms: v
            .get("total_handle_latency_ms")
            .and_then(|x| x.as_u64())
            .unwrap_or(0),
    }
}

/// Extract a string field from a JSON value, defaulting to empty string.
fn str_field(v: &serde_json::Value, key: &str) -> String {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("").to_owned()
}

// ---------------------------------------------------------------------------
// Canned messages (no I/O)
// ---------------------------------------------------------------------------

fn build_canned_messages() -> Vec<Result<ChunkDocumentResponse, Status>> {
    const DOC_ID: &str = "7f3b9c2a-1e4d-5f6b-8a9c-0d1e2f3a4b5c";
    const SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    let header = ChunkDocumentResponse {
        payload: Some(Payload::Header(DocumentHeader {
            schema_version: 1,
            document_id: DOC_ID.to_owned(),
            sha256: SHA256.to_owned(),
            filename: "sample_document.pdf".to_owned(),
            mimetype: "application/pdf".to_owned(),
            n_pages: 2,
            language: "en".to_owned(),
            title: "Sample Document for Mock Stream Testing".to_owned(),
            docling_version: "2.86.0".to_owned(),
            docling_core_version: "2.5.0".to_owned(),
            parse_latency_ms: 142,
            pages: vec![
                PageInfo {
                    page_no: 1,
                    width: 612.0,
                    height: 792.0,
                },
                PageInfo {
                    page_no: 2,
                    width: 612.0,
                    height: 792.0,
                },
            ],
        })),
    };

    let chunks: Vec<ChunkDocumentResponse> = [
        (
            0u32,
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "Introduction",
            "Introduction\n\nThis document describes the architecture of the agentic RAG system.",
            "This document describes the architecture of the agentic RAG system.",
            1u32,
            1u32,
        ),
        (
            1u32,
            "b2c3d4e5-f6a7-8901-bcde-f01234567891",
            "Retrieval Strategies",
            "Retrieval Strategies\n\nThe retrieval engine supports five strategies: SEMANTIC, KEYWORD, HYBRID, GRAPH, GRAPH_HYBRID.",
            "The retrieval engine supports five strategies: SEMANTIC, KEYWORD, HYBRID, GRAPH, GRAPH_HYBRID.",
            1u32,
            2u32,
        ),
        (
            2u32,
            "c3d4e5f6-a7b8-9012-cdef-012345678912",
            "Implementation Notes",
            "Implementation Notes\n\nAll components are implemented in Rust using the tokio async runtime.",
            "All components are implemented in Rust using the tokio async runtime.",
            2u32,
            2u32,
        ),
    ]
    .into_iter()
    .map(
        |(idx, chunk_id, section, embed, raw, page_start, page_end)| ChunkDocumentResponse {
            payload: Some(Payload::Chunk(Chunk {
                chunk_index: idx,
                total_chunks: 3,
                chunk_id: chunk_id.to_owned(),
                embed_text: embed.to_owned(),
                raw_text: raw.to_owned(),
                heading_path: vec![section.to_owned()],
                page_start,
                page_end,
                token_count: embed.len() as u32 / 4, // rough approximation
                char_count: embed.len() as u32,
                doc_items: vec![DocItem {
                    self_ref: format!("#/texts/{idx}"),
                    label: DocItemLabel::Text.into(),
                    raw_label: "text".to_owned(),
                    page_no: page_start,
                    bbox: None,
                    text: raw.to_owned(),
                    table_markdown: String::new(),
                    table_n_rows: 0,
                    table_n_cols: 0,
                    picture_caption: String::new(),
                    doc_item_id: String::new(),
                }],
                has_table: false,
                has_picture: false,
                has_code: false,
                has_formula: false,
                section_title: section.to_owned(),
                section_id: String::new(),
            })),
        },
    )
    .collect();

    let footer = ChunkDocumentResponse {
        payload: Some(Payload::Footer(DocumentFooter {
            n_chunks: 3,
            n_tables: 0,
            n_pictures: 0,
            n_formulas: 0,
            n_code: 0,
            total_handle_latency_ms: 287,
        })),
    };

    let mut messages = Vec::with_capacity(5);
    messages.push(Ok(header));
    for chunk in chunks {
        messages.push(Ok(chunk));
    }
    messages.push(Ok(footer));
    messages
}
