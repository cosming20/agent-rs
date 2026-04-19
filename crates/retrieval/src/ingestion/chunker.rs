//! Universal text chunking for document ingestion.
//!
//! Provides four configurable strategies:
//! - **Recursive character split** -- paragraphs, then sentences, then words
//! - **Section-based** -- split at detected headings/section boundaries
//! - **Sliding window** -- fixed-size window with configurable overlap
//! - **Token-aware paragraph** -- paragraph grouping bounded by cl100k_base token
//!   counts rather than character counts; respects embedding-model limits
//!   (text-embedding-3-large: 8 191 tokens)
//!
//! ## Chunk identity
//!
//! Chunk IDs are **deterministic UUID v5** values derived from the chunk content
//! using the OID namespace.  Re-ingesting the same document produces the same
//! IDs, enabling idempotent upserts into the vector store.
//!
//! ## Configuration defaults
//!
//! The default strategy is [`ChunkStrategy::TokenAwareParagraph`] with
//! `max_chunk_size = 1500` tokens, `chunk_overlap = 100` tokens, and
//! `min_chunk_size = 50` tokens.  These values comfortably fit within the
//! 8 191-token context window of text-embedding-3-large while leaving room
//! for any prepended instruction prefix.

use tiktoken_rs::cl100k_base;
use uuid::Uuid;

use super::metadata::{self, ChunkExtras, ChunkMetadata};

// ---- Configuration ----

/// Chunking strategy selector.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    /// Split by paragraphs, then sentences, then words.  Uses *character* counts.
    RecursiveCharacter,
    /// Split at heading/section boundaries detected via heuristics.  Uses *character* counts.
    SectionBased,
    /// Fixed-size sliding window with configurable overlap.  Uses *character* counts.
    SlidingWindow,
    /// Group paragraphs by *token* count (cl100k_base).  This is the recommended
    /// strategy when the downstream embedder is OpenAI text-embedding-3-large.
    #[default]
    TokenAwareParagraph,
}

/// Configuration for the text chunker.
///
/// The `max_chunk_size`, `chunk_overlap`, and `min_chunk_size` fields are
/// interpreted as **tokens** for [`ChunkStrategy::TokenAwareParagraph`] and as
/// **characters** for all other strategies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkerConfig {
    /// Which chunking strategy to use.
    pub strategy: ChunkStrategy,
    /// Maximum tokens (or characters) per chunk.
    pub max_chunk_size: usize,
    /// Overlap between adjacent chunks (tokens or characters).
    pub chunk_overlap: usize,
    /// Minimum chunk size; fragments smaller than this are merged or dropped.
    pub min_chunk_size: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::default(),
            // 1 500 tokens -- comfortably below the 8 191-token model limit.
            max_chunk_size: 1500,
            // 100-token overlap preserves cross-chunk context.
            chunk_overlap: 100,
            // Discard fragments shorter than 50 tokens / chars.
            min_chunk_size: 50,
        }
    }
}

// ---- Output types ----

/// A single chunk produced from a document.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentChunk {
    /// Deterministic identifier for this chunk (UUID v5, OID namespace, content-keyed).
    ///
    /// Re-ingesting identical content produces the same ID, enabling idempotent
    /// vector-store upserts.
    pub chunk_id: String,
    /// Parent document identifier.
    pub doc_id: String,
    /// The text content of this chunk.
    pub content: String,
    /// Extracted metadata for this chunk.
    pub metadata: ChunkMetadata,
    /// Zero-based position of this chunk within the document.
    pub chunk_index: usize,
    /// Total number of chunks the document was split into.
    pub total_chunks: usize,
    /// Section identifier from the structural layer (e.g. a Neo4j section UUID).
    ///
    /// Populated by the Docling ingestion path via `proto_chunk_to_document_chunk`;
    /// `None` on the legacy plain-text path where section structure is not available.
    pub section_id: Option<String>,
}

// ---- Public API ----

/// Chunk a document into [`DocumentChunk`]s using the configured strategy.
///
/// # Arguments
///
/// * `text`   -- full document text
/// * `doc_id` -- parent document identifier
/// * `source` -- filename or URL of the source document
/// * `config` -- chunking configuration
///
/// # Returns
///
/// A vector of chunks with metadata, ordered by position.  Empty or
/// whitespace-only input returns an empty vector.
///
/// # Notes
///
/// Chunk IDs are UUID v5 values derived from the content.  Two chunks with
/// identical content (across different documents) will therefore share an ID.
/// If that is undesirable, include `doc_id` in the ID seed before calling this
/// function -- or use a different namespace per document.
#[tracing::instrument(
    skip(text),
    fields(
        doc_id  = doc_id,
        source  = source,
        strategy = ?config.strategy,
        text_len = text.len(),
    )
)]
pub fn chunk_document(
    text: &str,
    doc_id: &str,
    source: &str,
    config: &ChunkerConfig,
) -> Vec<DocumentChunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    // Dispatch to the selected strategy.  Token-aware strategy returns rich
    // (content, extras) pairs; the character-based strategies return plain strings
    // and we synthesise byte offsets from the content.
    let raw: Vec<(String, ChunkExtras)> = match config.strategy {
        ChunkStrategy::RecursiveCharacter => {
            char_chunks_to_pairs(recursive_character_split(text, config), text)
        }
        ChunkStrategy::SectionBased => {
            char_chunks_to_pairs(section_based_split(text, config), text)
        }
        ChunkStrategy::SlidingWindow => {
            char_chunks_to_pairs(sliding_window_split(text, config), text)
        }
        ChunkStrategy::TokenAwareParagraph => token_aware_paragraph_split(text, config),
    };

    // Filter: for token-aware use token_count, otherwise use char length.
    let filtered: Vec<(String, ChunkExtras)> = raw
        .into_iter()
        .filter(|(content, extras)| {
            let size = if extras.token_count > 0 {
                extras.token_count
            } else {
                content.trim().len()
            };
            size >= config.min_chunk_size
        })
        .collect();

    let total_chunks = filtered.len();
    let mut headings: Vec<String> = Vec::new();

    filtered
        .into_iter()
        .enumerate()
        .map(|(chunk_index, (content, extras))| {
            let meta = metadata::extract_metadata(&content, source, &headings, extras);

            // Maintain heading hierarchy for subsequent chunks.
            if let Some(ref title) = meta.section_title {
                if let Some(pos) = headings.iter().position(|h| h == title) {
                    headings.truncate(pos + 1);
                } else {
                    headings.push(title.clone());
                }
            }

            // Deterministic chunk ID: UUID v5(OID namespace, content bytes).
            let chunk_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, content.as_bytes()).to_string();

            DocumentChunk {
                chunk_id,
                doc_id: doc_id.to_owned(),
                content,
                metadata: meta,
                chunk_index,
                total_chunks,
                section_id: None,
            }
        })
        .collect()
}

// ---- Token-aware paragraph strategy ----

/// Split `text` into token-bounded paragraph groups using cl100k_base.
///
/// Algorithm:
/// 1. Split on `\n\n` (paragraph breaks) and then `\n` (single newlines).
/// 2. Count tokens per paragraph with the cl100k_base BPE tokeniser.
/// 3. Group consecutive paragraphs until the token budget is exhausted.
/// 4. When a single paragraph exceeds `max_chunk_size`, split it by sentences.
/// 5. Apply a token-based overlap: carry the last `chunk_overlap` tokens from
///    the previous chunk into the current one.
///
/// Returns `(content, ChunkExtras)` pairs with populated `token_count`,
/// `byte_start`, `byte_end`, and `paragraph_ids`.
fn token_aware_paragraph_split(text: &str, config: &ChunkerConfig) -> Vec<(String, ChunkExtras)> {
    // Initialise the tokeniser.  cl100k_base is used by text-embedding-3-large.
    // `cl100k_base()` returns `Result<CoreBPE>` and is cheap after the first call
    // because tiktoken-rs caches the vocab internally.
    let bpe = match cl100k_base() {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = ?e, "failed to initialise cl100k_base tokeniser; falling back to recursive character split");
            return char_chunks_to_pairs(recursive_character_split(text, config), text);
        }
    };

    // ---- Step 1: collect paragraphs with their byte offsets ----
    // Split on blank lines first, then on single newlines so that a paragraph
    // that contains no blank lines is still processed as one unit.
    let paragraphs: Vec<(&str, usize)> = collect_paragraphs(text);

    if paragraphs.is_empty() {
        return Vec::new();
    }

    // ---- Step 2: tokenise each paragraph ----
    let para_tokens: Vec<(&str, usize, usize, usize)> = paragraphs
        .iter()
        .map(|(para, byte_start)| {
            let token_count = bpe.encode_ordinary(para).len();
            (*para, *byte_start, byte_start + para.len(), token_count)
        })
        .collect();

    // ---- Step 3 + 4 + 5: group into chunks ----
    let max = config.max_chunk_size;
    let overlap_budget = config.chunk_overlap;

    let mut chunks: Vec<(String, ChunkExtras)> = Vec::new();

    // Current accumulator.
    let mut current_paras: Vec<(&str, usize, usize, usize)> = Vec::new(); // (text, byte_start, byte_end, tokens)
    let mut current_tokens: usize = 0;

    // Overlap carry-over from the previous chunk (already tokenised).
    let mut overlap_text = String::new();
    let mut overlap_tokens: usize = 0;

    let flush = |current_paras: &[(&str, usize, usize, usize)],
                 overlap_prefix: &str|
     -> Option<(String, ChunkExtras)> {
        if current_paras.is_empty() {
            return None;
        }

        let body: String = current_paras
            .iter()
            .map(|(p, _, _, _)| *p)
            .collect::<Vec<_>>()
            .join("\n\n");

        let content = if overlap_prefix.is_empty() {
            body
        } else {
            format!("{overlap_prefix}\n\n{body}")
        };

        let byte_start = current_paras[0].1;
        let byte_end = current_paras[current_paras.len() - 1].2;
        let token_count = bpe.encode_ordinary(&content).len();

        let paragraph_ids = current_paras
            .iter()
            .map(|(p, _, _, _)| Uuid::new_v5(&Uuid::NAMESPACE_OID, p.as_bytes()).to_string())
            .collect();

        Some((
            content,
            ChunkExtras {
                token_count,
                byte_start,
                byte_end,
                paragraph_ids,
            },
        ))
    };

    for (para_text, byte_start, byte_end, para_token_count) in &para_tokens {
        let para_text = *para_text;
        let byte_start = *byte_start;
        let byte_end = *byte_end;
        let para_token_count = *para_token_count;

        if para_token_count > max {
            // This paragraph alone exceeds the budget.
            // Flush whatever we have accumulated first.
            if let Some(chunk) = flush(&current_paras, &overlap_text) {
                overlap_text = build_token_overlap(&chunk.0, overlap_budget, &bpe);
                chunks.push(chunk);
            }
            current_paras.clear();
            current_tokens = 0;

            // Split the oversized paragraph by sentences.
            let sentence_chunks =
                split_paragraph_by_sentences(para_text, byte_start, max, overlap_budget, &bpe);
            for sc in sentence_chunks {
                let saved_overlap = overlap_text.clone();
                overlap_text = build_token_overlap(&sc.0, overlap_budget, &bpe);
                // Prepend overlap from before this oversized paragraph.
                let (mut content, mut extras) = sc;
                if !saved_overlap.is_empty() {
                    let combined = format!("{saved_overlap}\n\n{content}");
                    extras.token_count = bpe.encode_ordinary(&combined).len();
                    content = combined;
                }
                chunks.push((content, extras));
            }
            // Recompute overlap_tokens from whatever overlap_text ended up as.
            overlap_tokens = bpe.encode_ordinary(&overlap_text).len();
            continue;
        }

        // Does adding this paragraph (plus separator) push us over the limit?
        let sep_tokens = if current_paras.is_empty() { 0 } else { 2 }; // "\n\n" ≈ 2 tokens
        let projected = current_tokens + overlap_tokens + sep_tokens + para_token_count;

        if projected > max && !current_paras.is_empty() {
            // Flush current accumulator.
            if let Some(chunk) = flush(&current_paras, &overlap_text) {
                overlap_text = build_token_overlap(&chunk.0, overlap_budget, &bpe);
                overlap_tokens = bpe.encode_ordinary(&overlap_text).len();
                chunks.push(chunk);
            }
            current_paras.clear();
            current_tokens = 0;
        }

        current_paras.push((para_text, byte_start, byte_end, para_token_count));
        current_tokens += para_token_count;
    }

    // Flush the final accumulator.
    if let Some(chunk) = flush(&current_paras, &overlap_text) {
        chunks.push(chunk);
    }

    chunks
}

/// Collect non-empty paragraph slices from `text` together with their byte offsets.
///
/// Splits on `\n\n` (blank line) and filters out whitespace-only segments.
fn collect_paragraphs(text: &str) -> Vec<(&str, usize)> {
    let mut result: Vec<(&str, usize)> = Vec::new();
    let mut offset = 0usize;

    for raw in text.split("\n\n") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            // Find the trimmed slice's actual start within `text`.
            let leading = raw.as_ptr() as usize - text.as_ptr() as usize;
            let trim_leading = raw.find(|c: char| !c.is_whitespace()).unwrap_or(0);
            result.push((trimmed, leading + trim_leading));
        }
        offset += raw.len() + 2; // +2 for the "\n\n" separator
        let _ = offset; // silence unused warning; offset is computed for clarity
    }

    result
}

/// Split a paragraph that exceeds `max_tokens` into sentence-bounded sub-chunks.
///
/// Falls back to hard token-split if no sentence boundaries are found.
fn split_paragraph_by_sentences(
    text: &str,
    byte_start: usize,
    max_tokens: usize,
    overlap_budget: usize,
    bpe: &tiktoken_rs::CoreBPE,
) -> Vec<(String, ChunkExtras)> {
    // Simple sentence splitter: split on ". ", "! ", "? ", ".\n", "!\n", "?\n".
    let mut sentences: Vec<&str> = Vec::new();
    let mut last = 0usize;
    let bytes = text.as_bytes();
    let len = bytes.len();

    let mut i = 0usize;
    while i < len {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            let next = i + 1;
            if next < len && (bytes[next] == b' ' || bytes[next] == b'\n') {
                sentences.push(&text[last..=i]);
                last = next + 1;
                i = last;
                continue;
            }
        }
        i += 1;
    }
    // Remainder
    if last < len {
        let remainder = text[last..].trim();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }
    }

    if sentences.is_empty() {
        // No sentence boundaries found; hard token-split.
        return hard_token_split(text, byte_start, max_tokens, overlap_budget, bpe);
    }

    // Group sentences into token-bounded chunks.
    let mut chunks: Vec<(String, ChunkExtras)> = Vec::new();
    let mut current = String::new();
    let mut current_byte_start = byte_start;
    let mut current_byte_end = byte_start;
    let mut overlap_text = String::new();
    let mut overlap_tokens = 0usize;

    for sentence in &sentences {
        let sentence_tokens = bpe.encode_ordinary(sentence).len();
        let sep_tokens: usize = if current.is_empty() { 0 } else { 1 };
        let current_tok = if current.is_empty() {
            0
        } else {
            bpe.encode_ordinary(&current).len()
        };
        let projected_tok = overlap_tokens + current_tok + sep_tokens + sentence_tokens;

        if projected_tok > max_tokens && !current.is_empty() {
            let content = if overlap_text.is_empty() {
                current.clone()
            } else {
                format!("{overlap_text} {current}")
            };
            let token_count = bpe.encode_ordinary(&content).len();
            let para_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, current.as_bytes()).to_string();
            chunks.push((
                content.clone(),
                ChunkExtras {
                    token_count,
                    byte_start: current_byte_start,
                    byte_end: current_byte_end,
                    paragraph_ids: vec![para_id],
                },
            ));
            overlap_text = build_token_overlap(&content, overlap_budget, bpe);
            overlap_tokens = bpe.encode_ordinary(&overlap_text).len();
            current_byte_start = current_byte_end;
            current = String::new();
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(sentence);
        current_byte_end = byte_start + text.find(&current[..]).unwrap_or(0) + current.len();
    }

    if !current.trim().is_empty() {
        let content = if overlap_text.is_empty() {
            current.clone()
        } else {
            format!("{overlap_text} {current}")
        };
        let token_count = bpe.encode_ordinary(&content).len();
        let para_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, current.as_bytes()).to_string();
        chunks.push((
            content,
            ChunkExtras {
                token_count,
                byte_start: current_byte_start,
                byte_end: current_byte_end,
                paragraph_ids: vec![para_id],
            },
        ));
    }

    chunks
}

/// Hard token-split: encode the full text and slice the token list into
/// windows of `max_tokens` with `overlap_budget` token overlap.
fn hard_token_split(
    text: &str,
    byte_start: usize,
    max_tokens: usize,
    overlap_budget: usize,
    bpe: &tiktoken_rs::CoreBPE,
) -> Vec<(String, ChunkExtras)> {
    let token_ids = bpe.encode_ordinary(text);
    if token_ids.is_empty() {
        return Vec::new();
    }

    let step = if max_tokens > overlap_budget {
        max_tokens - overlap_budget
    } else {
        max_tokens
    };

    let mut chunks: Vec<(String, ChunkExtras)> = Vec::new();
    let mut start = 0usize;

    while start < token_ids.len() {
        let end = (start + max_tokens).min(token_ids.len());
        let slice = &token_ids[start..end];
        // Decode the token slice back to text.
        let content = bpe.decode(slice.to_vec()).unwrap_or_else(|_| String::new());
        if !content.trim().is_empty() {
            let token_count = slice.len();
            let para_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, content.as_bytes()).to_string();
            chunks.push((
                content,
                ChunkExtras {
                    token_count,
                    // Byte offsets are approximate for hard-split; set to parent range.
                    byte_start,
                    byte_end: byte_start + text.len(),
                    paragraph_ids: vec![para_id],
                },
            ));
        }
        if end == token_ids.len() {
            break;
        }
        start += step;
    }

    chunks
}

/// Extract the last `overlap_budget` tokens from `text` and decode them back
/// to a string for use as the overlap prefix in the next chunk.
fn build_token_overlap(text: &str, overlap_budget: usize, bpe: &tiktoken_rs::CoreBPE) -> String {
    if overlap_budget == 0 {
        return String::new();
    }
    let token_ids = bpe.encode_ordinary(text);
    if token_ids.len() <= overlap_budget {
        return text.to_owned();
    }
    let start = token_ids.len() - overlap_budget;
    bpe.decode(token_ids[start..].to_vec()).unwrap_or_default()
}

// ---- Character-based strategy helpers ----

/// Wrap plain `Vec<String>` chunks from character-based strategies into
/// `(String, ChunkExtras)` pairs, computing byte offsets by searching for
/// each chunk's content in the original text.
fn char_chunks_to_pairs(chunks: Vec<String>, original: &str) -> Vec<(String, ChunkExtras)> {
    let mut search_offset = 0usize;

    chunks
        .into_iter()
        .map(|content| {
            // Try to locate the chunk in the remaining original text.
            let (byte_start, byte_end) =
                if let Some(pos) = original[search_offset..].find(content.trim()) {
                    let abs_start = search_offset + pos;
                    let abs_end = abs_start + content.trim().len();
                    search_offset = abs_start + 1; // advance past this occurrence
                    (abs_start, abs_end)
                } else {
                    (0, content.len())
                };

            let extras = ChunkExtras {
                token_count: 0, // not computed for char-based strategies
                byte_start,
                byte_end,
                paragraph_ids: Vec::new(),
            };
            (content, extras)
        })
        .collect()
}

/// Recursive character split: split by paragraph boundaries first, then
/// sentence boundaries, then word boundaries.
fn recursive_character_split(text: &str, config: &ChunkerConfig) -> Vec<String> {
    let separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "];
    split_recursive(
        text,
        &separators,
        config.max_chunk_size,
        config.chunk_overlap,
    )
}

/// Recursively split text using a hierarchy of separators.
fn split_recursive(
    text: &str,
    separators: &[&str],
    max_size: usize,
    overlap: usize,
) -> Vec<String> {
    if text.len() <= max_size {
        return vec![text.to_owned()];
    }

    if separators.is_empty() {
        return hard_char_split(text, max_size, overlap);
    }

    let sep = separators[0];
    let parts: Vec<&str> = text.split(sep).collect();

    if parts.len() <= 1 {
        return split_recursive(text, &separators[1..], max_size, overlap);
    }

    merge_with_overlap(&parts, sep, max_size, overlap, &separators[1..])
}

/// Merge split parts back together into chunks that respect `max_size`,
/// applying overlap between adjacent chunks.
fn merge_with_overlap(
    parts: &[&str],
    separator: &str,
    max_size: usize,
    overlap: usize,
    remaining_separators: &[&str],
) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for part in parts {
        let candidate = if current.is_empty() {
            part.to_string()
        } else {
            format!("{current}{separator}{part}")
        };

        if candidate.len() <= max_size {
            current = candidate;
        } else if current.is_empty() {
            let sub_chunks = split_recursive(part, remaining_separators, max_size, overlap);
            chunks.extend(sub_chunks);
        } else {
            chunks.push(current.clone());

            let overlap_text = tail_overlap(&current, overlap);
            current = if overlap_text.is_empty() {
                part.to_string()
            } else {
                format!("{overlap_text}{separator}{part}")
            };

            if current.len() > max_size {
                let sub_chunks = split_recursive(&current, remaining_separators, max_size, overlap);
                chunks.extend(sub_chunks);
                current = String::new();
            }
        }
    }

    if !current.trim().is_empty() {
        chunks.push(current);
    }

    chunks
}

/// Section-based split: detect heading lines and split at each heading.
fn section_based_split(text: &str, config: &ChunkerConfig) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sections: Vec<String> = Vec::new();
    let mut current_section = String::new();

    for line in &lines {
        let trimmed = line.trim();
        if is_section_boundary(trimmed) && !current_section.trim().is_empty() {
            sections.push(current_section.clone());
            current_section = String::new();
        }
        if !current_section.is_empty() {
            current_section.push('\n');
        }
        current_section.push_str(line);
    }

    if !current_section.trim().is_empty() {
        sections.push(current_section);
    }

    let mut result: Vec<String> = Vec::new();
    for section in sections {
        if section.len() <= config.max_chunk_size {
            result.push(section);
        } else {
            let sub = recursive_character_split(&section, config);
            result.extend(sub);
        }
    }

    result
}

/// Sliding window split: fixed-size window with configurable overlap.
fn sliding_window_split(text: &str, config: &ChunkerConfig) -> Vec<String> {
    hard_char_split(text, config.max_chunk_size, config.chunk_overlap)
}

/// Hard split at character boundaries, respecting overlap.
fn hard_char_split(text: &str, max_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + max_size).min(total);

        let actual_end = if end < total {
            let lookback = 50.min(end - start);
            let search_start = end - lookback;
            let mut best = end;
            for i in (search_start..end).rev() {
                if chars[i].is_whitespace() {
                    best = i + 1;
                    break;
                }
            }
            best
        } else {
            end
        };

        let chunk: String = chars[start..actual_end].iter().collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }

        let step = actual_end - start;
        if step <= overlap {
            start = actual_end;
        } else {
            start = actual_end - overlap;
        }

        if start >= total {
            break;
        }
    }

    chunks
}

// ---- Heading detection helpers ----

fn is_section_boundary(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.starts_with('#') {
        return true;
    }

    if trimmed.len() >= 4
        && trimmed.chars().filter(|c| c.is_alphabetic()).count() >= 4
        && trimmed
            .chars()
            .all(|c| c.is_uppercase() || !c.is_alphabetic())
    {
        return true;
    }

    is_numbered_heading_line(trimmed)
}

fn is_numbered_heading_line(line: &str) -> bool {
    if line.len() > 200 {
        return false;
    }

    if let Some(first) = line.chars().next() {
        if first.is_ascii_digit() {
            let num_end = line
                .find(|c: char| !c.is_ascii_digit() && c != '.')
                .unwrap_or(line.len());
            if num_end < line.len() {
                let rest = &line[num_end..];
                if rest.starts_with(' ') || rest.starts_with('\t') {
                    return true;
                }
            }
        }
    }

    let roman_end = line
        .find(|c: char| !matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
        .unwrap_or(line.len());
    if roman_end > 0 && roman_end < line.len() {
        let rest = &line[roman_end..];
        if rest.starts_with(". ") || rest.starts_with(".\t") {
            return true;
        }
    }

    false
}

/// Extract the tail of a string as character-based overlap text.
fn tail_overlap(text: &str, max_overlap: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_overlap {
        return text.to_owned();
    }

    let start = chars.len() - max_overlap;
    let mut actual_start = start;
    for (offset, &ch) in chars[start..].iter().enumerate().take(30) {
        if ch.is_whitespace() {
            actual_start = start + offset + 1;
            break;
        }
    }

    chars[actual_start..].iter().collect()
}
