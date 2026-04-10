//! Universal text chunking for document ingestion.
//!
//! Provides three configurable strategies:
//! - **Recursive character split** -- paragraphs, then sentences, then words
//! - **Section-based** -- split at detected headings/section boundaries
//! - **Sliding window** -- fixed-size window with configurable overlap
//!
//! All strategies are domain-agnostic and work on any document type.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::metadata::{self, ChunkMetadata};

// ---- Configuration ----

/// Chunking strategy selector.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    /// Split by paragraphs, then sentences, then words. Respects structural
    /// boundaries while staying within size limits.
    #[default]
    RecursiveCharacter,
    /// Split at heading/section boundaries detected via heuristics.
    SectionBased,
    /// Fixed-size sliding window with overlap.
    SlidingWindow,
}

/// Configuration for the text chunker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Which chunking strategy to use.
    pub strategy: ChunkStrategy,
    /// Maximum number of characters per chunk.
    pub max_chunk_size: usize,
    /// Number of overlapping characters between adjacent chunks.
    pub chunk_overlap: usize,
    /// Minimum chunk size; fragments smaller than this are merged or dropped.
    pub min_chunk_size: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::default(),
            max_chunk_size: 1000,
            chunk_overlap: 200,
            min_chunk_size: 50,
        }
    }
}

// ---- Output types ----

/// A single chunk produced from a document.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentChunk {
    /// Unique identifier for this chunk (UUID v4).
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
}

// ---- Public API ----

/// Chunk a document into [`DocumentChunk`]s using the configured strategy.
///
/// # Arguments
///
/// * `text` -- full document text
/// * `doc_id` -- parent document identifier
/// * `source` -- filename or URL of the source document
/// * `config` -- chunking configuration
///
/// # Returns
///
/// A vector of chunks with metadata, ordered by position. Empty or
/// whitespace-only input returns an empty vector.
#[tracing::instrument(skip(text), fields(doc_id = doc_id, source = source, strategy = ?config.strategy, text_len = text.len()))]
pub fn chunk_document(
    text: &str,
    doc_id: &str,
    source: &str,
    config: &ChunkerConfig,
) -> Vec<DocumentChunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let raw_chunks = match config.strategy {
        ChunkStrategy::RecursiveCharacter => recursive_character_split(text, config),
        ChunkStrategy::SectionBased => section_based_split(text, config),
        ChunkStrategy::SlidingWindow => sliding_window_split(text, config),
    };

    // Filter out chunks that are too small.
    let filtered: Vec<String> = raw_chunks
        .into_iter()
        .filter(|c| c.trim().len() >= config.min_chunk_size)
        .collect();

    let total_chunks = filtered.len();
    let mut headings: Vec<String> = Vec::new();

    filtered
        .into_iter()
        .enumerate()
        .map(|(chunk_index, content)| {
            let meta = metadata::extract_metadata(&content, source, &headings);

            // If this chunk has a section title, add it to the heading hierarchy.
            if let Some(ref title) = meta.section_title {
                // Simple heuristic: if the title matches an existing heading,
                // truncate back to that level. Otherwise, push it.
                if let Some(pos) = headings.iter().position(|h| h == title) {
                    headings.truncate(pos + 1);
                } else {
                    headings.push(title.clone());
                }
            }

            DocumentChunk {
                chunk_id: Uuid::new_v4().to_string(),
                doc_id: doc_id.to_owned(),
                content,
                metadata: meta,
                chunk_index,
                total_chunks,
            }
        })
        .collect()
}

// ---- Strategy implementations ----

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
        // Last resort: hard split at character boundary.
        return hard_split(text, max_size, overlap);
    }

    let sep = separators[0];
    let parts: Vec<&str> = text.split(sep).collect();

    if parts.len() <= 1 {
        // This separator doesn't appear; try the next one.
        return split_recursive(text, &separators[1..], max_size, overlap);
    }

    merge_with_overlap(&parts, sep, max_size, overlap, &separators[1..])
}

/// Merge split parts back together into chunks that respect `max_size`,
/// applying overlap between adjacent chunks. Recursively splits any
/// part that exceeds `max_size` on its own.
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
            // Single part exceeds max_size; recurse with finer separators.
            let sub_chunks = split_recursive(part, remaining_separators, max_size, overlap);
            chunks.extend(sub_chunks);
        } else {
            chunks.push(current.clone());

            // Build overlap: take the tail of the current chunk.
            let overlap_text = tail_overlap(&current, overlap);
            current = if overlap_text.is_empty() {
                part.to_string()
            } else {
                format!("{overlap_text}{separator}{part}")
            };

            // If the new current already exceeds max, flush and recurse.
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

    // Sub-split any section that exceeds max_chunk_size.
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
    hard_split(text, config.max_chunk_size, config.chunk_overlap)
}

/// Hard split at character boundaries, respecting overlap.
///
/// Attempts to split at word boundaries within a small lookahead window
/// to avoid cutting words in half.
fn hard_split(text: &str, max_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + max_size).min(total);

        // Try to find a word boundary (space) near the end.
        let actual_end = if end < total {
            // Look back up to 50 chars for a space.
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

        // Advance with overlap.
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

/// Check whether a line is a section boundary (heading).
fn is_section_boundary(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Markdown headers
    if trimmed.starts_with('#') {
        return true;
    }

    // All-caps line (at least 4 alphabetic chars)
    if trimmed.len() >= 4
        && trimmed.chars().filter(|c| c.is_alphabetic()).count() >= 4
        && trimmed
            .chars()
            .all(|c| c.is_uppercase() || !c.is_alphabetic())
    {
        return true;
    }

    // Numbered heading patterns
    is_numbered_heading_line(trimmed)
}

/// Check whether a line starts with a numbered heading prefix.
fn is_numbered_heading_line(line: &str) -> bool {
    // Must be reasonably short to be a heading.
    if line.len() > 200 {
        return false;
    }

    // "1. Title", "1.1 Title", "1.1.1 Title"
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

    // Roman numeral headings: "I. ", "II. ", "IV. "
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

/// Extract the tail of a string as overlap text.
///
/// Returns the last `max_overlap` characters, trying to start at a word boundary.
fn tail_overlap(text: &str, max_overlap: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_overlap {
        return text.to_owned();
    }

    let start = chars.len() - max_overlap;
    // Try to find a word boundary near the start of the overlap.
    let mut actual_start = start;
    for (offset, &ch) in chars[start..].iter().enumerate().take(30) {
        if ch.is_whitespace() {
            actual_start = start + offset + 1;
            break;
        }
    }

    chars[actual_start..].iter().collect()
}
