//! Automatic metadata extraction from text content.
//!
//! Detects language, tables, lists, section titles, and computes
//! word/character counts. Works on any document type -- no domain-specific
//! logic.

use serde::{Deserialize, Serialize};

/// Rich metadata extracted from a single chunk of text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source filename or URL.
    pub source: String,
    /// Page number (if known from source document).
    pub page_number: Option<usize>,
    /// First heading-like line detected in the chunk.
    pub section_title: Option<String>,
    /// Hierarchy of headings above this chunk.
    pub headings: Vec<String>,
    /// Whether the chunk contains table-like content.
    pub has_table: bool,
    /// Whether the chunk contains list items.
    pub has_list: bool,
    /// Detected language code (ISO 639-1) or `None` when uncertain.
    pub language: Option<String>,
    /// Total character count (Unicode scalar values).
    pub char_count: usize,
    /// Total word count (whitespace-delimited).
    pub word_count: usize,
    /// Exact token count using the cl100k_base tokeniser (text-embedding-3-large).
    ///
    /// Set to `0` when tokenisation is unavailable (e.g. non-token-aware strategies).
    pub token_count: usize,
    /// Byte offset of the first byte of this chunk within the original document text.
    pub byte_start: usize,
    /// Byte offset one past the last byte of this chunk within the original document text.
    pub byte_end: usize,
    /// Deterministic UUIDs (v5, OID namespace, content-keyed) for each paragraph
    /// that was grouped into this chunk.
    pub paragraph_ids: Vec<String>,
}

/// Extra positional + token information supplied by token-aware chunking strategies.
///
/// Callers that do not have this information available should pass `Default::default()`.
#[derive(Debug, Default, Clone)]
pub struct ChunkExtras {
    /// Exact token count (cl100k_base).  `0` when not applicable.
    pub token_count: usize,
    /// Byte start offset in the original document.
    pub byte_start: usize,
    /// Byte end offset (exclusive) in the original document.
    pub byte_end: usize,
    /// Deterministic paragraph UUIDs contained in this chunk.
    pub paragraph_ids: Vec<String>,
}

/// Extract metadata from a chunk of text.
///
/// Performs lightweight heuristic analysis -- no external calls required.
/// The `source` parameter is the filename or URL of the parent document.
/// `headings` is the heading hierarchy accumulated from previous chunks.
/// `extras` carries token/byte/paragraph information from token-aware strategies;
/// pass [`ChunkExtras::default()`] when that information is unavailable.
#[tracing::instrument(skip(text, headings, extras), fields(text_len = text.len()))]
pub fn extract_metadata(
    text: &str,
    source: &str,
    headings: &[String],
    extras: ChunkExtras,
) -> ChunkMetadata {
    let section_title = detect_section_title(text);
    let has_table = detect_table(text);
    let has_list = detect_list(text);
    let language = detect_language(text);
    let char_count = text.chars().count();
    let word_count = text.split_whitespace().count();

    ChunkMetadata {
        source: source.to_owned(),
        page_number: None,
        section_title,
        headings: headings.to_vec(),
        has_table,
        has_list,
        language,
        char_count,
        word_count,
        token_count: extras.token_count,
        byte_start: extras.byte_start,
        byte_end: extras.byte_end,
        paragraph_ids: extras.paragraph_ids,
    }
}

/// Detect the first heading-like line in the text.
///
/// Recognises:
/// - Markdown headers (`# Title`, `## Subtitle`)
/// - Numbered sections (`1.`, `1.1`, `I.`, `A.`)
/// - All-caps lines (common in legal/business documents)
/// - Underline headers (lines followed by `===` or `---`)
fn detect_section_title(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Markdown headers: # Title, ## Subtitle, etc.
        if trimmed.starts_with('#') {
            let title = trimmed.trim_start_matches('#').trim();
            if !title.is_empty() {
                return Some(title.to_owned());
            }
        }

        // Underline headers: line followed by === or ---
        if i + 1 < lines.len() {
            let next = lines[i + 1].trim();
            if (next.len() >= 3 && next.chars().all(|c| c == '='))
                || (next.len() >= 3 && next.chars().all(|c| c == '-'))
            {
                return Some(trimmed.to_owned());
            }
        }

        // All-caps line (at least 4 chars, all uppercase letters/spaces/punctuation)
        if trimmed.len() >= 4
            && trimmed.chars().filter(|c| c.is_alphabetic()).count() >= 4
            && trimmed
                .chars()
                .all(|c| c.is_uppercase() || !c.is_alphabetic())
        {
            return Some(trimmed.to_owned());
        }

        // Numbered section: "1.", "1.1", "I.", "A.", etc.
        if is_numbered_heading(trimmed) {
            return Some(trimmed.to_owned());
        }
    }

    None
}

/// Check whether a line looks like a numbered section heading.
///
/// Matches patterns like:
/// - `1. Introduction`
/// - `1.1 Background`
/// - `I. Overview`
/// - `A. Definitions`
/// - `(a) Definitions`
fn is_numbered_heading(line: &str) -> bool {
    let trimmed = line.trim();

    // Must have some alphabetic content after the number/prefix
    // and be reasonably short (headings are not paragraphs).
    if trimmed.len() > 200 {
        return false;
    }

    // "1. ", "1.1 ", "1.1.1 "
    if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit()) {
        let after_digits = rest.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        if after_digits.starts_with(' ') || after_digits.starts_with('\t') {
            return true;
        }
    }

    // "I. ", "II. ", "IV. " (Roman numerals)
    let roman_end = trimmed
        .find(|c: char| !matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
        .unwrap_or(trimmed.len());
    if roman_end > 0 && roman_end < trimmed.len() {
        let rest = &trimmed[roman_end..];
        if rest.starts_with(". ") || rest.starts_with(".\t") {
            return true;
        }
    }

    // "A. ", "B. " (single uppercase letter)
    if trimmed.len() >= 3 {
        let first = trimmed.as_bytes()[0];
        if first.is_ascii_uppercase()
            && trimmed.as_bytes()[1] == b'.'
            && trimmed.as_bytes()[2] == b' '
        {
            return true;
        }
    }

    // "(a) ", "(1) " (parenthesised)
    if trimmed.starts_with('(') {
        if let Some(close) = trimmed.find(')') {
            if close < 5 && close + 1 < trimmed.len() && trimmed.as_bytes()[close + 1] == b' ' {
                return true;
            }
        }
    }

    false
}

/// Detect whether the text contains table-like content.
///
/// Looks for:
/// - Pipe-delimited tables (`|` at least 2 occurrences on a line)
/// - Tab-separated lines with 3+ columns
fn detect_table(text: &str) -> bool {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Pipe-delimited table row: at least 2 pipe characters
        if trimmed.matches('|').count() >= 2 {
            return true;
        }

        // Tab-separated with 3+ columns
        if trimmed.matches('\t').count() >= 2 {
            return true;
        }
    }
    false
}

/// Detect whether the text contains list items.
///
/// Matches lines starting with: `- `, `* `, `1. `, `a) `, etc.
fn detect_list(text: &str) -> bool {
    for line in text.lines() {
        let trimmed = line.trim();

        // Bullet lists
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ ") {
            return true;
        }

        // Numbered lists: "1. ", "2. ", etc.
        if let Some(dot_pos) = trimmed.find(". ") {
            if dot_pos <= 3 && trimmed[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }

        // Letter lists: "a) ", "b) ", etc.
        if trimmed.len() >= 3
            && trimmed.as_bytes()[0].is_ascii_alphabetic()
            && trimmed.as_bytes()[1] == b')'
            && trimmed.as_bytes()[2] == b' '
        {
            return true;
        }
    }
    false
}

/// Detect the language of the text using simple character-frequency heuristics.
///
/// Returns an ISO 639-1 code or `None` when confidence is too low.
///
/// Detects:
/// - Romanian: ă, â, î, ș, ț
/// - French: heavy accent usage (e, a, c cedilla)
/// - German: umlauts (ae, oe, ue, ss)
/// - Spanish: n-tilde, inverted punctuation
/// - Default: "en" when enough ASCII alphabetic content is present
fn detect_language(text: &str) -> Option<String> {
    let total_alpha: usize = text.chars().filter(|c| c.is_alphabetic()).count();
    if total_alpha < 20 {
        return None;
    }

    // Romanian diacritics
    let ro_chars = text
        .chars()
        .filter(|c| matches!(c, 'ă' | 'â' | 'î' | 'ș' | 'ț' | 'Ă' | 'Â' | 'Î' | 'Ș' | 'Ț'))
        .count();
    if ro_chars > 0 && (ro_chars as f64 / total_alpha as f64) > 0.005 {
        return Some("ro".to_owned());
    }

    // German umlauts
    let de_chars = text
        .chars()
        .filter(|c| matches!(c, 'ä' | 'ö' | 'ü' | 'ß' | 'Ä' | 'Ö' | 'Ü'))
        .count();
    if de_chars > 0 && (de_chars as f64 / total_alpha as f64) > 0.005 {
        return Some("de".to_owned());
    }

    // Spanish
    let es_chars = text
        .chars()
        .filter(|c| matches!(c, 'ñ' | 'Ñ' | '¿' | '¡'))
        .count();
    if es_chars > 0 && (es_chars as f64 / total_alpha as f64) > 0.002 {
        return Some("es".to_owned());
    }

    // French -- check for frequent accented characters
    let fr_chars = text
        .chars()
        .filter(|c| {
            matches!(
                c,
                'é' | 'è'
                    | 'ê'
                    | 'ë'
                    | 'à'
                    | 'â'
                    | 'ù'
                    | 'û'
                    | 'ç'
                    | 'É'
                    | 'È'
                    | 'Ê'
                    | 'Ë'
                    | 'À'
                    | 'Â'
                    | 'Ù'
                    | 'Û'
                    | 'Ç'
                    | 'œ'
                    | 'Œ'
            )
        })
        .count();
    if fr_chars > 0 && (fr_chars as f64 / total_alpha as f64) > 0.01 {
        return Some("fr".to_owned());
    }

    // Default to English if the text is mostly ASCII
    let ascii_alpha = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if (ascii_alpha as f64 / total_alpha as f64) > 0.90 {
        return Some("en".to_owned());
    }

    None
}
