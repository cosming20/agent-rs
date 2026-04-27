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
    /// Whether the chunk contains a formula or equation-like fragment.
    pub has_formula: bool,
    /// Whether the chunk contains a source-code fragment.
    pub has_code: bool,
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
    let has_formula = detect_formula(text);
    let has_code = detect_code(text);
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
        has_formula,
        has_code,
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

/// Detect whether the text contains a formula or equation-like fragment.
///
/// Works cross-document / cross-lingual — detection is based on mathematical
/// glyphs and structural markers that are stable across languages, not on
/// vocabulary. Matches:
/// - inline or block LaTeX delimiters (`$…$`, `\[…\]`, `\(…\)`,
///   `\begin{equation}`, `\begin{align}`);
/// - Docling's sentinel comment `<!-- formula-not-decoded -->`;
/// - a run of at least two common math operators / relation glyphs
///   (`= + − − × ÷ ≤ ≥ ≠ ≈ ∑ ∫ ∏ ∞ √ π α β γ δ θ λ μ σ ω`) within ~200 chars,
///   which catches rendered equations that survived PDF extraction without
///   the surrounding LaTeX.
pub fn detect_formula(text: &str) -> bool {
    // Explicit LaTeX / Docling markers — high precision.
    const LATEX_MARKERS: &[&str] = &[
        "\\begin{equation",
        "\\begin{align",
        "\\begin{gather",
        "\\[",
        "\\(",
        "<!-- formula-not-decoded -->",
    ];
    for m in LATEX_MARKERS {
        if text.contains(m) {
            return true;
        }
    }

    // Paired `$…$` inline math. Require the closing `$` to be reasonably
    // close so we don't false-positive on currency amounts sprinkled
    // through prose.
    if let Some(open) = text.find('$') {
        let rest = &text[open + 1..];
        if let Some(close) = rest.find('$') {
            if close > 0 && close < 200 {
                return true;
            }
        }
    }

    // Glyph-density heuristic for rendered PDFs without LaTeX source.
    const MATH_GLYPHS: &str = "=+−×÷≤≥≠≈∑∫∏∞√πθλμσω≡⇒∀∃∈∉⊂⊃∪∩";
    let hits = text.chars().filter(|c| MATH_GLYPHS.contains(*c)).count();
    if hits >= 2 {
        // Further constrain to a window — two `=` in a 2 000-char chunk
        // would misfire on tables of key:value pairs. Two glyphs inside
        // any 200-char window is a much stronger signal.
        let bytes = text.as_bytes();
        let window = 400usize.min(bytes.len());
        if bytes.len() <= window {
            return true;
        }
        for start in (0..bytes.len().saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(bytes.len());
            // Byte slicing is safe for ASCII counting; we only care
            // about glyph density so char-decoding the window is OK.
            if let Ok(slice) = std::str::from_utf8(&bytes[start..end]) {
                let local: usize = slice.chars().filter(|c| MATH_GLYPHS.contains(*c)).count();
                if local >= 2 {
                    return true;
                }
            }
        }
    }

    false
}

/// Detect whether the text contains a source-code fragment.
///
/// Matches:
/// - triple-backtick fenced code blocks (`` ``` ``);
/// - inline backtick spans of at least 3 chars when multiple appear;
/// - indented-4-space blocks (Markdown code style) of at least 2 lines;
/// - common single-line code tokens (`fn `, `def `, `class `, `import `,
///   `function `, `return `, `console.log`, `System.out`, `#include`,
///   `pub fn`, `-> `, `=> `, `::`, `}\n`).
///
/// Deliberately language-agnostic: the signals are structural (fences,
/// indentation), not keyword-based. That keeps the detector useful for
/// documents written in any natural language that happen to include a
/// code snippet.
pub fn detect_code(text: &str) -> bool {
    // Fenced code blocks — highest precision.
    if text.contains("```") {
        return true;
    }

    // Indented-4-space block (Markdown code style): at least two
    // consecutive lines starting with four spaces or a tab.
    let mut run = 0usize;
    for line in text.lines() {
        if line.starts_with("    ") || line.starts_with('\t') {
            // Skip empty-but-indented lines.
            if line.trim().is_empty() {
                continue;
            }
            run += 1;
            if run >= 2 {
                return true;
            }
        } else if !line.trim().is_empty() {
            run = 0;
        }
    }

    // Structural single-line tokens that rarely appear in prose but are
    // ubiquitous in code. Matched as substrings to tolerate surrounding
    // punctuation.
    const CODE_TOKENS: &[&str] = &[
        "```",
        "pub fn ",
        "fn main",
        "def ",
        "class ",
        "import ",
        "function ",
        "console.log",
        "System.out.",
        "#include <",
        "#include \"",
        " -> ",
        " => {",
        "::",
    ];
    for tok in CODE_TOKENS {
        if text.contains(tok) {
            return true;
        }
    }

    // Multiple inline backtick spans (`foo` and `bar` in the same chunk)
    // strongly suggest code-referring prose.
    let backtick_count = text.matches('`').count();
    if backtick_count >= 4 {
        return true;
    }

    false
}

/// Minimum number of alphabetic characters required before a language
/// heuristic runs. Below this, character-frequency estimates are too
/// noisy to be trustworthy and we return `None` (language unknown).
const MIN_ALPHA_CHARS_FOR_DETECTION: usize = 20;

/// Minimum diacritic ratio (language-specific chars / total alphabetic
/// chars) for a language to be a candidate. Single threshold across
/// every non-English language so no language gets preferential treatment
/// from a looser bar — a mixed-diacritic text now picks the language
/// with the **highest** above-threshold ratio, not the first one in
/// source order.
const DIACRITIC_DETECTION_THRESHOLD: f64 = 0.005;

/// ASCII-alphabetic fraction above which a low-diacritic text is
/// classified as English.
const ENGLISH_ASCII_FRACTION: f64 = 0.90;

/// Detect the language of the text using simple character-frequency
/// heuristics.
///
/// Returns an ISO 639-1 code or `None` when confidence is too low.
///
/// Scores every supported language and returns the one with the
/// highest diacritic ratio above [`DIACRITIC_DETECTION_THRESHOLD`]. Ties
/// are broken by an alphabetical order of ISO codes so the function is
/// deterministic. This replaces the prior first-match-wins ordering
/// which had a subtle per-language-threshold bias — a document
/// containing one Romanian-looking character could win over a French
/// text purely because Romanian was checked first with a looser
/// threshold.
///
/// Supported languages (add new entries to the table below):
/// - Romanian: ă, â, î, ș, ț
/// - German:   ä, ö, ü, ß
/// - Spanish:  ñ, ¿, ¡
/// - French:   é è ê ë à â ù û ç œ (upper + lower)
///
/// When no diacritic-bearing language crosses the threshold, returns
/// `"en"` if the text is ≥ [`ENGLISH_ASCII_FRACTION`] ASCII-alphabetic;
/// otherwise `None` (no confident guess).
fn detect_language(text: &str) -> Option<String> {
    let total_alpha: usize = text.chars().filter(|c| c.is_alphabetic()).count();
    if total_alpha < MIN_ALPHA_CHARS_FOR_DETECTION {
        return None;
    }

    // Score every language; collect (code, ratio) pairs that exceed
    // the common threshold. A language counted once per character hit
    // — characters shared between languages (`â` in ro/fr) will score
    // for both, and whichever has the higher total ratio wins.
    let total_alpha_f = total_alpha as f64;
    let score = |predicate: fn(char) -> bool| -> f64 {
        text.chars().filter(|&c| predicate(c)).count() as f64 / total_alpha_f
    };

    let candidates: [(&str, f64); 4] = [
        (
            "ro",
            score(|c| matches!(c, 'ă' | 'â' | 'î' | 'ș' | 'ț' | 'Ă' | 'Â' | 'Î' | 'Ș' | 'Ț')),
        ),
        (
            "de",
            score(|c| matches!(c, 'ä' | 'ö' | 'ü' | 'ß' | 'Ä' | 'Ö' | 'Ü')),
        ),
        ("es", score(|c| matches!(c, 'ñ' | 'Ñ' | '¿' | '¡'))),
        (
            "fr",
            score(|c| {
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
            }),
        ),
    ];

    let best = candidates
        .iter()
        .filter(|(_, ratio)| *ratio > DIACRITIC_DETECTION_THRESHOLD)
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Break ties on ISO-code order for determinism.
                .then_with(|| b.0.cmp(a.0))
        });

    if let Some((code, _)) = best {
        return Some((*code).to_owned());
    }

    // Fallback: mostly-ASCII text gets tagged `"en"`.
    let ascii_alpha = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if (ascii_alpha as f64 / total_alpha_f) > ENGLISH_ASCII_FRACTION {
        return Some("en".to_owned());
    }

    None
}
