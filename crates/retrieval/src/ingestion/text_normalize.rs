//! Post-process raw text emitted by the Docling sidecar.
//!
//! PDF extraction frequently decomposes diacritics into a *spacing* mark
//! followed by the bare base letter, separated by whitespace — e.g. the
//! input sequence `" ˘ a"` (SPACING BREVE `U+02D8`, a space, then plain
//! `a`) appears where the original document contains a single precomposed
//! `ă` (`U+0103`). Left unchanged these artefacts:
//!
//! - tokenise incorrectly under BM25, splitting what should be one word into
//!   three tokens and surfacing mechanical-instructions chunks as false
//!   positives for definition queries;
//! - corrupt dense embeddings because the tokenizer emits unknown-token
//!   fragments in place of whole words;
//! - make literal-string queries against the stored payload impossible.
//!
//! This module applies a three-step normalisation that is language-agnostic
//! by construction:
//!
//! 1. Replace `<SPACING MARK> <space> <letter>` sequences with
//!    `<letter><COMBINING MARK>`, moving the accent next to the letter and
//!    swapping the spacing codepoint for its combining twin.
//! 2. Apply Unicode NFC composition so combining sequences collapse into a
//!    single precomposed codepoint (`U+0061 U+0306` → `U+0103`).
//! 3. Collapse runs of ASCII spaces left over from step 1.
//!
//! The transformation is idempotent — text that is already clean passes
//! through unchanged (no patterns match, NFC is a no-op on already-composed
//! strings, no runs of spaces to collapse).
//!
//! # Scope
//!
//! Only spacing marks known to appear in Docling output are handled:
//! breve (`U+02D8`), circumflex (`U+02C6`), cedilla (`U+00B8`), caron
//! (`U+02C7`), tilde (`U+02DC`), macron (`U+00AF`), ring (`U+02DA`), and
//! acute/grave (`U+00B4` / `U+0060`). Adding more requires a new row in
//! [`SPACING_MARKS`].
//!
//! No language-specific rules. Any script using combining diacritics
//! benefits equally — Romanian, French, Czech, Polish, Vietnamese, etc.

use std::sync::OnceLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// Mapping from a *spacing* accent codepoint (U+00xx / U+02xx range) to its
/// *combining* equivalent (U+03xx range) so NFC can fold the letter + mark
/// into a precomposed glyph.
struct SpacingMark {
    /// The spacing diacritic as it appears in Docling output.
    spacing: char,
    /// The combining counterpart that composes with the preceding letter.
    combining: char,
}

const SPACING_MARKS: &[SpacingMark] = &[
    SpacingMark {
        spacing: '\u{02D8}',
        combining: '\u{0306}',
    }, // BREVE        → combining (ă, ĕ, …)
    SpacingMark {
        spacing: '\u{02C6}',
        combining: '\u{0302}',
    }, // CIRCUMFLEX   → combining (â, ê, î, ô, û)
    SpacingMark {
        spacing: '\u{00B8}',
        combining: '\u{0327}',
    }, // CEDILLA      → combining (ç, ş, ţ)
    SpacingMark {
        spacing: '\u{02C7}',
        combining: '\u{030C}',
    }, // CARON        → combining (č, š, ž)
    SpacingMark {
        spacing: '\u{02DC}',
        combining: '\u{0303}',
    }, // TILDE        → combining (ã, ñ, õ)
    SpacingMark {
        spacing: '\u{00AF}',
        combining: '\u{0304}',
    }, // MACRON       → combining (ā, ē, ī)
    SpacingMark {
        spacing: '\u{02DA}',
        combining: '\u{030A}',
    }, // RING ABOVE   → combining (å, ů)
    SpacingMark {
        spacing: '\u{00B4}',
        combining: '\u{0301}',
    }, // ACUTE        → combining (á, é, í)
    SpacingMark {
        spacing: '\u{0060}',
        combining: '\u{0300}',
    }, // GRAVE        → combining (à, è, ì)
];

/// Per-spacing-mark regex pair: the mid-word variant runs first to collapse
/// `<letter><space?><spacing><space?><letter>` into a single glued word, and
/// the fallback variant handles remaining cases where the spacing mark sits
/// at a word boundary or follows punctuation.
struct MarkPatterns {
    mark: &'static SpacingMark,
    /// Consumes BOTH the space before and after the spacing mark when both
    /// sides are alphabetic, emitting `$1$2<combining>`. This is the common
    /// Docling failure mode: `histogram <BREVE> a` → `histogramă` with no
    /// stray separator left behind.
    mid_word: Regex,
    /// Catches the residual cases — spacing mark at a start of line, after
    /// punctuation, or before a non-letter. Preserves whatever preceded the
    /// match and only composes the trailing letter.
    fallback: Regex,
}

/// Compile once per `SpacingMark` and cache the [`Regex`] instances.
fn pattern_cache() -> &'static [MarkPatterns] {
    static CACHE: OnceLock<Vec<MarkPatterns>> = OnceLock::new();
    CACHE.get_or_init(|| {
        SPACING_MARKS
            .iter()
            .filter_map(|mark| {
                let escaped = regex::escape(&mark.spacing.to_string());
                // Mid-word: captures the preceding letter ($1) + following
                // letter ($2). Both optional spaces consumed by the match.
                let mid_word_pat = format!(r"(\p{{Alphabetic}})\s*{escaped}\s*(\p{{Alphabetic}})");
                // Fallback: only trailing letter captured as $1.
                let fallback_pat = format!(r"{escaped}\s?(\p{{Alphabetic}})");
                let mid_word = match Regex::new(&mid_word_pat) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!(
                            mark = ?mark.spacing,
                            error = %e,
                            "text_normalize: failed to compile mid-word pattern; skipping"
                        );
                        return None;
                    }
                };
                let fallback = match Regex::new(&fallback_pat) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!(
                            mark = ?mark.spacing,
                            error = %e,
                            "text_normalize: failed to compile fallback pattern; skipping"
                        );
                        return None;
                    }
                };
                Some(MarkPatterns {
                    mark,
                    mid_word,
                    fallback,
                })
            })
            .collect()
    })
}

/// Normalise Docling-emitted text.
///
/// See the module docs for the three-step pipeline. Returns the transformed
/// string; the input is not modified.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::ingestion::text_normalize::normalize_docling_text;
/// assert_eq!(normalize_docling_text("histogram \u{02D8} a"), "histogramă");
/// assert_eq!(
///     normalize_docling_text("Desf \u{02D8} a\u{00B8} surarea"),
///     "Desfășurarea"
/// );
/// assert_eq!(normalize_docling_text("already clean"), "already clean");
/// ```
pub fn normalize_docling_text(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }

    // Step 1a (mid-word): rewrite `<letter><space?><spacing><space?><letter>`
    //                     → `<letter><letter><combining mark>`.
    // Both surrounding spaces are consumed so `"histogram ˘ a"` collapses to
    // `"histogramă"` — otherwise BM25 would still tokenise `histogram` and
    // `ă` as two separate tokens after NFC composition.
    //
    // Step 1b (fallback): rewrite `<spacing><space?><letter>` →
    //                     `<letter><combining mark>` for positions where
    // the spacing mark sits at a word boundary or after punctuation. The
    // mid-word pass ran first and consumed the common-case substrings, so
    // this pass only sees the residual cases.
    let mut buf: String = raw.to_owned();
    for patterns in pattern_cache() {
        let combining = patterns.mark.combining;

        // Mid-word pass — preserves the preceding letter ($1) and glues
        // the trailing letter ($2) to it with the combining mark.
        let mid = patterns
            .mid_word
            .replace_all(&buf, |caps: &regex::Captures<'_>| {
                let prev = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
                let next = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
                let mut out = String::with_capacity(prev.len() + next.len() + combining.len_utf8());
                out.push_str(prev);
                out.push_str(next);
                out.push(combining);
                out
            });
        if let std::borrow::Cow::Owned(new) = mid {
            buf = new;
        }

        // Fallback pass — the preceding char (if any) is not captured; only
        // the trailing letter is composed.
        let fall = patterns
            .fallback
            .replace_all(&buf, |caps: &regex::Captures<'_>| {
                let letter = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
                let mut out = String::with_capacity(letter.len() + combining.len_utf8());
                out.push_str(letter);
                out.push(combining);
                out
            });
        if let std::borrow::Cow::Owned(new) = fall {
            buf = new;
        }
    }

    // Step 2: NFC compose combining-mark sequences into precomposed glyphs.
    let composed: String = buf.nfc().collect();

    // Step 3: map legacy cedilla forms to modern comma-below forms.
    //
    // Romanian orthography (Unicode 3.0+) uses U+0219 / U+021B (comma-below)
    // rather than U+015F / U+0163 (cedilla). Docling's input PDFs often
    // carry the legacy forms; BM25 won't match a user query spelled in
    // modern form against a corpus stored in legacy form.
    //
    // Applied unconditionally because the fold is reversible and symmetric:
    // query text is also normalised via `normalize_search_form`, so both sides
    // of the index always share the same form regardless of the source script.
    let diacritics_applied = fold_cedilla_to_comma_below(&composed);

    // Step 4: collapse runs of ASCII spaces left behind by step 1.
    //
    // We don't touch other whitespace (newlines, tabs, non-breaking spaces);
    // chunk boundaries and line formatting downstream rely on those.
    collapse_ascii_runs(&diacritics_applied)
}

/// Normalise text for indexing and retrieval so that ingested chunks and query
/// text share a canonical search form.
///
/// Wraps [`normalize_docling_text`] (which handles Docling spacing-mark
/// artefacts, NFC composition, and space collapsing) and unconditionally folds
/// legacy cedilla codepoints to their modern comma-below equivalents so that
/// BM25 and dense-embedding lookup are symmetric across both sides of the
/// index.
///
/// Call this function for **both** corpus text (at ingest time) and user query
/// text (before embedding); do **not** call [`normalize_docling_text`] directly
/// for the query side.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::ingestion::text_normalize::normalize_search_form;
///
/// // Comma-below and cedilla variants normalise to the same form.
/// let comma_below = "func\u{021B}ie"; // t with comma-below (U+021B)
/// let cedilla     = "func\u{0163}ie"; // t with cedilla    (U+0163)
/// assert_eq!(normalize_search_form(comma_below), normalize_search_form(cedilla));
/// ```
pub fn normalize_search_form(input: &str) -> String {
    let stage1 = normalize_docling_text(input);
    fold_cedilla_to_comma_below(&stage1)
}

/// Map the four legacy cedilla codepoints used in Romanian/Turkish typography
/// to their modern comma-below counterparts so that corpus and query text
/// share a single form.
///
/// | Input (cedilla)             | Output (comma-below)              |
/// |-----------------------------|-----------------------------------|
/// | U+015E LATIN CAPITAL S WITH CEDILLA | U+0218 LATIN CAPITAL S WITH COMMA BELOW |
/// | U+015F LATIN SMALL S WITH CEDILLA   | U+0219 LATIN SMALL S WITH COMMA BELOW   |
/// | U+0162 LATIN CAPITAL T WITH CEDILLA | U+021A LATIN CAPITAL T WITH COMMA BELOW |
/// | U+0163 LATIN SMALL T WITH CEDILLA   | U+021B LATIN SMALL T WITH COMMA BELOW   |
fn fold_cedilla_to_comma_below(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{015E}' => '\u{0218}', // Ş → Ș
            '\u{015F}' => '\u{0219}', // ş → ș
            '\u{0162}' => '\u{021A}', // Ţ → Ț
            '\u{0163}' => '\u{021B}', // ţ → ț
            other => other,
        })
        .collect()
}

/// Replace every run of two-or-more ASCII spaces with a single space.
/// Non-ASCII whitespace (NBSP, newlines, tabs) is preserved.
fn collapse_ascii_runs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_run = false;
    for ch in s.chars() {
        if ch == ' ' {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            out.push(ch);
            in_run = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_below_and_cedilla_normalize_to_same_form() {
        // Romanian "ț" with comma-below (U+021B) vs legacy "ţ" with cedilla (U+0163).
        let cb = "func\u{021B}ie"; // t with comma-below
        let ce = "func\u{0163}ie"; // t with cedilla
        assert_eq!(normalize_search_form(cb), normalize_search_form(ce));
    }

    #[test]
    fn normalize_search_form_passes_clean_text_unchanged() {
        let clean = "already clean";
        assert_eq!(normalize_search_form(clean), clean);
    }

    #[test]
    fn fold_cedilla_to_comma_below_covers_all_four_codepoints() {
        // ş Ş ţ Ţ  →  ș Ș ț Ț
        let input = "\u{015F}\u{015E}\u{0163}\u{0162}";
        let expected = "\u{0219}\u{0218}\u{021B}\u{021A}";
        assert_eq!(fold_cedilla_to_comma_below(input), expected);
    }

    #[test]
    fn normalize_search_form_wraps_docling_spacing_marks() {
        // Docling artefact: spacing breve + letter should still compose via
        // the inner normalize_docling_text call.
        assert_eq!(normalize_search_form("histogram \u{02D8} a"), "histogramă");
    }
}
