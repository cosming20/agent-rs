//! Query language detection.
//!
//! Wraps [`whatlang::detect`] and maps its [`whatlang::Lang`] result onto
//! the typed [`LangCode`] enum exposed by the core crate. The agent
//! pipeline calls [`detect_lang`] once per request (see
//! [`crate::orchestrator::AgentOrchestrator::process_query`]) and stores
//! the result on [`crate::context::QueryContext`] for downstream use
//! (prompt routing, lang-aware tokenization, telemetry).
//!
//! Only the seven languages defined on [`LangCode`] are mapped; every
//! other detected language collapses to [`LangCode::Other`]. This keeps
//! the supported surface tight and predictable — adding a new language
//! is one match arm here plus one variant on [`LangCode`].
//!
//! `whatlang::detect` returns `Option<Info>` and is unreliable on very
//! short or empty inputs; both yield [`LangCode::Other`] so callers can
//! treat "unknown" as the safe default rather than guessing.

use agent_rs_core::types::LangCode;

/// Detect the language of a free-form query string.
///
/// Returns [`LangCode::Other`] when the detector is uncertain (empty or
/// very short input, no script signal) or when the detected language is
/// outside the supported set on [`LangCode`].
///
/// # Examples
///
/// ```
/// use agent_rs_agent::lang::detect_lang;
/// use agent_rs_core::types::LangCode;
///
/// assert_eq!(
///     detect_lang("What is the capital of France?"),
///     LangCode::En
/// );
/// assert_eq!(detect_lang(""), LangCode::Other);
/// ```
pub fn detect_lang(query: &str) -> LangCode {
    if query.trim().is_empty() {
        return LangCode::Other;
    }
    match whatlang::detect_lang(query) {
        Some(whatlang::Lang::Eng) => LangCode::En,
        Some(whatlang::Lang::Ron) => LangCode::Ro,
        Some(whatlang::Lang::Spa) => LangCode::Es,
        Some(whatlang::Lang::Deu) => LangCode::De,
        Some(whatlang::Lang::Fra) => LangCode::Fr,
        Some(whatlang::Lang::Ita) => LangCode::It,
        Some(whatlang::Lang::Por) => LangCode::Pt,
        Some(_) | None => LangCode::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_returns_other() {
        assert_eq!(detect_lang(""), LangCode::Other);
        assert_eq!(detect_lang("   "), LangCode::Other);
    }

    #[test]
    fn obvious_english_detected() {
        assert_eq!(
            detect_lang("The quick brown fox jumps over the lazy dog repeatedly."),
            LangCode::En
        );
    }

    #[test]
    fn obvious_romanian_detected() {
        // Whatlang needs enough text to be confident; this sentence is well
        // above its minimum-length threshold.
        let q = "Aceasta este o propoziție în limba română despre câinele brun rapid.";
        assert_eq!(detect_lang(q), LangCode::Ro);
    }

    #[test]
    fn unsupported_language_collapses_to_other() {
        // Japanese — outside our seven-language allowlist.
        assert_eq!(detect_lang("これは日本語の文章です。"), LangCode::Other);
    }
}
