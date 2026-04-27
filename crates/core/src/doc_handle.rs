//! Canonical document handle + resolver for tool-call identifiers.
//!
//! The knowledge base identifies a document by its **MinIO object key**
//! (e.g. `users/<user-id>/docs/<uuid>.pdf`). That key is the value stored
//! on every chunk payload and the single ground truth the retrieval router
//! filters on.
//!
//! LLM-facing tools, however, must accept a range of user- or model-supplied
//! inputs and resolve them back to the canonical key:
//!
//! - the full key as emitted by `outline_corpus` / `document_info`;
//! - the UUID substring embedded in the key (LLMs tend to tokenise paths
//!   and copy only the bare UUID);
//! - the trailing filename (`08-cartePAI-regiuni.pdf`);
//! - any other fragment that uniquely identifies one pinned document.
//!
//! [`DocHandle::resolve`] is the single entry point. It returns the canonical
//! key on success or a structured error that the caller can surface back to
//! the LLM so it retries with a more specific value — turning a silent
//! "zero results" failure into a self-correcting loop.
//!
//! # Examples
//!
//! ```
//! use agent_rs_core::doc_handle::DocHandle;
//!
//! let known = vec![
//!     "users/u1/docs/567f3d77-5eb4-4865-ae70-392fb6be735e.pdf".to_string(),
//!     "users/u1/docs/11111111-2222-3333-4444-555555555555.pdf".to_string(),
//! ];
//!
//! // Exact key: trivially resolves.
//! let h = DocHandle::resolve(&known[0], &known).unwrap();
//! assert_eq!(h.as_str(), known[0]);
//!
//! // Bare UUID substring: resolves uniquely to the one key that contains it.
//! let h = DocHandle::resolve("567f3d77-5eb4-4865-ae70-392fb6be735e", &known).unwrap();
//! assert_eq!(h.as_str(), known[0]);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Canonical document handle: the MinIO object key that every chunk payload
/// stores under `minio_object_key`.
///
/// Construct via [`DocHandle::resolve`]. The inner string is intentionally
/// private — the only way to get a `DocHandle` is through resolution, which
/// guarantees the value is one of the caller-supplied known keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocHandle(String);

impl DocHandle {
    /// The canonical MinIO object key.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the handle and return the owned canonical key.
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Resolve an arbitrary identifier `input` against `known_keys`.
    ///
    /// Resolution rules, evaluated in order:
    ///
    /// 1. **Exact match** — `input` equals a known key.
    /// 2. **Unique case-insensitive substring** — exactly one known key
    ///    contains `input` as a case-folded substring.
    /// 3. **Ambiguous substring** — two or more known keys match →
    ///    [`DocHandleError::Ambiguous`] with the candidate list.
    /// 4. **No match** → [`DocHandleError::Unknown`] with the full known list.
    ///
    /// An empty / whitespace-only input always yields [`DocHandleError::Empty`]
    /// so callers catch "LLM sent `\"\"`" before it reaches the exact-match
    /// path (which would spuriously fail to match any non-empty key).
    ///
    /// # Errors
    ///
    /// Returns [`DocHandleError::Empty`] when `input` trims to empty,
    /// [`DocHandleError::Unknown`] when no known key matches, and
    /// [`DocHandleError::Ambiguous`] when multiple match.
    pub fn resolve(input: &str, known_keys: &[String]) -> Result<Self, DocHandleError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(DocHandleError::Empty);
        }

        if let Some(exact) = known_keys.iter().find(|k| k.as_str() == trimmed) {
            return Ok(Self(exact.clone()));
        }

        let needle = trimmed.to_lowercase();
        let mut matches: Vec<&String> = known_keys
            .iter()
            .filter(|k| k.to_lowercase().contains(&needle))
            .collect();

        match matches.len() {
            0 => Err(DocHandleError::Unknown {
                input: input.to_owned(),
                known: known_keys.to_vec(),
            }),
            1 => Ok(Self(matches.remove(0).clone())),
            _ => Err(DocHandleError::Ambiguous {
                input: input.to_owned(),
                candidates: matches.into_iter().cloned().collect(),
            }),
        }
    }
}

impl std::fmt::Display for DocHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Structured errors from [`DocHandle::resolve`].
///
/// The `Display` impl is deliberately LLM-friendly: it ends with a list of
/// the known keys (or the ambiguous candidates) so the model can see what
/// it should have passed and retry on the next turn without human help.
#[derive(Debug, Clone, Error)]
pub enum DocHandleError {
    /// Caller supplied an empty or whitespace-only identifier.
    #[error("doc identifier is empty; omit the field entirely to search all pinned documents")]
    Empty,

    /// No known key matches the input (not even as a substring).
    ///
    /// The `known` list is included so the caller — typically an LLM tool
    /// handler — can echo it back and retry with a valid value.
    #[error(
        "no pinned document matches identifier {input:?}; known handles: {known:?}. \
         Copy one of the known handles verbatim, or omit the field to search all of them."
    )]
    Unknown {
        /// The raw input the caller supplied.
        input: String,
        /// The full list of known pinned-document keys at resolution time.
        known: Vec<String>,
    },

    /// More than one known key matches the input as a substring.
    ///
    /// The `candidates` list is included so the caller can disambiguate by
    /// passing a more specific fragment (or the full key).
    #[error(
        "identifier {input:?} matches multiple pinned documents: {candidates:?}. \
         Pass a more specific fragment or the full key of the one you want."
    )]
    Ambiguous {
        /// The raw input the caller supplied.
        input: String,
        /// The known keys that matched `input` as a substring.
        candidates: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_keys() -> Vec<String> {
        vec![
            "users/u1/docs/567f3d77-5eb4-4865-ae70-392fb6be735e.pdf".to_string(),
            "users/u1/docs/11111111-2222-3333-4444-555555555555.pdf".to_string(),
            "users/u2/docs/99999999-aaaa-bbbb-cccc-dddddddddddd.pdf".to_string(),
        ]
    }

    #[test]
    fn resolves_exact_full_key() {
        let known = sample_keys();
        let h = DocHandle::resolve(&known[0], &known).expect("exact match should succeed");
        assert_eq!(h.as_str(), known[0]);
    }

    #[test]
    fn resolves_bare_uuid_substring() {
        let known = sample_keys();
        let h = DocHandle::resolve("567f3d77-5eb4-4865-ae70-392fb6be735e", &known)
            .expect("unique uuid substring must resolve");
        assert_eq!(h.as_str(), known[0]);
    }

    #[test]
    fn resolves_filename_only() {
        let mut known = sample_keys();
        known.push("users/u1/docs/08-cartePAI-regiuni.pdf".into());
        let h = DocHandle::resolve("08-cartePAI-regiuni.pdf", &known)
            .expect("unique filename must resolve");
        assert_eq!(h.as_str(), "users/u1/docs/08-cartePAI-regiuni.pdf");
    }

    #[test]
    fn resolves_case_insensitively() {
        let known = sample_keys();
        let h = DocHandle::resolve("567F3D77-5EB4-4865-AE70-392FB6BE735E", &known)
            .expect("case should not matter for substring match");
        assert_eq!(h.as_str(), known[0]);
    }

    #[test]
    fn rejects_empty_input() {
        let known = sample_keys();
        let err = DocHandle::resolve("", &known).expect_err("empty input must fail");
        assert!(matches!(err, DocHandleError::Empty));
    }

    #[test]
    fn rejects_whitespace_only_input() {
        let known = sample_keys();
        let err = DocHandle::resolve("   \t\n ", &known).expect_err("whitespace must fail");
        assert!(matches!(err, DocHandleError::Empty));
    }

    #[test]
    fn unknown_input_returns_known_list() {
        let known = sample_keys();
        let err = DocHandle::resolve("deadbeef-0000-0000-0000-000000000000", &known)
            .expect_err("nonexistent uuid must fail");
        match err {
            DocHandleError::Unknown {
                input,
                known: echoed,
            } => {
                assert_eq!(input, "deadbeef-0000-0000-0000-000000000000");
                assert_eq!(echoed, known);
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_substring_lists_candidates() {
        let known = vec![
            "users/u1/docs/aaa.pdf".to_string(),
            "users/u2/docs/aaa.pdf".to_string(),
        ];
        let err = DocHandle::resolve("aaa.pdf", &known).expect_err("ambiguous must fail");
        match err {
            DocHandleError::Ambiguous { input, candidates } => {
                assert_eq!(input, "aaa.pdf");
                assert_eq!(candidates.len(), 2);
                assert!(candidates.iter().all(|c| c.ends_with("aaa.pdf")));
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn exact_match_wins_over_substring() {
        // If one of the known keys IS the input verbatim, the exact-match
        // branch must short-circuit even when other keys contain the input
        // as a substring. This guards against a caller passing the canonical
        // key for a prefix and accidentally matching a longer key.
        let known = vec!["abc".to_string(), "abcdef".to_string(), "xabcx".to_string()];
        let h = DocHandle::resolve("abc", &known).expect("exact match must short-circuit");
        assert_eq!(h.as_str(), "abc");
    }

    #[test]
    fn unknown_on_empty_known_list_echoes_empty_list() {
        let err = DocHandle::resolve("anything", &[]).expect_err("empty known list");
        match err {
            DocHandleError::Unknown { known, .. } => assert!(known.is_empty()),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn display_renders_canonical_key() {
        let known = sample_keys();
        let h = DocHandle::resolve(&known[0], &known).unwrap();
        assert_eq!(format!("{h}"), known[0]);
    }

    #[test]
    fn into_inner_yields_owned_key() {
        let known = sample_keys();
        let h = DocHandle::resolve(&known[1], &known).unwrap();
        assert_eq!(h.into_inner(), known[1]);
    }

    #[test]
    fn trims_input_whitespace_before_matching() {
        let known = sample_keys();
        let h = DocHandle::resolve("  567f3d77-5eb4-4865-ae70-392fb6be735e  ", &known)
            .expect("leading/trailing whitespace must be trimmed");
        assert_eq!(h.as_str(), known[0]);
    }
}
