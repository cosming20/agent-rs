//! Deterministic subject-presence pre-filter for the synthesis pipeline.
//!
//! Cheap, language-agnostic check that runs **before** the synthesis LLM
//! call. When the agent's rewritten retrieval queries share zero content
//! tokens with the fused evidence pack, the topic is out-of-corpus and
//! synthesising an answer would either fabricate from general knowledge
//! or repeat the model's "not in evidence" hedge after wasting an LLM
//! call. Short-circuiting here saves ~40 s on out-of-corpus queries
//! while preserving the agent's already-honest stance.
//!
//! Migrated from the deleted `steps::grounding_gate` module — these
//! routines never relied on the LLM verifier and are kept verbatim
//! (modulo visibility / naming) so the pre-existing behaviour is
//! preserved bit-for-bit.

use std::cmp::Reverse;

use agent_rs_stores::types::SearchResult;

use crate::coverage::ToolCallRecord;

/// Extract the rewritten subject phrase from a tool-call's args JSON.
///
/// All retrieval tools that carry semantic content pass the phrase
/// under the `query` key (`search_knowledge`, `search_sections`). The
/// `fetch_section_chunks` tool uses a `section_id` key that encodes a
/// structural identifier rather than a semantic phrase, so those calls
/// are skipped (returning `None`).
pub fn extract_subject_phrase(args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    v.get("query")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string())
}

/// Collect all `≥min_len`-char alphabetic tokens from a query as
/// subject noun candidates.
///
/// Returns tokens sorted longest-first so the most specific term is
/// checked first when iterating. A query like
/// `"segmentare metode algoritmice"` with `min_len = 4` yields
/// `["algoritmice", "segmentare", "metode"]`.
///
/// `min_len` is sourced from
/// [`agent_rs_core::config::SubjectCheckConfig::subject_min_token_len`]
/// so the threshold can be tuned per deployment without a recompile.
pub fn subject_nouns(q: &str, min_len: usize) -> Vec<String> {
    let mut tokens: Vec<&str> = q
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| t.chars().count() >= min_len)
        .collect();
    tokens.sort_by_key(|t| Reverse(t.len()));
    tokens.iter().map(|s| s.to_lowercase()).collect()
}

/// Concatenate all chunk bodies into a single lowercased string for
/// substring lookups. Chunks lacking a `chunk_text` key are skipped.
pub fn build_haystack_lc(ev: &[SearchResult]) -> String {
    let mut s = String::new();
    for e in ev {
        if let Some(t) = e.payload.get("chunk_text").and_then(|v| v.as_str()) {
            s.push_str(&t.to_lowercase());
            s.push(' ');
        }
    }
    s
}

/// Returns `true` when all `≥min_len`-char alphabetic tokens from
/// `query` are absent from `haystack` (case-folded exact-substring
/// match).
///
/// An empty token set (query too short or all tokens < `min_len`
/// chars) returns `false` so ambiguous short queries do not trigger
/// the short-circuit.
pub fn all_tokens_absent(query: &str, haystack: &str, min_len: usize) -> bool {
    let nouns = subject_nouns(query, min_len);
    if nouns.is_empty() {
        return false;
    }
    nouns.iter().all(|noun| !haystack.contains(noun.as_str()))
}

/// Subject-presence check on the rewritten retrieval queries.
///
/// Reads the queries the agent issued to the retrieval tools (captured
/// in `tool_calls` by [`crate::agent_hook::AgentRsHook`] during the
/// agent run) rather than the raw user prompt — the raw prompt
/// contains instruction-meta tokens (`cheatsheet`, `summary`,
/// `organize`, …) that never appear in real corpora and would produce
/// false positives.
///
/// A query's subject is considered **present in evidence** when at
/// least one of its `≥min_len`-char content tokens appears as a
/// case-folded substring in the evidence haystack. The overall verdict
/// is "absent" only when **every** rewritten query's subject is absent
/// — i.e. no retrieved query's vocabulary overlaps with the evidence
/// corpus at all.
///
/// # Safe fallback
///
/// Returns `false` ("do not short-circuit") when:
///
/// - `evidence` is empty — synthesis already short-circuits on zero
///   evidence with an "I don't know" boundary, no need to duplicate;
/// - `tool_calls` is empty — no recorded queries to derive a subject
///   from;
/// - no `query` field could be extracted from any tool call.
pub fn subject_absent_v2(
    _raw_query: &str,
    tool_calls: &[ToolCallRecord],
    evidence: &[SearchResult],
    min_len: usize,
) -> bool {
    if evidence.is_empty() || tool_calls.is_empty() {
        return false;
    }
    let queries: Vec<String> = tool_calls
        .iter()
        .filter_map(|t| extract_subject_phrase(&t.args_json))
        .collect();
    if queries.is_empty() {
        return false;
    }

    let haystack = build_haystack_lc(evidence);

    // Subject is "absent" only if EVERY rewritten query has ALL its
    // tokens missing from the evidence. If even one token from any
    // tool call hits the haystack, we trust that the subject was
    // retrieved.
    queries
        .iter()
        .all(|q| all_tokens_absent(q, &haystack, min_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_rs_core::types::Score;
    use agent_rs_stores::types::SearchResult;

    fn dummy_evidence(text: &str) -> SearchResult {
        SearchResult {
            id: String::new(),
            score: Score::new(0.0),
            payload: serde_json::json!({"chunk_text": text}),
        }
    }

    fn dummy_tool_call(query: &str) -> ToolCallRecord {
        ToolCallRecord {
            tool_name: "search_knowledge".into(),
            args_json: format!(r#"{{"query":"{}"}}"#, query),
        }
    }

    #[test]
    fn instruction_heavy_query_is_no_longer_false_positive() {
        let raw = "vreau cheatsheet cu TOATE metodele de segmentare punctaj scriu organizat";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă.",
        )];
        let calls = vec![dummy_tool_call("segmentare metode algoritmice")];
        // v2 must extract "segmentare" subject from the rewritten query
        // (in tool_calls) and find it in evidence -> NOT absent.
        assert!(!subject_absent_v2(raw, &calls, &evidence, 4));
    }

    #[test]
    fn out_of_corpus_still_short_circuits() {
        let raw = "ce este React?";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă.",
        )];
        let calls = vec![dummy_tool_call("react javascript framework")];
        assert!(subject_absent_v2(raw, &calls, &evidence, 4));
    }

    #[test]
    fn falls_back_safely_when_no_tool_calls_recorded() {
        let raw = "anything";
        let evidence = vec![dummy_evidence("...")];
        // No tool calls -> v2 returns false (do not short-circuit).
        assert!(!subject_absent_v2(raw, &[], &evidence, 4));
    }

    #[test]
    fn falls_back_safely_when_evidence_empty() {
        let raw = "anything";
        let calls = vec![dummy_tool_call("anything")];
        // No evidence -> v2 returns false (let the existing zero-evidence
        // path in synthesis handle it).
        assert!(!subject_absent_v2(raw, &calls, &[], 4));
    }

    #[test]
    fn extract_subject_phrase_returns_none_for_section_id_args() {
        // fetch_section_chunks uses `section_id`, not `query` — must skip.
        let args = r#"{"section_id":"abc-def-123"}"#;
        assert!(extract_subject_phrase(args).is_none());
    }

    #[test]
    fn subject_nouns_orders_longest_first_and_lowercases() {
        // Use lengths that are strictly distinct so the
        // `sort_by_key(Reverse(len))` ordering is unambiguous (no
        // tie-breaking dependence on input order).
        let nouns = subject_nouns("Quick Segmentation Methods", 4);
        assert_eq!(nouns, vec!["segmentation", "methods", "quick"]);
        // Lower-casing happens regardless of input casing.
        for n in &nouns {
            assert!(n.chars().all(|c| !c.is_uppercase()));
        }
    }

    #[test]
    fn all_tokens_absent_returns_false_when_nouns_empty() {
        // All tokens shorter than min_len → empty noun set → must NOT
        // short-circuit (would otherwise short-circuit every short
        // query).
        assert!(!all_tokens_absent("hi", "anything", 4));
    }
}
