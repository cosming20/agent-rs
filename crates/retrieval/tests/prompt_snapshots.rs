//! Prompt snapshot tests: forbidden-lexicon guard with NFKC + case-fold
//! normalisation. Layer 2 of the bias guard.
//!
//! See `docs/superpowers/specs/2026-04-17-weighted-keywords-design.md` §8.4.
//!
//! Assertions per module:
//! 1. The module's public system prompt, after NFKC + lowercase + punctuation
//!    stripping + whitespace collapse, contains none of the forbidden-lexicon
//!    entries.
//! 2. The prompt matches its committed snapshot fixture so accidental edits
//!    surface during code review.

use std::fs;
use std::path::PathBuf;

use unicode_normalization::UnicodeNormalization;

fn normalize(s: &str) -> String {
    let nfkc: String = s.nfkc().collect();
    let lower = nfkc.to_lowercase();
    let stripped: String = lower
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

const FORBIDDEN: &[&str] = &[
    "penal",
    "criminal",
    "romanian",
    "codul",
    "articolul",
    "pedepsit",
    "cod penal",
];

fn assert_no_forbidden(label: &str, prompt: &str) {
    let n = normalize(prompt);
    for f in FORBIDDEN {
        assert!(
            !n.contains(f),
            "prompt {label} contains forbidden lexicon entry {f:?} after normalisation"
        );
    }
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("src/testdata/prompts");
    p.push(name);
    p
}

fn assert_matches_fixture(label: &str, prompt: &str, fixture_name: &str) {
    let path = fixture_path(fixture_name);
    // Set UPDATE_SNAPSHOTS=1 to (re)write the fixture from the live prompt.
    // Intended for initial creation and deliberate prompt updates only.
    if std::env::var("UPDATE_SNAPSHOTS").ok().as_deref() == Some("1") {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&path, prompt.trim_end())
            .unwrap_or_else(|e| panic!("failed to write fixture {}: {e}", path.display()));
        return;
    }
    let fixture = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {fixture_name} at {} — re-run with UPDATE_SNAPSHOTS=1 to create: {e}",
            path.display()
        )
    });
    assert_eq!(
        prompt.trim(),
        fixture.trim(),
        "prompt {label} diverged from committed fixture {fixture_name} (run with UPDATE_SNAPSHOTS=1 if this change is intentional)"
    );
}

#[test]
fn query_rewriter_system_prompt_is_universal_and_matches_fixture() {
    let p = agent_rs_retrieval::query_rewriter::SYSTEM_PROMPT_FOR_TESTING;
    assert_no_forbidden("query_rewriter", p);
    assert_matches_fixture("query_rewriter", p, "query_rewriter_system.txt");
}

#[test]
fn metadata_filter_system_prompt_is_universal_and_matches_fixture() {
    let p = agent_rs_retrieval::metadata_filter::SYSTEM_PROMPT_FOR_TESTING;
    assert_no_forbidden("metadata_filter", p);
    assert_matches_fixture("metadata_filter", p, "metadata_filter_system.txt");
}

#[test]
fn entity_linker_extract_prompt_is_universal_and_matches_fixture() {
    let p = agent_rs_retrieval::entity_linker::ENTITY_EXTRACT_PROMPT_FOR_TESTING;
    assert_no_forbidden("entity_linker", p);
    assert_matches_fixture("entity_linker", p, "entity_linker_extract.txt");
}

#[test]
fn graph_extractor_system_prompt_is_universal_and_matches_fixture() {
    let p = agent_rs_retrieval::ingestion::graph_extractor::SYSTEM_PROMPT_FOR_TESTING;
    assert_no_forbidden("graph_extractor", p);
    assert_matches_fixture("graph_extractor", p, "graph_extractor_system.txt");
}

#[test]
fn agent_default_system_prompt_is_universal_and_matches_fixture() {
    let p = agent_rs_agent::prompts::DEFAULT_SYSTEM_PROMPT;
    assert_no_forbidden("agent_default", p);
    assert_matches_fixture("agent_default", p, "agent_system.txt");
}
