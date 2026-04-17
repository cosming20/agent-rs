//! Determinism constants for ranking-signal LLM calls.
//!
//! Any LLM call whose output feeds into retrieval ranking — weighted
//! keywords, metadata filter, entity linker, query rewriter, graph-CoT
//! sufficiency checks — must be reproducible across runs so that:
//!
//! - A/B evaluation (comparing two retrieval configurations on the same
//!   query set) reflects ranker differences, not LLM sampling noise.
//! - Production queries converge to the same evidence pack when the
//!   corpus and code are unchanged.
//!
//! The two knobs OpenAI exposes for this are `temperature: 0.0` and a
//! fixed `seed`. `temperature: 0.0` makes the sampling deterministic for
//! non-reasoning models; `seed` is best-effort on the reasoning models
//! (gpt-5.x, o-series) but still drastically reduces variance.
//!
//! Generation-style LLM calls (summarisation, synthesis, HyDE when used
//! for diversity rather than ranking) may intentionally use higher
//! temperature; those call sites should not import [`RANKING_LLM_SEED`].

/// Seed used on every Chat Completions request whose output contributes
/// to retrieval ranking. Picked once and frozen so rerunning the same
/// query yields the same prompt-token → completion-token mapping.
pub const RANKING_LLM_SEED: u32 = 42;
