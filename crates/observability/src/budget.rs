//! Per-request LLM token + cost tracking.
//!
//! [`RequestBudget`] is created at the entry of `process_query` and shared
//! via `Arc` to every LLM-calling component that has access to the
//! [`crate`](agent-rs-agent)'s `QueryContext`. Each LLM call records its
//! prompt + completion tokens (typically read from the provider's
//! `usage` field on the response). At the end of the request, the
//! budget is logged with totals + cost estimate; histograms are
//! emitted per call for production observability.
//!
//! # Why a tracker, not a guard
//!
//! Aborting mid-request leaves the user with a partial answer and
//! forfeits already-spent tokens. This phase ships observability +
//! soft warnings; a hard `BudgetGuard::try_consume` abort path is a
//! follow-up.
//!
//! # Scope
//!
//! Query-time only. Ingestion-time LLM calls (summarizer, graph
//! extractor) are not query-scoped and therefore intentionally do not
//! update a [`RequestBudget`]; they still emit per-call histograms via
//! [`crate::metrics::RetrievalMetrics::record_llm_call`]. Per-corpus
//! ingestion-cost tracking is a follow-up.
//!
//! # Pricing table
//!
//! [`estimate_cost_usd`] uses a small built-in price table. Rates are
//! current OpenAI pricing as of 2026-Q1 — they should be updated when
//! pricing changes. Unknown models fall back to a conservative default
//! (gpt-4o-mini pricing) and emit a `tracing::warn` once per process.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Microcents per US dollar (USD * 1e6). Cost is held in atomics as
/// integer microcents so concurrent updates stay race-free without a
/// mutex; convert to floating-point USD at snapshot time.
const MICROCENTS_PER_USD: f64 = 1_000_000.0;

/// Tokens consumed by a single LLM call.
///
/// Both fields are typically lifted from the provider's `usage` object
/// on a chat-completion response. When the provider does not surface
/// usage (e.g. rig-core's `agent.prompt` opaque path), call sites
/// should pass `TokenUsage::default()` and rely on the latency-only
/// signal — the budget snapshot will under-count, which is preferable
/// to fabricating a number.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Prompt-side tokens (system + user + tool messages).
    pub prompt_tokens: u32,
    /// Completion-side tokens (assistant + tool-call output).
    pub completion_tokens: u32,
}

impl TokenUsage {
    /// Construct a [`TokenUsage`] from raw `u32` fields.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
        }
    }

    /// Sum of prompt and completion tokens.
    pub fn total(self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// Per-request token + cost aggregator.
///
/// Lock-free aggregation: one [`AtomicU64`] per counter. Cheap enough
/// to wrap every LLM call without measurable overhead. Constructed
/// once per `process_query` and shared via `Arc` across pipeline
/// steps.
#[derive(Debug, Default)]
pub struct RequestBudget {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    call_count: AtomicU64,
    /// Microcents (USD * 1e6) so cost stays integer-safe in atomics.
    cost_microcents: AtomicU64,
}

impl RequestBudget {
    /// Create a new empty budget wrapped in an `Arc` so callers can
    /// hand it to multiple pipeline steps without further wrapping.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a single LLM call.
    ///
    /// `model` drives cost estimation via [`estimate_cost_usd`]. All
    /// counters are updated with `Ordering::Relaxed` since the totals
    /// are read once at the end of the request and we don't need
    /// inter-thread happens-before ordering between increments.
    pub fn record_call(&self, model: &str, usage: TokenUsage) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(u64::from(usage.prompt_tokens), Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(u64::from(usage.completion_tokens), Ordering::Relaxed);

        let cost_usd = estimate_cost_usd(model, usage);
        // Cost is held in microcents (u64) so concurrent accumulation is
        // race-free without a mutex. Negative or non-finite costs are
        // coerced to zero — `estimate_cost_usd` itself never returns
        // negative for valid u32 token counts, this is just defensive.
        let microcents = if cost_usd.is_finite() && cost_usd >= 0.0 {
            (cost_usd * MICROCENTS_PER_USD) as u64
        } else {
            0
        };
        self.cost_microcents
            .fetch_add(microcents, Ordering::Relaxed);
    }

    /// Read a consistent snapshot of all counters.
    ///
    /// Reads are `Ordering::Relaxed` and are not synchronised against
    /// each other, so concurrent writers may produce a snapshot whose
    /// `total_tokens` differs from `prompt_tokens + completion_tokens`
    /// by one in-flight call. Acceptable for end-of-request logging.
    pub fn snapshot(&self) -> BudgetSnapshot {
        let prompt = self.prompt_tokens.load(Ordering::Relaxed);
        let completion = self.completion_tokens.load(Ordering::Relaxed);
        let calls = self.call_count.load(Ordering::Relaxed);
        let microcents = self.cost_microcents.load(Ordering::Relaxed);
        BudgetSnapshot {
            call_count: calls,
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt.saturating_add(completion),
            cost_usd: (microcents as f64) / MICROCENTS_PER_USD,
        }
    }
}

/// Immutable snapshot of [`RequestBudget`] state, suitable for
/// structured logging and metric emission at end-of-request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetSnapshot {
    /// Number of LLM calls aggregated.
    pub call_count: u64,
    /// Sum of prompt-side tokens across all recorded calls.
    pub prompt_tokens: u64,
    /// Sum of completion-side tokens across all recorded calls.
    pub completion_tokens: u64,
    /// Convenience: `prompt_tokens + completion_tokens`.
    pub total_tokens: u64,
    /// Estimated USD spend across all calls.
    pub cost_usd: f64,
}

// ---- Pricing table --------------------------------------------------------
//
// USD per 1M tokens. Rates as of 2026-Q1; update when OpenAI changes
// published pricing. Centralised so a price refresh is a one-line
// change. Unknown models fall back to gpt-4o-mini pricing
// (conservative — under-reports for expensive models, never
// over-reports) and emit a one-shot WARN log so missing entries
// surface in operator dashboards.

/// Per-million-token rate in USD. Float storage keeps the table
/// readable (rates match published OpenAI dollar prices verbatim) and
/// avoids the unit-conversion footguns of integer-microcent
/// representations.
#[derive(Debug, Clone, Copy)]
struct ModelRate {
    /// USD charged per 1M prompt tokens.
    prompt_per_1m_usd: f64,
    /// USD charged per 1M completion tokens.
    completion_per_1m_usd: f64,
}

/// Conservative fallback rate (gpt-4o-mini family) used when a model
/// name is not present in the pricing table. Picked so that the
/// estimator under-reports costs for expensive models — operators see
/// a low number and dig in, rather than seeing an inflated number that
/// silently wastes their alert budget.
const FALLBACK_RATE: ModelRate = ModelRate {
    prompt_per_1m_usd: 0.15,
    completion_per_1m_usd: 0.60,
};

/// Look up the per-million-token rate for a model identifier.
///
/// Match by prefix so date-stamped IDs (e.g. `gpt-5.4-2026-03-17`)
/// route to the right base price without a combinatorial explosion of
/// table entries. Order matters: most-specific prefixes first so
/// `gpt-5.4-mini` does not collide with `gpt-5.4`.
fn rate_for(model: &str) -> Option<ModelRate> {
    if model.starts_with("gpt-5.4-mini") || model.starts_with("gpt-4.7-mini") {
        // mini family: cheap reasoning workhorse.
        return Some(ModelRate {
            prompt_per_1m_usd: 0.15,
            completion_per_1m_usd: 0.60,
        });
    }
    if model.starts_with("gpt-4o-mini") {
        return Some(ModelRate {
            prompt_per_1m_usd: 0.15,
            completion_per_1m_usd: 0.60,
        });
    }
    if model.starts_with("gpt-5.4") || model.starts_with("gpt-4.7") || model.starts_with("gpt-4o") {
        // Full-size flagship pricing. gpt-4o-* dated variants land here.
        return Some(ModelRate {
            prompt_per_1m_usd: 2.50,
            completion_per_1m_usd: 10.00,
        });
    }
    if model.starts_with("o1") || model.starts_with("o3") {
        // Reasoning models are an order of magnitude pricier.
        return Some(ModelRate {
            prompt_per_1m_usd: 15.00,
            completion_per_1m_usd: 60.00,
        });
    }
    None
}

/// Tokens-per-million conversion constant — kept as a named `const`
/// so the multiplier in [`estimate_cost_usd`] documents itself.
const TOKENS_PER_MILLION: f64 = 1_000_000.0;

/// Estimate USD cost from token usage for a given model.
///
/// Uses a small built-in price table. Unknown models fall back to
/// gpt-4o-mini pricing and emit a `tracing::warn!` the first time a
/// given unknown name is seen per process.
///
/// Returns `0.0` for `usage.total() == 0` (no tokens consumed) without
/// touching the warn path.
pub fn estimate_cost_usd(model: &str, usage: TokenUsage) -> f64 {
    if usage.total() == 0 {
        return 0.0;
    }
    let rate = match rate_for(model) {
        Some(r) => r,
        None => {
            warn_unknown_model_once(model);
            FALLBACK_RATE
        }
    };
    let prompt_cost =
        (f64::from(usage.prompt_tokens) / TOKENS_PER_MILLION) * rate.prompt_per_1m_usd;
    let completion_cost =
        (f64::from(usage.completion_tokens) / TOKENS_PER_MILLION) * rate.completion_per_1m_usd;
    prompt_cost + completion_cost
}

/// Emit a `tracing::warn!` the first time `model` is observed without
/// a price-table entry. Subsequent calls for the same name are
/// silent — keeping logs quiet while still surfacing the gap to
/// operators.
fn warn_unknown_model_once(model: &str) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = match seen.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.insert(model.to_owned()) {
        tracing::warn!(
            model,
            "RequestBudget: model not in pricing table; using gpt-4o-mini fallback rates. \
             Add it to `rate_for()` in crates/observability/src/budget.rs to fix."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_total_sums_fields() {
        assert_eq!(TokenUsage::new(100, 200).total(), 300);
        assert_eq!(TokenUsage::default().total(), 0);
    }

    #[test]
    fn token_usage_total_saturates_on_overflow() {
        // u32::MAX + 1 would overflow without saturating_add.
        let usage = TokenUsage::new(u32::MAX, 1);
        assert_eq!(usage.total(), u32::MAX);
    }

    #[test]
    fn record_call_accumulates_atomically() {
        let budget = RequestBudget::new();
        budget.record_call("gpt-4o-mini", TokenUsage::new(100, 50));
        budget.record_call("gpt-4o-mini", TokenUsage::new(200, 75));

        let snap = budget.snapshot();
        assert_eq!(snap.call_count, 2);
        assert_eq!(snap.prompt_tokens, 300);
        assert_eq!(snap.completion_tokens, 125);
        assert_eq!(snap.total_tokens, 425);
        // gpt-4o-mini: $0.15/1M prompt + $0.60/1M completion.
        // (300/1e6)*0.15 + (125/1e6)*0.60 = 0.000045 + 0.000075 = 0.000120
        // Microcent quantisation in the atomic loses sub-microcent
        // precision, so compare with a 1e-6 tolerance not 1e-9.
        assert!(
            (snap.cost_usd - 0.000_120).abs() < 1e-6,
            "expected ≈ $0.000120, got {}",
            snap.cost_usd
        );
    }

    #[test]
    fn cost_estimate_uses_correct_rate_per_model_family() {
        // gpt-5.4: $2.50/1M prompt + $10.00/1M completion.
        let big = estimate_cost_usd("gpt-5.4-2026-03-17", TokenUsage::new(1000, 1000));
        // (1000/1e6)*2.50 + (1000/1e6)*10.00 = 0.0025 + 0.01 = 0.0125
        assert!((big - 0.0125).abs() < 1e-9);

        // o1: $15/1M prompt + $60/1M completion.
        let reasoning = estimate_cost_usd("o1-preview", TokenUsage::new(1000, 1000));
        // (1000/1e6)*15 + (1000/1e6)*60 = 0.015 + 0.060 = 0.075
        assert!((reasoning - 0.075).abs() < 1e-9);

        // gpt-4o-mini: $0.15/1M prompt + $0.60/1M completion.
        let mini = estimate_cost_usd("gpt-4o-mini", TokenUsage::new(1000, 1000));
        // (1000/1e6)*0.15 + (1000/1e6)*0.60 = 0.00015 + 0.00060 = 0.00075
        assert!((mini - 0.000_75).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_falls_back_to_mini_pricing() {
        // An unknown name should compute the same cost as gpt-4o-mini.
        let unknown =
            estimate_cost_usd("totally-fictional-model-2099", TokenUsage::new(1000, 1000));
        let mini = estimate_cost_usd("gpt-4o-mini", TokenUsage::new(1000, 1000));
        assert!((unknown - mini).abs() < 1e-9);
    }

    #[test]
    fn zero_usage_short_circuits_to_zero_cost() {
        let cost = estimate_cost_usd("never-seen-model", TokenUsage::default());
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn snapshot_returns_consistent_totals() {
        let budget = RequestBudget::new();
        budget.record_call("gpt-4o", TokenUsage::new(10, 20));
        budget.record_call("gpt-4o", TokenUsage::new(30, 40));
        let snap = budget.snapshot();
        assert_eq!(
            snap.prompt_tokens + snap.completion_tokens,
            snap.total_tokens
        );
        assert_eq!(snap.call_count, 2);
    }

    #[tokio::test]
    async fn multiple_concurrent_record_calls_produce_correct_sum() {
        let budget = RequestBudget::new();
        let mut handles = Vec::new();
        // Spawn 16 tasks each recording 100 calls of (5, 10) tokens.
        // Expected: 1600 calls, prompt=8000, completion=16000.
        for _ in 0..16 {
            let b = Arc::clone(&budget);
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    b.record_call("gpt-4o-mini", TokenUsage::new(5, 10));
                }
            }));
        }
        for h in handles {
            h.await.expect("task must not panic");
        }
        let snap = budget.snapshot();
        assert_eq!(snap.call_count, 1600);
        assert_eq!(snap.prompt_tokens, 8_000);
        assert_eq!(snap.completion_tokens, 16_000);
        assert_eq!(snap.total_tokens, 24_000);
    }
}
