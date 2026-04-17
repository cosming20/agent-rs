# Weighted-Keyword Generator + Prompt Universalization — Design Spec

**Status:** Draft · **Date:** 2026-04-17 · **Author:** agent-rs team
**Sub-project:** 1 of 4 in the retrieval-enhancement roadmap (see "Roadmap context")

---

## 1. Problem statement

The retrieval pipeline in `agent-rs` has a universal architecture but carries
two concrete gaps that this sub-project addresses together:

1. **Unweighted keyword signal.** The BM25 branch in `search/keyword` treats
   every query token as equally important, and neither the cross-encoder
   reranker nor the entity linker has any notion of which terms matter most to
   the user's intent. Rare / discriminative terms do not get extra weight, so
   retrieval under-serves queries that hinge on a small number of
   high-information tokens.

2. **Residual domain bias in LLM prompts.** Several retrieval and agent
   prompts still contain explicit framing around Romanian legal text
   (`articolul`, `Cod Penal`, `pedepsit_cu`, "For legal queries…"). This
   contradicts the project's universal-system principle and, more importantly,
   would create incoherent behavior if a *universal* weighted-keyword signal
   were wired into *domain-biased* consumers. Both must ship together.

## 2. Goal

Deliver one PR that:

- Adds a universal LLM-backed `WeightedKeywordExtractor` that emits a ranked,
  weighted-term view of the user query.
- Wires the output into three consumers — `search/keyword`, the rerank-time
  score adjustment (in `ranker.rs`), and `entity_linker` — all enabled by
  default.
- Removes Romanian / legal bias from the five prompt-bearing files enumerated
  in §8.
- Adds a CI guard that prevents re-introducing domain bias.
- Measures a Hit@1 improvement on a freshly-regenerated eval dataset built
  from the current `test.pdf`.

## 3. Non-goals

- **Learned sparse retrieval (SPLADE) / new Qdrant sparse index.** That is
  sub-project #4 in the roadmap.
- **Corpus-derived token bridger / fuzzy matching.** Sub-project #3.
- **LLM document scorer over full node text.** Sub-project #2.
- **Changing the Qdrant ingestion format or re-indexing.** This sub-project
  only adds a query-side component and must not require re-ingestion.
- **Per-sub-query keyword refinement.** Forward-compatible (see §5.2) but
  not built here.

## 4. Roadmap context

This is sub-project #1 of a four-part retrieval enhancement roadmap agreed
during brainstorming:

| # | Sub-project | Status |
|---|---|---|
| **1** | **LLM Weighted-Keyword Generator + Prompt Universalization** | This spec |
| 2 | LLM Document Scorer (full-text, post-merge) | Future |
| 3 | Corpus-derived Token Bridger (fuzzy / morphology) | Future |
| 4 | Learned Sparse channel (SPLADE-style) | Future |

Sub-projects 2–4 each get their own spec cycle. Sub-project #1 lands first
because it has the smallest blast radius (query-side, no ingestion or storage
changes) and its prompt-de-biasing work is a prerequisite for any future
universal feature regardless of which direction 2–4 are pursued.

## 5. Architecture

### 5.1 Module layout

One new module in the existing `agent-rs-retrieval` crate:

```
crates/retrieval/src/
  weighted_keywords.rs              (NEW)
```

No new crate, no new external dependencies. Uses `rig-core` for the LLM call
(same pattern as `query_rewriter.rs` and `LlmGraphExtractor`), `backon` for
retry with exponential backoff (existing dep), and `serde` for schema
parsing.

Public exports re-surfaced through `crates/retrieval/src/lib.rs`:

- `WeightedKeywordExtractor`
- `WeightedKeywordExtractorConfig`
- `WeightedKeywords`
- `WeightedTerm`
- `WeightedKeywordIntegrationConfig`

### 5.2 Public types

```rust
/// LLM-backed extractor that produces a ranked, weighted view of a query's
/// information-bearing terms, used to guide downstream retrieval consumers.
pub struct WeightedKeywordExtractor { /* opaque */ }

/// Typed configuration for the extractor.
pub struct WeightedKeywordExtractorConfig {
    /// OpenAI-compatible API key (same env convention as other LLM modules).
    pub api_key: String,
    /// Model identifier. Default comes from `agent_rs_core::config`.
    pub model: String,
    /// Max retries on transient failure. Default: 2.
    pub max_retries: u32,
    /// Per-call timeout. Default: 8 seconds (matches `query_rewriter`).
    pub timeout: Duration,
}

/// Output of the extractor. Stable public type — future enhancements (e.g.,
/// per-sub-query refinement) must not widen this type; they add new methods
/// on `WeightedKeywordExtractor` that return another `WeightedKeywords`.
pub struct WeightedKeywords {
    /// Ranked, weight-normalized terms.
    ///
    /// Invariants (enforced at construction):
    /// - Every `weight` is in `(0.0, 1.0]`.
    /// - Sorted by `weight` descending.
    /// - Deduplicated by `term` (case-insensitive Unicode folding, NFKC).
    pub terms: Vec<WeightedTerm>,
}

pub struct WeightedTerm {
    /// Surface form as emitted by the LLM, trimmed, with surrounding
    /// punctuation removed. No stemming, no language-specific morphology.
    pub term: String,
    /// Normalized importance score in `(0.0, 1.0]`.
    pub weight: f32,
    /// True if this term is a multi-token phrase to be treated as a
    /// contiguous token group by downstream consumers.
    pub is_phrase: bool,
}

/// Cross-consumer integration toggles. One field per consumer; any field set
/// to `None` disables that consumer's integration without touching the
/// extractor itself.
pub struct WeightedKeywordIntegrationConfig {
    pub bm25: Option<BM25WeightingConfig>,
    pub rerank_alpha: Option<f32>,
    pub entity_linker_top_k: Option<usize>,
}

pub struct BM25WeightingConfig {
    /// Terms with weight strictly less than `min_weight` are dropped from
    /// the rewritten BM25 query string.
    pub min_weight: f32,
    /// Upper cap on term repetition in the rewritten BM25 query string.
    pub max_repeats: u32,
}
```

All default values live in `agent-rs-core::config` (already the pattern for
similar components) — no magic values in code paths.

### 5.3 Forward compatibility for per-sub-query refinement

Approach 2 from brainstorming (per-sub-query weighted keywords) slots in
later without changing the output type. It will appear as:

```rust
impl WeightedKeywordExtractor {
    pub async fn extract(&self, query: &str)
        -> Result<WeightedKeywords, RetrievalError>;

    // Future — not built in this sub-project:
    pub async fn refine_for_sub_query(
        &self,
        global: &WeightedKeywords,
        sub_query: &SubQuery,
    ) -> Result<WeightedKeywords, RetrievalError>;
}
```

Consumers always take `&WeightedKeywords` — they never know or care whether
the instance was global or refined.

## 6. Data flow

Inside `router.rs`, the extractor runs as one more parallel branch in the
existing `tokio::join!` bundle that already fans out `query_rewriter`,
`metadata_filter`, and `entity_linker`:

```
user query
    │
    ├─► query_rewriter.rewrite(query)                 ──┐
    ├─► weighted_keyword_extractor.extract(query)     ──┤ parallel
    ├─► metadata_filter.extract(query)                ──┤ tokio::join!
    └─► entity_linker.link(query)                     ──┘
                        │
                        ▼
       Arc<WeightedKeywords> passed by reference into:
         • search/keyword   (BM25 query construction; §7.1)
         • ranker.rs        (post-rerank adjustment;    §7.2)
         • entity_linker    (candidate prioritization;  §7.3)
                        │
                        ▼
        per-strategy retrieval → RRF fusion → rerank → evidence pack
```

The extractor is held as `Arc<WeightedKeywordExtractor>` on the router
(consistent with other LLM extractors) and its output wrapped in `Arc<…>`
for cheap sharing among consumers.

When the extractor fails (all retries exhausted) the router receives
`WeightedKeywords::empty()` and the pipeline continues unmodified.

## 7. Consumer integrations

All three are enabled by default. Each is independently toggleable at
runtime via `WeightedKeywordIntegrationConfig`.

### 7.1 `search/keyword` — weighted BM25 query rewrite

External BM25 backends (Qdrant's server-side BM25, most others) accept only
a query string; they tokenize it themselves and cannot be given per-term
weight vectors without sharing the indexer's tokenizer. We therefore treat
weights as a **query rewrite signal**, not a BM25 API parameter.

One pure function:

```rust
pub fn weighted_bm25_query(
    original_query: &str,
    kw: &WeightedKeywords,
    cfg: &BM25WeightingConfig,
) -> String;
```

Algorithm:

1. If `kw.terms.is_empty()` → return `original_query.to_string()` unchanged.
2. Drop terms whose `weight < cfg.min_weight`.
3. For each surviving term:
   - If `is_phrase` is true, split on whitespace into component tokens.
   - Otherwise treat the term itself as one token.
4. Emit each token `repeats = max(1, ceil(weight * cfg.max_repeats as f32)) as u32`
   times.
5. Join all emitted tokens with a single space — this becomes the new BM25
   query text.

BM25's sub-linear TF saturation means the repetition produces a controlled
boost that asymptotes. `max_repeats` caps the amplification.

No tokenizer-sharing, no corpus priors, no stemming, no language detection.
Works with any BM25 backend Qdrant swaps in.

### 7.2 `ranker.rs` — post-rerank score adjustment

Cross-encoder rerankers (Jina, ZeroEntropy, etc.) are self-contained — they
do not accept weight hints on their API. Prepending weighted terms into the
query text risks confusing a model trained on natural-language queries.
Instead, we adjust the score **after** the cross-encoder returns it.

The adjustment lives in `ranker.rs` (post-rerank surface) — `reranker.rs`
stays a pure cross-encoder wrapper.

```rust
fn adjust_for_weighted_keywords(
    reranker_score: f32,
    candidate_text: &str,
    kw: &WeightedKeywords,
    alpha: f32,
) -> f32 {
    if kw.terms.is_empty() || alpha == 0.0 {
        return reranker_score;
    }
    let mut boost = 0.0_f32;
    let lower = candidate_text.to_lowercase();
    for t in &kw.terms {
        if lower.contains(&t.term.to_lowercase()) {
            boost += t.weight;
        }
    }
    reranker_score * (1.0 + alpha * boost)
}
```

Default `alpha = 0.2`. The cross-encoder remains the dominant signal;
weights bias toward candidates containing user-important terms. Empty
keywords → the function is a byte-identical no-op.

### 7.3 `entity_linker` — candidate prioritization

Today `entity_linker` LLM-extracts entity candidates from the query, then
looks them up in Neo4j. It treats every candidate equally. With weighted
keywords available, we prioritize:

- First: top-K highest-weight terms (K = `entity_linker_top_k`, default 5)
  become the first candidates tried against Neo4j.
- Then: remaining LLM-extracted entity candidates are appended.
- Early-exit: once `EntityLinkerConfig::max_matches` entities are found,
  remaining lookups are skipped.

No change to the `entity_linker::link()` signature — only its internal
candidate-ordering logic. When weighted keywords are empty, behavior is
byte-identical to today.

## 8. Prompt universalization (prerequisite work in this same PR)

Both features land together so the pipeline never has a universal component
feeding a domain-biased consumer.

### 8.1 Files changed

| File | Change |
|---|---|
| `crates/retrieval/src/query_rewriter.rs` | Remove "Romanian legal awareness" doc framing. Replace Romanian-legal example queries with mixed-domain abstract placeholders (software docs / fiction / business-notes styles). Task description stays strictly structural. |
| `crates/retrieval/src/metadata_filter.rs` | Generalize semantics of the `article_number` field to *numbered-section reference* (backward-compatible — same field, broader match). Strip `articolul`, `alin.`, `Cod Penal` examples. Replace with neutral `Section 15`, `§ 4.2`, `Chapter 3` style examples. |
| `crates/retrieval/src/entity_linker.rs` | Remove the "For legal queries: extract article numbers…" rider. Keep generic entity categories (Person, Organization, Location, Date, Event, Concept, Document). |
| `crates/retrieval/src/ingestion/graph_extractor.rs` | Strip Romanian legal examples. Keep universal entity type set. Replace the `pedepsit_cu` relationship example with multiple neutral-domain relationship examples (e.g., `authored_by`, `located_in`, `depends_on`). |
| `crates/agent/src/prompts.rs` | Rewrite the routing heuristic: *"Queries mentioning an identifier (number, code, section reference, ID) → keyword first"*. No domain words. |
| `crates/retrieval/src/router.rs` | Delete the "too specific for Romanian legal text" comment. |

### 8.2 Not changed (intentional)

- `crates/retrieval/src/ingestion/metadata.rs` — Romanian diacritic handling
  is Unicode character-set support, not domain bias. Stays.

### 8.3 Replacement example style

Prompts that contain few-shot examples get 2–3 examples drawn from
*different* neutral domains per prompt. Each prompt mixes domains so no
single one implicitly anchors the model. Suggested domain pool: software
documentation, fiction excerpts, meeting notes, encyclopedia entries,
product manuals, recipes. None tied to the current `test.pdf` content.

### 8.4 CI guard

One new CI step (and a matching pre-commit hook entry) greps the `crates/`
tree for a forbidden-word list:

```
penal | criminal | romanian | codul | articolul | pedepsit | Cod Penal
```

Matches in `crates/retrieval/src/ingestion/metadata.rs` are allowlisted
(the diacritic handling file). Any other match fails the build. This
stops bias from re-entering later.

## 9. Error handling

### 9.1 Safety principle

The extractor must never block a query. Every failure collapses to
`WeightedKeywords::empty()`, and every consumer's empty-input path is
byte-identical to today's behavior. This is the property that justifies
shipping all three integrations enabled by default.

### 9.2 Failure matrix

| Failure | Response |
|---|---|
| LLM 5xx / timeout / connection error | Retry with exponential backoff via `backon`; `empty()` after `max_retries` |
| LLM 429 rate-limit | Retry respecting `Retry-After` header when present |
| LLM 4xx (non-429) | No retry; `empty()` + `tracing::warn!` + failure metric |
| Malformed JSON / schema violation | Retry once with the same prompt; `empty()` if still malformed |
| Valid JSON but zero terms | Return `empty()` normally (not an error) |
| Weight outside `(0.0, 1.0]` | Clamp into range, log warning, keep the rest of the result |
| NaN / infinite weights | Drop that term, log warning, keep the rest |

### 9.3 Observability

- Traces: `#[instrument(skip(self))]` on `extract`, fields
  `%query_hash, terms_count, elapsed_ms`. Query is hashed (blake3),
  never logged verbatim.
- Metrics:
  - `weighted_keyword_extractor_calls_total`
  - `weighted_keyword_extractor_failures_total{kind=…}`
  - `weighted_keyword_extractor_latency_ms` histogram
- Per-consumer structured-field logs when weights changed behavior:
  `bm25_query_rewritten=true`, `rerank_adjusted=true`,
  `entity_linker_prioritized=true`.

## 10. Testing strategy

### 10.1 Unit tests (`#[cfg(test)] mod tests` in `weighted_keywords.rs`)

- Prompt schema construction matches the expected JSON shape.
- Response parser handles: valid JSON, malformed JSON, missing fields
  (error), extra fields (ignored), empty terms list (ok).
- Normalization invariants: output weights in `(0.0, 1.0]`, sorted
  descending, deduplicated by NFKC-lowercased term, `is_phrase` preserved.
- `WeightedKeywords::empty()` round-trip: construct → serialize → parse →
  equal.

### 10.2 Unit tests for pure integration functions

- `weighted_bm25_query`:
  - Empty keywords → output equals `original_query`.
  - Term repetition respects `max_repeats` cap.
  - Threshold filter drops terms below `min_weight`.
  - Phrase splitting: phrase tokens treated individually.
- `adjust_for_weighted_keywords`:
  - `alpha == 0` → returns `reranker_score` unchanged.
  - Empty keywords → returns `reranker_score` unchanged.
  - Boost monotonically non-decreasing in term coverage.
- Entity-linker top-K selection:
  - Terms returned in weight-descending order.
  - Respects `top_k` cap.

### 10.3 Property tests (`proptest`)

On `weighted_bm25_query`:

- **Idempotency:** applying the function with the output as the new
  `original_query` and the same keywords gives a stable result under
  repeated application (converges).
- **Monotonicity:** for two terms *a* and *b* with `weight_a > weight_b`
  and equal `is_phrase`, *a* appears at least as many times as *b* in the
  output.
- **Cap respect:** no individual token appears more than `max_repeats`
  times per original-term (across terms is unbounded, within is capped).

### 10.4 Integration tests (`crates/retrieval/tests/weighted_keywords_integration.rs`)

- Mock LLM returns canned `WeightedKeywords`; assert the same value is
  visible to all three consumers and actually modifies their behavior
  (observable via structured log fields or injected probes).
- Mock LLM fails (exhaust all retries); assert the router still produces
  a valid `EvidencePack` and consumers behaved as today.
- Mock LLM times out; assert the `tokio::join!` completes within the
  router's overall budget.

### 10.5 Doctests

Every public item (`WeightedKeywordExtractor`, `WeightedKeywords`,
`WeightedTerm`, `WeightedKeywordIntegrationConfig`,
`BM25WeightingConfig`, `weighted_bm25_query`) carries a runnable
`/// # Examples` block per the project Rust style guide.

### 10.6 End-to-end evaluation

**Step 1** — regenerate eval dataset from the current `test.pdf`:

```bash
python eval/generate_enterprise_v3.py \
    --pdf test.pdf \
    --output eval/test_pdf_v1.json
```

Produces a dataset with ~200 questions stratified across the existing 10
question types and 3 difficulty tiers. Committed alongside the spec for
reproducibility. The generator is already domain-agnostic by construction.

**Step 2** — after prompt de-biasing but *before* building the extractor,
capture a baseline:

```bash
EVAL_DATASET=eval/test_pdf_v1.json \
EVAL_SAVE_BASELINE=1 \
cargo run --release --example eval_harness
```

This baseline reflects the *post-de-bias, no-feature* state. Using a
post-de-bias baseline isolates the feature's true contribution from any
recovery-from-bias effect.

**Step 3** — after building the extractor and wiring consumers:

```bash
EVAL_DATASET=eval/test_pdf_v1.json \
cargo run --release --example eval_harness
```

Harness auto-compares against `baseline_eval.json`.

**Step 4** — ablation runs, each toggling one consumer while disabling
the other two via `WeightedKeywordIntegrationConfig`:

```
ablation_bm25_only.json
ablation_rerank_only.json
ablation_entity_only.json
```

## 11. Acceptance criteria

| Metric | Requirement |
|---|---|
| Hit@1 on `graph_hybrid` strategy | Strictly greater than post-de-bias baseline |
| Hit@1 on `lexical_exact` + `single_hop` types | Strictly greater than baseline |
| Hit@1 on any question type | No regression greater than 2 percentage points |
| End-to-end p50 latency | No regression greater than 10 % |
| LLM cost per query | Documented in PR description; not a hard gate |
| Static bias check (grep in CI) | Zero matches outside `metadata.rs` |
| `cargo fmt / check / clippy -D warnings / test` | All green |

## 12. Sequencing inside this sub-project

```
Step 1 · Regenerate eval/test_pdf_v1.json from current test.pdf
Step 2 · De-bias the five prompt-bearing files + delete the router comment
Step 3 · Add CI guard for forbidden domain-word list
Step 4 · Capture baseline eval (post-de-bias, no feature)
Step 5 · Implement WeightedKeywordExtractor + config + prompt
Step 6 · Implement weighted_bm25_query pure function + wire into search/keyword
Step 7 · Implement rerank adjustment in ranker.rs + wire into rerank flow
Step 8 · Implement entity_linker candidate prioritization
Step 9 · Wire extractor into router's parallel join
Step 10 · Unit + property + integration + doctests
Step 11 · Feature-on eval + ablation runs
Step 12 · PR with before/after metrics
```

Steps 1–4 are the de-bias half (committable as its own commit if desired,
but shipped in the same PR). Steps 5–11 are the weighted-keyword half.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM output quality varies across corpora | Prompt asks for information-content weighting based purely on term properties within the query, no corpus priors; empty output falls back safely |
| Added LLM call inflates eval cost | Latency hidden by parallel `join!`; cost per query documented in PR; `max_retries=2` caps worst-case cost |
| Reranker score distribution shifts enough to break downstream thresholds | `alpha=0.2` is small; acceptance gate catches latency + Hit@1 regressions per-type |
| BM25 repetition amplifies already-common terms | `min_weight` threshold drops noise; `max_repeats` caps amplification; BM25 TF saturation is sub-linear |
| Future contributor re-introduces domain bias in prompts | CI grep guard rejects commits reintroducing the forbidden word list |
| Per-sub-query keyword extraction later wants a different output shape | Forward-compat design (§5.3) — new method, same output type |

## 14. Open questions

None — all design points are resolved. Implementation-level naming and
exact default values (`min_weight`, `max_repeats`, `alpha`,
`entity_linker_top_k`) are deferred to the writing-plans phase and will
be selected there with reasoning recorded in the plan.
