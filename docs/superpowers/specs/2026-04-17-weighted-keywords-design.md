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
    /// - If non-empty, the maximum `weight` equals exactly `1.0` (linear
    ///   renormalization against the emitted max).
    /// - Sorted by `weight` descending. Ties preserve the LLM's emission
    ///   order (stable sort).
    /// - Deduplicated by `term` after NFKC + case folding; on collision
    ///   the higher-weight entry wins and `is_phrase` of the winner is
    ///   kept.
    /// - Every `term` is non-empty after normalization.
    pub terms: Vec<WeightedTerm>,
}

impl WeightedKeywords {
    /// The safe-fallback empty value. All consumers' empty-input paths are
    /// byte-identical to pre-feature behavior when they receive this.
    pub fn empty() -> Self { /* ... */ }

    /// Sum of all weights. Used by the rerank coverage ratio (§7.2).
    pub fn total_weight(&self) -> f32 { /* ... */ }

    /// First `k` entries (already sorted by weight desc).
    pub fn top_k(&self, k: usize) -> &[WeightedTerm] { /* ... */ }

    /// Convenience iterator; primarily exists to discourage consumers from
    /// indexing into `terms` directly.
    pub fn iter(&self) -> impl Iterator<Item = &WeightedTerm> { /* ... */ }
}

pub struct WeightedTerm {
    /// Surface form after NFKC normalization, trimmed, with surrounding
    /// punctuation removed. No stemming, no language-specific morphology.
    /// Guaranteed non-empty.
    pub term: String,
    /// Ordinal importance score in `(0.0, 1.0]`. **Not a calibrated
    /// probability, confidence, or corpus-aware relevance estimate** —
    /// only the ordering and relative magnitude are meaningful. Consumers
    /// must treat it as rank information, not as a scalar with absolute
    /// interpretation.
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
    pub rerank: Option<RerankAdjustmentConfig>,
    pub entity_linker: Option<EntityLinkerPrioritizationConfig>,
}

pub struct BM25WeightingConfig {
    /// Terms with `weight < min_weight` are dropped from the rewritten
    /// BM25 query string.
    pub min_weight: f32,
    /// Upper cap on per-term repetition in the rewritten BM25 query.
    /// Applied **after** token-level aggregation across term sources, so
    /// phrase expansion cannot accumulate above the cap for a single
    /// token.
    pub max_repeats: u32,
    /// Minimum Unicode grapheme-cluster count for a single-token term to
    /// be kept. Language-agnostic — drops short tokens by length alone,
    /// not by matching a stopword list. Default (picked in plan) is
    /// small (e.g., `2` or `3`). Phrase terms are exempt from this filter
    /// at the phrase level; it applies to tokens produced by phrase
    /// expansion.
    pub min_term_chars: usize,
    /// When `true`, phrase terms are emitted as both the full phrase
    /// (intact, once) and their component tokens (at reduced repetition).
    /// When `false`, only the full phrase is emitted.
    pub phrase_token_expansion: bool,
}

pub struct RerankAdjustmentConfig {
    /// Blend factor for the coverage-ratio adjustment (§7.2). Small by
    /// construction — the cross-encoder must remain the dominant signal.
    pub alpha: f32,
    /// Multiplier applied to phrase-term weights when computing both
    /// matched and total weight for the coverage ratio. Rewards phrase
    /// matches over equivalent-mass token matches. Default ≥ 1.0.
    pub phrase_bonus: f32,
}

pub struct EntityLinkerPrioritizationConfig {
    /// Max number of weighted terms considered as fallback probes when
    /// neither the overlap-first nor extracted-entity queues have
    /// exhausted `max_matches`.
    pub top_k: usize,
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

### 6.1 Timeout orchestration (router-level)

The weighted-keyword branch is **soft-optional**. Router orchestration
guarantees:

1. The extractor future is wrapped in a **branch-local timeout**
   (`router.weighted_keyword_branch_budget`, e.g. 6 s — concrete default
   chosen in the plan). This timeout lives at the router layer, **not**
   relying solely on the extractor's internal per-call timeout from
   `WeightedKeywordExtractorConfig`.
2. On branch-timeout expiry, the branch resolves to
   `Ok(WeightedKeywords::empty())` — it never propagates a timeout error
   that could force the other parallel branches to wait.
3. The weighted-keyword branch **never extends the router's overall
   deadline**. If the router's total budget has been exceeded by the
   time `tokio::join!` would otherwise still be waiting on this branch,
   the branch is cancelled and treated as empty.
4. Branch cancellation and branch-timeout hits are both recorded as
   `weighted_keyword_extractor_failures_total{kind="timeout"|"cancelled"}`
   metrics (see §9.3), so tail-latency regressions from this component
   are observable.

Consequence: when the extractor fails or times out (any cause, including
all retries exhausted), consumers receive `empty()` and the pipeline
continues unmodified — at no worse than today's p95/p99 latency.

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

**Algorithm (revised):**

1. If `kw.terms.is_empty()` → return `original_query.to_string()` unchanged.
2. Filter: drop terms whose `weight < cfg.min_weight`.
3. **Preserve phrases as phrases.** For each surviving term that is a
   phrase (`is_phrase == true`):
   - Emit the phrase **intact once** (contiguous, as a single
     space-separated group — surrounded by phrase markers if the
     backend supports them; otherwise as plain contiguous tokens).
   - If `cfg.phrase_token_expansion == true`, **additionally** emit
     each component token at **reduced** repetition
     (`ceil(weight * max_repeats / 2)`), subject to the
     `min_term_chars` filter below. When `false`, the phrase is
     emitted only as the intact form.
4. For single-token terms (`is_phrase == false`):
   - Drop if `token.graphemes().count() < cfg.min_term_chars`.
     Unicode grapheme clusters — language-agnostic — no stopword list.
   - Emit the token with multiplicity
     `max(1, ceil(weight * cfg.max_repeats))`.
5. **Token-level aggregation.** After steps 3–4, the same token may
   have been produced from multiple sources (e.g., a phrase expansion
   and a standalone single-token term). Aggregate per unique token:
   sum the repetition counts, **then** clamp the per-token total to
   `cfg.max_repeats`. This prevents amplification explosions from
   overlapping phrase-token expansions.
6. Join: intact phrases (from step 3) and aggregated single tokens
   (from step 5) with single spaces. Original query text is not
   included — the rewrite replaces it.

BM25's sub-linear TF saturation means the bounded repetition produces a
controlled boost that asymptotes. `max_repeats` caps the amplification
*per token* after aggregation, so phrase-token expansion cannot smuggle
in extra weight beyond the cap.

**Explicitly rejected alternative:** a hardcoded language-agnostic
stopword list (e.g., `the, and, of, in`). This would be English-only in
practice and reintroduces the same class of domain/language bias that
§8 is removing from the existing prompts. The combination of
information-content prompt + `min_weight` threshold + `min_term_chars`
filter handles function-word noise without a hardcoded list. A
corpus-derived stopword pass, if needed, is within the scope of
sub-project #3 (corpus-derived token bridger), not this one.

No tokenizer-sharing, no corpus priors, no stemming, no language
detection. Works with any BM25 backend Qdrant swaps in.

### 7.2 `ranker.rs` — post-rerank score adjustment

Cross-encoder rerankers (Jina, ZeroEntropy, etc.) are self-contained — they
do not accept weight hints on their API. Prepending weighted terms into the
query text risks confusing a model trained on natural-language queries.
Instead, we adjust the score **after** the cross-encoder returns it.

The adjustment lives in `ranker.rs` (post-rerank surface) — `reranker.rs`
stays a pure cross-encoder wrapper.

**Design — bounded coverage-ratio adjustment (revised from v1):**

The original substring-based additive boost had three structural flaws:

1. `contains()` matches across word boundaries (`art` inside `article`);
2. Summing over matched weights rewarded long / verbose candidates that
   happened to contain many terms "somewhere";
3. The signal was unbounded in principle — proportional to the sum of
   all emitted weights — making the adjustment distribution-sensitive
   across queries with different term counts.

The revised adjustment uses **boundary-aware matching** and a **coverage
ratio** that is naturally bounded in `[0, 1]`:

```rust
fn adjust_for_weighted_keywords(
    reranker_score: f32,
    candidate_text: &str,
    kw: &WeightedKeywords,
    cfg: &RerankAdjustmentConfig,
) -> f32 {
    if kw.terms.is_empty() || cfg.alpha == 0.0 {
        return reranker_score;
    }

    // 1. Normalize candidate once (NFKC + lowercase).
    let normalized = normalize_candidate(candidate_text);

    // 2. Per-term matching with boundary awareness:
    //    - single-token terms: Unicode-word-boundary match (`\b<term>\b`
    //      with \b defined over Unicode word characters, not ASCII).
    //    - phrase terms: exact contiguous token sequence match after
    //      NFKC + lowercase folding.
    let mut matched_weighted_mass = 0.0_f32;
    let mut total_weighted_mass = 0.0_f32;
    for t in &kw.terms {
        let bonus = if t.is_phrase { cfg.phrase_bonus } else { 1.0 };
        let mass = t.weight * bonus;
        total_weighted_mass += mass;
        if term_matches(&normalized, t) {
            matched_weighted_mass += mass;
        }
    }

    // 3. Coverage ratio — naturally bounded in [0, 1].
    let coverage = matched_weighted_mass / total_weighted_mass;

    reranker_score * (1.0 + cfg.alpha * coverage)
}
```

Properties:

- **Bounded:** `coverage ∈ [0, 1]`, so the multiplicative adjustment is
  bounded in `[1.0, 1.0 + alpha]`. Long verbose candidates cannot
  accumulate a runaway boost.
- **Coverage-aware:** matching three of five high-weight terms is
  rewarded more than matching one out of fifty low-weight terms, for
  the same total mass.
- **Phrase-preferring:** phrase terms contribute `weight * phrase_bonus`
  to both numerator and denominator; matching the phrase preserves the
  bonus in the numerator only if the phrase is actually matched as a
  contiguous sequence, which is strictly harder than matching its
  tokens individually. Phrase hits therefore earn disproportionately
  more coverage credit than the same tokens matched individually.
- **Boundary-aware:** `art` no longer matches `article`, `504` no
  longer matches `25041`, `AI` no longer matches random substrings.
- **Safe empty-path:** with `kw.terms.is_empty()` or `alpha == 0.0` the
  function is a byte-identical no-op.

Default `alpha = 0.2`, `phrase_bonus = 1.5` (concrete values selected in
the plan with reasoning recorded). The cross-encoder remains the
dominant signal by construction — at `alpha = 0.2` a fully-covered
adjustment is at most a 20 % multiplicative bump.

**Unicode word boundaries** use `regex::Regex` with `\b` in Unicode
mode (enabled by default in the `regex` crate). This handles Latin,
Cyrillic, Greek, Arabic, CJK, and mixed scripts uniformly — no
language-specific logic.

### 7.3 `entity_linker` — candidate prioritization (revised)

Today `entity_linker` LLM-extracts entity candidates from the query,
then looks them up in Neo4j. It treats every candidate equally.

**Problem with a naive top-K approach:** weighted keywords are not
necessarily entity-shaped. For a query like *"how to retry failed
webhook delivery after timeout"*, the top keywords might be `retry`,
`webhook delivery`, `timeout` — none of which is guaranteed to be a
graph entity. Blindly feeding them as Neo4j lookup candidates would
spend Neo4j budget on cold misses.

**Overlap-first precedence (revised design):**

The entity linker merges weighted keywords with extracted entity
candidates using a three-tier priority queue. Let:

- `E` = set of LLM-extracted entity candidates (today's source)
- `W` = set of weighted-keyword terms (NFKC + case-folded forms)

Lookup order:

1. **Intersection first (highest priority):** terms in `W ∩ E` — weighted
   keywords that are also LLM-identified as entity candidates. These
   are the highest-confidence lookups: the LLM flagged them as
   entities *and* they carry user-importance weight. Within this tier,
   order by descending weight.
2. **Extracted entities next:** entries in `E \ W` — LLM-identified
   entities not present in the weighted-keyword set. Today's behavior
   for these, preserved.
3. **Weighted-keyword fallback probes (lowest priority):** top
   `entity_linker.top_k` entries from `W \ E`, in weight order. These
   are "the user thinks these terms matter, try them as entity probes
   if we have budget left."

**Early-exit:** once `EntityLinkerConfig::max_matches` entities have
been resolved, remaining lookups in any tier are skipped. This
preserves the existing budget semantics — the overlap-first heuristic
only changes *which* lookups happen within that budget.

No change to the `entity_linker::link()` signature — only its internal
candidate-ordering logic. When weighted keywords are empty, the queue
collapses to tier 2 and behavior is byte-identical to today.

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

### 8.4 Forbidden-lexicon guard (two layers)

This is an enforceable tripwire against regression, **not** a
bias-proofing mechanism. Grep alone is easy to bypass with casing,
punctuation, unicode homoglyphs, or framings that don't contain any
listed token ("legal statute section" would still sail through). The
spec therefore pairs a coarse CI grep with a more robust
prompt-snapshot test.

**Layer 1 — CI grep (coarse tripwire).** One new CI step (and a
matching pre-commit hook entry) greps the `crates/` tree for a
forbidden-lexicon list:

```
penal | criminal | romanian | codul | articolul | pedepsit | Cod Penal
```

Matches in `crates/retrieval/src/ingestion/metadata.rs` are allowlisted
(the diacritic handling file). Any other match fails the build.

**Layer 2 — prompt snapshot tests (the real guard).** Canonical prompt
strings for each of the five prompt-bearing files (§8.1) are stored as
snapshot fixtures under `crates/retrieval/src/testdata/prompts/`. A
unit test:

1. Loads the prompt string at runtime from each module.
2. Normalizes it: NFKC, lowercase, punctuation stripped, Unicode
   homoglyph folding, whitespace collapsed.
3. Asserts the normalized form contains no entry from a **normalized
   forbidden lexicon** (the same list above, applied after the same
   normalization pipeline).
4. Also diff-compares against the stored snapshot and requires the
   diff to be reviewed and explicitly blessed on change.

The snapshot test catches:
- Casing / punctuation / homoglyph bypasses the grep misses.
- Silent prompt edits during code review (the diff surfaces them).

It does **not** catch semantic domain framing that uses none of the
forbidden tokens — that remains a human-review concern. The spec is
explicit that this guard is **"forbidden lexicon + snapshot diff",
not "bias proof"**.

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

Thin health metrics (call / failure / latency) are insufficient for
tuning a ranking component — when quality shifts, we need enough
distribution-level signal to localize the cause to extractor output,
thresholding, BM25 rewrite, rerank adjustment, or entity
prioritization. The metric set is therefore expanded.

**Traces.** `#[instrument(skip(self))]` on `extract`, fields
`%query_hash, terms_count, surviving_count, elapsed_ms`. Query is
hashed (blake3), never logged verbatim. Consumer-side adjustments add
fields `bm25_rewritten_tokens`, `rerank_coverage`,
`entity_priority_hits` on their respective spans.

**Extractor health.**
- `weighted_keyword_extractor_calls_total` (counter)
- `weighted_keyword_extractor_failures_total{kind}` (counter) —
  `kind ∈ {http_5xx, http_4xx, timeout, cancelled, rate_limit, malformed_json, schema_violation}`
- `weighted_keyword_extractor_latency_ms` (histogram)
- `weighted_keyword_extractor_nonempty_total` (counter) — paired with
  `calls_total` to derive a non-empty rate

**Extractor output distribution.** (Needed to tell whether the LLM is
emitting usable signal.)
- `weighted_keyword_terms_emitted` (histogram) — raw term count per
  call
- `weighted_keyword_terms_surviving_threshold` (histogram) — term
  count after `min_weight` + `min_term_chars` filters
- `weighted_keyword_phrase_ratio` (histogram) — phrase fraction of
  emitted terms

**BM25 consumer.** (Needed to tell whether query rewrite amplified
useful tokens vs noise.)
- `weighted_bm25_rewritten_tokens` (histogram) — output query token
  count after aggregation
- `weighted_bm25_unique_tokens` (histogram) — distinct tokens in
  output

**Rerank consumer.** (Needed to tell whether adjustment is moving the
distribution.)
- `weighted_rerank_coverage` (histogram) — the computed coverage
  ratio per candidate, bucketed
- `weighted_rerank_adjustment_ratio` (histogram) —
  `adjusted_score / reranker_score` per candidate

**Entity-linker consumer.** (Needed to tell whether overlap-first
precedence is actually finding overlaps.)
- `weighted_entity_tier_hits_total{tier}` (counter) —
  `tier ∈ {intersection, extracted, fallback}` — counts per-tier
  Neo4j lookups that resolved to an entity
- `weighted_entity_priority_hits_total` (counter) — convenience sum
  of `tier={intersection, fallback}`

Per-consumer structured-field logs when weights changed behavior:
`bm25_query_rewritten=true`, `rerank_adjusted=true`,
`entity_linker_prioritized=true`.

All metrics feed the existing OpenTelemetry pipeline — no new
collectors.

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

- `weighted_bm25_query` (revised algorithm, §7.1):
  - Empty keywords → output equals `original_query`.
  - Threshold filter drops terms below `min_weight`.
  - Length filter drops single-token terms shorter than `min_term_chars`;
    does **not** drop phrase terms at the phrase level.
  - Phrase preservation: with `phrase_token_expansion = false`, the
    phrase appears intact exactly once; with `true`, the phrase appears
    intact once and component tokens appear at reduced repetition.
  - Token-level aggregation: when a token appears from multiple term
    sources (e.g., a standalone single-token term and a phrase
    expansion), per-token total is aggregated, then clamped to
    `max_repeats` — never exceeds `max_repeats` for any unique token.
  - Rejected-alternative regression test: asserts the function does
    **not** drop common English function words (`the`, `and`, `of`,
    `in`) by any hardcoded list — only by threshold and length.
- `adjust_for_weighted_keywords` (revised algorithm, §7.2):
  - `alpha == 0` → returns `reranker_score` unchanged.
  - Empty keywords → returns `reranker_score` unchanged.
  - **Bounded adjustment:** output is always in
    `[reranker_score, reranker_score * (1.0 + alpha)]`.
  - **Boundary-aware matching:** query term `"art"` does **not** match
    candidate text `"article"`; term `"504"` does **not** match
    candidate text `"25041"`; term `"AI"` does **not** match
    candidate text `"brains"`.
  - **Coverage ratio:** for a candidate matching all terms, the
    adjustment equals `reranker_score * (1.0 + alpha)`; for a
    candidate matching none, the adjustment equals `reranker_score`.
  - **Phrase bonus:** matching a phrase earns strictly more coverage
    credit than matching each of its tokens individually (when
    `phrase_bonus > 1.0`).
  - **Monotonicity in coverage:** if candidate A matches a strict
    superset of the terms candidate B matches, A's adjusted score is
    ≥ B's adjusted score (at equal base `reranker_score`).
- Entity-linker overlap-first precedence (§7.3):
  - Intersection-tier entries come first, ordered by weight
    descending.
  - Extracted-entity tier next, preserving original ordering.
  - Fallback tier entries come last, capped at `top_k`.
  - When weighted keywords are empty, queue collapses to tier 2
    only and ordering matches today's behavior byte-identically.

### 10.3 Property tests (`proptest`)

On `weighted_bm25_query`:

- **Idempotency convergence:** repeatedly applying the function with
  the previous output as the new `original_query` and the same
  keywords converges to a fixpoint within 2 iterations.
- **Per-token cap:** no single unique token appears more than
  `max_repeats` times in the output, regardless of how many term
  sources produced it.
- **Weight-rank preservation (for non-phrase terms):** for two
  single-token terms *a* and *b* with `weight_a > weight_b` that both
  pass the filters, the count of *a* in the output is ≥ count of *b*.
- **Empty-input invariant:** if `kw.terms` is empty after filtering,
  output equals `original_query`.

On `adjust_for_weighted_keywords`:

- **Bounded adjustment invariant:** for any candidate text and any
  non-negative `alpha`, the output is in
  `[reranker_score, reranker_score * (1.0 + alpha)]`.
- **Coverage monotonicity:** adding a matched term to a candidate
  (shadowing one that wasn't matched) cannot decrease the adjusted
  score.
- **Empty-keywords no-op:** for any `cfg`, empty `kw` produces output
  equal to `reranker_score`.

### 10.4 Integration tests (`crates/retrieval/tests/weighted_keywords_integration.rs`)

- Mock LLM returns canned `WeightedKeywords`; assert the same value is
  visible to all three consumers and actually modifies their behavior
  (observable via structured log fields or injected probes).
- Mock LLM fails (exhaust all retries); assert the router still produces
  a valid `EvidencePack` and consumers behaved as today.
- Mock LLM times out; assert the `tokio::join!` completes within the
  router's overall budget.

### 10.5 Doctests

Every public item carries a runnable `/// # Examples` block per the
project Rust style guide:
`WeightedKeywordExtractor`, `WeightedKeywordExtractorConfig`,
`WeightedKeywords` (including `empty()`, `total_weight()`, `top_k()`,
`iter()`), `WeightedTerm`, `WeightedKeywordIntegrationConfig`,
`BM25WeightingConfig`, `RerankAdjustmentConfig`,
`EntityLinkerPrioritizationConfig`, `weighted_bm25_query`,
`adjust_for_weighted_keywords`.

### 10.6 End-to-end evaluation

**Step 1 — regenerate eval dataset from the current `test.pdf`:**

```bash
python eval/generate_enterprise_v3.py \
    --pdf test.pdf \
    --output eval/test_pdf_v1.json
```

Produces a dataset with ~200 questions stratified across the existing
10 question types and 3 difficulty tiers. Committed alongside the
spec for reproducibility. The generator is domain-agnostic by
construction.

**Step 2 — baseline capture** (post-de-bias, no feature):

```bash
EVAL_DATASET=eval/test_pdf_v1.json \
EVAL_SAVE_BASELINE=1 \
cargo run --release --example eval_harness
```

**Step 3 — feature-on run:**

```bash
EVAL_DATASET=eval/test_pdf_v1.json \
cargo run --release --example eval_harness
```

**Step 4 — ablation runs**, each toggling one consumer while disabling
the other two via `WeightedKeywordIntegrationConfig`:

```
ablation_bm25_only.json
ablation_rerank_only.json
ablation_entity_only.json
```

### 10.7 Metrics computed and compared

Hit@1 alone is too thin — this feature touches query-side recall
(BM25 rewrite, entity prioritization) *and* rerank ordering. We
cannot localize gains without separating the two surfaces.

Per strategy × question-type × difficulty tier, the harness computes
and reports:

| Metric | Surface | What it tells us |
|---|---|---|
| **Hit@1**, **Hit@5** | post-rerank | primary top-rank quality |
| **MRR** | post-rerank | rank position of the first correct result |
| **NDCG@5** | post-rerank | graded ranking quality (the reranker's main turf) |
| **Recall@5 (pre-rerank)** | fusion output | candidate-pool quality before rerank; isolates BM25 rewrite + entity prioritization lift |
| **Recall@5 (post-rerank)** | final | candidate pool after rerank adjustment |

Separating pre-rerank Recall@5 from post-rerank NDCG@5 is the key
diagnostic: it tells us whether lift came from *getting better
candidates in* (BM25 + entity) or from *reordering the same pool*
(rerank adjustment) — or both, or either but not the other.

### 10.8 Statistical rigor of per-type gates

"No regression greater than 2 percentage points on any question type"
is statistically noisy for types with small sample counts. The spec
therefore:

- **Reports sample counts per question type and difficulty tier** in
  the eval harness output.
- **Computes bootstrap 95% confidence intervals** for each metric
  (1000 resamples, stratified by question type) and prints them
  alongside point estimates.
- **Per-type regression gate:** for question types with sample count
  ≥ 20, the `> 2 pp` regression gate in §11 applies as a point
  comparison. For smaller samples, a "regression" requires the
  baseline-vs-feature-on 95% CIs to be **non-overlapping in the
  direction of degradation**. This avoids blocking merges on noisy
  single-digit-sample slices.

### 10.9 Cross-corpus generalization — acknowledged gap

A single-corpus eval (even one freshly regenerated from a
non-legal `test.pdf`) cannot prove universality. The extractor or
the de-biased prompts could overfit stylistic properties of
whatever corpus is in `test.pdf` today.

For this sub-project, a single corpus is accepted as the scope —
adding a second corpus belongs to the follow-up work listed in §13.
The spec documents this explicitly rather than pretending the
generalization claim is settled.

## 11. Acceptance criteria

Point estimates below are compared against the post-de-bias baseline
(§10.6 Step 2). Per-type regression gates follow the sample-size rule
in §10.8 (point comparison for n ≥ 20, non-overlapping 95 % CIs for
n < 20).

| Metric | Surface | Requirement |
|---|---|---|
| Hit@1 (`graph_hybrid`) | post-rerank | Strictly greater than baseline |
| MRR (`graph_hybrid`) | post-rerank | Strictly greater than baseline |
| Recall@5 pre-rerank (`graph_hybrid`) | fusion | Strictly greater than baseline — confirms BM25 rewrite + entity prioritization are the query-side driver |
| NDCG@5 (`graph_hybrid`) | post-rerank | Strictly greater than baseline — confirms rerank adjustment is helping, not hurting |
| Hit@1 on `lexical_exact` + `single_hop` types | post-rerank | Strictly greater than baseline |
| Any per-type Hit@1 regression | post-rerank | No regression per §10.8 rule |
| End-to-end p50 latency | router | No regression greater than 10 % |
| End-to-end p95 latency | router | No regression greater than 15 % (the branch-local timeout of §6.1 is the primary defense) |
| LLM cost per query | — | Documented in PR description; not a hard gate |
| Forbidden-lexicon guard (grep) | CI | Zero matches outside `metadata.rs` |
| Prompt snapshot tests (§8.4 Layer 2) | CI | All green |
| `cargo fmt / check / clippy -D warnings / test` | CI | All green |

## 12. Sequencing inside this sub-project

```
Step 1 · Regenerate eval/test_pdf_v1.json from current test.pdf
Step 2 · De-bias the five prompt-bearing files + delete the router comment
Step 3 · Add forbidden-lexicon guard — Layer 1 (CI grep) + Layer 2
         (prompt snapshot tests with NFKC/fold normalization)
Step 4 · Capture baseline eval (post-de-bias, no feature);
         baseline includes Hit@1/Hit@5/MRR/NDCG@5 + pre- and
         post-rerank Recall@5, with sample counts + bootstrap CIs
Step 5 · Implement WeightedKeywordExtractor + config + universal prompt
Step 6 · Implement weighted_bm25_query (revised algorithm, §7.1)
         + wire into search/keyword
Step 7 · Implement adjust_for_weighted_keywords (bounded coverage-ratio,
         §7.2) in ranker.rs + wire into rerank flow
Step 8 · Implement overlap-first entity-linker prioritization (§7.3)
Step 9 · Wire extractor into router's parallel join with
         branch-local timeout (§6.1); cancellation/timeout metrics
Step 10 · Unit + property + integration + doctests (§10.1–10.5)
Step 11 · Feature-on eval + per-consumer ablation runs; emit metrics
          distributions (§9.3) during ablation for tunability
Step 12 · PR with before/after metric tables (all metrics from §10.7),
          sample counts, bootstrap CIs, and acceptance-criteria
          checklist (§11) filled in
```

Steps 1–4 are the de-bias half (committable as its own commit if desired,
but shipped in the same PR). Steps 5–11 are the weighted-keyword half.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM output quality varies across corpora | Prompt asks for information-content weighting based purely on term properties within the query, no corpus priors; empty output falls back safely |
| Added LLM call inflates eval cost | Latency hidden by parallel `join!` + branch-local timeout (§6.1); cost per query documented in PR; `max_retries=2` caps worst-case cost |
| Weighted-keyword branch is the slowest and blows router p95 | Branch-local timeout at the router layer (§6.1) forces the branch to resolve to `empty()` independent of the extractor's own per-call timeout |
| Rerank coverage ratio misfires on score distributions with narrow dynamic range | `alpha=0.2` is small; coverage is bounded in `[0,1]`; §10.7 splits NDCG@5 (post-rerank) from Recall@5 (pre-rerank) so adjustment regressions are localizable |
| BM25 phrase-token expansion amplifies common short tokens | `min_term_chars` universal length filter + token-level aggregation with `max_repeats` cap applied *after* aggregation |
| Future contributor re-introduces domain bias in prompts | Forbidden-lexicon grep guard (Layer 1) + prompt snapshot tests with normalization (Layer 2); §8.4 documents the limitations of both honestly |
| Per-sub-query keyword extraction later wants a different output shape | Forward-compat design (§5.3) — new method, same output type |
| **Single-corpus eval overfits to whatever is in the current `test.pdf`** | Acknowledged in §10.9. Follow-up work: add a second eval corpus (small is fine) in a subsequent PR, ideally before sub-project #2 lands. Not a blocker here, but tracked. |
| Substring-free word-boundary matching regresses on scripts without explicit word separators (e.g., Thai, some CJK contexts) | Unicode-mode `\b` in the `regex` crate delegates to Unicode properties; residual script-specific issues, if they appear in eval, are captured by per-type Hit@1 and addressed via prompt iteration rather than adding language-specific logic |

## 14. Open questions

These are genuinely unresolved items whose answers belong to the
implementation plan, not to the spec. Listed honestly rather than
hidden as "None":

1. **Phrase preservation vs token expansion default.** The spec allows
   both (`phrase_token_expansion` flag). The plan must pick the
   default — `true` (preserve + expand at half repetition) or `false`
   (preserve only) — and justify it empirically after a sanity check
   on a few queries from the new `test.pdf`.
2. **`min_term_chars` default.** The spec uses a universal length
   filter but doesn't fix the threshold. Candidates to consider in
   the plan: `2` (aggressive — drops many single-letter tokens but
   may drop short numeric IDs), `3` (moderate), `4` (conservative,
   preserves most short IDs and common content words).
3. **Rerank `alpha` and `phrase_bonus` defaults.** Spec fixes the
   shape (bounded coverage ratio with phrase bonus) but not the
   exact numeric defaults. The plan selects them; a sweep over a
   small grid on the new `test.pdf` is appropriate.
4. **Entity-linker `top_k` default.** Spec picks overlap-first
   precedence; `top_k` governs how many fallback probes flow into
   tier 3. Small default (3–5) is appropriate; plan picks the
   concrete value.
5. **Router branch-local timeout for the weighted-keyword branch.**
   Spec requires this timeout but does not fix its value. Must be
   short enough to preserve p95, long enough to let typical LLM
   calls complete (e.g., 4–8 s). Plan picks based on measured LLM
   latency distribution on the existing extractors.
6. **Whether the CI lexicon grep runs on all `crates/` or a
   narrower allowlist of prompt-bearing files.** The spec as written
   runs on all `crates/` with a `metadata.rs` allowlist. The plan
   may narrow this if false-positive rate from legitimate code
   comments in unrelated modules proves too high in practice.

None of these affects the architecture, public types, or acceptance
criteria. They are concrete defaults to be picked with reasoning in
the plan phase and recorded alongside the plan.
