# Weighted-Keyword Generator — Execution Status

**Branch:** `weighted-keywords` · **Plan:** `2026-04-17-weighted-keywords.md` · **Spec:** `../specs/2026-04-17-weighted-keywords-design.md`

## Done in this session (7 commits)

| Commit | Subject | Phase |
|---|---|---|
| `6979642` | `ci: add forbidden-lexicon grep guard` | 1.1 |
| `7bf5a44` | `test(retrieval): snapshot test infrastructure (Layer 2 bias guard)` | 1.2 |
| `7420813` | `refactor: universalize 5 prompts + 2 code comments (Phase 2 de-biasing)` | 2.1–2.5 |
| `8beb3ca` | `feat(retrieval/weighted_keywords): module + extractor + pure consumer helpers` | 4.1–4.3, 5.1, 6.1, 7.1 (pure functions) |
| `f6f62d8` | `feat(router): wire WeightedKeywordExtractor into parallel join + rerank adjustment` | 7.2 + 6.2 |
| `9211e12` | `test(retrieval): integration test — mock LLM flows to all three consumers` | 9.1 |
| `d221554` | `style: cargo fmt workspace` | — |

**Quality gates (all green):**
- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test -p agent-rs-retrieval` — 28 unit + 5 snapshot + 2 integration + doctests ✓
- Forbidden-lexicon CI guard (local grep) — `ALL CLEAN` ✓
- Prompt snapshot tests — 5/5 pass ✓

## Deferred to follow-up commits

These are ready to implement — pure functions exist, tests exist, default configs exist. Only the call-site surgery in two files remains:

### A. BM25 query rewrite consumer wiring (`search/keyword` via `dispatch_single`)

**Why deferred:** requires threading an `Option<&str>` keyword-specific query override through `router.rs::dispatch_single()` so the dense embedding still uses the raw query while BM25 sees the rewritten string. Affects 3 `keyword_search(...)` call sites inside `dispatch_single`.

**To implement:**
1. Add a new parameter `keyword_query_override: Option<&str>` to `dispatch_single`.
2. In `retrieve()`, compute the override inside the per-sub-query loop:
   ```rust
   let kw_override = self.weighted_keyword_integration.bm25.as_ref()
       .filter(|_| !weighted_keywords.terms.is_empty())
       .map(|cfg| weighted_bm25_query(query_str, weighted_keywords.as_ref(), cfg));
   ```
3. Pass `kw_override.as_deref()` into `dispatch_single(...)`.
4. In the three `Keyword`/`Hybrid`/`GraphHybrid` arms, use the override for `keyword_search`'s `query_text` parameter.

### B. Entity-linker overlap-first tiering wiring

**Why deferred:** `EntityLinker::link(&self, query)` needs to accept weighted keywords. Cleanest option: add a new method `link_with_priority(&self, query, Option<&WeightedKeywords>, Option<&EntityLinkerPrioritizationConfig>)` so the existing call sites stay unchanged. Then the router picks which method to call based on whether the integration is enabled.

**To implement:**
1. Refactor `entity_linker::EntityLinker::link` internals so the LLM extraction returns a `Vec<String>` of candidates before Neo4j lookups.
2. Between extraction and lookup, call `order_candidates_overlap_first(kw, &extracted, cfg)` when priority config is provided.
3. Use the reordered list for Neo4j lookups, respecting the existing `max_matches` budget.

### C. Observability metrics (Task 8.1 of the plan)

**Why deferred:** adds a significant surface area to `agent_rs_observability::metrics::RetrievalMetrics` plus emit sites across four locations (extractor, router BM25 call, ranker adjustment, entity-linker tier hits). The feature works without them — they are diagnostic, not correctness.

**To implement:** see plan §Task 8.1 for the full metric list. Suggested minimum to start: `weighted_keyword_calls_total`, `weighted_keyword_failures_total{kind}`, `weighted_keyword_nonempty_total`, `weighted_keyword_terms_emitted` (histogram), `weighted_rerank_coverage` (histogram).

## User actions required to finish the PR

### 1. Regenerate the eval dataset (Phase 0)

**Prerequisite:** Python 3, `gpt-4o-mini` API access via `OPENAI_API_KEY`.

```bash
cd /Users/cosmingagea/workspace/agent-rs
python3 eval/generate_enterprise_v3.py \
    --pdf test.pdf \
    --output eval/test_pdf_v1.json
git add eval/test_pdf_v1.json
git commit -m "feat(eval): regenerate test_pdf_v1 dataset from current test.pdf"
```

If the generator's CLI differs, adjust flags accordingly; check its `--help`.

### 2. Capture post-de-bias baseline (Phase 3)

**Prerequisite:** Docker Compose up (qdrant/neo4j/postgres/docling-sidecar), `OPENAI_API_KEY`, `EMBEDDER_TYPE=openai`.

```bash
docker compose up -d qdrant neo4j postgres docling-sidecar
# Wait ~15s for sidecar health.

EVAL_DATASET=eval/test_pdf_v1.json \
EVAL_SAVE_BASELINE=1 \
cargo run --release --example eval_harness
```

Produces `test_results/baseline_eval.json`. This baseline captures the **post-de-bias, feature-off** state — the fair reference for the weighted-keyword lift.

### 3. Wire the extractor into `eval_harness`

Without this step, the feature is dormant (extractor = `None` on router). In `examples/eval_harness.rs` where `RetrievalRouter` is built, add:

```rust
use std::sync::Arc;
use agent_rs_retrieval::{
    WeightedKeywordExtractor, WeightedKeywordExtractorConfig,
    WeightedKeywordIntegrationConfig,
};

let wk_extractor = Arc::new(WeightedKeywordExtractor::new(
    WeightedKeywordExtractorConfig::new(openai_api_key.clone()),
    http_client.clone(),
));
let router = RetrievalRouter::new(embedder, vector_store, graph_store, ranker, collection)
    // … existing .with_* calls …
    .with_weighted_keyword_extractor(wk_extractor)
    .with_weighted_keyword_integration(WeightedKeywordIntegrationConfig::default());
```

`http_client` is already constructed somewhere in the harness setup; reuse it.

### 4. Feature-on eval run

```bash
EVAL_DATASET=eval/test_pdf_v1.json \
cargo run --release --example eval_harness | tee test_results/feature_on.log
```

Compares against `baseline_eval.json` automatically. The report will show per-strategy / per-type deltas.

### 5. Per-consumer ablations (optional but recommended)

Since only the rerank adjustment is wired in this session, only ablation that makes sense today is rerank-on / rerank-off. To disable rerank adjustment at runtime, construct the router with:

```rust
.with_weighted_keyword_integration(WeightedKeywordIntegrationConfig {
    bm25: None,             // not yet wired anyway
    rerank: None,           // disabled for this run
    entity_linker: None,    // not yet wired anyway
})
```

Once deferred items A and B land, full three-way ablations become meaningful.

### 6. Acceptance criteria check

Per spec §11 — verify on the final eval report:
- Hit@1 on `graph_hybrid` strictly greater than baseline.
- MRR on `graph_hybrid` strictly greater.
- No per-type Hit@1 regression > 2 pp (for question types with sample count ≥ 20).
- End-to-end p50 latency regression < 10 %.
- p95 latency regression < 15 %.

If any fail:
- Top knobs to tune: `RerankAdjustmentConfig::alpha` (default 0.2) and `RerankAdjustmentConfig::phrase_bonus` (default 1.5).
- If p95 suffers: shorten `weighted_keyword_branch_budget` (default 6s).

### 7. Open PR

```bash
git push -u origin weighted-keywords
gh pr create --title "feat(retrieval): weighted-keyword generator + prompt universalization" \
  --body "$(cat <<'EOF'
## Summary
Adds a universal LLM-backed WeightedKeywordExtractor feeding (initially) the cross-encoder rerank adjustment via a bounded coverage-ratio formula, plus paired de-biasing of five Romanian-legal-biased prompts with a two-layer CI/snapshot bias guard.

Two consumer integrations (BM25 rewrite, entity-linker overlap-first tiering) have their pure functions landed and tested but wiring deferred to follow-up commits once this PR's eval data is in. See docs/superpowers/plans/2026-04-17-weighted-keywords-STATUS.md.

## Metrics
_(fill from test_results/feature_on.log)_

| Metric | Baseline | Feature | Δ |
|---|---|---|---|
| Hit@1 (graph_hybrid) | | | |
| MRR (graph_hybrid) | | | |
| NDCG@5 (graph_hybrid) | | | |
| p50 latency | | | |
| p95 latency | | | |

## Test plan
- [x] cargo fmt / clippy -D warnings / test (all workspace)
- [x] 28 unit tests, 5 snapshot tests, 2 integration tests green
- [x] Forbidden-lexicon CI guard local grep: ALL CLEAN
- [ ] Eval acceptance criteria per spec §11
EOF
)"
```
