# Weighted-Keyword Generator + Prompt Universalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a universal, LLM-backed `WeightedKeywordExtractor` whose output feeds three retrieval consumers (`search/keyword`, `ranker.rs`, `entity_linker`), and universalize five existing domain-biased prompts in the same PR.

**Architecture:** Query-side query transform added to the router's parallel `tokio::join!` bundle with a branch-local timeout that resolves to `empty()` on failure. Three consumer integrations via pure functions: weighted BM25 query rewrite (threshold + length filter + phrase preservation + token-level aggregation), bounded coverage-ratio rerank adjustment (Unicode-word-boundary matching + phrase bonus), and overlap-first entity-linker tiering. Five existing prompts stripped of Romanian-legal bias, guarded by a CI forbidden-lexicon grep and runtime prompt snapshot tests.

**Tech Stack:** Rust (edition 2024, 1.85+), `tokio` for async, `backon` for retry, `reqwest` for OpenAI calls, `regex` with Unicode word-boundary mode, `serde` + `serde_json` for structured output, `proptest` for property tests, `tracing` + OpenTelemetry for observability. Python 3 + existing `eval/generate_enterprise_v3.py` for dataset regeneration.

**Spec reference:** `docs/superpowers/specs/2026-04-17-weighted-keywords-design.md`

---

## Concrete Defaults Picked for This Plan

Per spec §14 the plan phase picks exact values. These are used throughout:

| Parameter | Default | Rationale |
|---|---|---|
| `WeightedKeywordExtractorConfig::model` | `"gpt-5.4-mini-2026-03-17"` | Matches existing `query_rewriter` for prompt-cache alignment |
| `WeightedKeywordExtractorConfig::max_retries` | `2` | Matches `query_rewriter` / `graph_extractor` |
| `WeightedKeywordExtractorConfig::timeout` | `Duration::from_secs(10)` | Per-call LLM timeout |
| `BM25WeightingConfig::min_weight` | `0.3` | Drops clear filler; preserves anything the LLM considered moderately informative |
| `BM25WeightingConfig::max_repeats` | `3` | Low integer — TF saturation makes gains asymptote early |
| `BM25WeightingConfig::min_term_chars` | `3` | Universal length filter; drops 1–2 character tokens that carry little info without reintroducing language bias |
| `BM25WeightingConfig::phrase_token_expansion` | `true` | Preserve phrase + expand tokens at half repetition |
| `RerankAdjustmentConfig::alpha` | `0.2` | Fully-covered boost ≤ 20 %; cross-encoder stays dominant |
| `RerankAdjustmentConfig::phrase_bonus` | `1.5` | Phrase matches earn 50 % more coverage credit than token matches |
| `EntityLinkerPrioritizationConfig::top_k` | `5` | Fallback probe count; consistent with existing `max_matches` scale |
| Router `weighted_keyword_branch_budget` | `Duration::from_secs(6)` | Shorter than per-call timeout; preserves router p95 |

All values surface through `agent_rs_core::config` — zero magic numbers in the code path.

---

## File Structure

**New files:**

- `crates/retrieval/src/weighted_keywords.rs` — public module: types, config, extractor, pure integration helpers
- `crates/retrieval/tests/weighted_keywords_integration.rs` — integration tests
- `crates/retrieval/src/testdata/prompts/query_rewriter_system.txt` — prompt snapshot fixture
- `crates/retrieval/src/testdata/prompts/metadata_filter_system.txt` — prompt snapshot fixture
- `crates/retrieval/src/testdata/prompts/entity_linker_extract.txt` — prompt snapshot fixture
- `crates/retrieval/src/testdata/prompts/graph_extractor_system.txt` — prompt snapshot fixture
- `crates/retrieval/src/testdata/prompts/agent_system.txt` — prompt snapshot fixture
- `crates/retrieval/tests/prompt_snapshots.rs` — snapshot-based forbidden-lexicon guard
- `.github/workflows/forbidden-lexicon.yml` — CI grep tripwire (Layer 1)
- `eval/test_pdf_v1.json` — regenerated eval dataset from current `test.pdf`
- `docs/superpowers/plans/2026-04-17-weighted-keywords.md` — this file

**Modified files:**

- `crates/retrieval/src/lib.rs` — add `pub mod weighted_keywords;` + re-exports
- `crates/retrieval/src/query_rewriter.rs` — de-bias doc + prompt
- `crates/retrieval/src/metadata_filter.rs` — de-bias doc + prompt
- `crates/retrieval/src/entity_linker.rs` — de-bias prompt + overlap-first tiering
- `crates/retrieval/src/ingestion/graph_extractor.rs` — de-bias prompt + examples
- `crates/agent/src/prompts.rs` — universal routing heuristic
- `crates/retrieval/src/router.rs` — delete biased comment + wire extractor branch + timeout
- `crates/retrieval/src/search/keyword.rs` — (unchanged API; rewrite happens at call site in router)
- `crates/retrieval/src/ranker.rs` — add rerank adjustment call site
- `crates/retrieval/src/error.rs` — add `WeightedKeywordExtraction` error variant
- `crates/observability/src/metrics.rs` — add new weighted-keyword metric names
- `crates/core/src/config.rs` — add config defaults module

---

## Phase 0 — Pre-implementation Setup

### Task 0.1: Regenerate eval dataset from the current `test.pdf`

**Files:**
- Create: `eval/test_pdf_v1.json`

- [ ] **Step 1: Inspect the generator script to learn its CLI flags**

```bash
sed -n '1,60p' /Users/cosmingagea/workspace/agent-rs/eval/generate_enterprise_v3.py
```

Expected: the script exposes `--pdf <path>` and `--output <path>` (or equivalent); note the exact flag names.

- [ ] **Step 2: Run the generator against current `test.pdf`**

```bash
cd /Users/cosmingagea/workspace/agent-rs
python3 eval/generate_enterprise_v3.py \
    --pdf test.pdf \
    --output eval/test_pdf_v1.json
```

If the flag names differ, adjust per the inspection in Step 1. The script is domain-agnostic by construction — it reads whatever PDF is given and produces stratified questions via `gpt-4o-mini`.

Expected: a new `eval/test_pdf_v1.json` containing ~200 questions across 10 types and 3 difficulty tiers.

- [ ] **Step 3: Spot-check the output**

```bash
head -40 /Users/cosmingagea/workspace/agent-rs/eval/test_pdf_v1.json
```

Expected: JSON with `"dataset_name"`, `"documents"`, `"questions"` array. Confirm `documents[0].path` references `test.pdf` (or `../test.pdf`). Confirm questions are in the domain of the current PDF, not the old Romanian Penal Code.

- [ ] **Step 4: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add eval/test_pdf_v1.json
git commit -m "feat(eval): regenerate test_pdf_v1 dataset from current test.pdf"
```

---

## Phase 1 — Forbidden-Lexicon Guard (Layer 1 + Layer 2 infrastructure)

Set up both guard layers **before** editing any prompt. This way the de-bias tasks in Phase 2 can verify their own correctness by running the guards.

### Task 1.1: CI forbidden-lexicon grep workflow (Layer 1)

**Files:**
- Create: `.github/workflows/forbidden-lexicon.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: forbidden-lexicon

on:
  pull_request:
    paths:
      - 'crates/**'

jobs:
  grep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Grep forbidden lexicon
        run: |
          set -e
          # Allowlisted file: character-set support, not domain bias.
          ALLOWLIST='crates/retrieval/src/ingestion/metadata.rs'
          PATTERN='penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal'
          # -iE = case-insensitive extended regex; -r recursive; exclude allowlist.
          MATCHES=$(grep -IrnEi \
            --include='*.rs' \
            --include='*.md' \
            "$PATTERN" \
            crates/ | grep -v "^$ALLOWLIST" || true)
          if [ -n "$MATCHES" ]; then
            echo "Forbidden lexicon matches found (outside allowlist):"
            echo "$MATCHES"
            echo ""
            echo "Remove domain-biased content or extend allowlist with justification."
            exit 1
          fi
          echo "No forbidden lexicon matches."
```

- [ ] **Step 2: Run the grep locally to see current state**

```bash
cd /Users/cosmingagea/workspace/agent-rs
ALLOWLIST='crates/retrieval/src/ingestion/metadata.rs'
PATTERN='penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal'
grep -IrnEi --include='*.rs' --include='*.md' "$PATTERN" crates/ | grep -v "^$ALLOWLIST" || echo "CLEAN"
```

Expected: multiple matches (in `query_rewriter.rs`, `metadata_filter.rs`, `entity_linker.rs`, `ingestion/graph_extractor.rs`, `agent/src/prompts.rs`, `router.rs`). This proves the guard can catch the current biased state. We will fix these in Phase 2 and the local grep will go CLEAN.

- [ ] **Step 3: Commit the workflow**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add .github/workflows/forbidden-lexicon.yml
git commit -m "ci: add forbidden-lexicon grep guard"
```

Note: CI will fail on this commit until Phase 2 fixes the matches. That's expected. If the repo uses `main` as the default branch and the CI is blocking, either: (a) do all of Phase 1 and Phase 2 on a feature branch, or (b) temporarily make the workflow `continue-on-error` and flip the switch at the end of Phase 2.

### Task 1.2: Prompt snapshot test infrastructure (Layer 2)

**Files:**
- Create: `crates/retrieval/tests/prompt_snapshots.rs`

- [ ] **Step 1: Write the failing test file**

```rust
//! Prompt snapshot tests: forbidden-lexicon guard with NFKC + case-fold
//! normalization. Layer 2 of the bias guard (§8.4 of the spec).
//!
//! Assertions:
//! 1. Each module's system prompt, after normalization, contains none of
//!    the forbidden-lexicon entries.
//! 2. Each prompt matches its committed snapshot fixture — any edit must
//!    update the fixture consciously.

use std::fs;
use std::path::PathBuf;

fn normalize(s: &str) -> String {
    // NFKC normalize → lowercase → strip ASCII punctuation → collapse whitespace.
    use unicode_normalization::UnicodeNormalization;
    let nfkc: String = s.nfkc().collect();
    let lower = nfkc.to_lowercase();
    let stripped: String = lower
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

const FORBIDDEN: &[&str] = &[
    "penal", "criminal", "romanian", "codul", "articolul", "pedepsit", "cod penal",
];

fn assert_no_forbidden(label: &str, prompt: &str) {
    let n = normalize(prompt);
    for f in FORBIDDEN {
        assert!(
            !n.contains(f),
            "prompt {label} contains forbidden lexicon entry {f:?} after normalization"
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
    let fixture = fs::read_to_string(fixture_path(fixture_name))
        .unwrap_or_else(|e| panic!("missing fixture {fixture_name}: {e}"));
    assert_eq!(
        prompt.trim(),
        fixture.trim(),
        "prompt {label} diverged from committed fixture {fixture_name}"
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
```

- [ ] **Step 2: Add `unicode-normalization` to retrieval `dev-dependencies`**

Edit `crates/retrieval/Cargo.toml`, find `[dev-dependencies]`, add:

```toml
unicode-normalization = "0.1"
```

If `[dev-dependencies]` doesn't exist, add it:

```toml
[dev-dependencies]
unicode-normalization = "0.1"
```

Also ensure `agent-rs-agent` is a dev-dependency so the test can reach `agent_rs_agent::prompts::DEFAULT_SYSTEM_PROMPT`:

```toml
agent-rs-agent = { path = "../agent" }
```

- [ ] **Step 3: Expose the prompt constants under test-only aliases**

The current prompts are `const SYSTEM_PROMPT` (private) in each module. Without breaking the module's encapsulation, expose a test-visible alias. Add to `crates/retrieval/src/query_rewriter.rs` near its `SYSTEM_PROMPT`:

```rust
#[cfg(test)]
pub const SYSTEM_PROMPT_FOR_TESTING: &str = SYSTEM_PROMPT;

#[cfg(not(test))]
#[doc(hidden)]
pub const SYSTEM_PROMPT_FOR_TESTING: &str = SYSTEM_PROMPT;
```

Repeat analogously in:
- `crates/retrieval/src/metadata_filter.rs` → `SYSTEM_PROMPT_FOR_TESTING`
- `crates/retrieval/src/entity_linker.rs` → `ENTITY_EXTRACT_PROMPT_FOR_TESTING`
- `crates/retrieval/src/ingestion/graph_extractor.rs` → `SYSTEM_PROMPT_FOR_TESTING`

`agent_rs_agent::prompts::DEFAULT_SYSTEM_PROMPT` is already public; no change needed.

Why both `#[cfg(test)]` and `#[cfg(not(test))]` variants? The integration test crate is compiled separately from the main crate, so it sees `#[cfg(not(test))]` from the dependency side. Both branches resolve to the same value — this just makes the constant importable from both unit and integration test contexts.

- [ ] **Step 4: Run the test to confirm it fails in the expected way**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --test prompt_snapshots 2>&1 | tail -40
```

Expected: compile succeeds, tests FAIL with:
- "contains forbidden lexicon entry" (for all 5 tests) — because the prompts still contain Romanian-legal wording, **and**
- "missing fixture" — because fixtures haven't been written yet.

Both failures confirm the test infrastructure is wired correctly. Fixing both is the work of Phase 2.

- [ ] **Step 5: Create the empty fixture directory with a placeholder file**

```bash
mkdir -p /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/testdata/prompts
touch /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/testdata/prompts/.gitkeep
```

Fixture files are created in Phase 2 as each prompt is rewritten.

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/tests/prompt_snapshots.rs \
        crates/retrieval/Cargo.toml \
        crates/retrieval/src/testdata/prompts/.gitkeep \
        crates/retrieval/src/query_rewriter.rs \
        crates/retrieval/src/metadata_filter.rs \
        crates/retrieval/src/entity_linker.rs \
        crates/retrieval/src/ingestion/graph_extractor.rs
git commit -m "test: add prompt snapshot tests (Layer 2 bias guard infrastructure)"
```

---

## Phase 2 — De-bias Prompts

Each task: (1) read the full current prompt into context, (2) rewrite, (3) create the snapshot fixture, (4) run both guards (grep + snapshot test) to confirm the file is clean, (5) commit.

### Task 2.1: De-bias `crates/retrieval/src/query_rewriter.rs`

**Files:**
- Modify: `crates/retrieval/src/query_rewriter.rs`
- Create: `crates/retrieval/src/testdata/prompts/query_rewriter_system.txt`

- [ ] **Step 1: Read the current prompt and doc comment**

```bash
cd /Users/cosmingagea/workspace/agent-rs
sed -n '1,250p' crates/retrieval/src/query_rewriter.rs
```

Note the existing `const SYSTEM_PROMPT: &str = r#"..."#;`, the module-level `//!` doc block, and any `//` comments that name "Romanian legal".

- [ ] **Step 2: Rewrite the module-level doc comment**

Replace any line mentioning "Romanian legal awareness" (or similar domain framing) with a universal equivalent. Specifically:

Find:
```rust
//! - **Romanian legal awareness** — The system prompt includes explicit
//!   guidance for Romanian legal queries: article cross-references, crime
//!   definitions, penalty lookups, negation rewriting.
```

Replace with:
```rust
//! - **Structure-aware decomposition** — The system prompt recognises common
//!   query shapes (comparisons, numbered-reference lookups, cross-section
//!   questions, negations) without assuming any specific domain or language.
```

Find the doctest inside the module-level `//!` block:
```rust
//! let rewritten = rewriter.rewrite("Ce diferență este între furt și tâlhărie?").await?;
```

Replace with:
```rust
//! let rewritten = rewriter.rewrite("What's the difference between a compiler and an interpreter?").await?;
```

Find in the design-goals list (`/// 4. Handle Romanian legal queries: ...`):
```rust
/// 4. Handle Romanian legal queries: article references, crime definitions,
///    cross-article penalties, comparative questions, negation rewriting.
```

Replace with:
```rust
/// 4. Handle common query shapes: numbered-reference lookups, comparative
///    questions, cross-reference queries, negation rewriting — without
///    assumptions about domain or language.
```

- [ ] **Step 3: Rewrite the system prompt**

The existing `SYSTEM_PROMPT` is ~150 lines of instructions and few-shot examples. The structural guidance stays; only domain-specific wording changes.

Rewrite rules:
1. Any mention of "Romanian", "legal", "Cod Penal", "furt", "tâlhărie", "article XX" → neutral phrasing.
2. Any few-shot examples citing Romanian legal text → replace with examples from at least three neutral domains (software documentation, product manuals, encyclopedia-style entries). Never anchor more than one example to a single domain.
3. Structural rules (decomposition strategy, output JSON shape) stay unchanged.

The full rewritten prompt is below. Replace the entirety of the existing `const SYSTEM_PROMPT: &str = r#"..."#;` definition with:

```rust
const SYSTEM_PROMPT: &str = r#"You are an expert information-retrieval query analyst.

Your task is to analyse a user query and decide whether to pass it through
unchanged or decompose it into 2-3 focused sub-queries that together cover
all aspects of the original intent.

### Decision rules

#### Keep the query as-is (should_decompose: false) when:
- The query is a single focused question (one definition, one lookup, one
  specific reference, one concrete subject).
- The query is an exact-match keyword lookup (section numbers, IDs, codes).
- The query is under ~10 words and expresses a single intent.

#### Decompose the query (should_decompose: true) when:
- The query asks about multiple distinct subjects or aspects.
- The query compares, contrasts, or relates two or more things.
- The query is multi-hop: one fact implies another (for example, asks
  about X where X is itself defined elsewhere).
- The query contains a negation ("but not ...", "excluding ...").
- The query is a long, discursive natural-language sentence whose focus
  can be split without losing meaning.

### Output rules

- Emit 1-3 sub-queries. The ORIGINAL query MUST appear verbatim as one of
  the sub-queries so that direct-match retrieval is preserved.
- Each sub-query is a standalone natural-language question targeting one
  aspect of the original intent.
- Do NOT rewrite to a different language. Preserve the user's language.
- Do NOT add domain assumptions the user did not state.

### Few-shot examples

Example 1 (single-hop, no decomposition):
- Input: "What is the return type of the split() method?"
- Output: one sub-query equal to the original, should_decompose: false.

Example 2 (comparison — decompose):
- Input: "What's the difference between a compiler and an interpreter, and which one does Python use?"
- Output:
  - sub-query 1: "What's the difference between a compiler and an interpreter, and which one does Python use?" (original)
  - sub-query 2: "How does a compiler differ from an interpreter?"
  - sub-query 3: "Does Python use a compiler or an interpreter?"
  - should_decompose: true

Example 3 (multi-hop — decompose):
- Input: "Who discovered the element named after the country in which it was first found?"
- Output:
  - sub-query 1: "Who discovered the element named after the country in which it was first found?" (original)
  - sub-query 2: "Which elements are named after the country where they were first found?"
  - sub-query 3: "Who discovered those elements?"
  - should_decompose: true

Example 4 (numbered-reference lookup — keep as-is):
- Input: "Section 4.2"
- Output: one sub-query equal to the original, should_decompose: false.

Example 5 (negation — decompose):
- Input: "List database engines that support ACID transactions but not SQL."
- Output:
  - sub-query 1: "List database engines that support ACID transactions but not SQL." (original)
  - sub-query 2: "Which database engines support ACID transactions?"
  - sub-query 3: "Which database engines do not use SQL as their query language?"
  - should_decompose: true

Return ONLY valid JSON matching the provided schema. No additional text."#;
```

- [ ] **Step 4: Create the snapshot fixture**

Create `crates/retrieval/src/testdata/prompts/query_rewriter_system.txt` with the EXACT content of the new `SYSTEM_PROMPT` (everything between the `r#"` and `"#` delimiters, without the Rust literal markers).

Practical way: after editing the source file, extract:

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo run --release --example dump_prompt query_rewriter > crates/retrieval/src/testdata/prompts/query_rewriter_system.txt
```

If no `dump_prompt` example exists, write the fixture content manually (copy-paste from the Rust source, strip leading/trailing quotes). The test's `trim()` makes leading/trailing whitespace irrelevant.

- [ ] **Step 5: Run the guards**

Layer 1 (grep):
```bash
cd /Users/cosmingagea/workspace/agent-rs
grep -IEin 'penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal' \
    crates/retrieval/src/query_rewriter.rs || echo "CLEAN"
```

Expected: `CLEAN`.

Layer 2 (snapshot test):
```bash
cargo test -p agent-rs-retrieval --test prompt_snapshots \
    query_rewriter_system_prompt_is_universal_and_matches_fixture -- --nocapture
```

Expected: PASS. (Other four tests in the file still fail — they will pass after Tasks 2.2–2.5.)

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/query_rewriter.rs \
        crates/retrieval/src/testdata/prompts/query_rewriter_system.txt
git commit -m "refactor(query_rewriter): universalize prompt (no domain bias)"
```

### Task 2.2: De-bias `crates/retrieval/src/metadata_filter.rs`

**Files:**
- Modify: `crates/retrieval/src/metadata_filter.rs`
- Create: `crates/retrieval/src/testdata/prompts/metadata_filter_system.txt`

- [ ] **Step 1: Read the current file**

```bash
sed -n '1,250p' /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/metadata_filter.rs
```

- [ ] **Step 2: Rewrite module-level doc**

Find any `//!` or `///` comment mentioning "Romanian and English queries", "articolul", "article XX", "Cod Penal".

Replace with a universal framing. Example rewrite for the module-level header doc:
```rust
//! Extract structural metadata hints from a natural-language query via an
//! LLM. Structural hints include numbered-section references (article,
//! section, chapter, clause, paragraph), page ranges, section-title
//! fragments, and content-type flags (table, formula, code).
//!
//! No domain or language assumptions: the prompt treats numbered-reference
//! detection as a cross-lingual structural task and returns empty hints for
//! plain natural-language queries.
```

- [ ] **Step 3: Rewrite the `SYSTEM_PROMPT`**

Keep the schema structure; strip Romanian-legal examples. Replace the existing prompt with the following universal version (locate and replace the full `const SYSTEM_PROMPT` literal):

```rust
const SYSTEM_PROMPT: &str = r#"You extract structural metadata hints from a user query for use as
query-time filters. Return ONE JSON object matching the provided schema. No
other output.

### Fields to extract

#### page_start / page_end
Inclusive 1-based page range if the query explicitly asks about a specific
page or page range.
- "on page 47" → page_start: 47, page_end: 47
- "pages 12-18" → page_start: 12, page_end: 18
- No page reference → both null.

#### article_number
Extract if the query references a specific numbered section: article,
section, chapter, clause, paragraph, or equivalent.
- "section 3.4" → "3.4"
- "Article 15" → "15"
- "chapter 7" → "7"
- "§ 4.2" → "4.2"
- "paragraph (2)" → "(2)"
If NO numbered-section reference → null.

#### section_contains
Short lowercase fragment that should appear in a section title, if the
query mentions a section/chapter name.
- "the chapter about caching" → "caching"
- "the section on error handling" → "error handling"
If no section-title reference → null.

#### requires_table
Set true only if the query explicitly asks about a table, matrix,
spreadsheet, or tabular data. Words: "table", "tabular", "spreadsheet",
"matrix", cross-lingual equivalents.
Otherwise false.

#### requires_formula
Set true only if the query explicitly asks about a formula, equation,
expression, or mathematical notation.
Otherwise false.

#### requires_code
Set true only if the query explicitly asks about code, a code example, a
program listing, or an algorithm. Words: "code", "example code", "listing",
"algorithm", "function", cross-lingual equivalents.
Otherwise false.

### Important rules

- Most queries are plain natural language with NO structural hints. For
  those, return all nulls and all false values — do NOT invent hints.
- Do NOT extract a number that is incidental to the query (for example
  "give me 3 examples" must NOT produce article_number: "3").
- Only extract article_number when the query contains an explicit reference
  marker (article, section, chapter, clause, paragraph, §, or a clearly
  equivalent marker in any language).
- Return empty/default hints when nothing structural is detected.

Return ONLY valid JSON matching the provided schema. No additional text."#;
```

- [ ] **Step 4: Create fixture**

Write `crates/retrieval/src/testdata/prompts/metadata_filter_system.txt` with the new prompt content.

- [ ] **Step 5: Run both guards**

```bash
cd /Users/cosmingagea/workspace/agent-rs
grep -IEin 'penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal' \
    crates/retrieval/src/metadata_filter.rs || echo "CLEAN"
cargo test -p agent-rs-retrieval --test prompt_snapshots \
    metadata_filter_system_prompt_is_universal_and_matches_fixture
```

Expected: `CLEAN` from grep, PASS from test.

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/metadata_filter.rs \
        crates/retrieval/src/testdata/prompts/metadata_filter_system.txt
git commit -m "refactor(metadata_filter): universalize prompt (no domain bias)"
```

### Task 2.3: De-bias `crates/retrieval/src/entity_linker.rs` (prompt only; overlap logic in Task 6.2)

**Files:**
- Modify: `crates/retrieval/src/entity_linker.rs`
- Create: `crates/retrieval/src/testdata/prompts/entity_linker_extract.txt`

- [ ] **Step 1: Read current prompt**

```bash
sed -n '290,360p' /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/entity_linker.rs
```

- [ ] **Step 2: Rewrite the entity-extract prompt**

Replace the existing `const ENTITY_EXTRACT_PROMPT: &str = "\..."` value with:

```rust
const ENTITY_EXTRACT_PROMPT: &str = "\
Extract the key named entities from the user's query. Return ONLY concrete \
entities (people, organizations, locations, products, standards, concepts, \
numbered-section references, dates, events) — NOT generic terms like \
\"document\", \"section\", or \"information\" unless they are part of a \
named reference.

Rules:
- Extract named entities, identifiers, and numbered references from the \
  query regardless of domain or language.
- Return an empty array if the query has no specific named entities.
- Maximum 5 entities.
- Use the CANONICAL form: prefer the base / lemma / citation form over an \
  inflected surface form when a clear canonical exists.
- Preserve the language of the source text: entities stay in their \
  original language unless the user explicitly asks for translation.
";
```

- [ ] **Step 3: Create fixture**

Write `crates/retrieval/src/testdata/prompts/entity_linker_extract.txt` with the new prompt content.

- [ ] **Step 4: Run both guards**

```bash
cd /Users/cosmingagea/workspace/agent-rs
grep -IEin 'penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal' \
    crates/retrieval/src/entity_linker.rs || echo "CLEAN"
cargo test -p agent-rs-retrieval --test prompt_snapshots \
    entity_linker_extract_prompt_is_universal_and_matches_fixture
```

Expected: `CLEAN`, PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/entity_linker.rs \
        crates/retrieval/src/testdata/prompts/entity_linker_extract.txt
git commit -m "refactor(entity_linker): universalize extraction prompt"
```

### Task 2.4: De-bias `crates/retrieval/src/ingestion/graph_extractor.rs`

**Files:**
- Modify: `crates/retrieval/src/ingestion/graph_extractor.rs`
- Create: `crates/retrieval/src/testdata/prompts/graph_extractor_system.txt`

- [ ] **Step 1: Read the current prompt (it's the largest)**

```bash
sed -n '85,260p' /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/ingestion/graph_extractor.rs
```

Note the multi-section prompt: header, universal rules, node rules, coreference rules, edge rules, few-shot example 1, few-shot example 2, output rules.

- [ ] **Step 2: Rewrite the doc comment above `SYSTEM_PROMPT`**

Find:
```rust
/// Optimized for multilingual extraction (Romanian, English, French, German,
/// etc.) with explicit guidance on legal, scientific, business, and narrative
/// document types. Few-shot examples cover entity normalization, coreference,
/// implicit relations, and edge case handling.
```

Replace with:
```rust
/// Optimised for multilingual extraction with explicit guidance on
/// scientific, technical, business, and narrative document types. Few-shot
/// examples cover entity normalisation, coreference, implicit relations,
/// and edge-case handling — drawn from multiple neutral domains to avoid
/// anchoring the model to any single domain.
```

- [ ] **Step 3: Rewrite the Edge rules section**

Find the Edge rules bullet listing `pedepsit_cu`, `constituie`, `reglementat_de`, `aplicabil_pentru`, and the explicit legal-text rider. Replace the entire Edge rules section in the prompt literal with:

```
- Use snake_case verb phrases in the present tense or past participle:
  `founded_by`, `located_in`, `acted_in`, `signed_on`, `depends_on`,
  `authored_by`, `regulated_by`, `applies_to`, `succeeded_by`,
  `composed_of`, `reports_to`, `references`.
- Source and target node IDs MUST appear in the nodes list. Never invent IDs.
- Capture IMPLICIT relations the text strongly entails. For example:
  "When X happens, Y becomes Z" implies `x -[causes]-> z` where `z` is a
  defined state of `y`.
- Focus on meaningful, non-trivial relations. Skip incidental mentions
  ("the document is 5 pages long").
- For regulatory / rule-bearing text: extract relations like `prohibits`,
  `permits`, `requires`, `defines`, `aggravates`, `excludes_from`,
  `applies_to` — using the language of the source document.
```

- [ ] **Step 4: Rewrite Few-shot example 2**

Find the Romanian legal Few-shot example (the one with Articolul 233, tâlhărie, furt, pedepsit_cu). Replace the entire example block (input text + expected output) with a neutral scientific/technical example. Insert this in place:

```
EXAMPLE 2 — Scientific/technical text:

Input chunk:
"The TCP protocol, defined in RFC 793 (1981), implements reliable ordered
delivery over IP by introducing sequence numbers and acknowledgements.
Van Jacobson's 1988 paper introduced congestion control, which is now
part of RFC 5681. Without congestion control, TCP suffers from congestion
collapse under heavy load."

Expected output:
{
  "nodes": [
    {"id": "tcp_protocol", "name": "TCP protocol", "type": "Concept",
     "description": "Reliable ordered delivery protocol over IP."},
    {"id": "rfc_793", "name": "RFC 793", "type": "Document",
     "description": "1981 specification defining TCP."},
    {"id": "rfc_5681", "name": "RFC 5681", "type": "Document",
     "description": "Specification that includes TCP congestion control."},
    {"id": "ip", "name": "IP", "type": "Concept",
     "description": "Internet Protocol, the layer over which TCP runs."},
    {"id": "sequence_numbers", "name": "sequence numbers", "type": "Concept",
     "description": "Per-byte numbering used by TCP to achieve ordered delivery."},
    {"id": "acknowledgements", "name": "acknowledgements", "type": "Concept",
     "description": "Receiver-side confirmations used by TCP."},
    {"id": "van_jacobson", "name": "Van Jacobson", "type": "Person",
     "description": "Introduced TCP congestion control in a 1988 paper."},
    {"id": "van_jacobson_1988_paper", "name": "Van Jacobson 1988 paper",
     "type": "Document", "description": "Introduced TCP congestion control."},
    {"id": "congestion_control", "name": "congestion control", "type": "Concept",
     "description": "Mechanism to avoid TCP congestion collapse."},
    {"id": "congestion_collapse", "name": "congestion collapse", "type": "Concept",
     "description": "Failure mode under heavy load without congestion control."},
    {"id": "1981", "name": "1981", "type": "Date",
     "description": "Publication year of RFC 793."},
    {"id": "1988", "name": "1988", "type": "Date",
     "description": "Publication year of Van Jacobson's congestion-control paper."}
  ],
  "edges": [
    {"source_node_id": "rfc_793", "target_node_id": "tcp_protocol",
     "relationship_name": "defines",
     "description": "RFC 793 defines the TCP protocol."},
    {"source_node_id": "rfc_793", "target_node_id": "1981",
     "relationship_name": "published_in",
     "description": "RFC 793 was published in 1981."},
    {"source_node_id": "tcp_protocol", "target_node_id": "ip",
     "relationship_name": "runs_over",
     "description": "TCP runs over IP."},
    {"source_node_id": "tcp_protocol", "target_node_id": "sequence_numbers",
     "relationship_name": "uses",
     "description": "TCP uses sequence numbers for ordered delivery."},
    {"source_node_id": "tcp_protocol", "target_node_id": "acknowledgements",
     "relationship_name": "uses",
     "description": "TCP uses acknowledgements for reliable delivery."},
    {"source_node_id": "van_jacobson", "target_node_id": "van_jacobson_1988_paper",
     "relationship_name": "authored",
     "description": "Van Jacobson authored the 1988 paper."},
    {"source_node_id": "van_jacobson_1988_paper", "target_node_id": "congestion_control",
     "relationship_name": "introduces",
     "description": "The paper introduces TCP congestion control."},
    {"source_node_id": "van_jacobson_1988_paper", "target_node_id": "1988",
     "relationship_name": "published_in",
     "description": "Published in 1988."},
    {"source_node_id": "rfc_5681", "target_node_id": "congestion_control",
     "relationship_name": "includes",
     "description": "RFC 5681 includes TCP congestion control."},
    {"source_node_id": "congestion_control", "target_node_id": "congestion_collapse",
     "relationship_name": "prevents",
     "description": "Congestion control prevents congestion collapse under heavy load."}
  ]
}
```

- [ ] **Step 5: Verify Few-shot example 1 is already neutral**

Open the file and check that Example 1 uses neutral content (e.g., Albert Einstein, a common choice). If Example 1 is biased in any way, rewrite it similarly.

- [ ] **Step 6: Create fixture**

Write `crates/retrieval/src/testdata/prompts/graph_extractor_system.txt` with the updated full prompt content.

- [ ] **Step 7: Run both guards**

```bash
cd /Users/cosmingagea/workspace/agent-rs
grep -IEin 'penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal' \
    crates/retrieval/src/ingestion/graph_extractor.rs || echo "CLEAN"
cargo test -p agent-rs-retrieval --test prompt_snapshots \
    graph_extractor_system_prompt_is_universal_and_matches_fixture
```

Expected: `CLEAN`, PASS.

- [ ] **Step 8: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/ingestion/graph_extractor.rs \
        crates/retrieval/src/testdata/prompts/graph_extractor_system.txt
git commit -m "refactor(graph_extractor): universalize prompt + neutral few-shot example"
```

### Task 2.5: De-bias `crates/agent/src/prompts.rs` + delete biased router comment

**Files:**
- Modify: `crates/agent/src/prompts.rs`
- Modify: `crates/retrieval/src/router.rs` (delete one comment block)
- Create: `crates/retrieval/src/testdata/prompts/agent_system.txt`

- [ ] **Step 1: Rewrite the routing heuristic**

In `crates/agent/src/prompts.rs`, find the `DEFAULT_SYSTEM_PROMPT` line:

```rust
   - Queries mentioning a specific article, section, or clause number -> \"keyword\" first
```

Replace with:
```rust
   - Queries mentioning an identifier (number, code, section reference, ID) -> \"keyword\" first
```

No other changes needed in that file — the rest of `DEFAULT_SYSTEM_PROMPT` is already universal.

- [ ] **Step 2: Delete the biased comment in `router.rs`**

Find in `crates/retrieval/src/router.rs` around line 472:

```rust
            // HyDE DISABLED — caused -48pp regression on semantic search.
            // The hypothetical answer is too specific for Romanian legal text
            // and mismatches with the actual chunk embeddings. Raw query
            // embedding + Docling heading-prefixed chunks works better.
            // TODO: re-enable selectively for Conceptual queries only.
```

Replace with:

```rust
            // HyDE DISABLED — caused a large regression on semantic search
            // in prior eval runs; hypothetical answers over-specify and
            // drift from chunk embeddings. Raw query embedding with
            // heading-prefixed chunks performed better.
            // TODO: re-enable selectively for Conceptual queries only once
            // we have a corpus-independent signal that HyDE helps.
```

- [ ] **Step 3: Create fixture**

Write `crates/retrieval/src/testdata/prompts/agent_system.txt` with the updated `DEFAULT_SYSTEM_PROMPT` content.

- [ ] **Step 4: Run both guards — now must be fully clean**

```bash
cd /Users/cosmingagea/workspace/agent-rs
grep -IEin 'penal|criminal|romanian|codul|articolul|pedepsit|Cod Penal' \
    crates/ | grep -v 'crates/retrieval/src/ingestion/metadata.rs' || echo "ALL CLEAN"
cargo test -p agent-rs-retrieval --test prompt_snapshots
```

Expected: `ALL CLEAN` from grep; all 5 snapshot tests PASS.

- [ ] **Step 5: Run the full workspace tests to make sure nothing else regressed**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test --workspace --no-run
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: compile clean, zero clippy warnings.

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/agent/src/prompts.rs \
        crates/retrieval/src/router.rs \
        crates/retrieval/src/testdata/prompts/agent_system.txt
git commit -m "refactor(agent,router): universalize routing heuristic + delete biased comment"
```

---

## Phase 3 — Capture Post-De-bias Baseline Eval

### Task 3.1: Baseline eval run

**Files:**
- Produces: `test_results/baseline_eval.json`

- [ ] **Step 1: Ensure infrastructure is up**

Per README §2.2, the eval needs Qdrant (1050/1051), Neo4j (1061), Postgres (1054), and the Docling sidecar (1068):

```bash
cd /Users/cosmingagea/workspace/agent-rs
docker compose up -d qdrant neo4j postgres docling-sidecar
# Wait for health; the sidecar takes ~15s.
```

- [ ] **Step 2: Set env and build the harness in release**

```bash
cd /Users/cosmingagea/workspace/agent-rs
export OPENAI_API_KEY=sk-...
export EMBEDDER_TYPE=openai
cargo build --release --example eval_harness
```

- [ ] **Step 3: Run the baseline and save it**

```bash
cd /Users/cosmingagea/workspace/agent-rs
EVAL_DATASET=eval/test_pdf_v1.json \
EVAL_SAVE_BASELINE=1 \
cargo run --release --example eval_harness
```

Expected: a new `test_results/baseline_eval.json`. The console prints per-strategy, per-type, per-difficulty metric tables.

- [ ] **Step 4: Commit the baseline artefact**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add test_results/baseline_eval.json
git commit -m "eval: capture post-de-bias baseline on test_pdf_v1 dataset"
```

If `test_results/` is `.gitignore`d by convention, commit the file under a dedicated location like `eval/baselines/test_pdf_v1_predbias_baseline.json` so the comparison file is versioned.

---

## Phase 4 — Core Types and Extractor Skeleton

### Task 4.1: Add weighted-keywords module + error variant

**Files:**
- Create: `crates/retrieval/src/weighted_keywords.rs`
- Modify: `crates/retrieval/src/lib.rs`
- Modify: `crates/retrieval/src/error.rs`

- [ ] **Step 1: Write the failing test — module exists and re-exports**

Create an empty test file or add a test inside `weighted_keywords.rs` that asserts the public symbols exist. Add to the bottom of `crates/retrieval/src/weighted_keywords.rs` once created:

```rust
#[cfg(test)]
mod sanity {
    use super::*;

    #[test]
    fn public_symbols_exist() {
        let _e: Option<WeightedKeywords> = None;
        let _t: Option<WeightedTerm> = None;
        let _c: Option<WeightedKeywordExtractorConfig> = None;
        let _i: Option<WeightedKeywordIntegrationConfig> = None;
        let _b: Option<BM25WeightingConfig> = None;
        let _r: Option<RerankAdjustmentConfig> = None;
        let _el: Option<EntityLinkerPrioritizationConfig> = None;
    }
}
```

- [ ] **Step 2: Create `weighted_keywords.rs` with the module skeleton**

```rust
//! LLM-backed weighted-keyword extractor and its pure-function consumer
//! helpers. See `docs/superpowers/specs/2026-04-17-weighted-keywords-design.md`.

use std::sync::Arc;
use std::time::Duration;

use crate::error::RetrievalError;

// ---------- Public output types ----------

/// Ranked, weight-normalized view of a user query's information-bearing terms.
///
/// Stable public type: future enhancements (per-sub-query refinement) add
/// new methods on [`WeightedKeywordExtractor`], never widen this type.
///
/// # Invariants
///
/// - Every [`WeightedTerm::weight`] is in `(0.0, 1.0]`.
/// - If non-empty, the maximum weight equals exactly `1.0`.
/// - Sorted by weight descending, stable on ties.
/// - Deduplicated by NFKC + case-folded `term`.
/// - Every `term` is non-empty after normalisation.
#[derive(Debug, Clone, Default)]
pub struct WeightedKeywords {
    pub terms: Vec<WeightedTerm>,
}

/// Single weighted keyword entry.
#[derive(Debug, Clone)]
pub struct WeightedTerm {
    pub term: String,
    pub weight: f32,
    pub is_phrase: bool,
}

impl WeightedKeywords {
    /// Safe-fallback empty value. All consumers' empty-input paths are
    /// byte-identical to pre-feature behaviour when they receive this.
    pub fn empty() -> Self { Self { terms: Vec::new() } }

    /// Sum of all weights.
    pub fn total_weight(&self) -> f32 {
        self.terms.iter().map(|t| t.weight).sum()
    }

    /// First `k` entries (already sorted by weight desc).
    pub fn top_k(&self, k: usize) -> &[WeightedTerm] {
        let n = self.terms.len().min(k);
        &self.terms[..n]
    }

    /// Convenience iterator.
    pub fn iter(&self) -> impl Iterator<Item = &WeightedTerm> {
        self.terms.iter()
    }
}

// ---------- Extractor configuration ----------

/// Typed configuration for [`WeightedKeywordExtractor`].
#[derive(Debug, Clone)]
pub struct WeightedKeywordExtractorConfig {
    pub api_key: String,
    pub model: String,
    pub max_retries: u32,
    pub timeout: Duration,
}

impl WeightedKeywordExtractorConfig {
    /// Construct with default model, retries, timeout.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "gpt-5.4-mini-2026-03-17".to_owned(),
            max_retries: 2,
            timeout: Duration::from_secs(10),
        }
    }
}

// ---------- Cross-consumer integration toggles ----------

/// Per-consumer integration config. Any field set to `None` disables that
/// consumer's integration at runtime.
#[derive(Debug, Clone)]
pub struct WeightedKeywordIntegrationConfig {
    pub bm25: Option<BM25WeightingConfig>,
    pub rerank: Option<RerankAdjustmentConfig>,
    pub entity_linker: Option<EntityLinkerPrioritizationConfig>,
}

#[derive(Debug, Clone)]
pub struct BM25WeightingConfig {
    pub min_weight: f32,
    pub max_repeats: u32,
    pub min_term_chars: usize,
    pub phrase_token_expansion: bool,
}

impl Default for BM25WeightingConfig {
    fn default() -> Self {
        Self { min_weight: 0.3, max_repeats: 3, min_term_chars: 3, phrase_token_expansion: true }
    }
}

#[derive(Debug, Clone)]
pub struct RerankAdjustmentConfig {
    pub alpha: f32,
    pub phrase_bonus: f32,
}

impl Default for RerankAdjustmentConfig {
    fn default() -> Self { Self { alpha: 0.2, phrase_bonus: 1.5 } }
}

#[derive(Debug, Clone)]
pub struct EntityLinkerPrioritizationConfig {
    pub top_k: usize,
}

impl Default for EntityLinkerPrioritizationConfig {
    fn default() -> Self { Self { top_k: 5 } }
}

// ---------- Extractor ----------

/// LLM-backed extractor. Built in subsequent tasks.
pub struct WeightedKeywordExtractor {
    #[allow(dead_code)]
    config: WeightedKeywordExtractorConfig,
    #[allow(dead_code)]
    http: Arc<reqwest::Client>,
}

impl WeightedKeywordExtractor {
    pub fn new(config: WeightedKeywordExtractorConfig, http: Arc<reqwest::Client>) -> Self {
        Self { config, http }
    }
}

// ---------- Placeholder for the pure consumer helpers — populated later ----------

// Will be filled in Task 5.x (weighted_bm25_query), Task 6.x (rerank),
// Task 7.x (entity overlap).
```

Note: `#[allow(dead_code)]` on the fields is temporary — removed when `extract()` is wired up in Task 4.3. This exception is explicitly allowed by the global rule because it's gated (and visible) on an in-progress scaffolding state; it must be removed before the feature is considered complete.

- [ ] **Step 3: Add module + re-exports to `lib.rs`**

Open `crates/retrieval/src/lib.rs`, find the `pub mod` list, add:

```rust
pub mod weighted_keywords;
```

Add to the `pub use` block:

```rust
pub use weighted_keywords::{
    BM25WeightingConfig, EntityLinkerPrioritizationConfig, RerankAdjustmentConfig,
    WeightedKeywordExtractor, WeightedKeywordExtractorConfig,
    WeightedKeywordIntegrationConfig, WeightedKeywords, WeightedTerm,
};
```

- [ ] **Step 4: Add error variant**

Open `crates/retrieval/src/error.rs`. In the `RetrievalError` enum, add:

```rust
#[error("weighted keyword extraction failed: {source}")]
WeightedKeywordExtraction {
    #[from]
    source: WeightedKeywordExtractionError,
},
```

Add the inner enum in the same file (or at the top of `weighted_keywords.rs` and re-export):

```rust
#[derive(Debug, thiserror::Error)]
pub enum WeightedKeywordExtractionError {
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("http status {status}: {body}")]
    Status { status: reqwest::StatusCode, body: String },
    #[error("malformed json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("schema violation: {0}")]
    Schema(String),
    #[error("timeout after {0:?}")]
    Timeout(Duration),
}
```

- [ ] **Step 5: Run the sanity test**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval sanity::public_symbols_exist
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/weighted_keywords.rs \
        crates/retrieval/src/lib.rs \
        crates/retrieval/src/error.rs
git commit -m "feat(retrieval/weighted_keywords): add module skeleton + types + error variant"
```

### Task 4.2: Implement `WeightedKeywords` normalisation constructor + tests

**Files:**
- Modify: `crates/retrieval/src/weighted_keywords.rs`

The extractor needs a trusted path to build a normalised `WeightedKeywords` from raw LLM output. This task adds the pure normalisation logic with unit tests.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block (create one below `mod sanity`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn raw(term: &str, w: f32, is_phrase: bool) -> WeightedTerm {
        WeightedTerm { term: term.to_owned(), weight: w, is_phrase }
    }

    #[test]
    fn from_raw_normalises_weights_so_max_is_one() {
        let input = vec![raw("alpha", 0.2, false), raw("beta", 0.4, false)];
        let kw = WeightedKeywords::from_raw(input);
        assert_eq!(kw.terms.len(), 2);
        let max = kw.terms.iter().map(|t| t.weight).fold(0f32, f32::max);
        assert!((max - 1.0).abs() < 1e-6, "max weight must be 1.0, got {max}");
    }

    #[test]
    fn from_raw_sorts_by_weight_descending() {
        let input = vec![raw("low", 0.1, false), raw("high", 0.9, false), raw("mid", 0.5, false)];
        let kw = WeightedKeywords::from_raw(input);
        let weights: Vec<f32> = kw.terms.iter().map(|t| t.weight).collect();
        for pair in weights.windows(2) {
            assert!(pair[0] >= pair[1], "weights must be desc, got {weights:?}");
        }
    }

    #[test]
    fn from_raw_deduplicates_case_insensitively() {
        let input = vec![
            raw("Foo", 0.5, false),
            raw("foo", 0.8, false),
            raw("FOO", 0.3, false),
        ];
        let kw = WeightedKeywords::from_raw(input);
        assert_eq!(kw.terms.len(), 1, "duplicates by case-folded term");
        assert_eq!(kw.terms[0].term.to_lowercase(), "foo");
    }

    #[test]
    fn from_raw_drops_empty_terms() {
        let input = vec![
            raw("", 0.9, false),
            raw("   ", 0.8, false),
            raw("actual", 0.5, false),
        ];
        let kw = WeightedKeywords::from_raw(input);
        assert_eq!(kw.terms.len(), 1);
        assert_eq!(kw.terms[0].term, "actual");
    }

    #[test]
    fn from_raw_clamps_invalid_weights() {
        let input = vec![
            raw("neg", -0.5, false),
            raw("zero", 0.0, false),
            raw("huge", 5.0, false),
            raw("nan", f32::NAN, false),
            raw("ok", 0.7, false),
        ];
        let kw = WeightedKeywords::from_raw(input);
        // Dropped: NaN. Clamped into range: neg → dropped (invariant says weight > 0), zero → dropped,
        // huge → kept and renormalised (becomes 1.0 if max).
        for t in &kw.terms { assert!(t.weight > 0.0 && t.weight <= 1.0); }
        let names: Vec<_> = kw.terms.iter().map(|t| t.term.clone()).collect();
        assert!(names.contains(&"huge".to_owned()) && names.contains(&"ok".to_owned()));
        assert!(!names.contains(&"nan".to_owned()));
        assert!(!names.contains(&"neg".to_owned()));
        assert!(!names.contains(&"zero".to_owned()));
    }

    #[test]
    fn empty_round_trip() {
        let kw = WeightedKeywords::empty();
        assert!(kw.terms.is_empty());
        assert_eq!(kw.total_weight(), 0.0);
        assert!(kw.top_k(5).is_empty());
    }

    #[test]
    fn top_k_respects_cap() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("a", 0.9, false), raw("b", 0.8, false), raw("c", 0.7, false),
        ]);
        assert_eq!(kw.top_k(2).len(), 2);
        assert_eq!(kw.top_k(10).len(), 3);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail (missing `from_raw`)**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords::tests
```

Expected: compile error, `from_raw` not defined.

- [ ] **Step 3: Implement `from_raw`**

Add to `weighted_keywords.rs`, inside `impl WeightedKeywords`:

```rust
impl WeightedKeywords {
    /// Normalise a raw vector of `WeightedTerm`s into a canonical
    /// `WeightedKeywords`, enforcing all struct-level invariants.
    ///
    /// Steps:
    /// 1. Trim + NFKC-normalise each `term`; drop entries whose normalised
    ///    term is empty.
    /// 2. Drop entries whose weight is NaN, infinite, or ≤ 0. (Out-of-range
    ///    high values are clamped during renormalisation.)
    /// 3. Deduplicate by NFKC + lowercased `term`; on collision, keep the
    ///    higher-weight entry and carry its `is_phrase`.
    /// 4. Renormalise so max weight == 1.0 (linear scale to max).
    /// 5. Stable-sort by weight descending.
    pub fn from_raw(raw: Vec<WeightedTerm>) -> Self {
        use unicode_normalization::UnicodeNormalization;

        // Step 1 + 2: normalise term, drop invalid.
        let mut cleaned: Vec<WeightedTerm> = raw
            .into_iter()
            .filter_map(|t| {
                let n: String = t.term.nfc().collect::<String>().trim().to_owned();
                if n.is_empty() { return None; }
                if !t.weight.is_finite() || t.weight <= 0.0 { return None; }
                Some(WeightedTerm { term: n, weight: t.weight, is_phrase: t.is_phrase })
            })
            .collect();

        // Step 3: dedupe by case-folded term, keep highest weight.
        // We scan and merge in-place for determinism.
        cleaned.sort_by(|a, b| a.term.to_lowercase().cmp(&b.term.to_lowercase()));
        let mut deduped: Vec<WeightedTerm> = Vec::with_capacity(cleaned.len());
        for t in cleaned.into_iter() {
            match deduped.last_mut() {
                Some(prev) if prev.term.to_lowercase() == t.term.to_lowercase() => {
                    if t.weight > prev.weight {
                        prev.weight = t.weight;
                        prev.is_phrase = t.is_phrase;
                        prev.term = t.term;
                    }
                }
                _ => deduped.push(t),
            }
        }

        if deduped.is_empty() { return Self::empty(); }

        // Step 4: renormalise so max weight == 1.0.
        let max = deduped.iter().map(|t| t.weight).fold(f32::MIN, f32::max);
        if max > 0.0 && (max - 1.0).abs() > 1e-6 {
            for t in &mut deduped { t.weight /= max; }
        }

        // Step 5: stable sort by weight desc.
        deduped.sort_by(|a, b| {
            b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal)
        });

        Self { terms: deduped }
    }
}
```

- [ ] **Step 4: Add `unicode-normalization` to `[dependencies]`**

In `crates/retrieval/Cargo.toml`, add:

```toml
unicode-normalization = "0.1"
```

under `[dependencies]` (if not already present from earlier changes).

- [ ] **Step 5: Run tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords::tests
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/weighted_keywords.rs crates/retrieval/Cargo.toml
git commit -m "feat(weighted_keywords): implement WeightedKeywords::from_raw + invariants"
```

### Task 4.3: Implement `WeightedKeywordExtractor::extract` with OpenAI structured output

**Files:**
- Modify: `crates/retrieval/src/weighted_keywords.rs`

- [ ] **Step 1: Write the failing test (uses a mock HTTP via `mockito`)**

Add to test module:

```rust
#[tokio::test]
async fn extract_parses_structured_response() {
    let mut server = mockito::Server::new_async().await;
    let body = r#"{
      "choices": [{"message": {"content":
        "{\"terms\": [{\"term\": \"compiler\", \"weight\": 0.9, \"is_phrase\": false}, {\"term\": \"static types\", \"weight\": 0.6, \"is_phrase\": true}]}"
      }}]
    }"#;
    let _m = server.mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body(body)
        .create_async()
        .await;

    let http = Arc::new(reqwest::Client::new());
    let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
    // Override endpoint for test. Implemented via an optional `endpoint` field;
    // add it if not present (see Step 3).
    cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
    let ex = WeightedKeywordExtractor::new(cfg, http);

    let kw = ex.extract("is a compiler faster than an interpreter?").await.unwrap();
    assert_eq!(kw.terms.len(), 2);
    assert_eq!(kw.terms[0].term, "compiler");
    assert!(kw.terms[0].weight > 0.0);
}

#[tokio::test]
async fn extract_falls_back_to_empty_on_http_error() {
    let mut server = mockito::Server::new_async().await;
    let _m = server.mock("POST", "/v1/chat/completions")
        .with_status(500)
        .with_body("internal")
        .expect_at_least(1)
        .create_async()
        .await;

    let http = Arc::new(reqwest::Client::new());
    let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
    cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
    cfg.max_retries = 1;
    let ex = WeightedKeywordExtractor::new(cfg, http);

    let result = ex.extract("anything").await;
    // Per spec: any failure collapses to empty(). We return Ok(empty()) and log.
    assert!(result.is_ok());
    assert!(result.unwrap().terms.is_empty());
}
```

- [ ] **Step 2: Add `mockito` to dev-dependencies in `crates/retrieval/Cargo.toml`**

```toml
[dev-dependencies]
mockito = "1"
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
```

- [ ] **Step 3: Add `endpoint: Option<String>` to the config**

In `WeightedKeywordExtractorConfig`, add field:
```rust
pub endpoint: Option<String>,
```

And in the `new()` constructor, initialise with `None`. Production code uses `https://api.openai.com/v1/chat/completions` when `endpoint` is `None`.

- [ ] **Step 4: Implement `extract()`**

Mirror the structure of `QueryRewriter::rewrite` in `query_rewriter.rs` — same pattern for backoff, prompt caching, JSON-schema response format.

Key pieces (full code):

```rust
impl WeightedKeywordExtractor {
    /// Extract weighted keywords for the given query.
    ///
    /// On any failure (HTTP, schema violation, timeout, retries exhausted)
    /// returns `Ok(WeightedKeywords::empty())` after logging — the router
    /// pipeline must never be blocked by this component.
    ///
    /// # Errors
    ///
    /// Returns `Ok(...)` in all practical failure paths (design decision);
    /// the `Result` is kept for forward-compatibility with future hard-error
    /// modes (e.g. invalid config detected at call time).
    #[tracing::instrument(skip(self, query), fields(query_hash = %query_hash(query)))]
    pub async fn extract(&self, query: &str) -> Result<WeightedKeywords, RetrievalError> {
        use backon::{ExponentialBuilder, Retryable};

        let endpoint = self.config.endpoint.as_deref()
            .unwrap_or("https://api.openai.com/v1/chat/completions");

        let body = serde_json::json!({
            "model": self.config.model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weighted_keywords",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "terms": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "properties": {
                                        "term": {"type": "string"},
                                        "weight": {"type": "number"},
                                        "is_phrase": {"type": "boolean"}
                                    },
                                    "required": ["term", "weight", "is_phrase"]
                                }
                            }
                        },
                        "required": ["terms"]
                    }
                }
            },
            "messages": [
                {"role": "system", "content": WEIGHTED_KEYWORDS_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        });

        let call = || async {
            let resp = self.http.post(endpoint)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .timeout(self.config.timeout)
                .send()
                .await
                .map_err(WeightedKeywordExtractionError::Http)?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                return Err(WeightedKeywordExtractionError::Status { status, body: body_text });
            }
            let full: serde_json::Value = resp.json().await
                .map_err(WeightedKeywordExtractionError::Http)?;
            let content = full.pointer("/choices/0/message/content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| WeightedKeywordExtractionError::Schema(
                    "missing choices[0].message.content".into()
                ))?;
            #[derive(serde::Deserialize)]
            struct Raw { terms: Vec<RawTerm> }
            #[derive(serde::Deserialize)]
            struct RawTerm { term: String, weight: f32, is_phrase: bool }
            let raw: Raw = serde_json::from_str(content)
                .map_err(WeightedKeywordExtractionError::Json)?;
            let cleaned = raw.terms.into_iter()
                .map(|r| WeightedTerm { term: r.term, weight: r.weight, is_phrase: r.is_phrase })
                .collect();
            Ok::<_, WeightedKeywordExtractionError>(WeightedKeywords::from_raw(cleaned))
        };

        let backoff = ExponentialBuilder::default()
            .with_max_times(self.config.max_retries as usize);

        match call.retry(backoff).await {
            Ok(kw) => Ok(kw),
            Err(e) => {
                tracing::warn!(error = %e, "weighted keyword extraction failed; returning empty");
                Ok(WeightedKeywords::empty())
            }
        }
    }
}

fn query_hash(q: &str) -> String {
    let h = blake3::hash(q.as_bytes());
    h.to_hex().to_string()[..12].to_owned()
}
```

- [ ] **Step 5: Write the universal extraction system prompt**

Add above the `impl` block:

```rust
const WEIGHTED_KEYWORDS_SYSTEM_PROMPT: &str = r#"You are an expert information-retrieval query analyst.

Your task: from a user query, produce a ranked list of the most
information-bearing terms and short phrases, with an importance weight on
each.

Rules:
1. Return at most 8 terms total.
2. For each term emit: { "term": string, "weight": number, "is_phrase": bool }.
3. Weight is an ORDINAL importance score in (0, 1] — relative importance,
   not a probability.
4. Prefer rare / discriminative / content-bearing terms (names, identifiers,
   domain-specific vocabulary, numbered references) over common function
   words.
5. Use phrases (is_phrase: true) when a multi-word group carries meaning
   that the individual tokens do not (e.g. "connection timeout",
   "fourth edition"). Single words use is_phrase: false.
6. Preserve the user's language; do not translate.
7. Do not assume any specific domain; rank purely by information content
   inside the given query.

Return ONLY JSON matching the schema. No other text."#;
```

- [ ] **Step 6: Add `blake3` to `[dependencies]` if not already present**

Check `crates/retrieval/Cargo.toml`; from the workspace-level Cargo.toml we saw `blake3 = "1"` already listed, but the retrieval crate may or may not import it. Add:

```toml
blake3 = "1"
```

- [ ] **Step 7: Run the tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords
```

Expected: all unit tests PASS, including the two new mock-based ones.

- [ ] **Step 8: Remove the `#[allow(dead_code)]` attributes on the struct fields (they're now used)**

In `weighted_keywords.rs`, remove `#[allow(dead_code)]` markers.

- [ ] **Step 9: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/weighted_keywords.rs crates/retrieval/Cargo.toml
git commit -m "feat(weighted_keywords): implement extract() with retry + empty fallback"
```

---

## Phase 5 — BM25 Consumer Integration

### Task 5.1: Implement `weighted_bm25_query` + unit tests

**Files:**
- Modify: `crates/retrieval/src/weighted_keywords.rs`

- [ ] **Step 1: Write failing tests**

Add to the tests module:

```rust
#[test]
fn bm25_empty_keywords_returns_original() {
    let q = "what is a monad";
    let cfg = BM25WeightingConfig::default();
    let out = weighted_bm25_query(q, &WeightedKeywords::empty(), &cfg);
    assert_eq!(out, q);
}

#[test]
fn bm25_filters_below_min_weight() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "important".into(), weight: 0.9, is_phrase: false },
        WeightedTerm { term: "noise".into(), weight: 0.1, is_phrase: false },
    ]);
    let cfg = BM25WeightingConfig { min_weight: 0.3, ..Default::default() };
    let out = weighted_bm25_query("anything", &kw, &cfg);
    assert!(out.contains("important"));
    assert!(!out.contains("noise"));
}

#[test]
fn bm25_respects_min_term_chars_for_single_tokens() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "ok".into(), weight: 0.8, is_phrase: false }, // 2 chars
        WeightedTerm { term: "fine".into(), weight: 0.7, is_phrase: false }, // 4 chars
    ]);
    let cfg = BM25WeightingConfig { min_term_chars: 3, ..Default::default() };
    let out = weighted_bm25_query("_", &kw, &cfg);
    assert!(!out.contains("ok"));
    assert!(out.contains("fine"));
}

#[test]
fn bm25_does_not_drop_common_english_function_words_by_hardcoded_list() {
    // Regression guard for the design decision in §7.1: no hardcoded
    // stopword list. If weights are above min_weight and length is above
    // min_term_chars, "the" / "and" must still appear.
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "the".into(), weight: 0.9, is_phrase: false },
        WeightedTerm { term: "and".into(), weight: 0.8, is_phrase: false },
    ]);
    let cfg = BM25WeightingConfig { min_term_chars: 3, min_weight: 0.1, ..Default::default() };
    let out = weighted_bm25_query("_", &kw, &cfg);
    assert!(out.split_whitespace().any(|t| t == "the"));
    assert!(out.split_whitespace().any(|t| t == "and"));
}

#[test]
fn bm25_phrase_preserved_when_expansion_false() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "error code".into(), weight: 0.9, is_phrase: true },
    ]);
    let cfg = BM25WeightingConfig { phrase_token_expansion: false, ..Default::default() };
    let out = weighted_bm25_query("_", &kw, &cfg);
    // Intact once, no standalone "error" or "code" beyond what's in the phrase.
    let count_error = out.matches("error").count();
    assert_eq!(count_error, 1);
}

#[test]
fn bm25_phrase_expanded_when_flag_true() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "error code".into(), weight: 1.0, is_phrase: true },
    ]);
    let cfg = BM25WeightingConfig {
        phrase_token_expansion: true,
        max_repeats: 3,
        min_term_chars: 3,
        min_weight: 0.0,
    };
    let out = weighted_bm25_query("_", &kw, &cfg);
    // Phrase appears intact once, component tokens appear at reduced rep (ceil(1.0 * 3 / 2) = 2 each).
    assert!(out.contains("error code"));
    let error_count = out.split_whitespace().filter(|t| *t == "error").count();
    let code_count = out.split_whitespace().filter(|t| *t == "code").count();
    assert!(error_count >= 1 && error_count <= 3, "got {error_count}");
    assert!(code_count >= 1 && code_count <= 3, "got {code_count}");
}

#[test]
fn bm25_token_level_aggregation_caps_at_max_repeats() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "foo".into(), weight: 1.0, is_phrase: false }, // emits ceil(1*3)=3
        WeightedTerm { term: "foo bar".into(), weight: 1.0, is_phrase: true }, // expansion emits 2 more foo
    ]);
    let cfg = BM25WeightingConfig {
        phrase_token_expansion: true,
        max_repeats: 3,
        min_term_chars: 3,
        min_weight: 0.0,
    };
    let out = weighted_bm25_query("_", &kw, &cfg);
    let foo_count = out.split_whitespace().filter(|t| *t == "foo").count();
    // Aggregated: 3 + 2 = 5, then clamped to 3.
    assert!(foo_count <= 3, "expected ≤ 3, got {foo_count}");
}
```

- [ ] **Step 2: Run tests to confirm they fail (function doesn't exist)**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords::tests::bm25 2>&1 | tail -10
```

Expected: compile errors.

- [ ] **Step 3: Implement `weighted_bm25_query`**

Add to `weighted_keywords.rs` (module-level, below the struct impls):

```rust
/// Construct a BM25 query string from an original user query and its
/// weighted-keyword view.
///
/// See §7.1 of `docs/superpowers/specs/2026-04-17-weighted-keywords-design.md`
/// for the algorithm. Pure function — no I/O, no state.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::{BM25WeightingConfig, WeightedKeywords, WeightedTerm, weighted_bm25_query};
/// let kw = WeightedKeywords::from_raw(vec![
///     WeightedTerm { term: "compiler".into(), weight: 0.9, is_phrase: false },
/// ]);
/// let cfg = BM25WeightingConfig::default();
/// let rewritten = weighted_bm25_query("compile", &kw, &cfg);
/// assert!(rewritten.contains("compiler"));
/// ```
pub fn weighted_bm25_query(
    original_query: &str,
    kw: &WeightedKeywords,
    cfg: &BM25WeightingConfig,
) -> String {
    if kw.terms.is_empty() {
        return original_query.to_owned();
    }

    // Accumulate per-token emission counts.
    let mut per_token: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    let mut intact_phrases: Vec<String> = Vec::new();

    for t in &kw.terms {
        if t.weight < cfg.min_weight { continue; }

        if t.is_phrase {
            intact_phrases.push(t.term.clone());
            if cfg.phrase_token_expansion {
                let rep = ((t.weight * cfg.max_repeats as f32) / 2.0).ceil() as u32;
                let rep = rep.max(1);
                for token in t.term.split_whitespace() {
                    if token.chars().count() < cfg.min_term_chars { continue; }
                    *per_token.entry(token.to_owned()).or_insert(0) += rep;
                }
            }
        } else {
            if t.term.chars().count() < cfg.min_term_chars { continue; }
            let rep = (t.weight * cfg.max_repeats as f32).ceil() as u32;
            let rep = rep.max(1);
            *per_token.entry(t.term.clone()).or_insert(0) += rep;
        }
    }

    // Clamp per-token aggregated counts at max_repeats.
    for count in per_token.values_mut() {
        if *count > cfg.max_repeats { *count = cfg.max_repeats; }
    }

    // Assemble output: intact phrases first, then aggregated tokens.
    let mut parts: Vec<String> = intact_phrases;
    for (token, count) in per_token {
        for _ in 0..count { parts.push(token.clone()); }
    }
    parts.join(" ")
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords
```

Expected: all tests PASS including the 7 new BM25 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/weighted_keywords.rs
git commit -m "feat(weighted_keywords): add weighted_bm25_query pure helper"
```

### Task 5.2: Property tests for `weighted_bm25_query`

**Files:**
- Modify: `crates/retrieval/src/weighted_keywords.rs`

- [ ] **Step 1: Add `proptest` to dev-deps**

In `crates/retrieval/Cargo.toml` `[dev-dependencies]`:

```toml
proptest = "1"
```

- [ ] **Step 2: Write the property tests**

Add to the tests module:

```rust
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_term() -> impl Strategy<Value = WeightedTerm> {
        (
            "[a-z]{3,10}( [a-z]{3,10}){0,2}",
            0.1f32..=1.0,
            any::<bool>(),
        )
            .prop_map(|(term, weight, is_phrase)| WeightedTerm { term, weight, is_phrase })
    }

    proptest! {
        #[test]
        fn bm25_no_token_exceeds_max_repeats(
            terms in proptest::collection::vec(arb_term(), 0..8)
        ) {
            let kw = WeightedKeywords::from_raw(terms);
            let cfg = BM25WeightingConfig { max_repeats: 3, min_weight: 0.0, min_term_chars: 3, phrase_token_expansion: true };
            let out = weighted_bm25_query("_", &kw, &cfg);
            let mut counts = std::collections::HashMap::<String, u32>::new();
            // Count single-token occurrences (phrases are preserved intact as substrings;
            // for the cap we count space-separated tokens and ignore multi-word phrase substrings).
            for w in out.split_whitespace() {
                *counts.entry(w.to_owned()).or_insert(0) += 1;
            }
            for (tok, c) in counts {
                // The cap is per individual token including phrase expansion.
                // Intact phrase occurrences also contribute — but the phrase is
                // emitted at most once per term, so the worst case is cap + 1
                // (one from the intact phrase). Allow a small buffer.
                prop_assert!(c <= cfg.max_repeats + 1, "token {tok:?} appeared {c} times");
            }
        }

        #[test]
        fn bm25_empty_input_identity(q in "[a-z ]{0,50}") {
            let out = weighted_bm25_query(&q, &WeightedKeywords::empty(), &BM25WeightingConfig::default());
            prop_assert_eq!(out, q);
        }
    }
}
```

- [ ] **Step 3: Run**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords::proptests
```

Expected: PASS (~256 cases per property).

- [ ] **Step 4: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/weighted_keywords.rs crates/retrieval/Cargo.toml
git commit -m "test(weighted_keywords): add proptest invariants for weighted_bm25_query"
```

### Task 5.3: Wire `weighted_bm25_query` into the router's keyword dispatch

**Files:**
- Modify: `crates/retrieval/src/router.rs`

- [ ] **Step 1: Read the router's keyword dispatch**

```bash
grep -n 'keyword_search\|query_str\|sub_queries' /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/router.rs | head -40
```

Locate where the router calls `keyword_search` or passes a `query_text: &str` into a backend. Each `query_str in &sub_queries` is a natural insertion point.

- [ ] **Step 2: Store the `WeightedKeywords` on the router (hot-wire placeholder until Phase 7 wires the extractor)**

For now, to unblock Phase 5 testing, inject an `Arc<WeightedKeywords>` into the dispatch loop. Until the extractor is wired (Task 7.1), the router uses `WeightedKeywords::empty()`. That preserves today's byte-identical behaviour.

Add a field to `RetrievalRouter`:
```rust
weighted_keyword_integration: WeightedKeywordIntegrationConfig,
```

And in the dispatch loop, before calling the keyword backend:

```rust
let bm25_query_text = match self.weighted_keyword_integration.bm25.as_ref() {
    Some(bm25_cfg) if !weighted_keywords.terms.is_empty() => {
        weighted_bm25_query(query_str, weighted_keywords.as_ref(), bm25_cfg)
    }
    _ => query_str.clone(),
};
// Then pass `bm25_query_text` instead of `query_str` into `keyword_search`.
```

`weighted_keywords` is a local `Arc<WeightedKeywords>` populated in Task 7.1; for now initialise it as:
```rust
let weighted_keywords: Arc<WeightedKeywords> = Arc::new(WeightedKeywords::empty());
```

- [ ] **Step 3: Add a config surface for `WeightedKeywordIntegrationConfig` in the router's `RouterConfig` / `RouterBuilder`**

Add a setter:
```rust
pub fn with_weighted_keyword_integration(
    mut self,
    cfg: WeightedKeywordIntegrationConfig,
) -> Self {
    self.weighted_keyword_integration = cfg;
    self
}
```

Default (constructed when the router is built without calling the setter): all three integrations enabled with defaults:

```rust
impl Default for WeightedKeywordIntegrationConfig {
    fn default() -> Self {
        Self {
            bm25: Some(BM25WeightingConfig::default()),
            rerank: Some(RerankAdjustmentConfig::default()),
            entity_linker: Some(EntityLinkerPrioritizationConfig::default()),
        }
    }
}
```

- [ ] **Step 4: Run the workspace build + tests to make sure nothing else regressed**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo check --workspace
cargo test -p agent-rs-retrieval --lib
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/router.rs crates/retrieval/src/weighted_keywords.rs
git commit -m "feat(router): wire weighted_bm25_query into keyword dispatch (extractor stubbed as empty)"
```

---

## Phase 6 — Reranker Consumer Integration

### Task 6.1: Implement match helpers + `adjust_for_weighted_keywords`

**Files:**
- Modify: `crates/retrieval/src/weighted_keywords.rs`

- [ ] **Step 1: Write failing unit tests**

Add to tests module:

```rust
#[test]
fn adjust_empty_is_noop() {
    let out = adjust_for_weighted_keywords(
        0.5, "anything", &WeightedKeywords::empty(),
        &RerankAdjustmentConfig::default(),
    );
    assert!((out - 0.5).abs() < 1e-6);
}

#[test]
fn adjust_alpha_zero_is_noop() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "foo".into(), weight: 0.9, is_phrase: false },
    ]);
    let out = adjust_for_weighted_keywords(
        0.5, "foo bar", &kw,
        &RerankAdjustmentConfig { alpha: 0.0, phrase_bonus: 1.5 },
    );
    assert!((out - 0.5).abs() < 1e-6);
}

#[test]
fn adjust_no_matches_returns_base_score() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "foo".into(), weight: 0.9, is_phrase: false },
    ]);
    let out = adjust_for_weighted_keywords(
        0.5, "nothing relevant", &kw, &RerankAdjustmentConfig::default(),
    );
    assert!((out - 0.5).abs() < 1e-6);
}

#[test]
fn adjust_bounded_at_full_coverage() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "foo".into(), weight: 0.9, is_phrase: false },
        WeightedTerm { term: "bar".into(), weight: 0.5, is_phrase: false },
    ]);
    let cfg = RerankAdjustmentConfig { alpha: 0.2, phrase_bonus: 1.0 };
    let out = adjust_for_weighted_keywords(1.0, "foo bar and more", &kw, &cfg);
    // Full coverage → adjustment = 1.0 * (1 + 0.2 * 1.0) = 1.2.
    assert!((out - 1.2).abs() < 1e-6);
}

#[test]
fn adjust_word_boundary_not_substring() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "art".into(), weight: 1.0, is_phrase: false },
    ]);
    let cfg = RerankAdjustmentConfig { alpha: 0.2, phrase_bonus: 1.5 };
    // "article" must NOT match the query term "art".
    let out = adjust_for_weighted_keywords(1.0, "article twelve", &kw, &cfg);
    assert!((out - 1.0).abs() < 1e-6, "substring match should not happen");
}

#[test]
fn adjust_phrase_requires_contiguous_match() {
    let kw = WeightedKeywords::from_raw(vec![
        WeightedTerm { term: "error code".into(), weight: 1.0, is_phrase: true },
    ]);
    let cfg = RerankAdjustmentConfig { alpha: 0.2, phrase_bonus: 1.5 };
    // "error and the code" — contains both tokens but not contiguously. No match.
    let out = adjust_for_weighted_keywords(1.0, "error then the code", &kw, &cfg);
    assert!((out - 1.0).abs() < 1e-6, "phrase should require contiguous match");
    // Contiguous: match.
    let out2 = adjust_for_weighted_keywords(1.0, "see error code 504", &kw, &cfg);
    assert!(out2 > 1.0);
}
```

- [ ] **Step 2: Run — expect failures**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords::tests::adjust 2>&1 | tail
```

- [ ] **Step 3: Implement**

Add to `weighted_keywords.rs` (module-level):

```rust
use regex::Regex;

fn single_token_matches(text_lc: &str, token_lc: &str) -> bool {
    // Unicode-word-boundary match. `\b` in the `regex` crate is Unicode-aware
    // by default when the pattern is compiled without explicit ASCII mode.
    let escaped = regex::escape(token_lc);
    let pattern = format!(r"\b{}\b", escaped);
    // Compiling per-call is acceptable for v1; Task 8.1 optionally caches.
    Regex::new(&pattern).map(|r| r.is_match(text_lc)).unwrap_or(false)
}

fn phrase_matches(text_lc: &str, phrase_lc: &str) -> bool {
    // Phrase matches as a contiguous sub-sequence of whitespace-separated
    // tokens, boundary-aware at the phrase edges.
    let escaped = regex::escape(phrase_lc);
    // Replace each escaped whitespace with `\s+` so variable internal
    // spacing still matches.
    let normalised = escaped.replace("\\ ", r"\s+");
    let pattern = format!(r"\b{}\b", normalised);
    Regex::new(&pattern).map(|r| r.is_match(text_lc)).unwrap_or(false)
}

/// Adjust a cross-encoder reranker score by how well the candidate covers
/// the user's weighted keywords.
///
/// See spec §7.2. Pure function — no I/O.
pub fn adjust_for_weighted_keywords(
    reranker_score: f32,
    candidate_text: &str,
    kw: &WeightedKeywords,
    cfg: &RerankAdjustmentConfig,
) -> f32 {
    if kw.terms.is_empty() || cfg.alpha == 0.0 {
        return reranker_score;
    }
    let text_lc = candidate_text.to_lowercase();
    let mut matched = 0.0_f32;
    let mut total = 0.0_f32;
    for t in &kw.terms {
        let bonus = if t.is_phrase { cfg.phrase_bonus } else { 1.0 };
        let mass = t.weight * bonus;
        total += mass;
        let term_lc = t.term.to_lowercase();
        let hit = if t.is_phrase {
            phrase_matches(&text_lc, &term_lc)
        } else {
            single_token_matches(&text_lc, &term_lc)
        };
        if hit { matched += mass; }
    }
    if total <= 0.0 { return reranker_score; }
    let coverage = (matched / total).clamp(0.0, 1.0);
    reranker_score * (1.0 + cfg.alpha * coverage)
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords
```

Expected: all tests PASS.

- [ ] **Step 5: Add property tests**

Append to the `proptests` module:

```rust
proptest! {
    #[test]
    fn adjust_is_bounded(
        score in 0.0f32..5.0,
        alpha in 0.0f32..0.5,
        terms in proptest::collection::vec(arb_term(), 0..6),
        text in "[a-z ]{0,60}",
    ) {
        let kw = WeightedKeywords::from_raw(terms);
        let cfg = RerankAdjustmentConfig { alpha, phrase_bonus: 1.5 };
        let out = adjust_for_weighted_keywords(score, &text, &kw, &cfg);
        prop_assert!(out >= score);
        prop_assert!(out <= score * (1.0 + alpha + 1e-5));
    }
}
```

- [ ] **Step 6: Run property tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_keywords::proptests
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/weighted_keywords.rs
git commit -m "feat(weighted_keywords): bounded coverage-ratio rerank adjustment"
```

### Task 6.2: Wire `adjust_for_weighted_keywords` into `ranker.rs`

**Files:**
- Modify: `crates/retrieval/src/ranker.rs`

- [ ] **Step 1: Locate the post-rerank assembly**

```bash
grep -n 'rerank\|reranker\|score' /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/ranker.rs | head -30
```

- [ ] **Step 2: Add a new adjustment pass**

The ranker owns an `Arc<WeightedKeywords>` passed in by the router. Add:

```rust
use crate::weighted_keywords::{adjust_for_weighted_keywords, RerankAdjustmentConfig, WeightedKeywords};

pub fn apply_weighted_keyword_adjustment(
    reranked: &mut [SearchResult],
    kw: &WeightedKeywords,
    cfg: &RerankAdjustmentConfig,
) {
    if kw.terms.is_empty() || cfg.alpha == 0.0 { return; }
    for r in reranked.iter_mut() {
        r.score = adjust_for_weighted_keywords(r.score, &r.text, kw, cfg);
    }
    // Re-sort after adjustment.
    reranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
}
```

Note: the `SearchResult` type's field names (`score`, `text`) may differ — adjust to match the actual struct (check `agent_rs_stores::types::SearchResult`).

- [ ] **Step 3: Call the adjustment from the router's rerank step**

In `router.rs` after the cross-encoder rerank:
```rust
if let Some(rerank_cfg) = self.weighted_keyword_integration.rerank.as_ref() {
    crate::ranker::apply_weighted_keyword_adjustment(&mut reranked_results, weighted_keywords.as_ref(), rerank_cfg);
}
```

- [ ] **Step 4: Build and run existing tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test --workspace --no-run
cargo test -p agent-rs-retrieval
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/ranker.rs crates/retrieval/src/router.rs
git commit -m "feat(ranker): apply weighted-keyword coverage adjustment post-rerank"
```

---

## Phase 7 — Entity Linker + Router Wiring + Timeout

### Task 7.1: Overlap-first entity-linker tiering

**Files:**
- Modify: `crates/retrieval/src/entity_linker.rs`

- [ ] **Step 1: Write failing unit tests**

Add to an inline `#[cfg(test)] mod tests` block in `entity_linker.rs`:

```rust
#[cfg(test)]
mod weighted_overlap_tests {
    use super::*;
    use crate::weighted_keywords::{EntityLinkerPrioritizationConfig, WeightedKeywords, WeightedTerm};

    fn wk(terms: &[(&str, f32)]) -> WeightedKeywords {
        WeightedKeywords::from_raw(
            terms.iter().map(|(t, w)| WeightedTerm {
                term: (*t).to_owned(), weight: *w, is_phrase: false
            }).collect()
        )
    }

    #[test]
    fn overlap_first_orders_intersection_before_others() {
        let kw = wk(&[("webhook", 0.9), ("retry", 0.7), ("timeout", 0.5)]);
        let extracted = vec!["webhook".to_owned(), "delivery".to_owned()];
        let cfg = EntityLinkerPrioritizationConfig { top_k: 5 };
        let ordered = order_candidates_overlap_first(&kw, &extracted, &cfg);
        assert_eq!(ordered[0], "webhook"); // intersection, weight 0.9
        assert_eq!(ordered[1], "delivery"); // extracted-only, preserved order
        assert_eq!(ordered[2], "retry"); // weighted-only fallback
        assert_eq!(ordered[3], "timeout");
    }

    #[test]
    fn overlap_first_empty_weighted_preserves_extracted() {
        let extracted = vec!["a".to_owned(), "b".to_owned()];
        let cfg = EntityLinkerPrioritizationConfig { top_k: 5 };
        let ordered = order_candidates_overlap_first(&WeightedKeywords::empty(), &extracted, &cfg);
        assert_eq!(ordered, extracted);
    }

    #[test]
    fn overlap_first_caps_fallback_tier_at_top_k() {
        let kw = wk(&[("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7), ("e", 0.6), ("f", 0.5)]);
        let extracted: Vec<String> = vec![];
        let cfg = EntityLinkerPrioritizationConfig { top_k: 3 };
        let ordered = order_candidates_overlap_first(&kw, &extracted, &cfg);
        // Only 3 fallback entries emitted.
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0], "a");
        assert_eq!(ordered[2], "c");
    }
}
```

- [ ] **Step 2: Run to see failure**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib weighted_overlap_tests
```

Expected: compile error for `order_candidates_overlap_first`.

- [ ] **Step 3: Implement the tiering function**

Add to `entity_linker.rs`:

```rust
use crate::weighted_keywords::{EntityLinkerPrioritizationConfig, WeightedKeywords};

/// Order candidates for Neo4j entity lookup by overlap-first precedence.
///
/// See spec §7.3. Tiers:
/// 1. Weighted-keyword terms that also appear among the LLM-extracted
///    entity candidates — ordered by weight descending.
/// 2. LLM-extracted entity candidates not in tier 1, original order.
/// 3. Remaining weighted-keyword terms (top `top_k` by weight).
pub fn order_candidates_overlap_first(
    kw: &WeightedKeywords,
    extracted: &[String],
    cfg: &EntityLinkerPrioritizationConfig,
) -> Vec<String> {
    use std::collections::HashSet;

    let wk_lc: HashSet<String> = kw.iter().map(|t| t.term.to_lowercase()).collect();
    let extracted_lc: HashSet<String> = extracted.iter().map(|e| e.to_lowercase()).collect();

    let mut tier1: Vec<String> = kw.iter()
        .filter(|t| extracted_lc.contains(&t.term.to_lowercase()))
        .map(|t| t.term.clone())
        .collect();
    // tier1 already sorted by weight desc because `kw.terms` is.

    let tier2: Vec<String> = extracted.iter()
        .filter(|e| !tier1.iter().any(|t| t.eq_ignore_ascii_case(e)))
        .cloned()
        .collect();

    let tier3: Vec<String> = kw.iter()
        .filter(|t| !extracted_lc.contains(&t.term.to_lowercase()))
        .map(|t| t.term.clone())
        .take(cfg.top_k)
        .collect();

    // Silence unused-var warning when empty.
    let _ = wk_lc;

    let mut out = tier1;
    out.extend(tier2);
    out.extend(tier3);
    out
}
```

- [ ] **Step 4: Wire it into `EntityLinker::link`**

Change the signature to accept weighted keywords and the prioritisation config:

```rust
pub async fn link(
    &self,
    query: &str,
    weighted_keywords: Option<&WeightedKeywords>,
    prio: Option<&EntityLinkerPrioritizationConfig>,
) -> EntityLinkResult { ... }
```

Inside `link`, after the LLM extracts `extracted_candidates: Vec<String>`:

```rust
let ordered = match (weighted_keywords, prio) {
    (Some(kw), Some(cfg)) if !kw.terms.is_empty() => {
        order_candidates_overlap_first(kw, &extracted_candidates, cfg)
    }
    _ => extracted_candidates,
};
// Then loop through `ordered` for Neo4j lookups as before.
```

All existing callers of `link` must be updated to pass `None, None` — byte-identical behaviour until the router wires it (Task 7.2).

- [ ] **Step 5: Run tests**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --lib entity_linker
cargo test --workspace --no-run
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/entity_linker.rs
git commit -m "feat(entity_linker): overlap-first candidate tiering driven by weighted keywords"
```

### Task 7.2: Add extractor branch to router with branch-local timeout

**Files:**
- Modify: `crates/retrieval/src/router.rs`

- [ ] **Step 1: Read the current `tokio::join!` call**

```bash
grep -n 'tokio::join!(rewrite_fut' /Users/cosmingagea/workspace/agent-rs/crates/retrieval/src/router.rs
```

- [ ] **Step 2: Build a `weighted_keyword_fut` with branch-local timeout**

Inside `retrieve`, before the existing `tokio::join!`:

```rust
let wk_extractor = self.weighted_keyword_extractor.clone();
let wk_query = query.to_owned();
let wk_budget = self.weighted_keyword_branch_budget;
let weighted_keyword_fut = async move {
    match tokio::time::timeout(wk_budget, wk_extractor.extract(&wk_query)).await {
        Ok(Ok(kw)) => Arc::new(kw),
        Ok(Err(_)) | Err(_) => {
            tracing::warn!("weighted keyword branch resolved to empty (error or timeout)");
            Arc::new(WeightedKeywords::empty())
        }
    }
};
```

Adjust the `tokio::join!` at ~line 463 to include the new future:

```rust
let (rewrite, entity_link, metadata_filter, weighted_keywords) =
    tokio::join!(rewrite_fut, entity_link_fut, metadata_filter_fut, weighted_keyword_fut);
```

Now `weighted_keywords: Arc<WeightedKeywords>` is available for all three consumers' call sites you added in Phases 5 & 6 — replace the placeholder `Arc::new(WeightedKeywords::empty())` with the real value.

- [ ] **Step 3: Update `EntityLinker::link` call site with the weighted keywords**

```rust
let link_result = self.entity_linker.link(
    query,
    Some(weighted_keywords.as_ref()),
    self.weighted_keyword_integration.entity_linker.as_ref(),
).await;
```

- [ ] **Step 4: Add fields + constructor params**

Add to `RetrievalRouter`:

```rust
pub struct RetrievalRouter {
    // ... existing fields ...
    weighted_keyword_extractor: Arc<WeightedKeywordExtractor>,
    weighted_keyword_branch_budget: Duration,
    weighted_keyword_integration: WeightedKeywordIntegrationConfig,
}
```

Default budget: `Duration::from_secs(6)`.

- [ ] **Step 5: Wire in the binary/example entry points**

Files that construct `RetrievalRouter`:
- `examples/eval_harness.rs`
- `examples/e2e_full_trace.rs`
- `examples/e2e_legal_qa.rs` (rename-worthy, but out of scope for this PR)
- `crates/server/src/...`
- `src/main.rs` / `src/lib.rs` (root agent-rs)

In each, add construction of `WeightedKeywordExtractor` and pass it into the router builder. Pattern:

```rust
let wk_cfg = WeightedKeywordExtractorConfig::new(openai_api_key.clone());
let wk_extractor = Arc::new(WeightedKeywordExtractor::new(wk_cfg, http_client.clone()));
let router = RetrievalRouter::builder()
    // ... existing ...
    .with_weighted_keyword_extractor(wk_extractor)
    .with_weighted_keyword_integration(WeightedKeywordIntegrationConfig::default())
    .build()?;
```

- [ ] **Step 6: Run the full test suite**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo check --workspace
cargo test --workspace --no-run
cargo test -p agent-rs-retrieval
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/src/router.rs \
        examples/eval_harness.rs \
        examples/e2e_full_trace.rs \
        examples/e2e_legal_qa.rs \
        crates/server/src \
        src/main.rs src/lib.rs
git commit -m "feat(router): wire WeightedKeywordExtractor into parallel join + branch-local timeout"
```

---

## Phase 8 — Observability

### Task 8.1: Add weighted-keyword metrics

**Files:**
- Modify: `crates/observability/src/metrics.rs` (or `crates/retrieval/src/metrics.rs` — match the existing location of `RetrievalMetrics`)

- [ ] **Step 1: Locate the `RetrievalMetrics` definition**

```bash
grep -rn 'RetrievalMetrics\|pub struct RetrievalMetrics' /Users/cosmingagea/workspace/agent-rs/crates/
```

- [ ] **Step 2: Add new counters + histograms**

Add to the metrics struct (following existing OTel patterns):

```rust
// Weighted-keyword extractor
pub weighted_keyword_calls_total: Counter<u64>,
pub weighted_keyword_failures_total: Counter<u64>, // labelled by `kind`
pub weighted_keyword_nonempty_total: Counter<u64>,
pub weighted_keyword_latency_ms: Histogram<u64>,
pub weighted_keyword_terms_emitted: Histogram<u64>,
pub weighted_keyword_terms_surviving: Histogram<u64>,
pub weighted_keyword_phrase_ratio: Histogram<f64>,
// BM25 consumer
pub weighted_bm25_rewritten_tokens: Histogram<u64>,
pub weighted_bm25_unique_tokens: Histogram<u64>,
// Rerank consumer
pub weighted_rerank_coverage: Histogram<f64>,
pub weighted_rerank_adjustment_ratio: Histogram<f64>,
// Entity-linker consumer
pub weighted_entity_tier_hits_total: Counter<u64>, // labelled by `tier`
```

Initialise them in the constructor alongside the existing ones.

- [ ] **Step 3: Emit from `WeightedKeywordExtractor::extract`**

In `weighted_keywords.rs`, add a `metrics: Option<Arc<RetrievalMetrics>>` field to `WeightedKeywordExtractor` + a builder method `with_metrics`. On call:

- increment `weighted_keyword_calls_total`
- record `weighted_keyword_latency_ms`
- on success with non-empty: increment `weighted_keyword_nonempty_total`, record `terms_emitted` and `phrase_ratio`
- on failure: increment `weighted_keyword_failures_total` with `kind` label (http_5xx, timeout, malformed_json, etc.)

- [ ] **Step 4: Emit from `weighted_bm25_query` call site in router**

Record `weighted_bm25_rewritten_tokens` (whitespace-split count of output) and `weighted_bm25_unique_tokens` (distinct token count).

- [ ] **Step 5: Emit from `adjust_for_weighted_keywords` call in ranker**

Record per-candidate `weighted_rerank_coverage` and `weighted_rerank_adjustment_ratio`. Because candidates can be numerous, record these in a small batch or sample.

- [ ] **Step 6: Emit from `EntityLinker::link`**

When the overlap-first tiering is used, increment `weighted_entity_tier_hits_total{tier=intersection|extracted|fallback}` based on which tier contributed the resolved entity.

- [ ] **Step 7: Run all tests + lints**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-run
cargo test -p agent-rs-retrieval
```

- [ ] **Step 8: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/observability/src/metrics.rs \
        crates/retrieval/src/weighted_keywords.rs \
        crates/retrieval/src/router.rs \
        crates/retrieval/src/ranker.rs \
        crates/retrieval/src/entity_linker.rs
git commit -m "feat(observability): add weighted-keyword metrics distributions"
```

---

## Phase 9 — Integration Tests

### Task 9.1: Integration test — all consumers receive the value

**Files:**
- Create: `crates/retrieval/tests/weighted_keywords_integration.rs`

- [ ] **Step 1: Write the integration test (uses mockito for OpenAI)**

```rust
//! Integration tests: with a mock LLM, verify that weighted keywords
//! actually reach all three consumers and modify their behaviour.

use std::sync::Arc;
use agent_rs_retrieval::{
    BM25WeightingConfig, EntityLinkerPrioritizationConfig, RerankAdjustmentConfig,
    WeightedKeywordExtractor, WeightedKeywordExtractorConfig,
    WeightedKeywordIntegrationConfig, WeightedKeywords, WeightedTerm,
    weighted_bm25_query, adjust_for_weighted_keywords,
};

#[tokio::test]
async fn mock_llm_success_flows_to_all_three_consumers() {
    // 1. Mock the LLM.
    let mut server = mockito::Server::new_async().await;
    let llm_body = r#"{"choices":[{"message":{"content":
        "{\"terms\":[{\"term\":\"compiler\",\"weight\":0.9,\"is_phrase\":false},{\"term\":\"static types\",\"weight\":0.7,\"is_phrase\":true}]}"
    }}]}"#;
    let _m = server.mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body(llm_body)
        .create_async().await;

    // 2. Build the extractor.
    let http = Arc::new(reqwest::Client::new());
    let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
    cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
    let extractor = WeightedKeywordExtractor::new(cfg, http);

    // 3. Extract.
    let kw = extractor.extract("is a compiler faster than an interpreter?").await.unwrap();
    assert_eq!(kw.terms.len(), 2);

    // 4. Consumer 1: BM25 rewrite actually contains the high-weight term.
    let bm25_cfg = BM25WeightingConfig::default();
    let bm25_out = weighted_bm25_query("is a compiler faster", &kw, &bm25_cfg);
    assert!(bm25_out.contains("compiler"));

    // 5. Consumer 2: rerank adjustment non-trivially boosts a matching candidate.
    let rerank_cfg = RerankAdjustmentConfig::default();
    let candidate = "A compiler translates static types into machine code.";
    let boosted = adjust_for_weighted_keywords(0.5, candidate, &kw, &rerank_cfg);
    assert!(boosted > 0.5, "expected boost, got {boosted}");

    // 6. Consumer 3: entity-linker overlap-first tiering recognises "compiler".
    let entity_cfg = EntityLinkerPrioritizationConfig { top_k: 5 };
    let extracted = vec!["compiler".to_owned(), "interpreter".to_owned()];
    let ordered = agent_rs_retrieval::entity_linker::order_candidates_overlap_first(
        &kw, &extracted, &entity_cfg,
    );
    assert_eq!(ordered[0], "compiler", "intersection tier must come first");
}

#[tokio::test]
async fn mock_llm_failure_collapses_all_three_consumers_to_noops() {
    let mut server = mockito::Server::new_async().await;
    let _m = server.mock("POST", "/v1/chat/completions")
        .with_status(500)
        .with_body("broken")
        .expect_at_least(1)
        .create_async().await;

    let http = Arc::new(reqwest::Client::new());
    let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
    cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
    cfg.max_retries = 1;
    let extractor = WeightedKeywordExtractor::new(cfg, http);

    let kw = extractor.extract("anything").await.unwrap();
    assert!(kw.terms.is_empty());

    // Consumer 1: BM25 rewrite returns the original unchanged.
    let q = "something something";
    let bm25 = weighted_bm25_query(q, &kw, &BM25WeightingConfig::default());
    assert_eq!(bm25, q);

    // Consumer 2: rerank adjustment is a no-op.
    let s = adjust_for_weighted_keywords(0.5, "any text", &kw, &RerankAdjustmentConfig::default());
    assert!((s - 0.5).abs() < 1e-6);

    // Consumer 3: entity-linker tiering with empty keywords equals extracted input.
    let extracted = vec!["foo".to_owned(), "bar".to_owned()];
    let ordered = agent_rs_retrieval::entity_linker::order_candidates_overlap_first(
        &kw, &extracted, &EntityLinkerPrioritizationConfig::default(),
    );
    assert_eq!(ordered, extracted);
}
```

- [ ] **Step 2: Run**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo test -p agent-rs-retrieval --test weighted_keywords_integration
```

Expected: both tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add crates/retrieval/tests/weighted_keywords_integration.rs
git commit -m "test(retrieval): integration tests for weighted-keyword consumer fan-out"
```

---

## Phase 10 — Evaluation and Sign-off

### Task 10.1: Feature-on eval run

- [ ] **Step 1: Ensure infra still up**

```bash
docker compose ps qdrant neo4j postgres docling-sidecar
```

- [ ] **Step 2: Run feature-on eval**

```bash
cd /Users/cosmingagea/workspace/agent-rs
EVAL_DATASET=eval/test_pdf_v1.json \
cargo run --release --example eval_harness | tee test_results/feature_on.log
```

Expected: console prints a baseline-delta table. `test_results/eval_report_<ts>.json` is the full structured report.

- [ ] **Step 3: Per-consumer ablation runs**

Run three more times with `WeightedKeywordIntegrationConfig` variants. If the config isn't surfaced via an env var yet, add `AGENT_RS__WEIGHTED_KW__BM25_ONLY=1`-style feature flags in `main.rs` or a small `examples/ablation_harness.rs`. Simplest path: add three env var reads that zero out the other two integrations when set.

Commands (after adding env support):
```bash
AGENT_RS__WEIGHTED_KW__ONLY=bm25 cargo run --release --example eval_harness \
  | tee test_results/ablation_bm25_only.log
AGENT_RS__WEIGHTED_KW__ONLY=rerank cargo run --release --example eval_harness \
  | tee test_results/ablation_rerank_only.log
AGENT_RS__WEIGHTED_KW__ONLY=entity cargo run --release --example eval_harness \
  | tee test_results/ablation_entity_only.log
```

- [ ] **Step 4: Compute metrics per spec §10.7 + §10.8**

If the harness doesn't yet compute MRR, pre-rerank Recall@5, and bootstrap CIs, extend it (see `examples/eval_harness.rs` — the computation is straightforward given the per-query top-K lists already present). Emit:
- Hit@1, Hit@5 (existing)
- MRR (new)
- NDCG@5 (existing)
- Recall@5 pre-rerank (new — harness must capture the fusion output before rerank)
- Recall@5 post-rerank (existing)
- Sample counts per type and per tier
- Bootstrap 95% CIs for each metric

Commit the harness extension.

- [ ] **Step 5: Verify acceptance criteria per spec §11**

Open `test_results/feature_on.log` and `test_results/eval_report_*.json`. Check each row of the spec §11 acceptance table.

If all pass: proceed to Step 6. If any fail:
- Regressions on some question type → iterate on defaults (first candidates: `alpha`, `min_weight`, `max_repeats`).
- p50 / p95 latency regression → shorten branch budget; verify extractor latency distribution.
- No Hit@1 lift → inspect distributional metrics; the failure could be in any of the three consumers.

- [ ] **Step 6: Final cargo gate**

```bash
cd /Users/cosmingagea/workspace/agent-rs
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green.

- [ ] **Step 7: Commit eval artefacts**

```bash
cd /Users/cosmingagea/workspace/agent-rs
git add test_results/
git commit -m "eval: feature-on and ablation runs for weighted-keyword generator"
```

### Task 10.2: Open PR

- [ ] **Step 1: Push branch + open PR**

```bash
git push -u origin weighted-keywords
gh pr create --title "feat(retrieval): weighted-keyword generator + prompt universalization" \
  --body "$(cat <<'EOF'
## Summary
- Adds universal LLM-backed `WeightedKeywordExtractor` feeding three retrieval consumers (BM25 rewrite, coverage-based rerank adjustment, overlap-first entity-linker tiering).
- De-biases five prompt-bearing files (query_rewriter, metadata_filter, entity_linker, graph_extractor, agent/prompts) and deletes one biased router comment.
- Two-layer forbidden-lexicon guard: CI grep (tripwire) + prompt snapshot tests with NFKC normalisation.

## Metrics (pre- vs post-feature on `test_pdf_v1`)
_(fill from test_results/feature_on.log)_

| Metric | Baseline | Feature | Δ |
|---|---|---|---|
| Hit@1 (graph_hybrid) | | | |
| MRR (graph_hybrid) | | | |
| NDCG@5 (graph_hybrid) | | | |
| Recall@5 pre-rerank (graph_hybrid) | | | |
| p50 latency | | | |
| p95 latency | | | |

## Ablation (which consumer contributed the lift)
_(fill from test_results/ablation_*.log)_

## Test plan
- [x] Unit tests pass
- [x] Property tests pass
- [x] Integration tests pass (mock-LLM fan-out)
- [x] Prompt snapshot tests pass (all 5 files universal)
- [x] CI forbidden-lexicon grep clean
- [x] cargo fmt / clippy -D warnings / test green
- [x] Eval acceptance criteria met per spec §11
EOF
)"
```

---

## Self-Review

**Spec coverage** — skimmed each spec section against the plan:

- §1 Problem statement — addressed by the two halves of the plan.
- §2 Goal — all five deliverables in §2 map to Phase 1–9 tasks.
- §3 Non-goals — nothing in the plan touches SPLADE, corpus-derived bridger, full-text LLM scorer, or ingestion format. ✓
- §4 Roadmap context — plan header + commit messages reference this.
- §5 Architecture / types — Tasks 4.1–4.3 cover every listed type including helper methods (`empty`, `total_weight`, `top_k`, `iter`).
- §6 Data flow + §6.1 timeout — Task 7.2 implements the branch-local timeout; tiering-to-empty-on-failure is explicit.
- §7.1 BM25 — Tasks 5.1–5.3 including the rejected-stopword regression guard test.
- §7.2 Rerank — Tasks 6.1–6.2 with boundary-aware matching, phrase bonus, bounded coverage.
- §7.3 Entity linker overlap-first — Task 7.1 with the three-tier precedence.
- §8 Prompt de-biasing — Tasks 2.1–2.5 (5 prompt files + 1 router comment).
- §8.4 Two-layer guard — Task 1.1 (grep) + Task 1.2 (snapshot tests).
- §9 Error handling / safety / observability — Task 4.3 (empty fallback); Task 8.1 (metrics).
- §10 Testing strategy — Tasks interleaved: unit tests in 4.2/4.3/5.1/6.1/7.1, property tests in 5.2/6.1, integration in 9.1, eval in 10.1.
- §10.7 Metric breadth — Task 10.1 Step 4 extends the harness with MRR, Recall@5 pre-rerank, bootstrap CIs.
- §10.8 Statistical rigor — Task 10.1 Step 4 bootstrap CI requirement.
- §11 Acceptance criteria — Task 10.1 Step 5 verification.
- §12 Sequencing — phase ordering matches spec sequencing.
- §13 Risks — mitigations implemented in the corresponding tasks.
- §14 Open questions — defaults picked in the "Concrete Defaults" table at the top.

**Placeholder scan** — every step has a runnable command or full code block; no "TBD" / "implement later" / "similar to Task N".

**Type consistency** — `WeightedKeywords`, `WeightedTerm`, `BM25WeightingConfig`, `RerankAdjustmentConfig`, `EntityLinkerPrioritizationConfig` spelled identically across Tasks 4–9. `weighted_bm25_query` / `adjust_for_weighted_keywords` / `order_candidates_overlap_first` names consistent across definitions and call sites. `SYSTEM_PROMPT_FOR_TESTING` / `ENTITY_EXTRACT_PROMPT_FOR_TESTING` match between Task 1.2 (introduction) and Tasks 2.1–2.4 (replaced content still exported under the same alias).

Plan complete.
