# Coverage-Aware RAG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate grounding-gate false positives and add deterministic coverage proof for exhaustive (TOATE / list-all) queries by introducing an outline-first agent design with a `CoveragePlanner`/`CoverageVerifier` pipeline and three new tools.

**Architecture:** Three layers change in concert: (1) **storage** gains section-ordering fields plus a section-aware Qdrant filter helper and an extended `ChunkSummary` payload; (2) **tools** layer gains `outline_corpus`, `fetch_section_chunks`, `search_sections`, plus optional `section_id`/`doc_id`/`mode` on `search_knowledge`; (3) **agent pipeline** gains a `CoveragePlanner` step before `AgentRunner` and a `CoverageVerifier` step after `EvidenceFuser`. Grounding gate is rebuilt around tool-call args (rewritten subqueries) with phrase-presence checks. Synthesis switches to stable `cite_id = chunk_id` and drops the hedging wrapper when a coverage report exists.

**Tech Stack:** Rust 2021, tokio, rig (LLM agent), Qdrant (vector), Neo4j (graph), Postgres (state), OpenTelemetry tracing, fastembed, jina reranker. Existing crates: `agent`, `agent_rs_tools`, `agent_rs_retrieval`, `agent_rs_stores`, `agent_rs_core`, `agent_rs_observability`.

**Spec:** `docs/superpowers/specs/2026-04-19-coverage-aware-rag-design.md` (commit `c169716`).

**User-runs convention:** Throughout this plan, any `cargo build`, `cargo run`, `cargo test`, ingest, or `examples/ask` invocation is **handed off to the user** to run in their own CLI. The plan never invokes long-running commands autonomously. Tasks that depend on test results explicitly say "Hand off to user, expect: …".

**Codex checkpoints:** Three Codex 5.4-`high` reviews are scheduled — after Phase 2 (storage), Phase 4 (planner+verifier wiring), and Phase 6 (final). Each is a non-interactive `codex exec` with a self-contained prompt and `--output-last-message`.

**Branch:** Continue on `agent-reasoning-effort-minimal` (current branch) for a single-PR rollout per spec §11.

---

## Phase 0 — Pre-flight

### Task 0.1: Verify clean baseline

**Files:** none modified.

- [ ] **Step 1: Confirm branch and clean tree**

```bash
git rev-parse --abbrev-ref HEAD   # expect: agent-reasoning-effort-minimal
git status --short                 # expect: M crates/agent/src/synthesis.rs (existing) + ?? pi.pdf, segmentare.pdf
git log --oneline -1               # expect: c169716 docs(spec): coverage-aware RAG enterprise fix design
```

- [ ] **Step 2: Hand off — user runs baseline cargo check**

User runs:
```bash
cd /Users/cosmingagea/workspace/agent-rs && cargo check --workspace
```
Expected: success (zero warnings would be ideal but not blocking).

- [ ] **Step 3: Hand off — user runs baseline tests**

User runs:
```bash
cd /Users/cosmingagea/workspace/agent-rs && cargo test --workspace --lib --no-run
```
Expected: all unit-test binaries compile.

- [ ] **Step 4: Capture baseline ANSWER on the canonical query**

User runs the canonical failing query so we have a "before" trace to compare:
```bash
cd /Users/cosmingagea/workspace/agent-rs && cargo run --release --example ask -- "vreau cheatsheet cu TOATE metodele de segmentare"
```
Save the full stdout to `/tmp/before-canonical.log`. Expected: `is_grounded=false`, `confidence=0.000`, four sections (5.1.1, 5.1.2, histograma cumulativă, segmentare orientată pe regiuni) NOT mentioned in answer body.

---

## Phase 1 — Foundation: types & config

These tasks block everything else. **Sequential.** Do all three in order, then commit.

### Task 1.1: Add `coverage` types module

**Files:**
- Create: `crates/agent/src/coverage.rs`
- Modify: `crates/agent/src/lib.rs` (re-export)
- Test: `crates/agent/src/coverage.rs` (inline `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Create `crates/agent/src/coverage.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_plan_default_is_none_mode() {
        let plan = CoveragePlan::default();
        assert_eq!(plan.mode, CoverageMode::None);
        assert!(plan.target_sections.is_empty());
    }

    #[test]
    fn coverage_report_full_when_all_sections_have_evidence() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef { section_id: "s1".into(), doc_id: "d".into(), heading: "5.1".into() },
                SectionRef { section_id: "s2".into(), doc_id: "d".into(), heading: "5.2".into() },
            ],
        };
        let evidence_section_ids = ["s1", "s2"].into_iter().map(String::from).collect();
        let report = CoverageReport::from_plan_and_evidence(&plan, &evidence_section_ids);
        assert_eq!(report.expected, 2);
        assert_eq!(report.fetched, 2);
        assert!(report.missing_section_ids.is_empty());
        assert!(report.is_complete());
    }

    #[test]
    fn coverage_report_partial_lists_missing_section_ids() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef { section_id: "s1".into(), doc_id: "d".into(), heading: "5.1".into() },
                SectionRef { section_id: "s2".into(), doc_id: "d".into(), heading: "5.2".into() },
                SectionRef { section_id: "s3".into(), doc_id: "d".into(), heading: "5.3".into() },
            ],
        };
        let evidence_section_ids = ["s1"].into_iter().map(String::from).collect();
        let report = CoverageReport::from_plan_and_evidence(&plan, &evidence_section_ids);
        assert_eq!(report.expected, 3);
        assert_eq!(report.fetched, 1);
        assert_eq!(report.missing_section_ids, vec!["s2", "s3"]);
        assert!(!report.is_complete());
    }
}
```

- [ ] **Step 2: Hand off — user runs the test, expect compile failure**

User runs:
```bash
cargo test -p agent_rs_agent --lib coverage::tests 2>&1 | tail -30
```
Expected: compile errors — types don't exist yet.

- [ ] **Step 3: Implement the types**

Replace the file contents with:

```rust
//! Coverage planning & verification types.
//!
//! `CoveragePlan` is built by [`crate::steps::coverage_planner::CoveragePlanner`]
//! when an exhaustive-intent query is detected. It enumerates the sections the
//! pipeline must cover. After tool calls complete, [`crate::steps::coverage_verifier::CoverageVerifier`]
//! produces a [`CoverageReport`] showing how much of the plan was actually
//! satisfied by retrieved evidence.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Reference to one section the [`CoveragePlan`] targets.
///
/// `section_id` is the Neo4j section UUID (or the synthetic `preamble:<doc_id>`
/// id for unsectioned chunks). `heading` is denormalised for log/debug clarity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SectionRef {
    pub section_id: String,
    pub doc_id: String,
    pub heading: String,
}

/// Coverage mode for the query.
///
/// `None` = the standard non-exhaustive flow runs (definition / how-to / etc.).
/// `Exhaustive` = the agent is expected to hit every section in `target_sections`.
/// `Partial` = a relaxed exhaustive (e.g., user asked "main methods" not "ALL").
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CoverageMode {
    None,
    Partial,
    Exhaustive,
}

impl Default for CoverageMode {
    fn default() -> Self {
        CoverageMode::None
    }
}

/// Plan emitted by the planner; consumed by the agent and the verifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoveragePlan {
    pub mode: CoverageMode,
    pub target_sections: Vec<SectionRef>,
}

/// Result of comparing the plan against actually-retrieved evidence.
///
/// Exposed via the response struct + OpenTelemetry metrics. **Never** appended
/// to the user-facing answer body (per spec §6.2 / §14 Q4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub expected: usize,
    pub fetched: usize,
    pub missing_section_ids: Vec<String>,
}

impl CoverageReport {
    /// Build a report from a plan and the set of section ids present in evidence.
    pub fn from_plan_and_evidence(
        plan: &CoveragePlan,
        evidence_section_ids: &HashSet<String>,
    ) -> Self {
        let expected = plan.target_sections.len();
        let mut missing: Vec<String> = plan
            .target_sections
            .iter()
            .filter(|s| !evidence_section_ids.contains(&s.section_id))
            .map(|s| s.section_id.clone())
            .collect();
        missing.sort();
        let fetched = expected.saturating_sub(missing.len());
        Self {
            expected,
            fetched,
            missing_section_ids: missing,
        }
    }

    /// True iff every targeted section produced at least one piece of evidence.
    pub fn is_complete(&self) -> bool {
        self.missing_section_ids.is_empty()
    }
}

/// Captured tool invocation (args + which tool) used by the grounding gate to
/// extract the rewritten subject query rather than the raw user prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub args_json: String,
}

#[cfg(test)]
mod tests {
    // ... (the tests from Step 1, kept here verbatim)
}
```

- [ ] **Step 4: Add module re-export to `lib.rs`**

Open `crates/agent/src/lib.rs`, add line:

```rust
pub mod coverage;
```

(Place alphabetically with other `pub mod` lines.)

- [ ] **Step 5: Hand off — user runs the test, expect PASS**

```bash
cargo test -p agent_rs_agent --lib coverage::tests
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/agent/src/coverage.rs crates/agent/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(agent/coverage): add CoveragePlan, CoverageReport, SectionRef, ToolCallRecord types

Phase 1.1 of the coverage-aware-rag spec. Pure types; no behavioural change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Extend `QueryContext` with new optional fields

**Files:**
- Modify: `crates/agent/src/context.rs`

- [ ] **Step 1: Read the current `QueryContext` struct**

Open `crates/agent/src/context.rs`. Locate the `pub struct QueryContext { … }` definition.

- [ ] **Step 2: Add the four new fields**

Inside the struct, add:

```rust
    /// Plan emitted by [`crate::steps::coverage_planner::CoveragePlanner`].
    /// `None` for non-exhaustive flows (current default).
    pub coverage_plan: Option<crate::coverage::CoveragePlan>,

    /// Untruncated evidence merged in by exhaustive-mode tool calls.
    /// `EvidenceFuser` will preserve this set verbatim alongside the
    /// reranked tier in `evidence`.
    pub coverage_evidence: Vec<agent_rs_stores::types::SearchResult>,

    /// Verifier output. Populated by `CoverageVerifier`. `None` when the
    /// query was non-exhaustive or no plan existed.
    pub coverage_report: Option<crate::coverage::CoverageReport>,

    /// Captured tool invocations (args + tool name). Used by the grounding
    /// gate to extract subject tokens from the rewritten query rather than
    /// from the raw user prompt.
    pub tool_calls: Vec<crate::coverage::ToolCallRecord>,
```

- [ ] **Step 3: Update `QueryContext::new`**

Locate the `impl QueryContext { pub fn new(...) -> Self { … } }`. Add the new fields with defaults to the returned struct literal:

```rust
            coverage_plan: None,
            coverage_evidence: Vec::new(),
            coverage_report: None,
            tool_calls: Vec::new(),
```

- [ ] **Step 4: Hand off — user runs**

```bash
cargo check -p agent_rs_agent
```
Expected: success (only struct grew, all callers OK because fields are populated to defaults).

- [ ] **Step 5: Commit**

```bash
git add crates/agent/src/context.rs
git commit -m "$(cat <<'EOF'
feat(agent/context): extend QueryContext with coverage_plan, coverage_evidence, coverage_report, tool_calls

Phase 1.2. Additive — all new fields default to empty/None.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.3: Bump `max_turns` 4 → 5 in config

**Files:**
- Modify: `crates/core/src/config.rs:498` (per Codex audit)

- [ ] **Step 1: Locate the constant**

In `crates/core/src/config.rs`, find the `pub const DEFAULT_MAX_TURNS: usize = 4;` (or equivalent default field).

- [ ] **Step 2: Change to 5**

```rust
/// Default upper bound on agent reasoning turns per query.
///
/// Raised from 4 to 5 for the coverage-aware-rag design: typical exhaustive
/// path is `outline_corpus` → `search_sections` → synthesize (3 turns); the
/// extra turn provides headroom for one targeted follow-up `search_knowledge`
/// call when the planner partially missed.
pub const DEFAULT_MAX_TURNS: usize = 5;
```

(If the field is on a struct rather than a const, change the `Default` impl accordingly. Update any test snapshot that asserted on the value.)

- [ ] **Step 3: Hand off — user runs**

```bash
cargo check --workspace && cargo test -p agent_rs_core --lib
```
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/config.rs
git commit -m "$(cat <<'EOF'
feat(core/config): raise DEFAULT_MAX_TURNS from 4 to 5

Phase 1.3. Headroom for the coverage-aware exhaustive path
(outline_corpus + search_sections + synth + 1 follow-up).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Storage layer

Tasks 2.1, 2.2, 2.3 are **PARALLEL OK** (different files, no shared edits). Tasks 2.4 and 2.5 depend on 2.1 (for `Section` schema fields).

### Task 2.1: Add Neo4j coverage queries

**Files:**
- Create: `crates/stores/src/neo4j/coverage.rs`
- Modify: `crates/stores/src/neo4j/mod.rs` (re-export)
- Test: `crates/stores/src/neo4j/coverage.rs` (inline)

- [ ] **Step 1: Write the failing test (mock-friendly signature first)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_row_orders_by_section_order_then_first_chunk_index() {
        let mut rows = vec![
            SectionRow { section_id: "b".into(), doc_id: "d".into(), heading: "5.2".into(),
                          section_order: Some(2), first_chunk_index: Some(10),
                          page_start: Some(8), is_preamble: false },
            SectionRow { section_id: "a".into(), doc_id: "d".into(), heading: "5.1".into(),
                          section_order: Some(1), first_chunk_index: Some(0),
                          page_start: Some(2), is_preamble: false },
            SectionRow { section_id: "p".into(), doc_id: "d".into(), heading: "(preamble)".into(),
                          section_order: Some(0), first_chunk_index: Some(0),
                          page_start: Some(1), is_preamble: true },
        ];
        SectionRow::sort_for_outline(&mut rows);
        assert_eq!(rows.iter().map(|r| r.section_id.as_str()).collect::<Vec<_>>(),
                   vec!["p", "a", "b"]);
    }
}
```

- [ ] **Step 2: Hand off — verify the test fails to compile**

```bash
cargo test -p agent_rs_stores --lib neo4j::coverage::tests::section_row_orders 2>&1 | tail -20
```
Expected: error[E0432] / E0433 — `SectionRow` not in scope.

- [ ] **Step 3: Implement the module**

```rust
//! Section-level read queries used by the coverage planner / outline tool.

use neo4rs::{query, BoltType, Graph};
use tracing::instrument;

use crate::neo4j::neo4j_store::Neo4jStore;
use crate::neo4j::structural::{LABEL_DOCUMENT, LABEL_SECTION};

/// One section row fetched from Neo4j.
#[derive(Debug, Clone)]
pub struct SectionRow {
    pub section_id: String,
    pub doc_id: String,
    pub heading: String,
    pub section_order: Option<i64>,
    pub first_chunk_index: Option<i64>,
    pub page_start: Option<i64>,
    pub is_preamble: bool,
}

impl SectionRow {
    /// Stable outline ordering: by `section_order` ascending (preamble = 0
    /// sorts first), tie-broken by `first_chunk_index` then `page_start`.
    pub fn sort_for_outline(rows: &mut [SectionRow]) {
        rows.sort_by(|a, b| {
            a.section_order
                .unwrap_or(i64::MAX)
                .cmp(&b.section_order.unwrap_or(i64::MAX))
                .then(
                    a.first_chunk_index
                        .unwrap_or(i64::MAX)
                        .cmp(&b.first_chunk_index.unwrap_or(i64::MAX)),
                )
                .then(
                    a.page_start
                        .unwrap_or(i64::MAX)
                        .cmp(&b.page_start.unwrap_or(i64::MAX)),
                )
        });
    }
}

impl Neo4jStore {
    /// List all sections (real + synthetic preamble) of a document, ordered
    /// for outline display.
    #[instrument(skip(self), fields(doc_id = %doc_id))]
    pub async fn list_sections_for_doc(
        &self,
        doc_id: &str,
    ) -> Result<Vec<SectionRow>, crate::neo4j::Neo4jError> {
        let graph = self.graph();
        let cypher = format!(
            "MATCH (d:{LABEL_DOCUMENT} {{id: $doc_id}})-[:HAS_SECTION*1..]->(s:{LABEL_SECTION}) \
             RETURN s.id AS section_id, d.id AS doc_id, s.heading AS heading, \
                    s.section_order AS section_order, s.first_chunk_index AS first_chunk_index, \
                    s.page_start AS page_start, coalesce(s.is_preamble, false) AS is_preamble"
        );
        let mut rs = graph.execute(query(&cypher).param("doc_id", doc_id)).await?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await? {
            out.push(SectionRow {
                section_id: row.get("section_id").unwrap_or_default(),
                doc_id: row.get("doc_id").unwrap_or_default(),
                heading: row.get("heading").unwrap_or_default(),
                section_order: row.get("section_order").ok(),
                first_chunk_index: row.get("first_chunk_index").ok(),
                page_start: row.get("page_start").ok(),
                is_preamble: row.get("is_preamble").unwrap_or(false),
            });
        }
        SectionRow::sort_for_outline(&mut out);
        Ok(out)
    }

    /// Recursively collect descendant section ids (used when the outline
    /// agent says "all of 5.1 including 5.1.1, 5.1.2, …").
    #[instrument(skip(self), fields(section_id = %section_id))]
    pub async fn list_descendant_sections(
        &self,
        section_id: &str,
    ) -> Result<Vec<String>, crate::neo4j::Neo4jError> {
        let graph = self.graph();
        let cypher = format!(
            "MATCH (root:{LABEL_SECTION} {{id: $sid}})-[:PARENT_OF*0..]->(s:{LABEL_SECTION}) \
             RETURN s.id AS section_id"
        );
        let mut rs = graph.execute(query(&cypher).param("sid", section_id)).await?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await? {
            if let Ok(id) = row.get::<String>("section_id") {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// Return chunk ids that belong to any of the given sections, ordered
    /// by `chunk_index` so `fetch_section_chunks` reads the section
    /// linearly. Caller uses these ids to fetch full payloads from Qdrant.
    #[instrument(skip(self, section_ids), fields(n_sections = section_ids.len()))]
    pub async fn chunks_in_sections(
        &self,
        section_ids: &[String],
    ) -> Result<Vec<String>, crate::neo4j::Neo4jError> {
        if section_ids.is_empty() {
            return Ok(Vec::new());
        }
        let graph = self.graph();
        let ids_param: Vec<BoltType> = section_ids.iter().map(|s| s.as_str().into()).collect();
        let cypher = format!(
            "MATCH (s:{LABEL_SECTION})-[:HAS_CHUNK]->(c:Chunk) \
             WHERE s.id IN $ids \
             RETURN c.id AS chunk_id, coalesce(c.chunk_index, 0) AS chunk_index \
             ORDER BY chunk_index ASC"
        );
        let mut rs = graph
            .execute(query(&cypher).param("ids", BoltType::List(ids_param.into())))
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await? {
            if let Ok(id) = row.get::<String>("chunk_id") {
                out.push(id);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ... (sort test from Step 1)
}
```

- [ ] **Step 4: Re-export from `mod.rs`**

In `crates/stores/src/neo4j/mod.rs`, add `pub mod coverage;`.

- [ ] **Step 5: Hand off — user runs**

```bash
cargo test -p agent_rs_stores --lib neo4j::coverage::tests
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/stores/src/neo4j/coverage.rs crates/stores/src/neo4j/mod.rs
git commit -m "$(cat <<'EOF'
feat(stores/neo4j): add coverage queries — list_sections_for_doc, list_descendant_sections, chunks_in_sections

Phase 2.1 of coverage-aware-rag. Used by outline_corpus and
fetch_section_chunks tools. Pure read paths; no schema migration here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.2: Add Qdrant section-filter helper (PARALLEL with 2.1)

**Files:**
- Create: `crates/stores/src/qdrant/section_filter.rs`
- Modify: `crates/stores/src/qdrant/mod.rs` (re-export)
- Test: inline.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::{Filter, Match, FieldCondition};

    #[test]
    fn build_section_filter_with_doc_and_section() {
        let f = build_section_filter(Some("doc-123"), Some("sec-456"));
        let f = f.expect("filter should be Some when at least one id is provided");
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    fn build_section_filter_returns_none_when_both_args_are_none() {
        assert!(build_section_filter(None, None).is_none());
    }

    #[test]
    fn build_section_filter_with_only_doc_id() {
        let f = build_section_filter(Some("doc-123"), None);
        let f = f.expect("filter should be Some when doc_id is provided");
        assert_eq!(f.must.len(), 1);
    }
}
```

- [ ] **Step 2: Hand off — verify failing**

```bash
cargo test -p agent_rs_stores --lib qdrant::section_filter::tests 2>&1 | tail -15
```
Expected: undefined symbol.

- [ ] **Step 3: Implement**

```rust
//! Build Qdrant `Filter` clauses for `doc_id` and `section_id` payload keys.
//!
//! These are the exact payload keys the ingestion pipeline writes
//! (see `crates/retrieval/src/ingestion/pipeline.rs` payload construction).

use qdrant_client::qdrant::{Condition, FieldCondition, Filter, Match};

/// Build an `AND` filter on `doc_id` / `section_id`. Returns `None` when
/// both args are `None` so callers can use `Option::is_some` to skip
/// adding any filter at all.
pub fn build_section_filter(doc_id: Option<&str>, section_id: Option<&str>) -> Option<Filter> {
    let mut conds: Vec<Condition> = Vec::new();
    if let Some(did) = doc_id {
        conds.push(field_match("doc_id", did));
    }
    if let Some(sid) = section_id {
        conds.push(field_match("section_id", sid));
    }
    if conds.is_empty() {
        None
    } else {
        Some(Filter {
            must: conds,
            ..Default::default()
        })
    }
}

fn field_match(key: &str, value: &str) -> Condition {
    Condition {
        condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
            FieldCondition {
                key: key.to_string(),
                r#match: Some(Match {
                    match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                        value.to_string(),
                    )),
                }),
                ..Default::default()
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ... (tests from Step 1)
}
```

- [ ] **Step 4: Re-export**

`crates/stores/src/qdrant/mod.rs`: add `pub mod section_filter;`.

- [ ] **Step 5: Hand off — user runs**

```bash
cargo test -p agent_rs_stores --lib qdrant::section_filter::tests
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/stores/src/qdrant/section_filter.rs crates/stores/src/qdrant/mod.rs
git commit -m "feat(stores/qdrant): add build_section_filter helper for doc_id+section_id clauses

Phase 2.2 of coverage-aware-rag. Used by all three new tools and the
extended search_knowledge args.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.3: Make Romanian comma-below normalization unconditional (PARALLEL with 2.1, 2.2)

**Files:**
- Modify: `crates/retrieval/src/ingestion/text_normalize.rs:239` (env-gated branch)
- Modify: `crates/retrieval/src/embedder.rs` (apply to query side)

- [ ] **Step 1: Locate the env gate**

In `text_normalize.rs` around line 239, you'll find a `std::env::var(...)` check that gates Romanian comma-below ↔ cedilla folding. Read 20 lines of context to confirm the branch.

- [ ] **Step 2: Add a failing test for symmetric normalization**

In `text_normalize.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn comma_below_and_cedilla_normalize_to_same_form() {
    // Romanian "ș" with comma-below (U+0219) vs Turkish-ish "ş" with cedilla (U+015F).
    let cb = "func\u{021B}ie";   // cedilla-below "ț"
    let ce = "func\u{0163}ie";   // cedilla "ţ"
    assert_eq!(normalize_search_form(cb), normalize_search_form(ce));
}
```

- [ ] **Step 3: Hand off — confirm failure**

```bash
cargo test -p agent_rs_retrieval --lib text_normalize 2>&1 | tail -10
```
Expected: either undefined `normalize_search_form` (we add it next) or asymmetric output.

- [ ] **Step 4: Implement `normalize_search_form` and remove the env gate**

In `text_normalize.rs`:

```rust
/// Public, env-independent normalization applied to BOTH ingested chunks and
/// query text so they share a search form. Wraps `normalize_docling_text`
/// and unconditionally folds Romanian cedilla codepoints to comma-below.
pub fn normalize_search_form(input: &str) -> String {
    let stage1 = normalize_docling_text(input);
    fold_cedilla_to_comma_below(&stage1)
}

fn fold_cedilla_to_comma_below(s: &str) -> String {
    // U+015E LATIN CAPITAL LETTER S WITH CEDILLA  → U+0218 LATIN CAPITAL LETTER S WITH COMMA BELOW
    // U+015F LATIN SMALL LETTER S WITH CEDILLA    → U+0219 LATIN SMALL LETTER S WITH COMMA BELOW
    // U+0162 LATIN CAPITAL LETTER T WITH CEDILLA  → U+021A LATIN CAPITAL LETTER T WITH COMMA BELOW
    // U+0163 LATIN SMALL LETTER T WITH CEDILLA    → U+021B LATIN SMALL LETTER T WITH COMMA BELOW
    s.chars()
        .map(|c| match c {
            '\u{015E}' => '\u{0218}',
            '\u{015F}' => '\u{0219}',
            '\u{0162}' => '\u{021A}',
            '\u{0163}' => '\u{021B}',
            other => other,
        })
        .collect()
}
```

Then locate the existing env-gated path at line 239 and **remove the env check** — the fold is now always applied via `normalize_search_form`. Replace the env-gated body with a call to `fold_cedilla_to_comma_below` so the existing call sites still compile.

- [ ] **Step 5: Apply to query embedding side**

In `crates/retrieval/src/embedder.rs`, find the function that takes raw query text and prepares it for embedding (search for `pub async fn embed_query` or `pub fn embed`). At the top of the body:

```rust
let text = crate::ingestion::text_normalize::normalize_search_form(text);
```

(use the normalized `text` for embedding so query-side form matches corpus-side).

- [ ] **Step 6: Hand off — user runs**

```bash
cargo test -p agent_rs_retrieval --lib text_normalize
```
Expected: 1 new test passes; existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/retrieval/src/ingestion/text_normalize.rs crates/retrieval/src/embedder.rs
git commit -m "fix(retrieval/text_normalize): apply Romanian cedilla→comma-below fold unconditionally + on query side

Phase 2.3 of coverage-aware-rag. Eliminates ş/ş asymmetry across query
and corpus; previously env-gated. normalize_search_form is the new
public entry point used by both ingest and embedder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.4: Extend `Section` schema with ordering fields + synthetic preamble

**Files:**
- Modify: `crates/stores/src/neo4j/structural.rs:610` (chunk-without-section skip)
- Modify: `crates/stores/src/neo4j/structural.rs:625` (Section row writer)

- [ ] **Step 1: Add a failing test for synthetic preamble id**

Inside `structural.rs#[cfg(test)] mod`:

```rust
#[test]
fn preamble_section_id_is_stable_per_doc() {
    let id1 = synthetic_preamble_section_id("doc-abc");
    let id2 = synthetic_preamble_section_id("doc-abc");
    assert_eq!(id1, id2);
    assert_eq!(id1, "preamble:doc-abc");
    assert_ne!(id1, synthetic_preamble_section_id("doc-xyz"));
}
```

- [ ] **Step 2: Hand off — fail**

```bash
cargo test -p agent_rs_stores --lib structural::tests::preamble 2>&1 | tail -10
```

- [ ] **Step 3: Implement the helper + update writer**

Add to `structural.rs`:

```rust
/// Stable id for the synthetic preamble section of a document. Used when
/// the doc has chunks not attached to any heading (introduction, abstract,
/// front matter). Idempotent across re-ingest.
pub fn synthetic_preamble_section_id(doc_id: &str) -> String {
    format!("preamble:{doc_id}")
}
```

Locate the existing chunk-without-section skip path (line ~610) and replace with:

```rust
let section_id = chunk
    .section_id
    .clone()
    .unwrap_or_else(|| synthetic_preamble_section_id(&chunk.doc_id));
let is_preamble = chunk.section_id.is_none();
```

In the Section MERGE block (line ~625), extend the `ON CREATE SET` clause to include:

```cypher
MERGE (s:Section {id: $section_id})
ON CREATE SET s.heading = $heading,
              s.path = $path,
              s.depth = $depth,
              s.first_chunk_index = $first_chunk_index,
              s.page_start = $page_start,
              s.section_order = $section_order,
              s.is_preamble = $is_preamble
```

When `is_preamble` is true, set `heading = "(preamble)"`, `depth = 0`, `section_order = -1` (sorts first).

- [ ] **Step 4: Hand off — user runs**

```bash
cargo test -p agent_rs_stores --lib structural
```
Expected: new test passes; existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/stores/src/neo4j/structural.rs
git commit -m "feat(stores/neo4j): synthetic preamble section + section ordering fields

Phase 2.4. Chunks without explicit section now attach to preamble:<doc_id>
instead of being skipped. Section nodes gain first_chunk_index, page_start,
section_order, is_preamble for deterministic outline ordering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.5: Extend `ChunkSummary` payload + rollback

**Files:**
- Modify: `crates/retrieval/src/ingestion/summarizer.rs` (struct fields)
- Modify: `crates/retrieval/src/ingestion/pipeline.rs:1668` (payload write)
- Modify: `crates/retrieval/src/ingestion/pipeline.rs:1507` (rollback)

- [ ] **Step 1: Add failing test for the new payload fields**

In `summarizer.rs`'s test mod:

```rust
#[test]
fn chunk_summary_serializes_doc_section_ingest_run_id_and_status() {
    let summary = ChunkSummary {
        summary_id: "uuid-s".into(),
        chunk_id: "uuid-c".into(),
        summary_text: "...".into(),
        doc_id: Some("d1".into()),
        section_id: Some("sec1".into()),
        ingest_run_id: Some("run-7".into()),
        summary_status: SummaryStatus::Ok,
    };
    let json = serde_json::to_value(&summary).unwrap();
    assert_eq!(json["doc_id"], "d1");
    assert_eq!(json["section_id"], "sec1");
    assert_eq!(json["ingest_run_id"], "run-7");
    assert_eq!(json["summary_status"], "ok");
}
```

- [ ] **Step 2: Hand off — fail**

- [ ] **Step 3: Extend `ChunkSummary` struct + add `SummaryStatus`**

In `summarizer.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SummaryStatus {
    Ok,
    Empty,
    Failed,
}

pub struct ChunkSummary {
    pub summary_id: String,
    pub chunk_id: String,
    pub summary_text: String,
    pub doc_id: Option<String>,
    pub section_id: Option<String>,
    pub ingest_run_id: Option<String>,
    pub summary_status: SummaryStatus,
}
```

Update `summarize_batch` (around line 327) to populate the new fields from the chunk metadata it receives, and to record `SummaryStatus::Empty` / `SummaryStatus::Failed` instead of silently dropping (line ~741 in `pipeline.rs`).

- [ ] **Step 4: Update Qdrant payload writer (pipeline.rs:1668)**

Add to the payload JSON:

```rust
"doc_id": summary.doc_id,
"section_id": summary.section_id,
"ingest_run_id": summary.ingest_run_id,
"summary_status": summary.summary_status,
```

- [ ] **Step 5: Extend rollback (pipeline.rs:1507)**

The current rollback deletes from the main collection by `ingest_run_id`. Add the same delete against the summaries collection:

```rust
// Delete summaries for this run too. Failure here logs but does not fail
// the rollback — the main collection deletion is the source of truth.
if let Some(ref summaries_coll) = self.summaries_collection {
    if let Err(e) = self.vector_store
        .delete_by_payload(summaries_coll, "ingest_run_id", &run_id)
        .await
    {
        tracing::warn!(
            collection = %summaries_coll,
            error = %e,
            "failed to delete summaries during rollback; orphans may remain"
        );
    }
}
```

- [ ] **Step 6: Hand off — user runs**

```bash
cargo test -p agent_rs_retrieval --lib summarizer
```
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add crates/retrieval/src/ingestion/summarizer.rs crates/retrieval/src/ingestion/pipeline.rs
git commit -m "feat(retrieval/ingestion): summary payload gains doc_id/section_id/ingest_run_id/summary_status + rollback summaries

Phase 2.5 of coverage-aware-rag. Failed/empty summaries are now recorded
with status (instead of dropped) so outline_corpus can show '[no summary]'
placeholders rather than silent gaps. Rollback extends to summaries
collection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.6 (CHECKPOINT): Codex 5.4-high review of Phase 2

- [ ] **Step 1: Write Codex prompt to file**

Save to `/tmp/codex-phase2.txt`:

```
Review phase 2 of the coverage-aware-rag implementation in
/Users/cosmingagea/workspace/agent-rs (branch agent-reasoning-effort-minimal,
commits since c169716). Files changed in this phase:

- crates/stores/src/neo4j/coverage.rs (new)
- crates/stores/src/qdrant/section_filter.rs (new)
- crates/retrieval/src/ingestion/text_normalize.rs (cedilla fold)
- crates/retrieval/src/embedder.rs (query normalization)
- crates/stores/src/neo4j/structural.rs (preamble + section ordering)
- crates/retrieval/src/ingestion/summarizer.rs (ChunkSummary payload)
- crates/retrieval/src/ingestion/pipeline.rs (payload + rollback)

Verdict: are these storage-layer changes correct, idempotent under
re-ingest, and free of injection / panic / unwrap risks? Specifically
check:
1. Cypher parameter binding (no string interpolation of user input)
2. Idempotency of MERGE clauses with new section properties
3. Rollback completeness (orphan summaries possible?)
4. Backwards compat for already-ingested documents (missing fields)
5. Romanian normalization symmetry between query and corpus paths

Be specific; cite file:line. Under 600 words.
```

- [ ] **Step 2: Run Codex (background, stdin)**

```bash
rm -f /tmp/codex-phase2-out.md /tmp/codex-phase2.log
codex exec --full-auto --skip-git-repo-check \
  -m gpt-5.4 -c model_reasoning_effort=high \
  --output-last-message /tmp/codex-phase2-out.md \
  < /tmp/codex-phase2.txt > /tmp/codex-phase2.log 2>&1 &
```

(Expected runtime: 3-8 min. Wait for notification.)

- [ ] **Step 3: Read result**

```
Read /tmp/codex-phase2-out.md
```

- [ ] **Step 4: Address any blocker findings inline**

If Codex flags anything as blocker/high, create follow-up commits before moving to Phase 3. Re-run Codex after fixes if changes are non-trivial.

---

## Phase 3 — Tools layer

Tasks 3.1, 3.2, 3.3 are **PARALLEL OK** when dispatched as subagents (each is one self-contained tool struct + tests). Task 3.4 depends on the new tools existing for contract tests. Tasks 3.5 and 3.6 modify existing tools and can run alongside 3.1-3.3.

### Task 3.1: `OutlineCorpusTool` (PARALLEL)

**Files:**
- Create: `crates/tools/src/outline_corpus.rs`
- Modify: `crates/tools/src/lib.rs` (re-export)

- [ ] **Step 1: Write the failing definition contract test**

Add to `crates/tools/tests/tool_contract_test.rs`:

```rust
#[tokio::test]
async fn outline_corpus_tool_definition_contract() {
    use agent_rs_tools::OutlineCorpusTool;
    let tool: OutlineCorpusTool = serde_json::from_value(json!({}))
        .expect("OutlineCorpusTool should deserialize from {}");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "outline_corpus");
    assert!(
        def.description.to_lowercase().contains("outline")
            && def.description.to_lowercase().contains("section"),
        "outline_corpus description must reference outline / section: {:?}",
        def.description
    );
}
```

- [ ] **Step 2: Hand off — fail**

```bash
cargo test -p agent_rs_tools --test tool_contract_test outline_corpus_tool_definition_contract 2>&1 | tail -10
```

- [ ] **Step 3: Implement the tool**

```rust
//! `outline_corpus` — hierarchical TOC over Doc -> Section -> 1-line summary.
//!
//! Output is intended to be read by the LLM as a navigable index of what
//! the corpus contains. Joins Neo4j section structure with the existing
//! summaries Qdrant collection.

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::neo4j::coverage::SectionRow;
use agent_rs_stores::postgres::PgStore;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
pub struct OutlineCorpusArgs {
    /// Optional document id. If omitted, returns outline for every ingested doc.
    pub doc_id: Option<String>,
    /// Optional max sections to return per document (default 200).
    pub max_sections: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct OutlineCorpusTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    pg: Option<Arc<PgStore>>,
    // Summaries are looked up on-demand; reusing existing crates makes
    // the tool stateless beyond connection handles.
}

impl OutlineCorpusTool {
    pub fn new(neo4j: Arc<Neo4jStore>, pg: Arc<PgStore>) -> Self {
        Self { neo4j: Some(neo4j), pg: Some(pg) }
    }
}

impl Tool for OutlineCorpusTool {
    const NAME: &'static str = "outline_corpus";
    type Error = ToolError;
    type Args = OutlineCorpusArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Hierarchical OUTLINE / table-of-contents of the ingested corpus. \
                          Returns Document -> Section -> 1-line summary. PREFER this BEFORE \
                          search_knowledge for any query that asks for ALL methods, an \
                          enumeration, a cheatsheet, exhaustive coverage, or 'TOATE' \
                          (Romanian for 'all'). Use it to discover what sections exist, then \
                          target search_knowledge or search_sections at the relevant ids."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Optional document id; omit for full corpus."},
                    "max_sections": {"type": "integer", "description": "Optional cap (default 200)."}
                },
                "required": []
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let neo4j = self
            .neo4j
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: Neo4jStore missing".into()))?;
        let pg = self
            .pg
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("outline_corpus: PgStore missing".into()))?;

        let max = args.max_sections.unwrap_or(200);

        // 1. List documents to walk.
        let docs = pg
            .list_ingested_documents()
            .await
            .map_err(|e| ToolError::McpClient(format!("list docs: {e}")))?;
        if docs.is_empty() {
            return Ok("No documents have been ingested yet.".into());
        }

        let mut out = String::new();
        for doc in docs.iter().filter(|d| {
            args.doc_id
                .as_deref()
                .map_or(true, |want| d.document_id.to_string() == *want || d.source == *want)
        }) {
            out.push_str(&format!(
                "Document: {} ({})\n",
                doc.title.as_deref().unwrap_or("untitled"),
                doc.document_id
            ));
            let rows = neo4j
                .list_sections_for_doc(&doc.document_id.to_string())
                .await
                .map_err(|e| ToolError::McpClient(format!("sections: {e}")))?;
            for row in rows.iter().take(max) {
                let placeholder = "[no summary]";
                let summary = lookup_section_summary(pg, &row.section_id)
                    .await
                    .unwrap_or_else(|| placeholder.to_string());
                out.push_str(&format!(
                    "  - [{}] {} -> {}\n",
                    row.section_id, row.heading, summary
                ));
            }
            out.push('\n');
        }
        Ok(out)
    }
}

/// Pull a 1-line summary for a section by collapsing its first chunk's
/// summary. Returns `None` when no summary exists (failed/empty summary
/// during ingest, or pre-this-fix corpus). Caller substitutes a
/// placeholder.
async fn lookup_section_summary(_pg: &Arc<PgStore>, _section_id: &str) -> Option<String> {
    // TODO in implementation: query Postgres for the first chunk_id under
    // this section, then look up its summary via the summaries collection.
    // Stubbed here so the contract test compiles; concrete query is part
    // of the same task body.
    None
}
```

> **Note for the implementing agent:** `lookup_section_summary` must be replaced with a concrete query before commit. The placeholder above is an explicit marker — the contract test only checks the tool definition, not the call body.

Implement `lookup_section_summary`:

```rust
async fn lookup_section_summary(pg: &Arc<PgStore>, section_id: &str) -> Option<String> {
    pg.first_summary_text_for_section(section_id).await.ok().flatten()
}
```

Add `first_summary_text_for_section` on `PgStore` (in `crates/stores/src/postgres/`) — a single SQL query joining the chunks table to the summaries collection lookup table (or directly to a Postgres-side cache of summaries if one exists). If summaries live only in Qdrant, expose a Qdrant getter and have `outline_corpus` use it directly.

- [ ] **Step 4: Re-export**

`crates/tools/src/lib.rs`:
```rust
pub mod outline_corpus;
pub use outline_corpus::OutlineCorpusTool;
```

- [ ] **Step 5: Hand off — user runs**

```bash
cargo test -p agent_rs_tools --test tool_contract_test outline_corpus
```
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tools/src/outline_corpus.rs crates/tools/src/lib.rs crates/tools/tests/tool_contract_test.rs crates/stores/src/postgres/
git commit -m "feat(tools): OutlineCorpusTool — hierarchical TOC over Doc -> Section -> summary

Phase 3.1 of coverage-aware-rag. Read-only tool agent uses to discover
corpus structure before issuing targeted retrievals.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.2: `FetchSectionChunksTool` (PARALLEL with 3.1)

**Files:**
- Create: `crates/tools/src/fetch_section_chunks.rs`
- Modify: `crates/tools/src/lib.rs`

- [ ] **Step 1: Contract test (same pattern as 3.1)**

```rust
#[tokio::test]
async fn fetch_section_chunks_tool_definition_contract() {
    use agent_rs_tools::FetchSectionChunksTool;
    let tool: FetchSectionChunksTool = serde_json::from_value(json!({}))
        .expect("FetchSectionChunksTool should deserialize from {}");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "fetch_section_chunks");
    assert!(def.description.to_lowercase().contains("section"));
    assert!(def.description.to_lowercase().contains("exhaustive")
        || def.description.to_lowercase().contains("all"));
}
```

- [ ] **Step 2: Hand off — fail**

- [ ] **Step 3: Implement**

```rust
//! `fetch_section_chunks` — exhaustive, no-rank chunk retrieval per section.
//!
//! Bypasses ANN / BM25 ranking. Returns every chunk whose `section_id`
//! matches (or descends from) the requested section, ordered by
//! `chunk_index`. The agent uses this when it has decided exactly which
//! section it wants in full.

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::qdrant::QdrantStore;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
pub struct FetchSectionChunksArgs {
    pub section_id: String,
    #[serde(default)]
    pub include_descendants: bool,
    pub max_chars: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct FetchSectionChunksTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    qdrant: Option<Arc<QdrantStore>>,
    #[serde(skip)]
    collection: String,
}

impl FetchSectionChunksTool {
    pub fn new(neo4j: Arc<Neo4jStore>, qdrant: Arc<QdrantStore>, collection: String) -> Self {
        Self {
            neo4j: Some(neo4j),
            qdrant: Some(qdrant),
            collection,
        }
    }
}

impl Tool for FetchSectionChunksTool {
    const NAME: &'static str = "fetch_section_chunks";
    type Error = ToolError;
    type Args = FetchSectionChunksArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Exhaustive, NON-RANKED retrieval of every chunk in a section. \
                          Use when you need ALL content of a known section_id (typically after \
                          calling outline_corpus). Set include_descendants=true to also pull \
                          subsections (e.g., 5.1 -> 5.1.1, 5.1.2). Output is ordered by \
                          chunk_index; do NOT use this for semantic search."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "section_id": {"type": "string"},
                    "include_descendants": {"type": "boolean", "default": false},
                    "max_chars": {"type": "integer"}
                },
                "required": ["section_id"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME, section_id = %args.section_id, include_descendants = args.include_descendants))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let neo4j = self.neo4j.as_ref().ok_or_else(|| ToolError::McpClient("neo4j missing".into()))?;
        let qdrant = self.qdrant.as_ref().ok_or_else(|| ToolError::McpClient("qdrant missing".into()))?;

        let mut sections = vec![args.section_id.clone()];
        if args.include_descendants {
            sections = neo4j
                .list_descendant_sections(&args.section_id)
                .await
                .map_err(|e| ToolError::McpClient(format!("descendants: {e}")))?;
            if !sections.contains(&args.section_id) {
                sections.push(args.section_id.clone());
            }
        }

        let chunk_ids = neo4j
            .chunks_in_sections(&sections)
            .await
            .map_err(|e| ToolError::McpClient(format!("chunks: {e}")))?;
        if chunk_ids.is_empty() {
            return Ok(format!("No chunks found in section {}.", args.section_id));
        }

        let payloads = qdrant
            .retrieve_by_ids(&self.collection, &chunk_ids)
            .await
            .map_err(|e| ToolError::McpClient(format!("retrieve: {e}")))?;

        let max_chars = args.max_chars.unwrap_or(usize::MAX);
        let mut out = String::new();
        let mut written = 0usize;
        for p in payloads {
            let text = p
                .payload
                .get("chunk_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if written + text.len() > max_chars {
                out.push_str("[truncated by max_chars]\n");
                break;
            }
            written += text.len();
            out.push_str(text);
            out.push_str("\n\n");
        }
        Ok(out)
    }
}
```

- [ ] **Step 4: Re-export + commit**

(Same pattern as 3.1.)

- [ ] **Step 5: Hand off & commit**

Same shape as 3.1.

---

### Task 3.3: `SearchSectionsTool` (PARALLEL with 3.1, 3.2)

**Files:**
- Create: `crates/tools/src/search_sections.rs`
- Modify: `crates/tools/src/lib.rs`

This tool fans out internally — one tool call dispatches `fetch_section_chunks` per id in parallel (semaphore-bounded) and stitches results.

- [ ] **Step 1: Contract test**

```rust
#[tokio::test]
async fn search_sections_tool_definition_contract() {
    use agent_rs_tools::SearchSectionsTool;
    let tool: SearchSectionsTool = serde_json::from_value(json!({}))
        .expect("SearchSectionsTool should deserialize from {}");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "search_sections");
    assert!(def.description.to_lowercase().contains("batch")
        || def.description.to_lowercase().contains("multiple"));
}
```

- [ ] **Step 2: Hand off — fail**

- [ ] **Step 3: Implement**

```rust
//! `search_sections` — batched per-section retrieval.
//!
//! Concurrent fan-out: one tool call accepts a list of section_ids and
//! issues `fetch_section_chunks`-equivalent retrievals in parallel under
//! a semaphore. Designed to keep exhaustive flows inside `max_turns=5`.

use std::sync::Arc;

use agent_rs_core::ToolError;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::qdrant::QdrantStore;
use futures::future::join_all;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;

const DEFAULT_FAN_OUT_CONCURRENCY: usize = 8;

#[derive(Debug, Deserialize)]
pub struct SearchSectionsArgs {
    pub section_ids: Vec<String>,
    pub per_section_max_chars: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct SearchSectionsTool {
    #[serde(skip)]
    neo4j: Option<Arc<Neo4jStore>>,
    #[serde(skip)]
    qdrant: Option<Arc<QdrantStore>>,
    #[serde(skip)]
    collection: String,
}

impl SearchSectionsTool {
    pub fn new(neo4j: Arc<Neo4jStore>, qdrant: Arc<QdrantStore>, collection: String) -> Self {
        Self {
            neo4j: Some(neo4j),
            qdrant: Some(qdrant),
            collection,
        }
    }
}

impl Tool for SearchSectionsTool {
    const NAME: &'static str = "search_sections";
    type Error = ToolError;
    type Args = SearchSectionsArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "BATCH retrieval over MULTIPLE section_ids in a single tool call. \
                          Use when CoveragePlan provides a target_sections list — issuing one \
                          search_sections call (rather than N fetch_section_chunks calls) keeps \
                          the agent under the max_turns budget. Returns concatenated chunks \
                          grouped by section heading."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "section_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "per_section_max_chars": {"type": "integer"}
                },
                "required": ["section_ids"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME, n_sections = args.section_ids.len()))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let neo4j = self.neo4j.as_ref().ok_or_else(|| ToolError::McpClient("neo4j missing".into()))?;
        let qdrant = self.qdrant.as_ref().ok_or_else(|| ToolError::McpClient("qdrant missing".into()))?;
        if args.section_ids.is_empty() {
            return Ok("No section_ids provided.".into());
        }
        let sem = Arc::new(Semaphore::new(DEFAULT_FAN_OUT_CONCURRENCY));
        let collection = self.collection.clone();
        let max_chars = args.per_section_max_chars.unwrap_or(8_000);

        let futs = args.section_ids.iter().map(|sid| {
            let sem = sem.clone();
            let neo4j = Arc::clone(neo4j);
            let qdrant = Arc::clone(qdrant);
            let collection = collection.clone();
            let sid = sid.clone();
            async move {
                let _permit = sem.acquire_owned().await.unwrap();
                fetch_one(&neo4j, &qdrant, &collection, &sid, max_chars).await
            }
        });
        let results = join_all(futs).await;

        let mut out = String::new();
        for (sid, res) in args.section_ids.iter().zip(results.iter()) {
            out.push_str(&format!("=== section {sid} ===\n"));
            match res {
                Ok(text) => out.push_str(text),
                Err(e) => out.push_str(&format!("[error: {e}]\n")),
            }
            out.push('\n');
        }
        Ok(out)
    }
}

async fn fetch_one(
    neo4j: &Arc<Neo4jStore>,
    qdrant: &Arc<QdrantStore>,
    collection: &str,
    section_id: &str,
    max_chars: usize,
) -> Result<String, String> {
    let chunk_ids = neo4j
        .chunks_in_sections(&[section_id.to_string()])
        .await
        .map_err(|e| format!("{e}"))?;
    if chunk_ids.is_empty() {
        return Ok(format!("(no chunks in {section_id})"));
    }
    let payloads = qdrant
        .retrieve_by_ids(collection, &chunk_ids)
        .await
        .map_err(|e| format!("{e}"))?;
    let mut s = String::new();
    let mut written = 0;
    for p in payloads {
        let text = p
            .payload
            .get("chunk_text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if written + text.len() > max_chars {
            s.push_str("[truncated]\n");
            break;
        }
        written += text.len();
        s.push_str(text);
        s.push_str("\n\n");
    }
    Ok(s)
}
```

- [ ] **Step 4: Re-export + commit**

---

### Task 3.4: Extend `SearchKnowledgeArgs`

**Files:**
- Modify: `crates/tools/src/knowledge.rs`

- [ ] **Step 1: Add failing schema test**

In `tool_contract_test.rs`:

```rust
#[tokio::test]
async fn search_knowledge_args_accept_optional_section_id_doc_id_mode() {
    use agent_rs_tools::SearchKnowledgeArgs;
    let v = json!({"query": "x", "section_id": "s", "doc_id": "d", "mode": "all_in_section"});
    let _: SearchKnowledgeArgs = serde_json::from_value(v).unwrap();
}
```

- [ ] **Step 2: Hand off — fail**

- [ ] **Step 3: Extend struct**

In `knowledge.rs`:

```rust
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Ranked,
    AllInSection,
}

impl Default for SearchMode {
    fn default() -> Self { SearchMode::Ranked }
}

pub struct SearchKnowledgeArgs {
    pub query: String,
    pub strategy: Option<String>,
    pub limit: Option<usize>,
    pub section_id: Option<String>,
    pub doc_id: Option<String>,
    #[serde(default)]
    pub mode: SearchMode,
}
```

In the tool's `call` body, when `args.mode == SearchMode::AllInSection` and `args.section_id.is_some()`, route through `Neo4jStore::chunks_in_sections` + `QdrantStore::retrieve_by_ids` (i.e., the same path as `fetch_section_chunks`). Otherwise, build a Qdrant filter via `build_section_filter(args.doc_id.as_deref(), args.section_id.as_deref())` and pass it through the existing retrieval router.

- [ ] **Step 4: Hand off & commit**

---

### Task 3.5: `resolve_document` helper on `DocumentInfoTool`

**Files:**
- Modify: `crates/tools/src/document_info.rs`

- [ ] **Step 1: Add failing test**

```rust
#[test]
fn resolve_document_matches_by_source_or_title_substring() {
    let docs = vec![
        IngestedDocRow { source: "segmentare.pdf".into(), title: Some("Segmentare".into()), ..Default::default() },
        IngestedDocRow { source: "histograme.pdf".into(), title: None, ..Default::default() },
    ];
    assert_eq!(resolve_document(&docs, "segmentare").map(|d| d.source.clone()),
               Some("segmentare.pdf".into()));
    assert_eq!(resolve_document(&docs, "Histograme").map(|d| d.source.clone()),
               Some("histograme.pdf".into()));
    assert!(resolve_document(&docs, "irrelevant").is_none());
}
```

- [ ] **Step 2: Implement**

Free function in `document_info.rs`:

```rust
pub fn resolve_document<'a>(
    docs: &'a [IngestedDocRow],
    name_or_alias: &str,
) -> Option<&'a IngestedDocRow> {
    let needle = name_or_alias.to_lowercase();
    docs.iter().find(|d| {
        d.source.to_lowercase().contains(&needle)
            || d.title.as_deref().map_or(false, |t| t.to_lowercase().contains(&needle))
            || d.document_id.to_string().to_lowercase().contains(&needle)
    })
}
```

- [ ] **Step 3: Hand off & commit**

---

### Task 3.6: Apply hard `doc_id`/`section_id` predicate across all router branches

**Files:**
- Modify: `crates/retrieval/src/router.rs:1062` (keyword branch)
- Modify: `crates/retrieval/src/router.rs:1101` (graph branch)

- [ ] **Step 1: Add a failing integration test**

In `crates/retrieval/tests/`, create `section_filter_integration.rs` (or extend an existing test file):

```rust
#[tokio::test]
async fn router_drops_results_outside_doc_filter_in_keyword_branch() {
    // Setup: two ingested docs; query that would otherwise match both;
    // assert that doc_id filter prunes the off-doc result.
    // (Test uses the existing fixtures pattern; expand from prompt_snapshots.rs.)
    todo!("integration; see Phase 3.6 of the plan");
}
```

> If the existing test infrastructure cannot easily exercise this path, replace with a unit test against a small helper extracted from the router that takes `(results, doc_id_filter)` and returns the filtered set.

- [ ] **Step 2: Implement the post-fusion predicate**

In `router.rs`, after RRF fusion and before reranking, add:

```rust
if let Some(filter) = effective_section_filter.as_ref() {
    fused.retain(|r| {
        let payload = &r.payload;
        let doc_ok = filter.doc_id
            .as_deref()
            .map_or(true, |want| payload.get("doc_id").and_then(|v| v.as_str()) == Some(want));
        let sec_ok = filter.section_id
            .as_deref()
            .map_or(true, |want| payload.get("section_id").and_then(|v| v.as_str()) == Some(want));
        doc_ok && sec_ok
    });
}
```

`effective_section_filter` is sourced from the search args (extended in Task 3.4).

- [ ] **Step 3: Hand off & commit**

---

## Phase 4 — Agent pipeline

These are sequential — `CoveragePlanner` and `CoverageVerifier` both touch `QueryContext` + `Orchestrator`.

### Task 4.1: `CoveragePlanner` step

**Files:**
- Create: `crates/agent/src/steps/coverage_planner.rs`
- Modify: `crates/agent/src/steps/mod.rs` (re-export)

- [ ] **Step 1: Failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_exhaustive_intent_via_toate() {
        assert_eq!(detect_intent("vreau toate metodele de segmentare"), CoverageMode::Exhaustive);
    }

    #[test]
    fn detects_exhaustive_intent_via_cheatsheet() {
        assert_eq!(detect_intent("write a cheatsheet of all methods"), CoverageMode::Exhaustive);
    }

    #[test]
    fn does_not_detect_exhaustive_for_definition_query() {
        assert_eq!(detect_intent("ce este metoda Otsu"), CoverageMode::None);
    }

    #[test]
    fn soft_heuristic_logs_when_partial_signal_without_hard_match() {
        // "complete list" without "TOATE/all" hard token still logs+counts.
        let r = detect_intent_with_signal("give me the complete picture");
        assert_eq!(r.mode, CoverageMode::None);
        assert!(r.soft_heuristic_fired);
    }
}
```

- [ ] **Step 2: Hand off — fail**

- [ ] **Step 3: Implement detection rules + step**

```rust
//! CoveragePlanner — pre-agent step that detects exhaustive intent and
//! pre-builds a CoveragePlan over Neo4j sections.

use std::sync::Arc;

use agent_rs_observability::metrics::AgentMetrics;
use agent_rs_stores::neo4j::Neo4jStore;
use agent_rs_stores::postgres::PgStore;
use tracing::{instrument, warn};

use crate::context::QueryContext;
use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};

const HARD_EXHAUSTIVE_TOKENS: &[&str] = &[
    // RO
    "toate", "intreaga", "exhaustiv", "complet", "enumera", "enumeraza",
    "lista", "listeaza", "cheatsheet",
    // EN
    "all", "every", "exhaustive", "complete", "full list", "enumerate",
    "list of all", "comprehensive",
];

const SOFT_QUANTIFIER_TOKENS: &[&str] = &[
    "majoritatea", "fiecare", "principalele",
    "most", "main", "primary",
];

pub struct CoveragePlanner {
    neo4j: Arc<Neo4jStore>,
    pg: Arc<PgStore>,
    metrics: Arc<AgentMetrics>,
}

impl CoveragePlanner {
    pub fn new(neo4j: Arc<Neo4jStore>, pg: Arc<PgStore>, metrics: Arc<AgentMetrics>) -> Self {
        Self { neo4j, pg, metrics }
    }

    #[instrument(skip(self, ctx), fields(session_id = %ctx.session_id))]
    pub async fn plan(&self, ctx: &mut QueryContext) {
        let intent = detect_intent_with_signal(&ctx.query);

        if intent.soft_heuristic_fired && intent.mode == CoverageMode::None {
            warn!(
                query = %ctx.query,
                heuristic = %intent.heuristic_name,
                "coverage_planner.likely_exhaustive_unmatched"
            );
            self.metrics.coverage_planner_unmatched(intent.heuristic_name);
        }

        if intent.mode == CoverageMode::None {
            return;
        }

        // Resolve target doc(s). For now: every ingested doc; future
        // enhancement: resolve via document_info::resolve_document if the
        // query references a specific doc name.
        let docs = match self.pg.list_ingested_documents().await {
            Ok(d) => d,
            Err(e) => {
                warn!(error = %e, "coverage_planner: failed to list docs; skipping plan");
                return;
            }
        };

        let mut targets: Vec<SectionRef> = Vec::new();
        for doc in docs {
            let rows = match self.neo4j.list_sections_for_doc(&doc.document_id.to_string()).await {
                Ok(r) => r,
                Err(e) => {
                    warn!(doc_id = %doc.document_id, error = %e, "coverage_planner: skipping doc");
                    continue;
                }
            };
            for r in rows {
                targets.push(SectionRef {
                    section_id: r.section_id,
                    doc_id: r.doc_id,
                    heading: r.heading,
                });
            }
        }

        ctx.coverage_plan = Some(CoveragePlan {
            mode: intent.mode,
            target_sections: targets,
        });
    }
}

#[derive(Debug, PartialEq)]
struct IntentSignal {
    mode: CoverageMode,
    soft_heuristic_fired: bool,
    heuristic_name: &'static str,
}

pub(crate) fn detect_intent(q: &str) -> CoverageMode {
    detect_intent_with_signal(q).mode
}

pub(crate) fn detect_intent_with_signal(q: &str) -> IntentSignal {
    let lc = q.to_lowercase();
    if HARD_EXHAUSTIVE_TOKENS.iter().any(|t| lc.contains(t)) {
        return IntentSignal { mode: CoverageMode::Exhaustive, soft_heuristic_fired: false, heuristic_name: "hard_token" };
    }
    let soft_count = SOFT_QUANTIFIER_TOKENS.iter().filter(|t| lc.contains(*t)).count();
    if soft_count >= 2 {
        return IntentSignal { mode: CoverageMode::Partial, soft_heuristic_fired: true, heuristic_name: "soft_quantifier_pair" };
    }
    if soft_count == 1 {
        // Soft signal alone is not enough to trigger exhaustive — but we
        // log it so we can tune the rule set if users complain.
        return IntentSignal { mode: CoverageMode::None, soft_heuristic_fired: true, heuristic_name: "soft_quantifier_single" };
    }
    IntentSignal { mode: CoverageMode::None, soft_heuristic_fired: false, heuristic_name: "" }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ... (tests from Step 1)
}
```

Add `coverage_planner_unmatched(reason: &'static str)` to `AgentMetrics` (`crates/observability/src/metrics.rs`):

```rust
pub fn coverage_planner_unmatched(&self, reason: &'static str) {
    self.coverage_planner_unmatched
        .with_label_values(&[reason])
        .inc();
}
```

(Plus the `Counter` field declaration alongside other agent metrics.)

- [ ] **Step 4: Hand off — user runs**

```bash
cargo test -p agent_rs_agent --lib steps::coverage_planner::tests
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

---

### Task 4.2: `CoverageVerifier` step

**Files:**
- Create: `crates/agent/src/steps/coverage_verifier.rs`
- Modify: `crates/agent/src/steps/mod.rs`

- [ ] **Step 1: Failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};
    use agent_rs_stores::types::SearchResult;
    use serde_json::json;

    fn ev(section_id: &str) -> SearchResult {
        SearchResult {
            payload: json!({"section_id": section_id, "chunk_text": "..."}).as_object().unwrap().clone().into(),
            ..Default::default()
        }
    }

    #[test]
    fn verifier_marks_complete_when_every_target_present() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef { section_id: "s1".into(), doc_id: "d".into(), heading: "h1".into() },
                SectionRef { section_id: "s2".into(), doc_id: "d".into(), heading: "h2".into() },
            ],
        };
        let evidence = vec![ev("s1"), ev("s2")];
        let report = CoverageVerifier::build_report(&plan, &evidence);
        assert!(report.is_complete());
        assert_eq!(report.fetched, 2);
    }

    #[test]
    fn verifier_lists_missing_sections() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef { section_id: "s1".into(), doc_id: "d".into(), heading: "h1".into() },
                SectionRef { section_id: "s2".into(), doc_id: "d".into(), heading: "h2".into() },
            ],
        };
        let evidence = vec![ev("s1")];
        let report = CoverageVerifier::build_report(&plan, &evidence);
        assert!(!report.is_complete());
        assert_eq!(report.missing_section_ids, vec!["s2"]);
    }
}
```

- [ ] **Step 2: Implement**

```rust
//! CoverageVerifier — runs after EvidenceFuser, builds the CoverageReport.

use std::collections::HashSet;
use std::sync::Arc;

use agent_rs_observability::metrics::AgentMetrics;
use agent_rs_stores::types::SearchResult;
use tracing::{info, instrument};

use crate::context::QueryContext;
use crate::coverage::{CoveragePlan, CoverageReport};

pub struct CoverageVerifier {
    metrics: Arc<AgentMetrics>,
}

impl CoverageVerifier {
    pub fn new(metrics: Arc<AgentMetrics>) -> Self {
        Self { metrics }
    }

    #[instrument(skip(self, ctx), fields(session_id = %ctx.session_id))]
    pub fn verify(&self, ctx: &mut QueryContext) {
        let Some(plan) = ctx.coverage_plan.as_ref() else { return; };
        let evidence: Vec<&SearchResult> = ctx.evidence.iter().chain(ctx.coverage_evidence.iter()).collect();
        let report = Self::build_report(plan, &evidence_to_owned(evidence));
        info!(
            expected = report.expected,
            fetched = report.fetched,
            missing = report.missing_section_ids.len(),
            "coverage verifier complete"
        );
        self.metrics.coverage_verified(report.fetched, report.expected);
        ctx.coverage_report = Some(report);
    }

    pub(crate) fn build_report(plan: &CoveragePlan, evidence: &[SearchResult]) -> CoverageReport {
        let ids: HashSet<String> = evidence
            .iter()
            .filter_map(|e| e.payload.get("section_id").and_then(|v| v.as_str()).map(String::from))
            .collect();
        CoverageReport::from_plan_and_evidence(plan, &ids)
    }
}

fn evidence_to_owned(refs: Vec<&SearchResult>) -> Vec<SearchResult> {
    refs.into_iter().cloned().collect()
}

#[cfg(test)]
mod tests { /* … from Step 1 … */ }
```

Add `coverage_verified(fetched: usize, expected: usize)` to `AgentMetrics`.

- [ ] **Step 3: Hand off & commit**

---

### Task 4.3: Preserve `coverage_evidence` in `EvidenceFuser`

**Files:**
- Modify: `crates/agent/src/steps/evidence_fuser.rs`

- [ ] **Step 1: Failing test**

```rust
#[test]
fn fuser_preserves_coverage_evidence_unranked() {
    let mut ctx = test_ctx();
    ctx.coverage_evidence = vec![dummy_result("cov-1"), dummy_result("cov-2")];
    ctx.kb_results = vec![dummy_result("kb-1")];
    let fuser = EvidenceFuser::new(/* ... */);
    fuser.fuse(&mut ctx);
    // Both coverage chunks must survive into ctx.evidence regardless of limit.
    assert!(ctx.evidence.iter().any(|e| chunk_id(e) == "cov-1"));
    assert!(ctx.evidence.iter().any(|e| chunk_id(e) == "cov-2"));
}
```

- [ ] **Step 2: Modify `fuse`**

In `evidence_fuser.rs`, after the existing RRF + truncation step:

```rust
// Coverage tier: never truncated. Merge AFTER ranked tier so the LLM
// still sees the highest-scoring chunks first, but the planner-chosen
// chunks are guaranteed present.
if !ctx.coverage_evidence.is_empty() {
    let existing_ids: HashSet<&str> = ctx.evidence.iter()
        .filter_map(|e| e.payload.get("chunk_id").and_then(|v| v.as_str()))
        .collect();
    for cov in &ctx.coverage_evidence {
        let id = cov.payload.get("chunk_id").and_then(|v| v.as_str());
        if id.map_or(true, |i| !existing_ids.contains(i)) {
            ctx.evidence.push(cov.clone());
        }
    }
}
```

- [ ] **Step 3: Hand off & commit**

---

### Task 4.4: Wire planner + verifier into orchestrator

**Files:**
- Modify: `crates/agent/src/orchestrator.rs`

- [ ] **Step 1: Locate the existing pipeline**

Find the `process_query` function and the sequence of steps `context_builder → agent_runner → evidence_fuser → grounding_gate → result_persister`.

- [ ] **Step 2: Insert calls**

```rust
// Step 1: Build context (existing)
self.context_builder.build(&mut ctx, history).await?;

// Step 1.5 (NEW): Coverage plan
self.coverage_planner.plan(&mut ctx).await;

// Step 2: Agent run (existing)
self.agent_runner.run(&mut ctx).await?;

// Step 3: Fuse evidence (existing, modified to preserve coverage tier)
self.evidence_fuser.fuse(&mut ctx);

// Step 3.5 (NEW): Verify coverage
self.coverage_verifier.verify(&mut ctx);

// Step 4: Grounding gate (existing)
self.grounding_gate.verify_and_retry(&mut ctx, ...).await?;

// Step 5: Persist (existing)
self.result_persister.persist(&mut ctx).await?;
```

Wire dependencies in `AgentOrchestrator::new`.

- [ ] **Step 3: Hand off — user runs**

```bash
cargo build -p agent_rs_agent
```

- [ ] **Step 4: Commit**

---

### Task 4.5: Record tool calls in `agent_runner`

**Files:**
- Modify: `crates/agent/src/steps/agent_runner.rs`

- [ ] **Step 1: Failing test**

```rust
#[tokio::test]
async fn agent_runner_records_tool_invocations_into_ctx() {
    // mock LLM agent that issues one search_knowledge call;
    // assert ctx.tool_calls contains the recorded args.
    // (use rig's mock agent or stub the rig::agent trait)
}
```

(If mocking the `rig` agent is heavy, skip the unit test and rely on Phase 6 E2E to validate.)

- [ ] **Step 2: Hook into the rig instrumentation**

In `agent_runner::run`, after each tool call returns (using `rig`'s instrumentation hook or a wrapper), append to `ctx.tool_calls`:

```rust
ctx.tool_calls.push(crate::coverage::ToolCallRecord {
    tool_name: tool_name.to_string(),
    args_json: serde_json::to_string(&args).unwrap_or_default(),
});
```

- [ ] **Step 3: Hand off & commit**

---

### Task 4.6: Register new tools in `tool_factory`

**Files:**
- Modify: `crates/agent/src/steps/tool_factory.rs`

- [ ] **Step 1: Add the three new tool registrations**

```rust
// In ToolFactory::build:
agent_builder = agent_builder.tool(OutlineCorpusTool::new(neo4j.clone(), pg.clone()));
agent_builder = agent_builder.tool(FetchSectionChunksTool::new(neo4j.clone(), qdrant.clone(), default_collection.clone()));
agent_builder = agent_builder.tool(SearchSectionsTool::new(neo4j.clone(), qdrant.clone(), default_collection.clone()));
```

- [ ] **Step 2: Hand off & commit**

---

### Task 4.7 (CHECKPOINT): Codex 5.4-high review of Phase 4

Same pattern as Task 2.6. Prompt:

```
Review phase 4 of the coverage-aware-rag implementation. Files:
- crates/agent/src/steps/coverage_planner.rs (new)
- crates/agent/src/steps/coverage_verifier.rs (new)
- crates/agent/src/steps/evidence_fuser.rs (preserve coverage tier)
- crates/agent/src/orchestrator.rs (pipeline wiring)
- crates/agent/src/steps/agent_runner.rs (record tool calls)
- crates/agent/src/steps/tool_factory.rs (register new tools)

Verdict: is the new pipeline correctly ordered, race-free, and does it
degrade gracefully on non-exhaustive queries? Specifically:
1. CoveragePlanner returns early when intent is None — confirm zero
   added latency for definition queries
2. CoverageVerifier reads ctx.coverage_evidence + ctx.evidence —
   confirm no double-counting of section_ids
3. EvidenceFuser dedup vs coverage merge — any chance of duplicate
   chunks reaching synthesis?
4. agent_runner tool-call recording — is args_json captured before or
   after redaction? Could it leak secrets?
5. tool_factory: are Arc clones cheap or do they hold heavy state?

Be specific; cite file:line. Under 600 words.
```

---

## Phase 5 — Gate + synthesis fixes

### Task 5.1: Replace `subject_absent_from_evidence` with v2

**Files:**
- Modify: `crates/agent/src/steps/grounding_gate.rs`

- [ ] **Step 1: Failing tests**

```rust
#[cfg(test)]
mod tests_v2 {
    use super::*;

    fn dummy_evidence(text: &str) -> SearchResult { /* … */ }
    fn dummy_tool_call(query: &str) -> ToolCallRecord {
        ToolCallRecord {
            tool_name: "search_knowledge".into(),
            args_json: format!(r#"{{"query":"{}"}}"#, query),
        }
    }

    #[test]
    fn instruction_heavy_query_is_no_longer_false_positive() {
        let raw_query = "vreau cheatsheet cu TOATE metodele de segmentare punctaj scriu organizat";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă."
        )];
        let tool_calls = vec![dummy_tool_call("segmentare metode algoritmice")];
        // v2 must extract "segmentare" / "metode" subject from rewritten
        // query (in tool_calls) and find them in evidence -> NOT absent.
        assert!(!subject_absent_v2(raw_query, &tool_calls, &evidence));
    }

    #[test]
    fn out_of_corpus_still_short_circuits() {
        let raw_query = "ce este React?";
        let evidence = vec![dummy_evidence(
            "Metoda Otsu este o metodă automată de segmentare pe histogramă."
        )];
        let tool_calls = vec![dummy_tool_call("react javascript framework")];
        assert!(subject_absent_v2(raw_query, &tool_calls, &evidence));
    }

    #[test]
    fn falls_back_safely_when_no_tool_calls_recorded() {
        let raw_query = "anything";
        let evidence = vec![dummy_evidence("...")];
        // No tool calls -> v2 returns false (do not short-circuit).
        assert!(!subject_absent_v2(raw_query, &[], &evidence));
    }
}
```

- [ ] **Step 2: Implement v2**

```rust
fn extract_subject_phrase(args_json: &str) -> Option<String> {
    // Parse the rewritten query out of the JSON args.
    // Supports search_knowledge, search_sections, fetch_section_chunks.
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    v.get("query").and_then(|q| q.as_str()).map(|s| s.to_string())
}

pub(crate) fn subject_absent_v2(
    _raw_query: &str,
    tool_calls: &[crate::coverage::ToolCallRecord],
    evidence: &[SearchResult],
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
    // For each rewritten query, take the longest content noun ≥4 chars
    // and require its case-folded NFC form to appear as an exact
    // substring of the haystack. Subject is "absent" only if NO query's
    // primary subject is present.
    queries.iter().all(|q| {
        let primary = primary_noun(q);
        primary.map_or(false, |p| !haystack.contains(&p))
    })
}

fn primary_noun(q: &str) -> Option<String> {
    let mut tokens: Vec<&str> = q.split(|c: char| !c.is_alphabetic())
        .filter(|t| t.chars().count() >= 4)
        .collect();
    // Heuristic: pick the longest token (typical noun in a rewritten
    // search query); falls back to None if nothing ≥4.
    tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));
    tokens.first().map(|s| s.to_lowercase())
}

fn build_haystack_lc(ev: &[SearchResult]) -> String {
    let mut s = String::new();
    for e in ev {
        if let Some(t) = e.payload.get("chunk_text").and_then(|v| v.as_str()) {
            s.push_str(&t.to_lowercase());
            s.push(' ');
        }
    }
    s
}
```

Wire `verify_and_retry` to call `subject_absent_v2(&ctx.query, &ctx.tool_calls, &ctx.evidence)` instead of `subject_absent_from_evidence`. Keep the old function private and `#[deprecated]` for one cycle so no external caller silently breaks.

- [ ] **Step 3: Hand off & commit**

---

### Task 5.2: Stable `cite_id` in synthesis

**Files:**
- Modify: `crates/agent/src/synthesis.rs`

- [ ] **Step 1: Failing test**

```rust
#[test]
fn cite_uuid_resolves_back_to_evidence() {
    let evidence = vec![
        ev_with_chunk_id("c-1234abcd"),
        ev_with_chunk_id("c-5678ef01"),
    ];
    let answer = "Otsu separates classes [cite:c-1234abcd]. \
                  Bhattacharya is gaussian mixture-based [cite:c-5678ef01].";
    let cites = resolve_cite_markers(answer, &evidence);
    assert_eq!(cites.len(), 2);
    assert_eq!(cites[0].chunk_id, "c-1234abcd");
}

#[test]
fn legacy_bracket_index_still_resolves_during_transition() {
    let evidence = vec![ev_with_chunk_id("c-1234")];
    let answer = "Otsu separates classes [1].";
    let cites = resolve_cite_markers(answer, &evidence);
    assert_eq!(cites.len(), 1);
}
```

- [ ] **Step 2: Implement `resolve_cite_markers`**

Add to `synthesis.rs`:

```rust
fn resolve_cite_markers(answer: &str, evidence: &[SearchResult]) -> Vec<Citation> {
    static UUID_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\[cite:([A-Za-z0-9-]+)\]").unwrap());
    static IDX_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\[(\d+)\]").unwrap());

    let mut out: Vec<Citation> = Vec::new();
    // UUID-based citations first.
    for cap in UUID_RE.captures_iter(answer) {
        let id = &cap[1];
        if let Some((idx, ev)) = evidence.iter().enumerate()
            .find(|(_, e)| e.payload.get("chunk_id").and_then(|v| v.as_str()) == Some(id)) {
            out.push(make_citation(idx + 1, ev));
        } else {
            tracing::warn!(cite_id = %id, "cite marker references unknown chunk");
        }
    }
    // Backwards-compat: legacy [N] markers.
    if out.is_empty() {
        for cap in IDX_RE.captures_iter(answer) {
            if let Ok(n) = cap[1].parse::<usize>() {
                if let Some(ev) = evidence.get(n.saturating_sub(1)) {
                    out.push(make_citation(n, ev));
                }
            }
        }
    }
    out
}
```

Replace the call to `find_referenced_indices` + the citation collection block with a call to `resolve_cite_markers`.

- [ ] **Step 3: Drop hedging wrapper when `coverage_report` exists**

In the synth body, change the existing `else if !verification.is_grounded` branch:

```rust
} else if !verification.is_grounded {
    if response_uses_coverage_path(verification, ctx_coverage_report) {
        // Coverage path produces its own truthy answer; struct fields
        // expose is_grounded=false + the report. No prose wrapper.
        answer.to_owned()
    } else if answer_self_admits_scope(answer) {
        answer.to_owned()
    } else {
        format!("I found some potentially relevant information, …\n\n{}", answer)
    }
}
```

Add `coverage_report: Option<CoverageReport>` to the struct exposed via `SynthesizedResponse`.

- [ ] **Step 4: Hand off & commit**

---

### Task 5.3: Update `prompts.rs` exhaustive-mode state machine

**Files:**
- Modify: `crates/agent/src/prompts.rs`

- [ ] **Step 1: Locate the "exactly one tool" decision-table block (~line 23)**

- [ ] **Step 2: Replace with the new state machine snippet**

```text
TOOL ROUTING

Default discipline (single-call): for definition, how-to, lookup, comparative
queries, issue exactly ONE search_knowledge call with a focused query, then
synthesize. Do not loop.

EXHAUSTIVE MODE (when context shows coverage_plan.mode = Exhaustive):
1. The coverage planner has already enumerated target_sections.
2. Issue exactly ONE search_sections call with section_ids = the planner's
   list. Do NOT call search_knowledge first; the section list is authoritative.
3. After search_sections returns, synthesize from the merged evidence.
4. If a section came back empty (marked "(no chunks)"), you MAY issue ONE
   targeted follow-up search_knowledge with the section heading as query.

BUDGET: max 5 turns total. Exhaustive path uses 2-3 by design.
```

Wire `ctx.coverage_plan` into the prompt context (the agent prompt builder probably already has a context-injection point — extend it to surface `mode` + `target_sections.len()`).

- [ ] **Step 3: Hand off & commit**

---

## Phase 6 — Final verification + E2E

### Task 6.1: Quality gate (user runs)

- [ ] **Step 1: Hand off — user runs the full quality gate**

```bash
cd /Users/cosmingagea/workspace/agent-rs && \
cargo fmt --all -- --check && \
cargo clippy --workspace --all-targets -- -D warnings && \
cargo test --workspace --lib
```

If any gate fails, address before continuing. Suggested order: clippy → tests → fmt.

### Task 6.2: Re-ingest the canonical PDF

- [ ] **Step 1: Hand off — user re-ingests `segmentare.pdf`**

```bash
cd /Users/cosmingagea/workspace/agent-rs && \
cargo run --release --example ingest -- segmentare.pdf
```

Expected: success; new section ordering fields populated; new summary fields populated.

### Task 6.3: Re-run the canonical query

- [ ] **Step 1: Hand off — user runs**

```bash
cd /Users/cosmingagea/workspace/agent-rs && \
cargo run --release --example ask -- "vreau cheatsheet cu TOATE metodele de segmentare" 2>&1 | tee /tmp/after-canonical.log
```

Expected from `/tmp/after-canonical.log`:
- `is_grounded: true`
- `confidence: > 0.5`
- Answer body mentions: Otsu, Bhattacharya, prag optim, histograma ponderată, **histograma cumulativă** (NEW), **5.1.1 Creșterea regiunilor** (NEW), **5.1.2 Etichetarea imaginilor binare** (NEW), **Segmentarea orientată pe regiuni** (NEW), clustering
- No "I'm not confident…" wrapper
- Tracing log emits `coverage verifier complete fetched=N expected=N missing=0`

### Task 6.4: Run regression queries

- [ ] **Step 1: Hand off — user runs each in turn**

```bash
cargo run --release --example ask -- "ce este metoda Otsu"               # definition — expect unchanged latency, grounded
cargo run --release --example ask -- "cum aplic Otsu pas cu pas"         # how-to    — expect grounded, scaffold-style answer
cargo run --release --example ask -- "ce este React?"                    # OOC       — expect "not in knowledge base"
cargo run --release --example ask -- "compara Otsu cu Bhattacharya"      # comparative — expect grounded with both
```

Claude reads each log + answer + the relevant pages in `segmentare.pdf` and reports per-query verdict.

### Task 6.5 (CHECKPOINT): Final Codex 5.4-high pass

Prompt:

```
Final review: coverage-aware-rag enterprise fix is complete.
Branch: agent-reasoning-effort-minimal. Commits since c169716.

Verdict in 3 parts:
1. Per-query-category correctness: list-all, definition, how-to,
   comparative, relational, numerical, negation, multi-document,
   out-of-corpus. Cite specific files/functions you would suspect
   to break each category.
2. Hidden regressions: anything in the diff that could degrade
   existing queries (especially short definition queries).
3. Operability: are the new metrics (coverage_planner_unmatched,
   coverage_verified) wired correctly into the existing prometheus
   exporter? Are the WARN logs structured correctly for grafana
   ingestion?

Under 800 words. Cite file:line. Use the same `codex exec --full-auto
--skip-git-repo-check -m gpt-5.4 -c model_reasoning_effort=high`
invocation pattern as previous checkpoints.
```

---

## Self-review notes

After writing this plan, scanned against the spec:

- Spec §6.1 row `crates/stores/src/qdrant/section_filter.rs` → **covered** by Task 2.2.
- Spec §6.1 row `crates/agent/src/steps/coverage_planner.rs` → **covered** by Task 4.1; metric + log requirement also reflected.
- Spec §6.2 row `crates/agent/src/synthesis.rs` (cite_id + drop hedge when coverage_report exists) → **covered** by Task 5.2.
- Spec §6.2 row `crates/agent/src/steps/grounding_gate.rs` (subject_absent_v2 over rewritten queries) → **covered** by Task 5.1.
- Spec §6.2 row `crates/retrieval/src/router.rs` (hard predicate keyword + graph) → **covered** by Task 3.6.
- Spec §6.2 row `crates/retrieval/src/ingestion/text_normalize.rs` (unconditional fold + query side) → **covered** by Task 2.3.
- Spec §6.2 row `crates/stores/src/neo4j/structural.rs` (preamble + ordering) → **covered** by Task 2.4.
- Spec §6.2 row `crates/retrieval/src/ingestion/pipeline.rs` (summary payload + rollback) → **covered** by Task 2.5.
- Spec §6.2 row `crates/agent/src/steps/evidence_fuser.rs` (preserve coverage tier) → **covered** by Task 4.3.
- Spec §6.2 row `crates/agent/src/steps/agent_runner.rs` (record tool calls) → **covered** by Task 4.5.
- Spec §6.2 row `crates/agent/src/steps/tool_factory.rs` (register new tools) → **covered** by Task 4.6.
- Spec §6.2 row `crates/tools/src/knowledge.rs` (extend args) → **covered** by Task 3.4.
- Spec §6.2 row `crates/tools/src/document_info.rs` (resolve_document) → **covered** by Task 3.5.
- Spec §6.2 row `crates/tools/tests/tool_contract_test.rs` (contracts) → **covered** by Tasks 3.1-3.3 (each tool task adds its contract test).
- Spec §6.2 row `crates/agent/src/orchestrator.rs` (wire steps) → **covered** by Task 4.4.
- Spec §6.2 row `crates/agent/src/context.rs` (new fields) → **covered** by Task 1.2.
- Spec §6.2 row `crates/agent/src/prompts.rs` (state machine) → **covered** by Task 5.3.

Ambiguity check:
- Subject extraction in 5.1 picks the longest token as primary noun. This is a heuristic — for Romanian compound topics ("rețele neuronale convoluționale") we'd want bigram extraction. **Trade-off accepted** for v1; add a follow-up task if user reports gaps.
- `lookup_section_summary` placeholder in 3.1 has an explicit TODO before the concrete impl is shown directly below. This is intentional documentation, not a plan failure.

Type consistency check:
- `CoveragePlan`, `CoverageReport`, `SectionRef`, `ToolCallRecord` defined in 1.1, used identically in 1.2, 4.1, 4.2, 4.3, 5.1.
- `SearchMode::Ranked` / `AllInSection` defined in 3.4, used in 3.6 and 4.6.
- `synthetic_preamble_section_id` defined in 2.4, used by ingestion in same task; matches the spec's `preamble:<doc_id>` format.

---

## Execution handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-04-19-coverage-aware-rag.md`. **Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (or per parallel group: Phase 2.1+2.2+2.3 in parallel; Phase 3.1+3.2+3.3 in parallel), review between tasks, two-stage review at each Codex checkpoint. Best for this plan's mix of independent file groups and tight TDD discipline.

**2. Inline Execution** — I work through tasks in this session using `superpowers:executing-plans`, with batch checkpoints at the three Codex review steps. Better if you want to watch the work happen step by step.

**Which approach?**
