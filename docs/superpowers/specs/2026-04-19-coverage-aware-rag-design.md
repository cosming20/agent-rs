# Coverage-Aware RAG: Enterprise Fix for "Perfect on Documents"

**Status:** Draft for review
**Author:** orchestrated brainstorm (user + Claude + Codex 5.4-high audit)
**Date:** 2026-04-19
**Topic:** consolidated multi-prong fix for grounding-gate false negatives, retrieval coverage gaps, and intent-aware tool fan-out
**Repo:** `/Users/cosmingagea/workspace/agent-rs`

---

## 1. Problem statement

A Romanian-language query against `segmentare.pdf` (chapter 5 on image segmentation) returned an ungrounded answer with `is_grounded=false`, `confidence=0.000`, and a hedging wrapper, despite the retrieved evidence being clearly on-topic. End-to-end trace and corpus inspection revealed four independent failures that combine into the observed symptom:

1. **Grounding-gate false positive.** `subject_absent_from_evidence` (`crates/agent/src/steps/grounding_gate.rs:66`) requires every ≥4-char content token from the **raw** user query (minus a small stopword list) to appear in the evidence haystack. The query contained instruction-meta tokens — `cheatsheet`, `punctaj`, `organizat`, `scriu`, `gândește`, `învăț`, `rapid`, `test` — none of which can appear in any image-processing textbook. A single missing token short-circuits retries and forces `confidence=0`, then `synthesis.rs:142` adds the "I'm not confident…" wrapper to a structurally correct answer.
2. **Retrieval coverage gap.** The agent issued a single `search_knowledge` call with `limit=10`. After RRF fusion + cross-encoder rerank → 10 chunks. Cross-checking against `pdftotext segmentare.pdf` shows the chapter contains **at least four sections that were not retrieved**: 5.1.1 *Creșterea regiunilor* (region growing), 5.1.2 *Etichetarea imaginilor binare* (binary labeling), *Segmentarea pe histograma cumulativă*, and the umbrella paradigm *Segmentarea orientată pe regiuni*. The user explicitly asked for **TOATE** (ALL) methods, so omitting four of them is a correctness failure, not a relevance failure.
3. **No mechanism for the agent to discover document structure.** `DocumentInfoTool` (`crates/tools/src/document_info.rs`) returns top-level metadata only (page count, language, title). The agent has no way to enumerate sections or read per-section summaries before retrieval. Section structure exists in Neo4j and per-chunk summaries exist in a dedicated Qdrant collection, but they are not exposed to the LLM.
4. **No coverage proof.** The pipeline has no notion of "expected sections covered vs missing". Even if the agent stitched together five tool calls, nothing would warn that section 5.1.1 was never touched.

A previous attempt at LLM-driven query augmentation (`QueryAnalyzer`, commit `5905dca`) was reverted twice (`032baa6`, `e887b94`); independent Codex 5.4-high analysis identifies the cause as **diluted retrieval** from generated sub-queries (Hit@1 dropped on `graph_hybrid`) and a false "zero added latency" claim. We must not repeat that pattern.

## 2. Goals

- Achieve "perfect on documents" answer quality across the nine query categories listed in §3.
- Eliminate grounding-gate false positives on instruction-heavy queries while keeping out-of-corpus queries honestly closed.
- Provide deterministic coverage signal for exhaustive ("list-all") queries — the system must be able to *prove* it covered every relevant section.
- Backwards-compatible: existing flows (definition queries, single-shot facts) must not regress in latency or quality.

## 3. Query categories the design must handle

The following categories were validated against the canonical corpus and Codex audit. Each is a first-class success criterion.

| # | Category | Example (RO) | Primary mechanism in this design |
|---|----------|--------------|----------------------------------|
| 1 | list-all / cheatsheet | "TOATE metodele de segmentare" | CoveragePlanner + `outline_corpus` + `search_sections` (batch) |
| 2 | definition | "ce este Otsu" | Existing `search_knowledge` (unchanged) |
| 3 | procedural / how-to | "cum aplic Otsu pas cu pas" | `search_knowledge` + intent-aware scaffold (already in place) |
| 4 | comparative | "care e mai bună Otsu sau Bhattacharya" | `search_sections` over both targets + numeric-aware synth |
| 5 | relational | "care e relația 5.1.1 ↔ 5.2.2" | Graph traversal via existing `graph_hybrid` |
| 6 | numerical | "care metodă are eroarea cea mai mică" | `fetch_section_chunks` + verbatim-numeric prompt rule (already in place) |
| 7 | negation | "ce nu se poate face cu X" | `outline_corpus` for breadth + verifier-led elimination |
| 8 | multi-document | "compară X din doc A cu Y din doc B" | `resolve_document(name)` + per-doc outline |
| 9 | out-of-corpus | "ce e React?" against image-processing corpus | Tightened gate (entity-presence, not prefix-overlap) |

## 4. Non-goals

- Re-introducing an LLM-driven `QueryAnalyzer` for query augmentation (rejected per Codex audit of revert).
- Multi-modal retrieval (images, audio).
- Cross-corpus federation across separate Qdrant collections.
- Embedding model fine-tuning.
- Performance optimization beyond functional correctness; latency budget is "no worse than current for non-exhaustive queries; ≤2× current for exhaustive queries".

## 5. Design overview — Approach A+

The design moves coverage orchestration **outside the LLM** and exposes document structure to the agent via two new tools, while tightening the grounding gate and extending the evidence pipeline to support coverage proofs. Five mechanisms compose:

1. **`CoveragePlanner`** (new pipeline step before `AgentRunner`) — detects exhaustive intent deterministically and pre-builds a `CoveragePlan` with target sections. The agent sees the plan as part of its context, rather than being asked to discover and orchestrate it within `max_turns=4`.
2. **`outline_corpus`** (new tool) — hierarchical TOC over Neo4j `Document → Section`, joined with the existing summaries Qdrant collection. Optional `doc_id` filter, optional `query_relevance` sort.
3. **`fetch_section_chunks`** (new tool) — exhaustive, no-rank retrieval. Returns every chunk in a section (and optionally its descendants), ordered by `chunk_index`. Bypasses ANN/BM25 ranking entirely.
4. **`search_sections`** (new tool) — batched variant: one tool call accepts a list of `section_id`, returns evidence per section. Lets the agent fan out within a single LLM turn.
5. **Stable citation IDs** + **separate `coverage_evidence` pack** + **gate fix** + **summary schema fix** — covered in detail in §7 and §8.

**Tuning constants** changed by this design (`crates/core/src/config.rs`):

- `max_turns` raised from `4` to `5`. Rationale: with `SearchSectionsTool` collapsing fan-out into one call, the typical exhaustive path is `outline_corpus → search_sections → synthesize` (3 turns), but headroom is needed for the agent to issue one targeted follow-up `search_knowledge` (e.g., to disambiguate a section that came back empty) without hitting the cap. Non-exhaustive flows still finish in ≤2 turns.
- `reasoning_effort` stays `minimal`. The new tools eliminate the need for the model to reason about decomposition.

### Pipeline before / after

```
BEFORE:
  ContextBuilder → AgentRunner → EvidenceFuser → GroundingGate → ResultPersister

AFTER:
  ContextBuilder → CoveragePlanner → AgentRunner → EvidenceFuser → GroundingGate
                                                  ↘ CoverageVerifier ↗
                                                                     → ResultPersister
```

`CoveragePlanner` is a no-op for non-exhaustive intents — definition / how-to / numerical queries skip it entirely.

## 6. Components — file-by-file

### 6.1 New code

| File | Crate | Responsibility |
|------|-------|----------------|
| `crates/agent/src/steps/coverage_planner.rs` | `agent` | Detects exhaustive intent (rules-based + intent-prompt signal). Builds `CoveragePlan { target_sections: Vec<SectionRef>, mode: Exhaustive \| Partial \| None }`. **Telemetry:** when the rules do *not* match but a soft heuristic flags the query as likely-exhaustive (e.g. ≥2 quantifier tokens, presence of "list/all/full" without exact rule match, or LLM-side intent classification disagrees with rule output), emit a structured `WARN` log `coverage_planner.likely_exhaustive_unmatched` with the query text, the soft heuristic that fired, and the chosen `mode`. Increment Prometheus counter `agent_coverage_planner_unmatched_total{reason="<heuristic>"}`. This gives an observability hook for tuning the rule set over time. |
| `crates/agent/src/coverage.rs` | `agent` | `CoveragePlan`, `CoverageReport`, `SectionRef` types. Functions to merge per-section evidence into a `coverage_evidence` set that bypasses `EvidenceFuser` truncation. |
| `crates/agent/src/steps/coverage_verifier.rs` | `agent` | Runs after `EvidenceFuser`. Compares `CoveragePlan.target_sections` against sections present in `ctx.coverage_evidence`. Emits `CoverageReport { expected: N, fetched: M, empty: K, missing_section_ids: Vec<_> }`. Affects gate decision and synthesis hedging. |
| `crates/tools/src/outline_corpus.rs` | `tools` | `OutlineCorpusTool`. Args: `{ doc_id: Option<String>, query_relevance: Option<String>, max_sections: Option<usize> }`. Output: hierarchical text formatted as `Document → Section (id) → 1-line summary`. |
| `crates/tools/src/fetch_section_chunks.rs` | `tools` | `FetchSectionChunksTool`. Args: `{ section_id: String, include_descendants: bool, max_chars: Option<usize> }`. Output: every chunk in section (descendants traversed via Neo4j `PARENT_OF`), ordered by `chunk_index`. **Does not rerank.** |
| `crates/tools/src/search_sections.rs` | `tools` | `SearchSectionsTool`. Args: `{ section_ids: Vec<String>, query: Option<String>, per_section_limit: Option<usize> }`. Calls `fetch_section_chunks` per id (in parallel, bounded by semaphore) and merges results. |
| `crates/stores/src/neo4j/coverage.rs` | `stores` | `list_sections_for_doc(doc_id) -> Vec<SectionRow>`, `list_descendant_sections(section_id) -> Vec<String>`, `chunks_in_sections(section_ids) -> Vec<ChunkId>`. |
| `crates/stores/src/qdrant/section_filter.rs` | `stores` | Helper to build a Qdrant `Filter::must` clause keyed on `section_id` (and `doc_id`). Reusable across the new tools and the extended `search_knowledge`. |

### 6.2 Modified code

| File | Change |
|------|--------|
| `crates/agent/src/steps/grounding_gate.rs` | Replace `subject_absent_from_evidence` heuristic. New input source: `ctx.tool_calls[].args.query` (the rewritten queries the LLM actually issued). New algorithm: extract head noun phrase per query, require **exact phrase presence** in evidence haystack (case-folded, NFC-normalized). Falls through to retry rather than short-circuit when *any* tool query's subject is found. |
| `crates/agent/src/synthesis.rs` | (a) Use stable `cite_id = chunk_id` (UUID already in the Qdrant payload as `chunk_id`) instead of the answer-relative `[1] [2] [3]` index. The LLM is instructed (via prompt update) to cite as `[cite:<short-uuid>]`; the synthesizer resolves `cite:` markers back to evidence. Backwards-compatible: `[1]` markers still resolve via the existing index path during the transition. (b) Drop the "I'm not confident…" hedging wrapper unconditionally when `coverage_report.is_some()`. The answer text stays clean — no coverage footer appended. Coverage information is exposed only via the `SynthesizedResponse` struct fields (`is_grounded`, `confidence`, new optional `coverage: CoverageReport`) and via OpenTelemetry metrics; it never leaks into the user-facing answer body. |
| `crates/agent/src/prompts.rs` | Replace the "exactly one tool / do not loop" decision-table block (currently around line 23) with an exhaustive-mode state machine: when `ctx.coverage_plan.mode == Exhaustive`, the agent must call `search_sections(plan.target_sections)` exactly once. Non-exhaustive flows keep the current single-call discipline. |
| `crates/agent/src/orchestrator.rs` | Wire `CoveragePlanner` after `ContextBuilder` and `CoverageVerifier` after `EvidenceFuser`. |
| `crates/agent/src/context.rs` | `QueryContext` gains `coverage_plan: Option<CoveragePlan>`, `coverage_evidence: Vec<SearchResult>`, `coverage_report: Option<CoverageReport>`, and `tool_calls: Vec<ToolCallRecord>` (capturing args for the gate). |
| `crates/agent/src/steps/agent_runner.rs` | Record each tool invocation's args + result into `ctx.tool_calls`. Already has the hooks via `rig` instrumentation. |
| `crates/agent/src/steps/tool_factory.rs` | Register `OutlineCorpusTool`, `FetchSectionChunksTool`, `SearchSectionsTool`. |
| `crates/agent/src/steps/evidence_fuser.rs` | When `ctx.coverage_evidence` is populated, **do not truncate** it to `limit`; merge it into `ctx.evidence` as a separate, unranked tier. The reranker still applies to the ANN-derived tier only. |
| `crates/tools/src/knowledge.rs` | `SearchKnowledgeArgs` gains optional `section_id: Option<String>`, `doc_id: Option<String>`, `mode: Option<SearchMode>` (`Ranked` default, `AllInSection` bypasses ranking). Schema fields are additive — existing callers unaffected. |
| `crates/tools/src/document_info.rs` | Add `resolve_document(name_or_alias)` helper used by `outline_corpus` and `search_knowledge`. |
| `crates/retrieval/src/router.rs` | Apply `doc_id` / `section_id` as a **hard post-fusion predicate** so the keyword and graph branches honor the filter (currently only the vector branch does — see Codex gap #10). |
| `crates/retrieval/src/ingestion/pipeline.rs` | `ChunkSummary` payload gains `doc_id`, `section_id`, `ingest_run_id`. Rollback (`pipeline.rs:1507`) extends to delete from summaries collection by `ingest_run_id`. Failed/empty summaries are recorded with `summary_status: Empty` in payload (instead of being silently dropped) so `outline_corpus` can show `[no summary]` rather than a hole. |
| `crates/retrieval/src/ingestion/text_normalize.rs` | Make Romanian comma-below ↔ cedilla folding **unconditional** (currently env-gated, see line 239). Apply same fold to query text in `crates/retrieval/src/embedder.rs` so query and corpus share a search form. |
| `crates/stores/src/neo4j/structural.rs` | (a) `Section` rows store `first_chunk_index`, `page_start`, `section_order` for deterministic outline ordering. (b) Chunks without an explicit section attach to a synthetic section with `heading = "(preamble)"` and a new boolean property `is_preamble = true` (rather than being skipped at line 610). The synthetic section gets a stable derived id of `preamble:<doc_id>` so re-ingest is idempotent. |
| `crates/tools/tests/tool_contract_test.rs` | Contract tests for the three new tools: name, description, schema, deserialization. |

## 7. Data flow — per query category walkthrough

### 7.1 List-all / cheatsheet (the canonical case)

```
1. user query → CoveragePlanner detects exhaustive intent
   (rules: tokens like {TOATE, all, list, enumerate, cheatsheet, complet,
   exhaustive} + LLM intent classification already in prompt)
2. CoveragePlanner.build_plan(query, corpus):
     - extract topic noun phrase (existing entity_linker)
     - call outline_corpus(query_relevance=topic) → ranked sections
     - select top N sections above relevance threshold
     - emit CoveragePlan { target_sections: [5.1, 5.1.1, 5.1.2, 5.2,
                          5.2.1, 5.2.2, 5.3], mode: Exhaustive }
3. AgentRunner sees plan in context; system prompt routes to
   exhaustive-mode state machine → ONE tool call:
     search_sections(section_ids=[…7 ids…], query=topic)
4. SearchSectionsTool issues parallel fetch_section_chunks per id;
   results merged into ctx.coverage_evidence (no truncation).
5. EvidenceFuser leaves coverage_evidence intact, optionally adds
   ANN-derived chunks as supplementary tier.
6. CoverageVerifier compares plan vs coverage_evidence:
     CoverageReport { expected: 7, fetched: 7, empty: 0, missing: [] }
7. GroundingGate: subject extracted from the actual tool args
   ("segmentare metode algoritmice"), entity phrase present in
   evidence → grounded.
8. Synthesis emits answer with stable citations + coverage footer.
```

### 7.2 Definition (unchanged path)

```
1. CoveragePlanner: mode=None (no exhaustive triggers).
2. AgentRunner: single search_knowledge call (current behavior).
3. EvidenceFuser, GroundingGate, ResultPersister: unchanged.
```

### 7.3 Multi-document comparative

```
1. CoveragePlanner detects two doc references via resolve_document.
2. Plan: target_sections grouped per doc.
3. Agent calls search_sections once per doc (≤4 turns OK with two).
4. Coverage report shows per-doc completeness.
```

### 7.4 Out-of-corpus

```
1. CoveragePlanner: mode=None.
2. AgentRunner: single search_knowledge → low-score evidence.
3. GroundingGate: extracted topic entity not present as exact phrase
   anywhere in evidence → SubjectAbsent → skip retries (preserved
   from current behavior, but with phrase check, not prefix overlap).
4. Synthesis: honest "not in knowledge base" message.
```

## 8. Detailed mitigation per Codex-identified gap

| # | Gap | Mitigation in this design |
|---|-----|---------------------------|
| 1 | `section_id` filter still top-K | `FetchSectionChunksTool` with `mode=AllInSection`, no rerank |
| 2 | `EvidenceFuser` truncates coverage chunks | Separate `coverage_evidence` tier in `QueryContext`, untruncated |
| 3 | Citation numbering breaks across calls | Stable `cite_id` from chunk UUID emitted by tool, resolved by id in synthesis |
| 4 | Cannot fit fan-out in 4 turns | `SearchSectionsTool` is a single tool call that fans out internally |
| 5 | Prompt forbids looping / multiple tools | `prompts.rs` rewrite: exhaustive-mode state machine, non-exhaustive flows keep single-call discipline |
| 6 | Section ordering + preamble | `Section` schema gains `first_chunk_index, page_start, section_order`; synthetic `Preamble` section captures unsectioned chunks |
| 7 | Parent vs subsection semantics | `include_descendants` in `FetchSectionChunksTool`; Neo4j `PARENT_OF` traversal in `coverage.rs` |
| 8 | Summary collection lacks identity / freshness | `ChunkSummary` payload gains `doc_id, section_id, ingest_run_id`; rollback extended to summaries collection |
| 9 | Empty/failed summaries create silent holes | `summary_status: Empty` recorded; `outline_corpus` displays `[no summary]` placeholder |
| 10 | Filters not hard across all branches | Post-fusion hard predicate in `router.rs` for `doc_id` / `section_id` |
| 11 | No deterministic coverage proof | `CoverageVerifier` + `CoverageReport` |
| 12 | Gate over-correction | Phrase-presence on rewritten subqueries instead of prefix-overlap on raw query; tool-args persisted in `ctx.tool_calls` |
| 13 | Multi-document identity unresolved | `resolve_document(name)` helper in `document_info.rs` |
| 14 | Numerical/table answers | Out of scope for this spec; tracked separately. `has_table` metadata exposed in evidence so synth can flag "structured table omitted, see source". |
| 15 | Romanian Unicode partial | `text_normalize.rs` comma-below fold made unconditional; query side normalized in `embedder.rs` |
| 16 | Tool registration & contracts missing | `tool_factory.rs` + `tool_contract_test.rs` extended |

## 9. Error handling

| Failure | Fallback |
|---------|----------|
| `outline_corpus` Neo4j query fails | Tool returns explicit error string; agent prompt instructs fall-through to `search_knowledge` |
| `fetch_section_chunks` returns 0 chunks | Marked `coverage_status: Empty` in `CoverageReport`; verifier downgrades grounded confidence proportionally |
| `CoveragePlanner` cannot detect intent | `mode=None` (default) → existing pipeline behavior preserved |
| Summary join encounters orphan chunks | Outline shows `[summary unavailable]`; not a hard failure |
| `section_id` filter matches no chunks | Tool returns "no chunks found in section X"; coverage report flags missing |
| Gate's tool-args extraction finds no rewritten query | Returns `subject_absent = false` (i.e. "let the pipeline continue, do not short-circuit"). This is safer than reverting to the broken prefix-AND heuristic; the trade-off is one extra retry on a pathological case where no tool was called yet evidence exists. |
| Re-ingest leaves orphan summaries | Detected by `ingest_run_id` mismatch; `outline_corpus` filters them out and logs a warning |

## 10. Testing strategy

### 10.1 Unit tests (in-tree, `cargo test`)

| Module | Test |
|--------|------|
| `coverage_planner` | exhaustive triggers (RO + EN: TOATE, all, list, cheatsheet, enumerează, full list); non-triggers (definition, how-to); edge case (TOATE inside a quote); **soft-heuristic-fired-but-rule-missed** path emits the WARN log and increments the unmatched counter |
| `coverage::CoveragePlan` | merge / dedupe / target-section ordering |
| `coverage_verifier` | full coverage / partial / empty cases produce correct reports |
| `outline_corpus` | doc filter / query-relevance sort / max_sections cap / `[no summary]` placeholder |
| `fetch_section_chunks` | descendants on/off / chunk ordering / empty section |
| `grounding_gate::subject_absent_from_evidence` (renamed `subject_absent_v2`) | instruction-token query no longer false-positives; out-of-corpus query still fires; mixed case (one tool call has subject, another doesn't) |
| `synthesis::find_referenced_indices` (replaced by `cite_id` resolver) | resolves stable IDs; unknown ID logged and dropped, not panicked |
| `text_normalize` | comma-below ↔ cedilla bidirectional; query/payload symmetric |
| `tool_contract_test` | `outline_corpus`, `fetch_section_chunks`, `search_sections` schemas pinned |

### 10.2 End-to-end validation (manual, by user)

Per user direction: no eval harness. The user runs `cargo run --release --example ask -- "<query>"` for representative queries from §3 and judges quality from logs + answer + source PDFs. Claude Code reads the trace + evidence + source and reports findings. Suggested validation matrix:

1. The canonical failing query (cheatsheet TOATE metode segmentare) — expect coverage report 7/7, all four previously-missed methods present.
2. A definition query ("ce este metoda Otsu") — expect unchanged latency and quality.
3. A how-to query — expect unchanged.
4. An out-of-corpus query ("ce este React?") — expect honest "not in knowledge base", no hedging on a confident wrong answer.
5. A multi-doc comparative query if a second doc is ingested.

## 11. Rollout

Single PR. The changes are additive at the public-API level:

- New tools coexist with existing `search_knowledge` and `document_info`.
- `SearchKnowledgeArgs` schema gains optional fields (`section_id`, `doc_id`, `mode`); old callers unaffected.
- `ChunkSummary` payload gains optional fields; old summaries from before re-ingest are usable but flagged with `ingest_run_id: None`.
- `QueryContext` gains optional fields; non-exhaustive flows leave them `None` and behave as today.
- Grounding gate is in-place replacement; behavior change is restricted to instruction-heavy and multi-tool queries.

No feature flag. The change either lands fully or not at all; pragmatic given the user's "single spec" direction and the absence of a regression eval harness.

## 12. Migration

Existing ingested documents (e.g., `segmentare.pdf`) need re-ingest to get the new `doc_id / section_id / ingest_run_id` fields on summaries and the new `Section` ordering fields. The migration is a one-shot re-run of the existing ingest CLI; the spec does not require an online migration.

If the user wants to validate without re-ingesting, the design degrades gracefully:

- `outline_corpus` over un-migrated data falls back to Neo4j-only sections without summaries (heading-only outline).
- `CoverageVerifier` over un-migrated data still emits a report; missing fields just mean section-level coverage is at section heading granularity, not summary granularity.

## 13. Out of scope (explicit)

- Re-introducing `QueryAnalyzer` or any LLM-based query augmentation that competes with `QueryRewriter` (see §1 reverted-pattern note and Codex audit `/tmp/codex-revert.md`).
- Cross-corpus federation.
- New retrieval strategies (`Summaries` strategy is reused inside `outline_corpus` but not exposed as a top-level routing choice).
- Numerical / table-aware aggregation beyond exposing `has_table` metadata.
- Latency optimization beyond the "no worse than current for non-exhaustive, ≤2× for exhaustive" budget.
- Concurrency or quota changes to the existing reranker / embedder.

## 14. Open questions — resolved

Reviewer decisions captured here for traceability; the spec body above already reflects them.

- **Q1 → resolved (re-ingest as part of rollout).** Existing corpora will be re-ingested to populate the new `doc_id / section_id / ingest_run_id` fields on summaries. No separate backfill job.
- **Q2 → resolved (log + metric).** `CoveragePlanner` emits a `WARN` log and Prometheus counter when a soft heuristic flags the query as likely-exhaustive but the strict rules do not match. See `coverage_planner.rs` row in §6.1.
- **Q3 → resolved (raise to 5).** `max_turns` raised from 4 to 5 to give the agent one follow-up turn after the primary `search_sections` call.
- **Q4 → resolved (no user-facing footer).** Coverage information stays out of the answer body. It is exposed via the `SynthesizedResponse` struct (`is_grounded`, `confidence`, `coverage: CoverageReport`) and via OpenTelemetry metrics. The "I'm not confident…" hedging wrapper is dropped unconditionally when a `coverage_report` exists; if coverage is partial, the synthesizer still emits a clean answer and lets the caller (CLI / API consumer) decide how to surface partial coverage.

---

*End of design.*
