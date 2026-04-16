# Heavy Code Review — agent-rs

**Date:** 2026-04-16
**Reviewers:** 6 parallel specialized agents (Rust Pro, Bug Hunter, Error Hunter, Type Analyst, Standards, Security+Perf)
**Scope:** Entire workspace (7 crates + root binary + examples)
**Scores:** Type Design 6.2/10 | Security 6/10 | Performance 7/10 | Standards 74/100

---

## Consolidated Findings (deduplicated, severity-sorted)

### CRITICAL — Must fix before production

| # | Issue | Files | Workers | Description | Fix |
|---|---|---|---|---|---|
| C1 | **Evidence lost during grounding retries** | `agent/steps/grounding_gate.rs:47`, `agent/steps/agent_runner.rs:137-138` | W2 | `rerun()` clears `ctx.kb_evidence` and `ctx.web_evidence` before retry. All original evidence is permanently lost. Retry has LESS evidence than original — defeats the purpose. | Save evidence before rerun: `let prev_kb = ctx.kb_evidence.clone()`. After rerun, merge: `ctx.kb_evidence.extend(prev_kb)`. |
| C2 | **Cypher injection via raw query passthrough** | `stores/neo4j/mod.rs:182, 212-239` | W6 | `query_nodes()` and `query_raw()` accept arbitrary Cypher strings and execute directly. LLM prompt injection could craft malicious tool calls. | Remove or restrict to a whitelist of pre-approved query templates. All callers should use parameterized queries. |
| C3 | **Budget TOCTOU race condition** | `tools/budget.rs:62-65` | W1, W2 | `fetch_add` + conditional `fetch_sub` is not atomic. Two concurrent tool calls can both bypass the budget limit. | Use `compare_exchange` loop: `loop { let cur = load(); if cur >= max { return Err } match cas(cur, cur+1) { Ok => return Ok, Err => continue } }` |
| C4 | **Lossy error variant mapping** | `retrieval/error.rs:93-117` | W3, W4, W5 | Six distinct error variants (`Ingestion`, `PdfExtraction`, `NotImplemented`, `DoclingIngestFailed`, `IngestInProgress`, `PostgresCommitFailed`) ALL mapped to `core::RetrievalError::Ranking(...)`. Upstream code misclassifies every one. | Add matching variants to `core::RetrievalError`. |
| C5 | **Result persistence failures silently swallowed** | `agent/steps/result_persister.rs:80-82, 109-111, 115-121` | W3 | Three log-and-continue patterns: `persist_run`, `store_evidence_batch`, and `store_message` failures are logged but ignored. Conversation history silently corrupted. | Propagate errors or flag on `SynthesizedResponse`. |
| C6 | **No gRPC authentication/authorization** | `server/lib.rs:67-94` | W6 | No TLS, no auth interceptors, no OPA integration, no rate limiting. Any network client can submit queries. | Add tonic interceptors for API key validation and OPA policy checks. |
| C7 | **`.expect()` in production Neo4j code** | `stores/neo4j/structural.rs:773, 826` | W1, W3, W5 | `serde_json::to_value(r).expect(...)` on `NodeRow` and `EdgeRow`. If any field contains `f64::NAN`, the server panics and crashes. | Replace with `.map_err(|e| StoreError::Serialization(...))?` |

### HIGH — Should fix soon

| # | Issue | Files | Workers | Description | Fix |
|---|---|---|---|---|---|
| H1 | **Graph depth scoring is wrong** | `retrieval/search/graph.rs:85-88` | W2 | `enumerate()` index used as proxy for traversal depth, but `get_neighbors` returns unordered flat list. Scores are meaningless for all 5 graph strategies. | Either modify `GraphStore::get_neighbors` to return `(GraphNode, usize)` with depth, or assign uniform scores. |
| H2 | **Internal error details leaked to gRPC clients** | `server/error.rs:28-57` | W6 | `err.to_string()` passed directly as gRPC status message. Exposes internal paths, connection strings, query details. | Return generic messages to clients. Log details server-side with correlation ID. |
| H3 | **`fuse_evidence` returns unbounded results** | `retrieval/ranker.rs:240-241` | W2 | Limit is `max(kb.len(), web.len())` instead of a caller-specified top-K. Downstream gets unnecessarily large evidence packs. | Accept `limit` as a parameter, use `ctx.limit`. |
| H4 | **`std::sync::Mutex` in async context** | `tools/knowledge.rs:138`, `tools/tool_factory.rs:6,20,75`, `agent/steps/agent_runner.rs:8,201` | W1 | Evidence collector uses `std::sync::Mutex` in async code. Latent deadlock risk if any `.await` is added inside lock scope. | Replace with `tokio::sync::Mutex` across the evidence collector path. |
| H5 | **DNS rebinding TOCTOU in SSRF protection** | `tools/fetch.rs:175-187` | W6 | DNS resolved for SSRF check, then HTTP client resolves again independently. Window for DNS rebinding attack. | Pin resolved IP using `reqwest::ClientBuilder::resolve()`. |
| H6 | **Per-request HTTP client construction** | `tools/web.rs:138-141, 281-284`, `tools/fetch.rs:180-187` | W6 | Each tool call creates a new `reqwest::Client` — new connection pool, TLS cache, DNS resolver every time. | Create one `reqwest::Client` at `ToolFactory` construction, share via `Arc`. |
| H7 | **Sequential Neo4j node/edge insertion** | `stores/neo4j/mod.rs:72-78, 94-99` | W6 | Each node/edge upserted one at a time. O(n) round-trips. 1000 nodes ≈ 2 seconds. | Use `UNWIND` batch pattern (already proven in `structural.rs`). |
| H8 | **Double DB query for history** | `agent/steps/context_builder.rs:61, 90` | W2 | `load_history` called twice — once with hardcoded limit 10, again with `self.max_messages`. 2x Postgres hits per request, first ignores config. | Call once with `self.max_messages`, reuse result. |
| H9 | **Qdrant count error silently swallowed** | `stores/qdrant/mod.rs:323-326` | W3 | Double `unwrap_or(0)` on `delete_by_filter` pre-count. If Qdrant unreachable, logs "deleted 0" — a silent lie. | Propagate the count error. |
| H10 | **`SparseVector` length invariant unenforced** | `stores/types.rs` | W4 | `indices.len() == values.len()` documented but not enforced. Mismatch causes silent data corruption in Qdrant. | Private fields, `new()` that validates length match. |
| H11 | **Unbounded memory on fetch without Content-Length** | `tools/fetch.rs:219-235` | W6 | Without Content-Length header, full response buffered via `response.bytes().await` before size check. OOM possible. | Stream with `response.chunk()` in a loop, abort at `max_bytes`. |
| H12 | **Unbounded graph traversal depth** | `stores/neo4j/queries.rs:62-69` | W6 | `depth` interpolated into Cypher `[*1..{depth}]` with no upper bound. Large depth = DoS via exhaustive traversal. | Enforce `MAX_GRAPH_DEPTH = 5`, clamp input. |

### MEDIUM — Fix during next refactoring pass

| # | Issue | Files | Workers | Description | Fix |
|---|---|---|---|---|---|
| M1 | **6 newtypes have `pub` inner fields** | `core/types.rs` | W1, W4 | `SessionId(pub Uuid)`, `UserId(pub String)`, `Score(pub f64)`, `Embedding(pub Vec<f32>)`, `DatasetName(pub String)`, `CollectionName(pub String)` all bypass validation. | Private fields, `new()`/`TryFrom`, accessor methods. |
| M2 | **QueryContext has no typestate** | `agent/context.rs:18-62` | W1, W4 | 20 `pub` fields, any step can mutate anything. Spec calls for typestate but current impl is a flat mutable bag. | At minimum: immutable request fields behind getters, mutable fields `pub(crate)`. |
| M3 | **Verifier silently degrades to heuristic** | `agent/verifier.rs:122-135, 176-184` | W3 | LLM verifier failure falls back to heuristic token overlap with only WARN log. Massive quality regression invisible to user. | Log at ERROR. Tag `VerificationResult` with `method: "heuristic_fallback"`. |
| M4 | **13+ files exceed 200-line limit** | Multiple | W1, W5 | `pipeline.rs` (2009), `router.rs` (1010), `graph_cot.rs` (991), `graph_extractor.rs` (982), `structural.rs` (856), `qdrant/mod.rs` (597), `config.rs` (475), `embedder.rs` (454), etc. | Progressive splitting into submodules. |
| M5 | **Regex compiled on every invocation** | `agent/synthesis.rs:183`, `retrieval/ingestion/pipeline.rs:1864,1866` | W1, W5, W6 | `Regex::new().expect()` on hot paths. | Use `std::sync::LazyLock` for once-compiled regexes. |
| M6 | **Global Mutex on ONNX embedding models** | `retrieval/embedder.rs:27,283,344,374,415` | W6 | `Arc<Mutex<TextEmbedding>>` serializes all embedding calls. Two queries can't embed in parallel. | Use `tokio::sync::Mutex` with `spawn_blocking`, or pool of model instances per core. |
| M7 | **Tool `Option` wrapper anti-pattern** | `tools/*.rs` (all tool structs) | W1, W4 | Required deps wrapped in `Option<Arc<T>>` for serde, every `call()` unwraps. Runtime check for compile-time guarantee. | Document with `// INVARIANT: always Some after new()` or restructure serde approach. |
| M8 | **Sequential sub-query processing** | `retrieval/router.rs:454-478` | W6 | Multiple decomposed sub-queries embedded and dispatched sequentially in a loop. 3 sub-queries = 3x latency. | Use `futures::future::join_all()` or `tokio::join!` for parallel dispatch. |
| M9 | **Long-term memory failure silently ignored** | `agent/steps/context_builder.rs:76-85` | W3 | `get_long_term_memories` failure logged at WARN, pipeline continues without user's stored context. | Inject note into context informing agent that LTM was unavailable. |
| M10 | **Triplet hydration errors logged at DEBUG** | `retrieval/router.rs:312-319` | W3 | If Qdrant is down, every hydration silently fails. Evidence pack silently degraded. | Log at WARN, include count of failed hydrations in evidence metadata. |
| M11 | **`core` couples to storage crate errors** | `core/Cargo.toml` | W5 | `core` imports `qdrant-client`, `neo4rs`, `diesel` for `#[from]` conversions in errors.rs. | Use `String`-wrapped error variants in core; define `From` impls in `stores`. |
| M12 | **Duplicate `SparseVector` type** | `retrieval/embedder.rs:46` vs `stores/types.rs:21` | W5 | Two independent definitions. | Consolidate into one location. |
| M13 | **`truncate_to_char_boundary` iterates twice** | `core/types.rs:296-306` | W1 | `chars().count()` (O(n)) then `char_indices().nth()` (another O(n)). Hot path for long strings. | Single-pass: `match s.char_indices().nth(max) { Some((idx, _)) => &s[..idx], None => s }` |
| M14 | **Inconsistent `Score` usage** | `stores/types.rs`, `agent/synthesis.rs` | W4 | `EvidenceItem.score` and `Citation.score` use raw `f64` instead of `Score` newtype. | Use `Score` everywhere scores appear. |
| M15 | **`#[allow(...)]` without justification** | Multiple | W5 | 4 of 6 `#[allow]` annotations lack proper justification comments. | Add `// SAFETY:` or justification comments. |

### LOW — Nice to have

| # | Issue | Files | Workers |
|---|---|---|---|
| L1 | Wrong error variant for KB errors (uses `ToolError::WebSearch`) | `tools/knowledge.rs:118,134` | W1 |
| L2 | `Vec::new()` without `with_capacity` where size is known (~60 instances) | Multiple | W1 |
| L3 | Intermediate Vec allocations in verifier | `agent/verifier.rs:205-217, 257-289` | W1 |
| L4 | Default credentials hardcoded (dev defaults) | `core/config.rs:370-380` | W6 |
| L5 | No input length validation on user queries | `agent/steps/agent_runner.rs:59-60` | W6 |
| L6 | Neo4j `query_nodes` silently skips rows without "n" column | `stores/neo4j/mod.rs:181-197` | W3 |
| L7 | `UncertaintyType` is dead code | `agent/verifier.rs` | W4 |
| L8 | 3 TODO comments in production code | `agent/verifier.rs:247`, `retrieval/router.rs:424,459` | W5 |
| L9 | Rayon parallel threshold too low (100) | `retrieval/ranker.rs:173-176` | W6 |
| L10 | Double HashMap lookup per document in ranker | `retrieval/ranker.rs:157-169` | W6 |
| L11 | `BraveConfig` cloned twice in tool_factory | `agent/steps/tool_factory.rs:84,88` | W1 |
| L12 | `eprintln!` in TracingGuard::Drop (justified) | `observability/tracing.rs:56,59` | W5 |
| L13 | Regex for match with single error arm | `agent/steps/agent_runner.rs:150-158` | W1 |

---

## Refactoring Action Plan (Suggested Order)

### Sprint 1 — Critical Correctness (1-2 days)

| Task | Issue | Impact |
|---|---|---|
| Fix evidence preservation during retries | C1 | Grounding retry actually works |
| Fix budget TOCTOU with compare_exchange | C3 | Budget enforcement is correct under concurrency |
| Replace `.expect()` with `?` in neo4j structural | C7 | Server won't crash on NaN in graph data |
| Fix lossy error variant mapping | C4 | Error diagnosis works correctly |
| Fix `fuse_evidence` limit | H3 | Evidence packs are bounded |
| Fix graph depth scoring | H1 | Graph-based strategies produce correct rankings |
| Fix double history query | H8 | Correct config respected, halves DB load |

### Sprint 2 — Security Hardening (1-2 days)

| Task | Issue | Impact |
|---|---|---|
| Restrict raw Cypher methods | C2 | Eliminates injection surface |
| Add gRPC auth interceptors | C6 | System is not open to any network client |
| Fix DNS rebinding TOCTOU | H5 | SSRF protection actually works |
| Sanitize gRPC error messages | H2 | No internal details leaked |
| Enforce max graph traversal depth | H12 | Prevents DoS via deep graph traversal |
| Stream fetch responses for size limit | H11 | Prevents OOM on malicious servers |

### Sprint 3 — Error Handling Integrity (1 day)

| Task | Issue | Impact |
|---|---|---|
| Fix result_persister log-and-continue | C5 | Persistence failures are visible |
| Fix Qdrant count swallow | H9 | Delete operations report accurately |
| Tag verifier fallback in results | M3 | Users know when quality is degraded |
| Fix triplet hydration logging level | M10 | Degraded evidence is visible |
| Fix LTM failure notification | M9 | Agent knows when memory unavailable |
| Fix SparseVector invariant | H10 | No silent data corruption |

### Sprint 4 — Type Safety Hardening (1-2 days)

| Task | Issue | Impact |
|---|---|---|
| Make newtype inner fields private | M1 | Newtypes actually enforce invariants |
| QueryContext encapsulation (pub(crate) + getters) | M2 | Pipeline steps can't corrupt state |
| Consistent Score usage | M14 | Type safety throughout evidence chain |
| Fix tool Option anti-pattern (document invariant) | M7 | Code intent is clear |
| Consolidate SparseVector | M12 | No type duplication |

### Sprint 5 — Performance (1 day)

| Task | Issue | Impact |
|---|---|---|
| Share reqwest::Client across tools | H6 | Connection reuse, -5-20ms per tool call |
| Batch Neo4j inserts with UNWIND | H7 | O(n) → O(n/100) round-trips |
| Replace std::sync::Mutex with tokio::sync::Mutex | H4 | Eliminates latent deadlock risk |
| LazyLock for compiled regexes | M5 | No repeated compilation |
| Fix truncate_to_char_boundary double iteration | M13 | Halves cost on long strings |
| Parallelize sub-query dispatch | M8 | 3x latency → 1x for decomposed queries |

### Sprint 6 — Module Splitting (2-3 days, can be incremental)

| Task | Issue | Impact |
|---|---|---|
| Split `pipeline.rs` (2009 lines) | M4 | Manageable modules |
| Split `router.rs` (1010 lines) | M4 | Per-strategy modules |
| Split `graph_cot.rs` (991 lines) | M4 | Separated concerns |
| Split `graph_extractor.rs` (982 lines) | M4 | Prompt templates separate |
| Split remaining oversized files | M4 | All files under 200 lines |

---

## Cross-Cutting Themes

1. **Silent degradation is the biggest systemic risk.** The codebase has a pattern of logging errors and continuing with degraded results. While this keeps the system running, users receive lower-quality answers with no indication. Every fallback/degradation should be flagged in the response metadata.

2. **Type safety is inconsistent.** `Confidence`, `NodeLabel`, `RetrievalStrategy` are gold-standard types. But `SessionId`, `UserId`, `Score`, `Embedding` have `pub` fields that defeat the newtype pattern. The pattern exists — it just needs to be applied uniformly.

3. **The error hierarchy foundation is strong but the conversions are lossy.** `thiserror` enums are well-designed. The problem is in the `From` conversion between crate-local and core error types, where information is destroyed.

4. **Performance architecture is solid but has localized bottlenecks.** `tokio::join!` for parallel retrieval, `rayon` for CPU-bound work, `spawn_blocking` for ONNX — all correct. The issues are specific: per-request HTTP clients, sequential Neo4j inserts, sequential sub-queries.

5. **Security has strong foundations but critical gaps.** SSRF IP validation, NodeLabel injection prevention, secret masking in Debug — all good. But raw Cypher passthrough, no gRPC auth, and leaked error details are production blockers.
