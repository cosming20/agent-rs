# Type Design Analysis Report

**Project:** agentic-rag
**Date:** 2026-03-26
**Analyzer:** pr-review-toolkit:type-design-analyzer
**Scope:** Focused analysis of recently modified files with type design concerns

---

## Type 1: Tool Config Dicts (`tools.py` / `agent.py`)

### Invariants Identified
- A tool config must have exactly four keys: `name` (str), `description` (str), `function` (callable), `parameters` (JSON Schema dict)
- `name` must match a known tool identifier
- `parameters` must be valid JSON Schema
- `function` must be a synchronous callable (Haystack constraint)

### Ratings
- **Encapsulation**: 2/10
  Tool configs are raw `dict[str, Any]`. Any code can construct malformed dicts. The invariant (four required keys with specific types) exists only in documentation and is validated implicitly when `Tool()` is constructed in `build_agent`.

- **Invariant Expression**: 2/10
  Nothing in the type system communicates what a valid tool config looks like. The `dict[str, Any]` signature erases all structure. A reader must inspect `build_agent` lines 48-55 to learn the required shape.

- **Invariant Usefulness**: 8/10
  The invariant (four specific keys) is valuable -- it prevents runtime KeyError crashes in `build_agent` and ensures Haystack can instantiate `Tool` objects correctly.

- **Invariant Enforcement**: 2/10
  Zero compile-time or construction-time validation. Errors only surface at `Tool()` instantiation, deep in `build_agent`. A missing key produces a cryptic `KeyError`, not a descriptive validation error.

### Strengths
- Factory pattern (`make_*`) centralizes tool creation, reducing ad-hoc dict construction
- Each factory produces a complete, correct config

### Concerns
- **Critical: `dict[str, Any]` erases all type safety.** If a factory forgets a key or misspells one, the error surfaces at runtime in a different module.
- `make_ingest_tool` accepts `cognee_store: Any` to "avoid circular import" -- this is a design smell. The circular dependency should be resolved structurally, not by dropping type safety.
- `_run_async` accepts and returns `Any` -- loses type information for all async bridge calls.

### Recommended Improvements

1. **Introduce a `ToolConfig` dataclass:**

```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class ToolConfig:
    name: str
    description: str
    function: Callable[..., dict[str, Any]]
    parameters: dict[str, Any]  # JSON Schema
```

Change all `make_*` return types to `-> ToolConfig` and `build_agent` to accept `list[ToolConfig]`. Cost: ~20 lines changed, zero runtime overhead, full mypy coverage.

2. **Type `_run_async` with a TypeVar:**

```python
from typing import TypeVar, Coroutine
T = TypeVar("T")
def _run_async(coro: Coroutine[Any, Any, T]) -> T: ...
```

3. **Fix the circular import** by using `TYPE_CHECKING` guard + string annotation for `CogneeStore`, not `Any`.

---

## Type 2: `_extract_evidence` return type (`orchestrator.py`)

### Invariants Identified
- Returns a list of Haystack `Document` objects
- Each document represents evidence extracted from agent tool calls
- Documents contain `content` (str) and `meta` with at least `source` key

### Ratings
- **Encapsulation**: 5/10
  The method is on the class (good) and static (appropriate). But the return type annotation `list[Any]` undermines all downstream type checking.

- **Invariant Expression**: 3/10
  The annotation says `list[Any]` but the implementation creates `list[Document]`. The import of `Document` inside the method body (lazy import) further obscures the actual type. Every caller must cast or assume.

- **Invariant Usefulness**: 9/10
  Knowing that evidence is `list[Document]` is critical -- it flows into `verifier.verify()`, `synthesizer.synthesize()`, and `ranker.rank()`, all of which expect `Document` instances.

- **Invariant Enforcement**: 3/10
  The implementation correctly builds `Document` objects, but the type system cannot verify callers handle them correctly. The `Any` return type silences all mypy checks downstream.

### Strengths
- The extraction logic itself is reasonable -- iterating messages in reverse for the answer, forward for evidence

### Concerns
- **The `list[Any]` return type is a lie.** It actually returns `list[Document]`. This breaks the type chain: `orchestrator.process_query` passes evidence to `verifier.verify(evidence=...)` which expects `tuple[Document, ...] | list[Document]`. With `list[Any]`, mypy cannot verify this contract.
- The lazy import of `Document` inside the method suggests a potential circular dependency concern, but `haystack.Document` is a third-party type with no circular risk.

### Recommended Improvements

1. **Fix the return annotation** to `list[Document]` and move the import to module level (it is already imported via `haystack.dataclasses.ChatMessage` path):

```python
@staticmethod
def _extract_evidence(agent_result: dict[str, Any]) -> list[Document]:
```

2. **Consider typing `agent_result`** more precisely. Haystack's `Agent.run()` returns `dict[str, list[ChatMessage]]` with a `"messages"` key. A `TypedDict` would help:

```python
class AgentResult(TypedDict):
    messages: list[ChatMessage]
```

---

## Type 3: `strategy` parameter -- enum vs raw string regression (`query.py` vs `router.py`)

### Invariants Identified
- Valid strategies are exactly: `semantic`, `keyword`, `hybrid`, `graph`, `graph_hybrid`
- `RetrievalStrategy` enum encodes this constraint in `router.py`
- The query route (`query.py`) and orchestrator both pass `strategy` as `str`

### Ratings
- **Encapsulation**: 4/10
  `RetrievalStrategy` is well-defined in `router.py`, but its use is not enforced at the API boundary. `QueryRequest.strategy` is a raw `str` with no enum validation.

- **Invariant Expression**: 5/10
  The enum exists and is clear, but the type is "leaky" -- it is used in `router.retrieve()` but abandoned everywhere else. The orchestrator's `process_query(strategy: str)`, `QueryRequest.strategy: str`, `SynthesizedResponse.strategy: str`, and `EvidencePack.strategy: str` all use raw strings.

- **Invariant Usefulness**: 9/10
  Invalid strategies cause silent failures or routing errors. The constraint is business-critical.

- **Invariant Enforcement**: 4/10
  Only `router.retrieve()` and `search_knowledge()` (in tools.py) validate via `RetrievalStrategy(strategy)`. The API layer, orchestrator, synthesizer, and evidence pack all accept arbitrary strings. A typo like `"hybrd"` passes through multiple layers before failing.

### Strengths
- `RetrievalStrategy(enum.StrEnum)` is excellent -- subclasses `str` so it works as a string wherever needed
- The enum's JSON Schema is properly used in tool parameters

### Concerns
- **Type safety regression at API boundary.** `QueryRequest.strategy: str` should validate against the enum. Pydantic supports enum fields natively.
- The string passes through 4 layers (API -> orchestrator -> agent tool -> router) before being validated. Fail-early principle violated.
- `SynthesizedResponse.strategy: str` and `EvidencePack.strategy: str` could be `RetrievalStrategy` for downstream safety.

### Recommended Improvements

1. **Use the enum in `QueryRequest`:**

```python
class QueryRequest(BaseModel):
    strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID, ...)
```

Pydantic will validate and coerce string inputs to the enum automatically.

2. **Propagate the enum type** through `AgentOrchestrator.process_query(strategy: RetrievalStrategy)`, `SynthesizedResponse.strategy`, and `EvidencePack.strategy`.

3. At minimum, if Haystack forces string usage internally, validate at the API boundary and convert early.

---

## Type 4: `RAGPipeline` dataclass (`factory.py`)

### Invariants Identified
- All components must be initialized and wired correctly
- `cognee_store` may be `None` (Cognee disabled), but if the orchestrator's tools reference it, it must be non-None
- The pipeline must be closeable (graceful shutdown)

### Ratings
- **Encapsulation**: 4/10
  All fields are public (bare `@dataclass`). Any code can replace `orchestrator`, `postgres`, etc. after construction. The pipeline is a bag of components with no protection.

- **Invariant Expression**: 5/10
  The optional `cognee_store: CogneeStore | None = None` correctly expresses that Cognee is optional. However, the relationship between `cognee_store` being None and the tools configuration is not expressed -- a `# type: ignore[arg-type]` on line 119 of `factory.py` papers over this.

- **Invariant Usefulness**: 7/10
  The pipeline container is useful for lifecycle management (startup/shutdown) and dependency injection. The invariants are practical.

- **Invariant Enforcement**: 3/10
  No validation in `__init__` or `__post_init__`. The `build_pipeline` factory is the only safe constructor, but nothing prevents direct `RAGPipeline(...)` with incompatible components. The `# type: ignore[arg-type]` on `RetrievalRouter(cognee_store=cognee_store)` when `cognee_store` could be `None` is a real bug risk.

### Strengths
- Factory function (`build_pipeline`) centralizes wiring logic
- `close()` method handles graceful shutdown with individual failure tolerance
- Clean separation of concerns in the assembled components

### Concerns
- **The `# type: ignore[arg-type]` on line 119 hides a real invariant violation.** `RetrievalRouter.__init__` requires `CogneeStore`, but `build_pipeline` may pass `None`. This will crash at runtime if Cognee is disabled but a query arrives.
- `close()` only closes Postgres. If CogneeStore or other components need cleanup, they are leaked.
- The dataclass is mutable -- fields can be reassigned after construction.

### Recommended Improvements

1. **Guard the `cognee_store=None` case** either by making `RetrievalRouter` accept `CogneeStore | None` (and raise a clear error on search), or by making the pipeline construction fail-fast when Cognee is disabled but retrieval is configured.

2. **Make the dataclass frozen:** `@dataclass(frozen=True)` -- the pipeline should not be mutated after assembly.

3. **Add `__post_init__` validation** to check invariants (e.g., if `cognee_store` is None, verify that no tools reference it).

---

## Type 5: `VerificationResult` (`verifier.py`)

### Invariants Identified
- `confidence` must be in range [0.0, 1.0]
- `is_grounded` should be consistent with `confidence` relative to the threshold
- `ungrounded_claims` should be empty when `is_grounded` is True
- Result is immutable after creation

### Ratings
- **Encapsulation**: 9/10
  `frozen=True` dataclass. Immutable. No setters. Clean.

- **Invariant Expression**: 7/10
  The fields clearly communicate the verification result. The `tuple[str, ...]` for `ungrounded_claims` (not `list`) reinforces immutability. However, the `confidence` range [0.0, 1.0] is not enforced in the type.

- **Invariant Usefulness**: 9/10
  These invariants directly prevent bugs: confidence used in metrics, retry logic, and user-facing responses. An out-of-range value would corrupt all three.

- **Invariant Enforcement**: 6/10
  Immutability is enforced (frozen). But `confidence` range is not validated -- `VerificationResult(is_grounded=True, confidence=5.0, reasoning="...")` would be accepted. The LLM verification path could return out-of-range values from a malformed JSON response.

### Strengths
- Excellent use of frozen dataclass for immutability
- `tuple[str, ...]` instead of `list` for the claims -- correct immutability
- Clean, focused type with clear purpose
- `needs_retry()` method encapsulates retry logic on the verifier, not on the result

### Concerns
- No `confidence` range validation. The LLM path parses `float(parsed.get("confidence", 0.0))` which could return any float value.
- Potential inconsistency: `is_grounded=True` with `ungrounded_claims=("claim1",)` is representable but semantically invalid.

### Recommended Improvements

1. **Add `__post_init__` validation:**

```python
def __post_init__(self) -> None:
    if not 0.0 <= self.confidence <= 1.0:
        object.__setattr__(self, 'confidence', max(0.0, min(1.0, self.confidence)))
    if self.is_grounded and self.ungrounded_claims:
        raise ValueError("Grounded result cannot have ungrounded claims")
```

2. Alternatively, clamp confidence at the creation site in `_verify_with_llm`.

---

## Type 6: `EvidencePack` (`ranker.py`)

### Invariants Identified
- `documents` is an immutable tuple of ranked Documents
- Documents are ordered by descending RRF score
- Documents are deduplicated (no duplicate content)
- `source_counts` records pre-dedup counts

### Ratings
- **Encapsulation**: 8/10
  Frozen dataclass with tuple for documents. The `source_counts` uses `dict` (mutable default via factory), which is a minor encapsulation leak -- callers could mutate it.

- **Invariant Expression**: 7/10
  `tuple[Document, ...]` clearly signals immutability and ordering. The relationship between `source_counts` and `documents` (pre-dedup vs post-dedup) could be confusing without reading the docstring.

- **Invariant Usefulness**: 8/10
  Dedup and ordering invariants prevent duplicate evidence in responses and ensure highest-quality evidence appears first.

- **Invariant Enforcement**: 6/10
  The `ResultRanker.rank()` method correctly enforces these invariants during construction. But direct `EvidencePack(documents=(...), ...)` construction bypasses all ranking/dedup logic. The `source_counts: dict` default_factory creates a new dict per instance (correct), but the dict itself remains mutable after construction.

### Strengths
- Frozen dataclass with tuple -- excellent immutability pattern
- `source_counts` provides useful observability metadata
- Clear docstring explaining the type's role

### Concerns
- `source_counts: dict[str, int]` on a frozen dataclass is technically still mutable (the dict contents can be changed). Use `types.MappingProxyType` or document the intended immutability.
- No way to enforce the "ordered by RRF score" invariant at the type level (acceptable -- that is the ranker's responsibility).

### Recommended Improvements

Minor: consider `frozenset` or `MappingProxyType` for `source_counts` if true deep immutability is desired. This is a low-priority improvement.

---

## Type 7: `SynthesizedResponse` and `Citation` (`synthesis.py`)

### Invariants Identified
- `Citation.index` is 1-based and sequential
- `Citation.score` should be non-negative
- `SynthesizedResponse.citations` uses tuple (immutable)
- `confidence` should be in [0.0, 1.0]

### Ratings
- **Encapsulation**: 9/10
  Both are frozen dataclasses with tuple collections. Immutable and clean.

- **Invariant Expression**: 8/10
  Clear field names and types. The frozen tuple pattern is consistently applied.

- **Invariant Usefulness**: 8/10
  These types are the final output to the API layer. Correct structure directly prevents serialization bugs.

- **Invariant Enforcement**: 6/10
  No range validation on `confidence` or `score`. No check that citation indices are sequential. The `ResponseSynthesizer._build_citations` method enforces sequential indexing, but direct construction does not.

### Strengths
- Excellent frozen dataclass pattern
- Clean separation between Citation and SynthesizedResponse
- Direct correspondence with the API response model (`QueryResponse`)

### Concerns
- Same `confidence` range issue as `VerificationResult`
- `strategy: str = ""` -- empty string default is a weak sentinel. Consider making it required or using `RetrievalStrategy`.

### Recommended Improvements
- Make `strategy` a `RetrievalStrategy` type (ties into finding #3 above)
- Validate `confidence` range at construction

---

## Type 8: `RetrievalStrategy` enum (`router.py`)

### Invariants Identified
- Exactly five valid strategies
- Values are lowercase string identifiers
- Subclasses `str` for seamless string interop

### Ratings
- **Encapsulation**: 10/10 -- Enum is inherently encapsulated.
- **Invariant Expression**: 10/10 -- The finite set of valid values is perfectly expressed.
- **Invariant Usefulness**: 9/10 -- Prevents invalid strategy strings throughout the pipeline.
- **Invariant Enforcement**: 10/10 -- Enum construction raises `ValueError` for invalid values.

### Strengths
- `StrEnum` is the ideal choice -- works as both enum and string
- Clean, self-documenting values
- Module docstring explains what each strategy maps to in Cognee

### Concerns
- The enum is excellent. The problem is it is not used consistently (see Type 3 analysis).

---

## Cross-Cutting Findings Summary

### Priority 1 -- High Impact, Low Cost

| Finding | File | Fix |
|---------|------|-----|
| `_extract_evidence` returns `list[Any]` but should be `list[Document]` | `orchestrator.py:195` | Change annotation, move import to module level |
| `_extract_answer` input typed as `dict[str, Any]` | `orchestrator.py:186` | Add `AgentResult` TypedDict |
| `strategy: str` at API boundary -- use `RetrievalStrategy` enum | `query.py:25` | Change to `RetrievalStrategy` field |
| `tools_config: list[dict[str, Any]]` -- introduce `ToolConfig` | `agent.py:41`, `tools.py` | Add frozen dataclass, ~20-line change |

### Priority 2 -- Medium Impact, Low Cost

| Finding | File | Fix |
|---------|------|-----|
| `_run_async` accepts/returns `Any` | `tools.py:32` | Add TypeVar for return type |
| `cognee_store: Any` in `make_ingest_tool` | `tools.py:220` | Use TYPE_CHECKING guard |
| `memory_store: Any` in `make_remember_tool` | `tools.py:290` | Use TYPE_CHECKING guard |
| `confidence` range not validated | `verifier.py:39`, `synthesis.py:31` | Add `__post_init__` clamping |
| `# type: ignore[arg-type]` hides None bug | `factory.py:119` | Guard or make router accept Optional |

### Priority 3 -- Low Impact, Design Improvement

| Finding | File | Fix |
|---------|------|-----|
| `RAGPipeline` dataclass is mutable | `factory.py:32` | Add `frozen=True` |
| `strategy: str` in `SynthesizedResponse` / `EvidencePack` | Multiple | Propagate `RetrievalStrategy` enum |
| `source_counts: dict` mutable on frozen dataclass | `ranker.py:30` | Use MappingProxyType or document |
| `strategy: str = ""` sentinel | `synthesis.py:48` | Use `str \| None = None` |

---

## Overall Assessment

**Aggregate Score: 6.2 / 10**

The codebase demonstrates strong fundamentals: frozen dataclasses for value objects, a clean error hierarchy, good use of `StrEnum`, and consistent factory patterns. The main weakness is **incomplete type propagation** -- good types exist (`RetrievalStrategy`, `Document`, `VerificationResult`) but are abandoned at module boundaries in favor of `str`, `Any`, and `dict[str, Any]`. The four Priority 1 fixes are straightforward, require no architectural changes, and would raise the aggregate score to approximately 8/10.

### Summary by File

| File | Primary Issue | Severity |
|------|--------------|----------|
| `tools.py` | `dict[str, Any]` tool configs, `Any` params | High |
| `agent.py` | `list[dict[str, Any]]` input, no `ToolConfig` type | High |
| `orchestrator.py` | `list[Any]` return, `dict[str, Any]` input | High |
| `query.py` | `strategy: str` instead of `RetrievalStrategy` | High |
| `router.py` | Enum is excellent, no issues | None |
| `factory.py` | Mutable dataclass, `type: ignore` hiding None bug | Medium |
| `verifier.py` | `confidence` range unvalidated | Medium |
| `synthesis.py` | `confidence` range unvalidated, `str = ""` sentinel | Medium |
| `ranker.py` | Mutable dict on frozen dataclass | Low |
