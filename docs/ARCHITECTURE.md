# Agentic RAG -- Complete Architecture Document

> **Generated:** 2026-03-27
> **Version:** 0.1.0
> **Status:** Active development -- Phase 1-7 implemented

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Request Flow](#3-request-flow)
4. [Project Structure](#4-project-structure)
5. [Core Foundation](#5-core-foundation)
6. [Data Layer (Stores)](#6-data-layer-stores)
7. [Retrieval Engine](#7-retrieval-engine)
8. [Agent Core](#8-agent-core)
9. [Gateway & API](#9-gateway--api)
10. [Security & Authorization](#10-security--authorization)
11. [Observability](#11-observability)
12. [Infrastructure](#12-infrastructure)
13. [Dependencies](#13-dependencies)
14. [Data Models](#14-data-models)
15. [Error Hierarchy](#15-error-hierarchy)
16. [Configuration Reference](#16-configuration-reference)
17. [Scripts & Tooling](#17-scripts--tooling)
18. [Design Decisions](#18-design-decisions)

---

## 1. System Overview

Agentic RAG is a **near-perfect agentic Retrieval-Augmented Generation system** built on **Haystack + Hayhooks**. It implements a multi-strategy retrieval engine with a hardcoded quality gate (grounding verification) and automatic retry, backed by a unified knowledge graph engine (Cognee) that manages both vector search (Qdrant) and graph traversal (Neo4j).

**Key properties:**
- Pure Python 3.12+ backend, no frontend
- Agent uses ReACT reasoning loop (Haystack Agent)
- 5 retrieval strategies through a single Cognee backend
- Hardcoded grounding verification -- the agent cannot skip this step
- Per-user dataset isolation (RBAC) via OPA + scoped tool injection
- Full audit trail (every query run is persisted to Postgres)
- OpenTelemetry distributed tracing + custom metrics

---

## 2. High-Level Architecture

```
                                  +-------------------+
                                  |   Client (HTTP)   |
                                  +--------+----------+
                                           |
                                  POST /api/v1/query
                                  POST /api/v1/ingest
                                           |
                              +------------v-----------+
                              |    FastAPI Gateway      |
                              |  (uvicorn @ port 1059) |
                              +--+--+--+--+--+--+-----+
                                 |  |  |  |  |  |
                    +------------+  |  |  |  |  +------------+
                    |               |  |  |  |               |
              +-----v-----+  +-----v--v--v--v-----+  +------v------+
              |    OPA     |  |  Agent Orchestrator |  | OTel/Jaeger |
              | (AuthZ)    |  |  (Haystack Agent)   |  | (Tracing)   |
              | port 1055  |  +--+--+--+--+--+-----+  | port 1058   |
              +------------+     |  |  |  |  |         +-------------+
                                 |  |  |  |  |
            +--------------------+  |  |  |  +-------------------+
            |                       |  |  |                      |
      +-----v------+         +-----v--v--v-----+         +------v-------+
      |  MemoryStore |         |   CogneeStore   |         |  RunStore    |
      |  (Postgres)  |         | (Qdrant + Neo4j)|         | (Postgres)   |
      +--------------+         +--+-----+--+----+         +--------------+
                                  |     |  |
                        +---------+     |  +----------+
                        |               |             |
                  +-----v----+   +------v----+  +-----v--------+
                  |  Qdrant   |   |   Neo4j   |  |  PostgreSQL  |
                  | (vectors) |   |  (graph)  |  | (state/audit)|
                  | port 1051 |   | port 1061 |  |  port 1054   |
                  +-----------+   +-----------+  +--------------+
```

---

## 3. Request Flow

### Query Pipeline (10 steps)

```
1. POST /api/v1/query arrives
   Headers: X-User-Id, X-User-Role, X-Session-Id

2. OPA middleware checks authorization
   -> admin: full access
   -> user: /api/v1/query + /api/v1/ingest only
   -> denied: 403

3. Orchestrator.process_query() begins
   -> Records query metric
   -> Stores user message in Postgres (ChatMessage)

4. Load conversation history (last 10 messages)
   -> Build ChatMessage list for Haystack Agent
   -> Inject strategy hint into user message

5. Build per-request tools scoped to user's dataset
   -> search_knowledge (router + ranker, dataset=user_{user_id})
   -> search_web (placeholder)
   -> remember (memory store, user-scoped)
   -> ingest (cognee_store, dataset-scoped)

6. Run Haystack Agent (ReACT tool-calling loop, offloaded to thread)
   -> Agent calls tools iteratively
   -> Evidence accumulates via closure
   -> Max steps: configurable (default 10)

7. Extract answer from agent result

8. Verify grounding (LLM-based faithfulness check)
   -> is_grounded: true/false
   -> confidence: 0.0-1.0
   -> ungrounded_claims: [...]

9. If not grounded AND retries remaining:
   -> Inject feedback message ("search again with broader strategy")
   -> Re-run agent with retry messages
   -> Re-verify grounding
   -> Record stage2_retry metric

10. Synthesize response with citations
    -> Build Citation objects from evidence
    -> Persist PipelineRun to Postgres
    -> Store assistant message in chat history
    -> Return SynthesizedResponse
```

### Ingestion Pipeline

```
1. POST /api/v1/ingest with file(s) or text

2. EmbeddingPipeline parses file
   -> Docling: structure-aware parsing (PDF, DOCX, PPTX, HTML, etc.)
   -> HybridChunker: token-aware splitting (max 512 tokens)
   -> build_breadcrumb_chunks(): heading-prefixed text chunks

3. CogneeStore.add_chunks() ingests to Cognee
   -> Each chunk added as separate document
   -> cognee.cognify() builds knowledge graph (entities + relations)
   -> Vectors stored in Qdrant
   -> Graph stored in Neo4j

4. DocumentStore persists DoclingDocument JSON to Postgres
```

---

## 4. Project Structure

```
agentic-rag/
  src/
    agentic_ra/
      __init__.py
      core/
        __init__.py
        config.py              # Pydantic Settings (43+ env vars)
        errors.py              # 14-class exception hierarchy
        logging.py             # JSON + text structured logging
      stores/
        __init__.py
        postgres/
          __init__.py
          client.py            # Async SQLAlchemy engine + sessions
          models.py            # ChatMessage, MemoryEntry ORM models
          documents.py         # DocumentStore (parsed doc persistence)
          runs.py              # RunStore (pipeline run audit trail)
        qdrant/
          __init__.py
          client.py            # Qdrant client wrapper (legacy, now via Cognee)
        cognee/
          __init__.py
          client.py            # CogneeStore: graph + vector + lexical search
      retrieval/
        __init__.py
        embedder.py            # Docling + HybridChunker + breadcrumb chunks
        router.py              # 5-strategy retrieval dispatcher
        ranker.py              # RRF dedup + evidence pack assembly
      agent/
        __init__.py
        agent.py               # Haystack Agent builder
        tools.py               # Tool factories (search, web, ingest, remember)
        memory.py              # Short-term + long-term memory over Postgres
        orchestrator.py        # 10-step query pipeline + quality gate
        verifier.py            # LLM-based faithfulness/grounding check
        synthesis.py           # Citation builder + response assembly
      tools/
        __init__.py
        mcp_client.py          # MCP client (official Python SDK)
      auth/
        __init__.py
        opa.py                 # OPA async client (httpx)
      gateway/
        __init__.py
        app.py                 # FastAPI app factory + lifespan
        routes/
          health.py            # GET /health
          query.py             # POST /api/v1/query
          ingest.py            # POST /api/v1/ingest
        middleware/
          __init__.py
          auth.py              # OPA authorization middleware
      observability/
        __init__.py
        tracing.py             # OTel TracerProvider + OTLP export
        metrics.py             # RAGMetrics (7 custom OTel metrics)
      pipelines/
        __init__.py
        factory.py             # build_pipeline() assembles everything
  infra/
    docker-compose.yml         # 6 services on agentic-ra-net
    .env                       # 24 environment variables
    qdrant/
      config.yaml              # API key + log level
    postgres/
      init.sql                 # Extensions (uuid-ossp, pgcrypto)
    opa/
      policies/
        authz.rego             # RBAC: admin/user roles
    otel/
      collector.yaml           # OTLP receivers -> batch -> Jaeger export
  scripts/                     # 11 operational scripts
  tests/
    e2e/                       # End-to-end tests (Playwright)
  docs/
    superpowers/
      specs/                   # Design specs
      plans/                   # Implementation plans
  pyproject.toml               # uv + hatchling
```

---

## 5. Core Foundation

### 5.1 Configuration (`core/config.py`)

Centralized configuration via **pydantic-settings**. All values from environment variables + `.env` file. Zero hardcoded values.

| Group | Settings | Defaults |
|-------|----------|----------|
| **Qdrant** | `qdrant_host`, `qdrant_grpc_port` (1050), `qdrant_http_port` (1051), `qdrant_api_key` (secret), `qdrant_collection` ("documents") | localhost |
| **PostgreSQL** | `postgres_user` ("agentic_ra"), `postgres_password` (secret), `postgres_db` ("agentic_ra"), `postgres_port` (1054), `postgres_host` | localhost |
| **OPA** | `opa_port` (1055), `opa_host`, `opa_log_level` ("info"), `opa_auth_enabled` (False) | localhost |
| **Hayhooks** | `hayhooks_port` (1056) | reserved |
| **OTel/Jaeger** | `otel_collector_port` (1057), `otel_host`, `jaeger_ui_port` (1058), `jaeger_host` | localhost |
| **OpenAI** | `openai_api_key` (secret), `llm_model` ("gpt-4o-mini") | - |
| **Embedding** | `embedding_model` ("text-embedding-3-small"), `embedding_dimension` (1536), `embedding_batch_size` (32), `chunk_max_tokens` (512), `chunk_merge_peers` (True) | - |
| **Cognee** | `cognee_dataset` ("agentic_rag"), `cognee_graph_db_provider` ("neo4j"), `cognee_vector_db_provider` ("qdrant"), `cognee_data_root`, `cognee_system_root`, `cognee_enabled` (True) | - |
| **Neo4j** | `neo4j_host`, `neo4j_bolt_port` (1061), `neo4j_http_port` (1062), `neo4j_user` ("neo4j"), `neo4j_password` (secret), `neo4j_db` ("neo4j") | localhost |
| **Application** | `log_level` ("INFO") | - |
| **Agent** | `agent_model` ("gpt-4o"), `agent_max_steps` (10), `agent_max_retries` (1) | - |

**Computed fields** (auto-derived):
- `qdrant_url` -> `http://{host}:{http_port}`
- `postgres_dsn` -> `postgresql://...`
- `postgres_dsn_async` -> `postgresql+asyncpg://...`
- `opa_url` -> `http://{host}:{port}`
- `otel_endpoint` -> `http://{host}:{port}`
- `jaeger_ui_url` -> `http://{host}:{port}`
- `neo4j_bolt_url` -> `bolt://{host}:{port}`

**Singleton access:** `get_settings()` via `@lru_cache(maxsize=1)`.

### 5.2 Structured Logging (`core/logging.py`)

Dual-format logging:
- **Development:** human-readable `%(asctime)s %(levelname)-5s [%(name)s] %(message)s`
- **Production:** JSON with structured fields (`timestamp`, `level`, `logger`, `message`, `exception`, + domain extras: `query`, `confidence`, `source`, `operation`)

Suppressed noisy loggers: `httpx`, `httpcore`, `opentelemetry`, `urllib3`.

### 5.3 Error Hierarchy (`core/errors.py`)

```
AgenticRAError (base)
  +-- message: str
  +-- context: dict[str, object]  # structured metadata for observability
  +-- __str__: "message [key='value', ...]"
  |
  +-- ConfigError
  |
  +-- StoreError
  |     +-- PostgresError
  |     +-- CogneeError
  |
  +-- RetrievalError
  |     +-- EmbeddingError
  |     +-- RoutingError
  |     +-- RankingError
  |
  +-- AgentError
  |     +-- MemoryStoreError
  |     +-- OrchestratorError
  |     +-- VerificationError
  |     +-- SynthesisError
  |
  +-- ToolError
  |     +-- MCPClientError
  |
  +-- AuthError
  |     +-- OPAError
  |
  +-- GatewayError
```

Every exception carries structured context (`**kwargs`) for logging/tracing. Example:
```python
raise CogneeError("Graph query failed", dataset="agentic_rag", operation="search")
# str: "Graph query failed [dataset='agentic_rag', operation='search']"
```

---

## 6. Data Layer (Stores)

### 6.1 PostgreSQL (`stores/postgres/`)

**Client** (`client.py`): Async SQLAlchemy engine with `asyncpg` driver. Provides `session()` context manager for transactions. Schema creation via `Base.metadata.create_all()`.

**Models** (`models.py`):

#### ChatMessage
| Column | Type | Notes |
|--------|------|-------|
| `id` | `UUID` | PK, auto-generated |
| `session_id` | `UUID` | Indexed, groups conversations |
| `role` | `MessageRole` (StrEnum) | `user` / `assistant` / `system` |
| `content` | `Text` | Message body |
| `metadata_` | `JSONB` | Arbitrary metadata (mapped as `metadata`) |
| `created_at` | `DateTime(tz)` | `server_default=func.now()` |

#### MemoryEntry
| Column | Type | Notes |
|--------|------|-------|
| `id` | `UUID` | PK, auto-generated |
| `user_id` | `String(255)` | Indexed, user scope |
| `key` | `String(255)` | Memory lookup key |
| `value` | `Text` | Memory content |
| `metadata_` | `JSONB` | Arbitrary metadata |
| `created_at` | `DateTime(tz)` | `server_default=func.now()` |
| `updated_at` | `DateTime(tz)` | `server_default=func.now()`, `onupdate=func.now()` |

#### ParsedDocument
| Column | Type | Notes |
|--------|------|-------|
| `id` | `UUID` | PK, auto-generated |
| `source_path` | `String(1024)` | File path or URL |
| `filename` | `String(512)` | Original filename |
| `content_hash` | `String(64)` | SHA-256 for dedup |
| `chunk_count` | `Integer` | Number of chunks produced |
| `docling_json` | `JSONB` | Full DoclingDocument JSON |
| `metadata_` | `JSONB` | Extra metadata |
| `created_at` | `DateTime(tz)` | `server_default=func.now()` |

#### PipelineRun
| Column | Type | Notes |
|--------|------|-------|
| `id` | `UUID` | PK, auto-generated |
| `session_id` | `UUID` | Links to conversation |
| `user_id` | `String(255)` | Indexed |
| `query` | `Text` | Original user query |
| `rewritten_query` | `Text` | After query rewriting |
| `strategy` | `String(50)` | Retrieval strategy used |
| `answer` | `Text` | Final synthesized answer |
| `confidence` | `Float` | Grounding confidence 0.0-1.0 |
| `is_grounded` | `Boolean` | Passed verification? |
| `citation_count` | `Integer` | Number of citations |
| `retrieval_ms` | `Float` | Retrieval latency |
| `evidence_count` | `Integer` | Number of evidence docs |
| `retries` | `Integer` | Stage-2 retry count |
| `created_at` | `DateTime(tz)` | `server_default=func.now()` |

**Constant:** `ANONYMOUS_USER_ID = "anonymous"` for unauthenticated operations.

**Additional stores:**
- `DocumentStore` (`documents.py`): Saves/retrieves `ParsedDocument` records.
- `RunStore` (`runs.py`): Saves `PipelineRun` audit records.

### 6.2 Cognee Knowledge Graph (`stores/cognee/client.py`)

`CogneeStore` wraps the Cognee SDK, managing both **vector search** (via Qdrant) and **knowledge graph** (via Neo4j).

**Configuration:** Programmatic setup at startup -- LLM API key, model, graph/vector providers, storage directories, Qdrant/Neo4j connection details.

**Qdrant Compatibility Shim:** Custom `_CompatQDrantAdapter` wraps the community Qdrant adapter to absorb extra kwargs (`database_name`, `include_payload`, `node_name`) that Cognee 0.5.5 passes but the adapter v0.1.0 doesn't accept.

**Methods:**

| Method | Description |
|--------|-------------|
| `configure(...)` | Set up LLM, graph DB, vector DB, storage roots |
| `add_and_cognify(text, dataset)` | Ingest text + build knowledge graph in one step |
| `add_chunks(chunks, dataset)` | Ingest individual chunks, then cognify once |
| `add(text, dataset)` | Ingest without building graph (for batching) |
| `cognify(dataset)` | Build/update knowledge graph for a dataset |
| `prune()` | Reset entire knowledge base (irreversible) |
| `search(query, search_type, top_k, dataset)` | Graph-augmented search returning Haystack Documents |

**Search types** (Cognee `SearchType`):
- `GRAPH_COMPLETION` -- Graph traversal + LLM reasoning
- `CHUNKS` -- Vector similarity search (Qdrant)
- `CHUNKS_LEXICAL` -- Lexical/BM25 search
- `TRIPLET_COMPLETION` -- Triplet-based graph search
- `SUMMARIES` -- Summary-based search

**Content extraction:** `_extract_content()` normalizes all Cognee result shapes (dict, str, SearchResult with .payload/.text/.content) to plain text.

### 6.3 Qdrant (`stores/qdrant/client.py`)

Legacy direct Qdrant client wrapper. Now superseded by Cognee-managed Qdrant access. Retained for potential direct vector operations outside the Cognee pipeline.

---

## 7. Retrieval Engine

### 7.1 Embedder (`retrieval/embedder.py`)

**`EmbeddingPipeline`**: Structure-aware document parsing and chunking.

**Components:**
- **Docling `DocumentConverter`**: Parses PDF, DOCX, PPTX, HTML, Markdown, images into structured `DoclingDocument`
- **Docling `HybridChunker`**: Token-aware splitting with heading hierarchy preservation
  - `max_tokens`: 512 (configurable)
  - `merge_peers`: True (merges small adjacent chunks)
  - Tokenizer: `sentence-transformers/all-MiniLM-L6-v2`

**Methods:**

| Method | Description |
|--------|-------------|
| `chunk_text(documents)` | Chunk Haystack Documents into contextualized pieces |
| `chunk_file(source)` | Parse file/URL with Docling, then chunk |
| `build_breadcrumb_chunks(docling_doc)` | Build heading-prefixed text chunks |
| `warm_up()` | Pre-load models |

**Output:** `ChunkResult` dataclass with `documents` (list of Haystack Documents with breadcrumb metadata) and `docling_document` (full DoclingDocument for persistence).

**Breadcrumb chunks:** Each chunk is prefixed with its heading hierarchy path, e.g.:
```
Chapter 1 > Section 1.2 > Subsection 1.2.3
Actual chunk text here...
```

### 7.2 Router (`retrieval/router.py`)

**`RetrievalRouter`**: Dispatches searches to Cognee based on strategy.

**5 retrieval strategies:**

| Strategy | Cognee Search Types | Description |
|----------|-------------------|-------------|
| `SEMANTIC` | `CHUNKS` | Vector similarity search (Qdrant) |
| `KEYWORD` | `CHUNKS_LEXICAL` | Lexical/BM25 search |
| `HYBRID` | `CHUNKS` + `CHUNKS_LEXICAL` (parallel) | Vector + lexical fusion |
| `GRAPH` | `GRAPH_COMPLETION` | Graph traversal + LLM reasoning (Neo4j) |
| `GRAPH_HYBRID` | `GRAPH_COMPLETION` + `CHUNKS` + `CHUNKS_LEXICAL` (parallel) | Triple fusion: graph + vector + lexical |

**Parallel dispatch:** `HYBRID` and `GRAPH_HYBRID` strategies use `asyncio.gather()` to run multiple Cognee searches concurrently.

**Interface:**
```python
async def retrieve(
    query: str,
    strategy: RetrievalStrategy,
    top_k: int = 10,
    dataset: str | None = None,
) -> list[Document]
```

### 7.3 Ranker (`retrieval/ranker.py`)

**`ResultRanker`**: Deduplicates and ranks results using **Reciprocal Rank Fusion (RRF)**.

**Algorithm:**
1. Group documents by content hash (deduplication)
2. For each unique document, compute RRF score: `sum(1 / (k + rank))` across all source lists
3. Sort by RRF score descending
4. Return top-N as `EvidencePack`

**RRF constant `k`:** Default 60 (dampens high-rank contributions).

**Output:** `EvidencePack` dataclass:
```python
@dataclass(frozen=True)
class EvidencePack:
    documents: tuple[Document, ...]  # Ranked, deduplicated
    query: str                        # Original query
    strategy: str                     # Strategy used
    source_counts: dict[str, int]    # Pre-dedup counts per source
```

---

## 8. Agent Core

### 8.1 Agent Builder (`agent/agent.py`)

Builds a **Haystack `Agent`** with OpenAI chat generator and tool configurations.

```python
def build_agent(
    model: str,          # e.g. "gpt-4o"
    api_key: str,
    tools_config: list,  # Tool descriptor dicts
    max_steps: int = 10,
) -> Agent
```

Base agent is built with **no tools** at pipeline startup. Tools are injected per-request by the orchestrator for RBAC dataset scoping via `build_agent_with_tools()`.

### 8.2 Tools (`agent/tools.py`)

Four tool factories, each returns a tool descriptor dict:

#### `search_knowledge`
- Routes through `RetrievalRouter` -> `ResultRanker`
- Scoped to user's dataset (`user_{sanitized_user_id}`)
- Admin role: fan-out search across all datasets
- Evidence accumulates via closure reference (`evidence_accumulator`)
- Returns formatted search results as string

#### `search_web`
- Placeholder -- returns a "not yet configured" message
- Designed for future web search integration

#### `remember`
- Saves key-value facts to `MemoryStore` (long-term memory)
- Scoped to user's ID

#### `ingest_document`
- Ingests text into Cognee via `add_and_cognify()`
- Scoped to user's dataset

**User ID sanitization:** `_sanitize_user_id()` strips non-alphanumeric characters to prevent dataset name injection.

### 8.3 Memory (`agent/memory.py`)

**`MemoryStore`**: Dual-layer memory over PostgreSQL.

**Short-term (session-scoped):**
- `add_message(session_id, role, content, metadata)` -- Persist chat message
- `get_messages(session_id, limit=50)` -- Retrieve conversation history (oldest-first)

**Long-term (persistent facts):**
- `save_memory(user_id, key, value, metadata)` -- Upsert key-value fact (uses `SELECT ... FOR UPDATE`)
- `get_memory(user_id, key)` -- Retrieve single fact
- `get_memories(user_id, limit=100)` -- Retrieve all user facts (most recently updated first)
- `delete_memory(user_id, key)` -- Remove a fact

### 8.4 Orchestrator (`agent/orchestrator.py`)

**`AgentOrchestrator`**: The main query processing engine. Runs the 10-step pipeline described in [Section 3](#3-request-flow).

**Key design decisions:**
- **Hardcoded quality gate**: Grounding verification runs AFTER the agent, outside its control. The agent cannot skip or override it.
- **Per-request tool injection**: Tools are built fresh per request with the user's dataset scope, ensuring RBAC isolation.
- **Evidence accumulation via closure**: Tools append to a shared `evidence` list, so even if the agent fails, accumulated evidence is available.
- **Graceful agent failure**: If the agent throws (including serialization errors), the orchestrator falls back to accumulated evidence.
- **Async offloading**: Agent.run() (sync) is offloaded via `asyncio.to_thread()`.

### 8.5 Verifier (`agent/verifier.py`)

**`GroundingVerifier`**: LLM-based faithfulness/grounding check.

Uses an OpenAI chat model to evaluate whether the answer is grounded in the evidence. Returns structured JSON:

```python
@dataclass(frozen=True)
class VerificationResult:
    is_grounded: bool       # Every claim supported by evidence?
    confidence: float       # 0.0-1.0
    reasoning: str          # Explanation
    ungrounded_claims: tuple[str, ...]  # Claims not in evidence
```

**Retry logic:** `needs_retry(result)` returns `True` if `not is_grounded` and `confidence > 0.3` (low confidence = no retry, not worth trying again).

**Verification prompt enforces:**
- `is_grounded: true` ONLY if every factual claim is supported
- If answer says "no evidence found", is_grounded=false, confidence=1.0
- Lists all ungrounded claims

### 8.6 Synthesis (`agent/synthesis.py`)

**`ResponseSynthesizer`**: Builds final responses with citations and provenance.

**Output types:**

```python
@dataclass(frozen=True)
class Citation:
    index: int              # 1-based citation number
    content_snippet: str    # Excerpt from source
    source: str             # e.g. "cognee_chunks", "cognee_graph_completion"
    score: float            # Relevance score

@dataclass(frozen=True)
class SynthesizedResponse:
    answer: str
    citations: tuple[Citation, ...]
    query: str
    confidence: float
    is_grounded: bool
    strategy: str
```

---

## 9. Gateway & API

### 9.1 Application Factory (`gateway/app.py`)

**FastAPI application** with async lifespan management:

**Startup:**
1. `setup_logging(level)` -- Configure structured logging
2. `setup_tracing(settings)` -- Initialize OTel TracerProvider
3. `build_pipeline(settings)` -- Assemble entire RAG pipeline
4. `instrument_fastapi(app)` -- Add OTel auto-instrumentation
5. `pipeline.embedding_pipeline.warm_up()` -- Pre-load models

**Shutdown:**
1. Graceful close of all components (tolerates individual failures)

**Middleware:**
- `OPAAuthMiddleware` -- Authorization check before every request

**Routes mounted:**
- `health_router` -> `/health`
- `query_router` -> `/api/v1/query`
- `ingest_router` -> `/api/v1/ingest`

### 9.2 API Endpoints

#### `GET /health`
Returns service health status. Public (no auth required).

#### `POST /api/v1/query`
```json
{
  "query": "string",
  "strategy": "hybrid",  // semantic|keyword|hybrid|graph|graph_hybrid
  "limit": 10
}
```

**Headers:**
- `X-User-Id` (optional, default: "anonymous")
- `X-User-Role` (optional, default: "user")
- `X-Session-Id` (optional, auto-generated UUID if missing)

**Response:** `SynthesizedResponse` JSON with answer, citations, confidence, grounding status.

#### `POST /api/v1/ingest`
Accepts file uploads or text for ingestion into the knowledge graph.

---

## 10. Security & Authorization

### 10.1 OPA Integration (`auth/opa.py`)

**`OPAClient`**: Async HTTP client (httpx) for OPA policy evaluation.

```python
async def check(input_data: dict, policy_path: str | None) -> AuthzDecision
```

**`AuthzDecision`** dataclass:
```python
@dataclass(frozen=True)
class AuthzDecision:
    allowed: bool
    reason: str
    policy_path: str
```

### 10.2 RBAC Policy (`infra/opa/policies/authz.rego`)

| Role | Permissions |
|------|-------------|
| **admin** | Full access to all endpoints |
| **user** | `POST /api/v1/query` and `POST /api/v1/ingest` only |
| **public** | `/health`, `/docs/*`, `/openapi.json` |

Default: **deny** (all requests denied unless explicitly allowed).

### 10.3 Data Isolation

- **Per-user datasets**: Each user's data is stored in a Cognee dataset named `user_{sanitized_user_id}`
- **Tool scoping**: Tools are built per-request with the user's dataset, so queries only search the user's data
- **Admin fan-out**: Admin role can search across all datasets
- **User ID sanitization**: Non-alphanumeric characters stripped to prevent dataset name injection

### 10.4 Auth Middleware (`gateway/middleware/auth.py`)

`OPAAuthMiddleware` (Starlette BaseHTTPMiddleware):
- Skips public paths: `/health`, `/docs`, `/openapi.json`, `/redoc`
- Reads `X-User-Id` and `X-User-Role` from request headers
- Sends request context to OPA for authorization decision
- Returns 403 if denied, 503 if OPA unavailable
- Configurable `enabled` flag for development bypass

---

## 11. Observability

### 11.1 Tracing (`observability/tracing.py`)

**OpenTelemetry distributed tracing:**
- `TracerProvider` with `BatchSpanProcessor`
- Exports via OTLP gRPC to OTel Collector
- Service: `agentic-ra` v0.1.0
- FastAPI auto-instrumentation via `FastAPIInstrumentor`

**Pipeline:** App -> OTLP gRPC -> OTel Collector (port 1057) -> batch -> Jaeger (port 4317)

### 11.2 Metrics (`observability/metrics.py`)

**`RAGMetrics`**: 7 custom OpenTelemetry metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `rag.query.count` | Counter | Total queries processed (by strategy) |
| `rag.retrieval.latency` | Histogram | Retrieval latency in ms (by strategy) |
| `rag.retrieval.document_count` | Histogram | Docs retrieved per query (by strategy) |
| `rag.verification.confidence` | Histogram | Grounding confidence scores |
| `rag.verification.pass_count` | Counter | Queries passing grounding check |
| `rag.verification.fail_count` | Counter | Queries failing grounding check |
| `rag.retrieval.stage2_retry_count` | Counter | Stage-2 retry count |

**Export:** OTLP gRPC to OTel Collector, 30s export interval via `PeriodicExportingMetricReader`.

### 11.3 Trace UI

**Jaeger** at `http://localhost:1058` -- in-memory storage (ephemeral, dev only).

---

## 12. Infrastructure

### 12.1 Docker Compose Services

All services on `agentic-ra-net` bridge network.

| Service | Image | Ports | Resources | Restart |
|---------|-------|-------|-----------|---------|
| **Qdrant** | `qdrant/qdrant:v1.14.0` | 1050 (gRPC), 1051 (HTTP) | 2GB / 1GB / 1.5 CPU | on-failure:5 |
| **PostgreSQL** | `postgres:17.4-bookworm` | 1054 | 1GB / 512MB / 1.0 CPU | on-failure:5 |
| **Neo4j** | `neo4j:5.26-community` | 1061 (Bolt), 1062 (HTTP) | 2GB / 1GB / 1.0 CPU | on-failure:5 |
| **OPA** | `openpolicyagent/opa:1.4.2-debug` | 1055 | 256MB / 128MB / 0.5 CPU | unless-stopped |
| **Jaeger** | `jaegertracing/all-in-one:1.67` | 1058 (UI), 4317 (OTLP), 14269 (admin) | 1GB / 512MB / 0.5 CPU | unless-stopped |
| **OTel Collector** | `otel/opentelemetry-collector:0.121.0` | 1057 (OTLP gRPC), 4318 (OTLP HTTP) | 512MB / 256MB / 0.5 CPU | unless-stopped |

**Total resources:** ~6.8 GB memory limit, ~3.9 GB reservation, 5.0 CPUs.

### 12.2 Persistent Volumes

| Volume | Service | Mount |
|--------|---------|-------|
| `qdrant_storage` | Qdrant | `/qdrant/storage` |
| `postgres_data` | PostgreSQL | `/var/lib/postgresql/data` |
| `neo4j_data` | Neo4j | `/data` |

### 12.3 Health Checks

| Service | Check | Interval/Start/Retries |
|---------|-------|----------------------|
| Qdrant | TCP echo to port 6333 | 10s start, 5 retries |
| PostgreSQL | `pg_isready` + `SELECT 1` | 15s start, 10 retries |
| Neo4j | `wget` to port 7474 | 30s start, 10 retries |
| OPA | `wget` to `/v1/policies` | 10s start, 5 retries |
| Jaeger | `wget` to 14269 + netcat 4317 | 10s start, 5 retries |
| OTel | Depends on Jaeger healthy | - |

### 12.4 Port Map

| Port | Service | Protocol |
|------|---------|----------|
| 1050 | Qdrant | gRPC |
| 1051 | Qdrant | HTTP/REST |
| 1054 | PostgreSQL | PostgreSQL |
| 1055 | OPA | HTTP/REST |
| 1056 | Hayhooks | Reserved |
| 1057 | OTel Collector | OTLP gRPC |
| 1058 | Jaeger UI | HTTP |
| 1059 | API Gateway | HTTP (FastAPI) |
| 1060 | Reserved | - |
| 1061 | Neo4j | Bolt |
| 1062 | Neo4j | HTTP |

### 12.5 Network Topology

```
                     agentic-ra-net (bridge)
    +----------------------------------------------------+
    |                                                    |
    |  qdrant:6333/6334  <--  cognee (via config)        |
    |  postgres:5432     <--  sqlalchemy (asyncpg)       |
    |  neo4j:7687/7474   <--  cognee (via config)        |
    |  opa:8181          <--  gateway middleware (httpx)  |
    |  jaeger:4317       <--  otel-collector (grpc)      |
    |  otel:4317/4318    <--  app (otlp exporter)        |
    |                                                    |
    +----------------------------------------------------+
```

### 12.6 OTel Collector Pipeline

```yaml
Receivers:
  - otlp/grpc (0.0.0.0:4317)
  - otlp/http (0.0.0.0:4318)

Processors:
  - batch (timeout: 200ms, send_batch_size: 512)

Exporters:
  - otlp (jaeger:4317, tls: insecure)

Pipeline:
  traces: receivers [otlp] -> processors [batch] -> exporters [otlp]
```

### 12.7 PostgreSQL Init

```sql
-- infra/postgres/init.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";    -- Cryptographic functions
SELECT 1;                                      -- Health check verification
-- Table schemas managed by Alembic (Phase 2)
```

---

## 13. Dependencies

### Runtime Dependencies (`pyproject.toml`)

| Package | Version | Purpose |
|---------|---------|---------|
| `haystack-ai` | >=2.13 | Agent framework (ReACT loop, components) |
| `cognee[neo4j]` | >=0.3 | Knowledge graph engine (entity extraction, graph building) |
| `cognee-community-vector-adapter-qdrant` | >=0.1 | Qdrant adapter for Cognee |
| `docling` | >=2.31 | Structure-aware document parsing |
| `docling-core` | >=2.31 | Docling core types (DoclingDocument, DocItemLabel) |
| `fastapi` | >=0.115 | API gateway framework |
| `uvicorn` | >=0.34 | ASGI server |
| `sqlalchemy[asyncio]` | >=2.0 | ORM + async sessions |
| `asyncpg` | >=0.30 | Async PostgreSQL driver |
| `alembic` | >=1.15 | Database migrations |
| `qdrant-client` | >=1.13 | Qdrant vector DB client |
| `httpx` | >=0.28 | Async HTTP client (OPA, MCP) |
| `mcp` | >=1.12 | MCP Python SDK |
| `pydantic-settings` | >=2.9 | Configuration management |
| `opentelemetry-api` | >=1.29 | OTel tracing/metrics API |
| `opentelemetry-sdk` | >=1.29 | OTel tracing/metrics SDK |
| `opentelemetry-exporter-otlp` | >=1.29 | OTLP gRPC exporter |
| `opentelemetry-instrumentation-fastapi` | >=0.50b0 | FastAPI auto-instrumentation |

### Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mypy` | >=1.19.1 | Static type checking |
| `ruff` | >=0.15.7 | Linting + formatting |

### Build System

- **Backend:** `hatchling`
- **Package manager:** `uv`
- **Python:** >=3.12
- **Wheel packages:** `["src/agentic_ra"]`

### Dependency Overrides

```toml
[tool.uv]
override-dependencies = [
    "onnxruntime>=1.24",                    # No cp314 wheels for older versions
    "kuzu; python_version < '3.14'",        # Can't build from source on 3.14
]
```

---

## 14. Data Models

### Request/Response Types

#### QueryRequest
```python
query: str                    # User's question
strategy: str = "hybrid"     # Retrieval strategy
limit: int = 10              # Max results
```

#### SynthesizedResponse
```python
answer: str                   # Final answer text
citations: tuple[Citation]    # Source citations
query: str                    # Original query
confidence: float             # 0.0-1.0
is_grounded: bool             # Passed grounding check?
strategy: str                 # Strategy used
```

#### Citation
```python
index: int                    # 1-based number
content_snippet: str          # Source excerpt
source: str                   # Origin ("cognee_chunks", etc.)
score: float                  # Relevance score
```

#### VerificationResult
```python
is_grounded: bool
confidence: float
reasoning: str
ungrounded_claims: tuple[str, ...]
```

#### EvidencePack
```python
documents: tuple[Document, ...]
query: str
strategy: str
source_counts: dict[str, int]
```

#### ChunkResult
```python
documents: list[Document]     # Chunked Haystack Documents
docling_document: DoclingDocument  # Full parsed document
```

### Enums

#### MessageRole (StrEnum)
```python
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"
```

#### RetrievalStrategy (StrEnum)
```python
SEMANTIC = "semantic"
KEYWORD = "keyword"
HYBRID = "hybrid"
GRAPH = "graph"
GRAPH_HYBRID = "graph_hybrid"
```

---

## 15. Error Hierarchy

See [Section 5.3](#53-error-hierarchy-coreerrorspy) for full tree. Key patterns:

- **All errors** carry `message` + `context` dict
- **Store errors** include `dataset`, `operation`, `original_error`
- **Retrieval errors** include `model`, `operation`, `original_error`
- **Agent errors** include `session_id`, `query`, `operation`
- **Auth errors** include `policy_path`, `reason`

---

## 16. Configuration Reference

### Environment Variables (`.env`)

```bash
# Qdrant
QDRANT_GRPC_PORT=1050
QDRANT_HTTP_PORT=1051
QDRANT_HOST=localhost
QDRANT_API_KEY=changeme
QDRANT_COLLECTION=documents

# PostgreSQL
POSTGRES_USER=agentic_ra
POSTGRES_PASSWORD=changeme
POSTGRES_DB=agentic_ra
POSTGRES_PORT=1054
POSTGRES_HOST=localhost

# OPA
OPA_PORT=1055
OPA_HOST=localhost
OPA_LOG_LEVEL=info
OPA_AUTH_ENABLED=false

# Hayhooks
HAYHOOKS_PORT=1056

# OTel / Jaeger
OTEL_COLLECTOR_PORT=1057
OTEL_HOST=localhost
JAEGER_UI_PORT=1058
JAEGER_HOST=localhost

# OpenAI
OPENAI_API_KEY=sk-...

# Embedding / Chunking
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=32
CHUNK_MAX_TOKENS=512
CHUNK_MERGE_PEERS=true

# Cognee
COGNEE_DATASET=agentic_rag
COGNEE_GRAPH_DB_PROVIDER=neo4j
COGNEE_VECTOR_DB_PROVIDER=qdrant
COGNEE_ENABLED=true

# Neo4j
NEO4J_HOST=localhost
NEO4J_BOLT_PORT=1061
NEO4J_HTTP_PORT=1062
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
NEO4J_DB=neo4j

# Application
LOG_LEVEL=INFO

# Agent
AGENT_MODEL=gpt-4o
AGENT_MAX_STEPS=10
AGENT_MAX_RETRIES=1
```

---

## 17. Scripts & Tooling

### Operational Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `verify_infra.py` | Check Neo4j, Qdrant, Postgres ports before running |
| `verify_db.py` | Direct database connectivity checks |
| `stores_check.py` | Test store connections without modification |
| `stores_reset.py` | Reset Qdrant collection and knowledge graph |
| `data_chunks.py` | Preview chunks stored in Cognee/Qdrant |
| `data_list.py` | List all parsed documents from Postgres |
| `data_stats.py` | Show stats across Postgres, Qdrant, Neo4j |
| `ingest_sample.py` | Load test data into pipeline |
| `ingest_file.py` | Ingest multiple files (PDF, DOCX, PPTX, HTML) via CLI |
| `e2e_ingest_and_query.py` | Full end-to-end testing with data validation |
| `mcp_test.py` | Test MCP client-server round trips |

All scripts: async-first, configured via `get_settings()`.

---

## 18. Design Decisions

### Why Cognee as Sole Retrieval Backend

Cognee manages both vector search (Qdrant) and knowledge graph (Neo4j). Rather than maintaining separate clients for each, all retrieval routes through `CogneeStore`. This:
- Eliminates dual-client complexity
- Ensures entity/relation consistency between graph and vectors
- Enables unified dataset scoping for RBAC
- Allows transparent search type switching via `SearchType` enum

### Why Hardcoded Quality Gate

The grounding verifier runs **outside** the agent's ReACT loop, in the orchestrator. This ensures:
- The agent cannot skip verification
- Quality gate is deterministic (same code path every time)
- Retry logic is centralized, not scattered across tools
- Metrics capture real verification outcomes

### Why Per-Request Tool Construction

Tools are built fresh for each request with the user's dataset scope baked in. This:
- Prevents data leakage between users
- Makes RBAC enforcement automatic (tools can only access user's data)
- Allows admin role to fan-out across all datasets
- Avoids mutable shared state between requests

### Why Evidence Accumulation via Closure

Agent tools append to a shared `evidence` list via closure. This:
- Survives agent failures (SerializationError, etc.)
- Provides evidence for grounding verification even if agent crashes
- Avoids requiring the agent to return structured evidence
- Works with Haystack's tool interface without modification

### Why Docling for Parsing

Docling provides structure-aware parsing that preserves heading hierarchies, table structures, and document layout. Combined with HybridChunker, this produces:
- Token-aware chunks (respects model context limits)
- Breadcrumb-prefixed text (preserves section context)
- Multi-format support (PDF, DOCX, PPTX, HTML, images)

### Why Async Throughout

The entire pipeline is async (asyncpg, httpx, Cognee SDK). Sync operations (Haystack Agent.run(), GroundingVerifier.verify()) are offloaded via `asyncio.to_thread()`. This:
- Maximizes throughput for I/O-bound operations
- Prevents blocking the event loop
- Allows parallel retrieval dispatch (hybrid/graph_hybrid strategies)
