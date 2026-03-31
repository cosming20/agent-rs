# Agentic RAG — Complete Project Documentation

> **Version:** 0.1.0
> **Language:** Python 3.12+
> **Package Manager:** uv (Hatchling build backend)
> **Generated:** 2026-03-30

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Infrastructure & Docker Services](#4-infrastructure--docker-services)
5. [Core Modules](#5-core-modules)
6. [Data Stores](#6-data-stores)
7. [Retrieval Engine](#7-retrieval-engine)
8. [Agent Core](#8-agent-core)
9. [Tool Integration (MCP)](#9-tool-integration-mcp)
10. [Gateway (FastAPI)](#10-gateway-fastapi)
11. [Security & Authorization](#11-security--authorization)
12. [Observability](#12-observability)
13. [Pipeline Factory](#13-pipeline-factory)
14. [Database Migrations](#14-database-migrations)
15. [Scripts & E2E Testing](#15-scripts--e2e-testing)
16. [Dependencies](#16-dependencies)
17. [Git History & Milestones](#17-git-history--milestones)
18. [Project Structure](#18-project-structure)
19. [Configuration Reference](#19-configuration-reference)

---

## 1. Project Overview

**Agentic RAG** is a near-perfect agentic Retrieval-Augmented Generation system built on **Haystack** and served via **FastAPI**. It combines:

- **Knowledge graph** retrieval (Cognee + Neo4j)
- **Vector** similarity search (Qdrant, via Cognee)
- **Keyword/BM25** search (OpenSearch, legacy — Cognee lexical is primary)
- **LLM-based** query rewriting, answer generation, and faithfulness verification
- **MCP tool calling** (Brave Search, knowledge search, ingestion, memory)
- **Full observability** via OpenTelemetry (traces + metrics → Jaeger)
- **OPA-based authorization** middleware

The system ingests documents (PDF, DOCX, PPTX, HTML) through **Docling** (structure-aware parsing), builds a knowledge graph via **Cognee** (entities + relations → Neo4j, embeddings → Qdrant), and answers queries with grounded, cited responses.

---

## 2. Architecture

### High-Level Flow

```
Client
  → FastAPI Gateway (auth middleware → OPA)
    → AgentOrchestrator
      ├── MemoryStore (short-term session + long-term facts → Postgres)
      ├── LLMGenerator (query rewrite → RetrievalRouter → ResultRanker)
      │     ├── CogneeStore.search(CHUNKS)         — vector search via Qdrant
      │     ├── CogneeStore.search(CHUNKS_LEXICAL)  — keyword/BM25
      │     └── CogneeStore.search(GRAPH_COMPLETION) — graph traversal via Neo4j
      ├── LLMGenerator (answer generation with evidence)
      ├── GroundingVerifier (faithfulness check → retry if failed)
      ├── MCPToolClient (Brave Search, web fallback)
      └── ResponseSynthesizer (citations + provenance assembly)
    → Postgres (chat history, memory, runs, parsed documents, LLM costs)
    → OpenTelemetry → OTel Collector → Jaeger
```

### Pipeline Flow (10 Steps)

1. User query + context arrives at `/api/v1/query`
2. Agent loads short/long-term memory from Postgres
3. LLMGenerator rewrites query for better retrieval recall
4. **Stage 1 retrieval**: CogneeStore search (vector + keyword + graph)
5. ResultRanker: dedup via content hashing + Reciprocal Rank Fusion (RRF)
6. LLMGenerator produces answer grounded in evidence
7. GroundingVerifier runs faithfulness/grounding check (LLM-as-judge)
8. If fail or low confidence: **Stage 2 retrieval** (broadened query, fallback strategy)
9. If still insufficient: MCPToolClient invokes Brave Search for web evidence
10. ResponseSynthesizer builds final answer with citations + provenance

---

## 3. Technology Stack

| Layer                  | Technology                      | Version     | Purpose                                    |
| ---------------------- | ------------------------------- | ----------- | ------------------------------------------ |
| Agent framework        | Haystack (`haystack-ai`)        | >=2.13      | Agent, generators, embedders, tools        |
| API server             | FastAPI + Uvicorn               | >=0.115     | HTTP gateway, route handling               |
| Document parsing       | Docling (`docling`, `docling-core`) | >=2.31  | Structure-aware PDF/HTML/DOCX parsing      |
| Knowledge graph engine | Cognee (`cognee[neo4j]`)        | >=0.3       | Entity extraction, graph building, search  |
| Vector DB              | Qdrant (`qdrant-client`)        | >=1.13      | Embedding storage and similarity search    |
| Graph DB               | Neo4j                           | latest      | Entity/relation graph store (via Cognee)   |
| Keyword search         | OpenSearch (`opensearch-py`)    | >=2.8       | BM25 full-text search (legacy backend)     |
| State store            | PostgreSQL + SQLAlchemy         | >=2.0       | Chat history, memory, runs, documents      |
| Migrations             | Alembic                         | >=1.15      | Database schema management                 |
| Async Postgres driver  | asyncpg                         | >=0.30      | High-performance async Postgres            |
| MCP tools              | MCP Python SDK                  | >=1.12      | Tool discovery + invocation (Brave Search) |
| HTTP client            | httpx                           | >=0.28      | Async HTTP for external service calls      |
| Config management      | pydantic-settings               | >=2.9       | Typed env-based configuration              |
| Tracing                | OpenTelemetry (API+SDK+OTLP)    | >=1.29      | Distributed tracing                        |
| FastAPI instrumentation| OTel FastAPI instrumentation    | >=0.50b0    | Auto-instrument HTTP spans                 |
| Build system           | Hatchling                       | -           | PEP 517 build backend                      |
| Package manager        | uv                              | -           | Fast Python package management             |
| Dev tools              | ruff (>=0.15.7), mypy (>=1.19.1)| -           | Linting, formatting, type checking         |

---

## 4. Infrastructure & Docker Services

All services run via `infra/docker-compose.yml` on the `agentic-ra-net` bridge network.

### Service Map

| Service            | Image                                      | Host Port(s)       | Purpose                              | Resource Limits       |
| ------------------ | ------------------------------------------ | ------------------ | ------------------------------------ | --------------------- |
| **Qdrant**         | `qdrant/qdrant:latest`                     | `1050` (gRPC), `1051` (HTTP) | Vector similarity search    | 2GB RAM, 1.5 CPU      |
| **OpenSearch**     | `opensearchproject/opensearch:latest`      | `1052` (HTTP), `1053` (perf) | BM25 keyword search (legacy) | 2GB RAM, 1.5 CPU     |
| **PostgreSQL**     | `postgres:17`                              | `1054`             | State store (all tables)             | 1GB RAM, 1.0 CPU      |
| **OPA**            | `openpolicyagent/opa:latest`               | `1055`             | Policy-based authorization           | 256MB RAM, 0.25 CPU   |
| **Jaeger**         | `jaegertracing/all-in-one:1.67`            | `1058` (UI)        | Trace storage + visualization        | 1GB RAM, 0.5 CPU      |
| **OTel Collector** | `otel/opentelemetry-collector-contrib:0.121.0` | `1057` (gRPC)  | Telemetry pipeline                   | 512MB RAM, 0.5 CPU    |
| **Neo4j**          | `neo4j:5`                                  | `1061` (Bolt), `1062` (HTTP) | Knowledge graph store       | 2GB RAM, 1.0 CPU      |
| **Brave Search MCP** | `node:22-slim`                           | `1063`             | Web search MCP server                | 512MB RAM, 0.25 CPU   |

### Volumes

| Volume             | Mounted To                      | Purpose                    |
| ------------------ | ------------------------------- | -------------------------- |
| `qdrant_storage`   | `/qdrant/storage`               | Qdrant persistent data     |
| `opensearch_data`  | `/usr/share/opensearch/data`    | OpenSearch indices          |
| `postgres_data`    | `/var/lib/postgresql/data`      | PostgreSQL databases        |
| `neo4j_data`       | `/data`                         | Neo4j graph data            |

### Infrastructure Config Files

| File                           | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `infra/docker-compose.yml`     | All service definitions, ports, volumes, health checks   |
| `infra/.env` / `.env.example`  | Port mappings, credentials, API keys                     |
| `infra/qdrant/config.yaml`     | Qdrant production config override (mounted read-only)    |
| `infra/postgres/init.sql`      | Initial DB setup (uuid-ossp, pgcrypto extensions)        |
| `infra/opa/policies/authz.rego`| Authorization policy skeleton (GET /health allowed)      |
| `infra/otel/collector.yaml`    | OTel Collector config (receivers, exporters, pipelines)  |

### Health Checks

All services have health checks configured:
- **Qdrant**: TCP probe on port 6333
- **OpenSearch**: `curl` to `/_cluster/health`
- **PostgreSQL**: `pg_isready`
- **OPA**: HTTP GET to `/health`
- **Jaeger**: `wget --spider` + `nc` TCP probe
- **Neo4j**: `cypher-shell` connectivity check
- **Brave MCP**: `wget --spider` on `/sse`

---

## 5. Core Modules

### `core/config.py` — Centralized Configuration

Pydantic Settings-based configuration loaded from environment variables. All values configurable — zero hardcoding.

**Key settings groups:**

| Group              | Settings                                                                                           |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| **LLM**            | `llm_api_key`, `llm_model`, `embedding_model`, `embedding_dimension`, `llm_enrichment_model`       |
| **Agent**          | `agent_model`, `agent_max_steps`, `agent_max_retries`                                              |
| **Postgres**       | `postgres_dsn_async` (async DSN with asyncpg)                                                      |
| **Qdrant**         | `qdrant_url`, `qdrant_api_key`, `qdrant_collection`                                                |
| **OpenSearch**     | `opensearch_host`, `opensearch_port`, `opensearch_user`, `opensearch_password`, `opensearch_index`  |
| **Cognee**         | `cognee_enabled`, `cognee_dataset`, `cognee_graph_db_provider`, `cognee_vector_db_provider`, Neo4j connection details |
| **Chunking**       | `chunk_max_tokens`, `chunk_merge_peers`                                                            |
| **OPA**            | `opa_url`, `opa_auth_enabled`                                                                      |
| **MCP**            | `brave_mcp_url`                                                                                    |
| **Observability**  | `otel_endpoint`, `log_level`                                                                       |
| **Verification**   | `grounding_confidence_threshold`                                                                   |

### `core/errors.py` — Structured Error Hierarchy

Custom exception hierarchy rooted at `AgenticRAError`:

```
AgenticRAError (base)
├── ConfigError          — Configuration / environment issues
├── PostgresError        — Database operations
├── QdrantError          — Vector store operations
├── OpenSearchError      — Keyword search operations
├── CogneeError          — Knowledge graph operations
├── EmbeddingError       — Document parsing / chunking / embedding
├── RetrievalError       — Retrieval routing failures
├── RankingError         — Dedup / rank fusion failures
├── OrchestratorError    — Agent pipeline failures
├── VerificationError    — Grounding / faithfulness check failures
├── SynthesisError       — Response assembly failures
├── MemoryStoreError     — Memory read/write failures
├── MCPClientError       — MCP tool invocation failures
└── OPAError             — Authorization policy failures
```

Each exception carries structured context (operation, model, source, etc.) for debugging.

### `core/logging.py` — Structured Logging

JSON-structured logging via Python's `logging` module with:
- Configurable log levels via `Settings.log_level`
- Structured format with timestamps, module, level, message
- No `print()` statements — all logging through the standard library

---

## 6. Data Stores

### PostgreSQL (`stores/postgres/`)

**Client** (`client.py`): Async SQLAlchemy client using `asyncpg` driver. Manages session lifecycle with async context managers.

**Models** (`models.py`): 5 SQLAlchemy ORM models:

| Model             | Table               | Purpose                                               | Key Columns                                                       |
| ----------------- | ------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| `ChatMessage`     | `chat_messages`     | Session conversation history                          | `id`, `session_id`, `role` (enum), `content`, `metadata`, `created_at` |
| `MemoryEntry`     | `memory_entries`    | Long-term agent memory (key-value facts)              | `id`, `session_id`, `key`, `value`, `metadata`, `created_at`      |
| `ParsedDocument`  | `parsed_documents`  | DoclingDocument JSON persistence                      | `id`, `source_path`, `user_id`, `docling_json`, `chunk_count`, `metadata`, `created_at` |
| `PipelineRun`     | `pipeline_runs`     | Query execution tracking                              | `id`, `session_id`, `user_id`, `query`, `rewritten_query`, `strategy`, `answer`, `confidence`, `is_grounded`, `citation_count`, `retrieval_ms`, `evidence_count`, `retries`, `metadata`, `created_at` |
| `LLMCall`         | `llm_calls`         | LLM cost tracking (tokens, model, cost)               | `id`, `model`, `input_tokens`, `output_tokens`, `cost`, `call_type`, `metadata`, `created_at` |

**Specialized Stores:**
- `DocumentStore` (`documents.py`): Persists DoclingDocument JSON with dedup by `(user_id, source_path)`
- `RunStore` (`runs.py`): Records each pipeline run for observability and auditing

### Qdrant (`stores/qdrant/`)

**Client** (`client.py`): Qdrant client wrapper for vector operations. Managed by Cognee — direct Qdrant access is available but Cognee handles embedding storage for the primary retrieval path.

**Collections** (`collections.py`): Collection creation and management utilities.

### Cognee (`stores/cognee/`)

**Client** (`client.py`): Wrapper around the Cognee SDK providing three core operations:

| Operation   | Method                   | What It Does                                              |
| ----------- | ------------------------ | --------------------------------------------------------- |
| **add**     | `add(text, dataset)`     | Ingest text into a Cognee dataset                         |
| **cognify** | `cognify(datasets)`      | Extract entities/relations → Neo4j, embed chunks → Qdrant |
| **search**  | `search(query, type)`    | Graph-augmented retrieval across multiple search types     |

**Search Types (via `cognee.api.v1.search.SearchType`):**
- `CHUNKS` — Vector similarity search (embeddings in Qdrant)
- `CHUNKS_LEXICAL` — Keyword/BM25 search
- `GRAPH_COMPLETION` — Graph traversal for entity-relation completion

**Configure** method sets up all Cognee internals: LLM keys, Neo4j connection, Qdrant connection, data directories.

### OpenSearch (`stores/opensearch/`)

**Client** (`client.py`): Async OpenSearch wrapper for BM25 keyword search. Legacy backend — Cognee's `CHUNKS_LEXICAL` is the primary keyword search path. Retained for fallback and infrastructure completeness.

---

## 7. Retrieval Engine

### Embedder (`retrieval/embedder.py`)

**Document parsing and chunking pipeline using Docling + OpenAI embeddings.**

**Architecture:**
1. **Parse** source → `DoclingDocument` (preserves hierarchy, tables, provenance)
2. **Chunk** with `HybridChunker` (structure-aware + token-budget-aware)
3. **Contextualize** each chunk (inject heading/caption context from doc structure)
4. **Embed** contextualized text via OpenAI (`text-embedding-3-small` default)

**Key classes:**

| Class               | Purpose                                                          |
| ------------------- | ---------------------------------------------------------------- |
| `EmbeddingPipeline` | Main pipeline: chunk files/HTML → `ChunkResult` with Haystack Documents |
| `QueryEmbedder`     | Embeds queries at search time using `OpenAITextEmbedder`         |
| `ChunkResult`       | Frozen dataclass: `documents`, `docling_doc`, `chunk_count`      |
| `CharApproxTokenizer` | Char-based tokenizer (4 chars ≈ 1 token) — avoids BERT dependency for Py3.14 compat |

**Docling Configuration:**
- PDF pipeline with `TableStructureOptions` enabled
- `HybridChunker` with configurable `max_tokens` and `merge_peers`
- Automatic detection and preprocessing of `legislatie.just.ro` HTML (Romanian legal docs)

**HTML Preprocessor** (`retrieval/html_preprocessor.py`):
- Detects `legislatie.just.ro` format by `<span class="S_*">` CSS class patterns
- Converts span-based styling into semantic HTML (headings, articles, paragraphs)
- Enables Docling to correctly parse Romanian legislative documents

### Router (`retrieval/router.py`)

**Retrieval strategy router — dispatches queries to appropriate search backends.**

**Strategies** (enum `RetrievalStrategy`):
- `VECTOR` — Cognee CHUNKS (embedding similarity)
- `KEYWORD` — Cognee CHUNKS_LEXICAL (BM25)
- `HYBRID` — Both vector + keyword, results merged
- `GRAPH` — Cognee GRAPH_COMPLETION (entity-relation traversal)
- `GRAPH_HYBRID` — Graph + vector + keyword (all three)

The router accepts a strategy string hint and dispatches to Cognee's search API. Results are tagged with `meta["source"]` for downstream ranking.

### Ranker (`retrieval/ranker.py`)

**Dedup + Reciprocal Rank Fusion (RRF) + evidence pack assembly.**

**Algorithm:**
1. **Dedup**: Hash document content → keep first occurrence per content hash
2. **RRF scoring**: `score(d) = Σ 1/(k + rank_in_source)` across all sources (default k=60)
3. **Sort** by fused score descending
4. **Assemble** `EvidencePack`: ranked documents + query + strategy + source counts

**Output**: `EvidencePack` (frozen dataclass) with:
- `documents`: Tuple of ranked Haystack `Document` objects
- `query`: Original query
- `strategy`: Strategy used
- `source_counts`: Per-source result counts before dedup

---

## 8. Agent Core

### Memory (`agent/memory.py`)

**Dual-layer memory abstraction over PostgreSQL:**

| Layer          | Scope               | Storage             | Operations                        |
| -------------- | -------------------- | ------------------- | --------------------------------- |
| **Short-term** | Session (ephemeral)  | `chat_messages`     | `add_message`, `get_history`      |
| **Long-term**  | Persistent (global)  | `memory_entries`    | `remember`, `recall`, `recall_all`|

- Short-term: Conversation history per session, ordered by `created_at`
- Long-term: Key-value facts that persist across sessions (e.g., "user prefers Romanian")

### Generator (`agent/generator.py`)

**LLM-based answer generation and query rewriting using OpenAI via Haystack.**

Two modes:
1. **Query rewriting**: Takes user question + conversation history → optimized search query (expands abbreviations, resolves pronouns, adds keywords)
2. **Answer generation**: Takes user question + evidence documents → grounded answer with citation references `[index]`

Uses `OpenAIChatGenerator` from Haystack with configurable model, temperature, and max tokens.

### Orchestrator (`agent/orchestrator.py`)

**Central pipeline coordinator — wires all components together.**

```python
class AgentOrchestrator:
    """Flow: memory → query rewrite → retrieval → rank → LLM generate →
    verify → (Stage-2 retry if needed) → synthesize → persist."""
```

**Query processing flow:**
1. Load conversation history from MemoryStore
2. Rewrite query via LLMGenerator
3. Route to RetrievalRouter with selected strategy
4. Rank results via ResultRanker (RRF)
5. Generate answer via LLMGenerator with evidence
6. Verify groundedness via GroundingVerifier
7. **If verification fails** → Stage 2: broaden query, try different strategy, re-retrieve, re-generate, re-verify
8. **If still fails** → MCPToolClient invokes Brave Search for web evidence
9. Synthesize response with citations via ResponseSynthesizer
10. Persist conversation + run metrics to Postgres

**Configurable**: `max_retries` for Stage-2 retry attempts, confidence threshold from settings.

### Verifier (`agent/verifier.py`)

**LLM-as-judge faithfulness/grounding evaluator.**

Sends a structured prompt to the LLM asking it to evaluate:
- `is_grounded`: Whether every factual claim is supported by evidence
- `confidence`: 0.0–1.0 confidence in the assessment
- `reasoning`: Brief explanation of the judgment
- `ungrounded_claims`: List of claims not supported by evidence

Returns `VerificationResult` (frozen dataclass).

### Synthesis (`agent/synthesis.py`)

**Response builder with citations and provenance.**

Produces `SynthesizedResponse`:
- `answer`: Final answer text
- `citations`: Tuple of `Citation` objects (index, content_snippet, source, score)
- `query`: Original query
- `confidence`: Verification confidence
- `is_grounded`: Pass/fail
- `strategy`: Retrieval strategy used

### Agent Builder (`agent/agent.py`)

**Haystack Agent builder — configures a ReACT tool-calling agent.**

System prompt instructs the agent to:
1. Analyze the query
2. Search using appropriate strategy (semantic/keyword/graph/hybrid)
3. Cite evidence with `[1]`, `[2]` markers
4. Use `search_web` only if local knowledge is insufficient
5. Use `remember` to save important facts

### Agent Tools (`agent/tools.py`)

Four high-level tools exposed to the Haystack Agent:

| Tool               | Description                                    | Backend                         |
| ------------------ | ---------------------------------------------- | ------------------------------- |
| `search_knowledge` | Search the knowledge base                      | RetrievalRouter → Cognee        |
| `search_web`       | Search the web via Brave                       | MCPToolClient → Brave MCP       |
| `ingest`           | Ingest a document into the knowledge base      | EmbeddingPipeline → Cognee      |
| `remember`         | Save a fact to long-term memory                | MemoryStore                     |

---

## 9. Tool Integration (MCP)

### MCP Client (`tools/mcp_client.py`)

**Async MCP client using the official Python SDK.**

| Method        | Purpose                                              |
| ------------- | ---------------------------------------------------- |
| `list_tools`  | Discover available tools on an MCP server            |
| `call_tool`   | Invoke a tool with structured arguments              |

Returns `ToolResult` (frozen dataclass): `tool_name`, `content`, `is_error`, `raw`.

**Transport**: stdio-based subprocess per call (production TODO: shared subprocess lifecycle).

### Brave Search MCP Server

Runs as a Docker service (`node:22-slim`) on port **1063**, using `@brave/brave-search-mcp-server` via npx with HTTP transport. Requires `BRAVE_API_KEY` environment variable.

---

## 10. Gateway (FastAPI)

### Application Factory (`gateway/app.py`)

Creates the FastAPI app with:
- **Lifespan**: Builds full `RAGPipeline` on startup, tears down on shutdown
- **Middleware**: OPA authorization (configurable enable/disable)
- **Error handling**: Global exception handler for `AgenticRAError` hierarchy
- **OpenTelemetry**: Auto-instruments all HTTP spans via `FastAPIInstrumentor`
- **Embedding warmup**: Pre-loads embedding models on startup

### API Endpoints

| Method | Path            | Handler               | Purpose                                  |
| ------ | --------------- | --------------------- | ---------------------------------------- |
| `GET`  | `/health`       | `health_check`        | Service health status + version          |
| `POST` | `/api/v1/query` | `query`               | Main RAG query (returns answer + citations) |
| `POST` | `/api/v1/ingest`| `ingest_documents`    | Document ingestion (text → Cognee)       |

### Query Request/Response

**Request** (`QueryRequest`):
```json
{
  "query": "What does Article 5 say?",
  "session_id": "uuid (optional)",
  "user_id": "anonymous",
  "strategy": "hybrid",
  "limit": 10
}
```

**Response** (`QueryResponse`):
```json
{
  "answer": "Article 5 states that...",
  "citations": [
    {"index": 1, "content_snippet": "...", "source": "cognee_chunks", "score": 0.95}
  ],
  "query": "What does Article 5 say?",
  "confidence": 0.92,
  "is_grounded": true,
  "strategy": "hybrid",
  "session_id": "uuid"
}
```

### Ingest Request/Response

**Request** (`IngestRequest`):
```json
{
  "documents": [{"content": "...", "metadata": {"title": "..."}}],
  "source_path": "api_input"
}
```

**Response** (`IngestResponse`):
```json
{
  "ingested_count": 5,
  "chunk_count": 42,
  "parsed_document_id": "uuid",
  "cognee_ingested": true
}
```

---

## 11. Security & Authorization

### OPA Integration (`auth/opa.py`)

`OPAClient`: HTTP client that queries the OPA policy engine at `/v1/data/authz/allow`.

### Auth Middleware (`gateway/middleware/auth.py`)

`OPAAuthMiddleware` (Starlette `BaseHTTPMiddleware`):
- Intercepts all requests
- Sends `{method, path}` to OPA for policy evaluation
- Returns 403 if denied
- Configurable enable/disable via `Settings.opa_auth_enabled`
- Skip paths: `/health`, `/docs`, `/openapi.json`

### Current Policy (`infra/opa/policies/authz.rego`)

Skeleton policy — allows `GET /health` only. Production policies to be implemented in Phase 6.

```rego
package authz
default allow := false
allow if {
    input.method == "GET"
    input.path == ["health"]
}
```

---

## 12. Observability

### Tracing (`observability/tracing.py`)

- **Provider**: OpenTelemetry `TracerProvider` with `BatchSpanProcessor`
- **Exporter**: OTLP gRPC to OTel Collector (`settings.otel_endpoint`)
- **Resource**: `service.name=agentic-ra`, `service.version=0.1.0`
- **FastAPI**: Auto-instrumented via `FastAPIInstrumentor`

### Metrics (`observability/metrics.py`)

`RAGMetrics` class with OpenTelemetry meters:

| Metric                    | Type       | Purpose                              |
| ------------------------- | ---------- | ------------------------------------ |
| `retrieval_latency`       | Histogram  | Retrieval duration in ms             |
| `retrieval_hit_rate`      | Counter    | Successful retrievals                |
| `confidence_distribution` | Histogram  | Verification confidence scores       |
| `evidence_count`          | Histogram  | Documents per retrieval              |
| `retry_count`             | Counter    | Stage-2 retry attempts               |

**Export**: OTLP gRPC with 30-second periodic export interval.

### Cost Tracking (`observability/cost_tracker.py`)

- Hooks into **LiteLLM callbacks** (used by Cognee internally)
- Records every LLM call to the `llm_calls` Postgres table
- Tracks: model, input/output tokens, cost, call type
- Activated via `setup_cost_tracking(pg_dsn=...)` at pipeline startup

### Trace Flow

```
FastAPI request
  → OTel auto-span (HTTP)
    → Custom spans (retrieval, generation, verification)
      → BatchSpanProcessor
        → OTLP gRPC export
          → OTel Collector (port 1057)
            → Jaeger (port 1058 UI)
```

---

## 13. Pipeline Factory

### `pipelines/factory.py`

**Assembles all components into a working `RAGPipeline`.**

```python
@dataclass
class RAGPipeline:
    orchestrator: AgentOrchestrator
    postgres: PostgresClient
    qdrant: QdrantStore
    opensearch: OpenSearchStore
    embedding_pipeline: EmbeddingPipeline
    query_embedder: QueryEmbedder
    document_store: DocumentStore
    run_store: RunStore
```

**`build_pipeline(settings)` wiring:**
1. PostgresClient + DocumentStore + RunStore
2. QdrantStore + OpenSearchStore (legacy)
3. CogneeStore with full configuration (Neo4j, Qdrant, LLM)
4. RetrievalRouter (Cognee-only: graph + vector + lexical)
5. ResultRanker (RRF with k=60)
6. EmbeddingPipeline + QueryEmbedder
7. LLMGenerator (OpenAI chat model)
8. GroundingVerifier (OpenAI with verification prompt)
9. MemoryStore (over Postgres)
10. ResponseSynthesizer
11. MCPToolClient (Brave Search at configured URL)
12. RAGMetrics (OpenTelemetry)
13. AgentOrchestrator (wires everything together)

**Lifecycle**: `RAGPipeline.close()` gracefully shuts down all clients (Qdrant, OpenSearch, Postgres).

---

## 14. Database Migrations

Managed by **Alembic** (`alembic.ini` + `stores/postgres/migrations/`).

### Migration History

| Revision       | Description                              | Table(s) Affected      |
| -------------- | ---------------------------------------- | ---------------------- |
| `1da85e5b5d58` | Create `chat_messages` table             | `chat_messages`        |
| `a2b3c4d5e6f7` | Create `memory_entries` table            | `memory_entries`       |
| `b3c4d5e6f7a8` | Create `parsed_documents` table          | `parsed_documents`     |
| `c4d5e6f7a8b9` | Create `pipeline_runs` table             | `pipeline_runs`        |
| `d5e6f7a8b9c0` | Add composite index `(user_id, source_path)` | `parsed_documents` |
| `71bd3ba7d1c8` | Add `user_id` column to `parsed_documents` | `parsed_documents`   |
| `a24399359597` | Create `llm_calls` table                 | `llm_calls`            |

### Running Migrations

```bash
# Apply all pending migrations
uv run alembic upgrade head

# Generate new migration after model changes
uv run alembic revision --autogenerate -m "description"
```

---

## 15. Scripts & E2E Testing

### Scripts (`scripts/`)

| Script                    | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `ingest_file.py`          | Ingest a file (PDF, DOCX, HTML, etc.) into the RAG pipeline |
| `ingest_sample.py`        | Ingest sample/test documents                                 |
| `e2e_ingest_and_query.py` | Full E2E test: ingest PDF → query API → validate response    |
| `create_qdrant_collection.py` | Create Qdrant collection with configured settings        |
| `query_cognee.py`         | Direct Cognee search testing                                 |
| `reset_cognee.py`         | Reset Cognee data (prune + re-initialize)                    |

### E2E Test Flow (`e2e_ingest_and_query.py`)

```
1. Initialize all services (Postgres, Cognee, Embedding pipeline)
2. Parse test PDF via Docling → DoclingDocument
3. Persist DoclingDocument JSON to Postgres
4. Ingest chunks into Cognee (add + cognify → Qdrant + Neo4j)
5. Query the live FastAPI gateway API at /api/v1/query
6. Validate response: answer present, citations exist, is_grounded=true
7. Report: timing, chunk counts, retrieval results, confidence
```

**Usage:**
```bash
# Full E2E (ingest + query)
uv run python scripts/e2e_ingest_and_query.py

# Custom file
uv run python scripts/e2e_ingest_and_query.py my_document.pdf

# Query only (skip ingestion)
uv run python scripts/e2e_ingest_and_query.py --skip-ingest
```

### Ingestion Flow (`ingest_file.py`)

```
Docling parse → Postgres (DoclingDocument JSON) → Cognee add + cognify
                                                    ├── Qdrant (embeddings)
                                                    └── Neo4j (entities + relations)
```

**Usage:**
```bash
uv run python scripts/ingest_file.py document.pdf
uv run python scripts/ingest_file.py file1.pdf file2.docx file3.html
```

---

## 16. Dependencies

### Runtime Dependencies

```toml
dependencies = [
    "alembic>=1.15",           # Database migrations
    "asyncpg>=0.30",           # Async PostgreSQL driver
    "cognee[neo4j]>=0.3",      # Knowledge graph engine + Neo4j support
    "cognee-community-vector-adapter-qdrant>=0.1",  # Cognee → Qdrant adapter
    "docling>=2.31",           # Structure-aware document parsing
    "docling-core>=2.31",      # Docling core types + chunker
    "fastapi>=0.115",          # API framework
    "haystack-ai>=2.13",       # Agent framework + generators + embedders
    "httpx>=0.28",             # Async HTTP client
    "mcp>=1.12",               # Model Context Protocol SDK
    "pydantic-settings>=2.9",  # Typed config from env vars
    "qdrant-client>=1.13",     # Qdrant vector DB client
    "opentelemetry-api>=1.29",                          # OTel API
    "opentelemetry-exporter-otlp>=1.29",                # OTel OTLP exporter
    "opentelemetry-instrumentation-fastapi>=0.50b0",    # FastAPI auto-instrumentation
    "opentelemetry-sdk>=1.29",                          # OTel SDK
    "sqlalchemy[asyncio]>=2.0",# Async ORM
    "uvicorn>=0.34",           # ASGI server
]
```

### Dev Dependencies

```toml
[dependency-groups]
dev = [
    "mypy>=1.19.1",   # Static type checking
    "ruff>=0.15.7",    # Linting + formatting
]
```

---

## 17. Git History & Milestones

### Recent Commits (latest first)

| Hash      | Message                                                                  |
| --------- | ------------------------------------------------------------------------ |
| `e2be4b8` | feat(embedder): add legislatie.just.ro HTML preprocessor + fix Py3.14 chunker |
| `5c4aa9c` | feat(cost): record agent + verifier LLM costs to Postgres               |
| `4ce74d5` | fix: wire Brave MCP client in factory, fix docker-compose               |
| `9d0836b` | refactor(embedder): replace BERT tokenizer with char-based approximation |
| `fc21798` | feat(embedder): align Docling settings with lawren-docling              |
| `aad6463` | fix: resolve heavy review v2 findings (2 critical, 4 important)         |
| `50109e0` | docs: add complete system architecture diagram and inventory             |
| `03b5941` | fix: resolve all heavy review findings (6 critical, 7 important)        |
| `00f18ab` | refactor(query): accept strategy as string hint for agent-based dispatch |
| `3bfb5af` | refactor(orchestrator): replace sequential pipeline with Haystack Agent  |
| `2ab0aff` | feat(agent): add Haystack Agent builder with ReACT tool-calling          |
| `8d31571` | feat(agent): add 4 high-level tools — search_knowledge, search_web, ingest, remember |
| `39ff0c0` | feat(config): add agent_model, agent_max_steps, agent_max_retries        |
| `966a7d8` | fix: resolve all remaining audit issues — explicit only_context, stale refs |
| `65fff44` | docs: update CLAUDE.md for Cognee-first architecture                     |
| `0db5aaa` | refactor(core): rewrite router+factory+app for Cognee-only pipeline      |
| `b5dafb6` | fix(cognee): add only_context=True to search + add prune()              |
| `f8401a4` | fix: address all heavy review findings (7 critical, 8 important)        |

### Architecture Evolution

1. **Phase 1-2**: Foundation — Docker infra, config, errors, logging, Postgres models + migrations
2. **Phase 3**: Retrieval — Embedder (Docling + OpenAI), Router, Ranker (RRF)
3. **Phase 4**: Agent — Memory, Generator, Verifier, Synthesis, Orchestrator
4. **Phase 5**: Tools — MCP client, Brave Search integration
5. **Cognee migration**: Replaced direct Qdrant/OpenSearch with Cognee as sole retrieval backend
6. **Agent refactor**: Replaced sequential pipeline with Haystack Agent + ReACT tool-calling
7. **Embedder refinements**: legislatie.just.ro preprocessor, char-based tokenizer for Py3.14
8. **Cost tracking**: LiteLLM callback → Postgres `llm_calls` table

---

## 18. Project Structure

```
agentic-rag/
├── src/agentic_ra/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py              # Pydantic Settings (all env vars)
│   │   ├── errors.py              # Exception hierarchy (14 types)
│   │   └── logging.py             # Structured logging setup
│   ├── stores/
│   │   ├── postgres/
│   │   │   ├── client.py          # Async SQLAlchemy client
│   │   │   ├── models.py          # 5 ORM models
│   │   │   ├── documents.py       # DoclingDocument JSON persistence
│   │   │   ├── runs.py            # Pipeline run tracking
│   │   │   └── migrations/        # 7 Alembic migrations
│   │   ├── qdrant/
│   │   │   ├── client.py          # Qdrant client wrapper
│   │   │   └── collections.py     # Collection management
│   │   ├── opensearch/
│   │   │   └── client.py          # OpenSearch BM25 wrapper (legacy)
│   │   └── cognee/
│   │       └── client.py          # Cognee SDK wrapper (add/cognify/search)
│   ├── retrieval/
│   │   ├── embedder.py            # Docling parse → HybridChunker → OpenAI embed
│   │   ├── html_preprocessor.py   # legislatie.just.ro HTML → semantic HTML
│   │   ├── router.py              # Strategy router (vector/keyword/hybrid/graph)
│   │   └── ranker.py              # Dedup + RRF + EvidencePack assembly
│   ├── agent/
│   │   ├── agent.py               # Haystack Agent builder (ReACT)
│   │   ├── tools.py               # 4 tools: search_knowledge, search_web, ingest, remember
│   │   ├── memory.py              # Short-term (session) + long-term (persistent)
│   │   ├── generator.py           # LLM query rewrite + answer generation
│   │   ├── orchestrator.py        # Central pipeline: memory→retrieve→generate→verify→synthesize
│   │   ├── verifier.py            # LLM-as-judge faithfulness checker
│   │   └── synthesis.py           # Citation + provenance assembly
│   ├── tools/
│   │   └── mcp_client.py          # MCP tool discovery + invocation (Brave Search)
│   ├── auth/
│   │   └── opa.py                 # OPA policy client
│   ├── gateway/
│   │   ├── app.py                 # FastAPI factory (lifespan, middleware, error handling)
│   │   ├── routes/
│   │   │   ├── health.py          # GET /health
│   │   │   ├── query.py           # POST /api/v1/query
│   │   │   └── ingest.py          # POST /api/v1/ingest
│   │   └── middleware/
│   │       └── auth.py            # OPA authorization middleware
│   ├── observability/
│   │   ├── tracing.py             # OTel tracing + FastAPI instrumentation
│   │   ├── metrics.py             # RAGMetrics (latency, confidence, hit rates)
│   │   └── cost_tracker.py        # LiteLLM callback → Postgres cost recording
│   └── pipelines/
│       └── factory.py             # RAGPipeline assembly (13 components wired)
├── infra/
│   ├── docker-compose.yml         # 8 services, all health-checked
│   ├── .env / .env.example        # Port mappings + credentials
│   ├── postgres/init.sql          # uuid-ossp + pgcrypto extensions
│   ├── qdrant/config.yaml         # Production config override
│   ├── opa/policies/authz.rego    # Authorization policy skeleton
│   └── otel/collector.yaml        # OTel Collector pipeline config
├── scripts/
│   ├── ingest_file.py             # CLI file ingestion
│   ├── ingest_sample.py           # Sample document ingestion
│   ├── e2e_ingest_and_query.py    # Full E2E test (ingest + query + validate)
│   ├── create_qdrant_collection.py# Qdrant collection setup
│   ├── query_cognee.py            # Direct Cognee search test
│   └── reset_cognee.py            # Cognee data reset
├── docs/
│   └── superpowers/specs/         # Design specs and plans
├── alembic.ini                    # Alembic configuration
└── pyproject.toml                 # Project metadata + dependencies
```

---

## 19. Configuration Reference

All configuration is via environment variables, managed by `core/config.py` (Pydantic Settings).

| Variable                        | Default                          | Description                              |
| ------------------------------- | -------------------------------- | ---------------------------------------- |
| `LLM_API_KEY`                   | (required)                       | OpenAI API key                           |
| `LLM_MODEL`                     | `gpt-4o`                         | Chat model for generation                |
| `AGENT_MODEL`                   | `gpt-4o`                         | Model for the Haystack Agent             |
| `AGENT_MAX_STEPS`               | `10`                             | Max ReACT agent steps per query          |
| `AGENT_MAX_RETRIES`             | `2`                              | Max Stage-2 retry attempts               |
| `EMBEDDING_MODEL`               | `text-embedding-3-small`         | OpenAI embedding model                   |
| `EMBEDDING_DIMENSION`           | `1536`                           | Embedding vector dimension               |
| `LLM_ENRICHMENT_MODEL`          | `openai/gpt-4o-mini`            | Model for Cognee entity extraction       |
| `CHUNK_MAX_TOKENS`              | `512`                            | Max tokens per chunk                     |
| `CHUNK_MERGE_PEERS`             | `true`                           | Merge adjacent small chunks              |
| `POSTGRES_DSN_ASYNC`            | `postgresql+asyncpg://...`       | Async Postgres connection string         |
| `QDRANT_URL`                    | `http://localhost:1051`          | Qdrant HTTP endpoint                     |
| `QDRANT_API_KEY`                | (optional)                       | Qdrant authentication key                |
| `QDRANT_COLLECTION`             | `agentic_rag`                    | Default Qdrant collection name           |
| `COGNEE_ENABLED`                | `true`                           | Enable Cognee knowledge graph            |
| `COGNEE_DATASET`                | `agentic_rag`                    | Cognee dataset name                      |
| `COGNEE_GRAPH_DB_PROVIDER`      | `neo4j`                          | Graph database backend                   |
| `COGNEE_VECTOR_DB_PROVIDER`     | `qdrant`                         | Vector database backend                  |
| `NEO4J_BOLT_URL`                | `bolt://localhost:1061`          | Neo4j Bolt protocol endpoint             |
| `NEO4J_USER`                    | `neo4j`                          | Neo4j username                           |
| `NEO4J_PASSWORD`                | (required)                       | Neo4j password                           |
| `NEO4J_DB`                      | `neo4j`                          | Neo4j database name                      |
| `OPA_URL`                       | `http://localhost:1055`          | OPA policy engine URL                    |
| `OPA_AUTH_ENABLED`              | `false`                          | Enable/disable OPA middleware            |
| `BRAVE_MCP_URL`                 | `http://localhost:1063`          | Brave Search MCP server URL              |
| `OTEL_ENDPOINT`                 | `http://localhost:1057`          | OTel Collector gRPC endpoint             |
| `LOG_LEVEL`                     | `INFO`                           | Application log level                    |
| `GROUNDING_CONFIDENCE_THRESHOLD`| `0.7`                            | Min confidence to accept answer          |

---

## Quick Start

```bash
# 1. Start infrastructure
cd infra && docker compose up -d

# 2. Run database migrations
uv run alembic upgrade head

# 3. Ingest a document
uv run python scripts/ingest_file.py my_document.pdf

# 4. Start the gateway
uv run uvicorn agentic_ra.gateway.app:create_app --factory --host 0.0.0.0 --port 8000

# 5. Query the API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Article 5 say?", "strategy": "hybrid"}'

# 6. Run E2E tests
uv run python scripts/e2e_ingest_and_query.py
```
