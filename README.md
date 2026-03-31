# Agentic RAG

A self-learning Retrieval-Augmented Generation system that improves with every query.

Built on **Haystack** + **Cognee** + **Neo4j** + **Qdrant**, served via **FastAPI**, with full observability through **Grafana** dashboards.

## What Makes This Different

Most RAG systems are static --- they retrieve and generate the same way forever, regardless of whether the results are good or bad.

This system has **three feedback loops** that learn from every query:

```
Query --> Retrieve --> Generate --> Verify --> Respond
              ^                       |
              |                 training signal
              |                       |
         +---------+----------+-------+
         |         |          |
      Loop 1    Loop 2     Loop 3
      Graph     Strategy   Vector
      Learning  Learning   Learning
```

**Loop 1 --- Graph Learning:** After each query, the verifier's confidence score feeds back into the knowledge graph. Neo4j nodes and edges that contributed to grounded answers get upweighted; those that led to hallucinations get downweighted. Powered by Cognee's memify pipeline with exponential moving average updates.

**Loop 2 --- Strategy Learning:** A dedicated Qdrant collection stores (query embedding, strategy, confidence) tuples. Before routing, the system finds similar past queries and picks the strategy that historically performed best. 10 retrieval strategies available including graph-constrained vector search.

**Loop 3 --- Vector Index Learning:** RuVector's GNN-guided HNSW replaces traditional greedy nearest-neighbor search with a trained neural router that learns which index paths lead to better results for which query patterns.

Each loop is independently deployable, feature-flagged, and observable through Grafana dashboards with regression detection.

## Architecture

```
Client
  --> FastAPI Gateway (OPA auth middleware)
    --> Haystack Agent (ReACT tool-calling)
      |-- MemoryStore (session chat + long-term facts --> Postgres)
      |-- search_knowledge tool
      |     |-- Strategy Learner (Loop 2) predicts best strategy
      |     |-- RetrievalRouter dispatches to Cognee
      |     |     |-- CHUNKS (vector search via Qdrant)
      |     |     |-- CHUNKS_LEXICAL (keyword/BM25)
      |     |     |-- GRAPH_COMPLETION (entity-relation traversal via Neo4j)
      |     |     |-- GRAPH_CONSTRAINED (graph narrows vector search)
      |     |     +-- 6 more strategies (RAG_COMPLETION, SUMMARIES, TEMPORAL, ...)
      |     |-- ResultRanker (dedup + Reciprocal Rank Fusion)
      |     +-- Evidence fed to LLM Generator
      |-- search_web tool (Brave Search via MCP)
      |-- remember tool (long-term memory)
      |-- ingest tool (document ingestion)
      |
      --> LLM Generator (answer with citations)
      --> Grounding Verifier (faithfulness check)
      --> FeedbackAdapter (Loop 1) --> Cognee memify --> Neo4j weights
      --> Strategy Learner records outcome (Loop 2)
      --> ResponseSynthesizer (citations + provenance)
      --> RunStore (pipeline_runs --> Postgres)
      --> CostTracker (llm_calls --> Postgres)
    --> OpenTelemetry --> Prometheus --> Grafana
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent Framework | Haystack 2.13+ | ReACT agent, generators, embedders, tools |
| Knowledge Graph | Cognee 0.5.5 + Neo4j 5 | Entity extraction, graph building, graph-augmented search |
| Vector DB | Qdrant 1.14 | Embedding storage, similarity search, strategy collection |
| State Store | PostgreSQL 17 + SQLAlchemy 2 | Chat history, memory, pipeline runs, LLM cost tracking |
| Document Parsing | Docling 2.31 | Structure-aware PDF/HTML/DOCX parsing with HybridChunker |
| API Server | FastAPI + Uvicorn | HTTP gateway with OPA authorization middleware |
| MCP Tools | MCP Python SDK | Brave Search, web fetching, tool discovery |
| Observability | OpenTelemetry + Prometheus + Grafana | Traces, metrics, dashboards, regression alerts |
| Future Vector | RuVector | Self-learning HNSW with GNN-guided routing |
| Language | Python 3.12+ | Async throughout, full type annotations |
| Package Manager | uv | Fast dependency resolution |

## Infrastructure

All services run via Docker Compose on a single bridge network.

| Service | Port | Purpose |
|---|---|---|
| Qdrant | 1050 (gRPC), 1051 (HTTP) | Vector similarity search |
| PostgreSQL | 1054 | State store (5 tables, 7 migrations) |
| OPA | 1055 | Policy-based authorization |
| OTel Collector | 1057 | Telemetry pipeline (traces + metrics) |
| Jaeger | 1058 | Distributed trace visualization |
| Neo4j | 1061 (Bolt), 1062 (HTTP) | Knowledge graph store |
| Brave Search MCP | 1063 | Web search via Model Context Protocol |
| Prometheus | 1066 | Metrics storage (scrapes OTel Collector) |
| Grafana | 1067 | Dashboards, alerts, regression detection |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd agentic-rag
uv sync

# 2. Configure
cp infra/.env.example infra/.env
# Edit infra/.env --- set LLM_API_KEY, POSTGRES_PASSWORD, NEO4J_PASSWORD, etc.

# 3. Start infrastructure
cd infra && docker compose up -d

# 4. Run database migrations
uv run alembic upgrade head

# 5. Ingest a document
uv run python scripts/ingest_file.py test.pdf

# 6. Start the gateway
uv run uvicorn agentic_ra.gateway.app:create_app --factory --host 0.0.0.0 --port 8000

# 7. Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Article 5 say?", "strategy": "hybrid"}'
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health + version |
| `POST` | `/api/v1/query` | RAG query --- returns answer + citations + confidence |
| `POST` | `/api/v1/ingest` | Document ingestion --- text to Cognee knowledge graph |

### Query Request

```json
{
  "query": "What are the penalties for Article 12 violations?",
  "session_id": "optional-uuid",
  "user_id": "anonymous",
  "strategy": "hybrid",
  "limit": 10
}
```

### Query Response

```json
{
  "answer": "Article 12 violations are subject to...",
  "citations": [
    {"index": 1, "content_snippet": "...", "source": "cognee_chunks", "score": 0.95}
  ],
  "confidence": 0.92,
  "is_grounded": true,
  "strategy": "hybrid",
  "session_id": "uuid"
}
```

## Retrieval Strategies

| Strategy | Search Types | Best For |
|---|---|---|
| `semantic` | Vector similarity (Qdrant) | Factual lookups, keyword-rich queries |
| `keyword` | BM25 lexical (Cognee) | Exact term matching |
| `hybrid` | Vector + keyword | General purpose (default) |
| `graph` | Graph traversal (Neo4j) | Entity relationships, multi-hop reasoning |
| `graph_hybrid` | Graph + vector + keyword | Complex questions needing both |
| `graph_constrained` | Graph narrows vector search | Entity-focused questions |
| `rag_completion` | Traditional RAG via Cognee | Simple Q&A |
| `summaries` | Pre-generated summaries | High-level overviews |
| `temporal` | Time-aware search | Questions about events, dates |
| `graph_cot` | Chain-of-thought over graph | Multi-step reasoning |

## Self-Learning Loops

### Loop 1: Graph Feedback Learning

Every query's verification score flows back to the knowledge graph:

1. Query runs with `session_id` --- Cognee tracks which graph nodes/edges were used
2. Verifier scores confidence (0.0--1.0) --- mapped to Cognee feedback score (1--5)
3. `FeedbackAdapter` submits score via `cognee.session.add_feedback()`
4. Periodic `apply_feedback_weights_pipeline()` updates Neo4j node/edge weights using EMA

```
Confidence 0.0-0.2 --> Score 1 (suppress these nodes)
Confidence 0.2-0.4 --> Score 2
Confidence 0.4-0.6 --> Score 3 (neutral)
Confidence 0.6-0.8 --> Score 4
Confidence 0.8-1.0 --> Score 5 (boost these nodes)
```

**Safety:** Kill switch (`feedback_enabled`), dry-run mode, weight floor/ceiling (0.1--0.9), rollback script (`scripts/reset_feedback_weights.py`).

### Loop 2: Strategy Learning

A `query_strategies` Qdrant collection remembers which strategies worked:

1. Before routing: embed query, find 20 similar past queries, pick highest-confidence strategy
2. After verification: record (embedding, strategy, confidence) to the collection
3. Cold start: falls back to "hybrid" until enough data accumulates (min 5 neighbors)
4. On verifier rejection: PIKE-RAG pattern --- try all strategies, record which one works

### Loop 3: RuVector GNN (Future)

Replaces Qdrant with RuVector for vector search. The GNN learns HNSW routing paths from query patterns. Blocked on Python SDK availability --- REST client and Cognee adapter are ready.

## Dashboards

Two Grafana dashboards auto-provision on startup at `http://localhost:1067`:

### RAG Overview Dashboard

| Panel | Data Source | Shows |
|---|---|---|
| Queries/Hour | Prometheus | Query volume trend |
| Avg Confidence (24h) | PostgreSQL | Verification quality gauge |
| Grounding Rate (24h) | PostgreSQL | % of grounded answers |
| Total LLM Cost | PostgreSQL | Cumulative spend |
| Retrieval Latency P95 | Prometheus | Performance |
| Confidence by Strategy | PostgreSQL | Which strategy produces best results |
| Strategy Usage | PostgreSQL | Distribution of strategy selection |
| Retries by Strategy | PostgreSQL | Which strategies need retries |
| Cost by Model | PostgreSQL | Spend breakdown by LLM model |
| Cost per Query Trend | PostgreSQL | Cost efficiency over time |
| Token Usage by Model | PostgreSQL | Input/output token breakdown |

### Loop 1 Feedback Dashboard

| Panel | Data Source | Shows |
|---|---|---|
| Feedback Applied/Hour | Prometheus | Rate of graph weight updates |
| Feedback Score Distribution | Prometheus | Score histogram (1--5) |
| Confidence Trend (regression detector) | PostgreSQL | 24h vs 7d baseline --- alerts on >10% drop |
| Ungrounded Rate Trend | PostgreSQL | Should decrease over time |
| Confidence Over Time | PostgreSQL | Long-term quality trend |
| Queries Processed Today | PostgreSQL | Daily volume |
| Average Retries | PostgreSQL | Should decrease with learning |

## Document Ingestion

Supports PDF, DOCX, PPTX, HTML, and more via Docling:

```bash
# Single file
uv run python scripts/ingest_file.py document.pdf

# Multiple files
uv run python scripts/ingest_file.py file1.pdf file2.docx file3.html

# Batch feedback apply (after queries accumulate)
uv run python scripts/apply_feedback.py --session <session-id>

# Emergency weight rollback
uv run python scripts/reset_feedback_weights.py --confirm
```

### Ingestion Pipeline

```
Document --> Docling parse --> DoclingDocument (preserves structure)
         --> HybridChunker (structure-aware + token-budget)
         --> Contextualize (inject heading/caption context)
         --> Postgres (persist DoclingDocument JSON)
         --> Cognee add + cognify
              |--> Qdrant (chunk embeddings)
              +--> Neo4j (entities + relations)
```

Special handling for `legislatie.just.ro` Romanian legal HTML --- auto-detected and preprocessed into semantic HTML before Docling conversion.

## Database Schema

| Table | Purpose | Key Columns |
|---|---|---|
| `chat_messages` | Session conversation history | session_id, role, content |
| `memory_entries` | Long-term agent memory (key-value) | session_id, key, value |
| `parsed_documents` | DoclingDocument JSON persistence | source_path, user_id, docling_json, chunk_count |
| `pipeline_runs` | Query execution tracking | query, strategy, confidence, is_grounded, retrieval_ms, retries |
| `llm_calls` | LLM cost tracking | model, prompt_tokens, completion_tokens, cost_usd |

## Configuration

All configuration via environment variables. See `infra/.env.example` for the complete list.

Key settings:

```bash
# LLM
LLM_API_KEY=sk-...                    # OpenAI API key
LLM_MODEL=gpt-4o                      # Chat model
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model

# Feedback (Loop 1)
FEEDBACK_ENABLED=true                   # Kill switch
FEEDBACK_ALPHA=0.1                      # EMA learning rate
FEEDBACK_DRY_RUN=false                  # Compute but don't write
FEEDBACK_WEIGHT_FLOOR=0.1              # Min weight (prevents zero)
FEEDBACK_WEIGHT_CEILING=0.9            # Max weight (prevents runaway)

# Strategy Learning (Loop 2)
STRATEGY_LEARNING_ENABLED=false         # Enable after collection setup
STRATEGY_COLLECTION=query_strategies    # Qdrant collection name
STRATEGY_MIN_NEIGHBORS=5               # Min similar queries for prediction

# RuVector (Loop 3)
RUVECTOR_ENABLED=false                  # Requires RuVector service
RUVECTOR_URL=http://localhost:1064
```

## Testing

```bash
# Self-learning feature tests (37 test cases)
uv run python scripts/e2e_self_learning.py

# Full pipeline E2E (requires running infrastructure)
uv run python scripts/e2e_full_pipeline.py

# Query-only (skip ingestion)
uv run python scripts/e2e_full_pipeline.py --skip-ingest

# Loop-specific tests
uv run python scripts/e2e_self_learning.py --loop 1
uv run python scripts/e2e_self_learning.py --loop 2
```

## Project Structure

```
agentic-rag/
  src/agentic_ra/
    core/                    # Config, errors, logging
    stores/
      postgres/              # Client, 5 ORM models, 7 Alembic migrations
      qdrant/                # Qdrant client (strategy collection)
      cognee/                # Cognee SDK wrapper (add/cognify/search)
      ruvector/              # RuVector REST client + Cognee adapter
    retrieval/
      embedder.py            # Docling + OpenAI embedding pipeline
      router.py              # 10 retrieval strategies
      ranker.py              # Dedup + RRF + EvidencePack
      strategy_learner.py    # Embedding-similarity strategy prediction
    agent/
      agent.py               # Haystack Agent builder (ReACT)
      tools.py               # search_knowledge, search_web, ingest, remember
      orchestrator.py        # Central pipeline (all loops wired)
      feedback.py            # FeedbackAdapter (Loop 1)
      memory.py              # Short-term + long-term memory
      generator.py           # LLM query rewrite + answer generation
      verifier.py            # LLM-as-judge faithfulness checker
      synthesis.py           # Citation + provenance assembly
    tools/
      mcp_client.py          # MCP tool discovery + invocation
    auth/
      opa.py                 # OPA policy client
    gateway/
      app.py                 # FastAPI factory
      routes/                # health, query, ingest endpoints
      middleware/             # OPA auth middleware
    observability/
      tracing.py             # OpenTelemetry + FastAPI instrumentation
      metrics.py             # 12 custom OTel metrics
      cost_tracker.py        # LiteLLM callback --> Postgres
      gnn_trainer.py         # GNN training pipeline (Loop 3)
    pipelines/
      factory.py             # RAGPipeline assembly (all components)
  infra/
    docker-compose.yml       # 9 services, all health-checked
    grafana/                 # 2 dashboards + datasource provisioning
    prometheus/              # Scrape config
    otel/                    # Collector pipeline config
    postgres/                # init.sql (uuid-ossp, pgcrypto)
    opa/policies/            # Authorization policy skeleton
  scripts/
    ingest_file.py           # CLI document ingestion
    e2e_self_learning.py     # 37 self-learning feature tests
    e2e_full_pipeline.py     # Full integration test with Playwright
    apply_feedback.py        # Batch feedback processing
    reset_feedback_weights.py # Emergency weight rollback
    train_gnn.py             # Manual GNN training trigger
  docs/
    superpowers/specs/       # Design specifications
    superpowers/plans/       # Implementation plans
    reviews/                 # Code review reports
```

## Research Validation

The self-learning architecture is independently validated by:

- **PIKE-RAG** (Microsoft Research, 2025) --- same 3 patterns: log-based feedback, strategy evaluation on failures, multi-layer graph learning
- **GraphRAG** (Microsoft) --- community hierarchy + Leiden detection for stable feedback weights
- **Fabric IQ** (Microsoft) --- graph-constrained vector search pattern
- **Adaptive-RAG** (NAACL 2024) --- trained classifier for query complexity routing
- **RouteRAG** (arXiv, Dec 2025) --- RL-based graph+text retrieval routing

## License

Private project.
