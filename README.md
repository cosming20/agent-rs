# agent-rs

Production-grade agentic RAG system in Rust. Ingests documents via Docling, builds a knowledge graph, and retrieves evidence across vector, keyword, and graph search — fused with RRF ranking.

**84.8% Hit@1** on a 125-page Romanian legal corpus (33 diverse questions, 10 question types).

## Architecture

```
PDF/DOCX/HTML
    |
    v  gRPC streaming (port 1068)
+-----------------------------------+
| Docling Sidecar (Python Docker)   |
| DocumentConverter (TableFormer)   |
| HybridChunker (cl100k/1024)      |
| heading-prefixed embed_text      |
+-----------------------------------+
    |
    v  tonic streaming client
+-----------------------------------+
| agent-rs (Rust)                   |
|                                   |
| Ingestion:                        |
|   embed (OpenAI 3-large 3072)    |
|   Qdrant upsert (dense + BM25)   |
|   LLM KG extraction (gpt-5.4)   |
|   Neo4j structural + entity graph|
|   triplet indexing + summaries    |
|   Postgres state machine         |
|                                   |
| Retrieval:                        |
|   query decomposition (LLM)      |
|   metadata Qdrant filtering      |
|   entity -> KG linking           |
|   5 strategies: semantic,        |
|     keyword, hybrid, graph,      |
|     graph_hybrid                  |
|   RRF fusion + Jina reranking    |
|   context expansion (FOLLOWS)    |
|   sufficiency evaluation         |
+-----------------------------------+
    |
    v
Qdrant (vectors) + Neo4j (graph) + Postgres (state)
```

## Quick start

### 1. Prerequisites

- Rust 1.85+ (`rustup update`)
- Docker Desktop with **16 GB RAM** (Settings > Resources > Memory)
- OpenAI API key
- Jina API key (optional, for reranking)

### 2. Start infrastructure

```bash
cd infra
docker compose up -d
```

This starts: Qdrant (1050/1051), PostgreSQL (1054), Neo4j (1061), Jaeger (1058), OPA (1055), Prometheus, Grafana.

### 3. Build the Docling sidecar

```bash
docker compose build docling-sidecar
docker compose up -d docling-sidecar
```

Wait for healthy (~15s):
```bash
docker exec agentic-ra-docling-sidecar /usr/local/bin/grpc_health_probe -addr=:1068
# Expected: status: SERVING
```

### 4. Configure environment

```bash
cp .env.example .env   # or create manually:
```

```env
# Required
OPENAI_API_KEY=sk-...
LLM_API_KEY=sk-...          # same key, used by KG extraction

# Optional
JINA_API_KEY=jina_...       # for cross-encoder reranking
EMBEDDER_TYPE=openai         # or "local" for fastembed
AGENT_RS__INGEST_BACKEND=docling   # or "legacy" for pdf-extract

# Infrastructure (defaults match docker-compose)
AGENT_RS__QDRANT__HOST=localhost
AGENT_RS__QDRANT__GRPC_PORT=1050
AGENT_RS__QDRANT__HTTP_PORT=1051
AGENT_RS__QDRANT__API_KEY=changeme-qdrant-key
AGENT_RS__POSTGRES__HOST=localhost
AGENT_RS__POSTGRES__PORT=1054
AGENT_RS__POSTGRES__USER=agentic_ra
AGENT_RS__POSTGRES__PASSWORD=changeme-postgres
AGENT_RS__POSTGRES__DB=agentic_ra
AGENT_RS__NEO4J__HOST=localhost
AGENT_RS__NEO4J__BOLT_PORT=1061
AGENT_RS__NEO4J__USER=neo4j
AGENT_RS__NEO4J__PASSWORD=changeme-neo4j
AGENT_RS__DOCLING__ENDPOINT=http://localhost:1068
```

### 5. Run the evaluation

```bash
# Build release binary
cargo build --release --example eval_harness

# Run with a PDF
PDF_PATH=../test.pdf \
EVAL_DATASET=eval/test_questions_v2.json \
cargo run --release --example eval_harness
```

Results are written to `test_results/eval_report_<timestamp>.json`.

### 6. Ingest your own documents

```bash
PDF_PATH=/path/to/your/document.pdf \
EVAL_DATASET=eval/your_questions.json \
cargo run --release --example eval_harness
```

The eval dataset JSON schema:
```json
{
  "schema_version": "2",
  "dataset_name": "my_eval",
  "allowed_difficulties": ["easy", "medium", "hard"],
  "allowed_types": ["single_hop", "multi_hop", "lexical_exact", "conceptual",
                     "negation", "temporal", "comparison", "disambiguation",
                     "summary_only", "bridge"],
  "documents": [
    { "path": "/path/to/doc.pdf", "doc_id": "my_doc", "source": "my_source" }
  ],
  "questions": [
    {
      "id": "q01",
      "difficulty": "easy",
      "type": "single_hop",
      "question": "What is X?",
      "expected_doc_ids": [],
      "expected_text_contains": ["keyword"]
    }
  ]
}
```

## Workspace structure

```
agent-rs/
  crates/
    core/           # Config, errors, types, logging
    stores/         # Qdrant, Neo4j, Postgres adapters + traits
    retrieval/      # Embedder, search, routing, ranking, ingestion
    agent/          # Orchestrator, verifier, synthesis, memory
    tools/          # search_knowledge, search_web, fetch_url, remember
    server/         # tonic gRPC service layer
    observability/  # OpenTelemetry tracing + metrics
  proto/            # Protobuf service definitions
  eval/             # Evaluation datasets + generator
  examples/         # eval_harness, e2e_legal_qa, e2e_full_trace
  experiments/      # Docling chunker feedback loop lab
```

## Retrieval strategies

| Strategy | What it does | Best for |
|----------|-------------|----------|
| `semantic` | Dense vector similarity (OpenAI 3-large) | Concept matching |
| `keyword` | Server-side BM25 via Qdrant | Exact term lookup |
| `hybrid` | Dense + BM25 fused via RRF | General purpose |
| `graph` | Neo4j entity traversal | Relational questions |
| `graph_hybrid` | All three fused | Best overall (84.8%) |

## Cognitive retrieval features

| Feature | What it does |
|---------|-------------|
| **Query decomposition** | LLM splits complex queries into 2-3 focused sub-queries (>12 words only) |
| **Metadata filtering** | LLM extracts page range, article number, content type from query -> Qdrant filter |
| **Entity linking** | Checks if query entities exist in Neo4j -> bias toward graph strategy |
| **Context expansion** | After retrieval, fetches +-2 neighbor chunks via FOLLOWS edges |
| **Sufficiency evaluation** | LLM evaluates if evidence is enough -> refined query if not |
| **Diagnostic retry** | On grounding failure, retries with specific ungrounded claims |
| **Jina reranking** | Cross-encoder reranking with 2s/4s/8s exponential backoff |

## Skip flags

Disable features via environment variables for debugging:

```bash
EVAL_SKIP_KG_EXTRACTION=1    # Skip LLM knowledge graph extraction
EVAL_SKIP_SUMMARIES=1         # Skip chunk summarization
EVAL_SKIP_RERANKING=1         # Skip Jina reranker
EVAL_SKIP_GRAPH_COT=1         # Skip GraphCoT iterative retrieval
EVAL_SKIP_QUERY_REWRITER=1    # Skip LLM query decomposition
EVAL_SKIP_METADATA_FILTER=1   # Skip metadata filter extraction
EVAL_SKIP_CONTEXT_EXPANDER=1  # Skip FOLLOWS/IN_SECTION expansion
EVAL_SKIP_ENTITY_LINKER=1     # Skip entity -> KG linking
```

## Infrastructure ports

| Port | Service |
|------|---------|
| 1050 | Qdrant gRPC |
| 1051 | Qdrant HTTP |
| 1054 | PostgreSQL |
| 1055 | OPA |
| 1057 | OTel Collector |
| 1058 | Jaeger UI |
| 1061 | Neo4j Bolt |
| 1062 | Neo4j HTTP |
| 1063 | Brave Search MCP |
| 1068 | Docling sidecar gRPC |
| 1069 | MinIO S3 API |
| 1070 | MinIO Console UI |

## Multi-tenant rollout (April 2026)

This release introduces strict per-user document scoping across Postgres, Qdrant, Neo4j, and MinIO. Documents uploaded under one `user_id` are invisible to any other `user_id` at every retrieval layer. Upgrade steps (destroys all existing data — reingest required):

```bash
just infra-reset      # wipes Postgres/Qdrant/Neo4j volumes
just migrations       # applies all diesel migrations (incl. multi_tenant)
just minio-bootstrap  # creates the agent-rs-docs bucket
just ingest "<user_id>" "<path/to/doc.pdf>"
just ask "<user_id>" "" "<question>"   # empty session_id = fresh UUID
```

`<user_id>` accepts `[A-Za-z0-9_-]{1,128}` and is validated at every entry point. PDF bytes are uploaded to MinIO at `users/{user_id}/docs/{document_id}.pdf` before Docling runs; the object persists even if Docling fails so retries don't require re-upload.

The three stores (Qdrant / Neo4j / Postgres / MinIO) enforce user scoping independently — defence in depth. See the spec at `docs/superpowers/specs/2026-04-21-multi-tenant-docs-design.md` for the full threat model, transactional guarantees, and security matrix.

## Performance

Measured on Romanian Penal Code (2.4 MB, 125 pages, 412 Docling chunks):

| Metric | Value |
|--------|-------|
| **Hit@1 (graph_hybrid)** | **84.8%** |
| Hit@5 (hybrid) | 87.9% |
| Ingestion time | ~10 min (KG + summaries at concurrency 20) |
| Query latency | ~2.5s (4 parallel LLM calls + retrieval + reranking) |
| Ingestion cost | ~$0.42/document |
| Eval cost (33q) | ~$0.15/run |

## License

MIT
