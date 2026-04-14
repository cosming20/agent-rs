# agent-rs — System Architecture (April 2026)

## High-Level System Overview

```
                                    ┌─────────────────────────────────────────┐
                                    │           gRPC Clients                  │
                                    │     (grpcurl, UI, other services)       │
                                    └──────────────────┬──────────────────────┘
                                                       │ QueryRequest (query, session_id,
                                                       │   user_id, strategy, limit)
                                                       ▼
                              ┌─────────────────────────────────────────────────┐
                              │              gRPC Server (:1059)                 │
                              │  ┌─────────────────┐  ┌──────────────────────┐  │
                              │  │  QueryService    │  │   HealthService      │  │
                              │  │  rpc Query(...)   │  │   rpc Check(...)     │  │
                              │  └────────┬─────────┘  └──────────────────────┘  │
                              └───────────┼─────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          AgentOrchestrator.process_query()                            │
│                                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │  Step 1       │    │  Step 2       │    │  Step 3       │    │  Step 4             │  │
│  │  Context      │───▶│  Agent        │───▶│  Evidence     │───▶│  Grounding          │  │
│  │  Builder      │    │  Runner       │    │  Fuser        │    │  Gate               │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └───────┬────────────┘  │
│         │                   │                   │                    │                │
│         │                   │                   │              ┌─────┴──────┐         │
│         │                   │                   │              │  If failed  │         │
│         │                   │                   │              │  & retries  │         │
│         │                   │                   │              │  < max:     │         │
│         │                   │                   │              │  RETRY ─────┼──┐      │
│         │                   │                   │              └────────────┘  │      │
│         │                   │                   │                    │         │      │
│         │                   │                   │                    ▼         │      │
│         │                   │                   │              ┌────────────┐  │      │
│         │                   │                   │              │  Step 5     │  │      │
│         │                   │                   │              │  Result     │  │      │
│         │                   │                   │              │  Persister  │  │      │
│         │                   │                   │              └──────┬─────┘  │      │
│         │                   │                   │                    │         │      │
│         ▼                   ▼                   ▼                    ▼         │      │
│    ┌─────────┐        ┌──────────┐        ┌─────────┐        ┌──────────┐    │      │
│    │Postgres │        │ OpenAI   │        │  RRF    │        │Synthesize│    │      │
│    │history  │        │ LLM API  │        │ Fusion  │        │+Persist  │    │      │
│    │+memory  │        │(GPT-4o)  │        │ k=60    │        │          │    │      │
│    └─────────┘        └──────────┘        └─────────┘        └──────────┘    │      │
│                             │                                       │         │      │
│                             │ Agent calls tools ◄───────────────────┼─────────┘      │
│                             ▼                                       │                │
│                    ┌─────────────────┐                              │                │
│                    │  ToolFactory    │                              │                │
│                    │  (per-request)  │                              │                │
│                    └────────┬────────┘                              │                │
│                             │                                       ▼                │
│         ┌───────────────────┼──────────────────────┐        QueryResponse            │
│         │                   │                      │        (answer, citations,       │
│         ▼                   ▼                      ▼         confidence, strategy)    │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐                             │
│  │search_        │  │search_web     │  │fetch_url       │                             │
│  │knowledge      │  │search_news    │  │remember        │                             │
│  │               │  │               │  │ingest          │                             │
│  └──────┬───────┘  └───────┬───────┘  └────────┬───────┘                             │
│         │                   │                   │                                    │
└─────────┼───────────────────┼───────────────────┼────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
```

## Retrieval Engine Detail

```
search_knowledge tool
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RetrievalRouter                              │
│                                                                  │
│  1. Embed query (fastembed ONNX: dense + sparse vectors)         │
│  2. Look up strategy descriptor from registry                    │
│  3. Dispatch to search backend(s)                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Strategy Dispatch Table                        │  │
│  │                                                            │  │
│  │  SEMANTIC ─────────► vector_search()      [Qdrant dense]   │  │
│  │  KEYWORD ──────────► keyword_search()     [Qdrant BM25]    │  │
│  │  HYBRID ───────────► hybrid_search()      [parallel both]  │  │
│  │  GRAPH ────────────► graph_search()       [Neo4j Cypher]   │  │
│  │  GRAPH_COT ────────► graph_cot_search()   [Neo4j + CoT]    │  │
│  │  GRAPH_CONSTRAINED ► graph_constrained()  [Neo4j filtered] │  │
│  │  TRIPLET ──────────► triplet_search()     [Qdrant triplets]│  │
│  │  SUMMARIES ────────► summaries_search()   [Qdrant summaries│  │
│  │  TEMPORAL ─────────► temporal_search()     [Qdrant + time]  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. RRF Rank Fusion (ResultRanker)                               │
│     score += 1/(k + rank) per list, k=60                         │
│     Dedup by doc_id, sort by fused_score desc                    │
│                                                                  │
│  5. Optional CrossEncoder Reranking (BGE-reranker-v2-m3)         │
│                                                                  │
│  6. Return EvidencePack (ranked docs + scores + strategy)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Storage Layer

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Storage Traits (stores/traits.rs)                   │
│                                                                            │
│  VectorStore              GraphStore              StateStore               │
│  ──────────              ──────────              ──────────               │
│  search()                add_node()              save_message()            │
│  search_hybrid()         add_edge()              load_messages()           │
│  search_with_filter()    get_neighbors()         store_memory()            │
│  upsert()                traverse()              retrieve_memory()         │
│  create_collection()     query_nodes()           persist_run()             │
│  delete()                delete_by_filter()      begin_ingestion()         │
│  collection_exists()                             mark_complete()           │
└───────┬──────────────────────┬────────────────────────┬────────────────────┘
        │                      │                        │
        ▼                      ▼                        ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  QdrantStore   │    │   Neo4jStore      │    │      PgStore          │
│  :1050 (gRPC)  │    │   :1061 (Bolt)    │    │      :1054            │
│  :1051 (HTTP)  │    │                   │    │                       │
│                │    │                   │    │  Tables:               │
│  Collections:  │    │  Nodes:           │    │  - chat_messages       │
│  - chunks      │    │  - Entity         │    │  - long_term_memory    │
│    (dense vec) │    │  - Document       │    │  - query_runs          │
│  - chunks_bm25 │    │  - Section        │    │  - ingested_documents  │
│    (sparse)    │    │  - Table/Picture  │    │                       │
│  - triplets    │    │  Edges:           │    │                       │
│  - summaries   │    │  - RELATES_TO     │    │                       │
│                │    │  - HAS_SECTION    │    │                       │
│                │    │  - HAS_CHUNK      │    │                       │
└───────────────┘    └──────────────────┘    └──────────────────────┘
```

## Ingestion Pipeline

```
                PDF / Document
                      │
                      ▼
          ┌──────────────────────┐
          │  Docling Sidecar     │  (Python, gRPC streaming)
          │  PDF/DOCX → chunks   │
          └──────────┬───────────┘
                     │ chunked text + metadata
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                                    │
│                                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐                   │
│  │ 1. Dedup    │──▶│ 2. Chunk     │──▶│ 3. Embed    │                   │
│  │ (Postgres   │   │ (sentences/  │   │ (fastembed  │                   │
│  │  sha256)    │   │  paragraphs/ │   │  ONNX)      │                   │
│  │             │   │  hybrid)     │   │             │                   │
│  └─────────────┘   └──────────────┘   └──────┬──────┘                   │
│                                              │                           │
│                    ┌─────────────────────────┬┴──────────────────┐        │
│                    ▼                         ▼                   ▼        │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────┐  │
│  │ 4a. Qdrant Upsert    │  │ 4b. Neo4j Structure  │  │ 4c. LLM KG    │  │
│  │ chunks + chunks_bm25 │  │ doc → sections →     │  │ Extraction     │  │
│  │ (27-field payload)   │  │ chunks hierarchy     │  │ (gpt-5.4-mini) │  │
│  └──────────────────────┘  └──────────────────────┘  │ entities +     │  │
│                                                      │ relations →    │  │
│                                                      │ Neo4j nodes +  │  │
│                                                      │ edges          │  │
│                                                      └───────┬────────┘  │
│                                                              │           │
│                    ┌─────────────────────────┬────────────────┘           │
│                    ▼                         ▼                            │
│  ┌──────────────────────┐  ┌──────────────────────┐                      │
│  │ 5a. Triplet Indexing │  │ 5b. Chunk Summaries  │                      │
│  │ linearize triplets   │  │ LLM summarize each   │                      │
│  │ → embed → Qdrant     │  │ chunk → embed →      │                      │
│  │ (triplets collection)│  │ Qdrant (summaries)   │                      │
│  └──────────────────────┘  └──────────────────────┘                      │
│                                                                          │
│  ┌──────────────────────┐                                                │
│  │ 6. Postgres          │                                                │
│  │ mark_complete()      │  ← 3-retry exponential backoff                 │
│  └──────────────────────┘                                                │
│                                                                          │
│  Result: IngestionResult { doc_id, chunks_created, vectors_upserted,     │
│          graph_nodes, graph_edges, triplets, summaries, duration_ms }    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Verification & Retry Loop

```
                    ┌──────────────────────────────┐
                    │      GroundingGate            │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     GroundingVerifier          │
                    │                               │
                    │  Path A: LLM Verification     │
                    │  ┌─────────────────────────┐  │
                    │  │ OpenAI call with         │  │
                    │  │ <evidence> + <answer>    │  │
                    │  │ → JSON {is_grounded,     │  │
                    │  │   confidence, reasoning, │  │
                    │  │   ungrounded_claims}     │  │
                    │  └─────────────────────────┘  │
                    │                               │
                    │  Path B: Heuristic Fallback   │
                    │  ┌─────────────────────────┐  │
                    │  │ Token overlap ratio      │  │
                    │  │ (answer ∩ evidence)      │  │
                    │  │ with stopword filtering  │  │
                    │  └─────────────────────────┘  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  is_grounded?                  │
                    │                               │
                    │  YES ──► proceed to Step 5     │
                    │                               │
                    │  NO & retries < max ──►        │
                    │    Build retry prompt with:    │
                    │    - ungrounded claims          │
                    │    - evidence gaps              │
                    │    - "search for more info"     │
                    │    → AgentRunner.rerun()        │
                    │    → EvidenceFuser.fuse()       │
                    │    → GroundingVerifier again    │
                    │                               │
                    │  NO & retries >= max ──►       │
                    │    Return with is_grounded=false│
                    └───────────────────────────────┘
```

## Full Query Flow (End-to-End)

```
Client ──grpcurl──► :1059/QueryService/Query
                         │
                         ▼
              QueryRequest { query: "What is X?", session_id, user_id,
                            strategy: HYBRID, limit: 10 }
                         │
          ╔══════════════╧══════════════════════════════════════╗
          ║  STEP 1: ContextBuilder                             ║
          ║  ├── Postgres: load_messages(session_id, last 20)   ║
          ║  ├── Postgres: retrieve_memory(user_id)             ║
          ║  └── Build conversation_context string              ║
          ╚══════════════╤══════════════════════════════════════╝
                         │
          ╔══════════════╧══════════════════════════════════════╗
          ║  STEP 2: AgentRunner                                ║
          ║  ├── ToolFactory: build tools with WebBudgetTracker ║
          ║  │   ├── SearchKnowledgeTool(router, evidence_sink) ║
          ║  │   ├── SearchWebTool(brave_client, budget)        ║
          ║  │   ├── SearchNewsTool(brave_client, budget)       ║
          ║  │   ├── FetchUrlTool(reqwest, budget)              ║
          ║  │   ├── RememberTool(pg_store)                     ║
          ║  │   └── IngestTool(pipeline)                       ║
          ║  │                                                  ║
          ║  ├── Rig Agent: client.agent(gpt-4o)                ║
          ║  │   .preamble(system_prompt + context)             ║
          ║  │   .tools([...])                                  ║
          ║  │   .build()                                       ║
          ║  │                                                  ║
          ║  ├── agent.prompt(query).max_turns(N)               ║
          ║  │   │                                              ║
          ║  │   │  Turn 1: LLM decides to call search_knowledge║
          ║  │   │  ├── RetrievalRouter.search(HYBRID, query)   ║
          ║  │   │  │   ├── fastembed: dense + sparse vectors   ║
          ║  │   │  │   ├── tokio::join! {                      ║
          ║  │   │  │   │   Qdrant: vector_search(dense)        ║
          ║  │   │  │   │   Qdrant: keyword_search(sparse/BM25) ║
          ║  │   │  │   │ }                                     ║
          ║  │   │  │   ├── RRF fusion (k=60)                   ║
          ║  │   │  │   └── Optional: CrossEncoder rerank       ║
          ║  │   │  └── evidence_sink ← results                 ║
          ║  │   │                                              ║
          ║  │   │  Turn 2: LLM may call search_web             ║
          ║  │   │  ├── Brave API search                        ║
          ║  │   │  └── web_evidence ← results                  ║
          ║  │   │                                              ║
          ║  │   │  Turn N: LLM generates final answer          ║
          ║  │   │                                              ║
          ║  │   └── answer + accumulated evidence              ║
          ║  └── ctx.kb_evidence, ctx.web_evidence, ctx.answer  ║
          ╚══════════════╤══════════════════════════════════════╝
                         │
          ╔══════════════╧══════════════════════════════════════╗
          ║  STEP 3: EvidenceFuser                              ║
          ║  ├── Merge kb_evidence (weight 1.0)                 ║
          ║  │        + web_evidence (weight 0.8)               ║
          ║  ├── RRF rank fusion across sources                 ║
          ║  └── ctx.evidence = fused & ranked list             ║
          ╚══════════════╤══════════════════════════════════════╝
                         │
          ╔══════════════╧══════════════════════════════════════╗
          ║  STEP 4: GroundingGate                              ║
          ║  ├── GroundingVerifier.verify(answer, evidence)     ║
          ║  │   ├── LLM path: OpenAI → JSON verdict            ║
          ║  │   └── Heuristic fallback: token overlap          ║
          ║  │                                                  ║
          ║  ├── if !grounded && retries < max:                 ║
          ║  │   ├── Build retry prompt with gaps               ║
          ║  │   ├── AgentRunner.rerun() → more evidence        ║
          ║  │   ├── EvidenceFuser.fuse() again                 ║
          ║  │   └── GroundingVerifier.verify() again           ║
          ║  │                                                  ║
          ║  └── ctx.verification = { is_grounded, confidence,  ║
          ║       reasoning, ungrounded_claims }                ║
          ╚══════════════╤══════════════════════════════════════╝
                         │
          ╔══════════════╧══════════════════════════════════════╗
          ║  STEP 5: ResultPersister                            ║
          ║  ├── ResponseSynthesizer.synthesize()               ║
          ║  │   ├── Extract citations from evidence            ║
          ║  │   │   (source, snippet, score per doc)           ║
          ║  │   ├── Match [N] refs in answer text              ║
          ║  │   ├── Confidence inversion if ungrounded         ║
          ║  │   └── SynthesizedResponse                        ║
          ║  │                                                  ║
          ║  ├── Postgres: persist_run()                        ║
          ║  │   (query, answer, citations, confidence,         ║
          ║  │    strategy, latency, is_grounded)               ║
          ║  │                                                  ║
          ║  └── Postgres: save_message(assistant, answer)      ║
          ╚══════════════╤══════════════════════════════════════╝
                         │
                         ▼
              QueryResponse {
                answer: "X is ... [1][3]",
                citations: [
                  { index: 1, snippet: "...", source: "qdrant", score: 0.89 },
                  { index: 3, snippet: "...", source: "brave",  score: 0.72 }
                ],
                confidence: 0.85,
                is_grounded: true,
                strategy: "HYBRID"
              }
```

## Infrastructure Topology

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Compose                               │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Qdrant      │  │   Neo4j       │  │  PostgreSQL   │             │
│  │  :1050 gRPC   │  │  :1061 Bolt   │  │  :1054        │             │
│  │  :1051 HTTP   │  │  :1062 HTTP   │  │               │             │
│  │               │  │               │  │               │             │
│  │ Collections:  │  │ Entities +    │  │ chat_messages │             │
│  │ · chunks      │  │ Relations +   │  │ long_term_mem │             │
│  │ · chunks_bm25 │  │ Doc structure │  │ query_runs    │             │
│  │ · triplets    │  │               │  │ ingested_docs │             │
│  │ · summaries   │  │               │  │               │             │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  OTel         │  │   Jaeger      │  │     OPA       │             │
│  │  Collector    │  │   Tracing UI  │  │   Policies    │             │
│  │  :1057        │  │   :1058       │  │   :1055       │             │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Brave Search  │                                                   │
│  │ MCP :1063     │                                                   │
│  └──────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

         ┌────────────────────────────┐
         │      agent-rs binary       │
         │         :1059              │
         │                            │
         │  gRPC server (tonic)       │
         │  ├── QueryService          │
         │  └── HealthService         │
         │                            │
         │  Connects to:              │
         │  → Qdrant :1050/:1051      │
         │  → Neo4j  :1061            │
         │  → Postgres :1054          │
         │  → OpenAI API (HTTPS)      │
         │  → Brave API (HTTPS)       │
         │  → OTel Collector :1057    │
         │                            │
         │  Embeds locally:           │
         │  → fastembed (ONNX runtime)│
         └────────────────────────────┘
```

## Crate Dependency Graph

```
                  ┌──────────────┐
                  │    server     │  (tonic gRPC, entry point)
                  └──────┬───────┘
                         │ depends on
            ┌────────────┼──────────────┐
            ▼            ▼              ▼
     ┌────────────┐ ┌──────────┐ ┌──────────────┐
     │   agent     │ │  tools   │ │observability │
     │(orchestrate)│ │(rig tools│ │(tracing/otel)│
     └──────┬─────┘ └────┬─────┘ └──────────────┘
            │             │
            ▼             ▼
     ┌─────────────────────────┐
     │       retrieval          │  (router, ranker, search, ingestion)
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │        stores            │  (qdrant, neo4j, postgres + traits)
     └────────────┬────────────┘
                  │
                  ▼
     ┌─────────────────────────┐
     │         core             │  (config, errors, types, logging)
     │    (ZERO workspace deps) │
     └─────────────────────────┘
```
