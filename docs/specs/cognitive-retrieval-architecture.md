# Cognitive Retrieval Architecture — "The Thinking Brain"

> Transform agent-rs from a retrieval wrapper into an intelligent system that
> analyzes, plans, adapts, and learns from every query.

---

## Current State: What's Missing

```
CLIENT → "strategy: hybrid" → search → fuse → verify → answer
                ↑                                    ↓
           client decides                    generic retry
           no query analysis                 "try harder"
           no adaptive behavior              no gap analysis
```

**7 gaps:**

| Gap | Impact |
|-----|--------|
| Strategy is client-chosen, not system-decided | Wrong strategy → poor results |
| No query analysis before retrieval | Wastes tokens, misses decomposition |
| Retry prompt is generic ("try graph_hybrid") | Wastes retry budget |
| No evidence sufficiency check | Synthesizes from weak evidence |
| Query rewriting not implemented | Raw queries → suboptimal retrieval |
| No learning across queries | Same mistakes repeated forever |
| Tool selection is pure LLM | No cost/benefit analysis, no planning |

---

## Proposed Architecture: 4 Cognitive Layers

```
         ┌──────────────────────────────────────────────────────────────┐
         │                   LAYER 4: META-COGNITION                    │
         │  "Do I know enough? What am I uncertain about?"              │
         │                                                              │
         │  • Evidence sufficiency scoring                              │
         │  • Per-claim confidence tracking                             │
         │  • Knowledge gap detection                                   │
         │  • "I don't know" decision boundary                          │
         └──────────────────────────────┬───────────────────────────────┘
                                        │ feeds back to
         ┌──────────────────────────────▼───────────────────────────────┐
         │                   LAYER 3: ADAPTIVE EXECUTION                │
         │  "This didn't work. What should I try differently?"          │
         │                                                              │
         │  • Intelligent retry with gap analysis                       │
         │  • Strategy escalation cascade                               │
         │  • Query reformulation on failure                            │
         │  • Cross-strategy evidence correlation                       │
         └──────────────────────────────┬───────────────────────────────┘
                                        │ executes plan from
         ┌──────────────────────────────▼───────────────────────────────┐
         │                   LAYER 2: QUERY PLANNING                    │
         │  "How should I search for this information?"                 │
         │                                                              │
         │  • Query decomposition (multi-hop, parallel, sequential)     │
         │  • Strategy selection per sub-query                          │
         │  • Tool orchestration plan with budget allocation            │
         │  • Query rewriting (HyDE, expansion, reformulation)          │
         └──────────────────────────────┬───────────────────────────────┘
                                        │ informed by
         ┌──────────────────────────────▼───────────────────────────────┐
         │                   LAYER 1: QUERY UNDERSTANDING               │
         │  "What is actually being asked?"                             │
         │                                                              │
         │  • Intent classification (factual / relational / temporal /  │
         │    comparative / procedural / aggregation)                   │
         │  • Entity recognition + linking to knowledge graph           │
         │  • Complexity assessment (simple / multi-hop / composite)    │
         │  • Temporal/constraint detection                             │
         └─────────────────────────────────────────────────────────────┘
```

---

## LAYER 1: Query Understanding

### What it does

Before ANY retrieval happens, analyze the query to understand intent, complexity,
entities, and constraints. This replaces the current "client picks a strategy" model.

### Implementation: QueryAnalyzer

```
Input:  raw query string + conversation history + user memory
Output: QueryAnalysis {
            intent: QueryIntent,
            complexity: Complexity,
            entities: Vec<RecognizedEntity>,
            temporal_constraints: Option<TemporalRange>,
            sub_questions: Vec<SubQuestion>,     // if decomposed
            suggested_strategies: Vec<(RetrievalStrategy, f64)>,  // ranked
            rewritten_queries: Vec<RewrittenQuery>,
        }
```

#### 1a. Intent Classification

```rust
enum QueryIntent {
    /// "What is X?" — direct fact lookup
    Factual,
    /// "How are X and Y related?" — entity relationship traversal
    Relational,
    /// "What happened after X?" — time-ordered events
    Temporal,
    /// "Compare X and Y" — multi-entity parallel retrieval
    Comparative,
    /// "How do I do X?" — step-by-step procedural
    Procedural,
    /// "How many X have Y?" — aggregation over results
    Aggregation,
    /// "If X, what would Y?" — hypothetical reasoning
    Hypothetical,
    /// "Summarize section Z" — content summary
    Summarization,
}
```

**How:** LLM call with structured output (JSON schema). Lightweight — model sees
only the query + few-shot examples, not the full context. Can use a smaller/faster
model (gpt-4o-mini or local).

**Why this matters:** Intent directly maps to optimal strategy:

| Intent | Primary Strategy | Fallback |
|--------|-----------------|----------|
| Factual | SEMANTIC or KEYWORD | HYBRID |
| Relational | GRAPH | GRAPH_COT |
| Temporal | TEMPORAL | HYBRID + time filter |
| Comparative | parallel SEMANTIC × N | HYBRID × N |
| Procedural | HYBRID | SUMMARIES |
| Aggregation | GRAPH_CONSTRAINED | HYBRID + post-filter |
| Hypothetical | GRAPH_COT | HYBRID + reasoning |
| Summarization | SUMMARIES | SEMANTIC |

#### 1b. Entity Recognition + KG Linking

Extract entities from the query and check if they exist in our Neo4j graph.

```
Query: "What is the relationship between Acme Corp and Project Alpha?"
       ─────────────────────────── ─────────── ─────────────────────
Entities: [                                    
  Entity("Acme Corp",  type=Organization, kg_node_id=Some("node_123")),
  Entity("Project Alpha", type=Project,   kg_node_id=Some("node_456")),
]
```

**How:** 
- LLM extraction of entity mentions
- Neo4j lookup: `MATCH (n) WHERE n.name =~ '(?i).*acme corp.*' RETURN n LIMIT 3`
- If found → we can do targeted graph traversal from known nodes
- If not found → fall back to vector search

**Why this matters:** If entities are IN the graph, graph search is dramatically better.
If they're NOT, vector search is better. Currently, the system has no way to know.

#### 1c. Complexity Assessment

```rust
enum Complexity {
    /// Single fact, single search — "What is X?"
    Simple,
    /// Requires combining 2+ facts — "Where does the CEO of X live?"
    MultiHop { hops: u8 },
    /// Multiple independent sub-questions — "Compare X and Y"
    Composite { sub_queries: Vec<SubQuestion> },
    /// Requires reasoning over aggregated results
    Analytical,
}
```

**Why this matters:** Complex queries need decomposition BEFORE search, not after.
The current system sends the whole complex query as-is and hopes the LLM figures it out.

#### 1d. Query Rewriting

Transform the raw query into better search queries.

**Three rewriting strategies:**

```
Raw: "What did the board decide about the merger last quarter?"

1. HyDE (Hypothetical Document Embedding):
   → Generate a hypothetical answer: "The board voted to approve the merger 
     with XYZ Corp in Q3 2025, subject to regulatory approval..."
   → Embed THIS instead of the question (closer to actual document vectors)

2. Expansion:
   → "board decision merger Q1 2026" (keyword-optimized)
   → "board of directors merger acquisition resolution" (synonym expansion)

3. Decomposition:
   → Sub-query 1: "board decisions about merger" (HYBRID)
   → Sub-query 2: "events in Q1 2026" (TEMPORAL)
   → Merge results
```

### QueryAnalyzer Pipeline

```
raw_query ──► LLM: classify intent + extract entities + assess complexity
                │
                ├──► Neo4j: link entities to graph nodes
                │
                ├──► LLM: generate rewritten queries (HyDE + expansion)
                │
                └──► Build QueryAnalysis struct
                         │
                         ├── intent: Relational
                         ├── complexity: MultiHop { hops: 2 }
                         ├── entities: [Acme(linked), Alpha(linked)]
                         ├── suggested_strategies: [(GRAPH, 0.9), (HYBRID, 0.6)]
                         ├── rewritten_queries: [hyde_query, expanded_query]
                         └── sub_questions: [q1, q2] (if decomposed)
```

**Cost:** 1 LLM call (fast model) + 1-2 Neo4j lookups. ~200-500ms added latency.
**Value:** Dramatically better strategy selection and retrieval quality.

---

## LAYER 2: Query Planning

### What it does

Takes QueryAnalysis and creates a concrete execution plan: what to search,
in what order, with what budget, and when to stop.

### Implementation: QueryPlanner

```
Input:  QueryAnalysis + available tools + budget constraints
Output: ExecutionPlan {
            steps: Vec<PlannedStep>,
            budget: BudgetAllocation,
            stop_conditions: Vec<StopCondition>,
            fallback_chain: Vec<FallbackStrategy>,
        }
```

#### 2a. Search Plan Generation

**Simple query → single step:**
```
Query: "What is GDPR?"
Plan: [
  Step 1: SEMANTIC search("What is GDPR?", limit=10)
  Stop: if top_score > 0.85 and results >= 3
]
```

**Multi-hop query → sequential steps:**
```
Query: "Where is the headquarters of the company that made Product X?"
Plan: [
  Step 1: HYBRID search("Product X manufacturer company", limit=5)
  Step 2: Extract company_name from Step 1 results
  Step 3: KEYWORD search("{company_name} headquarters location", limit=5)
  Merge: combine results with provenance chain
]
```

**Comparative query → parallel steps:**
```
Query: "Compare GDPR and CCPA data breach notification requirements"
Plan: [
  Step 1a: HYBRID search("GDPR data breach notification", limit=8) ─┐
  Step 1b: HYBRID search("CCPA data breach notification", limit=8) ─┤ parallel
  Step 2: Merge + deduplicate                                        │
  Step 3: If coverage < threshold, GRAPH search for regulatory links │
]
```

**Relational query with linked entities → graph-first:**
```
Query: "How are Acme Corp and Project Alpha related?"
Plan: [
  Step 1: GRAPH traverse(start="node_123", end="node_456", max_depth=3)
  Step 2: If no path found, HYBRID search("Acme Corp Project Alpha relationship")
  Step 3: If Step 2 has results but Step 1 doesn't, run GRAPH_COT
]
```

#### 2b. Budget Allocation

Currently, web budgets are flat per-query. The planner should allocate budgets
based on the plan:

```rust
struct BudgetAllocation {
    /// How many retrieval steps before we must answer
    max_retrieval_rounds: u8,        // default 3, up to 5 for complex
    /// Web search budget allocation per step
    web_searches_per_step: Vec<u8>,  // e.g., [0, 0, 2, 1] — web only in later steps
    /// Time budget per step
    time_budget_per_step_ms: Vec<u64>,
    /// Total query budget
    total_time_budget_ms: u64,
    /// Evidence target: minimum docs before we attempt synthesis
    min_evidence_docs: usize,
    /// Quality target: minimum average relevance score
    min_avg_relevance: f64,
}
```

**Key insight:** Don't blow the web budget on the first search. KB first, web as fallback.

#### 2c. Strategy Cascade (Fallback Chain)

Instead of single strategy → generic retry, define an intelligent fallback chain:

```
Level 1 (fast, cheap):   SEMANTIC vector search
                         ↓ if confidence < 0.6 or results < 3
Level 2 (broader):       HYBRID (vector + keyword in parallel)
                         ↓ if still insufficient
Level 3 (deep):          GRAPH traversal from linked entities
                         ↓ if entities not in graph or no paths
Level 4 (creative):      HyDE rewrite + HYBRID retry
                         ↓ if still insufficient
Level 5 (external):      Web search (Brave API)
                         ↓ if web also fails
Level 6 (admission):     "I don't have enough information to answer this."
```

Each level only triggers if the previous level's evidence is insufficient.
The system KNOWS what "insufficient" means (see Layer 4: Meta-Cognition).

#### 2d. Stop Conditions

```rust
enum StopCondition {
    /// Enough high-quality evidence to answer
    EvidenceSufficient {
        min_docs: usize,
        min_avg_score: f64,
        min_coverage: f64,  // % of query aspects covered
    },
    /// Budget exhausted
    BudgetExhausted,
    /// Time limit reached
    TimeoutReached { max_ms: u64 },
    /// Maximum retrieval rounds reached
    MaxRoundsReached { max: u8 },
}
```

---

## LAYER 3: Adaptive Execution

### What it does

Executes the plan but adapts in real-time based on intermediate results.
This is where the "thinking" happens during retrieval.

### 3a. Intelligent Retry (replaces generic "try harder")

**Current retry prompt (BROKEN):**
```
"Your previous answer was not well-grounded. Please search again 
 with a broader strategy (try 'graph_hybrid')."
```

**Proposed: Diagnostic Retry**

When grounding fails, run a diagnostic analysis BEFORE retrying:

```
Input:  answer + evidence + VerificationResult (with ungrounded_claims)
Output: RetryDiagnosis {
            ungrounded_claims: Vec<UngroundedClaim>,
            evidence_gaps: Vec<EvidenceGap>,
            suggested_actions: Vec<RetryAction>,
        }

enum RetryAction {
    /// Search for specific missing information
    TargetedSearch {
        query: String,
        strategy: RetrievalStrategy,
        reason: String,
    },
    /// Reformulate the query approach
    ReformulateQuery {
        new_query: String,
        rationale: String,
    },
    /// Try a completely different strategy
    StrategySwitch {
        from: RetrievalStrategy,
        to: RetrievalStrategy,
        rationale: String,
    },
    /// Admit partial knowledge
    PartialAnswer {
        answerable_parts: Vec<String>,
        unanswerable_parts: Vec<String>,
    },
    /// Give up — evidence doesn't exist in our corpus
    AdmitIgnorance {
        reason: String,
    },
}
```

**Example:**

```
Query: "What was Acme Corp's revenue in 2024 and how did it compare to competitors?"

Grounding fails. Diagnostic:
  Ungrounded claims:
    1. "Revenue was $4.2B" — no evidence supporting this number
    2. "Competitors like XYZ had lower revenue" — no competitor data found

  Evidence gaps:
    1. Financial data for Acme Corp 2024 — NOT FOUND in KB
    2. Competitor revenue data — NOT FOUND in KB

  Suggested actions:
    1. TargetedSearch("Acme Corp revenue 2024 financial", KEYWORD, 
       "financial data often uses exact terms")
    2. TargetedSearch("Acme Corp competitors revenue comparison", HYBRID,
       "comparative data may be in reports")
    3. If KB fails: web search for public financial data
    4. If still nothing: PartialAnswer("I found information about Acme Corp 
       but their specific 2024 revenue and competitor comparison are not in 
       the knowledge base.")
```

### 3b. Evidence-Aware Strategy Switching

After each retrieval step, evaluate results and decide next step:

```
┌─────────────────────────────────────────────────────┐
│              Retrieval Step N complete                │
│                                                      │
│  Results: 8 docs, avg_score: 0.72                    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ EvidenceEvaluator.assess()                    │    │
│  │                                               │    │
│  │ Coverage: 60% (2 of 3 query aspects covered)  │    │
│  │ Quality: MEDIUM (scores 0.6-0.8)              │    │
│  │ Diversity: LOW (all from same document)        │    │
│  │ Missing: "competitor comparison data"          │    │
│  │                                               │    │
│  │ Verdict: INSUFFICIENT — need more evidence     │    │
│  │ Next action: search for "competitor data"      │    │
│  │              with GRAPH strategy (relational)  │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  → Execute next retrieval step with specific target  │
└─────────────────────────────────────────────────────┘
```

### 3c. Cross-Strategy Evidence Correlation

When results come from multiple strategies (e.g., HYBRID = vector + keyword + graph),
correlate them to assess agreement:

```rust
struct EvidenceCorrelation {
    /// Documents found by multiple strategies (high confidence)
    corroborated: Vec<SearchResult>,
    /// Documents found by only one strategy (lower confidence)
    uncorroborated: Vec<SearchResult>,
    /// Contradictions between sources
    contradictions: Vec<Contradiction>,
    /// Agreement score (0.0 = all disagree, 1.0 = all agree)
    agreement_score: f64,
}
```

**Why:** If vector search and graph traversal find the SAME document,
confidence should be much higher than if only one found it.

---

## LAYER 4: Meta-Cognition

### What it does

The system's awareness of its own knowledge state. "Do I know enough?
What am I uncertain about? Should I answer or admit ignorance?"

### 4a. Evidence Sufficiency Scoring

Before synthesis, evaluate if evidence is ADEQUATE to answer the query:

```rust
struct SufficiencyAssessment {
    /// Does the evidence address all aspects of the query?
    coverage: f64,          // 0.0 - 1.0
    /// Is the evidence from diverse, independent sources?
    diversity: f64,         // 0.0 - 1.0
    /// Are key claims supported by multiple documents?
    corroboration: f64,     // 0.0 - 1.0
    /// How relevant is the best evidence?
    top_relevance: f64,     // 0.0 - 1.0
    /// Combined sufficiency score
    overall: f64,           // weighted combination
    /// Specific gaps identified
    gaps: Vec<String>,
    /// Verdict
    verdict: SufficiencyVerdict,
}

enum SufficiencyVerdict {
    /// Evidence is strong — proceed to synthesis
    Sufficient,
    /// Evidence is partial — answer with caveats
    Partial { answerable: Vec<String>, gaps: Vec<String> },
    /// Evidence is insufficient — try more retrieval
    Insufficient { suggested_actions: Vec<RetryAction> },
    /// No relevant evidence — admit ignorance
    NoEvidence,
}
```

**Implementation:** LLM call that sees the query + retrieved evidence summaries.
Structured output. Fast model is fine.

**This replaces the current binary grounding check with a richer assessment:**

```
BEFORE (current):
  is_grounded: true/false
  confidence: 0.73

AFTER (proposed):
  coverage: 0.80  (covers 4 of 5 query aspects)
  diversity: 0.40  (all evidence from 2 documents — weak)
  corroboration: 0.60  (main claim supported by 2 sources)
  top_relevance: 0.91  (best match is very relevant)
  overall: 0.68
  gaps: ["No competitor comparison data found"]
  verdict: Partial {
    answerable: ["Acme Corp's 2024 revenue", "business segments"],
    gaps: ["Competitor comparison", "Year-over-year growth"]
  }
```

### 4b. Per-Claim Confidence Tracking

Instead of one confidence score for the whole answer, track confidence per claim:

```
Answer: "Acme Corp reported revenue of $4.2B in 2024 [1]. This represents 
         a 12% increase from 2023 [2]. Their main competitor, XYZ Corp, 
         reported $3.8B in the same period."

Claim analysis:
  ✅ "revenue of $4.2B in 2024" — confidence: 0.95 (directly from evidence [1])
  ✅ "12% increase from 2023" — confidence: 0.88 (computed from evidence [1][2])
  ⚠️ "competitor XYZ Corp $3.8B" — confidence: 0.30 (NO evidence found)

Per-claim output:
  claims: [
    Claim { text: "revenue $4.2B", confidence: 0.95, sources: [1], grounded: true },
    Claim { text: "12% increase",  confidence: 0.88, sources: [1,2], grounded: true },
    Claim { text: "XYZ Corp $3.8B", confidence: 0.30, sources: [], grounded: false },
  ]
```

**Why:** The client can display per-claim confidence visually.
Users can see exactly which parts to trust and which to verify.

### 4c. Structured Uncertainty

```rust
enum UncertaintyType {
    /// Evidence exists but is ambiguous/contradictory
    EvidenceConflict { sources: Vec<Citation>, conflict: String },
    /// Evidence is old / potentially outdated
    Staleness { latest_evidence_date: chrono::NaiveDate, query_implies_recent: bool },
    /// Evidence partially addresses the question
    PartialCoverage { covered: Vec<String>, missing: Vec<String> },
    /// No evidence found at all
    NoEvidence { searched_strategies: Vec<RetrievalStrategy> },
    /// Evidence found but low relevance scores
    LowRelevance { best_score: f64 },
}
```

### 4d. "I Don't Know" Decision Boundary

**Current behavior:** Always answers, even with zero evidence.
The LLM hallucinates when evidence is insufficient.

**Proposed:** Explicit decision boundary:

```
if sufficiency.verdict == NoEvidence {
    return "I don't have information about this topic in my knowledge base. 
            Would you like me to search the web for this?"
}

if sufficiency.verdict == Partial { gaps, .. } {
    return "Based on available evidence, [answer the covered parts]. 
            However, I couldn't find information about [gaps]. 
            My confidence in this partial answer is [X]%."
}
```

**This is the most important cognitive feature:** knowing what you don't know.

---

## LAYER 5 (BONUS): Learning & Adaptation

### 5a. Strategy Performance Tracking

After each query, log:

```sql
INSERT INTO strategy_performance (
    query_hash,
    query_intent,        -- from QueryAnalyzer
    strategy_used,
    evidence_count,
    avg_relevance_score,
    is_grounded,
    confidence,
    latency_ms,
    required_retry,
    created_at
) VALUES (...);
```

Over time, build a model:
- "For RELATIONAL queries, GRAPH strategy succeeds 87% of the time"
- "For FACTUAL queries, SEMANTIC is fastest and succeeds 92%"
- "HYBRID is a safe default but adds 300ms of latency"

### 5b. Adaptive Strategy Weights

Use historical performance to adjust strategy selection:

```rust
struct StrategyPerformance {
    strategy: RetrievalStrategy,
    success_rate: f64,      // % of queries where grounding passed
    avg_confidence: f64,    // average confidence when this strategy is used
    avg_latency_ms: u64,    // cost
    sample_count: usize,    // how many queries contributed to this
}

// QueryPlanner uses this to rank strategies:
fn select_strategy(intent: QueryIntent, history: &[StrategyPerformance]) -> Vec<(Strategy, f64)> {
    // Combine intent-based heuristic with historical performance
    // Thompson sampling or similar exploration/exploitation strategy
}
```

### 5c. User-Specific Adaptation

Track per-user patterns:

```
User "alice" tends to ask RELATIONAL questions → bias toward GRAPH
User "bob" asks lots of TEMPORAL questions → bias toward TEMPORAL
User "carol" always needs broad context → increase evidence limit
```

---

## Revised Pipeline Flow

```
Query arrives
     │
     ▼
╔════════════════════════════════════════════════════════════════╗
║  LAYER 1: Query Understanding                                  ║
║                                                                ║
║  QueryAnalyzer                                                 ║
║  ├── LLM: classify intent + extract entities + assess complexity║
║  ├── Neo4j: link entities to graph nodes                       ║
║  ├── LLM: generate HyDE + expanded queries                    ║
║  └── Output: QueryAnalysis                                     ║
║       intent: Comparative                                      ║
║       complexity: Composite(2 sub-queries)                     ║
║       entities: [Acme(linked), XYZ(not found)]                 ║
║       strategies: [(HYBRID, 0.8), (GRAPH, 0.7)]               ║
║       rewrites: [hyde_query, expanded_query]                   ║
╚════════════════════════╤═══════════════════════════════════════╝
                         │
                         ▼
╔════════════════════════════════════════════════════════════════╗
║  LAYER 2: Query Planning                                       ║
║                                                                ║
║  QueryPlanner                                                  ║
║  ├── Decompose into sub-queries (if composite)                 ║
║  ├── Assign strategy per sub-query                             ║
║  ├── Allocate budget (KB first, web as fallback)               ║
║  ├── Define stop conditions                                    ║
║  └── Output: ExecutionPlan                                     ║
║       steps: [                                                 ║
║         Step 1a: HYBRID("Acme Corp revenue 2024") ──┐ parallel ║
║         Step 1b: HYBRID("XYZ Corp revenue 2024")  ──┘          ║
║         Step 2:  GRAPH(from node_123, depth=2) if Step 1 weak  ║
║         Step 3:  Web search if KB insufficient                 ║
║       ]                                                        ║
║       stop: EvidenceSufficient(min_docs=5, coverage=0.8)       ║
╚════════════════════════╤═══════════════════════════════════════╝
                         │
                         ▼
╔════════════════════════════════════════════════════════════════╗
║  LAYER 3: Adaptive Execution                                   ║
║                                                                ║
║  Execute Step 1a + 1b in parallel (tokio::join!)               ║
║  │                                                             ║
║  ├── EvidenceEvaluator: assess intermediate results            ║
║  │   coverage: 0.65, diversity: 0.50, gaps: ["competitor data"]║
║  │   verdict: INSUFFICIENT                                     ║
║  │                                                             ║
║  ├── Execute Step 2: GRAPH traversal from Acme node            ║
║  │   Found path: Acme → COMPETES_WITH → XYZ                   ║
║  │   New evidence from graph context                           ║
║  │                                                             ║
║  ├── EvidenceEvaluator: re-assess                              ║
║  │   coverage: 0.85, diversity: 0.70                           ║
║  │   verdict: SUFFICIENT                                       ║
║  │                                                             ║
║  └── Skip Step 3 (web) — KB evidence is sufficient             ║
╚════════════════════════╤═══════════════════════════════════════╝
                         │
                         ▼
╔════════════════════════════════════════════════════════════════╗
║  LAYER 4: Meta-Cognition                                       ║
║                                                                ║
║  SufficiencyAssessor                                           ║
║  ├── Coverage: 0.85 — good                                     ║
║  ├── Diversity: 0.70 — acceptable (3 unique sources)           ║
║  ├── Corroboration: 0.80 — key claims multi-sourced            ║
║  ├── Verdict: SUFFICIENT                                       ║
║  │                                                             ║
║  Synthesize with per-claim confidence:                         ║
║  ├── "Acme revenue $4.2B" → 0.95 (direct evidence)            ║
║  ├── "12% YoY growth" → 0.88 (computed from evidence)         ║
║  └── "XYZ revenue $3.8B" → 0.82 (graph-sourced)               ║
║                                                                ║
║  GroundingVerifier: final check                                ║
║  ├── is_grounded: true                                         ║
║  └── confidence: 0.88                                          ║
╚════════════════════════╤═══════════════════════════════════════╝
                         │
                         ▼
              QueryResponse {
                answer: "...",
                citations: [...],
                confidence: 0.88,
                claim_confidences: [...],     // NEW
                coverage: 0.85,              // NEW
                uncertainty: None,           // NEW
                retrieval_trace: [...],      // NEW: what was searched and why
              }
```

---

## Implementation Priority

### Phase A: Highest Impact, Lowest Effort

| # | Optimization | Effort | Impact | Why |
|---|-------------|--------|--------|-----|
| A1 | **Query rewriting (HyDE)** | LOW | HIGH | Already a placeholder in router.rs. Generate hypothetical answer, embed that instead. Dramatic improvement for semantic search. |
| A2 | **Intelligent retry prompts** | LOW | HIGH | VerificationResult.ungrounded_claims already exists but is IGNORED. Wire it into the retry prompt. |
| A3 | **Evidence sufficiency check** | MEDIUM | HIGH | Add EvidenceEvaluator between fuse and synthesis. Prevents synthesizing from weak evidence. |

### Phase B: Core Cognitive Layer

| # | Optimization | Effort | Impact | Why |
|---|-------------|--------|--------|-----|
| B1 | **QueryAnalyzer (intent + entities)** | MEDIUM | VERY HIGH | Enables automatic strategy selection. Single LLM call with structured output. |
| B2 | **Strategy cascade (fallback chain)** | MEDIUM | HIGH | Replaces "client picks strategy" with intelligent escalation. |
| B3 | **Entity → KG linking** | LOW | HIGH | Neo4j lookup before search. If entities exist in graph, use GRAPH strategy. |

### Phase C: Full Planning System

| # | Optimization | Effort | Impact | Why |
|---|-------------|--------|--------|-----|
| C1 | **Query decomposition** | HIGH | VERY HIGH | Multi-hop and comparative queries. Biggest improvement for complex questions. |
| C2 | **Budget-aware planning** | MEDIUM | MEDIUM | Allocate web budget intelligently. KB first, web as fallback. |
| C3 | **Per-claim confidence** | MEDIUM | HIGH | Transparency for users. Know exactly what to trust. |

### Phase D: Learning & Adaptation

| # | Optimization | Effort | Impact | Why |
|---|-------------|--------|--------|-----|
| D1 | **Strategy performance tracking** | LOW | MEDIUM | Postgres table + INSERT after each query. Data for future adaptation. |
| D2 | **Adaptive strategy selection** | HIGH | HIGH | Use historical data to adjust strategy weights. Requires enough data first. |
| D3 | **User-specific adaptation** | HIGH | MEDIUM | Per-user patterns. Requires user interaction history. |

---

## Recommended Implementation Order

```
Phase A (do first — immediate wins, low risk):
  A1. HyDE query rewriting in router.rs
  A2. Wire ungrounded_claims into retry prompt
  A3. EvidenceEvaluator before synthesis

Phase B (core intelligence — biggest architectural change):
  B3. Entity → KG linking (quick Neo4j lookup)
  B1. QueryAnalyzer (LLM classification + entity extraction)
  B2. Strategy cascade with fallback chain

Phase C (full planning — complex queries):
  C1. Query decomposition (multi-hop, parallel sub-queries)
  C3. Per-claim confidence tracking
  C2. Budget-aware planning

Phase D (learning — long-term improvement):
  D1. Strategy performance tracking (start collecting data ASAP)
  D2. Adaptive strategy selection (after enough data)
  D3. User-specific adaptation
```

---

## New Crate Structure Changes

```
crates/
  retrieval/src/
    rewriter.rs        ← NEW: HyDE + expansion + reformulation
  agent/src/
    analyzer.rs        ← NEW: QueryAnalyzer (intent, entities, complexity)
    planner.rs         ← NEW: QueryPlanner (execution plan, budget, cascade)
    evaluator.rs       ← NEW: EvidenceEvaluator (sufficiency, coverage, gaps)
    metacognition.rs   ← NEW: Per-claim confidence, uncertainty tracking
    steps/
      grounding_gate.rs ← MODIFIED: intelligent retry with gap analysis
      evidence_fuser.rs ← MODIFIED: correlation analysis, diversity scoring
  stores/src/
    postgres/
      analytics.rs     ← NEW: strategy_performance table + queries
```

---

## Key Architectural Decisions

### 1. Where does the "thinking" happen?

**Hybrid: LLM for understanding + deterministic code for planning.**

- LLM: intent classification, entity extraction, HyDE generation, sufficiency assessment
- Code: strategy mapping, cascade logic, budget allocation, stop conditions
- LLM: retry diagnosis (what's missing and why)
- Code: strategy switching, fallback chain execution

**NOT** "give the LLM a bigger system prompt and hope it figures it out."
Structured code for structured decisions. LLM for understanding and generation.

### 2. Latency budget

```
Current pipeline:    ~2-5 seconds
Added by Layer 1:    ~200-500ms (1 LLM call + Neo4j lookups)
Added by Layer 2:    ~50ms (pure code, no LLM)
Added by Layer 3:    ~0ms baseline (runs only on retry/escalation)
Added by Layer 4:    ~200-300ms (1 LLM call for sufficiency)

Total added: ~450-800ms for much better results
```

Use a fast model (gpt-4o-mini, or even a local classifier) for Layer 1
to minimize latency. The planning and cascade layers are pure Rust — near zero cost.

### 3. When to use LLM vs. code

| Task | LLM or Code? | Why |
|------|-------------|-----|
| Intent classification | LLM (structured output) | Natural language understanding |
| Entity extraction | LLM (structured output) | NER from free text |
| Strategy selection | Code (lookup table) | Deterministic, fast, testable |
| Query rewriting (HyDE) | LLM (generation) | Needs creative text generation |
| Budget allocation | Code (rules engine) | Deterministic, configurable |
| Evidence sufficiency | LLM (structured output) | Needs understanding of coverage |
| Per-claim confidence | LLM (structured output) | Needs claim extraction + assessment |
| Retry diagnosis | LLM (structured output) | Needs gap analysis |
| Cascade logic | Code (state machine) | Deterministic, no ambiguity |
| Performance tracking | Code (SQL) | Data persistence |
