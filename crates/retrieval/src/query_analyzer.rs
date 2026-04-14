//! Query understanding and intent-aware augmentation.
//!
//! Implements spec item **B1** from the Cognitive Retrieval Architecture:
//! before retrieval dispatch, analyse the query to understand its intent
//! and generate **augmented search queries** that target the specific
//! retrieval need.
//!
//! This module fixes three retrieval gaps:
//!
//! - **Conceptual** (50% Hit@1): "What does X mean?" → generate synonym
//!   expansions and definition-oriented reformulations.
//! - **Negation** (67% Hit@1): "What is NOT X?" → split into a positive
//!   definition query + an exceptions/exclusions query.
//! - **Disambiguation** (50% Hit@1): ambiguous references → generate 2-3
//!   concrete interpretation variants as separate search queries.
//!
//! The module is **domain-agnostic** — the LLM prompt works on any corpus
//! (legal, technical, medical, financial) without domain-specific rules.

use std::sync::Arc;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MAX_RETRIES: u32 = 2;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Classified query intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    /// Direct fact lookup: "What is X?", "When did Y happen?"
    Factual,
    /// Abstract concept understanding: "What does X mean?", "Explain Y"
    Conceptual,
    /// Negation / exclusion: "What is NOT X?", "When does Y not apply?"
    Negation,
    /// Compare two or more things: "Difference between X and Y"
    Comparative,
    /// Ambiguous query needing multiple interpretations
    Disambiguation,
    /// Time-related: "What happened after X?", "When was Y introduced?"
    Temporal,
    /// Relational: "How are X and Y related?"
    Relational,
    /// Summary / overview: "Summarize section X"
    Summary,
    /// Procedural / how-to: "How to do X?"
    Procedural,
}

/// Result of query analysis.
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// The original query text.
    pub original: String,
    /// Classified intent.
    pub intent: QueryIntent,
    /// Augmented search queries generated based on intent.
    ///
    /// Always includes the original query. For conceptual/negation/disambiguation
    /// intents, additional queries are generated to target the specific retrieval
    /// need. Factual queries pass through with just the original.
    pub augmented_queries: Vec<String>,
    /// LLM reasoning for the classification and augmentation.
    pub reasoning: String,
}

/// Configuration for the query analyzer.
#[derive(Debug, Clone)]
pub struct QueryAnalyzerConfig {
    pub api_key: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for QueryAnalyzerConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "gpt-5.4-mini-2026-03-17".to_owned(),
            temperature: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryAnalyzer
// ---------------------------------------------------------------------------

/// Analyses query intent and generates augmented search queries.
///
/// Single LLM call that does BOTH intent classification AND query
/// augmentation, keeping latency to one round-trip (~200-400ms).
#[derive(Clone)]
pub struct QueryAnalyzer {
    client: reqwest::Client,
    api_key: Arc<String>,
    model: Arc<String>,
    temperature: f32,
}

impl QueryAnalyzer {
    pub fn new(config: QueryAnalyzerConfig) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            api_key: Arc::new(config.api_key),
            model: Arc::new(config.model),
            temperature: config.temperature,
        }
    }

    /// Analyse the query and produce augmented search queries.
    ///
    /// On any failure, returns the original query as a factual pass-through.
    /// Never blocks the retrieval pipeline.
    #[instrument(skip(self), fields(query_len = query.len()))]
    pub async fn analyze(&self, query: &str) -> QueryAnalysis {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return QueryAnalysis::passthrough("");
        }

        let start = std::time::Instant::now();

        match self.analyze_inner(trimmed).await {
            Ok(analysis) => {
                tracing::info!(
                    intent = ?analysis.intent,
                    augmented_count = analysis.augmented_queries.len(),
                    latency_ms = start.elapsed().as_millis(),
                    "query analysis complete"
                );
                analysis
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    latency_ms = start.elapsed().as_millis(),
                    "query analysis failed — passing through original"
                );
                QueryAnalysis::passthrough(trimmed)
            }
        }
    }

    async fn analyze_inner(&self, query: &str) -> Result<QueryAnalysis, RetrievalError> {
        let schema = serde_json::json!({
            "name": "query_analysis",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "factual", "conceptual", "negation", "comparative",
                            "disambiguation", "temporal", "relational", "summary",
                            "procedural"
                        ]
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this intent was chosen"
                    },
                    "augmented_queries": {
                        "type": "array",
                        "description": "Augmented search queries including the original",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 4
                    }
                },
                "required": ["intent", "reasoning", "augmented_queries"]
            }
        });

        let body = serde_json::json!({
            "model": self.model.as_str(),
            "messages": [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": query }
            ],
            "temperature": self.temperature,
            "response_format": { "type": "json_schema", "json_schema": schema }
        });

        let body_bytes = serde_json::to_vec(&body).map_err(|e| RetrievalError::Ingestion {
            reason: format!("serialize query analysis request: {e}"),
        })?;

        let response_bytes = self.call_api(body_bytes).await?;

        #[derive(Deserialize)]
        struct ChatResp {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Msg,
        }
        #[derive(Deserialize)]
        struct Msg {
            content: Option<String>,
        }

        let chat: ChatResp =
            serde_json::from_slice(&response_bytes).map_err(|e| RetrievalError::Ingestion {
                reason: format!("parse chat envelope: {e}"),
            })?;

        let content = chat
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        #[derive(Deserialize)]
        struct LlmResponse {
            intent: String,
            reasoning: String,
            augmented_queries: Vec<String>,
        }

        let llm: LlmResponse =
            serde_json::from_str(&content).map_err(|e| RetrievalError::Ingestion {
                reason: format!("parse analysis JSON: {e}"),
            })?;

        let intent = parse_intent(&llm.intent);

        // Ensure original query is always present.
        let mut queries: Vec<String> = llm
            .augmented_queries
            .into_iter()
            .filter(|q| !q.trim().is_empty())
            .collect();
        if !queries.iter().any(|q| q.trim() == query.trim()) {
            queries.insert(0, query.to_owned());
        }

        Ok(QueryAnalysis {
            original: query.to_owned(),
            intent,
            augmented_queries: queries,
            reasoning: llm.reasoning,
        })
    }

    async fn call_api(&self, body_bytes: Vec<u8>) -> Result<Vec<u8>, RetrievalError> {
        let client = self.client.clone();
        let api_key = Arc::clone(&self.api_key);
        let body_arc = Arc::new(body_bytes);

        let op = move || {
            let client = client.clone();
            let api_key = Arc::clone(&api_key);
            let body = Arc::clone(&body_arc);
            async move {
                let resp = client
                    .post(CHAT_COMPLETIONS_URL)
                    .header("Authorization", format!("Bearer {}", api_key.as_str()))
                    .header("Content-Type", "application/json")
                    .body((*body).clone())
                    .send()
                    .await
                    .map_err(|e| RetrievalError::Ingestion {
                        reason: format!("query analyzer HTTP: {e}"),
                    })?;

                let status = resp.status();
                let bytes = resp.bytes().await.map_err(|e| RetrievalError::Ingestion {
                    reason: format!("query analyzer read body: {e}"),
                })?;

                if status.is_success() {
                    Ok(bytes.to_vec())
                } else if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    Err(RetrievalError::Ingestion {
                        reason: format!("retryable query analyzer error (HTTP {status})"),
                    })
                } else {
                    Err(RetrievalError::Ingestion {
                        reason: format!("query analyzer error (HTTP {status})"),
                    })
                }
            }
        };

        op.retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_millis(500))
                .with_max_delay(Duration::from_secs(10))
                .with_max_times(DEFAULT_MAX_RETRIES as usize),
        )
        .when(|e: &RetrievalError| matches!(e, RetrievalError::Ingestion { reason } if reason.contains("retryable")))
        .await
    }
}

impl QueryAnalysis {
    fn passthrough(query: &str) -> Self {
        Self {
            original: query.to_owned(),
            intent: QueryIntent::Factual,
            augmented_queries: vec![query.to_owned()],
            reasoning: "pass-through (analysis skipped or failed)".to_owned(),
        }
    }
}

fn parse_intent(s: &str) -> QueryIntent {
    match s.to_lowercase().trim() {
        "factual" => QueryIntent::Factual,
        "conceptual" => QueryIntent::Conceptual,
        "negation" => QueryIntent::Negation,
        "comparative" => QueryIntent::Comparative,
        "disambiguation" => QueryIntent::Disambiguation,
        "temporal" => QueryIntent::Temporal,
        "relational" => QueryIntent::Relational,
        "summary" => QueryIntent::Summary,
        "procedural" => QueryIntent::Procedural,
        _ => QueryIntent::Factual,
    }
}

// ---------------------------------------------------------------------------
// System prompt — domain-agnostic query understanding
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are a query understanding engine for a retrieval system. Your job is to \
classify the user's query intent and generate AUGMENTED search queries that \
improve retrieval quality. You work on ANY domain (legal, technical, medical, \
financial, academic).

## Intent classification

Classify the query into one of these intents:
- **factual**: Direct fact lookup (\"What is X?\", \"When did Y happen?\")
- **conceptual**: Abstract concept understanding (\"What does X mean?\", \
  \"Explain the concept of Y\") — the user wants to UNDERSTAND something, \
  not just find a specific fact
- **negation**: Query about what is NOT something, exceptions, exclusions \
  (\"What does NOT qualify as X?\", \"When does Y NOT apply?\")
- **comparative**: Compare multiple things (\"Difference between X and Y\")
- **disambiguation**: Query has ambiguous references that could match \
  multiple distinct things (\"What about the article?\", \"The two types\")
- **temporal**: Time-related (\"What happened after X?\", \"Changes since Y\")
- **relational**: How things relate to each other
- **summary**: Overview or summary request
- **procedural**: How-to or process questions

## Augmented query generation rules

Based on the intent, generate 1-4 search queries (ALWAYS include the original):

### factual / temporal / summary / procedural / relational
→ Return ONLY the original query. No augmentation needed — these work \
  well with direct embedding.

### conceptual
→ Generate 2-3 augmented queries:
  1. The original query
  2. A reformulation using SYNONYMS and related terms \
     (e.g., \"continuous offense\" → also search \"repeated criminal acts\", \
     \"ongoing infraction\", \"persistent violation\")
  3. A definition-oriented query: \"definition of X\" / \"X is defined as\"

### negation
→ Generate 3 augmented queries:
  1. The original query
  2. The POSITIVE definition query (remove the negation): \
     \"What does NOT constitute X?\" → \"definition of X\" / \"what constitutes X\"
  3. An exceptions/exclusions query: \"exceptions to X\", \"cases excluded from X\", \
     \"when X does not apply\"

### comparative
→ Generate 2-3 augmented queries:
  1. The original query
  2. One query per entity being compared: \"X characteristics\" + \"Y characteristics\"

### disambiguation
→ Generate 2-3 augmented queries representing DIFFERENT possible interpretations:
  1. The original query
  2-3. Concrete interpretations: if \"the article\" could refer to article 188 \
  or article 345, generate a query for each plausible referent.

## Language rules
- Generate augmented queries in the SAME LANGUAGE as the original query.
- If the query is in Romanian, augmented queries must be in Romanian.
- If mixed language, follow the dominant language.

## Output rules
- ALWAYS include the original query as the first augmented query.
- Maximum 4 augmented queries total.
- Each augmented query should be a SEARCH query (not a question to the user).
- Keep augmented queries concise (under 20 words each).
";
