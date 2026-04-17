//! Entity → Knowledge Graph linker.
//!
//! Before retrieval dispatch, check whether entities mentioned in the query
//! already exist as nodes in Neo4j.  When linked entities are found, the
//! retrieval router can bias toward graph-based strategies (starting traversal
//! from known nodes) rather than relying purely on vector similarity.
//!
//! This implements spec item **B3** from the Cognitive Retrieval Architecture.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::RetrievalError;
use agent_rs_stores::traits::GraphStore;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An entity recognized in the query, possibly linked to a Neo4j node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedEntity {
    /// The entity mention as it appeared in the query.
    pub mention: String,
    /// Neo4j node ID if a matching node was found.
    pub node_id: Option<String>,
    /// Node name from the graph (may differ from the mention due to normalization).
    pub node_name: Option<String>,
    /// Whether this entity was successfully linked to the KG.
    pub is_linked: bool,
}

/// Result of entity linking for a query.
#[derive(Debug, Clone)]
pub struct EntityLinkResult {
    /// All entities extracted from the query.
    pub entities: Vec<LinkedEntity>,
    /// How many entities were successfully linked to KG nodes.
    pub linked_count: usize,
    /// How many entities were extracted but NOT found in the KG.
    pub unlinked_count: usize,
    /// Whether graph-based strategies should be preferred based on linking results.
    /// True when >= 1 entity is linked.
    pub prefer_graph: bool,
}

/// Configuration for the entity linker.
#[derive(Debug, Clone)]
pub struct EntityLinkerConfig {
    /// Maximum entities to extract per query (default 5).
    pub max_entities: usize,
    /// OpenAI API key for entity extraction.
    pub api_key: String,
    /// Model to use (default gpt-5.4-mini-2026-03-17).
    pub model: String,
}

impl Default for EntityLinkerConfig {
    fn default() -> Self {
        Self {
            max_entities: 5,
            api_key: String::new(),
            model: "gpt-5.4-mini-2026-03-17".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// EntityLinker
// ---------------------------------------------------------------------------

/// Links entity mentions in a query to Neo4j Knowledge Graph nodes.
///
/// Two-step process:
/// 1. **Extract** entities from the query via LLM (structured JSON output).
/// 2. **Link** each entity by searching Neo4j for matching nodes.
#[derive(Clone)]
pub struct EntityLinker {
    graph_store: Arc<dyn GraphStore + Send + Sync>,
    client: Arc<reqwest::Client>,
    api_key: String,
    model: String,
    max_entities: usize,
}

impl EntityLinker {
    pub fn new(
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        config: EntityLinkerConfig,
        client: Arc<reqwest::Client>,
    ) -> Self {
        Self {
            graph_store,
            client,
            api_key: config.api_key,
            model: config.model,
            max_entities: config.max_entities,
        }
    }

    /// Extract entities from the query and attempt to link them to KG nodes.
    ///
    /// On any failure (LLM error, Neo4j unavailable), returns an empty result
    /// with `prefer_graph = false` — never blocks the retrieval pipeline.
    #[instrument(skip(self), fields(query_len = query.len()))]
    pub async fn link(&self, query: &str) -> EntityLinkResult {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return EntityLinkResult::empty();
        }

        // Step 1: extract entity mentions via LLM.
        let mentions = match self.extract_entities(trimmed).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "entity extraction failed — skipping linking");
                return EntityLinkResult::empty();
            }
        };

        if mentions.is_empty() {
            tracing::debug!("no entities extracted from query");
            return EntityLinkResult::empty();
        }

        // Step 2: link each mention to Neo4j.
        let mut entities = Vec::with_capacity(mentions.len());
        for mention in mentions.into_iter().take(self.max_entities) {
            let linked = self.try_link_entity(&mention).await;
            entities.push(linked);
        }

        let linked_count = entities.iter().filter(|e| e.is_linked).count();
        let unlinked_count = entities.len() - linked_count;

        tracing::info!(
            total = entities.len(),
            linked = linked_count,
            unlinked = unlinked_count,
            "entity linking complete"
        );

        EntityLinkResult {
            prefer_graph: linked_count > 0,
            linked_count,
            unlinked_count,
            entities,
        }
    }

    /// Extract entity mentions from the query via LLM.
    async fn extract_entities(&self, query: &str) -> Result<Vec<String>, RetrievalError> {
        let body = serde_json::json!({
            "model": &self.model,
            "messages": [
                { "role": "system", "content": ENTITY_EXTRACT_PROMPT },
                { "role": "user", "content": query },
            ],
            "temperature": 0.0_f32,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "entities",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": { "type": "string" },
                                "maxItems": self.max_entities
                            }
                        },
                        "required": ["entities"]
                    }
                }
            }
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| RetrievalError::Ingestion {
                reason: format!("entity extraction API call: {e}"),
            })?;

        let bytes = resp.bytes().await.map_err(|e| RetrievalError::Ingestion {
            reason: format!("entity extraction read body: {e}"),
        })?;

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
        #[derive(Deserialize)]
        struct Entities {
            entities: Vec<String>,
        }

        let chat: ChatResp =
            serde_json::from_slice(&bytes).map_err(|e| RetrievalError::Ingestion {
                reason: format!("entity extraction parse envelope: {e}"),
            })?;

        let content = chat
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        let parsed: Entities =
            serde_json::from_str(&content).map_err(|e| RetrievalError::Ingestion {
                reason: format!("entity extraction parse JSON: {e}"),
            })?;

        Ok(parsed
            .entities
            .into_iter()
            .filter(|e| !e.trim().is_empty())
            .collect())
    }

    /// Try to link a single entity mention to a Neo4j node.
    async fn try_link_entity(&self, mention: &str) -> LinkedEntity {
        // Case-insensitive substring search in Neo4j.
        // Uses the query_raw method (added by W2) with parameterised Cypher.
        let cypher = "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($mention) \
                      RETURN n.id AS id, n.name AS name LIMIT 1";

        let mut params = HashMap::new();
        params.insert("mention".to_owned(), mention.to_owned());

        match self
            .graph_store
            .query_raw(cypher, params, &["id", "name"])
            .await
        {
            Ok(rows) if !rows.is_empty() => {
                let row = &rows[0];
                let node_id = row.get("id").and_then(|v| v.as_str()).map(|s| s.to_owned());
                let node_name = row
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());

                tracing::debug!(
                    mention,
                    node_id = ?node_id,
                    node_name = ?node_name,
                    "entity linked to KG node"
                );

                LinkedEntity {
                    mention: mention.to_owned(),
                    node_id,
                    node_name,
                    is_linked: true,
                }
            }
            Ok(_) => {
                tracing::debug!(mention, "entity not found in KG");
                LinkedEntity {
                    mention: mention.to_owned(),
                    node_id: None,
                    node_name: None,
                    is_linked: false,
                }
            }
            Err(e) => {
                tracing::warn!(mention, error = %e, "KG lookup failed for entity");
                LinkedEntity {
                    mention: mention.to_owned(),
                    node_id: None,
                    node_name: None,
                    is_linked: false,
                }
            }
        }
    }
}

impl EntityLinkResult {
    fn empty() -> Self {
        Self {
            entities: Vec::new(),
            linked_count: 0,
            unlinked_count: 0,
            prefer_graph: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Entity extraction prompt
// ---------------------------------------------------------------------------

const ENTITY_EXTRACT_PROMPT: &str = "\
Extract the key named entities from the user's query. Return ONLY concrete \
entities (people, organizations, legal articles, concepts, places, events) \
— NOT generic terms like \"document\", \"section\", or \"information\".

Rules:
- For legal queries: extract article numbers (\"articolul 188\"), legal \
  concepts (\"omor\", \"furt\", \"tâlhărie\"), and institutions.
- For technical queries: extract product names, technologies, standards.
- Return an empty array if the query has no specific named entities.
- Maximum 5 entities.
- Use the CANONICAL form: \"furt\" not \"furtul\", \"omor\" not \"omorul\".
- Match the language of the source text (Romanian entities stay Romanian).
";
