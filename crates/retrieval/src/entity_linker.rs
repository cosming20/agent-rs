//! Entity → Knowledge Graph linker.
//!
//! Before retrieval dispatch, check whether entities mentioned in the query
//! already exist as nodes in Neo4j.  When linked entities are found, the
//! retrieval router can bias toward graph-based strategies (starting traversal
//! from known nodes) rather than relying purely on vector similarity.
//!
//! This implements spec item **B3** from the Cognitive Retrieval Architecture.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::RetrievalError;
use agent_rs_core::{UserId, config::DEFAULT_LLM_MODEL};
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
    /// Model to use. Defaults to
    /// [`agent_rs_core::config::DEFAULT_LLM_MODEL`] when constructed via
    /// [`EntityLinkerConfig::default`].
    pub model: String,
}

impl Default for EntityLinkerConfig {
    fn default() -> Self {
        Self {
            max_entities: 5,
            api_key: String::new(),
            model: DEFAULT_LLM_MODEL.to_owned(),
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
    /// Results are scoped to `user_id` — the underlying store enforces
    /// `node.user_id = $user_id` inside the fulltext query so cross-tenant
    /// entity bleed is impossible.
    ///
    /// On any failure (LLM error, Neo4j unavailable), returns an empty result
    /// with `prefer_graph = false` — never blocks the retrieval pipeline.
    #[instrument(skip(self, user_id), fields(query_len = query.len(), user_id = %user_id))]
    pub async fn link(&self, user_id: &UserId, query: &str) -> EntityLinkResult {
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
            let linked = self.try_link_entity(user_id, &mention).await;
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
        // Build the request as a mutable `serde_json::Map` so we can
        // conditionally include `temperature` (non-reasoning models only).
        // gpt-5.x / o-series reject any explicit `temperature` value.
        let mut body = serde_json::Map::new();
        body.insert(
            "model".into(),
            serde_json::Value::String(self.model.clone()),
        );
        body.insert(
            "messages".into(),
            serde_json::json!([
                { "role": "system", "content": ENTITY_EXTRACT_PROMPT },
                { "role": "user", "content": query },
            ]),
        );
        if let Some(t) = agent_rs_core::config::temperature_for(&self.model, 0.0) {
            body.insert(
                "temperature".into(),
                serde_json::Number::from_f64(f64::from(t))
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        body.insert(
            "response_format".into(),
            serde_json::json!({
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
            }),
        );

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
    ///
    /// Delegates to [`GraphStore::find_entities_by_mention`] which is
    /// backed by a Lucene fulltext index over every name-like property
    /// (`name`, `title`, `heading`) across `Document|Section|Chunk|Entity`
    /// labels. The previous implementation hard-coded `n.name CONTAINS`
    /// which silently missed every structural node (Section / Document /
    /// Chunk), producing a zero-linkage rate on graph-backed corpora
    /// that hadn't run the LLM knowledge-graph extractor.
    async fn try_link_entity(&self, user_id: &UserId, mention: &str) -> LinkedEntity {
        match self
            .graph_store
            .find_entities_by_mention(user_id, mention, 1)
            .await
        {
            Ok(matches) if !matches.is_empty() => {
                let m = &matches[0];
                tracing::debug!(
                    mention,
                    node_id = %m.node_id,
                    display_name = %m.display_name,
                    label = %m.label,
                    score = m.score,
                    "entity linked to KG node"
                );
                LinkedEntity {
                    mention: mention.to_owned(),
                    node_id: Some(m.node_id.clone()),
                    node_name: Some(m.display_name.clone()),
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

/// Test-visible alias for the entity-extract prompt. Integration tests
/// (prompt snapshot bias guard) import this to verify universality.
#[doc(hidden)]
pub const ENTITY_EXTRACT_PROMPT_FOR_TESTING: &str = ENTITY_EXTRACT_PROMPT;

const ENTITY_EXTRACT_PROMPT: &str = "\
Extract the key named entities from the user's query. Return ONLY concrete \
entities (people, organizations, locations, products, standards, concepts, \
numbered-section references, dates, events) — NOT generic terms like \
\"document\", \"section\", or \"information\" unless they are part of a \
named reference.

Rules:
- Extract named entities, identifiers, and numbered references from the \
  query regardless of domain or language.
- Return an empty array if the query has no specific named entities.
- Maximum 5 entities.
- Use the CANONICAL form: prefer the base / lemma / citation form over an \
  inflected surface form when a clear canonical exists.
- Preserve the language of the source text: entities stay in their \
  original language unless the user explicitly asks for translation.
";
