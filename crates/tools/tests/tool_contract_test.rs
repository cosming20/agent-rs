//! Dataset-independent contract tests for every agent tool.
//!
//! These tests pin down the *interface* each tool exposes to the LLM —
//! the tool name, its description string, the JSON-schema shape of its
//! arguments, and a minimal happy-path through `call` where feasible
//! (i.e. for tools backed by a trait that can be mocked without
//! spinning up HTTP / database / vector-store backends).
//!
//! Nothing in this file references a specific ingested document. Every
//! assertion is about structure, not content. The tests are designed
//! to catch:
//!
//! - missing or trivially-short tool descriptions (which starve the
//!   LLM of decision signal);
//! - typos in the `NAME` constant that would break tool dispatch;
//! - broken `parameters` JSON schemas (missing `required`, invalid
//!   types) that would reject well-formed LLM calls;
//! - arg-struct drift where a field was renamed in the struct but not
//!   in the schema (or vice versa).

use std::sync::Arc;

use agent_rs_core::UserId;
use agent_rs_core::config::{BraveConfig, FetchConfig, WebBudgetConfig};
use agent_rs_stores::traits::StateStore;
use agent_rs_stores::types::{EvidenceItem, Memory, QueryRun};
use agent_rs_tools::{
    FetchSectionChunksTool, FetchUrlTool, OutlineCorpusTool, RememberTool, SearchKnowledgeArgs,
    SearchKnowledgeTool, SearchMode, SearchNewsTool, SearchSectionsTool, SearchWebTool,
    WebBudgetTracker,
};
use async_trait::async_trait;
use rig::tool::Tool;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Shared contract assertions
// ---------------------------------------------------------------------------

/// Minimum description length below which a tool's affordances become too
/// thin for a tool-calling LLM to route correctly. 60 chars is low enough
/// to accept terse descriptions while catching stub-sized ones.
const MIN_DESCRIPTION_CHARS: usize = 60;

/// Verify the static tool contract: description is substantive, parameter
/// schema is a JSON object, every listed `required` field exists as a
/// declared property.
fn assert_tool_contract(name: &str, description: &str, parameters: &Value) {
    assert!(!name.is_empty(), "tool name must be non-empty");
    assert!(
        description.chars().count() >= MIN_DESCRIPTION_CHARS,
        "tool {name:?} description is too short ({} chars < {}): {description:?}",
        description.chars().count(),
        MIN_DESCRIPTION_CHARS
    );
    let obj = parameters
        .as_object()
        .unwrap_or_else(|| panic!("tool {name:?} parameters must be a JSON object"));
    assert_eq!(
        obj.get("type").and_then(Value::as_str),
        Some("object"),
        "tool {name:?} parameters.type must be \"object\""
    );
    let properties = obj
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("tool {name:?} parameters must declare properties"));
    if let Some(required) = obj.get("required").and_then(Value::as_array) {
        for req in required {
            let field = req
                .as_str()
                .unwrap_or_else(|| panic!("tool {name:?} required entry must be a string"));
            assert!(
                properties.contains_key(field),
                "tool {name:?} required field {field:?} is not declared in properties"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Common fixtures
// ---------------------------------------------------------------------------

fn test_budget_tracker() -> Arc<WebBudgetTracker> {
    Arc::new(WebBudgetTracker::new(&WebBudgetConfig {
        max_searches: 5,
        max_news: 5,
        max_fetches: 5,
        timeout_s: 5.0,
        max_bytes: 1024,
    }))
}

fn test_fetch_config() -> FetchConfig {
    FetchConfig {
        max_length: 1024,
        max_bytes: 1024,
        timeout: 5,
        max_redirects: 3,
    }
}

fn test_brave_config() -> BraveConfig {
    BraveConfig {
        api_key: "test-key".into(),
        base_url: "https://api.search.brave.com/res/v1".into(),
        timeout_s: 5,
    }
}

// ---------------------------------------------------------------------------
// RememberTool — full round-trip against an in-memory StateStore mock
// ---------------------------------------------------------------------------

/// Minimal in-memory `StateStore` impl — only covers the subset of the
/// trait the `RememberTool` exercises. Any other method panics, which is
/// the intended behaviour for a narrow unit-test fixture: if a future
/// change to `RememberTool` reaches into a new state-store method, the
/// test will fail loudly rather than pretend to succeed.
#[derive(Default)]
struct InMemoryStateStore {
    memories: tokio::sync::Mutex<Vec<Memory>>,
}

#[async_trait]
impl StateStore for InMemoryStateStore {
    async fn get_memories(
        &self,
        user_id: &UserId,
    ) -> Result<Vec<Memory>, agent_rs_core::StoreError> {
        let all = self.memories.lock().await;
        Ok(all
            .iter()
            .filter(|m| m.user_id == *user_id)
            .cloned()
            .collect())
    }

    async fn store_memory(&self, memory: &Memory) -> Result<(), agent_rs_core::StoreError> {
        self.memories.lock().await.push(memory.clone());
        Ok(())
    }

    async fn persist_run(&self, _run: &QueryRun) -> Result<(), agent_rs_core::StoreError> {
        unimplemented!("persist_run is not exercised by RememberTool")
    }

    async fn get_run(
        &self,
        _id: &uuid::Uuid,
    ) -> Result<Option<QueryRun>, agent_rs_core::StoreError> {
        unimplemented!("get_run is not exercised by RememberTool")
    }

    async fn store_evidence(&self, _item: &EvidenceItem) -> Result<(), agent_rs_core::StoreError> {
        unimplemented!("store_evidence is not exercised by RememberTool")
    }

    async fn store_evidence_batch(
        &self,
        _items: &[EvidenceItem],
    ) -> Result<(), agent_rs_core::StoreError> {
        unimplemented!("store_evidence_batch is not exercised by RememberTool")
    }

    async fn get_evidence(
        &self,
        _run_id: &uuid::Uuid,
    ) -> Result<Vec<EvidenceItem>, agent_rs_core::StoreError> {
        unimplemented!("get_evidence is not exercised by RememberTool")
    }
}

#[tokio::test]
async fn remember_tool_definition_contract() {
    let store: Arc<dyn StateStore> = Arc::new(InMemoryStateStore::default());
    let tool = RememberTool::new_for_user(store, UserId::from_string("test-user".to_owned()));
    let def = tool.definition(String::new()).await;

    assert_eq!(def.name, "remember");
    assert_tool_contract(&def.name, &def.description, &def.parameters);
    let low = def.description.to_lowercase();
    assert!(
        low.contains("memory") || low.contains("remember") || low.contains("recall"),
        "RememberTool description should mention memory / remember / recall: {:?}",
        def.description
    );

    let required = def
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .expect("remember must have required fields declared");
    assert_eq!(
        required
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["content"]
    );

    // user_id must NOT appear in the schema — it is baked in at construction
    // time and the LLM must never be able to choose or override it.
    let props = def.parameters.get("properties").and_then(Value::as_object);
    assert!(
        props.is_some_and(|p| !p.contains_key("user_id")),
        "tool schema must NOT expose user_id to the LLM"
    );
}

#[tokio::test]
async fn remember_tool_stores_memory_round_trip() {
    let store_impl = Arc::new(InMemoryStateStore::default());
    let store: Arc<dyn StateStore> = store_impl.clone();
    let user = UserId::from_string("alice".to_owned());
    let tool = RememberTool::new_for_user(store, user.clone());

    let args_json = json!({ "content": "prefers concise answers" });
    let args: agent_rs_tools::remember::RememberArgs = serde_json::from_value(args_json).unwrap();
    let reply = tool.call(args).await.expect("call should succeed");

    assert!(
        reply.to_lowercase().contains("remember"),
        "tool reply should confirm the save: {reply:?}"
    );

    let stored = store_impl.memories.lock().await;
    assert_eq!(
        stored.len(),
        1,
        "exactly one memory should have been stored"
    );
    assert_eq!(stored[0].user_id, user);
    assert_eq!(stored[0].content, "prefers concise answers");
}

// ---------------------------------------------------------------------------
// FetchUrlTool — definition contract + local input-validation behaviour
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fetch_tool_definition_contract() {
    let tool = FetchUrlTool::new(test_fetch_config(), test_budget_tracker());
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "fetch_url");
    assert_tool_contract(&def.name, &def.description, &def.parameters);

    let required = def
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .expect("fetch_url must declare required fields");
    assert_eq!(
        required
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["url"]
    );
}

#[tokio::test]
async fn fetch_tool_rejects_non_http_scheme() {
    let tool = FetchUrlTool::new(test_fetch_config(), test_budget_tracker());
    let args: agent_rs_tools::fetch::FetchUrlArgs =
        serde_json::from_value(json!({ "url": "file:///etc/passwd" })).unwrap();
    let result = tool.call(args).await;
    assert!(result.is_err(), "fetch_url must reject non-http(s) schemes");
}

#[tokio::test]
async fn fetch_tool_rejects_private_ip() {
    let tool = FetchUrlTool::new(test_fetch_config(), test_budget_tracker());
    let args: agent_rs_tools::fetch::FetchUrlArgs =
        serde_json::from_value(json!({ "url": "http://127.0.0.1/admin" })).unwrap();
    let result = tool.call(args).await;
    assert!(
        result.is_err(),
        "fetch_url must reject private-IP addresses (SSRF protection)"
    );
}

// ---------------------------------------------------------------------------
// SearchWebTool + SearchNewsTool — definition contracts only (network-bound)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_web_tool_definition_contract() {
    let tool = SearchWebTool::new(
        test_budget_tracker(),
        test_brave_config(),
        reqwest::Client::new(),
    );
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "search_web");
    assert_tool_contract(&def.name, &def.description, &def.parameters);

    let low = def.description.to_lowercase();
    assert!(
        low.contains("web") || low.contains("search") || low.contains("brave"),
        "search_web description must reference web/search: {:?}",
        def.description
    );
}

#[tokio::test]
async fn search_news_tool_definition_contract() {
    let tool = SearchNewsTool::new(
        test_budget_tracker(),
        test_brave_config(),
        reqwest::Client::new(),
    );
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "search_news");
    assert_tool_contract(&def.name, &def.description, &def.parameters);
}

// ---------------------------------------------------------------------------
// SearchKnowledgeTool — definition contract (router mock is out of scope)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_knowledge_tool_definition_contract() {
    // Router is concrete (not trait-backed), so behavioural testing lives
    // at the retrieval-crate integration layer. Here we pin down only the
    // static definition.
    let tool: SearchKnowledgeTool = serde_json::from_value(json!({}))
        .expect("SearchKnowledgeTool should deserialize from {} via serde-skip defaults");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "search_knowledge");
    assert_tool_contract(&def.name, &def.description, &def.parameters);

    // Strategy enum surface must be preserved so the LLM can choose.
    let strategy = def
        .parameters
        .get("properties")
        .and_then(|p| p.get("strategy"))
        .expect("search_knowledge must expose a strategy parameter");
    let strategy_enum = strategy
        .get("enum")
        .and_then(Value::as_array)
        .expect("strategy must declare its enum values so the LLM can pick");

    // Every strategy advertised in prompts / docs is also accepted by the tool.
    let advertised = [
        "semantic",
        "keyword",
        "hybrid",
        "graph",
        "graph_hybrid",
        "rag_completion",
        "summaries",
        "temporal",
        "graph_cot",
        "graph_constrained",
    ];
    let enum_vals: Vec<&str> = strategy_enum.iter().filter_map(Value::as_str).collect();
    for want in advertised {
        assert!(
            enum_vals.contains(&want),
            "strategy enum missing advertised value {want:?}; tool enum = {enum_vals:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// FetchSectionChunksTool — definition contract (stores require live backends).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fetch_section_chunks_tool_definition_contract() {
    // All store fields are `#[serde(skip)]`, so `{}` deserializes to `None`
    // for each. `definition()` never touches the stores.
    let tool: FetchSectionChunksTool = serde_json::from_value(json!({}))
        .expect("FetchSectionChunksTool should deserialize from {}");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "fetch_section_chunks");
    assert_tool_contract(&def.name, &def.description, &def.parameters);

    let desc_lc = def.description.to_lowercase();
    assert!(
        desc_lc.contains("section"),
        "fetch_section_chunks description must contain 'section': {:?}",
        def.description
    );
    assert!(
        desc_lc.contains("exhaustive") || desc_lc.contains("all"),
        "fetch_section_chunks description must mention exhaustive retrieval: {:?}",
        def.description
    );

    // section_id must be a required field.
    let required = def
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .expect("fetch_section_chunks must declare required fields");
    let req_vals: Vec<&str> = required.iter().filter_map(Value::as_str).collect();
    assert!(
        req_vals.contains(&"section_id"),
        "fetch_section_chunks must list section_id as required; got: {req_vals:?}"
    );
}

// ---------------------------------------------------------------------------
// SearchSectionsTool — definition contract (stores require live backends).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_sections_tool_definition_contract() {
    let tool: SearchSectionsTool = serde_json::from_value(json!({}))
        .expect("SearchSectionsTool should deserialize from {}");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "search_sections");
    assert_tool_contract(&def.name, &def.description, &def.parameters);
    let desc_lc = def.description.to_lowercase();
    assert!(
        desc_lc.contains("batch") || desc_lc.contains("multiple"),
        "search_sections description must mention batch or multiple: {:?}",
        def.description
    );
}

// ---------------------------------------------------------------------------
// OutlineCorpusTool — definition contract (stores require live backends).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn outline_corpus_tool_definition_contract() {
    // All store fields are `#[serde(skip)]`, so `{}` deserializes to `None`
    // for each. `definition()` never touches the stores.
    let tool: OutlineCorpusTool = serde_json::from_value(json!({}))
        .expect("OutlineCorpusTool should deserialize from {}");
    let def = tool.definition(String::new()).await;
    assert_eq!(def.name, "outline_corpus");
    assert_tool_contract(&def.name, &def.description, &def.parameters);
    assert!(
        def.description.to_lowercase().contains("outline")
            && def.description.to_lowercase().contains("section"),
        "outline_corpus description must reference outline / section: {:?}",
        def.description
    );
}

// ---------------------------------------------------------------------------
// SearchKnowledgeArgs — schema evolution tests (Task 3.4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_knowledge_args_accept_optional_section_id_doc_id_mode() {
    let v = json!({"query": "x", "section_id": "s", "doc_id": "d", "mode": "all_in_section"});
    let parsed: SearchKnowledgeArgs = serde_json::from_value(v).expect("must parse");
    // existing query field still works
    assert_eq!(parsed.query, "x");
    assert_eq!(parsed.section_id.as_deref(), Some("s"));
    assert_eq!(parsed.doc_id.as_deref(), Some("d"));
    assert!(matches!(parsed.mode, SearchMode::AllInSection));
}

#[tokio::test]
async fn search_knowledge_args_default_mode_is_ranked_when_omitted() {
    let parsed: SearchKnowledgeArgs =
        serde_json::from_value(json!({"query": "x"})).expect("must parse");
    assert!(matches!(parsed.mode, SearchMode::Ranked));
    assert!(parsed.section_id.is_none());
    assert!(parsed.doc_id.is_none());
}
