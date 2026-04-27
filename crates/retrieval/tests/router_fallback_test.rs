//! Router integration tests focused on the Qdrant circuit-breaker
//! fallback path introduced in Phase 4.
//!
//! # What this file covers
//!
//! These tests close the gap flagged in the Phase 4 implementer review:
//! the dense-search → keyword-search degradation path was previously
//! exercised only by unit tests of the [`CircuitBreaker`] state machine
//! itself. Here we wire a [`RetrievalRouter`] up to test-only stub
//! `VectorStore` / `GraphStore` / `Embedder` impls and assert the
//! *router-level* behaviour end-to-end:
//!
//! - When the breaker is Open, dense-using strategies degrade to a
//!   BM25 fallback rather than 500'ing the whole query.
//! - The breaker actually short-circuits subsequent calls (the dense
//!   store sees no traffic until the recovery window elapses).
//! - The breaker recovers — Closed → Open → HalfOpen — under the
//!   public API, without needing private `*_at` test variants.
//! - `Hybrid` and `GraphConstrained` keep their non-dense branches
//!   running while the dense branch is short-circuited.
//!
//! # Mocks
//!
//! All store / embedder mocks live at the top of this file. They are
//! deliberately narrow: any method not exercised by these tests
//! `panic!`s with an explanatory message so a future test addition
//! that reaches a new method gets a clear, actionable failure rather
//! than a silent default.
//!
//! # Production-side change
//!
//! The router gained a single new builder method
//! [`RetrievalRouter::with_qdrant_circuit_breaker_arc`] so tests can
//! inject a pre-tripped breaker. The production
//! [`RetrievalRouter::with_qdrant_circuit_breaker`] entry point that
//! takes a `CircuitBreakerConfig` is unchanged.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use agent_rs_core::{CollectionName, Embedding, RetrievalStrategy, Score, StoreError, UserId};
use agent_rs_observability::circuit_breaker::{BreakerState, CircuitBreaker};
use agent_rs_retrieval::{
    Embedder, ResultRanker, RetrievalError, RetrievalRouter, embedder::SparseVector,
};
use agent_rs_stores::traits::{GraphStore, VectorStore};
use agent_rs_stores::types::{
    CollectionConfig, EntityMatch, FilterSpec, GraphEdge, GraphNode, GraphPath, PointPayload,
    RelationType, SearchResult, VectorPoint,
};

// ---------------------------------------------------------------------------
// Mock: Embedder
// ---------------------------------------------------------------------------

/// Test embedder that returns a fixed-length zero vector for every input.
///
/// Production embedders (FastEmbed / OpenAI) are far too heavy for
/// integration tests — they download ONNX weights or hit a remote API.
/// The router only cares that the embedder produces an [`Embedding`]
/// with a sensible dimensionality; the actual values are routed to the
/// vector store, which we mock anyway.
struct FakeEmbedder {
    dim: usize,
}

#[async_trait]
impl Embedder for FakeEmbedder {
    async fn embed_text(&self, _text: &str) -> Result<Embedding, RetrievalError> {
        Ok(Embedding::from_vec(vec![0.0_f32; self.dim]))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        Ok((0..texts.len())
            .map(|_| Embedding::from_vec(vec![0.0_f32; self.dim]))
            .collect())
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Embedding>, RetrievalError> {
        self.embed_batch(texts).await
    }

    async fn embed_sparse(&self, _text: &str) -> Result<SparseVector, RetrievalError> {
        // Sparse embedding isn't called by the router (server-side BM25 lives
        // in Qdrant); panic loudly if a future test path reaches it.
        panic!("FakeEmbedder::embed_sparse not exercised in this test")
    }

    async fn embed_sparse_batch(
        &self,
        _texts: &[&str],
    ) -> Result<Vec<SparseVector>, RetrievalError> {
        panic!("FakeEmbedder::embed_sparse_batch not exercised in this test")
    }

    fn dimensions(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Mock: GraphStore
// ---------------------------------------------------------------------------

/// Graph store that returns "no neighbours" for every traversal.
///
/// The router's graph branches always fall back to an empty result list
/// when no seeds reach a chunk; this matches what would happen in a
/// fresh tenant with no graph data. Methods unrelated to read paths
/// the router exercises panic so we notice if a new code path silently
/// reaches them.
struct EmptyGraphStore;

#[async_trait]
impl GraphStore for EmptyGraphStore {
    async fn add_node(&self, _node: &GraphNode) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::add_node not exercised in this test")
    }

    async fn add_nodes(&self, _nodes: &[GraphNode]) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::add_nodes not exercised in this test")
    }

    async fn add_edge(&self, _edge: &GraphEdge) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::add_edge not exercised in this test")
    }

    async fn add_edges(&self, _edges: &[GraphEdge]) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::add_edges not exercised in this test")
    }

    async fn get_node(&self, _id: &str) -> Result<Option<GraphNode>, StoreError> {
        Ok(None)
    }

    async fn get_neighbors(
        &self,
        _node_id: &str,
        _depth: usize,
    ) -> Result<Vec<GraphNode>, StoreError> {
        Ok(Vec::new())
    }

    // The trait provides a default `get_neighbors_of_labels` that delegates to
    // `get_neighbors`; we accept it.

    async fn traverse(
        &self,
        _start_id: &str,
        _relation_types: &[RelationType],
        _max_depth: usize,
    ) -> Result<Vec<GraphPath>, StoreError> {
        Ok(Vec::new())
    }

    async fn query_nodes(&self, _cypher: &str) -> Result<Vec<GraphNode>, StoreError> {
        Ok(Vec::new())
    }

    async fn query_raw(
        &self,
        _cypher: &str,
        _params: std::collections::HashMap<String, String>,
        _columns: &[&str],
    ) -> Result<Vec<std::collections::HashMap<String, serde_json::Value>>, StoreError> {
        Ok(Vec::new())
    }

    async fn find_entities_by_mention(
        &self,
        _user_id: &UserId,
        _mention: &str,
        _limit: usize,
    ) -> Result<Vec<EntityMatch>, StoreError> {
        Ok(Vec::new())
    }

    async fn delete_node(&self, _id: &str) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::delete_node not exercised in this test")
    }

    async fn rollback_by_run_id(&self, _ingest_run_id: Uuid) -> Result<u64, StoreError> {
        panic!("EmptyGraphStore::rollback_by_run_id not exercised in this test")
    }

    async fn ensure_structural_constraints(&self) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::ensure_structural_constraints not exercised in this test")
    }

    async fn upsert_document_structure(
        &self,
        _header: &agent_rs_stores::DocumentHeaderSnapshot,
        _chunks: &[agent_rs_stores::StructuralChunk],
        _tables: &[agent_rs_stores::StructuralTable],
        _pictures: &[agent_rs_stores::StructuralPicture],
    ) -> Result<(), StoreError> {
        panic!("EmptyGraphStore::upsert_document_structure not exercised in this test")
    }
}

// ---------------------------------------------------------------------------
// Mock: VectorStore — dense always errors, keyword always returns canned hits
// ---------------------------------------------------------------------------

/// Test fixture vector store: dense `search_for_user` always returns
/// `StoreError::QueryFailed`, while `keyword_search_for_user` always
/// returns one canned hit. This is the ideal shape for exercising the
/// breaker's fallback contract: the dense branch hits a real error
/// (the kind that's "trippable"), and there's a non-empty keyword
/// branch waiting to take over.
///
/// Both methods bump dedicated counters so tests can assert e.g. "the
/// dense store was called exactly N times before the breaker
/// short-circuited the rest".
struct DenseFailKeywordOkStore {
    dense_calls: AtomicUsize,
    keyword_calls: AtomicUsize,
    /// Canned chunk id returned by every keyword search. Tests can
    /// match on this id to confirm the result set really came from
    /// the keyword branch.
    keyword_chunk_id: String,
}

impl DenseFailKeywordOkStore {
    fn new() -> Self {
        Self {
            dense_calls: AtomicUsize::new(0),
            keyword_calls: AtomicUsize::new(0),
            keyword_chunk_id: "kw-chunk-1".to_owned(),
        }
    }

    fn dense_calls(&self) -> usize {
        self.dense_calls.load(Ordering::SeqCst)
    }

    fn keyword_calls(&self) -> usize {
        self.keyword_calls.load(Ordering::SeqCst)
    }

    fn canned_keyword_hit(&self) -> SearchResult {
        SearchResult {
            id: self.keyword_chunk_id.clone(),
            score: Score::new(0.95),
            payload: serde_json::json!({
                "chunk_text": "fallback keyword hit",
                "section_id": "s1",
                "minio_object_key": "doc-1",
            }),
        }
    }
}

#[async_trait]
impl VectorStore for DenseFailKeywordOkStore {
    async fn create_collection(
        &self,
        _name: &CollectionName,
        _config: &CollectionConfig,
    ) -> Result<(), StoreError> {
        panic!("DenseFailKeywordOkStore::create_collection not exercised in this test")
    }

    async fn upsert(
        &self,
        _collection: &CollectionName,
        _points: Vec<VectorPoint>,
    ) -> Result<(), StoreError> {
        panic!("DenseFailKeywordOkStore::upsert not exercised in this test")
    }

    async fn delete(
        &self,
        _collection: &CollectionName,
        _ids: Vec<String>,
    ) -> Result<(), StoreError> {
        panic!("DenseFailKeywordOkStore::delete not exercised in this test")
    }

    async fn delete_by_filter(
        &self,
        _collection: &CollectionName,
        _filter: FilterSpec,
    ) -> Result<u64, StoreError> {
        panic!("DenseFailKeywordOkStore::delete_by_filter not exercised in this test")
    }

    async fn collection_exists(&self, _name: &CollectionName) -> Result<bool, StoreError> {
        Ok(true)
    }

    async fn search_for_user(
        &self,
        _collection: &CollectionName,
        _user_id: &UserId,
        _query: &Embedding,
        _limit: usize,
        _extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError> {
        self.dense_calls.fetch_add(1, Ordering::SeqCst);
        Err(StoreError::QueryFailed(
            "synthetic Qdrant outage for breaker test".into(),
        ))
    }

    async fn keyword_search_for_user(
        &self,
        _collection: &CollectionName,
        _user_id: &UserId,
        _query_text: &str,
        _limit: usize,
        _extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError> {
        self.keyword_calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![self.canned_keyword_hit()])
    }

    async fn retrieve_chunks_for_user(
        &self,
        _collection: &CollectionName,
        _user_id: &UserId,
        _ids: &[String],
    ) -> Result<Vec<PointPayload>, StoreError> {
        // Used by graph_search's chunk rehydration. Not needed when seeds are
        // empty (graph_search short-circuits before this call).
        Ok(Vec::new())
    }
}

// ---------------------------------------------------------------------------
// Mock: VectorStore — dense succeeds with canned hits, counts both methods
// ---------------------------------------------------------------------------

/// Test fixture vector store where both dense and keyword search return
/// canned hits. Used by the [`RetrievalStrategy::GraphConstrained`] +
/// [`RetrievalStrategy::Hybrid`] tests: the router's
/// [`derive_seed_ids`](agent_rs_retrieval::router::RetrievalRouter)
/// helper calls `search_for_user` *before* the breaker short-circuits
/// the strategy's own dense branch. We need that call to succeed so
/// the strategy actually reaches `dispatch_single`, where the breaker
/// check then takes effect.
struct CountingVectorStore {
    dense_calls: AtomicUsize,
    keyword_calls: AtomicUsize,
    dense_chunk_id: String,
    keyword_chunk_id: String,
}

impl CountingVectorStore {
    fn new() -> Self {
        Self {
            dense_calls: AtomicUsize::new(0),
            keyword_calls: AtomicUsize::new(0),
            dense_chunk_id: "dense-chunk-1".to_owned(),
            keyword_chunk_id: "kw-chunk-1".to_owned(),
        }
    }

    fn dense_calls(&self) -> usize {
        self.dense_calls.load(Ordering::SeqCst)
    }

    fn keyword_calls(&self) -> usize {
        self.keyword_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl VectorStore for CountingVectorStore {
    async fn create_collection(
        &self,
        _name: &CollectionName,
        _config: &CollectionConfig,
    ) -> Result<(), StoreError> {
        panic!("CountingVectorStore::create_collection not exercised in this test")
    }

    async fn upsert(
        &self,
        _collection: &CollectionName,
        _points: Vec<VectorPoint>,
    ) -> Result<(), StoreError> {
        panic!("CountingVectorStore::upsert not exercised in this test")
    }

    async fn delete(
        &self,
        _collection: &CollectionName,
        _ids: Vec<String>,
    ) -> Result<(), StoreError> {
        panic!("CountingVectorStore::delete not exercised in this test")
    }

    async fn delete_by_filter(
        &self,
        _collection: &CollectionName,
        _filter: FilterSpec,
    ) -> Result<u64, StoreError> {
        panic!("CountingVectorStore::delete_by_filter not exercised in this test")
    }

    async fn collection_exists(&self, _name: &CollectionName) -> Result<bool, StoreError> {
        Ok(true)
    }

    async fn search_for_user(
        &self,
        _collection: &CollectionName,
        _user_id: &UserId,
        _query: &Embedding,
        _limit: usize,
        _extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError> {
        self.dense_calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![SearchResult {
            id: self.dense_chunk_id.clone(),
            score: Score::new(0.9),
            payload: serde_json::json!({
                "chunk_text": "dense canned hit",
                "section_id": "s1",
                "minio_object_key": "doc-1",
            }),
        }])
    }

    async fn keyword_search_for_user(
        &self,
        _collection: &CollectionName,
        _user_id: &UserId,
        _query_text: &str,
        _limit: usize,
        _extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError> {
        self.keyword_calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![SearchResult {
            id: self.keyword_chunk_id.clone(),
            score: Score::new(0.85),
            payload: serde_json::json!({
                "chunk_text": "keyword canned hit",
                "section_id": "s1",
                "minio_object_key": "doc-1",
            }),
        }])
    }

    async fn retrieve_chunks_for_user(
        &self,
        _collection: &CollectionName,
        _user_id: &UserId,
        ids: &[String],
    ) -> Result<Vec<PointPayload>, StoreError> {
        // Echo the requested IDs back as payload-only projections. The router
        // calls this from `graph_search` to rehydrate chunk payloads — but
        // only when seeds reached chunks, which `EmptyGraphStore` never lets
        // happen.
        Ok(ids
            .iter()
            .map(|id| PointPayload {
                id: id.clone(),
                payload: serde_json::json!({
                    "chunk_text": format!("payload for {id}"),
                }),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_collection() -> CollectionName {
    CollectionName::from_string("test_collection".to_owned())
}

fn test_user() -> UserId {
    UserId::try_new("test-user").expect("valid user id")
}

fn build_router_with_breaker(
    vector: Arc<dyn VectorStore + Send + Sync>,
    breaker: Arc<CircuitBreaker>,
) -> RetrievalRouter {
    let embedder: Arc<dyn Embedder + Send + Sync> = Arc::new(FakeEmbedder { dim: 4 });
    let graph: Arc<dyn GraphStore + Send + Sync> = Arc::new(EmptyGraphStore);
    RetrievalRouter::new(
        embedder,
        vector,
        graph,
        ResultRanker::default(),
        test_collection(),
    )
    .with_qdrant_circuit_breaker_arc(breaker)
}

fn assert_is_store_error(err: &RetrievalError) {
    // `RetrievalError::Store(_)` is the variant `is_breaker_trippable`
    // matches on; if the variant wrapping ever changes, this assertion
    // anchors the test to the exact contract the router relies on.
    assert!(
        matches!(err, RetrievalError::Store(_)),
        "expected RetrievalError::Store, got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// After the breaker trips on a dense failure, a subsequent `Semantic`
/// retrieve falls back to the keyword path and returns non-empty
/// evidence rather than 500'ing.
///
/// Sequence:
///   1. First retrieve: breaker is Closed → dense branch is attempted →
///      vector store errors → router returns `Err(Store(...))` and the
///      breaker records its first failure (threshold = 1 ⇒ Open).
///   2. Second retrieve: breaker is Open → `breaker_allows_qdrant`
///      short-circuits the dense branch → keyword branch supplies a
///      single canned hit → router returns `Ok(EvidencePack)` with one
///      document.
#[tokio::test]
async fn semantic_strategy_falls_back_to_keyword_when_vector_store_errors() {
    let store = Arc::new(DenseFailKeywordOkStore::new());
    let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_secs(60), 1));
    let router = build_router_with_breaker(Arc::clone(&store) as _, Arc::clone(&breaker));
    let user = test_user();

    // Step 1: first call surfaces the dense error and trips the breaker.
    let first = router
        .retrieve_filtered(&user, "any query", RetrievalStrategy::Semantic, 5, None)
        .await;
    let err = first.expect_err("dense failure must propagate to caller before fallback kicks in");
    assert_is_store_error(&err);
    assert_eq!(
        store.dense_calls(),
        1,
        "first retrieve must hit the dense branch exactly once",
    );
    assert_eq!(
        breaker.state(),
        BreakerState::Open,
        "single failure with threshold=1 must trip the breaker",
    );

    // Step 2: second call must skip the dense branch and serve from
    // the keyword fallback.
    let pack = router
        .retrieve_filtered(&user, "any query", RetrievalStrategy::Semantic, 5, None)
        .await
        .expect("breaker-open path must serve from keyword fallback");
    assert!(
        !pack.is_empty(),
        "keyword fallback must surface the canned hit",
    );
    assert_eq!(
        pack.documents.len(),
        1,
        "exactly one canned keyword hit was registered",
    );
    assert_eq!(
        store.dense_calls(),
        1,
        "post-trip retrieve must NOT touch the dense branch",
    );
    assert!(
        store.keyword_calls() >= 1,
        "fallback must invoke the keyword branch at least once",
    );
}

/// With `failure_threshold = 2`, two real dense failures are required
/// to trip the breaker. The third retrieve must short-circuit before
/// touching the dense branch and serve from the keyword fallback.
#[tokio::test]
async fn breaker_opens_after_failure_threshold_and_short_circuits_subsequent_calls() {
    let store = Arc::new(DenseFailKeywordOkStore::new());
    let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60), 1));
    let router = build_router_with_breaker(Arc::clone(&store) as _, Arc::clone(&breaker));
    let user = test_user();

    // Failure 1 — breaker still Closed afterwards.
    let r1 = router
        .retrieve_filtered(&user, "q", RetrievalStrategy::Semantic, 5, None)
        .await;
    assert!(r1.is_err(), "first failure must propagate");
    assert_eq!(breaker.state(), BreakerState::Closed);

    // Failure 2 — breaker trips to Open.
    let r2 = router
        .retrieve_filtered(&user, "q", RetrievalStrategy::Semantic, 5, None)
        .await;
    assert!(r2.is_err(), "second failure must propagate");
    assert_eq!(breaker.state(), BreakerState::Open);

    // Call 3 — short-circuited; keyword fallback path runs instead.
    let r3 = router
        .retrieve_filtered(&user, "q", RetrievalStrategy::Semantic, 5, None)
        .await
        .expect("post-trip call must serve from keyword fallback");
    assert!(!r3.is_empty(), "fallback must surface the canned hit");

    assert_eq!(
        store.dense_calls(),
        2,
        "exactly the failure_threshold dense calls were attempted; the third was short-circuited",
    );
    assert_eq!(
        store.keyword_calls(),
        1,
        "only the post-trip retrieve hit the keyword fallback",
    );
}

/// The breaker progresses Closed → Open → HalfOpen under the public API
/// once the wall-clock recovery window elapses. This proves that test 2's
/// "Open" assertion isn't sticking forever — recovery genuinely admits
/// probes again, which is the property a real backend outage relies on.
///
/// We use a 50 ms recovery window and `tokio::time::sleep` past it so
/// the test stays deterministic without reaching into private
/// `*_at(now)` variants of the breaker (which are file-private and not
/// reachable from an integration test).
#[tokio::test]
async fn breaker_recovery_admits_probe_after_recovery_window() {
    let breaker = CircuitBreaker::new(1, Duration::from_millis(50), 1);
    assert_eq!(breaker.state(), BreakerState::Closed);

    breaker.record_failure();
    assert_eq!(
        breaker.state(),
        BreakerState::Open,
        "single failure with threshold=1 must trip to Open",
    );
    assert!(
        !breaker.allow(),
        "Open breaker must short-circuit before the recovery window elapses",
    );

    // Wait past the recovery window with a generous safety margin so
    // CI scheduling jitter cannot flake the test.
    tokio::time::sleep(Duration::from_millis(120)).await;

    assert_eq!(
        breaker.state(),
        BreakerState::HalfOpen,
        "after the recovery window the breaker must admit a probe",
    );
    assert!(
        breaker.allow(),
        "HalfOpen breaker must admit at least one probe call",
    );
}

/// `Hybrid` retrieval normally runs `dense || keyword`. When the
/// breaker is Open, the dense leg is dropped and the strategy keeps
/// the keyword contribution alive — preserving partial-availability
/// semantics under a Qdrant outage.
#[tokio::test]
async fn hybrid_strategy_drops_dense_branch_when_breaker_open() {
    let store = Arc::new(CountingVectorStore::new());
    let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_secs(60), 1));
    // Pre-trip the breaker so the very first retrieve already takes
    // the fallback path. This is what
    // `with_qdrant_circuit_breaker_arc` was added for — production
    // wiring uses the config-driven builder, tests inject a
    // pre-tripped instance.
    breaker.record_failure();
    assert_eq!(breaker.state(), BreakerState::Open);

    let router = build_router_with_breaker(Arc::clone(&store) as _, Arc::clone(&breaker));
    let user = test_user();

    let pack = router
        .retrieve_filtered(&user, "hybrid query", RetrievalStrategy::Hybrid, 5, None)
        .await
        .expect("Hybrid with breaker Open must still complete via keyword");

    assert!(
        !pack.is_empty(),
        "keyword branch must contribute when dense is short-circuited",
    );
    assert_eq!(
        store.dense_calls(),
        0,
        "dense branch must NOT be attempted while breaker is Open",
    );
    assert!(
        store.keyword_calls() >= 1,
        "keyword branch must run on every Hybrid retrieval",
    );
    // The single hit must come from the keyword fallback, not the
    // dense branch.
    assert!(
        pack.documents.iter().any(|d| d.id == "kw-chunk-1"),
        "evidence must contain the keyword canned hit",
    );
    assert!(
        !pack.documents.iter().any(|d| d.id == "dense-chunk-1"),
        "evidence must NOT contain the dense canned hit",
    );
}

/// `GraphConstrained` is documented to fall back from `dense` to
/// `keyword` when the Qdrant breaker is Open, while keeping the graph
/// branch unchanged. This test characterises the *currently reachable*
/// behaviour and makes the architectural gap visible.
///
/// # The gap
///
/// `GraphConstrained::dispatch_single` calls `derive_seed_ids` before
/// the breaker-aware `dense_fut`. `derive_seed_ids` itself uses
/// `timed_dense_search`, which (a) does NOT consult the breaker and
/// (b) calls `record_success` on a successful response — re-closing
/// any breaker we pre-tripped via `record_failure`. As a consequence:
///
/// - With a healthy dense store + a pre-tripped breaker:
///   `derive_seed_ids` succeeds → `record_success` resets the breaker
///   to `Closed` → the strategy's own dense branch is admitted.
///   The breaker-Open fallback path is therefore not reached.
///
/// - With a broken dense store (returns `Err`): `derive_seed_ids`
///   propagates the error before `dispatch_single` ever reaches the
///   strategy body, so the fallback path is again not reached.
///
/// The test below uses the healthy-store flavour and asserts both the
/// observable behaviour (the strategy completes, the dense branch
/// runs) and the precise state transition that hides the fallback
/// from external observation. A future change that gates
/// `derive_seed_ids` on the breaker (or that drops `record_success`
/// for the seed-derivation pass) would make the fallback reachable
/// and would invalidate the assertions below — pointing operators at
/// exactly what changed.
///
/// NOTE: this is a *characterisation test*, not a behavioural one.
/// When the gap is closed, replace the assertions below with the
/// stronger contract: dense_calls == 1 (only `derive_seed_ids`),
/// keyword_calls >= 1 (the substituted fallback), and the evidence
/// pack contains the keyword hit.
#[tokio::test]
async fn graph_constrained_seed_derivation_resets_pretripped_breaker() {
    let store = Arc::new(CountingVectorStore::new());
    let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_secs(60), 1));
    breaker.record_failure();
    assert_eq!(
        breaker.state(),
        BreakerState::Open,
        "pre-test setup: breaker must start Open",
    );

    let router = build_router_with_breaker(Arc::clone(&store) as _, Arc::clone(&breaker));
    let user = test_user();

    let pack = router
        .retrieve_filtered(
            &user,
            "graph constrained query",
            RetrievalStrategy::GraphConstrained,
            5,
            None,
        )
        .await
        .expect("GraphConstrained with healthy dense store must complete");

    // Two dense calls confirm both the seed-derivation pass AND the
    // strategy's own dense branch ran — the latter is admitted only
    // because `derive_seed_ids` reset the breaker via record_success.
    assert_eq!(
        store.dense_calls(),
        2,
        "expected derive_seed_ids + dispatch_single dense branch; gap-doc test asserting current behaviour",
    );
    // The strategy's dense branch ran instead of the keyword fallback,
    // so keyword_search_for_user is never called.
    assert_eq!(
        store.keyword_calls(),
        0,
        "fallback path is not reachable today; see the test docstring for the architectural gap",
    );
    // Final state: derive_seed_ids' record_success reset the breaker
    // to Closed.
    assert_eq!(
        breaker.state(),
        BreakerState::Closed,
        "successful seed-derivation call resets a pre-tripped breaker — this is the gap",
    );
    // Evidence still surfaces the dense canned hit; the graph branch
    // is empty because EmptyGraphStore returns no neighbours.
    assert!(
        pack.documents.iter().any(|d| d.id == "dense-chunk-1"),
        "evidence must contain the dense canned hit (the fallback path was bypassed)",
    );
}
