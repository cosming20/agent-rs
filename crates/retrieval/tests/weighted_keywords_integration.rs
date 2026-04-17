//! Integration test: with a mock LLM, the extractor produces
//! `WeightedKeywords` that flow into all three consumer pure functions
//! (`weighted_bm25_query`, `adjust_for_weighted_keywords`,
//! `order_candidates_overlap_first`) and actually modify their output.

use std::sync::Arc;

use agent_rs_retrieval::{
    BM25WeightingConfig, EntityLinkerPrioritizationConfig, RerankAdjustmentConfig,
    WeightedKeywordExtractor, WeightedKeywordExtractorConfig, adjust_for_weighted_keywords,
    order_candidates_overlap_first, weighted_bm25_query,
};

#[tokio::test]
async fn mock_llm_success_flows_to_all_three_consumers() {
    let mut server = mockito::Server::new_async().await;
    let llm_body = r#"{"choices":[{"message":{"content":
        "{\"terms\":[{\"term\":\"compiler\",\"weight\":0.9,\"is_phrase\":false},{\"term\":\"static types\",\"weight\":0.7,\"is_phrase\":true}]}"
    }}]}"#;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body(llm_body)
        .create_async()
        .await;

    let http = Arc::new(reqwest::Client::new());
    let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
    cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
    let extractor = WeightedKeywordExtractor::new(cfg, http);

    let kw = extractor
        .extract("is a compiler faster than an interpreter?")
        .await
        .expect("extract");
    assert_eq!(kw.terms.len(), 2, "expected two terms from mock");

    // Consumer #1: BM25 rewrite surfaces the high-weight term.
    let bm25 = weighted_bm25_query("is a compiler faster", &kw, &BM25WeightingConfig::default());
    assert!(bm25.contains("compiler"));

    // Consumer #2: rerank adjustment boosts a matching candidate.
    let candidate = "A compiler translates static types into machine code.";
    let adjusted =
        adjust_for_weighted_keywords(1.0, candidate, &kw, &RerankAdjustmentConfig::default());
    assert!(adjusted > 1.0, "adjusted={adjusted}, expected boost");

    // Consumer #3: entity-linker tiering puts the intersection member first.
    let extracted = vec!["compiler".to_owned(), "interpreter".to_owned()];
    let ordered = order_candidates_overlap_first(
        &kw,
        &extracted,
        &EntityLinkerPrioritizationConfig { top_k: 5 },
    );
    assert_eq!(ordered[0], "compiler");
}

#[tokio::test]
async fn mock_llm_failure_keeps_all_consumers_as_noops() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/v1/chat/completions")
        .with_status(500)
        .with_body("broken")
        .expect_at_least(1)
        .create_async()
        .await;

    let http = Arc::new(reqwest::Client::new());
    let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
    cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
    cfg.max_retries = 1;
    let extractor = WeightedKeywordExtractor::new(cfg, http);

    let kw = extractor.extract("anything").await.expect("fallback is Ok");
    assert!(kw.terms.is_empty(), "failure path must produce empty()");

    // All three consumers no-op on empty.
    let q = "some query";
    assert_eq!(
        weighted_bm25_query(q, &kw, &BM25WeightingConfig::default()),
        q
    );
    let base = 0.5_f32;
    let adj =
        adjust_for_weighted_keywords(base, "any text", &kw, &RerankAdjustmentConfig::default());
    assert!((adj - base).abs() < 1e-6);
    let extracted = vec!["foo".to_owned(), "bar".to_owned()];
    let ordered = order_candidates_overlap_first(
        &kw,
        &extracted,
        &EntityLinkerPrioritizationConfig::default(),
    );
    assert_eq!(ordered, extracted);
}
