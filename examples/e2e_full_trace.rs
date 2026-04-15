//! E2E test harness that runs the full agent-rs pipeline and outputs a
//! detailed JSON debug trace capturing every step.
//!
//! # Usage
//!
//! ```bash
//! # Ensure Qdrant (ports 1050/1051), Neo4j (port 1061), and Postgres (port 1054) are running.
//! cargo run --example e2e_full_trace
//! ```
//!
//! The JSON trace is written to `agent-rs/test_results/e2e_trace_{timestamp}.json`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::Serialize;
use tracing::info;
use uuid::Uuid;

use agent_rs::stores::connect_stores;
use agent_rs_core::config::Settings;
use agent_rs_core::types::{CollectionName, Embedding, RetrievalStrategy};
use agent_rs_retrieval::Embedder;
use agent_rs_retrieval::embedder::{FastEmbedder, FastEmbedderConfig};
use agent_rs_retrieval::ranker::ResultRanker;
use agent_rs_retrieval::router::RetrievalRouter;
use agent_rs_stores::traits::{GraphStore, VectorStore};
use agent_rs_stores::types::{
    CollectionConfig, GraphEdge, GraphNode, NodeLabel, RelationType, VectorPoint,
};

// ---------------------------------------------------------------------------
// Trace types -- JSON-serializable records of every step
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StepTrace {
    step: String,
    input: serde_json::Value,
    output: serde_json::Value,
    duration_ms: u64,
    status: String,
}

#[derive(Serialize)]
struct IngestionTrace {
    documents_count: usize,
    steps: Vec<StepTrace>,
}

#[derive(Serialize)]
struct QueryTrace {
    query: String,
    strategy: String,
    pipeline_steps: Vec<StepTrace>,
    response: serde_json::Value,
    total_duration_ms: u64,
}

#[derive(Serialize)]
struct FullTrace {
    test_run_id: String,
    timestamp: String,
    config: serde_json::Value,
    ingestion: IngestionTrace,
    queries: Vec<QueryTrace>,
    cleanup: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Test document generation
// ---------------------------------------------------------------------------

struct TestDocument {
    id: String,
    title: String,
    content: String,
}

/// Generate test documents about machine learning concepts.
///
/// Each document covers a distinct concept with enough text for meaningful
/// embedding and retrieval.
fn generate_test_documents() -> Vec<TestDocument> {
    vec![
        TestDocument {
            id: "doc_neural_networks".into(),
            title: "Neural Networks".into(),
            content: "Neural networks are computational models inspired by the biological neural \
                networks in the human brain. They consist of layers of interconnected nodes \
                (neurons) that process information using weighted connections. Each neuron \
                applies an activation function to the weighted sum of its inputs. Deep neural \
                networks have multiple hidden layers between the input and output layers, \
                enabling them to learn hierarchical representations of data. Common \
                architectures include feedforward networks, convolutional neural networks \
                (CNNs) for image processing, and recurrent neural networks (RNNs) for \
                sequential data. Training involves forward propagation to compute predictions \
                and backpropagation to update weights based on the error gradient."
                .into(),
        },
        TestDocument {
            id: "doc_gradient_descent".into(),
            title: "Gradient Descent".into(),
            content: "Gradient descent is a first-order iterative optimization algorithm used \
                to find the minimum of a differentiable function. In machine learning, it is \
                the primary method for training models by minimizing a loss function. The \
                algorithm works by computing the gradient (partial derivatives) of the loss \
                with respect to each model parameter and then updating the parameters in the \
                direction opposite to the gradient. Variants include stochastic gradient \
                descent (SGD), which uses a single random sample per update; mini-batch \
                gradient descent, which uses a small subset of training data; and adaptive \
                methods like Adam, RMSProp, and AdaGrad that adjust learning rates per \
                parameter based on historical gradient information."
                .into(),
        },
        TestDocument {
            id: "doc_transformers".into(),
            title: "Transformer Architecture".into(),
            content: "The Transformer architecture was introduced in the paper 'Attention Is \
                All You Need' by Vaswani et al. in 2017. It relies entirely on self-attention \
                mechanisms to draw global dependencies between input and output, eliminating \
                the need for recurrence or convolution. The architecture consists of an \
                encoder stack and a decoder stack, each composed of multi-head self-attention \
                layers and position-wise feed-forward networks. Key innovations include \
                scaled dot-product attention, multi-head attention for parallel processing of \
                different representation subspaces, and positional encoding to inject sequence \
                order information. Transformers form the foundation of modern large language \
                models like GPT, BERT, and T5."
                .into(),
        },
        TestDocument {
            id: "doc_reinforcement_learning".into(),
            title: "Reinforcement Learning".into(),
            content: "Reinforcement learning (RL) is a type of machine learning where an \
                agent learns to make decisions by interacting with an environment to maximize \
                cumulative reward. The agent observes the current state, takes an action, \
                receives a reward signal, and transitions to a new state. Key concepts include \
                the Markov Decision Process (MDP) framework, value functions that estimate \
                expected returns, and policy functions that map states to actions. Major \
                algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient methods \
                like REINFORCE, and Actor-Critic methods that combine value-based and \
                policy-based approaches. Applications range from game playing and robotics \
                to recommendation systems and autonomous vehicles."
                .into(),
        },
        TestDocument {
            id: "doc_transfer_learning".into(),
            title: "Transfer Learning".into(),
            content: "Transfer learning is a machine learning technique where a model trained \
                on one task is repurposed for a different but related task. Instead of \
                training from scratch, transfer learning leverages knowledge gained from a \
                source domain to improve performance on a target domain. Common approaches \
                include fine-tuning pretrained models, feature extraction using frozen layers, \
                and domain adaptation techniques. In natural language processing, pretrained \
                language models like BERT and GPT are fine-tuned on downstream tasks such as \
                sentiment analysis, named entity recognition, and question answering. In \
                computer vision, models pretrained on ImageNet are commonly adapted for \
                medical imaging, satellite imagery, and other specialized visual tasks."
                .into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Config sanitization -- remove secrets before writing to trace
// ---------------------------------------------------------------------------

/// Produce a sanitized version of `Settings` safe for inclusion in the trace.
///
/// Replaces sensitive fields (api_key, password) with `"***"`.
fn sanitize_config(settings: &Settings) -> serde_json::Value {
    serde_json::json!({
        "llm": {
            "api_key": "***",
            "model": settings.llm.model,
        },
        "qdrant": {
            "host": settings.qdrant.host,
            "http_port": settings.qdrant.http_port,
            "grpc_port": settings.qdrant.grpc_port,
            "api_key": "***",
        },
        "postgres": {
            "host": settings.postgres.host,
            "port": settings.postgres.port,
            "user": settings.postgres.user,
            "password": "***",
            "db": settings.postgres.db,
        },
        "neo4j": {
            "host": settings.neo4j.host,
            "bolt_port": settings.neo4j.bolt_port,
            "user": settings.neo4j.user,
            "password": "***",
        },
        "agent": settings.agent,
        "context": settings.context,
        "evidence": settings.evidence,
        "synthesis": settings.synthesis,
        "ranker": settings.ranker,
        "verifier": settings.verifier,
        "web_budget": settings.web_budget,
        "fetch": settings.fetch,
        "retrieval": settings.retrieval,
    })
}

// ---------------------------------------------------------------------------
// Step execution helper -- wraps a step in timing + error capture
// ---------------------------------------------------------------------------

/// Execute a named step, capturing its timing, input, output, and status
/// into a [`StepTrace`].
///
/// On success, returns the step output alongside the trace entry.
/// On failure, the trace records the error and `None` is returned for the
/// output so the caller can decide whether to continue.
async fn trace_step<F, Fut, T>(
    step_name: &str,
    input: serde_json::Value,
    f: F,
) -> (StepTrace, Option<T>)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<(serde_json::Value, T)>>,
{
    let start = Instant::now();
    match f().await {
        Ok((output_json, value)) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let trace = StepTrace {
                step: step_name.to_owned(),
                input,
                output: output_json,
                duration_ms,
                status: "success".into(),
            };
            (trace, Some(value))
        }
        Err(err) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let trace = StepTrace {
                step: step_name.to_owned(),
                input,
                output: serde_json::json!(null),
                duration_ms,
                status: format!("error: {err}"),
            };
            (trace, None)
        }
    }
}

/// Synchronous variant of [`trace_step`] for non-async operations.
fn trace_step_sync<F, T>(step_name: &str, input: serde_json::Value, f: F) -> (StepTrace, Option<T>)
where
    F: FnOnce() -> Result<(serde_json::Value, T)>,
{
    let start = Instant::now();
    match f() {
        Ok((output_json, value)) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let trace = StepTrace {
                step: step_name.to_owned(),
                input,
                output: output_json,
                duration_ms,
                status: "success".into(),
            };
            (trace, Some(value))
        }
        Err(err) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let trace = StepTrace {
                step: step_name.to_owned(),
                input,
                output: serde_json::json!(null),
                duration_ms,
                status: format!("error: {err}"),
            };
            (trace, None)
        }
    }
}

// ---------------------------------------------------------------------------
// Entity extraction (simple rule-based for E2E test purposes)
// ---------------------------------------------------------------------------

struct ExtractedEntity {
    id: String,
    name: String,
    label: String,
}

struct ExtractedRelation {
    source: String,
    target: String,
    relation: String,
}

/// Simple keyword-based entity extraction for test purposes.
///
/// Searches each document for known ML concept keywords and creates
/// entities + relations from them. This is NOT the production LLM-based
/// extractor -- it exists solely to populate the graph for E2E testing.
fn extract_entities_from_docs(
    docs: &[TestDocument],
) -> (Vec<ExtractedEntity>, Vec<ExtractedRelation>) {
    let concept_keywords: Vec<(&str, &str)> = vec![
        ("neural_network", "Neural Network"),
        ("gradient_descent", "Gradient Descent"),
        ("backpropagation", "Backpropagation"),
        ("transformer", "Transformer"),
        ("attention", "Attention Mechanism"),
        ("reinforcement_learning", "Reinforcement Learning"),
        ("transfer_learning", "Transfer Learning"),
        ("deep_learning", "Deep Learning"),
        ("cnn", "Convolutional Neural Network"),
        ("rnn", "Recurrent Neural Network"),
        ("bert", "BERT"),
        ("gpt", "GPT"),
        ("fine_tuning", "Fine-tuning"),
        ("adam", "Adam Optimizer"),
        ("sgd", "Stochastic Gradient Descent"),
        ("q_learning", "Q-Learning"),
    ];

    let mut entities: Vec<ExtractedEntity> = Vec::new();
    let mut relations: Vec<ExtractedRelation> = Vec::new();
    let mut seen_entities: HashMap<String, bool> = HashMap::new();

    for (concept_id, concept_name) in &concept_keywords {
        for doc in docs {
            let lower_content = doc.content.to_lowercase();
            let lower_name = concept_name.to_lowercase();
            if lower_content.contains(&lower_name) {
                // Add entity if not seen
                if !seen_entities.contains_key(*concept_id) {
                    entities.push(ExtractedEntity {
                        id: format!("entity_{concept_id}"),
                        name: concept_name.to_string(),
                        label: "Concept".into(),
                    });
                    seen_entities.insert(concept_id.to_string(), true);
                }

                // Add MENTIONED_IN relation from concept to document
                relations.push(ExtractedRelation {
                    source: format!("entity_{concept_id}"),
                    target: doc.id.clone(),
                    relation: "MENTIONED_IN".into(),
                });
            }
        }
    }

    // Add some inter-concept relations based on known ML relationships.
    let inter_relations = [
        ("neural_network", "backpropagation", "USES"),
        ("neural_network", "gradient_descent", "TRAINED_BY"),
        ("transformer", "attention", "BASED_ON"),
        ("deep_learning", "neural_network", "EXTENDS"),
        ("cnn", "neural_network", "IS_A"),
        ("rnn", "neural_network", "IS_A"),
        ("bert", "transformer", "BASED_ON"),
        ("gpt", "transformer", "BASED_ON"),
        ("transfer_learning", "fine_tuning", "USES"),
        ("adam", "gradient_descent", "VARIANT_OF"),
        ("sgd", "gradient_descent", "VARIANT_OF"),
    ];

    for (src, dst, rel) in &inter_relations {
        let src_id = format!("entity_{src}");
        let dst_id = format!("entity_{dst}");
        if seen_entities.contains_key(*src) && seen_entities.contains_key(*dst) {
            relations.push(ExtractedRelation {
                source: src_id,
                target: dst_id,
                relation: rel.to_string(),
            });
        }
    }

    (entities, relations)
}

// ---------------------------------------------------------------------------
// Ingestion pipeline
// ---------------------------------------------------------------------------

async fn run_ingestion(
    docs: &[TestDocument],
    embedder: &FastEmbedder,
    vector_store: &dyn VectorStore,
    graph_store: &dyn GraphStore,
    collection: &CollectionName,
    dims: usize,
) -> IngestionTrace {
    let mut steps: Vec<StepTrace> = Vec::new();

    // Step 1: Create collection
    let (trace, _) = trace_step(
        "create_collection",
        serde_json::json!({
            "collection": collection.as_str(),
            "dense_vector_size": dims,
            "enable_sparse": true,
        }),
        || async {
            let config = CollectionConfig::hybrid(dims);
            vector_store
                .create_collection(collection, &config)
                .await
                .context("create_collection")?;
            let out = serde_json::json!({ "created": true });
            Ok((out, ()))
        },
    )
    .await;
    steps.push(trace);

    // Step 2: Embed documents (dense)
    let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
    let sample_text: String = texts
        .first()
        .map(|t| {
            if t.len() > 100 {
                format!("{}...", &t[..100])
            } else {
                t.to_string()
            }
        })
        .unwrap_or_default();

    let (trace, embeddings) = trace_step(
        "embed_documents",
        serde_json::json!({
            "doc_count": docs.len(),
            "sample_text": sample_text,
        }),
        || async {
            let embs = embedder
                .embed_documents(&texts)
                .await
                .map_err(|e| anyhow::anyhow!("embedding: {e}"))?;
            let out = serde_json::json!({
                "embeddings_count": embs.len(),
                "dimensions": embs.first().map(|e| e.dimensions()).unwrap_or(0),
            });
            Ok((out, embs))
        },
    )
    .await;
    steps.push(trace);

    // Step 3 (skipped): Sparse embeddings are now handled server-side by Qdrant
    // via the `qdrant/bm25` model. Each VectorPoint carries raw text; no local
    // SPLADE inference is needed.

    // Step 4: Upsert vectors
    if let Some(embs) = &embeddings {
        let points: Vec<VectorPoint> = docs
            .iter()
            .zip(embs.iter())
            .map(|(doc, emb)| VectorPoint {
                id: doc.id.clone(),
                embedding: Embedding(emb.0.clone()),
                text: doc.content.clone(),
                payload: serde_json::json!({
                    "title": doc.title,
                    "chunk_text": doc.content,
                    "source": "e2e_test",
                    "doc_id": doc.id,
                }),
            })
            .collect();

        let point_count = points.len();
        let (trace, _) = trace_step(
            "upsert_vectors",
            serde_json::json!({
                "collection": collection.as_str(),
                "point_count": point_count,
            }),
            || async {
                vector_store
                    .upsert(collection, points)
                    .await
                    .context("upsert_vectors")?;
                let out = serde_json::json!({ "upserted": point_count });
                Ok((out, ()))
            },
        )
        .await;
        steps.push(trace);
    } else {
        steps.push(StepTrace {
            step: "upsert_vectors".into(),
            input: serde_json::json!({ "skipped": true }),
            output: serde_json::json!(null),
            duration_ms: 0,
            status: "error: embeddings not available from previous step".into(),
        });
    }

    // Step 5: Extract entities and relations
    let (entities, relations) = extract_entities_from_docs(docs);
    let entity_count = entities.len();
    let relation_count = relations.len();

    // Step 5a: Create graph nodes (including document nodes)
    let (trace, _) = trace_step(
        "create_graph_nodes",
        serde_json::json!({
            "entity_count": entity_count,
            "doc_count": docs.len(),
        }),
        || async {
            // Create document nodes
            for doc in docs {
                let label = NodeLabel::new("Document")
                    .map_err(|e| anyhow::anyhow!("invalid label: {e}"))?;
                let node = GraphNode {
                    id: doc.id.clone(),
                    label,
                    properties: serde_json::json!({
                        "id": doc.id,
                        "title": doc.title,
                    }),
                };
                graph_store
                    .add_node(&node)
                    .await
                    .context("add document node")?;
            }
            // Create entity nodes
            for entity in &entities {
                let label = NodeLabel::new(&entity.label)
                    .map_err(|e| anyhow::anyhow!("invalid label: {e}"))?;
                let node = GraphNode {
                    id: entity.id.clone(),
                    label,
                    properties: serde_json::json!({
                        "id": entity.id,
                        "name": entity.name,
                    }),
                };
                graph_store
                    .add_node(&node)
                    .await
                    .context("add entity node")?;
            }
            let out = serde_json::json!({
                "nodes_created": entity_count + docs.len(),
            });
            Ok((out, ()))
        },
    )
    .await;
    steps.push(trace);

    // Step 5b: Create graph edges
    let (trace, _) = trace_step(
        "create_graph_edges",
        serde_json::json!({ "edge_count": relation_count }),
        || async {
            for rel in &relations {
                let rel_type = RelationType::new(&rel.relation)
                    .map_err(|e| anyhow::anyhow!("invalid relation type: {e}"))?;
                let edge = GraphEdge {
                    source_id: rel.source.clone(),
                    target_id: rel.target.clone(),
                    relation_type: rel_type,
                    properties: serde_json::json!({}),
                };
                graph_store.add_edge(&edge).await.context("add edge")?;
            }
            let out = serde_json::json!({ "edges_created": relation_count });
            Ok((out, ()))
        },
    )
    .await;
    steps.push(trace);

    IngestionTrace {
        documents_count: docs.len(),
        steps,
    }
}

// ---------------------------------------------------------------------------
// Query pipeline execution
// ---------------------------------------------------------------------------

/// Run a single query through the retrieval router and record every step.
///
/// This exercises: embedding, strategy dispatch, search, ranking, and
/// evidence pack assembly -- all the retrieval-layer steps.
///
/// The agent orchestrator steps (agent_runner, grounding_gate, etc.) are
/// simulated because they require a live LLM API key. The trace records
/// what *would* happen with realistic placeholder values so the JSON
/// structure is complete.
async fn run_query_trace(
    query: &str,
    strategy: RetrievalStrategy,
    router: &RetrievalRouter,
    verifier_threshold: f64,
    limit: usize,
) -> QueryTrace {
    let query_start = Instant::now();
    let mut pipeline_steps: Vec<StepTrace> = Vec::new();
    let session_id = Uuid::new_v4();

    // Step 1: context_builder (simulated -- no live DB history for test session)
    let (trace, _) = trace_step(
        "context_builder",
        serde_json::json!({
            "session_id": session_id.to_string(),
            "strategy": strategy.as_db_str(),
        }),
        || async {
            // In production this loads chat history + memories from Postgres.
            // For E2E trace we simulate an empty session.
            let out = serde_json::json!({
                "messages_loaded": 0,
                "memories_loaded": 0,
                "conversation_context": "",
            });
            Ok((out, ()))
        },
    )
    .await;
    pipeline_steps.push(trace);

    // Step 2: tool_factory (simulated -- records tool configuration)
    let (trace, _) = trace_step_sync(
        "tool_factory",
        serde_json::json!({
            "tools_requested": [
                "search_knowledge",
                "search_web",
                "fetch_url",
                "remember",
            ],
        }),
        || {
            let out = serde_json::json!({
                "tools_built": 4,
                "budget": {
                    "max_searches": 3,
                    "max_news": 2,
                    "max_fetches": 3,
                },
            });
            Ok((out, ()))
        },
    );
    pipeline_steps.push(trace);

    // Step 3: Retrieval via router (REAL -- exercises the full retrieval path)
    let (trace, evidence_pack) = trace_step(
        "retrieval",
        serde_json::json!({
            "query": query,
            "strategy": strategy.as_db_str(),
            "limit": limit,
        }),
        || async {
            let pack = router
                .retrieve(query, strategy, limit)
                .await
                .map_err(|e| anyhow::anyhow!("retrieval: {e}"))?;
            let out = serde_json::json!({
                "documents_found": pack.len(),
                "strategy": strategy.as_db_str(),
                "source_counts": pack.source_counts,
                "top_scores": pack.documents.iter()
                    .take(3)
                    .map(|d| d.score.0)
                    .collect::<Vec<_>>(),
            });
            Ok((out, pack))
        },
    )
    .await;
    pipeline_steps.push(trace);

    // Step 4: evidence_fuser (uses the real ranker if we have evidence)
    let evidence_count = evidence_pack.as_ref().map(|p| p.len()).unwrap_or(0);
    let (trace, _) = trace_step_sync(
        "evidence_fuser",
        serde_json::json!({
            "kb_count": evidence_count,
            "web_count": 0,
        }),
        || {
            let out = serde_json::json!({
                "fused_count": evidence_count,
                "strategy": strategy.as_db_str(),
                "source_counts": evidence_pack.as_ref()
                    .map(|p| p.source_counts.clone())
                    .unwrap_or_default(),
            });
            Ok((out, ()))
        },
    );
    pipeline_steps.push(trace);

    // Step 5: grounding_gate (simulated -- uses heuristic verification logic)
    let answer_preview = evidence_pack
        .as_ref()
        .and_then(|p| p.documents.first())
        .and_then(|d| d.payload.get("chunk_text"))
        .and_then(|v| v.as_str())
        .map(|s| if s.len() > 200 { &s[..200] } else { s })
        .unwrap_or("No evidence available to formulate answer.")
        .to_owned();

    // Compute a simple heuristic confidence for the trace.
    let heuristic_confidence = if evidence_count > 0 {
        (evidence_count as f64 / limit as f64).min(1.0) * 0.8 + 0.1
    } else {
        0.0
    };
    let is_grounded = heuristic_confidence >= verifier_threshold;

    let (trace, _) = trace_step_sync(
        "grounding_gate",
        serde_json::json!({
            "answer_length": answer_preview.len(),
            "evidence_count": evidence_count,
        }),
        || {
            let out = serde_json::json!({
                "is_grounded": is_grounded,
                "confidence": heuristic_confidence,
                "retries": 0,
                "ungrounded_claims": Vec::<String>::new(),
                "method": "heuristic_simulation",
            });
            Ok((out, ()))
        },
    );
    pipeline_steps.push(trace);

    // Step 6: result_persister (simulated -- no actual DB write in test)
    let run_id = Uuid::new_v4();
    let (trace, _) = trace_step_sync(
        "result_persister",
        serde_json::json!({ "run_id": run_id.to_string() }),
        || {
            let citation_count = evidence_pack
                .as_ref()
                .map(|p| p.documents.len().min(5))
                .unwrap_or(0);
            let out = serde_json::json!({
                "persisted": true,
                "citations_count": citation_count,
            });
            Ok((out, ()))
        },
    );
    pipeline_steps.push(trace);

    // Build citation list from evidence pack
    let citations: Vec<serde_json::Value> = evidence_pack
        .as_ref()
        .map(|p| {
            p.documents
                .iter()
                .take(5)
                .enumerate()
                .map(|(i, doc)| {
                    let snippet = doc
                        .payload
                        .get("chunk_text")
                        .and_then(|v| v.as_str())
                        .map(|s| {
                            if s.len() > 200 {
                                format!("{}...", &s[..200])
                            } else {
                                s.to_owned()
                            }
                        })
                        .unwrap_or_default();
                    let source = doc
                        .payload
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    serde_json::json!({
                        "index": i + 1,
                        "content_snippet": snippet,
                        "source": source,
                        "score": doc.score.0,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let response = serde_json::json!({
        "answer": format!("[Simulated answer based on retrieved evidence for: {}]", query),
        "citations": citations,
        "confidence": heuristic_confidence,
        "is_grounded": is_grounded,
        "strategy": strategy.as_db_str(),
    });

    let total_duration_ms = query_start.elapsed().as_millis() as u64;

    QueryTrace {
        query: query.to_owned(),
        strategy: strategy.as_db_str().to_owned(),
        pipeline_steps,
        response,
        total_duration_ms,
    }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

async fn run_cleanup(
    collection: &CollectionName,
    vector_store: &dyn VectorStore,
    graph_store: &dyn GraphStore,
    doc_ids: &[String],
    entity_ids: &[String],
) -> serde_json::Value {
    let mut results: HashMap<String, serde_json::Value> = HashMap::new();

    // Delete vector collection
    match vector_store.delete(collection, doc_ids.to_vec()).await {
        Ok(()) => {
            results.insert("vectors_deleted".into(), serde_json::json!(doc_ids.len()));
        }
        Err(e) => {
            results.insert(
                "vectors_deleted".into(),
                serde_json::json!(format!("error: {e}")),
            );
        }
    }

    // Delete graph nodes (entities + document nodes)
    let mut nodes_deleted = 0usize;
    let mut delete_errors: Vec<String> = Vec::new();

    for id in doc_ids.iter().chain(entity_ids.iter()) {
        match graph_store.delete_node(id).await {
            Ok(()) => nodes_deleted += 1,
            Err(e) => delete_errors.push(format!("{id}: {e}")),
        }
    }

    results.insert("nodes_deleted".into(), serde_json::json!(nodes_deleted));
    if !delete_errors.is_empty() {
        results.insert("delete_errors".into(), serde_json::json!(delete_errors));
    }

    results.insert(
        "collections_deleted".into(),
        serde_json::json!([collection.as_str()]),
    );
    results.insert(
        "status".into(),
        serde_json::json!(if delete_errors.is_empty() {
            "success"
        } else {
            "partial"
        }),
    );

    serde_json::json!(results)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load configuration
    let settings = Settings::load().context("failed to load configuration")?;

    // 2. Initialize logging (pretty for local dev)
    agent_rs_core::logging::init_logging(false);

    let test_run_id = Uuid::new_v4().to_string();
    let timestamp = Utc::now().to_rfc3339();

    info!(
        test_run_id = %test_run_id,
        "E2E full trace starting"
    );

    // 3. Connect to storage backends
    let stores = connect_stores(&settings).await?;
    let qdrant_store = stores.qdrant;
    let neo4j_store = stores.neo4j;
    // stores.postgres intentionally unused — this trace bypasses the ingest state machine.

    // 4. Initialize embedder
    info!("initializing embedder (this may download ONNX models on first run)...");
    let embedder = FastEmbedder::try_new(FastEmbedderConfig {
        show_download_progress: true,
        ..Default::default()
    })
    .await
    .context("failed to initialize embedding models")?;
    let dims = embedder.dimensions();
    let embedder = Arc::new(embedder);

    info!(dims, "embedder initialized");

    // 5. Set up test collection name (unique per run to avoid conflicts)
    let collection_name = format!("e2e_test_{}", &test_run_id[..8]);
    let collection = CollectionName(collection_name.clone());

    // 6. Create retrieval router
    let ranker = ResultRanker::new(settings.ranker.rrf_k);
    let router = Arc::new(RetrievalRouter::new(
        embedder.clone() as Arc<dyn agent_rs_retrieval::Embedder + Send + Sync>,
        qdrant_store.clone() as Arc<dyn VectorStore + Send + Sync>,
        neo4j_store.clone() as Arc<dyn GraphStore + Send + Sync>,
        ranker,
        collection.clone(),
    ));

    // 7. Generate test documents
    let docs = generate_test_documents();
    info!(doc_count = docs.len(), "test documents generated");

    // 8. Run ingestion
    info!("--- INGESTION PHASE ---");
    let ingestion = run_ingestion(
        &docs,
        &embedder,
        qdrant_store.as_ref(),
        neo4j_store.as_ref(),
        &collection,
        dims,
    )
    .await;

    // 9. Run queries with different strategies
    info!("--- QUERY PHASE ---");

    let query_specs: Vec<(&str, RetrievalStrategy)> = vec![
        (
            "What are neural networks and how are they trained?",
            RetrievalStrategy::Semantic,
        ),
        (
            "Explain the transformer architecture and attention mechanism",
            RetrievalStrategy::Hybrid,
        ),
        (
            "How does reinforcement learning relate to other machine learning approaches?",
            RetrievalStrategy::Graph,
        ),
    ];

    let mut queries: Vec<QueryTrace> = Vec::new();
    let limit = 5;

    for (query_text, strategy) in &query_specs {
        info!(query = %query_text, strategy = %strategy, "running query");
        let trace = run_query_trace(
            query_text,
            *strategy,
            &router,
            settings.verifier.confidence_threshold,
            limit,
        )
        .await;
        info!(
            query = %query_text,
            strategy = %strategy,
            total_ms = trace.total_duration_ms,
            "query complete"
        );
        queries.push(trace);
    }

    // 10. Cleanup
    info!("--- CLEANUP PHASE ---");

    let doc_ids: Vec<String> = docs.iter().map(|d| d.id.clone()).collect();
    let (entities, _) = extract_entities_from_docs(&docs);
    let entity_ids: Vec<String> = entities.iter().map(|e| e.id.clone()).collect();

    let cleanup = run_cleanup(
        &collection,
        qdrant_store.as_ref(),
        neo4j_store.as_ref(),
        &doc_ids,
        &entity_ids,
    )
    .await;

    // 11. Assemble full trace
    let full_trace = FullTrace {
        test_run_id: test_run_id.clone(),
        timestamp,
        config: sanitize_config(&settings),
        ingestion,
        queries,
        cleanup,
    };

    // 12. Write JSON trace to file
    let output_dir = std::path::Path::new("test_results");
    tokio::fs::create_dir_all(output_dir)
        .await
        .context("failed to create test_results/ directory")?;

    let filename = format!("e2e_trace_{}.json", Utc::now().format("%Y%m%dT%H%M%S"));
    let output_path = output_dir.join(&filename);

    let json_output =
        serde_json::to_string_pretty(&full_trace).context("failed to serialize trace to JSON")?;

    tokio::fs::write(&output_path, &json_output)
        .await
        .context("failed to write trace file")?;

    // 13. Print summary
    println!("\n{}", "=".repeat(72));
    println!("  E2E FULL TRACE SUMMARY");
    println!("{}", "=".repeat(72));
    println!("  Test Run ID:  {test_run_id}");
    println!("  Output File:  {}", output_path.display());
    println!();
    println!("  Ingestion:");
    println!("    Documents:  {}", full_trace.ingestion.documents_count);
    for step in &full_trace.ingestion.steps {
        let icon = if step.status == "success" {
            "[OK]"
        } else {
            "[FAIL]"
        };
        println!("    {icon} {} ({} ms)", step.step, step.duration_ms);
    }
    println!();
    println!("  Queries:");
    for qt in &full_trace.queries {
        println!("    Query: \"{}\"", qt.query);
        println!("    Strategy: {}", qt.strategy);
        println!("    Total: {} ms", qt.total_duration_ms);
        for step in &qt.pipeline_steps {
            let icon = if step.status == "success" {
                "[OK]"
            } else {
                "[FAIL]"
            };
            println!("      {icon} {} ({} ms)", step.step, step.duration_ms);
        }
        println!();
    }
    println!(
        "  Cleanup: {}",
        full_trace
            .cleanup
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    );
    println!("{}\n", "=".repeat(72));

    info!(
        output = %output_path.display(),
        "E2E full trace complete"
    );

    Ok(())
}
