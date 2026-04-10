//! Qdrant vector store implementation.
//!
//! Provides [`QdrantStore`], a gRPC-backed implementation of [`VectorStore`]
//! that supports:
//! - Named-vector collections (dense + optional sparse).
//! - Dense-only similarity search via the `query` API.
//! - Hybrid (dense + sparse) search via prefetch + RRF fusion.
//! - Point upsert with both dense and optional sparse vectors.
//! - Point deletion by string ID.
//! - Collection existence checks.

mod collections;

use std::collections::HashMap;

use agent_rs_core::{CollectionName, Embedding, StoreError};
use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    DeletePointsBuilder, DenseVector, NamedVectors, PointId, PointStruct, PointsIdsList,
    PrefetchQueryBuilder, QueryPointsBuilder, UpsertPointsBuilder, Vector, Vectors,
};
use tracing::instrument;

use crate::traits::VectorStore;
use crate::types::{CollectionConfig, SearchResult, SparseVector, VectorPoint};
use agent_rs_core::Score;

// ---- QdrantStore ----

/// Qdrant-backed implementation of [`VectorStore`].
///
/// Connects via gRPC using [`agent_rs_core::config::QdrantConfig`].
/// All methods are fully instrumented with `tracing`.
pub struct QdrantStore {
    client: Qdrant,
}

impl QdrantStore {
    /// Construct a new [`QdrantStore`] from a [`QdrantConfig`].
    ///
    /// Establishes a lazy gRPC connection; errors surface on the first request.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ConnectionFailed`] if the client cannot be built.
    pub fn new(config: &agent_rs_core::config::QdrantConfig) -> Result<Self, StoreError> {
        let mut builder = Qdrant::from_url(&config.grpc_url());

        if !config.api_key.is_empty() {
            builder = builder.api_key(config.api_key.clone());
        }

        let client = builder.build().map_err(|e| StoreError::ConnectionFailed {
            service: "qdrant".into(),
            reason: e.to_string(),
        })?;

        Ok(Self { client })
    }
}

// ---- VectorStore impl ----

#[async_trait]
impl VectorStore for QdrantStore {
    /// Create a Qdrant collection with named dense (and optional sparse) vectors.
    #[instrument(skip(self), fields(collection = %name))]
    async fn create_collection(
        &self,
        name: &CollectionName,
        config: &CollectionConfig,
    ) -> Result<(), StoreError> {
        collections::create_named_vector_collection(&self.client, name.as_str(), config)
            .await
            .map_err(StoreError::from)
    }

    /// Upsert vector points. Each point may carry a dense embedding and an
    /// optional sparse vector for hybrid search.
    #[instrument(skip(self, points), fields(collection = %collection, count = points.len()))]
    async fn upsert(
        &self,
        collection: &CollectionName,
        points: Vec<VectorPoint>,
    ) -> Result<(), StoreError> {
        let structs: Vec<PointStruct> = points
            .into_iter()
            .map(|p| {
                let mut vectors: HashMap<String, Vector> = HashMap::new();

                // Dense vector under the named vector key "dense".
                // `DenseVector` implements `Into<Vector>` (non-deprecated path).
                let dense_vector: Vector = DenseVector {
                    data: p.embedding.0,
                }
                .into();
                vectors.insert("dense".into(), dense_vector);

                // Sparse vector under "sparse" when present.
                // `qdrant_client::qdrant::SparseVector` implements `Into<Vector>`.
                if let Some(sv) = p.sparse_vector {
                    let sparse_vector: Vector = qdrant_client::qdrant::SparseVector {
                        values: sv.values,
                        indices: sv.indices,
                    }
                    .into();
                    vectors.insert("sparse".into(), sparse_vector);
                }

                let payload = payload_from_json(p.payload);

                // Qdrant requires point IDs to be UUID or u64.
                // If the ID is a valid UUID, use it directly; otherwise
                // generate a deterministic UUID v5 from the string.
                let point_id = match uuid::Uuid::parse_str(&p.id) {
                    Ok(uuid) => PointId::from(uuid.to_string()),
                    Err(_) => {
                        let ns = uuid::Uuid::NAMESPACE_OID;
                        let uuid = uuid::Uuid::new_v5(&ns, p.id.as_bytes());
                        PointId::from(uuid.to_string())
                    }
                };

                PointStruct {
                    id: Some(point_id),
                    vectors: Some(Vectors {
                        vectors_options: Some(
                            qdrant_client::qdrant::vectors::VectorsOptions::Vectors(NamedVectors {
                                vectors,
                            }),
                        ),
                    }),
                    payload,
                }
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection.as_str(), structs).wait(true))
            .await
            .map_err(StoreError::from)?;

        Ok(())
    }

    /// Dense vector similarity search using the "dense" named vector.
    ///
    /// Uses the modern `query` API (not the deprecated `search_points`).
    #[instrument(skip(self, query), fields(collection = %collection, limit = limit))]
    async fn search(
        &self,
        collection: &CollectionName,
        query: &Embedding,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StoreError> {
        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection.as_str())
                    .query(qdrant_client::qdrant::Query::from(query.0.clone()))
                    .using("dense")
                    .limit(limit as u64)
                    .with_payload(true),
            )
            .await
            .map_err(StoreError::from)?;

        response
            .result
            .into_iter()
            .map(scored_point_to_result)
            .collect()
    }

    /// Hybrid search: dense + sparse prefetch fused via RRF.
    ///
    /// Two prefetch sub-queries are issued (one per vector type), then the
    /// top-level query merges them with Reciprocal Rank Fusion.
    #[instrument(skip(self, dense, sparse), fields(collection = %collection, limit = limit))]
    async fn search_hybrid(
        &self,
        collection: &CollectionName,
        dense: &Embedding,
        sparse: &SparseVector,
        limit: usize,
    ) -> Result<Vec<SearchResult>, StoreError> {
        let prefetch_dense = PrefetchQueryBuilder::default()
            .query(qdrant_client::qdrant::Query::from(dense.0.clone()))
            .using("dense")
            .limit(limit as u64)
            .build();

        // Sparse query uses Vec<(index, value)> — the supported From impl.
        let sparse_pairs: Vec<(u32, f32)> = sparse
            .indices
            .iter()
            .copied()
            .zip(sparse.values.iter().copied())
            .collect();

        let prefetch_sparse = PrefetchQueryBuilder::default()
            .query(qdrant_client::qdrant::Query::from(sparse_pairs))
            .using("sparse")
            .limit(limit as u64)
            .build();

        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection.as_str())
                    .prefetch(vec![prefetch_dense, prefetch_sparse])
                    .query(qdrant_client::qdrant::Query::new_fusion(
                        qdrant_client::qdrant::Fusion::Rrf,
                    ))
                    .limit(limit as u64)
                    .with_payload(true),
            )
            .await
            .map_err(StoreError::from)?;

        response
            .result
            .into_iter()
            .map(scored_point_to_result)
            .collect()
    }

    /// Delete points by string IDs.
    #[instrument(skip(self, ids), fields(collection = %collection, count = ids.len()))]
    async fn delete(
        &self,
        collection: &CollectionName,
        ids: Vec<String>,
    ) -> Result<(), StoreError> {
        let point_ids: Vec<PointId> = ids
            .into_iter()
            .map(|id| PointId::from(id.as_str()))
            .collect();

        self.client
            .delete_points(
                DeletePointsBuilder::new(collection.as_str())
                    .points(PointsIdsList { ids: point_ids })
                    .wait(true),
            )
            .await
            .map_err(StoreError::from)?;

        Ok(())
    }

    /// Check whether a Qdrant collection exists.
    #[instrument(skip(self), fields(collection = %name))]
    async fn collection_exists(&self, name: &CollectionName) -> Result<bool, StoreError> {
        self.client
            .collection_exists(name.as_str())
            .await
            .map_err(StoreError::from)
    }
}

// ---- Private helpers ----

/// Convert a `serde_json::Value` object into a Qdrant payload map.
///
/// Non-object JSON values are wrapped under the key `"value"` so that the
/// payload is always a valid `HashMap<String, qdrant_client::qdrant::Value>`.
fn payload_from_json(value: serde_json::Value) -> HashMap<String, qdrant_client::qdrant::Value> {
    match value {
        serde_json::Value::Object(map) => map
            .into_iter()
            .map(|(k, v)| (k, json_to_qdrant_value(v)))
            .collect(),
        other => {
            let mut m = HashMap::new();
            m.insert("value".into(), json_to_qdrant_value(other));
            m
        }
    }
}

/// Recursively convert a `serde_json::Value` into a `qdrant_client::qdrant::Value`.
fn json_to_qdrant_value(value: serde_json::Value) -> qdrant_client::qdrant::Value {
    use qdrant_client::qdrant::value::Kind;
    use qdrant_client::qdrant::{ListValue, Struct, Value};

    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Kind::IntegerValue(i)
            } else {
                Kind::DoubleValue(n.as_f64().unwrap_or_else(|| {
                    tracing::warn!(number = %n, "JSON number cannot be represented as f64, using 0.0");
                    0.0
                }))
            }
        }
        serde_json::Value::String(s) => Kind::StringValue(s),
        serde_json::Value::Array(arr) => Kind::ListValue(ListValue {
            values: arr.into_iter().map(json_to_qdrant_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(Struct {
            fields: map
                .into_iter()
                .map(|(k, v)| (k, json_to_qdrant_value(v)))
                .collect(),
        }),
    };

    Value { kind: Some(kind) }
}

/// Convert a scored point from the Qdrant response into a domain [`SearchResult`].
///
/// Returns `Err` if the point has no valid ID — never produces empty-ID results.
fn scored_point_to_result(
    point: qdrant_client::qdrant::ScoredPoint,
) -> Result<SearchResult, StoreError> {
    let id = match point.id {
        Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)),
        }) => s,
        Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)),
        }) => n.to_string(),
        other => {
            return Err(StoreError::InvalidData(format!(
                "scored point has no valid ID: {other:?}"
            )));
        }
    };

    let payload = serde_json::Value::Object(
        point
            .payload
            .into_iter()
            .map(|(k, v)| (k, qdrant_value_to_json(v)))
            .collect(),
    );

    Ok(SearchResult {
        id,
        score: Score(point.score as f64),
        payload,
    })
}

/// Recursively convert a `qdrant_client::qdrant::Value` back into `serde_json::Value`.
fn qdrant_value_to_json(value: qdrant_client::qdrant::Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;

    match value.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(b),
        Some(Kind::IntegerValue(i)) => serde_json::Value::Number(i.into()),
        Some(Kind::DoubleValue(d)) => serde_json::json!(d),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s),
        Some(Kind::ListValue(lst)) => {
            serde_json::Value::Array(lst.values.into_iter().map(qdrant_value_to_json).collect())
        }
        Some(Kind::StructValue(st)) => serde_json::Value::Object(
            st.fields
                .into_iter()
                .map(|(k, v)| (k, qdrant_value_to_json(v)))
                .collect(),
        ),
    }
}
