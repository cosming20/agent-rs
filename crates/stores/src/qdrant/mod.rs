//! Qdrant vector store implementation.
//!
//! Provides [`QdrantStore`], a gRPC-backed implementation of [`VectorStore`]
//! that supports:
//! - Named-vector collections (dense + optional sparse).
//! - User-scoped dense vector search (mandatory `user_id` filter).
//! - Point upsert with both dense and optional sparse vectors.
//! - Point deletion by ID and by filter.
//! - Collection existence checks.
//!
//! Unscoped search methods (`search`, `search_hybrid`, `search_with_filter`,
//! `get_by_ids`) are intentionally absent — multi-tenancy requires every
//! read path to carry a `user_id` scope.

mod collections;
pub mod section_filter;

use std::collections::HashMap;

use agent_rs_core::{CollectionName, Embedding, StoreError, UserId};
use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    DeletePointsBuilder, DenseVector, Document, GetPointsBuilder, NamedVectors, PointId,
    PointStruct, PointsIdsList, QueryPointsBuilder, ScrollPointsBuilder, UpsertPointsBuilder,
    Vector, Vectors,
};
use tracing::instrument;

use crate::traits::VectorStore;
use crate::types::{CollectionConfig, FilterSpec, SearchResult, VectorPoint};
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

    /// Upsert vector points. Each point carries a dense embedding and,
    /// when `text` is non-empty, a server-side BM25 sparse vector inferred
    /// by Qdrant via `Document { model: "qdrant/bm25" }`.
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
                    data: p.embedding.as_slice().to_vec(),
                }
                .into();
                vectors.insert("dense".into(), dense_vector);

                // Sparse vector via server-side BM25 inference.
                //
                // When the point carries raw chunk text, we let Qdrant infer
                // the sparse vector using the `qdrant/bm25` model. This
                // requires the collection's sparse vector to be configured
                // with `Modifier::Idf` (set in `create_named_vector_collection`).
                // No local SPLADE/fastembed model is needed.
                if !p.text.is_empty() {
                    let sparse_vector: Vector = Document::new(&p.text, "qdrant/bm25").into();
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

    /// Delete all points matching a [`FilterSpec`] from a collection.
    ///
    /// Converts `FilterSpec` to a Qdrant `Filter` and calls the delete-by-filter
    /// API, which returns the count of deleted points.
    ///
    /// # Filter mapping
    ///
    /// - `FilterSpec::Eq { field, value }` → `Condition::matches(field, value)` in a `must` list.
    /// - `FilterSpec::And(specs)` → each spec maps to a `must` condition.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    #[instrument(skip(self, filter), fields(collection = %collection))]
    async fn delete_by_filter(
        &self,
        collection: &CollectionName,
        filter: FilterSpec,
    ) -> Result<u64, StoreError> {
        let qdrant_filter = filter_spec_to_qdrant(filter)?;

        // Pre-count: Qdrant's delete API does not return the number of
        // deleted points, so we count matching points BEFORE deletion.
        // This is a race-free approximation (no concurrent deletes expected
        // for a specific ingest_run_id).
        let count = match self
            .client
            .count(
                qdrant_client::qdrant::CountPointsBuilder::new(collection.as_str())
                    .filter(qdrant_filter.clone())
                    .exact(true),
            )
            .await
        {
            Ok(r) => r.result.map(|c| c.count).unwrap_or(0),
            Err(e) => {
                tracing::warn!(
                    collection = %collection,
                    error = %e,
                    "pre-delete count failed; proceeding with delete anyway"
                );
                0
            }
        };

        // Delete all matching points.
        self.client
            .delete_points(
                DeletePointsBuilder::new(collection.as_str())
                    .points(qdrant_filter)
                    .wait(true),
            )
            .await
            .map_err(StoreError::from)?;

        tracing::info!(
            collection = %collection,
            deleted = count,
            "delete_by_filter complete"
        );
        Ok(count)
    }

    /// Check whether a Qdrant collection exists.
    #[instrument(skip(self), fields(collection = %name))]
    async fn collection_exists(&self, name: &CollectionName) -> Result<bool, StoreError> {
        self.client
            .collection_exists(name.as_str())
            .await
            .map_err(StoreError::from)
    }

    /// Dense vector similarity search scoped to a specific user.
    ///
    /// Overrides the trait default to use native Qdrant filter construction
    /// via [`crate::qdrant::section_filter::build_user_filter`] and
    /// [`crate::qdrant::section_filter::merge_user_filter`], which produces
    /// an optimised `Filter` proto instead of the JSON-shim path.
    ///
    /// `extra_filter_json` conditions are converted to Qdrant `must` conditions
    /// and merged with the mandatory user scope filter.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    #[instrument(skip(self, query, extra_filter_json), fields(collection = %collection, user_id = %user_id, limit = limit))]
    async fn search_for_user(
        &self,
        collection: &CollectionName,
        user_id: &UserId,
        query: &Embedding,
        limit: usize,
        extra_filter_json: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>, StoreError> {
        use crate::qdrant::section_filter::{build_user_filter, merge_user_filter};

        let user_scope = build_user_filter(user_id.as_str(), None, None);
        let extra_qdrant = extra_filter_json.map(json_to_qdrant_filter).transpose()?;
        let merged = merge_user_filter(user_scope, extra_qdrant);

        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection.as_str())
                    .query(qdrant_client::qdrant::Query::from(
                        query.as_slice().to_vec(),
                    ))
                    .using("dense")
                    .limit(limit as u64)
                    .filter(merged)
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

    /// Fetch payload-only projections scoped to a user.
    ///
    /// Delegates to [`QdrantStore::get_payloads_by_ids`] and then drops any
    /// point whose payload `user_id` does not match — defence-in-depth against
    /// cross-tenant leakage even when chunk IDs are globally unique.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    ///
    /// # Panics
    ///
    /// Panics when `user_id` is empty.
    #[instrument(skip(self, ids), fields(collection = %collection, user_id = %user_id, n_ids = ids.len()))]
    async fn retrieve_chunks_for_user(
        &self,
        collection: &CollectionName,
        user_id: &UserId,
        ids: &[String],
    ) -> Result<Vec<crate::types::PointPayload>, StoreError> {
        assert!(
            !user_id.as_str().is_empty(),
            "retrieve_chunks_for_user: empty user_id is forbidden"
        );

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let points = self.get_payloads_by_ids(collection, ids).await?;

        let expected = user_id.as_str();
        Ok(points
            .into_iter()
            .filter(|p| p.payload.get("user_id").and_then(|v| v.as_str()) == Some(expected))
            .collect())
    }
}

// ---- QdrantStore inherent methods ----

impl QdrantStore {
    /// Dense vector similarity search scoped to a specific user.
    ///
    /// Builds a mandatory `user_id` filter via [`build_user_filter`] and
    /// merges it with any caller-supplied `extra_filter`. This is the
    /// preferred search entry point for multi-tenant contexts.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    #[instrument(skip(self, query, extra_filter), fields(collection = %collection, user_id = %user_id, limit = limit))]
    pub async fn search_for_user(
        &self,
        collection: &CollectionName,
        user_id: &UserId,
        query: &Embedding,
        limit: usize,
        extra_filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Vec<crate::types::SearchResult>, StoreError> {
        use crate::qdrant::section_filter::{build_user_filter, merge_user_filter};

        let user_scope = build_user_filter(user_id.as_str(), None, None);
        let merged = merge_user_filter(user_scope, extra_filter);

        let response = self
            .client
            .query(
                QueryPointsBuilder::new(collection.as_str())
                    .query(qdrant_client::qdrant::Query::from(
                        query.as_slice().to_vec(),
                    ))
                    .using("dense")
                    .limit(limit as u64)
                    .filter(merged)
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

    /// Fetch payload-only projections for a list of chunk ids, enforcing
    /// user-scope as defence-in-depth.
    ///
    /// Fetches points by ID from Qdrant and then drops any point whose
    /// payload `user_id` does not match the requested `user_id`. IDs are
    /// globally unique so a fetch-by-id normally returns only the right
    /// point, but this secondary check prevents cross-tenant reads if a
    /// caller ever supplies a chunk ID belonging to a different tenant.
    ///
    /// Ids that do not resolve to a stored point are silently dropped; the
    /// caller sees only hits.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    ///
    /// # Panics
    ///
    /// Panics if `user_id` is empty — empty user scope is forbidden.
    #[instrument(skip(self, ids), fields(collection, user_id = %user_id, n_ids = ids.len()))]
    pub async fn retrieve_chunks_by_ids(
        &self,
        collection: &str,
        user_id: &UserId,
        ids: &[String],
    ) -> Result<Vec<crate::types::PointPayload>, StoreError> {
        assert!(
            !user_id.as_str().is_empty(),
            "retrieve_chunks_by_ids: empty user_id is forbidden"
        );

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let coll = agent_rs_core::CollectionName::from_string(collection.to_owned());
        let points = self.get_payloads_by_ids(&coll, ids).await?;

        // Defence-in-depth: drop any point whose payload user_id does not match.
        let expected = user_id.as_str();
        let filtered: Vec<crate::types::PointPayload> = points
            .into_iter()
            .filter(|p| p.payload.get("user_id").and_then(|v| v.as_str()) == Some(expected))
            .collect();

        Ok(filtered)
    }

    /// Fetch payload-only projections for a list of point IDs.
    ///
    /// Internal helper used by [`retrieve_chunks_by_ids`]. Ids that don't
    /// resolve to a stored point are silently dropped.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    #[instrument(skip(self, ids), fields(collection = %collection, n_ids = ids.len()))]
    async fn get_payloads_by_ids(
        &self,
        collection: &CollectionName,
        ids: &[String],
    ) -> Result<Vec<crate::types::PointPayload>, StoreError> {
        let point_ids: Vec<PointId> = ids
            .iter()
            .map(|id| match uuid::Uuid::parse_str(id) {
                Ok(uuid) => PointId::from(uuid.to_string()),
                Err(_) => {
                    let ns = uuid::Uuid::NAMESPACE_OID;
                    let uuid = uuid::Uuid::new_v5(&ns, id.as_bytes());
                    PointId::from(uuid.to_string())
                }
            })
            .collect();

        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(collection.as_str(), point_ids)
                    .with_payload(true)
                    .with_vectors(false),
            )
            .await
            .map_err(StoreError::from)?;

        let mut out: Vec<crate::types::PointPayload> = Vec::with_capacity(response.result.len());
        for point in response.result {
            let id_str = match point.id {
                Some(qdrant_client::qdrant::PointId {
                    point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s)),
                }) => s,
                Some(qdrant_client::qdrant::PointId {
                    point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)),
                }) => n.to_string(),
                _ => {
                    tracing::warn!(collection = %collection, "qdrant returned a point with no id");
                    continue;
                }
            };
            let payload_json = serde_json::Value::Object(
                point
                    .payload
                    .into_iter()
                    .map(|(k, v)| (k, qdrant_value_to_json(v)))
                    .collect(),
            );
            out.push(crate::types::PointPayload {
                id: id_str,
                payload: payload_json,
            });
        }
        Ok(out)
    }

    /// Return the `summary_text` of the first stored summary whose payload
    /// matches both `user_id` and `section_id`. Returns `Ok(None)` when no
    /// matching point exists.
    ///
    /// Uses the scroll API (no vector required) with a mandatory user-scope
    /// filter ANDed with the `section_id` condition, limiting to 1 result.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] on Qdrant gRPC failure.
    #[instrument(skip(self), fields(collection = %collection, user_id = %user_id, section_id = %section_id))]
    pub async fn first_summary_by_section(
        &self,
        collection: &str,
        user_id: &UserId,
        section_id: &str,
    ) -> Result<Option<String>, StoreError> {
        use crate::qdrant::section_filter::build_user_filter;

        let filter = build_user_filter(user_id.as_str(), None, Some(section_id));

        let response = self
            .client
            .scroll(
                ScrollPointsBuilder::new(collection)
                    .filter(filter)
                    .limit(1u32)
                    .with_payload(true)
                    .with_vectors(false),
            )
            .await
            .map_err(StoreError::from)?;

        let summary = response.result.into_iter().next().and_then(|point| {
            point.payload.get("summary_text").and_then(|v| {
                use qdrant_client::qdrant::value::Kind;
                match v.kind.as_ref() {
                    Some(Kind::StringValue(s)) => Some(s.clone()),
                    _ => None,
                }
            })
        });

        Ok(summary)
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
        score: Score::new(point.score as f64),
        payload,
    })
}

/// Convert a JSON filter specification into a Qdrant `Filter`.
///
/// Accepts either:
/// - A JSON object with `"must"` array of conditions.
/// - A single condition object with `"key"` and `"match"` fields.
///
/// Each condition must have:
/// - `"key"`: the payload field name (string)
/// - `"match"`: `{"value": <string|integer|bool>}`
fn json_to_qdrant_filter(
    json: serde_json::Value,
) -> Result<qdrant_client::qdrant::Filter, StoreError> {
    use qdrant_client::qdrant::{Condition, Filter};

    let conditions_json = match &json {
        serde_json::Value::Object(map) if map.contains_key("must") => match map.get("must") {
            Some(serde_json::Value::Array(arr)) => arr.clone(),
            _ => {
                return Err(StoreError::InvalidData(
                    "filter 'must' field must be an array".into(),
                ));
            }
        },
        serde_json::Value::Object(_) => vec![json],
        _ => {
            return Err(StoreError::InvalidData(
                "filter must be a JSON object".into(),
            ));
        }
    };

    let mut must = Vec::with_capacity(conditions_json.len());

    for cond in conditions_json {
        let key = cond
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StoreError::InvalidData("filter condition missing 'key'".into()))?
            .to_string();

        // Dispatch on condition type: match (exact or text) vs range.
        if let Some(match_obj) = cond.get("match") {
            if let Some(text_val) = match_obj.get("text").and_then(|v| v.as_str()) {
                // Full-text tokenized substring match (e.g., "15" matches
                // "Section 15 - Overview").  Uses Qdrant's MatchText.
                must.push(Condition::matches_text(key, text_val));
            } else if let Some(match_value) = match_obj.get("value") {
                // Exact value match.
                let condition = match match_value {
                    serde_json::Value::String(s) => Condition::matches(key, s.clone()),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            Condition::matches(key, i)
                        } else {
                            return Err(StoreError::InvalidData(format!(
                                "filter match value must be string, integer, or bool; got: {n}"
                            )));
                        }
                    }
                    serde_json::Value::Bool(b) => Condition::matches(key, *b),
                    other => {
                        return Err(StoreError::InvalidData(format!(
                            "unsupported filter match value type: {other}"
                        )));
                    }
                };
                must.push(condition);
            } else {
                return Err(StoreError::InvalidData(
                    "filter match object must contain 'value' or 'text'".into(),
                ));
            }
        } else if let Some(range_obj) = cond.get("range") {
            // Numeric range condition.
            let mut range = qdrant_client::qdrant::Range::default();
            if let Some(gte) = range_obj.get("gte").and_then(|v| v.as_f64()) {
                range.gte = Some(gte);
            }
            if let Some(lte) = range_obj.get("lte").and_then(|v| v.as_f64()) {
                range.lte = Some(lte);
            }
            must.push(Condition::range(key, range));
        } else {
            return Err(StoreError::InvalidData(format!(
                "filter condition for key '{key}' has neither 'match' nor 'range'"
            )));
        }
    }

    Ok(Filter {
        must,
        ..Default::default()
    })
}

/// Convert a [`FilterSpec`] into a Qdrant [`Filter`].
///
/// # Mapping
///
/// - [`FilterSpec::Eq`] → single `Condition::matches` in the `must` list.
/// - [`FilterSpec::And`] → each sub-spec is recursively flattened into `must`.
///
/// # Errors
///
/// Returns [`StoreError::InvalidData`] if a nested `FilterSpec::And` is empty.
fn filter_spec_to_qdrant(spec: FilterSpec) -> Result<qdrant_client::qdrant::Filter, StoreError> {
    use qdrant_client::qdrant::{Condition, Filter};

    fn collect_conditions(spec: FilterSpec, out: &mut Vec<Condition>) -> Result<(), StoreError> {
        match spec {
            FilterSpec::Eq { field, value } => {
                out.push(Condition::matches(field, value));
                Ok(())
            }
            FilterSpec::And(specs) => {
                for s in specs {
                    collect_conditions(s, out)?;
                }
                Ok(())
            }
        }
    }

    let mut conditions = Vec::new();
    collect_conditions(spec, &mut conditions)?;

    Ok(Filter {
        must: conditions,
        ..Default::default()
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
