//! Neo4j graph store implementation.
//!
//! [`Neo4jStore`] implements [`GraphStore`] via the `neo4rs` async driver.
//! The connection pool is managed internally by `neo4rs::Graph`, which is
//! cheap to clone and share across tasks.

pub mod coverage;
mod queries;
pub mod structural;

use agent_rs_core::{StoreError, config::Neo4jConfig};
use async_trait::async_trait;
use neo4rs::{Graph, query};
use tracing::instrument;
use uuid::Uuid;

use crate::{
    traits::GraphStore,
    types::{GraphEdge, GraphNode, GraphPath, NodeLabel, RelationType},
};

// ---- Neo4jStore ----

/// Neo4j-backed implementation of [`GraphStore`].
///
/// Constructed with [`Neo4jStore::connect`], which establishes the
/// connection pool using `neo4rs::Graph::new`. The `Graph` handle is
/// `Clone + Send + Sync`, so `Neo4jStore` can be shared freely across
/// async tasks.
#[derive(Clone)]
pub struct Neo4jStore {
    graph: Graph,
}

impl Neo4jStore {
    /// Connect to Neo4j using the provided [`Neo4jConfig`].
    ///
    /// `Neo4jConfig::endpoint()` returns `host:port` — neo4rs takes this
    /// directly without a `bolt://` prefix.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ConnectionFailed`] if the driver cannot
    /// establish the initial connection.
    #[instrument(name = "neo4j_store.connect", skip(config), fields(endpoint = %config.endpoint()))]
    pub async fn connect(config: &Neo4jConfig) -> Result<Self, StoreError> {
        let endpoint = config.endpoint();
        let graph = Graph::new(&endpoint, &config.user, &config.password)
            .await
            .map_err(|e| StoreError::ConnectionFailed {
                service: format!("neo4j@{endpoint}"),
                reason: e.to_string(),
            })?;
        tracing::info!("connected to neo4j at {}", endpoint);
        Ok(Self { graph })
    }
}

// ---- GraphStore impl ----

#[async_trait]
impl GraphStore for Neo4jStore {
    /// Add a single node, using MERGE on `id` for idempotency.
    #[instrument(name = "neo4j_store.add_node", skip(self, node), fields(node_id = %node.id))]
    async fn add_node(&self, node: &GraphNode) -> Result<(), StoreError> {
        let q = queries::upsert_node(node).map_err(StoreError::from)?;
        self.graph.run(q).await.map_err(StoreError::from)?;
        Ok(())
    }

    /// Add multiple nodes in a batch, grouped by label and UNWIND'd.
    ///
    /// Reduces O(n) round-trips to O(labels) by batching nodes with the
    /// same label into a single `UNWIND $rows` Cypher statement.
    #[instrument(name = "neo4j_store.add_nodes", skip(self, nodes), fields(count = nodes.len()))]
    async fn add_nodes(&self, nodes: &[GraphNode]) -> Result<(), StoreError> {
        use std::collections::HashMap;

        if nodes.is_empty() {
            return Ok(());
        }

        // Group nodes by label for batched UNWIND.
        let mut by_label: HashMap<&str, Vec<serde_json::Value>> = HashMap::new();
        for node in nodes {
            let row = serde_json::json!({
                "id": node.id,
                "props": node.properties,
            });
            by_label.entry(node.label.as_str()).or_default().push(row);
        }

        for (label, rows) in &by_label {
            let cypher =
                format!("UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props");
            let batch_val = serde_json::Value::Array(rows.clone());
            let bolt = neo4rs::BoltType::try_from(batch_val)
                .map_err(|e| StoreError::Serialization(format!("node batch for {label}: {e}")))?;
            self.graph
                .run(neo4rs::query(&cypher).param("rows", bolt))
                .await
                .map_err(StoreError::from)?;
        }

        tracing::debug!(
            labels = by_label.len(),
            total = nodes.len(),
            "batch upserted nodes"
        );
        Ok(())
    }

    /// Add a directed edge. Nodes must already exist.
    #[instrument(
        name = "neo4j_store.add_edge",
        skip(self, edge),
        fields(src = %edge.source_id, dst = %edge.target_id, rel = %edge.relation_type)
    )]
    async fn add_edge(&self, edge: &GraphEdge) -> Result<(), StoreError> {
        let q = queries::create_edge(edge).map_err(StoreError::from)?;
        self.graph.run(q).await.map_err(StoreError::from)?;
        Ok(())
    }

    /// Add multiple edges in a batch, grouped by relationship type and UNWIND'd.
    ///
    /// Reduces O(n) round-trips to O(rel_types) by batching edges with the
    /// same relationship type into a single `UNWIND $rows` Cypher statement.
    #[instrument(name = "neo4j_store.add_edges", skip(self, edges), fields(count = edges.len()))]
    async fn add_edges(&self, edges: &[GraphEdge]) -> Result<(), StoreError> {
        use std::collections::HashMap;

        if edges.is_empty() {
            return Ok(());
        }

        // Group edges by relationship type for batched UNWIND.
        let mut by_rel: HashMap<&str, Vec<serde_json::Value>> = HashMap::new();
        for edge in edges {
            let row = serde_json::json!({
                "src": edge.source_id,
                "dst": edge.target_id,
                "props": edge.properties,
            });
            by_rel
                .entry(edge.relation_type.as_str())
                .or_default()
                .push(row);
        }

        for (rel_type, rows) in &by_rel {
            let cypher = format!(
                "UNWIND $rows AS row \
                 MATCH (a {{id: row.src}}), (b {{id: row.dst}}) \
                 MERGE (a)-[r:{rel_type}]->(b) SET r += row.props"
            );
            let batch_val = serde_json::Value::Array(rows.clone());
            let bolt = neo4rs::BoltType::try_from(batch_val).map_err(|e| {
                StoreError::Serialization(format!("edge batch for {rel_type}: {e}"))
            })?;
            self.graph
                .run(neo4rs::query(&cypher).param("rows", bolt))
                .await
                .map_err(StoreError::from)?;
        }

        tracing::debug!(
            rel_types = by_rel.len(),
            total = edges.len(),
            "batch upserted edges"
        );
        Ok(())
    }

    /// Get a node by its domain `id` property, returning `None` if absent.
    #[instrument(name = "neo4j_store.get_node", skip(self), fields(node_id = %id))]
    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, StoreError> {
        let mut result = self
            .graph
            .execute(queries::get_node_by_id(id))
            .await
            .map_err(StoreError::from)?;

        match result.next().await.map_err(StoreError::from)? {
            None => Ok(None),
            Some(row) => {
                let node: neo4rs::Node = row.get("n").map_err(|e| {
                    StoreError::QueryFailed(format!("failed to deserialize node: {e}"))
                })?;
                Ok(Some(bolt_node_to_graph_node(node)?))
            }
        }
    }

    /// Return all neighbor nodes reachable within `depth` hops (undirected).
    #[instrument(name = "neo4j_store.get_neighbors", skip(self), fields(node_id = %node_id, depth))]
    async fn get_neighbors(
        &self,
        node_id: &str,
        depth: usize,
    ) -> Result<Vec<GraphNode>, StoreError> {
        let mut result = self
            .graph
            .execute(queries::get_neighbors(node_id, depth))
            .await
            .map_err(StoreError::from)?;

        let mut nodes = Vec::new();
        while let Some(row) = result.next().await.map_err(StoreError::from)? {
            let node: neo4rs::Node = row.get("neighbor").map_err(|e| {
                StoreError::QueryFailed(format!("failed to deserialize neighbor: {e}"))
            })?;
            nodes.push(bolt_node_to_graph_node(node)?);
        }
        Ok(nodes)
    }

    /// Cypher-level label-filtered neighbor expansion.
    ///
    /// Overrides the trait default (which filters client-side) so the
    /// predicate is pushed into the database and only matching rows cross the
    /// network boundary.
    #[instrument(
        name = "neo4j_store.get_neighbors_of_labels",
        skip(self, labels),
        fields(node_id = %node_id, depth, n_labels = labels.len())
    )]
    async fn get_neighbors_of_labels(
        &self,
        node_id: &str,
        depth: usize,
        labels: &[NodeLabel],
    ) -> Result<Vec<GraphNode>, StoreError> {
        let mut result = self
            .graph
            .execute(queries::get_neighbors_of_labels(node_id, depth, labels))
            .await
            .map_err(StoreError::from)?;

        let mut nodes = Vec::new();
        while let Some(row) = result.next().await.map_err(StoreError::from)? {
            let node: neo4rs::Node = row.get("neighbor").map_err(|e| {
                StoreError::QueryFailed(format!(
                    "failed to deserialize label-filtered neighbor: {e}"
                ))
            })?;
            nodes.push(bolt_node_to_graph_node(node)?);
        }
        Ok(nodes)
    }

    /// Traverse from `start_id`, optionally filtering by relationship types.
    ///
    /// Returns [`GraphPath`] values — each path contains the alternating
    /// sequence of nodes and edges collected from the neo4rs `Path` type.
    #[instrument(
        name = "neo4j_store.traverse",
        skip(self, relation_types),
        fields(start_id, max_depth, rel_count = relation_types.len())
    )]
    async fn traverse(
        &self,
        start_id: &str,
        relation_types: &[RelationType],
        max_depth: usize,
    ) -> Result<Vec<GraphPath>, StoreError> {
        let mut result = self
            .graph
            .execute(queries::traverse(start_id, relation_types, max_depth))
            .await
            .map_err(StoreError::from)?;

        let mut paths = Vec::new();
        while let Some(row) = result.next().await.map_err(StoreError::from)? {
            let path: neo4rs::Path = row
                .get("p")
                .map_err(|e| StoreError::QueryFailed(format!("failed to deserialize path: {e}")))?;
            paths.push(bolt_path_to_graph_path(path)?);
        }
        Ok(paths)
    }

    /// Execute a raw Cypher string and return all nodes bound to the
    /// first column in the result set.
    ///
    /// The column name `n` is assumed. Rows that do not contain a node
    /// at `n` are silently skipped.
    #[instrument(name = "neo4j_store.query_nodes", skip(self), fields(cypher_len = cypher.len()))]
    async fn query_nodes(&self, cypher: &str) -> Result<Vec<GraphNode>, StoreError> {
        let mut result = self
            .graph
            .execute(query(cypher))
            .await
            .map_err(StoreError::from)?;

        let mut nodes = Vec::new();
        while let Some(row) = result.next().await.map_err(StoreError::from)? {
            // Raw Cypher may return nodes under any alias; try "n" first.
            if let Ok(node) = row.get::<neo4rs::Node>("n") {
                nodes.push(bolt_node_to_graph_node(node)?);
            }
        }
        Ok(nodes)
    }

    /// Execute a raw Cypher query with string parameters and return scalar rows.
    ///
    /// Each row is materialised as a `HashMap<String, serde_json::Value>` by
    /// extracting the caller-supplied `columns`.  Column values that cannot be
    /// deserialised as JSON are silently skipped per row.
    ///
    /// `neo4rs::Row` does not expose a `keys()` iterator, so the caller must
    /// enumerate the expected column aliases explicitly.
    #[instrument(
        name = "neo4j_store.query_raw",
        skip(self, params, columns),
        fields(cypher_len = cypher.len(), param_count = params.len(), col_count = columns.len())
    )]
    async fn query_raw(
        &self,
        cypher: &str,
        params: std::collections::HashMap<String, String>,
        columns: &[&str],
    ) -> Result<Vec<std::collections::HashMap<String, serde_json::Value>>, StoreError> {
        let mut q = query(cypher);
        for (k, v) in params {
            q = q.param(k.as_str(), v);
        }
        let mut result = self.graph.execute(q).await.map_err(StoreError::from)?;

        let mut rows = Vec::new();
        while let Some(row) = result.next().await.map_err(StoreError::from)? {
            let mut map = std::collections::HashMap::new();
            for &col in columns {
                match row.get::<serde_json::Value>(col) {
                    Ok(val) => {
                        map.insert(col.to_owned(), val);
                    }
                    Err(e) => {
                        tracing::warn!(column = %col, error = %e, "failed to deserialise query_raw column; skipping");
                    }
                }
            }
            rows.push(map);
        }
        Ok(rows)
    }

    /// Delete a node and all its incident edges by domain `id`.
    #[instrument(name = "neo4j_store.delete_node", skip(self), fields(node_id = %id))]
    async fn delete_node(&self, id: &str) -> Result<(), StoreError> {
        self.graph
            .run(queries::delete_node_by_id(id))
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }

    /// Delete all nodes (and their edges) tagged with the given `ingest_run_id`.
    ///
    /// Delegates to the inherent `rollback_by_run_id` method on [`Neo4jStore`]
    /// implemented in `structural.rs`, so that the [`GraphStore`] trait can be
    /// used polymorphically for rollback without callers holding a concrete
    /// `Neo4jStore` reference.
    async fn rollback_by_run_id(&self, ingest_run_id: Uuid) -> Result<u64, StoreError> {
        // Explicit UFCS to call the inherent method on Neo4jStore, avoiding
        // recursion through the trait method of the same name.
        // SAFETY: this is a plain async fn delegation — no unsafe code.
        Neo4jStore::rollback_by_run_id(self, ingest_run_id).await
    }

    /// Delegate to the inherent `ensure_structural_constraints` on
    /// [`Neo4jStore`] implemented in `structural.rs`.
    async fn ensure_structural_constraints(&self) -> Result<(), StoreError> {
        Neo4jStore::ensure_structural_constraints(self).await
    }

    /// Delegate to the inherent `upsert_document_structure` on [`Neo4jStore`]
    /// implemented in `structural.rs`.
    async fn upsert_document_structure(
        &self,
        header: &crate::neo4j::structural::DocumentHeaderSnapshot,
        chunks: &[crate::neo4j::structural::StructuralChunk],
        tables: &[crate::neo4j::structural::StructuralTable],
        pictures: &[crate::neo4j::structural::StructuralPicture],
    ) -> Result<(), StoreError> {
        Neo4jStore::upsert_document_structure(self, header, chunks, tables, pictures).await
    }
}

// ---- Conversion helpers ----

/// Convert a `neo4rs::Node` (row result) into a domain [`GraphNode`].
///
/// The domain `id` property must be stored on the node as a string property
/// named `"id"`. The first label is used as the [`crate::types::NodeLabel`].
// StoreError is 184 bytes due to string variants — acceptable for internal conversion helper
#[allow(clippy::result_large_err)]
fn bolt_node_to_graph_node(node: neo4rs::Node) -> Result<GraphNode, StoreError> {
    let id: String = node
        .get("id")
        .map_err(|e| StoreError::QueryFailed(format!("node missing 'id' property: {e}")))?;

    let label_str = node
        .labels()
        .into_iter()
        .next()
        .ok_or_else(|| StoreError::QueryFailed("node has no labels".into()))?;

    let label = crate::types::NodeLabel::new(label_str)
        .map_err(|e| StoreError::QueryFailed(format!("invalid node label '{label_str}': {e}")))?;

    // Reconstruct properties from the node's property keys using the
    // neo4rs serde bridge (node.get::<serde_json::Value>(key)).
    let mut map = serde_json::Map::new();
    for key in node.keys() {
        match node.get::<serde_json::Value>(key) {
            Ok(val) => {
                map.insert(key.to_owned(), val);
            }
            Err(e) => {
                tracing::warn!(property = %key, error = %e, "failed to deserialize node property; skipping");
            }
        }
    }
    let properties = serde_json::Value::Object(map);

    Ok(GraphNode {
        id,
        label,
        properties,
    })
}

/// Convert a `neo4rs::Path` into a domain [`GraphPath`].
///
/// Nodes in the path are converted individually; edges are reconstructed
/// from the alternating node / relationship sequence. Bolt path
/// relationships carry no start/end node ids, so we zip consecutive
/// node pairs with the relation list.
#[allow(clippy::result_large_err)]
fn bolt_path_to_graph_path(path: neo4rs::Path) -> Result<GraphPath, StoreError> {
    let graph_nodes: Vec<GraphNode> = path
        .nodes()
        .into_iter()
        .map(bolt_node_to_graph_node)
        .collect::<Result<Vec<_>, _>>()?;

    let rels = path.rels();
    let mut graph_edges: Vec<GraphEdge> = Vec::with_capacity(rels.len());

    for (i, rel) in rels.into_iter().enumerate() {
        let src_id = graph_nodes.get(i).map(|n| n.id.clone()).ok_or_else(|| {
            StoreError::QueryFailed(format!("path relationship {i} has no source node"))
        })?;
        let dst_id = graph_nodes
            .get(i + 1)
            .map(|n| n.id.clone())
            .ok_or_else(|| {
                StoreError::QueryFailed(format!(
                    "path relationship {i} has no target node (index {})",
                    i + 1
                ))
            })?;

        let rel_type_str = rel.typ();
        let relation_type = RelationType::new(rel_type_str).map_err(|e| {
            StoreError::QueryFailed(format!("invalid relation type '{rel_type_str}': {e}"))
        })?;

        let mut prop_map = serde_json::Map::new();
        for key in rel.keys() {
            match rel.get::<serde_json::Value>(key) {
                Ok(val) => {
                    prop_map.insert(key.to_owned(), val);
                }
                Err(e) => {
                    tracing::warn!(property = %key, error = %e, "failed to deserialize relationship property; skipping");
                }
            }
        }

        graph_edges.push(GraphEdge {
            source_id: src_id,
            target_id: dst_id,
            relation_type,
            properties: serde_json::Value::Object(prop_map),
        });
    }

    Ok(GraphPath {
        nodes: graph_nodes,
        edges: graph_edges,
    })
}
