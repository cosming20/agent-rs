//! Neo4j graph store implementation.
//!
//! [`Neo4jStore`] implements [`GraphStore`] via the `neo4rs` async driver.
//! The connection pool is managed internally by `neo4rs::Graph`, which is
//! cheap to clone and share across tasks.

mod queries;

use agent_rs_core::{StoreError, config::Neo4jConfig};
use async_trait::async_trait;
use neo4rs::{Graph, query};
use tracing::instrument;

use crate::{
    traits::GraphStore,
    types::{GraphEdge, GraphNode, GraphPath, RelationType},
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

    /// Add multiple nodes in a batch, each MERGE'd by id.
    #[instrument(name = "neo4j_store.add_nodes", skip(self, nodes), fields(count = nodes.len()))]
    async fn add_nodes(&self, nodes: &[GraphNode]) -> Result<(), StoreError> {
        for node in nodes {
            let q = queries::upsert_node(node).map_err(StoreError::from)?;
            self.graph.run(q).await.map_err(StoreError::from)?;
        }
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

    /// Add multiple edges in a batch.
    #[instrument(name = "neo4j_store.add_edges", skip(self, edges), fields(count = edges.len()))]
    async fn add_edges(&self, edges: &[GraphEdge]) -> Result<(), StoreError> {
        for edge in edges {
            let q = queries::create_edge(edge).map_err(StoreError::from)?;
            self.graph.run(q).await.map_err(StoreError::from)?;
        }
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

    /// Delete a node and all its incident edges by domain `id`.
    #[instrument(name = "neo4j_store.delete_node", skip(self), fields(node_id = %id))]
    async fn delete_node(&self, id: &str) -> Result<(), StoreError> {
        self.graph
            .run(queries::delete_node_by_id(id))
            .await
            .map_err(StoreError::from)?;
        Ok(())
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
