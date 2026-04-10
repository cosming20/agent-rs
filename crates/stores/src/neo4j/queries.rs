//! Cypher query builder helpers for Neo4j operations.
//!
//! All functions return a [`neo4rs::Query`] with parameters already bound.
//! Labels and relationship types are interpolated via `format!()` — safe
//! because [`NodeLabel`] and [`RelationType`] only permit ASCII alphanumeric
//! + underscore characters.

use neo4rs::query;

use crate::types::{GraphEdge, GraphNode, RelationType};

/// Convert a `serde_json::Value` to `neo4rs::types::BoltType`.
///
/// Uses the `TryFrom` impl provided by neo4rs when compiled with the
/// `"json"` Cargo feature (enabled in this workspace).
///
/// # Errors
///
/// Returns `neo4rs::Error` if the JSON value cannot be represented as a
/// Bolt type — for example when a JSON number exceeds `i64::MAX`.
pub(crate) fn json_to_bolt(value: serde_json::Value) -> Result<neo4rs::BoltType, neo4rs::Error> {
    neo4rs::BoltType::try_from(value)
}

/// BUILD: `MERGE (n:Label {id: $id}) SET n += $props`
///
/// Uses MERGE on `id` so re-ingesting the same node is idempotent.
pub(crate) fn upsert_node(node: &GraphNode) -> Result<neo4rs::Query, neo4rs::Error> {
    let label = node.label.as_str();
    let cypher = format!("MERGE (n:{label} {{id: $id}}) SET n += $props");
    let props = json_to_bolt(node.properties.clone())?;
    Ok(query(&cypher)
        .param("id", node.id.clone())
        .param("props", props))
}

/// BUILD: `MATCH + CREATE` for a single directed edge.
///
/// The relationship type is interpolated — safe because `RelationType`
/// validates to ASCII alphanumeric + underscore.
pub(crate) fn create_edge(edge: &GraphEdge) -> Result<neo4rs::Query, neo4rs::Error> {
    let rel = edge.relation_type.as_str();
    let cypher = format!(
        "MATCH (a {{id: $src}}), (b {{id: $dst}}) \
         MERGE (a)-[r:{rel}]->(b) SET r += $props"
    );
    let props = json_to_bolt(edge.properties.clone())?;
    Ok(query(&cypher)
        .param("src", edge.source_id.clone())
        .param("dst", edge.target_id.clone())
        .param("props", props))
}

/// BUILD: `MATCH (n {id: $id}) RETURN n`
pub(crate) fn get_node_by_id(id: &str) -> neo4rs::Query {
    query("MATCH (n {id: $id}) RETURN n").param("id", id)
}

/// BUILD: variable-length neighbor expansion up to `depth`.
///
/// Returns distinct neighbor nodes (excludes the start node).
pub(crate) fn get_neighbors(node_id: &str, depth: usize) -> neo4rs::Query {
    let cypher = format!(
        "MATCH (start {{id: $id}})-[*1..{depth}]-(neighbor) \
         WHERE neighbor.id <> $id \
         RETURN DISTINCT neighbor"
    );
    query(&cypher).param("id", node_id)
}

/// BUILD: traverse with optional relationship type filter, returning paths.
///
/// Each relation type name is ASCII-safe (validated by `RelationType`).
pub(crate) fn traverse(
    start_id: &str,
    relation_types: &[RelationType],
    max_depth: usize,
) -> neo4rs::Query {
    let rel_filter = if relation_types.is_empty() {
        String::new()
    } else {
        let types = relation_types
            .iter()
            .map(RelationType::as_str)
            .collect::<Vec<_>>()
            .join("|");
        format!(":{types}")
    };
    let cypher = format!(
        "MATCH p = (start {{id: $id}})-[r{rel_filter}*1..{max_depth}]->(end) \
         RETURN p"
    );
    query(&cypher).param("id", start_id)
}

/// BUILD: `DETACH DELETE` for a node by id.
pub(crate) fn delete_node_by_id(id: &str) -> neo4rs::Query {
    query("MATCH (n {id: $id}) DETACH DELETE n").param("id", id)
}
