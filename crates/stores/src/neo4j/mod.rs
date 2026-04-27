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

    /// Find knowledge-graph nodes whose name-like properties match a
    /// free-form mention.
    ///
    /// Backed by the `entity_mention_fts` Lucene fulltext index created
    /// by [`crate::neo4j::structural::Neo4jStore::ensure_structural_constraints`].
    /// The index spans `(name, title, heading)` across
    /// `Document|Section|Chunk|Entity`, so a mention like `"FCM"` finds
    /// Entity nodes, Section nodes (e.g. *"Clustering fuzzy (FCM)"*),
    /// and Chunk nodes uniformly.
    ///
    /// Results are scoped by a `WHERE node.user_id = $user_id` clause
    /// — the fulltext index cannot be a multi-property composite with a
    /// user-scope filter in Neo4j, so enforcement happens at the MATCH
    /// layer.
    ///
    /// Returns `Ok(Vec::new())` when:
    /// - `mention` trims to empty;
    /// - the fulltext index is missing (e.g. on a cold cluster before
    ///   `ensure_structural_constraints` has been called);
    /// - the Lucene query is malformed after escaping.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::InvalidData`] when `user_id` is empty — an
    /// empty tenant id would cause the `WHERE node.user_id = $uid` clause
    /// to match nothing tenant-scoped (and would accept any entity with
    /// a missing `user_id` property, breaking tenant isolation). Every
    /// other failure is returned from the underlying driver via
    /// [`StoreError::from`].
    #[instrument(
        name = "neo4j_store.find_entities_by_mention",
        skip(self, mention),
        fields(
            user_id = %user_id,
            mention_len = mention.len(),
            limit = limit,
            n_matches = tracing::field::Empty,
        )
    )]
    pub async fn find_entities_by_mention(
        &self,
        user_id: &agent_rs_core::UserId,
        mention: &str,
        limit: usize,
    ) -> Result<Vec<crate::types::EntityMatch>, StoreError> {
        if user_id.as_str().is_empty() {
            return Err(StoreError::InvalidData(
                "find_entities_by_mention: empty user_id is forbidden".into(),
            ));
        }

        let mention_trim = mention.trim();
        if mention_trim.is_empty() || limit == 0 {
            // Not a warn — empty mention / zero limit is a legitimate
            // input pattern from the router. DEBUG keeps the signal
            // queryable without polluting production logs.
            tracing::debug!(
                reason = "empty_mention_or_zero_limit",
                mention_len = mention.len(),
                limit,
                "find_entities_by_mention: early-returning empty result"
            );
            tracing::Span::current().record("n_matches", 0u64);
            return Ok(Vec::new());
        }

        // Escape Lucene reserved characters then build a tolerant query:
        // exact-phrase boost ∘ single-token wildcard. The wildcard form
        // lets a mention like "FCM" match against a node titled
        // "Fuzzy ISODATA (FCM)" where the indexed token chain keeps FCM
        // as its own token. Prefix wildcards are disabled by default in
        // Lucene (they force a full-index scan), so we keep a suffix
        // wildcard plus the exact-phrase form to cover both positions.
        //
        // `escaped_phrase` is safe to embed inside quotes as-is because
        // `lucene_escape` also escapes `"`. `escaped_term` strips any
        // trailing special characters so the appended `*` does not
        // produce a malformed parse like `foo\**` when the input ends in
        // a reserved char.
        let escaped_phrase = lucene_escape(mention_trim);
        let escaped_term = trim_trailing_lucene_escape(&escaped_phrase);
        // When the trimmed term is empty, the mention was made up entirely
        // of Lucene-reserved characters (e.g. `"&&"`). Appending `*` to
        // nothing produces a bare-wildcard query which Lucene rejects, so
        // fall back to the exact-phrase form only.
        let lucene_query = if escaped_term.is_empty() {
            format!("\"{escaped_phrase}\"^2")
        } else {
            format!("\"{escaped_phrase}\"^2 OR {escaped_term}*")
        };

        let cypher = "CALL db.index.fulltext.queryNodes(\"entity_mention_fts\", $q) \
                      YIELD node, score \
                      WHERE node.user_id = $uid \
                      RETURN node.id AS id, \
                             coalesce(node.name, node.title, node.heading, '') AS display_name, \
                             head(labels(node)) AS label, \
                             score \
                      ORDER BY score DESC LIMIT $lim";

        let q = query(cypher)
            .param("q", lucene_query)
            .param("uid", user_id.as_str())
            .param("lim", limit as i64);

        let mut result = match self.graph.execute(q).await {
            Ok(r) => r,
            Err(e) => {
                // Missing fulltext index → treat as "no matches" so the
                // entity-linking pipeline fails open instead of breaking.
                //
                // Detection uses a structured Neo4jError code match (via
                // `is_fulltext_index_missing`) with a free-text fallback
                // kept only as a last resort. Every fail-open path emits
                // a `reason=` field so dashboards can alert on sustained
                // rates of each specific degradation cause rather than
                // merging everything into a single "entity linker failed"
                // counter.
                if is_fulltext_index_missing(&e) {
                    tracing::warn!(
                        reason = "fulltext_index_missing",
                        index = "entity_mention_fts",
                        user_id = %user_id,
                        "fulltext index missing; entity linking returns empty. \
                         Run `ensure_structural_constraints` on this cluster."
                    );
                    tracing::Span::current().record("n_matches", 0u64);
                    return Ok(Vec::new());
                }
                tracing::warn!(
                    reason = "query_execute_failed",
                    user_id = %user_id,
                    error = %e,
                    "find_entities_by_mention: propagating execute error to caller"
                );
                return Err(StoreError::from(e));
            }
        };

        let mut out: Vec<crate::types::EntityMatch> = Vec::new();
        while let Some(row) = result.next().await.map_err(StoreError::from)? {
            let id: String = match row.get::<String>("id") {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(
                        reason = "row_missing_id_field",
                        user_id = %user_id,
                        error = ?err,
                        "fulltext row lacks `id` field; skipping"
                    );
                    continue;
                }
            };
            let display_name: String = row.get::<String>("display_name").unwrap_or_default();
            let label: String = row.get::<String>("label").unwrap_or_default();
            let score: f64 = row.get::<f64>("score").unwrap_or(0.0);
            out.push(crate::types::EntityMatch {
                node_id: id,
                display_name,
                label,
                score,
            });
        }
        tracing::Span::current().record("n_matches", out.len());
        Ok(out)
    }
}

/// Return `true` iff `e` signals a missing `entity_mention_fts`
/// fulltext index. Prefers the structured Neo4j error code
/// (`Neo.ClientError.Procedure.ProcedureCallFailed` or
/// `Neo.ClientError.Schema.IndexNotFound`) and falls back to the raw
/// message only for driver versions that do not populate the code.
fn is_fulltext_index_missing(e: &neo4rs::Error) -> bool {
    if let neo4rs::Error::Neo4j(ne) = e
        && (ne.code().contains("IndexNotFound")
            || (ne.code().contains("Procedure") && ne.message().contains("entity_mention_fts")))
    {
        return true;
    }
    let msg = e.to_string();
    msg.contains("entity_mention_fts")
        && (msg.contains("not found") || msg.contains("There is no such"))
}

/// Escape Lucene query-syntax reserved characters so that an arbitrary
/// user mention can be embedded into a fulltext query without aborting
/// the parse or changing the query semantics.
///
/// Handles both the single-char reserved set
/// (`+ - ! ( ) { } [ ] ^ " ~ * ? : \ /`) and the two-char boolean
/// operators `&&` and `||`. The two-char operators are substituted
/// first (`&& → \&\&`, `|| → \|\|`) so a char-by-char pass cannot
/// accidentally split them. Bare `&` and `|` are not operators in
/// Lucene and pass through unchanged.
///
/// The output is used exactly once per mention, so a preallocated
/// `String` keeps the overhead trivial.
fn lucene_escape(q: &str) -> String {
    // Pre-pass: substitute the two-char operators. Walk the chars and
    // emit `\&\&` / `\|\|` when a pair is seen; fall through to the
    // single-char pass otherwise. We write directly into `out` to avoid
    // an intermediate allocation.
    let mut out = String::with_capacity(q.len() + 8);
    let mut chars = q.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                out.push_str("\\&\\&");
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                out.push_str("\\|\\|");
            }
            '+' | '-' | '!' | '(' | ')' | '{' | '}' | '[' | ']' | '^' | '"' | '~' | '*' | '?'
            | ':' | '\\' | '/' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Strip every trailing `\X` escape pair from an already-escaped Lucene
/// term so that appending a wildcard suffix cannot produce malformed
/// output like `foo\**` (single-char case) or `foo\&*` (two-char
/// boolean-op case where `lucene_escape` emits `\&\&` for `&&`).
///
/// Iterates: a single call strips every trailing `\X` pair until the
/// string either becomes empty or ends in a non-escape character. This
/// matters for inputs like `"foo&&"` whose escape output is `foo\&\&`
/// (4 bytes of trailing escape) — a single-pass trim would leave `\&`
/// dangling and the appended `*` would attach to a literal `&` instead
/// of operating as a prefix wildcard on `foo`.
///
/// All reserved chars handled by `lucene_escape` are single-byte ASCII,
/// so byte-indexing from the end is UTF-8 safe: `bytes[len - 2] == b'\\'`
/// can only be true when the previous byte is a literal backslash, not
/// the final continuation byte of a multi-byte character.
fn trim_trailing_lucene_escape(mut escaped: &str) -> &str {
    loop {
        let bytes = escaped.as_bytes();
        if bytes.len() < 2 || bytes[bytes.len() - 2] != b'\\' {
            return escaped;
        }
        let last = bytes[bytes.len() - 1];
        if !matches!(
            last,
            b'+' | b'-'
                | b'!'
                | b'('
                | b')'
                | b'{'
                | b'}'
                | b'['
                | b']'
                | b'^'
                | b'"'
                | b'~'
                | b'*'
                | b'?'
                | b':'
                | b'\\'
                | b'/'
                | b'&'
                | b'|'
        ) {
            return escaped;
        }
        escaped = &escaped[..escaped.len() - 2];
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

    /// Delegate to the inherent `find_entities_by_mention`.
    async fn find_entities_by_mention(
        &self,
        user_id: &agent_rs_core::UserId,
        mention: &str,
        limit: usize,
    ) -> Result<Vec<crate::types::EntityMatch>, StoreError> {
        Neo4jStore::find_entities_by_mention(self, user_id, mention, limit).await
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

#[cfg(test)]
mod lucene_escape_tests {
    use super::*;

    #[test]
    fn escapes_single_char_reserved() {
        assert_eq!(lucene_escape("foo+bar"), "foo\\+bar");
        assert_eq!(lucene_escape("a*b"), "a\\*b");
        assert_eq!(lucene_escape("A (B)"), "A \\(B\\)");
    }

    #[test]
    fn escapes_boolean_operators() {
        assert_eq!(lucene_escape("A && B"), "A \\&\\& B");
        assert_eq!(lucene_escape("A || B"), "A \\|\\| B");
        // Bare `&` / `|` are not operators, so pass through.
        assert_eq!(lucene_escape("Tom & Jerry"), "Tom & Jerry");
        assert_eq!(lucene_escape("pipe | sep"), "pipe | sep");
    }

    #[test]
    fn escapes_backslash() {
        assert_eq!(lucene_escape("a\\b"), "a\\\\b");
    }

    #[test]
    fn trims_trailing_escape_before_wildcard() {
        // Input: "foo*" → escaped "foo\*" → trimmed "foo" so suffix `*`
        // yields the expected wildcard "foo*", not "foo\**".
        let escaped = lucene_escape("foo*");
        assert_eq!(escaped, "foo\\*");
        assert_eq!(trim_trailing_lucene_escape(&escaped), "foo");
    }

    #[test]
    fn trim_is_noop_on_plain_term() {
        let escaped = lucene_escape("FCM");
        assert_eq!(escaped, "FCM");
        assert_eq!(trim_trailing_lucene_escape(&escaped), "FCM");
    }

    #[test]
    fn trim_strips_trailing_two_char_operators() {
        // "foo&&" → escape "foo\&\&" → trim iteratively drops 4 bytes,
        // leaving "foo" so the caller's `*` suffix attaches cleanly.
        let escaped = lucene_escape("foo&&");
        assert_eq!(escaped, "foo\\&\\&");
        assert_eq!(trim_trailing_lucene_escape(&escaped), "foo");

        let escaped = lucene_escape("bar||");
        assert_eq!(escaped, "bar\\|\\|");
        assert_eq!(trim_trailing_lucene_escape(&escaped), "bar");
    }

    #[test]
    fn trim_fully_empties_all_special_term() {
        // A mention made entirely of reserved chars trims to an empty
        // term; the caller must fall back to a phrase-only query.
        let escaped = lucene_escape("&&||**");
        assert_eq!(trim_trailing_lucene_escape(&escaped), "");
    }

    #[test]
    fn lucene_escape_preserves_unicode() {
        // Byte indexing in `trim_trailing_lucene_escape` is only safe
        // because reserved chars are ASCII — pin that contract with a
        // mixed-alphabet input so any regression in the escape path or
        // the trim helper shows up here.
        let escaped = lucene_escape("北京 (PEK)");
        assert_eq!(escaped, "北京 \\(PEK\\)");
        assert_eq!(trim_trailing_lucene_escape(&escaped), "北京 \\(PEK");
    }
}
