//! Post-retrieval context expansion via Neo4j structural graph edges.
//!
//! When a retrieval hit lands at a chunk boundary the answer may span adjacent
//! chunks.  [`ContextExpander`] takes a list of top-K hit chunk IDs and
//! enriches each one by fetching:
//!
//! - **FOLLOWS neighbors** — up to `follows_depth` chunks immediately before
//!   and after the hit in document order (via the `FOLLOWS` edge chain).
//! - **IN_SECTION siblings** — other chunks that belong to the same `Section`
//!   node (via the `IN_SECTION` edge), deduplicated against FOLLOWS results.
//!
//! On any graph query failure the original hit is returned unchanged (graceful
//! degradation).  This ensures the calling pipeline always receives at least
//! the original evidence even if Neo4j is unreachable.
//!
//! # Example
//!
//! ```rust,ignore
//! let expander = ContextExpander::new(graph_store, ContextExpanderConfig::default());
//! let expanded = expander.expand(&hit_ids).await?;
//! // expanded[i].context_before / context_after contain ordered neighbors
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use agent_rs_stores::traits::GraphStore;
use tracing::{instrument, warn};

use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Column aliases returned by expansion queries
// ---------------------------------------------------------------------------

const COL_ID: &str = "id";
const COL_TEXT: &str = "text";
const COL_PAGE_START: &str = "page_start";
const COL_SECTION_TITLE: &str = "section_title";

const EXPANSION_COLUMNS: &[&str] = &[COL_ID, COL_TEXT, COL_PAGE_START, COL_SECTION_TITLE];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ContextExpander`].
#[derive(Debug, Clone)]
pub struct ContextExpanderConfig {
    /// How many chunks to fetch before and after each hit via `FOLLOWS` edges.
    ///
    /// A value of `2` expands by at most two hops in each direction (±2 chunks).
    /// Default: `2`.
    pub follows_depth: usize,

    /// When `true`, also fetch same-section chunks via `IN_SECTION` edges and
    /// include them as [`ExpandedChunk::section_siblings`].
    ///
    /// Default: `true`.
    pub expand_section: bool,

    /// Maximum total chunks in the expanded output (prevents combinatorial
    /// explosion on large sections).  Hit chunks that would push the total past
    /// this limit are included but with fewer neighbors.
    ///
    /// Default: `15`.
    pub max_expanded: usize,
}

impl Default for ContextExpanderConfig {
    fn default() -> Self {
        Self {
            follows_depth: 2,
            expand_section: true,
            max_expanded: 15,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A neighbor chunk fetched from the structural graph.
#[derive(Debug, Clone)]
pub struct ContextChunk {
    /// Domain ID of the chunk node.
    pub chunk_id: String,
    /// Full chunk text.
    pub text: String,
    /// First page this chunk appears on (1-based), if stored.
    pub page_start: Option<i64>,
    /// Heading title of the containing section, if stored.
    pub section_title: Option<String>,
}

/// A retrieval hit enriched with its graph neighborhood.
#[derive(Debug, Clone)]
pub struct ExpandedChunk {
    /// The original hit chunk ID.
    pub chunk_id: String,
    /// Text of the original hit chunk.
    pub original_text: String,
    /// Chunks immediately before the hit in document order, oldest-first.
    pub context_before: Vec<ContextChunk>,
    /// Chunks immediately after the hit in document order, oldest-first.
    pub context_after: Vec<ContextChunk>,
    /// Same-section siblings not already present in `context_before` or
    /// `context_after`, ordered by their chunk index where available.
    pub section_siblings: Vec<ContextChunk>,
}

// ---------------------------------------------------------------------------
// ContextExpander
// ---------------------------------------------------------------------------

/// Post-retrieval context expander using Neo4j structural graph edges.
///
/// Constructed with an `Arc<dyn GraphStore>` and a [`ContextExpanderConfig`].
/// The `GraphStore` must be backed by Neo4j (or a compatible implementation
/// that understands `FOLLOWS` / `IN_SECTION` structural edges).
pub struct ContextExpander {
    graph_store: Arc<dyn GraphStore + Send + Sync>,
    config: ContextExpanderConfig,
}

impl ContextExpander {
    /// Create a new [`ContextExpander`].
    ///
    /// # Parameters
    ///
    /// - `graph_store` — shared reference to the graph backend.
    /// - `config` — expansion parameters (depth, section expansion, max total).
    pub fn new(
        graph_store: Arc<dyn GraphStore + Send + Sync>,
        config: ContextExpanderConfig,
    ) -> Self {
        Self {
            graph_store,
            config,
        }
    }

    /// Expand context for a list of hit chunk IDs.
    ///
    /// Each element of `hit_chunks` is a `(chunk_id, original_text)` pair.
    /// Returns one [`ExpandedChunk`] per input hit, preserving order.
    ///
    /// On graph query failure the hit is returned with empty neighbor lists
    /// (graceful degradation) and a warning is logged.
    ///
    /// The combined total of all returned chunks (originals + neighbors) is
    /// bounded by [`ContextExpanderConfig::max_expanded`].
    ///
    /// # Errors
    ///
    /// Only returns [`RetrievalError`] for internal logic failures.  Graph
    /// backend errors are demoted to warnings and produce degraded output.
    #[instrument(
        name = "context_expander.expand",
        skip(self, hit_chunks),
        fields(
            hit_count = hit_chunks.len(),
            follows_depth = self.config.follows_depth,
            expand_section = self.config.expand_section,
        )
    )]
    pub async fn expand(
        &self,
        hit_chunks: &[(String, String)],
    ) -> Result<Vec<ExpandedChunk>, RetrievalError> {
        if hit_chunks.is_empty() {
            return Ok(Vec::new());
        }

        // IDs of all direct hits — used for cross-hit deduplication.
        let hit_ids: HashSet<String> = hit_chunks.iter().map(|(id, _)| id.clone()).collect();

        let mut results: Vec<ExpandedChunk> = Vec::with_capacity(hit_chunks.len());
        let mut total_chunks: usize = 0;

        for (chunk_id, original_text) in hit_chunks {
            if total_chunks >= self.config.max_expanded {
                // Budget exhausted; return remaining hits without expansion.
                results.push(ExpandedChunk {
                    chunk_id: chunk_id.clone(),
                    original_text: original_text.clone(),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                    section_siblings: Vec::new(),
                });
                total_chunks += 1;
                continue;
            }

            let expanded = self.expand_single(chunk_id, original_text, &hit_ids).await;

            let neighbor_count = {
                let e = match &expanded {
                    Ok(e) => e,
                    Err(_) => {
                        results.push(ExpandedChunk {
                            chunk_id: chunk_id.clone(),
                            original_text: original_text.clone(),
                            context_before: Vec::new(),
                            context_after: Vec::new(),
                            section_siblings: Vec::new(),
                        });
                        total_chunks += 1;
                        continue;
                    }
                };
                e.context_before.len() + e.context_after.len() + e.section_siblings.len()
            };

            tracing::debug!(
                chunk_id = %chunk_id,
                neighbors = neighbor_count,
                "context expanded"
            );

            total_chunks += 1 + neighbor_count;
            results.push(expanded?);
        }

        tracing::info!(
            hits = hit_chunks.len(),
            total_after_expansion = total_chunks,
            "context expansion complete"
        );

        Ok(results)
    }

    /// Expand a single chunk: fetch FOLLOWS neighbors and (optionally)
    /// IN_SECTION siblings.
    ///
    /// Returns `Ok(ExpandedChunk)` on success or logs a warning and returns
    /// an empty-neighbor chunk on graph failure.
    async fn expand_single(
        &self,
        chunk_id: &str,
        original_text: &str,
        all_hit_ids: &HashSet<String>,
    ) -> Result<ExpandedChunk, RetrievalError> {
        let depth = self.config.follows_depth;

        // ---- FOLLOWS: chunks before this one ----
        let context_before = match self.fetch_follows_before(chunk_id, depth).await {
            Ok(chunks) => chunks,
            Err(e) => {
                warn!(
                    chunk_id = %chunk_id,
                    error = %e,
                    "FOLLOWS-before query failed; skipping context_before"
                );
                Vec::new()
            }
        };

        // ---- FOLLOWS: chunks after this one ----
        let context_after = match self.fetch_follows_after(chunk_id, depth).await {
            Ok(chunks) => chunks,
            Err(e) => {
                warn!(
                    chunk_id = %chunk_id,
                    error = %e,
                    "FOLLOWS-after query failed; skipping context_after"
                );
                Vec::new()
            }
        };

        // Build the deduplication set: hit IDs + before/after IDs.
        let mut seen_ids: HashSet<String> = all_hit_ids.clone();
        seen_ids.insert(chunk_id.to_owned());
        for c in &context_before {
            seen_ids.insert(c.chunk_id.clone());
        }
        for c in &context_after {
            seen_ids.insert(c.chunk_id.clone());
        }

        // ---- IN_SECTION siblings (optional) ----
        let section_siblings = if self.config.expand_section {
            match self.fetch_section_siblings(chunk_id, &seen_ids).await {
                Ok(siblings) => siblings,
                Err(e) => {
                    warn!(
                        chunk_id = %chunk_id,
                        error = %e,
                        "IN_SECTION query failed; skipping section_siblings"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        Ok(ExpandedChunk {
            chunk_id: chunk_id.to_owned(),
            original_text: original_text.to_owned(),
            context_before,
            context_after,
            section_siblings,
        })
    }

    // -----------------------------------------------------------------------
    // Graph query helpers
    // -----------------------------------------------------------------------

    /// Fetch up to `depth` chunks that precede `chunk_id` via FOLLOWS edges,
    /// returned in document order (ascending chunk_index where available,
    /// otherwise in traversal order).
    async fn fetch_follows_before(
        &self,
        chunk_id: &str,
        depth: usize,
    ) -> Result<Vec<ContextChunk>, RetrievalError> {
        // Walk backwards along FOLLOWS edges: prev -[FOLLOWS]-> target
        // Multiple hops: (target)<-[:FOLLOWS*1..N]-(prev)
        let cypher = format!(
            "MATCH (target:Chunk {{id: $chunk_id}})<-[:FOLLOWS*1..{depth}]-(prev:Chunk) \
             RETURN prev.id AS id, \
                    prev.chunk_text AS text, \
                    prev.page_start AS page_start, \
                    prev.section_title AS section_title \
             ORDER BY prev.chunk_index ASC"
        );

        let params = build_params(chunk_id);
        let rows = self
            .graph_store
            .query_raw(&cypher, params, EXPANSION_COLUMNS)
            .await
            .map_err(RetrievalError::Store)?;

        Ok(rows_to_context_chunks(rows))
    }

    /// Fetch up to `depth` chunks that follow `chunk_id` via FOLLOWS edges,
    /// returned in document order (ascending chunk_index).
    async fn fetch_follows_after(
        &self,
        chunk_id: &str,
        depth: usize,
    ) -> Result<Vec<ContextChunk>, RetrievalError> {
        // Walk forwards: target -[FOLLOWS*1..N]-> next
        let cypher = format!(
            "MATCH (target:Chunk {{id: $chunk_id}})-[:FOLLOWS*1..{depth}]->(next:Chunk) \
             RETURN next.id AS id, \
                    next.chunk_text AS text, \
                    next.page_start AS page_start, \
                    next.section_title AS section_title \
             ORDER BY next.chunk_index ASC"
        );

        let params = build_params(chunk_id);
        let rows = self
            .graph_store
            .query_raw(&cypher, params, EXPANSION_COLUMNS)
            .await
            .map_err(RetrievalError::Store)?;

        Ok(rows_to_context_chunks(rows))
    }

    /// Fetch same-section chunks via IN_SECTION edges, excluding IDs already
    /// in `exclude_ids`.  Limited to 5 siblings to prevent section-wide explosion.
    async fn fetch_section_siblings(
        &self,
        chunk_id: &str,
        exclude_ids: &HashSet<String>,
    ) -> Result<Vec<ContextChunk>, RetrievalError> {
        // Fetch up to 5 siblings from the same section, ordered by chunk_index.
        // We cannot push the exclude set into Cypher without a list parameter,
        // so we fetch the raw candidates and filter in Rust instead.
        const SECTION_SIBLING_LIMIT: usize = 10;

        let cypher = format!(
            "MATCH (target:Chunk {{id: $chunk_id}})-[:IN_SECTION]->(s:Section) \
             MATCH (sibling:Chunk)-[:IN_SECTION]->(s) \
             WHERE sibling.id <> $chunk_id \
             RETURN sibling.id AS id, \
                    sibling.chunk_text AS text, \
                    sibling.page_start AS page_start, \
                    sibling.section_title AS section_title \
             ORDER BY sibling.chunk_index ASC \
             LIMIT {SECTION_SIBLING_LIMIT}"
        );

        let params = build_params(chunk_id);
        let rows = self
            .graph_store
            .query_raw(&cypher, params, EXPANSION_COLUMNS)
            .await
            .map_err(RetrievalError::Store)?;

        let siblings = rows_to_context_chunks(rows)
            .into_iter()
            .filter(|c| !exclude_ids.contains(&c.chunk_id))
            .take(5)
            .collect();

        Ok(siblings)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build the single-parameter map used by all expansion queries.
fn build_params(chunk_id: &str) -> HashMap<String, String> {
    let mut params = HashMap::with_capacity(1);
    params.insert("chunk_id".to_owned(), chunk_id.to_owned());
    params
}

/// Convert raw query rows into [`ContextChunk`] values.
///
/// Rows where the `id` column is absent or null are silently dropped.
fn rows_to_context_chunks(rows: Vec<HashMap<String, serde_json::Value>>) -> Vec<ContextChunk> {
    rows.into_iter()
        .filter_map(|mut row| {
            let id = match row.remove(COL_ID) {
                Some(serde_json::Value::String(s)) if !s.is_empty() => s,
                _ => return None,
            };

            let text = match row.remove(COL_TEXT) {
                Some(serde_json::Value::String(s)) => s,
                _ => String::new(),
            };

            let page_start = match row.remove(COL_PAGE_START) {
                Some(serde_json::Value::Number(n)) => n.as_i64(),
                _ => None,
            };

            let section_title = match row.remove(COL_SECTION_TITLE) {
                Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s),
                _ => None,
            };

            Some(ContextChunk {
                chunk_id: id,
                text,
                page_start,
                section_title,
            })
        })
        .collect()
}
