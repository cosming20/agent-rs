//! Section-level read queries used by the coverage planner / outline tool.

use neo4rs::query;
use serde_json::Value;
use tracing::instrument;

use crate::neo4j::structural::{LABEL_DOCUMENT, LABEL_SECTION};
use agent_rs_core::StoreError;

use super::Neo4jStore;

/// One section row fetched from Neo4j.
#[derive(Debug, Clone)]
pub struct SectionRow {
    pub section_id: String,
    pub doc_id: String,
    pub heading: String,
    pub section_order: Option<i64>,
    pub first_chunk_index: Option<i64>,
    pub page_start: Option<i64>,
    pub is_preamble: bool,
}

impl SectionRow {
    /// Stable outline ordering: by `section_order` ascending (preamble = 0
    /// sorts first), tie-broken by `first_chunk_index` then `page_start`.
    pub fn sort_for_outline(rows: &mut [SectionRow]) {
        rows.sort_by(|a, b| {
            a.section_order
                .unwrap_or(i64::MAX)
                .cmp(&b.section_order.unwrap_or(i64::MAX))
                .then(
                    a.first_chunk_index
                        .unwrap_or(i64::MAX)
                        .cmp(&b.first_chunk_index.unwrap_or(i64::MAX)),
                )
                .then(
                    a.page_start
                        .unwrap_or(i64::MAX)
                        .cmp(&b.page_start.unwrap_or(i64::MAX)),
                )
        });
    }
}

impl Neo4jStore {
    /// List all sections (real + synthetic preamble) of a document, ordered
    /// for outline display.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the Neo4j query fails or the driver returns
    /// a connection-level error.
    #[instrument(skip(self), fields(doc_id = %doc_id))]
    pub async fn list_sections_for_doc(
        &self,
        doc_id: &str,
    ) -> Result<Vec<SectionRow>, StoreError> {
        let cypher = format!(
            "MATCH (d:{LABEL_DOCUMENT} {{id: $doc_id}})-[:HAS_SECTION*1..]->(s:{LABEL_SECTION}) \
             RETURN s.id AS section_id, d.id AS doc_id, coalesce(s.title, s.heading) AS heading, \
                    s.section_order AS section_order, s.first_chunk_index AS first_chunk_index, \
                    s.page_start AS page_start, coalesce(s.is_preamble, false) AS is_preamble"
        );
        let mut rs = self
            .graph
            .execute(query(&cypher).param("doc_id", doc_id))
            .await
            .map_err(StoreError::from)?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await.map_err(StoreError::from)? {
            out.push(SectionRow {
                section_id: row.get("section_id").unwrap_or_default(),
                doc_id: row.get("doc_id").unwrap_or_default(),
                heading: row.get("heading").unwrap_or_default(),
                section_order: row.get("section_order").ok(),
                first_chunk_index: row.get("first_chunk_index").ok(),
                page_start: row.get("page_start").ok(),
                is_preamble: row.get("is_preamble").unwrap_or(false),
            });
        }
        SectionRow::sort_for_outline(&mut out);
        Ok(out)
    }

    /// Recursively collect descendant section ids (used when the outline
    /// agent says "all of 5.1 including 5.1.1, 5.1.2, ...").
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the Neo4j query fails.
    #[instrument(skip(self), fields(section_id = %section_id))]
    pub async fn list_descendant_sections(
        &self,
        section_id: &str,
    ) -> Result<Vec<String>, StoreError> {
        let cypher = format!(
            "MATCH (root:{LABEL_SECTION} {{id: $sid}})-[:PARENT_OF*0..]->(s:{LABEL_SECTION}) \
             RETURN s.id AS section_id"
        );
        let mut rs = self
            .graph
            .execute(query(&cypher).param("sid", section_id))
            .await
            .map_err(StoreError::from)?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await.map_err(StoreError::from)? {
            if let Ok(id) = row.get::<String>("section_id") {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// Return chunk ids that belong to any of the given sections, ordered
    /// by `chunk_index` so `fetch_section_chunks` reads the section
    /// linearly. Caller uses these ids to fetch full payloads from Qdrant.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if serialization or the Neo4j query fails.
    #[instrument(skip(self, section_ids), fields(n_sections = section_ids.len()))]
    pub async fn chunks_in_sections(
        &self,
        section_ids: &[String],
    ) -> Result<Vec<String>, StoreError> {
        if section_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids_json = Value::Array(
            section_ids
                .iter()
                .map(|s| Value::String(s.clone()))
                .collect(),
        );
        let ids_bolt = neo4rs::BoltType::try_from(ids_json)
            .map_err(|e| StoreError::Serialization(format!("section_ids serialize: {e}")))?;
        let cypher = format!(
            "MATCH (s:{LABEL_SECTION})-[:HAS_CHUNK]->(c:Chunk) \
             WHERE s.id IN $ids \
             RETURN c.id AS chunk_id, coalesce(c.chunk_index, 0) AS chunk_index \
             ORDER BY chunk_index ASC"
        );
        let mut rs = self
            .graph
            .execute(query(&cypher).param("ids", ids_bolt))
            .await
            .map_err(StoreError::from)?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await.map_err(StoreError::from)? {
            if let Ok(id) = row.get::<String>("chunk_id") {
                out.push(id);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_row_orders_by_section_order_then_first_chunk_index() {
        let mut rows = vec![
            SectionRow {
                section_id: "b".into(),
                doc_id: "d".into(),
                heading: "5.2".into(),
                section_order: Some(2),
                first_chunk_index: Some(10),
                page_start: Some(8),
                is_preamble: false,
            },
            SectionRow {
                section_id: "a".into(),
                doc_id: "d".into(),
                heading: "5.1".into(),
                section_order: Some(1),
                first_chunk_index: Some(0),
                page_start: Some(2),
                is_preamble: false,
            },
            SectionRow {
                section_id: "p".into(),
                doc_id: "d".into(),
                heading: "(preamble)".into(),
                section_order: Some(0),
                first_chunk_index: Some(0),
                page_start: Some(1),
                is_preamble: true,
            },
        ];
        SectionRow::sort_for_outline(&mut rows);
        assert_eq!(
            rows.iter()
                .map(|r| r.section_id.as_str())
                .collect::<Vec<_>>(),
            vec!["p", "a", "b"]
        );
    }
}
