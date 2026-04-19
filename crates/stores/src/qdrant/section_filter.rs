//! Build Qdrant `Filter` clauses for `doc_id` / `section_id` payload keys.
//!
//! These are the exact payload keys the ingestion pipeline writes
//! (see `crates/retrieval/src/ingestion/pipeline.rs` payload construction).

use qdrant_client::qdrant::{Condition, Filter};

/// Build an `AND` filter on `doc_id` / `section_id`. Returns `None` when
/// both args are `None` so callers can use `Option::is_some` to skip
/// adding any filter at all.
///
/// # Examples
///
/// ```rust
/// use agent_rs_stores::qdrant::section_filter::build_section_filter;
///
/// let f = build_section_filter(Some("doc-123"), Some("sec-456"));
/// assert!(f.is_some());
///
/// let none = build_section_filter(None, None);
/// assert!(none.is_none());
/// ```
pub fn build_section_filter(doc_id: Option<&str>, section_id: Option<&str>) -> Option<Filter> {
    let mut conds: Vec<Condition> = Vec::new();

    if let Some(did) = doc_id {
        conds.push(Condition::matches("doc_id", did.to_string()));
    }

    if let Some(sid) = section_id {
        conds.push(Condition::matches("section_id", sid.to_string()));
    }

    if conds.is_empty() {
        None
    } else {
        Some(Filter {
            must: conds,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_section_filter_with_doc_and_section() {
        let f = build_section_filter(Some("doc-123"), Some("sec-456"));
        let f = f.expect("filter should be Some when at least one id is provided");
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    fn build_section_filter_returns_none_when_both_args_are_none() {
        assert!(build_section_filter(None, None).is_none());
    }

    #[test]
    fn build_section_filter_with_only_doc_id() {
        let f = build_section_filter(Some("doc-123"), None);
        let f = f.expect("filter should be Some when doc_id is provided");
        assert_eq!(f.must.len(), 1);
    }
}
