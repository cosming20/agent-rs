//! Build Qdrant `Filter` clauses for payload keys.
//!
//! These are the exact payload keys the ingestion pipeline writes
//! (see `crates/retrieval/src/ingestion/pipeline.rs` payload construction).
//!
//! ## Multi-tenancy
//!
//! Use [`build_user_filter`] for all new code. Every chunk read from Qdrant
//! must be scoped to a `user_id`; passing an empty string panics because
//! strict multi-tenancy has no valid "no user" code path.
//!
//! [`build_section_filter`] is deprecated (retained for one release cycle
//! until Phase 6/7 migrates all callers).

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
#[deprecated(
    since = "0.2.0",
    note = "use build_user_filter — strict multi-tenancy requires user_id"
)]
#[allow(dead_code)]
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

/// Build a **mandatory** user-scoped Qdrant filter.
///
/// Unlike the legacy [`build_section_filter`], `user_id` is required here:
/// passing an empty string panics, because strict multi-tenancy has no valid
/// "no user" code path. Optional `doc_id` / `section_id` narrow the filter
/// further within the user's scope.
///
/// # Panics
///
/// Panics when `user_id` is empty — callers must supply a non-empty id.
///
/// # Examples
///
/// ```rust
/// use agent_rs_stores::qdrant::section_filter::build_user_filter;
///
/// let f = build_user_filter("alice", None, None);
/// assert_eq!(f.must.len(), 1);
///
/// let f2 = build_user_filter("alice", Some("doc-1"), Some("sec-1"));
/// assert_eq!(f2.must.len(), 3);
/// ```
pub fn build_user_filter(user_id: &str, doc_id: Option<&str>, section_id: Option<&str>) -> Filter {
    assert!(
        !user_id.is_empty(),
        "user_id must not be empty for Qdrant user filter"
    );
    let mut conds: Vec<Condition> = Vec::new();
    conds.push(Condition::matches("user_id", user_id.to_string()));
    if let Some(did) = doc_id {
        conds.push(Condition::matches("doc_id", did.to_string()));
    }
    if let Some(sid) = section_id {
        conds.push(Condition::matches("section_id", sid.to_string()));
    }
    Filter {
        must: conds,
        ..Default::default()
    }
}

/// Merge a mandatory user-scope filter with an optional caller-supplied filter.
///
/// Used where a caller already holds a narrowed [`Filter`] (e.g., custom
/// metadata constraints) that must be ANDed together with the mandatory
/// user-scope filter. Preserves `must_not` and `should` from the caller's
/// filter; prepends the user clause into `must`.
///
/// # Examples
///
/// ```rust
/// use agent_rs_stores::qdrant::section_filter::{build_user_filter, merge_user_filter};
///
/// let user_scope = build_user_filter("alice", None, None);
/// let merged = merge_user_filter(user_scope, None);
/// assert_eq!(merged.must.len(), 1);
/// ```
pub fn merge_user_filter(user_scope: Filter, extra: Option<Filter>) -> Filter {
    let Some(extra) = extra else {
        return user_scope;
    };
    let mut must = user_scope.must;
    must.extend(extra.must);
    Filter {
        must,
        must_not: extra.must_not,
        should: extra.should,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_section_filter_with_doc_and_section() {
        #[allow(deprecated)]
        let f = build_section_filter(Some("doc-123"), Some("sec-456"));
        let f = f.expect("filter should be Some when at least one id is provided");
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    fn build_section_filter_returns_none_when_both_args_are_none() {
        #[allow(deprecated)]
        let result = build_section_filter(None, None);
        assert!(result.is_none());
    }

    #[test]
    fn build_section_filter_with_only_doc_id() {
        #[allow(deprecated)]
        let f = build_section_filter(Some("doc-123"), None);
        let f = f.expect("filter should be Some when doc_id is provided");
        assert_eq!(f.must.len(), 1);
    }

    #[test]
    fn build_user_filter_with_user_only() {
        let f = build_user_filter("alice", None, None);
        assert_eq!(f.must.len(), 1, "user_id only -> 1 must clause");
    }

    #[test]
    fn build_user_filter_with_user_doc_section() {
        let f = build_user_filter("alice", Some("doc-1"), Some("sec-1"));
        assert_eq!(f.must.len(), 3, "user+doc+section -> 3 must clauses");
    }

    #[test]
    fn build_user_filter_with_user_and_doc() {
        let f = build_user_filter("alice", Some("doc-1"), None);
        assert_eq!(f.must.len(), 2);
    }

    #[test]
    #[should_panic(expected = "user_id must not be empty")]
    fn build_user_filter_panics_on_empty_user() {
        // Strict multi-tenancy has no "no user" case. An empty user_id
        // must panic — silently returning an unscoped filter would allow
        // cross-tenant reads.
        let _ = build_user_filter("", None, None);
    }
}
