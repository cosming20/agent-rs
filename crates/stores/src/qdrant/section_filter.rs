//! Build Qdrant `Filter` clauses for payload keys.
//!
//! These are the payload keys the indexer (RabbitMQ `IndexerWorker`,
//! `crates/queue`) writes on each chunk point.
//!
//! ## Multi-tenancy
//!
//! Every chunk read from Qdrant must be scoped to a `user_id`; passing an
//! empty string panics because strict multi-tenancy has no valid "no user"
//! code path. Optional `minio_object_key` / `section_id` narrow the filter
//! further within the user's scope.
//!
//! ## Payload field naming
//!
//! After the 2026-04-21 stateless pivot the canonical document identifier
//! is the MinIO object key (matching the `AskRequest.active_document_keys`
//! shape on the wire). The Qdrant payload therefore carries
//! `minio_object_key` rather than the legacy `doc_id` field.

use qdrant_client::qdrant::{Condition, Filter};

/// Qdrant payload key used to scope chunks to their owning document.
///
/// Matches the canonical key emitted by `IndexerWorker` and the
/// `minio_object_key` field on the proto `Citation` message.
const PAYLOAD_KEY_DOCUMENT: &str = "minio_object_key";

/// Qdrant payload key used to scope chunks to a section.
const PAYLOAD_KEY_SECTION: &str = "section_id";

/// Qdrant payload key used to scope chunks to their owner user.
const PAYLOAD_KEY_USER: &str = "user_id";

/// Build a **mandatory** user-scoped Qdrant filter.
///
/// `user_id` is required: passing an empty string panics, because strict
/// multi-tenancy has no valid "no user" code path. Optional
/// `minio_object_key` / `section_id` narrow the filter further within the
/// user's scope.
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
/// let f2 = build_user_filter(
///     "alice",
///     Some("users/alice/docs/abc.pdf"),
///     Some("sec-1"),
/// );
/// assert_eq!(f2.must.len(), 3);
/// ```
pub fn build_user_filter(
    user_id: &str,
    minio_object_key: Option<&str>,
    section_id: Option<&str>,
) -> Filter {
    assert!(
        !user_id.is_empty(),
        "user_id must not be empty for Qdrant user filter"
    );
    let mut conds: Vec<Condition> = Vec::new();
    conds.push(Condition::matches(
        PAYLOAD_KEY_USER,
        user_id.to_string(),
    ));
    if let Some(key) = minio_object_key {
        conds.push(Condition::matches(PAYLOAD_KEY_DOCUMENT, key.to_string()));
    }
    if let Some(sid) = section_id {
        conds.push(Condition::matches(PAYLOAD_KEY_SECTION, sid.to_string()));
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
    fn build_user_filter_with_user_only() {
        let f = build_user_filter("alice", None, None);
        assert_eq!(f.must.len(), 1, "user_id only -> 1 must clause");
    }

    #[test]
    fn build_user_filter_with_user_doc_section() {
        let f = build_user_filter(
            "alice",
            Some("users/alice/docs/abc.pdf"),
            Some("sec-1"),
        );
        assert_eq!(f.must.len(), 3, "user+doc+section -> 3 must clauses");
    }

    #[test]
    fn build_user_filter_with_user_and_doc() {
        let f = build_user_filter("alice", Some("users/alice/docs/abc.pdf"), None);
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
