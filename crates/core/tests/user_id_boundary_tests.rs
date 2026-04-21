//! Boundary tests for [`agent_rs_core::types::UserId`] validation.
//!
//! These tests exercise the security-critical rejection paths of
//! [`UserId::try_new`] across several attack classes.  No live
//! infrastructure is needed — they run on every plain `cargo test`.
//!
//! The goal is to demonstrate that `UserId` acts as an input-sanitation
//! gate before any user-controlled string reaches SQL, Cypher, or vector
//! payload filter contexts.
//!
//! ## Coverage
//! - SQL injection patterns
//! - Path traversal patterns
//! - Unicode homoglyph patterns
//! - Null byte injection
//! - Boundary lengths (too-long / exact max)
//! - Valid identifier forms

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::UserId;

// ---- SQL injection ----

/// Every attempt must be rejected with [`AgentError::InvalidUserId`].
///
/// Rejection happens because the ASCII-alphanumeric + `[_-]` allowlist
/// blocks the `'`, `"`, `;`, space, `=`, and `*` characters that all
/// SQL injection patterns require.
#[test]
fn sql_injection_via_user_id_rejected_at_type_boundary() {
    let attempts = [
        "'; DROP TABLE users; --",
        "admin'--",
        "' OR 1=1 --",
        "\"; DROP TABLE x;",
        "' UNION SELECT * FROM secrets --",
        "1; UPDATE sessions SET user_id='admin' WHERE 1=1;",
        "/**/OR/**/1=1",
        "' AND SLEEP(5)--",
    ];
    for attempt in &attempts {
        let result = UserId::try_new(attempt);
        assert!(
            matches!(result, Err(AgentError::InvalidUserId(_))),
            "SQL injection attempt must be rejected: {attempt:?}, got {result:?}",
        );
    }
}

// ---- Path traversal ----

/// Path-traversal sequences must be rejected.
///
/// The `..` and `/` characters are not in the `[A-Za-z0-9_-]` allowlist,
/// so they trigger the disallowed-character error rather than reaching any
/// filesystem or object-store key-building logic.
#[test]
fn path_traversal_via_user_id_rejected() {
    let attempts = [
        "../alice",
        "alice/../bob",
        "/etc/passwd",
        "../../../",
        "alice/bob",
        "..%2Falice",
    ];
    for attempt in &attempts {
        assert!(
            UserId::try_new(attempt).is_err(),
            "path traversal must be rejected: {attempt:?}",
        );
    }
}

// ---- Unicode homoglyphs ----

/// Visually similar Unicode characters are rejected.
///
/// The ASCII-only allowlist means that Cyrillic, Greek, or other lookalike
/// code points cannot create a user ID that collides with a legitimate
/// ASCII-only identifier when compared visually.
#[test]
fn unicode_homoglyphs_rejected_at_type_boundary() {
    // Cyrillic SMALL LETTER A (U+0430) — looks identical to Latin 'a' in most fonts.
    let cyrillic_a = "аlice"; // first char is U+0430
    assert!(
        UserId::try_new(cyrillic_a).is_err(),
        "Cyrillic 'а' homoglyph must be rejected to prevent collision with 'alice'",
    );

    // Greek SMALL LETTER ALPHA (U+03B1) — another common homoglyph.
    let greek_alpha = "αlice"; // first char is U+03B1
    assert!(
        UserId::try_new(greek_alpha).is_err(),
        "Greek 'α' homoglyph must be rejected",
    );

    // Full-width ASCII digit (U+FF10–U+FF19) — another homoglyph class.
    let fullwidth_zero = "user０one"; // U+FF10
    assert!(
        UserId::try_new(fullwidth_zero).is_err(),
        "Full-width digit homoglyph must be rejected",
    );
}

// ---- Null byte ----

/// Null bytes must be rejected.
///
/// A null byte embedded in a string can terminate C-style strings early and
/// bypass length checks in some contexts.  The character is not in the
/// allowlist, so it is caught at construction time.
#[test]
fn null_byte_rejected() {
    assert!(
        UserId::try_new("alice\0bob").is_err(),
        "null byte must be rejected",
    );
    assert!(
        UserId::try_new("\0").is_err(),
        "lone null byte must be rejected",
    );
}

// ---- Length boundaries ----

#[test]
fn user_id_at_max_length_is_accepted() {
    let max = "a".repeat(UserId::MAX_LEN);
    assert!(
        UserId::try_new(&max).is_ok(),
        "exactly {} chars must be accepted",
        UserId::MAX_LEN,
    );
}

#[test]
fn user_id_one_over_max_length_is_rejected() {
    let over = "a".repeat(UserId::MAX_LEN + 1);
    assert!(
        matches!(UserId::try_new(&over), Err(AgentError::InvalidUserId(_))),
        "one character over max length must be rejected",
    );
}

// ---- Valid forms ----

/// Confirm that identifiers with only the allowed characters succeed.
#[test]
fn valid_user_id_forms_accepted() {
    let valid = [
        "alice",
        "alice_01",
        "user-123",
        "ABC",
        "User-ID_42",
        "a",
        "0",
    ];
    for id in &valid {
        assert!(
            UserId::try_new(id).is_ok(),
            "valid user id must be accepted: {id:?}",
        );
    }
}
