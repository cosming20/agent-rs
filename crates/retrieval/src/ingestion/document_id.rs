//! Deterministic document UUID derivation.
//!
//! The document UUID is derived as a UUIDv5 of `"{user_id}|{sha256}"` in the
//! `NAMESPACE_OID` namespace.  Salting with `user_id` ensures two different
//! users uploading the same bytes receive distinct `document_id` values,
//! preventing PRIMARY KEY collisions on `ingested_documents.document_id`.
//!
//! Within a single user the derivation is stable: re-uploading the same file
//! always yields the same UUID, which powers the dedup path in
//! [`super::pipeline`] (the `(user_id, sha256)` unique index in Postgres).

use agent_rs_core::types::{UserId, NAMESPACE_OID};
use uuid::Uuid;

/// Derive the canonical document UUID for `(user_id, sha256)`.
///
/// The derivation is:
/// ```text
/// UUIDv5(NAMESPACE_OID, "{user_id}|{sha256}")
/// ```
///
/// # Parameters
///
/// - `user_id` — the tenant who owns the document.
/// - `sha256` — hex-encoded SHA-256 digest of the raw document bytes.
///
/// # Examples
///
/// ```rust,ignore
/// let id = derive_document_id(&user_id, &sha256_hex);
/// ```
pub fn derive_document_id(user_id: &UserId, sha256: &str) -> Uuid {
    let name = format!("{}|{}", user_id.as_str(), sha256);
    Uuid::new_v5(&NAMESPACE_OID, name.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_document_id_is_user_scoped() {
        let alice = UserId::try_new("alice").expect("valid user id");
        let bob = UserId::try_new("bob").expect("valid user id");
        let sha = "deadbeef00000000000000000000000000000000000000000000000000000000";

        let alice_id = derive_document_id(&alice, sha);
        let bob_id = derive_document_id(&bob, sha);

        assert_ne!(
            alice_id, bob_id,
            "different users uploading the same file must yield different document_ids"
        );
        assert_eq!(
            derive_document_id(&alice, sha),
            alice_id,
            "derivation must be stable for the same (user_id, sha256)"
        );
    }

    #[test]
    fn derive_document_id_same_user_is_stable() {
        let user = UserId::try_new("carol").expect("valid user id");
        let sha = "cafebabe00000000000000000000000000000000000000000000000000000000";
        assert_eq!(
            derive_document_id(&user, sha),
            derive_document_id(&user, sha),
            "same (user_id, sha256) must always produce the same UUID"
        );
    }
}
