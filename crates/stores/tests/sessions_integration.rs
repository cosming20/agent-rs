//! Integration tests for `PgStore::create_or_fetch_session`.
//!
//! Requires a live Postgres instance with migrations applied.
//! Start infrastructure via `just infra-up` before running.
//!
//! Feature-gated behind `integration-tests` to keep default `cargo test` fast.
//! Run with:
//! ```text
//! cargo test -p agent-rs-stores --features integration-tests --test sessions_integration
//! ```

#![cfg(feature = "integration-tests")]

use agent_rs_core::errors::StoreError;
use agent_rs_core::types::{SessionId, UserId};
use agent_rs_stores::postgres::PgStore;
use uuid::Uuid;

/// Construct a [`PgStore`] from the environment, falling back to the
/// well-known local dev URL from `docker-compose`.
async fn test_pg() -> PgStore {
    let url = std::env::var("AGENT_RS_DATABASE_URL")
        .or_else(|_| std::env::var("DATABASE_URL"))
        .unwrap_or_else(|_| {
            "postgres://agentic_ra:changeme-postgres@localhost:1054/agentic_ra".to_string()
        });
    PgStore::new(&url).await.expect("pg connect")
}

#[tokio::test]
async fn session_create_or_fetch_creates_new_row() {
    let pg = test_pg().await;
    let user = UserId::try_new("alice-test-3-1a").expect("valid user id");
    let session_id = SessionId::from_uuid(Uuid::new_v4());

    let row = pg
        .create_or_fetch_session(&user, &session_id)
        .await
        .expect("should create session");

    assert_eq!(&row.session_id, session_id.inner());
    assert_eq!(row.user_id, user.as_str());
}

#[tokio::test]
async fn session_create_or_fetch_returns_existing_for_same_user() {
    let pg = test_pg().await;
    let user = UserId::try_new("alice-test-3-1b").expect("valid user id");
    let session_id = SessionId::from_uuid(Uuid::new_v4());

    let first = pg
        .create_or_fetch_session(&user, &session_id)
        .await
        .expect("first call should succeed");
    let second = pg
        .create_or_fetch_session(&user, &session_id)
        .await
        .expect("second call should succeed");

    assert_eq!(first.session_id, second.session_id);
    assert_eq!(
        first.created_at, second.created_at,
        "idempotent call must not re-create the row"
    );
}

#[tokio::test]
async fn session_create_or_fetch_rejects_cross_user_hijack() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-1c").expect("valid user id");
    let bob = UserId::try_new("bob-test-3-1c").expect("valid user id");
    let session_id = SessionId::from_uuid(Uuid::new_v4());

    // Alice creates the session.
    pg.create_or_fetch_session(&alice, &session_id)
        .await
        .expect("alice should create session");

    // Bob must not be able to read or hijack alice's session.
    let err = pg
        .create_or_fetch_session(&bob, &session_id)
        .await
        .expect_err("bob must not access alice's session");

    match err {
        StoreError::SessionNotFound { .. } => {}
        other => panic!("expected StoreError::SessionNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn list_ingested_documents_scoped_to_user() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-2a").expect("valid");
    let bob = UserId::try_new("bob-test-3-2a").expect("valid");

    pg.test_insert_document_complete(&alice, "alice-3-2.pdf")
        .await
        .unwrap();
    pg.test_insert_document_complete(&bob, "bob-3-2.pdf")
        .await
        .unwrap();

    let alice_docs = pg.list_ingested_documents(&alice).await.unwrap();
    assert!(alice_docs.iter().all(|d| d.user_id == alice.as_str()));
    assert!(alice_docs.iter().any(|d| d.source == "alice-3-2.pdf"));
    assert!(
        alice_docs.iter().all(|d| d.source != "bob-3-2.pdf"),
        "alice must not see bob's doc"
    );

    let bob_docs = pg.list_ingested_documents(&bob).await.unwrap();
    assert!(bob_docs.iter().any(|d| d.source == "bob-3-2.pdf"));
    assert!(bob_docs.iter().all(|d| d.source != "alice-3-2.pdf"));
}

#[tokio::test]
async fn list_ingested_documents_filters_out_non_complete() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-2b").expect("valid");

    pg.test_insert_document_complete(&alice, "complete.pdf")
        .await
        .unwrap();
    pg.test_insert_document_with_status(&alice, "pending.pdf", "pending")
        .await
        .unwrap();
    pg.test_insert_document_with_status(&alice, "failed.pdf", "failed")
        .await
        .unwrap();

    let docs = pg.list_ingested_documents(&alice).await.unwrap();
    assert!(docs.iter().any(|d| d.source == "complete.pdf"));
    assert!(docs.iter().all(|d| d.source != "pending.pdf"));
    assert!(docs.iter().all(|d| d.source != "failed.pdf"));
}

#[tokio::test]
async fn two_users_same_sha_get_separate_document_ids() {
    let pg = test_pg().await;

    // Unique user ids + sha per run so re-running the test doesn't
    // hit the (user_id, sha256) unique index from a prior attempt.
    // The test validates per-user dedup, not idempotency of the test
    // fixture itself; prefixing with a UUID gives each run a fresh
    // namespace.
    let run = uuid::Uuid::new_v4();
    let run_hex = run.simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-2c-{}", &run_hex[..8])).expect("valid");
    let bob = UserId::try_new(&format!("bob-3-2c-{}", &run_hex[..8])).expect("valid");
    // sha256 is bpchar(64); compose a 64-char hex string from the run uuid.
    let shared_sha: String = format!("{run_hex}{run_hex}").chars().take(64).collect();

    let a = pg
        .test_insert_document_with_sha(&alice, "same.pdf", &shared_sha)
        .await
        .unwrap();
    let b = pg
        .test_insert_document_with_sha(&bob, "same.pdf", &shared_sha)
        .await
        .unwrap();

    assert_ne!(a.document_id, b.document_id);
    assert_eq!(a.user_id, alice.as_str());
    assert_eq!(b.user_id, bob.as_str());
    assert_eq!(a.sha256.trim(), shared_sha);
    assert_eq!(b.sha256.trim(), shared_sha);
}

// ---------------------------------------------------------------------------
// Task 3.3: memories + chat_messages scoped to (user_id, session_id)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn memories_scoped_to_user() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-3a").expect("valid");
    let bob = UserId::try_new("bob-test-3-3a").expect("valid");

    pg.save_memory(&alice, "alice knows Rust").await.unwrap();
    pg.save_memory(&bob, "bob prefers Python").await.unwrap();

    let alice_mem = pg.list_memories(&alice).await.unwrap();
    assert!(alice_mem.iter().any(|m| m.content.contains("Rust")));
    assert!(
        alice_mem.iter().all(|m| !m.content.contains("Python")),
        "alice must not see bob's memory"
    );
}

#[tokio::test]
async fn chat_messages_scoped_to_user_and_session() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-3b").expect("valid");
    let session = SessionId::from_uuid(Uuid::new_v4());
    pg.create_or_fetch_session(&alice, &session).await.unwrap();

    pg.append_chat_message(&alice, &session, "user", "hello from alice")
        .await
        .unwrap();

    let msgs = pg.list_chat_messages(&alice, &session).await.unwrap();
    assert!(msgs.iter().any(|m| m.content == "hello from alice"));
    assert!(msgs.iter().all(|m| m.user_id.as_str() == alice.as_str()));
}

#[tokio::test]
async fn chat_messages_cross_user_read_returns_empty() {
    let pg = test_pg().await;
    let alice = UserId::try_new("alice-test-3-3c").expect("valid");
    let bob = UserId::try_new("bob-test-3-3c").expect("valid");
    let session = SessionId::from_uuid(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session).await.unwrap();
    pg.append_chat_message(&alice, &session, "user", "private")
        .await
        .unwrap();

    // If someone bypasses authz and passes (bob, alice_session) directly,
    // the filter-by-user_id in the SELECT ensures zero rows — defence in depth.
    let bob_view = pg.list_chat_messages(&bob, &session).await.unwrap();
    assert!(bob_view.is_empty(), "bob must not see alice's chat");
}

// ---------------------------------------------------------------------------
// Task 3.4: session_active_documents with ownership-verified INSERT
// ---------------------------------------------------------------------------

#[tokio::test]
async fn activate_document_for_session_with_valid_owner() {
    let pg = test_pg().await;
    let run = uuid::Uuid::new_v4().simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-4a-{}", &run[..8])).expect("valid");
    let session = SessionId::from_uuid(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session).await.unwrap();
    let doc = pg
        .test_insert_document_complete(&alice, "alice-3-4.pdf")
        .await
        .unwrap();

    pg.activate_document_for_session(&alice, &session, &doc.document_id)
        .await
        .expect("valid owner");

    let active = pg.list_active_documents(&alice, &session).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].document_id, doc.document_id);
}

#[tokio::test]
async fn activate_document_rejects_cross_user() {
    let pg = test_pg().await;
    let run = uuid::Uuid::new_v4().simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-4b-{}", &run[..8])).expect("valid");
    let bob = UserId::try_new(&format!("bob-3-4b-{}", &run[..8])).expect("valid");
    let bob_session = SessionId::from_uuid(Uuid::new_v4());

    pg.create_or_fetch_session(&bob, &bob_session)
        .await
        .unwrap();
    let alice_doc = pg
        .test_insert_document_complete(&alice, "alice-3-4b.pdf")
        .await
        .unwrap();

    let err = pg
        .activate_document_for_session(&bob, &bob_session, &alice_doc.document_id)
        .await
        .expect_err("bob cannot activate alice's doc");

    match err {
        StoreError::DocumentNotOwnedByUser { .. } => {}
        other => panic!("expected DocumentNotOwnedByUser, got {other:?}"),
    }
}

#[tokio::test]
async fn activate_document_idempotent() {
    let pg = test_pg().await;
    let run = uuid::Uuid::new_v4().simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-4c-{}", &run[..8])).expect("valid");
    let session = SessionId::from_uuid(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session).await.unwrap();
    let doc = pg
        .test_insert_document_complete(&alice, "alice-3-4c.pdf")
        .await
        .unwrap();

    // First activation succeeds.
    pg.activate_document_for_session(&alice, &session, &doc.document_id)
        .await
        .unwrap();
    // Second activation (same session + doc) must NOT error — idempotent.
    pg.activate_document_for_session(&alice, &session, &doc.document_id)
        .await
        .unwrap();

    let active = pg.list_active_documents(&alice, &session).await.unwrap();
    assert_eq!(
        active.len(),
        1,
        "duplicate activations must collapse to one row"
    );
}

/// Regression test for Codex Phase 3 Bug 1: idempotency fallback must
/// re-verify ownership rather than matching any existing (session, doc) row.
///
/// Before the fix, bob calling `activate(bob, alice_session, alice_doc)` after
/// alice had already activated her doc would pass the fallback COUNT check and
/// return `Ok(())`, silently succeeding without owning anything.
#[tokio::test]
async fn activate_document_idempotency_does_not_leak_cross_user() {
    let pg = test_pg().await;
    let run = uuid::Uuid::new_v4().simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-4e-{}", &run[..8])).expect("valid");
    let bob = UserId::try_new(&format!("bob-3-4e-{}", &run[..8])).expect("valid");
    let alice_session = SessionId::from_uuid(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &alice_session)
        .await
        .unwrap();
    let alice_doc = pg
        .test_insert_document_complete(&alice, "alice.pdf")
        .await
        .unwrap();

    // Alice activates her doc — creates the row.
    pg.activate_document_for_session(&alice, &alice_session, &alice_doc.document_id)
        .await
        .unwrap();

    // Bob MUST NOT succeed on the existing row.
    let err = pg
        .activate_document_for_session(&bob, &alice_session, &alice_doc.document_id)
        .await
        .expect_err("bob must not succeed on an existing alice activation");
    match err {
        StoreError::DocumentNotOwnedByUser { .. } => {}
        other => panic!("expected DocumentNotOwnedByUser, got {other:?}"),
    }
}

#[tokio::test]
async fn activate_document_rejects_non_complete_status() {
    let pg = test_pg().await;
    let run = uuid::Uuid::new_v4().simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-4d-{}", &run[..8])).expect("valid");
    let session = SessionId::from_uuid(Uuid::new_v4());

    pg.create_or_fetch_session(&alice, &session).await.unwrap();
    let pending = pg
        .test_insert_document_with_status(&alice, "pending.pdf", "pending")
        .await
        .unwrap();

    let err = pg
        .activate_document_for_session(&alice, &session, &pending.document_id)
        .await
        .expect_err("cannot activate pending doc");

    match err {
        StoreError::DocumentNotOwnedByUser { .. } => {}
        other => panic!("expected DocumentNotOwnedByUser (doc not 'complete'), got {other:?}"),
    }
}

/// Regression test for Codex Phase 3 Bug 2: sha256 lookups must be scoped to
/// the requesting user, not global across all tenants.
///
/// Before the fix, bob could call `is_already_ingested_complete(sha256)` or
/// `get_ingested_document_by_sha256(sha256)` and receive positive results for
/// alice's ingestion, leaking the fact that alice had processed the file.
#[tokio::test]
async fn sha256_lookup_is_user_scoped() {
    let pg = test_pg().await;
    let run = uuid::Uuid::new_v4().simple().to_string();
    let alice = UserId::try_new(&format!("alice-3-sha-{}", &run[..8])).expect("valid");
    let bob = UserId::try_new(&format!("bob-3-sha-{}", &run[..8])).expect("valid");
    let shared_sha: String = format!("{}{}", &run, &run).chars().take(64).collect();

    pg.test_insert_document_with_sha(&alice, "alice.pdf", &shared_sha)
        .await
        .unwrap();

    // Bob probes with the same sha256 — MUST not see alice's doc.
    let bob_has = pg
        .is_already_ingested_complete(&bob, &shared_sha)
        .await
        .unwrap();
    assert!(!bob_has, "bob must not see alice's ingest via sha probe");

    let bob_fetch = pg
        .get_ingested_document_by_sha256(&bob, &shared_sha)
        .await
        .unwrap();
    assert!(
        bob_fetch.is_none(),
        "bob must not retrieve alice's row by sha"
    );

    // Alice herself must still see her own doc.
    let alice_has = pg
        .is_already_ingested_complete(&alice, &shared_sha)
        .await
        .unwrap();
    assert!(alice_has, "alice must still find her own doc");
}
