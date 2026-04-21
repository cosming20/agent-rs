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
