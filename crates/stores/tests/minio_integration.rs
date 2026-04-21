//! MinIO integration tests. Require a live MinIO instance via
//! `docker compose up -d minio` and a pre-created bucket
//! (`docker compose exec minio mc mb local/agent-rs-docs`).
//!
//! Feature-gated behind `integration-tests`.

#![cfg(feature = "integration-tests")]

use agent_rs_core::config::MinioConfig;
use agent_rs_core::types::UserId;
use agent_rs_stores::minio::{MinioStore, object_key};
use uuid::Uuid;

fn test_config() -> MinioConfig {
    MinioConfig {
        endpoint: std::env::var("AGENT_RS__MINIO__ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:1069".into()),
        access_key: std::env::var("AGENT_RS__MINIO__ACCESS_KEY")
            .unwrap_or_else(|_| "minioadmin".into()),
        secret_key: std::env::var("AGENT_RS__MINIO__SECRET_KEY")
            .unwrap_or_else(|_| "minioadmin".into()),
        bucket: std::env::var("AGENT_RS__MINIO__BUCKET").unwrap_or_else(|_| "agent-rs-docs".into()),
    }
}

#[tokio::test]
async fn put_get_roundtrip() {
    let store = MinioStore::new(test_config()).await.expect("connect");
    let user = UserId::try_new("alice-minio-test").expect("valid");
    let doc_id = Uuid::new_v4();

    let payload = b"hello world".to_vec();
    store
        .put_object(&user, doc_id, &payload, "application/pdf")
        .await
        .expect("put");

    let got = store.get_object(&user, doc_id).await.expect("get");
    assert_eq!(got, payload);

    store.delete_object(&user, doc_id).await.expect("delete");
}

#[tokio::test]
async fn cross_user_read_is_forbidden_by_key_namespacing() {
    let store = MinioStore::new(test_config()).await.expect("connect");
    let alice = UserId::try_new("alice-minio-cross").expect("valid");
    let bob = UserId::try_new("bob-minio-cross").expect("valid");
    let doc_id = Uuid::new_v4();

    store
        .put_object(&alice, doc_id, b"alice-secret", "application/pdf")
        .await
        .expect("alice put");

    // Bob asking for the same doc_id must fail — the key is under
    // users/alice-..., not users/bob-....
    let got = store.get_object(&bob, doc_id).await;
    assert!(got.is_err(), "bob must not get alice's object");

    store
        .delete_object(&alice, doc_id)
        .await
        .expect("alice delete");
}

#[tokio::test]
async fn delete_missing_object_is_idempotent() {
    let store = MinioStore::new(test_config()).await.expect("connect");
    let user = UserId::try_new("alice-minio-idem").expect("valid");
    let doc_id = Uuid::new_v4();
    // Never put — delete must still succeed (Ok).
    store
        .delete_object(&user, doc_id)
        .await
        .expect("idempotent delete");
}

#[test]
fn key_layout_is_user_scoped() {
    let user = UserId::try_new("alice-key-test").expect("valid");
    let doc_id = Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0);
    let key = object_key(&user, doc_id, "pdf");
    assert_eq!(
        key,
        "users/alice-key-test/docs/12345678-9abc-def0-1234-56789abcdef0.pdf"
    );
}
