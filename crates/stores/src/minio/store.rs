//! S3-compatible object storage (MinIO by default).

use agent_rs_core::config::MinioConfig;
use agent_rs_core::errors::StoreError;
use agent_rs_core::types::UserId;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::{Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;
use uuid::Uuid;

/// Construct the canonical object key for a user's document.
///
/// Format: `users/{user_id}/docs/{document_id}.{ext}`
///
/// The user prefix ensures that list-based lookups are scoped to a
/// single user and cannot accidentally resolve another user's object.
pub fn object_key(user_id: &UserId, document_id: Uuid, ext: &str) -> String {
    format!("users/{}/docs/{}.{}", user_id.as_str(), document_id, ext)
}

/// Map a content-type string to a canonical filename extension.
fn content_type_to_ext(ct: &str) -> &'static str {
    match ct {
        "application/pdf" => "pdf",
        "text/html" => "html",
        "text/plain" => "txt",
        "text/markdown" => "md",
        _ => "bin",
    }
}

/// Per-user object storage backed by a MinIO (S3-compatible) instance.
///
/// All public methods require a `&UserId` so the key namespace is
/// always user-scoped at the call site — there is no escape hatch to
/// read or write across user boundaries through this type.
pub struct MinioStore {
    client: Client,
    bucket: String,
}

impl MinioStore {
    /// Connect to the configured MinIO endpoint. Does NOT create the
    /// bucket — that is a one-off bootstrap concern handled by
    /// `just minio-bootstrap`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Minio`] if the SDK config cannot be built
    /// (in practice this is infallible with static credentials, but the
    /// signature is kept fallible for forward-compatibility).
    pub async fn new(cfg: MinioConfig) -> Result<Self, StoreError> {
        let creds = Credentials::new(&cfg.access_key, &cfg.secret_key, None, None, "static");
        let s3_cfg = aws_sdk_s3::config::Builder::new()
            .endpoint_url(&cfg.endpoint)
            .region(Region::new("us-east-1"))
            .credentials_provider(creds)
            .force_path_style(true)
            .behavior_version_latest()
            .build();
        Ok(Self {
            client: Client::from_conf(s3_cfg),
            bucket: cfg.bucket,
        })
    }

    /// Upload `bytes` under the user's namespace and return the full
    /// object key.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Minio`] if the S3 `PutObject` call fails.
    pub async fn put_object(
        &self,
        user_id: &UserId,
        document_id: Uuid,
        bytes: &[u8],
        content_type: &str,
    ) -> Result<String, StoreError> {
        let ext = content_type_to_ext(content_type);
        let key = object_key(user_id, document_id, ext);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(bytes.to_vec()))
            .content_type(content_type)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("put_object {key}: {e}")))?;
        Ok(key)
    }

    /// Download and return the bytes for a document owned by `user_id`.
    ///
    /// The object key extension is discovered via a prefix listing so
    /// callers do not need to supply the content-type.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Minio`] if the object does not exist or
    /// the S3 call fails. Cross-user reads return an error because the
    /// prefix `users/{bob}/docs/{doc}` will never resolve an object
    /// stored under `users/{alice}/docs/{doc}`.
    pub async fn get_object(
        &self,
        user_id: &UserId,
        document_id: Uuid,
    ) -> Result<Vec<u8>, StoreError> {
        let key = self
            .resolve_existing_key(user_id, document_id)
            .await?
            .ok_or_else(|| {
                StoreError::Minio(format!(
                    "no object for user={} doc={}",
                    user_id.as_str(),
                    document_id
                ))
            })?;
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("get_object {key}: {e}")))?;
        let body = resp
            .body
            .collect()
            .await
            .map_err(|e| StoreError::Minio(format!("collect {key}: {e}")))?;
        Ok(body.to_vec())
    }

    /// Delete the document object for `user_id`. Idempotent — returns
    /// `Ok(())` even if the object never existed.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Minio`] if the S3 `DeleteObject` call fails.
    pub async fn delete_object(
        &self,
        user_id: &UserId,
        document_id: Uuid,
    ) -> Result<(), StoreError> {
        let Some(key) = self.resolve_existing_key(user_id, document_id).await? else {
            return Ok(()); // already gone — idempotent
        };
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("delete_object {key}: {e}")))?;
        Ok(())
    }

    /// Look up an object's full key by listing the `users/{u}/docs/{d}`
    /// prefix. The extension is discovered at runtime because callers
    /// of `get_object` / `delete_object` don't carry the content-type.
    async fn resolve_existing_key(
        &self,
        user_id: &UserId,
        document_id: Uuid,
    ) -> Result<Option<String>, StoreError> {
        let prefix = format!("users/{}/docs/{}", user_id.as_str(), document_id);
        let resp = self
            .client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(&prefix)
            .max_keys(1)
            .send()
            .await
            .map_err(|e| StoreError::Minio(format!("list_objects_v2 {prefix}: {e}")))?;
        let key = resp
            .contents
            .and_then(|v| v.into_iter().next())
            .and_then(|o| o.key);
        Ok(key)
    }
}
