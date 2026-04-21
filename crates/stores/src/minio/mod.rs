//! MinIO / S3-compatible object storage for per-user document bytes.
//!
//! Keys: `users/{user_id}/docs/{document_id}.{ext}`. User scoping is
//! by prefix — bucket policies are not required for correctness
//! because every read/write goes through this store and always
//! accepts a `&UserId`.

mod store;

pub use store::{MinioStore, object_key};
