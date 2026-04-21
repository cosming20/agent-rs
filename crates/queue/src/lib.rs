//! RabbitMQ transport for the agent-rs indexing pipeline.
//!
//! Two-way split between the producer (the gRPC service, which publishes a
//! job when the web app calls `EnqueueIndex`) and the consumer (the
//! `IndexerWorker`, which runs chunking + embedding + Qdrant/Neo4j upserts).
//!
//! ```text
//!    AskRequest                       RabbitMQ
//!     (web → gRPC)                  "agent-rs.index"
//!         │                               │
//!         ▼                               ▼
//!   ┌──────────────┐   IndexJob    ┌──────────────┐
//!   │  Publisher   │ ────────────► │  Consumer    │
//!   │  (this crate)│               │  (IndexerWorker)
//!   └──────────────┘               └──────────────┘
//! ```
//!
//! The contract carries a stable job-id so operator logs and the
//! `document_index_state` table can be correlated across hops.

pub mod error;
pub mod publisher;
pub mod types;
pub mod worker;

pub use error::QueueError;
pub use publisher::IndexPublisher;
pub use types::{IndexJob, QUEUE_NAME};
pub use worker::{IndexHandler, run_indexer};
