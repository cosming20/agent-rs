//! Hybrid multi-backend search dispatcher.
//!
//! Fans out to vector + keyword + graph back-ends in parallel via
//! `tokio::join!` and fuses results with RRF via the ranker.
//! Full implementation will be added in Phase 3 Step 8.
