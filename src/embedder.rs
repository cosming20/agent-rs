//! Embedder initialization.

use std::sync::Arc;

use agent_rs_retrieval::embedder::{Embedder, FastEmbedder, FastEmbedderConfig};
use agent_rs_retrieval::openai_embedder::{OpenAIEmbedder, OpenAIEmbedderConfig};
use anyhow::{Context, Result};
use tracing::info;

/// Build the dense embedder based on `EMBEDDER_TYPE` env var.
///
/// - `"openai"` → OpenAI text-embedding-3-large (requires `OPENAI_API_KEY`)
/// - anything else → local fastembed ONNX model
///
/// Returns `(embedder, label)` where label is a human-readable string
/// like `"openai-text-embedding-3-large-3072"`.
pub async fn build_embedder() -> Result<(Arc<dyn Embedder + Send + Sync>, String)> {
    let embedder_type = std::env::var("EMBEDDER_TYPE").unwrap_or_else(|_| "local".to_owned());

    if embedder_type.eq_ignore_ascii_case("openai") {
        info!("initializing OpenAI dense embedder (sparse BM25 handled server-side by Qdrant)...");
        let api_key = std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("LLM_API_KEY"))
            .context("OPENAI_API_KEY or LLM_API_KEY must be set when EMBEDDER_TYPE=openai")?;
        let model = std::env::var("OPENAI_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-large".to_owned());
        let dims: usize = std::env::var("OPENAI_EMBEDDING_DIMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3072);

        let config = OpenAIEmbedderConfig::new(api_key)
            .with_model(model.clone())
            .with_dimensions(dims);
        let dense = OpenAIEmbedder::new(config);
        let label = format!("openai-{}-{}", model, dense.dimensions());
        Ok((Arc::new(dense) as Arc<dyn Embedder + Send + Sync>, label))
    } else {
        info!("initializing local fastembed embedder...");
        let emb = FastEmbedder::try_new(FastEmbedderConfig {
            show_download_progress: true,
            ..Default::default()
        })
        .await
        .context("initialize fastembed")?;
        let label = format!("fastembed-{}", emb.dimensions());
        Ok((Arc::new(emb), label))
    }
}
