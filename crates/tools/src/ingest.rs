//! `ingest` tool -- triggers document ingestion.
//!
//! Currently a stub. In production, this will call the Python Docling
//! sidecar for document parsing and then feed the chunks through the
//! embedding + storage pipeline.

use agent_rs_core::ToolError;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---- Args ----

/// Arguments for the `ingest` tool.
#[derive(Debug, Deserialize)]
pub struct IngestArgs {
    /// The document path or raw content to ingest.
    pub source: String,
    /// Optional description of the document for metadata.
    pub description: Option<String>,
}

// ---- Tool struct ----

/// Triggers ingestion of a document into the knowledge base.
///
/// Phase 5 stub: returns a "not yet available" message. Production
/// implementation will call the Python Docling sidecar for document
/// parsing, then embed and store the resulting chunks.
#[derive(Serialize, Deserialize)]
pub struct IngestTool;

impl IngestTool {
    /// Create a new `IngestTool`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for IngestTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for IngestTool {
    const NAME: &'static str = "ingest";
    type Error = ToolError;
    type Args = IngestArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Ingest a document into the knowledge base. Provide either a file \
                          path or raw text content. The document will be parsed, chunked, \
                          embedded, and stored for future retrieval."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "File path or raw text content to ingest."
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the document for metadata."
                    }
                },
                "required": ["source"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!(
            source = %args.source,
            description = ?args.description,
            "ingestion requested (stub)"
        );

        // Phase 5b: call Python Docling sidecar, embed chunks, store in
        // vector + graph stores.
        Ok(format!(
            "Ingestion is not yet available. Source: '{}'. \
             Document ingestion requires the Docling sidecar service.",
            args.source
        ))
    }
}
