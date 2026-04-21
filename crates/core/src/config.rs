//! Centralized configuration for the agent-rs system.
//!
//! Uses `figment` with layered providers:
//! 1. Programmatic defaults via [`Serialized`]
//! 2. Optional `config.toml` file
//! 3. Environment variables with `AGENT_RS__` prefix and `__` nesting separator
//!
//! [`dotenvy`] loads `.env` files before figment extraction so that
//! `.env` values are available as process environment variables.

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::ConfigError;

// ---- Sub-configs ----

/// LLM provider configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct LlmConfig {
    pub api_key: String,
    pub model: String,
}

impl fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmConfig")
            .field("api_key", &"***")
            .field("model", &self.model)
            .finish()
    }
}

/// Qdrant vector database configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct QdrantConfig {
    pub host: String,
    pub http_port: u16,
    pub grpc_port: u16,
    pub api_key: String,
}

impl fmt::Debug for QdrantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QdrantConfig")
            .field("host", &self.host)
            .field("http_port", &self.http_port)
            .field("grpc_port", &self.grpc_port)
            .field("api_key", &"***")
            .finish()
    }
}

impl QdrantConfig {
    /// HTTP REST URL for the Qdrant instance.
    pub fn http_url(&self) -> String {
        format!("http://{}:{}", self.host, self.http_port)
    }

    /// gRPC URL for the Qdrant instance.
    pub fn grpc_url(&self) -> String {
        format!("http://{}:{}", self.host, self.grpc_port)
    }
}

/// PostgreSQL database configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct PostgresConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub db: String,
}

impl fmt::Debug for PostgresConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PostgresConfig")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("user", &self.user)
            .field("password", &"***")
            .field("db", &self.db)
            .finish()
    }
}

impl PostgresConfig {
    /// Construct the full connection DSN string.
    pub fn dsn(&self) -> String {
        format!(
            "postgres://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, self.db
        )
    }
}

/// Neo4j graph database configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct Neo4jConfig {
    pub host: String,
    pub bolt_port: u16,
    pub user: String,
    pub password: String,
}

impl fmt::Debug for Neo4jConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Neo4jConfig")
            .field("host", &self.host)
            .field("bolt_port", &self.bolt_port)
            .field("user", &self.user)
            .field("password", &"***")
            .finish()
    }
}

impl Neo4jConfig {
    /// Returns `host:port` for neo4rs (no `bolt://` prefix -- neo4rs
    /// takes host:port directly).
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.host, self.bolt_port)
    }
}

/// Agent orchestration configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub max_retries: usize,
    pub system_prompt_file: String,
    /// Reasoning-effort hint forwarded to OpenAI Chat Completions for
    /// gpt-5.x / o-series models. Valid values: `"minimal"`, `"low"`,
    /// `"medium"`, `"high"`. Empty string disables the field (older
    /// non-reasoning models simply ignore unknown fields, so leaving the
    /// default is safe). `"minimal"` skips most hidden thinking tokens,
    /// which is the largest single lever on final-answer latency for
    /// reasoning models.
    #[serde(default)]
    pub reasoning_effort: String,
}

/// Context window management configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContextConfig {
    pub max_messages: usize,
    pub max_chars_per_msg: usize,
}

/// Evidence pack configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvidenceConfig {
    pub max_evidence_chars: usize,
    pub max_context_chars: usize,
}

/// Response synthesis configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SynthesisConfig {
    pub max_snippet_length: usize,
}

/// Reciprocal rank fusion configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RankerConfig {
    pub rrf_k: usize,
}

/// Cross-encoder reranker provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RerankerProvider {
    /// Jina `jina-reranker-v2-base-multilingual` via `api.jina.ai`.
    #[default]
    Jina,
    /// ZeroEntropy `zerank-2` via `api.zeroentropy.dev`.
    ZeroEntropy,
}

/// ZeroEntropy inference latency mode. Maps to the `latency` field on
/// `POST /v1/models/rerank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LatencyMode {
    /// Request sub-second inference. 429 on rate-limit overflow (lower limit).
    #[default]
    Fast,
    /// Request higher-rate-limit inference, may take >10s.
    Slow,
}

/// Cross-encoder reranker configuration.
///
/// Provider selection is compile-time-safe via [`RerankerProvider`]. The
/// `score_threshold` filter enables calibrated-score culling for providers
/// that return calibrated relevance scores (ZeroEntropy does; Jina does not —
/// leave `None` when using Jina).
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct RerankerConfig {
    /// Reranker provider.
    #[serde(default)]
    pub provider: RerankerProvider,
    /// Optional override for the provider's default model. `None` = provider default.
    #[serde(default)]
    pub model: Option<String>,
    /// API key. Empty means "read from provider's env var at runtime".
    #[serde(default)]
    pub api_key: String,
    /// Optional score threshold: docs with `relevance_score < threshold` are
    /// dropped after reranking. Only useful when the provider emits
    /// calibrated scores (e.g. ZeroEntropy zerank-2). `None` disables.
    #[serde(default)]
    pub score_threshold: Option<f64>,
    /// Default inference latency mode (ZeroEntropy only).
    #[serde(default)]
    pub latency_mode: LatencyMode,
    /// Maximum concurrent rerank calls across the workspace.
    #[serde(default = "default_rerank_concurrency")]
    pub max_concurrency: usize,
    /// Disable the reranker entirely. Takes precedence over env-var detection.
    #[serde(default)]
    pub disabled: bool,
}

fn default_rerank_concurrency() -> usize {
    8
}

impl fmt::Debug for RerankerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankerConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &"***")
            .field("score_threshold", &self.score_threshold)
            .field("latency_mode", &self.latency_mode)
            .field("max_concurrency", &self.max_concurrency)
            .field("disabled", &self.disabled)
            .finish()
    }
}

/// Faithfulness verifier configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerifierConfig {
    pub confidence_threshold: f64,
}

/// Web search budget configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebBudgetConfig {
    pub max_searches: usize,
    pub max_news: usize,
    pub max_fetches: usize,
    pub timeout_s: f64,
    pub max_bytes: usize,
}

/// Brave Search API configuration.
#[derive(Clone, Deserialize, Serialize)]
pub struct BraveConfig {
    /// Brave Search API key (X-Subscription-Token header).
    pub api_key: String,
    /// Base URL for the Brave Search API.
    pub base_url: String,
    /// Request timeout in seconds.
    pub timeout_s: u64,
}

impl fmt::Debug for BraveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BraveConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("timeout_s", &self.timeout_s)
            .finish()
    }
}

/// URL fetch configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FetchConfig {
    pub max_length: usize,
    pub max_bytes: usize,
    pub timeout: u64,
    pub max_redirects: usize,
}

/// OpenTelemetry collector configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OtelConfig {
    pub host: String,
    pub collector_port: u16,
}

impl OtelConfig {
    /// Full HTTP endpoint for the OTel collector.
    pub fn endpoint(&self) -> String {
        format!("http://{}:{}", self.host, self.collector_port)
    }
}

/// Open Policy Agent configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpaConfig {
    pub host: String,
    pub port: u16,
}

impl OpaConfig {
    /// Full HTTP URL for the OPA instance.
    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// Retrieval system configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RetrievalConfig {
    pub dataset: String,
    pub offline: bool,
}

// ---- Ingest backend ----

/// Selects the PDF/document ingestion backend at runtime.
///
/// Controlled by `AGENT_RS__INGEST_BACKEND` env var.
/// Default: `docling` (sidecar path only).
///
/// See the Strangler-Fig pattern note in the spec: `DoclingWithLegacyFallback`
/// lets a controlled fraction of traffic validate the new path while the
/// legacy path remains operational.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestBackend {
    /// Use the legacy `pdf_extract` path.
    ///
    /// Requires the `legacy_pdf_extract` compile-time feature in `agent-rs-retrieval`.
    Legacy,
    /// Use the Docling sidecar exclusively. Errors propagate to the caller.
    ///
    /// Default — new deployments must explicitly opt in to the legacy path via
    /// `AGENT_RS__INGEST_BACKEND=legacy`.
    #[default]
    Docling,
    /// Try Docling first; fall back to legacy on errors where
    /// `triggers_legacy_fallback = true`.
    ///
    /// On `retryable = true` errors, retry Docling once with backoff before
    /// falling back.
    DoclingWithLegacyFallback,
}

/// Configuration for the Docling sidecar gRPC connection.
///
/// All fields are overridable via `AGENT_RS__DOCLING__<FIELD>` env vars.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DoclingConfig {
    /// Full gRPC endpoint URL, e.g. `http://localhost:1068`.
    ///
    /// Override via `AGENT_RS__DOCLING__ENDPOINT`.
    pub endpoint: String,
    /// Timeout in seconds for establishing the gRPC channel.
    ///
    /// Override via `AGENT_RS__DOCLING__CONNECT_TIMEOUT_S`.
    pub connect_timeout_s: u64,
    /// Maximum idle time between stream messages before the client cancels
    /// the RPC (seconds).  Matches the sidecar's own idle-timeout so that
    /// the Rust client detects half-open connections promptly.
    ///
    /// Override via `AGENT_RS__DOCLING__STREAM_IDLE_TIMEOUT_S`.
    pub stream_idle_timeout_s: u64,
    /// Maximum document size the client will send (bytes).
    ///
    /// The sidecar enforces `min(server_default, request_value)` so this
    /// value cannot exceed the sidecar's configured cap.
    ///
    /// Override via `AGENT_RS__DOCLING__MAX_PDF_BYTES`.
    pub max_pdf_bytes: u32,
    /// Maximum page count the client will request.
    ///
    /// Override via `AGENT_RS__DOCLING__MAX_PAGES`.
    pub max_pages: u32,
}

/// MinIO / S3-compatible object storage config.
///
/// Loaded from `AGENT_RS__MINIO__*` env vars. Defaults are suitable
/// for the docker-compose dev stack (minioadmin/minioadmin on
/// localhost:1069); production deployments override via env.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MinioConfig {
    /// S3 API endpoint URL, e.g. `http://localhost:1069`.
    ///
    /// Override via `AGENT_RS__MINIO__ENDPOINT`.
    pub endpoint: String,
    /// S3 access key (MinIO root user).
    ///
    /// Override via `AGENT_RS__MINIO__ACCESS_KEY`.
    pub access_key: String,
    /// S3 secret key (MinIO root password).
    ///
    /// Override via `AGENT_RS__MINIO__SECRET_KEY`.
    pub secret_key: String,
    /// Bucket where user documents are stored.
    ///
    /// Override via `AGENT_RS__MINIO__BUCKET`.
    pub bucket: String,
}

impl Default for MinioConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:1069".into(),
            access_key: "minioadmin".into(),
            secret_key: "minioadmin".into(),
            bucket: "agent-rs-docs".into(),
        }
    }
}

/// RabbitMQ connection configuration for the indexing queue.
///
/// All values are layered: compiled-in defaults → `config.toml` →
/// `AGENT_RS__RABBITMQ__*` env vars. The default endpoint points at the
/// docker-compose-provided broker on ports 1075 (AMQP) / 1076 (management).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RabbitmqConfig {
    /// AMQP URI, e.g. `amqp://user:pass@host:port`.
    ///
    /// Override via `AGENT_RS__RABBITMQ__ENDPOINT`.
    pub endpoint: String,
}

impl Default for RabbitmqConfig {
    fn default() -> Self {
        Self {
            endpoint: "amqp://agentrs:agentrs-local@localhost:1075".into(),
        }
    }
}

/// gRPC server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// Host address the gRPC server binds to.
    pub host: String,
    /// Port the gRPC server listens on (default: 1059).
    pub port: u16,
    /// Maximum number of results a caller may request in a single query.
    ///
    /// Any `limit` field in [`QueryRequest`] is clamped to this value before
    /// being forwarded to the retrieval engine.
    ///
    /// Override via `AGENT_RS__SERVER__MAX_LIMIT_PER_QUERY`.
    pub max_limit_per_query: usize,
    /// Wall-clock timeout applied to every gRPC request (seconds).
    ///
    /// Requests that exceed this deadline are cancelled with
    /// `DEADLINE_EXCEEDED`. Protects the server from runaway agent pipelines.
    ///
    /// Override via `AGENT_RS__SERVER__REQUEST_TIMEOUT_S`.
    pub request_timeout_s: u64,
    /// Maximum number of concurrent HTTP/2 streams per connection.
    ///
    /// Maps directly to tonic's `max_concurrent_streams` setting.
    ///
    /// Override via `AGENT_RS__SERVER__MAX_CONCURRENT_STREAMS`.
    pub max_concurrent_streams: u32,
    /// Maximum number of in-flight gRPC calls per TCP connection.
    ///
    /// Maps directly to tonic's `concurrency_limit_per_connection` setting.
    ///
    /// Override via `AGENT_RS__SERVER__CONCURRENCY_PER_CONNECTION`.
    pub concurrency_per_connection: u32,
}

impl ServerConfig {
    /// Full bind address string (e.g., `[::]:1059`).
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ---- Root Settings ----

/// Root configuration struct containing all subsystem configs.
///
/// Load with [`Settings::load`] which layers defaults, config file,
/// and environment variables.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Settings {
    pub llm: LlmConfig,
    pub qdrant: QdrantConfig,
    pub postgres: PostgresConfig,
    pub neo4j: Neo4jConfig,
    pub agent: AgentConfig,
    pub context: ContextConfig,
    pub evidence: EvidenceConfig,
    pub synthesis: SynthesisConfig,
    pub ranker: RankerConfig,
    #[serde(default)]
    pub reranker: RerankerConfig,
    pub verifier: VerifierConfig,
    pub web_budget: WebBudgetConfig,
    pub brave: BraveConfig,
    pub fetch: FetchConfig,
    pub otel: OtelConfig,
    pub opa: OpaConfig,
    pub retrieval: RetrievalConfig,
    pub server: ServerConfig,
    /// Which PDF/document ingestion backend to use.
    pub ingest_backend: IngestBackend,
    /// Docling sidecar gRPC connection settings.
    pub docling: DoclingConfig,
    /// MinIO / S3-compatible object storage settings.
    #[serde(default)]
    pub minio: MinioConfig,
    /// RabbitMQ connection settings for the indexing queue.
    #[serde(default)]
    pub rabbitmq: RabbitmqConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            llm: LlmConfig {
                api_key: String::new(),
                model: "gpt-4o".into(),
            },
            qdrant: QdrantConfig {
                host: "localhost".into(),
                http_port: 1051,
                grpc_port: 1050,
                api_key: String::new(),
            },
            postgres: PostgresConfig {
                host: "localhost".into(),
                port: 1054,
                user: "agentic".into(),
                password: "agentic".into(),
                db: "agentic_rag".into(),
            },
            neo4j: Neo4jConfig {
                host: "localhost".into(),
                bolt_port: 1061,
                user: "neo4j".into(),
                password: "neo4j".into(),
            },
            agent: AgentConfig {
                // Raised from 4 to 5 for the coverage-aware-rag design.
                // Typical exhaustive path: outline_corpus -> search_sections
                // -> synthesize (3 turns); the extra turn provides headroom
                // for one targeted follow-up search_knowledge call when the
                // planner partially missed. Non-exhaustive flows still
                // finish in <=2 turns.
                max_steps: 5,
                max_retries: 1,
                system_prompt_file: String::new(),
                reasoning_effort: "minimal".into(),
            },
            context: ContextConfig {
                max_messages: 10,
                max_chars_per_msg: 2000,
            },
            evidence: EvidenceConfig {
                max_evidence_chars: 60000,
                max_context_chars: 3000,
            },
            synthesis: SynthesisConfig {
                max_snippet_length: 5000,
            },
            ranker: RankerConfig { rrf_k: 60 },
            reranker: RerankerConfig {
                provider: RerankerProvider::Jina,
                model: None,
                api_key: String::new(),
                score_threshold: None,
                latency_mode: LatencyMode::Fast,
                max_concurrency: 8,
                disabled: false,
            },
            verifier: VerifierConfig {
                confidence_threshold: 0.5,
            },
            web_budget: WebBudgetConfig {
                max_searches: 3,
                max_news: 2,
                max_fetches: 3,
                timeout_s: 45.0,
                max_bytes: 6_000_000,
            },
            brave: BraveConfig {
                api_key: String::new(),
                base_url: "https://api.search.brave.com/res/v1".into(),
                timeout_s: 15,
            },
            fetch: FetchConfig {
                max_length: 30000,
                max_bytes: 5_000_000,
                timeout: 30,
                max_redirects: 3,
            },
            otel: OtelConfig {
                host: "localhost".into(),
                collector_port: 1057,
            },
            opa: OpaConfig {
                host: "localhost".into(),
                port: 1055,
            },
            retrieval: RetrievalConfig {
                dataset: "default".into(),
                offline: false,
            },
            server: ServerConfig {
                host: "[::]".into(),
                port: 1059,
                max_limit_per_query: 200,
                request_timeout_s: 120,
                max_concurrent_streams: 256,
                concurrency_per_connection: 32,
            },
            ingest_backend: IngestBackend::default(),
            docling: DoclingConfig {
                endpoint: "http://localhost:1068".into(),
                connect_timeout_s: 10,
                stream_idle_timeout_s: 600,
                max_pdf_bytes: 100 * 1024 * 1024, // 100 MiB
                max_pages: 500,
            },
            minio: MinioConfig::default(),
            rabbitmq: RabbitmqConfig::default(),
        }
    }
}

impl Settings {
    /// Load settings: `.env` file -> defaults -> `config.toml` -> env vars (`AGENT_RS__` prefix).
    ///
    /// Calls [`dotenvy::dotenv().ok()`] first so `.env` file values are
    /// available as process env vars before figment extracts them.
    ///
    /// Env var mapping uses double-underscore as nesting separator:
    /// - `AGENT_RS__QDRANT__HOST` -> `qdrant.host`
    /// - `AGENT_RS__LLM__API_KEY` -> `llm.api_key`
    pub fn load() -> Result<Self, ConfigError> {
        // Load .env file — distinguish "not found" (ok) from "unreadable" (error)
        match dotenvy::dotenv() {
            Ok(path) => tracing::debug!("loaded .env from {}", path.display()),
            Err(dotenvy::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!("no .env file found; using env vars and defaults");
            }
            Err(e) => {
                return Err(ConfigError::LoadFailed(format!(".env load error: {e}")));
            }
        }

        Figment::from(Serialized::defaults(Settings::default()))
            .merge(Toml::file("config.toml"))
            .merge(Env::prefixed("AGENT_RS__").split("__"))
            .extract()
            .map_err(|e| ConfigError::LoadFailed(e.to_string()))
    }
}
