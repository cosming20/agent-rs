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

// ---- Constants ----

/// Default model identifier used across every LLM-calling component
/// (summarizer, KG extractor, entity linker, agent orchestrator) when
/// no explicit override is configured.
///
/// Centralised here so a version bump touches one line instead of
/// fanning out across every binary and every config fixture. Deployments
/// that want a different model set `AGENT_RS__LLM__MODEL` or pass
/// `--model` / the component-specific override env var.
pub const DEFAULT_LLM_MODEL: &str = "gpt-5.4-mini-2026-03-17";

/// Return the `temperature` to serialize for an OpenAI Chat Completion
/// request, or `None` when the field must be omitted from the body.
///
/// gpt-5.x / o1 / o3 **reasoning models** only accept the default
/// `temperature` of `1`; sending any other value (including `0.0` for
/// deterministic extraction) returns HTTP 400
/// `Unsupported value: 'temperature' does not support X`. For those
/// models, return `None` — call sites thread this through
/// `#[serde(skip_serializing_if = "Option::is_none")]` or conditional
/// `json!{}` inclusion so the field is dropped from the wire body.
///
/// For non-reasoning models (gpt-4o, gpt-4.1, older) the caller's
/// `preferred` value is forwarded. Callers should pass the value they
/// *want* (e.g. `0.0` for deterministic JSON extraction); this helper
/// either honours it or omits it depending on what the model supports.
///
/// Determinism on reasoning models comes from `reasoning_effort` +
/// schema-constrained structured output, not from `temperature`.
pub fn temperature_for(model: &str, preferred: f32) -> Option<f32> {
    if is_reasoning_model(model) {
        None
    } else {
        Some(preferred)
    }
}

/// Return `true` iff `model` belongs to a family that rejects
/// non-default `temperature`. Currently the gpt-5.x, o1-*, o3-* lines;
/// extend here when OpenAI ships new families with the same constraint.
fn is_reasoning_model(model: &str) -> bool {
    model.starts_with("gpt-5") || model.starts_with("o1") || model.starts_with("o3")
}

/// Default `reasoning_effort` forwarded to OpenAI Chat Completions for
/// gpt-5.x / o-series reasoning models.
///
/// OpenAI's enum for this field shifted over the lifetime of gpt-5.x:
/// the previously-valid `"minimal"` now returns HTTP 400 on
/// `gpt-5.4-mini-2026-03-17` and related models. The current accepted
/// set is `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`. `"low"`
/// is the lowest setting that still leaves non-zero reasoning on (as
/// opposed to `"none"` which disables it entirely), which is the
/// latency/quality sweet spot for summarisation + KG extraction +
/// metadata-filter rewriting — the places that rely on *some* hidden
/// reasoning to avoid degraded structured-output quality.
///
/// Centralised here so a future enum shift is a one-line change, same
/// rationale as [`DEFAULT_LLM_MODEL`].
pub const DEFAULT_REASONING_EFFORT: &str = "low";

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
    /// Hard cap on simultaneous pooled connections per process. Must be
    /// sized against Postgres's `max_connections` so all agent-rs /
    /// indexer-worker / web replicas fit. Override via
    /// `AGENT_RS__POSTGRES__MAX_CONNECTIONS`.
    #[serde(default = "default_pg_max_connections")]
    pub max_connections: u32,
    /// How long a caller waits for a free pool slot before the request
    /// errors. Keep strictly below any outer request deadline so failures
    /// surface at the pool layer, not as a client timeout. Override via
    /// `AGENT_RS__POSTGRES__CONNECTION_TIMEOUT_MS`.
    #[serde(default = "default_pg_connection_timeout_ms")]
    pub connection_timeout_ms: u64,
}

fn default_pg_max_connections() -> u32 {
    50
}

fn default_pg_connection_timeout_ms() -> u64 {
    5_000
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
    /// gpt-5.x / o-series models. Valid values as of
    /// `gpt-5.4-mini-2026-03-17`: `"none"`, `"low"`, `"medium"`,
    /// `"high"`, `"xhigh"`. Empty string disables the field (non-reasoning
    /// models ignore unknown fields, so leaving the default is safe).
    ///
    /// See [`DEFAULT_REASONING_EFFORT`] for rationale on the low-setting
    /// default.
    #[serde(default)]
    pub reasoning_effort: String,
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
    /// Number of top-evidence items to use as fallback citations when the
    /// LLM produced no inline `[cite:<id>]` markers in its answer. Tunable
    /// because it is a UX/quality trade-off: too few hides the source set
    /// from the user, too many dilutes attribution.
    ///
    /// Override via `AGENT_RS__SYNTHESIS__FALLBACK_CITATION_COUNT`.
    #[serde(default = "default_fallback_citation_count")]
    pub fallback_citation_count: usize,
}

fn default_fallback_citation_count() -> usize {
    5
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
    /// Minimum confidence threshold for the heuristic fallback verifier
    /// to return `is_grounded = true`. The LLM-based verifier ignores
    /// this field; its verdict is derived from typed claims.
    pub confidence_threshold: f64,
    /// Optional model override for the grounding verifier. When `None`
    /// (default), the verifier uses the global `llm.model`. Set this to
    /// a stronger model (e.g. `gpt-5.4-mini`) to improve claim
    /// classification quality without upgrading the primary retrieval
    /// LLM. Reads from `AGENT_RS__VERIFIER__MODEL`.
    #[serde(default)]
    pub model: Option<String>,
}

/// Grounding gate heuristic configuration.
///
/// Controls the language-agnostic subject-presence check that runs before
/// the LLM verifier as a cheap pre-filter. Centralised here so corpus
/// operators can tune the heuristic per dataset (e.g. a corpus of
/// short-form Q&A may want `subject_min_token_len = 3`) without a
/// recompile.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GroundingConfig {
    /// Minimum character length for a query token to be considered a
    /// content-bearing word. Tokens shorter than this are treated as
    /// function words / particles regardless of language.
    ///
    /// `4` is the empirical sweet spot across EN, RO, ES, DE — it drops
    /// pronouns and short conjunctions while preserving real nouns. Bump
    /// to `5` for noisier corpora; lower only with care.
    ///
    /// Override via `AGENT_RS__GROUNDING__SUBJECT_MIN_TOKEN_LEN`.
    #[serde(default = "default_subject_min_token_len")]
    pub subject_min_token_len: usize,
}

fn default_subject_min_token_len() -> usize {
    4
}

impl Default for GroundingConfig {
    fn default() -> Self {
        Self {
            subject_min_token_len: default_subject_min_token_len(),
        }
    }
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
    /// Maximum number of results to surface to the agent per
    /// `search_web` / `search_news` invocation. Brave returns more than
    /// this in many cases — capping the formatted output keeps token
    /// pressure on the agent predictable.
    ///
    /// Override via `AGENT_RS__BRAVE__MAX_RESULTS`.
    #[serde(default = "default_brave_max_results")]
    pub max_results: usize,
}

fn default_brave_max_results() -> usize {
    10
}

impl fmt::Debug for BraveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BraveConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("timeout_s", &self.timeout_s)
            .field("max_results", &self.max_results)
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
    /// Use the Marker sidecar exclusively. Marker is MPS-native on Apple
    /// Silicon (surya layout + texify formulas) — ~10-20× faster than
    /// Docling on macOS for formula-heavy PDFs while matching quality on
    /// well-typeset printed math.
    Marker,
    /// Try Marker first; fall back to legacy on errors where
    /// `triggers_legacy_fallback = true`. Useful when Marker is the
    /// primary but the deployment still needs a graceful degradation
    /// path for malformed PDFs Marker rejects.
    MarkerWithLegacyFallback,
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

/// Configuration for the Marker sidecar gRPC connection.
///
/// Same shape as [`DoclingConfig`] — Marker is a drop-in backend that
/// speaks the same proto. All fields override via
/// `AGENT_RS__MARKER__<FIELD>` env vars.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MarkerConfig {
    /// Full gRPC endpoint URL, e.g. `http://localhost:1073`.
    ///
    /// Override via `AGENT_RS__MARKER__ENDPOINT`. Defaults to 1073
    /// (not 1069 — that's taken by MinIO in the dev compose stack).
    /// Docling on 1068 can run in parallel for A/B comparison.
    pub endpoint: String,
    /// Timeout in seconds for establishing the gRPC channel.
    pub connect_timeout_s: u64,
    /// Maximum idle time between stream messages (seconds).
    pub stream_idle_timeout_s: u64,
    /// Maximum document size the client will send (bytes).
    pub max_pdf_bytes: u32,
    /// Maximum page count the client will request.
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
    /// Global cap on in-flight gRPC calls across every connection. The
    /// tower `ConcurrencyLimitLayer` shed-sheds above this: callers
    /// receive `Status::resource_exhausted` rather than queueing
    /// indefinitely behind a saturated Postgres / Qdrant / LLM tier.
    ///
    /// Override via `AGENT_RS__SERVER__MAX_IN_FLIGHT`.
    #[serde(default = "default_max_in_flight")]
    pub max_in_flight: usize,
}

fn default_max_in_flight() -> usize {
    512
}

/// Default Marker sidecar connection settings. Used both by
/// [`Settings::default`] and as a serde default so older config files
/// missing the `marker.*` section still parse.
fn default_marker_config() -> MarkerConfig {
    MarkerConfig {
        endpoint: "http://localhost:1073".into(),
        connect_timeout_s: 10,
        stream_idle_timeout_s: 600,
        max_pdf_bytes: 100 * 1024 * 1024, // 100 MiB
        max_pages: 500,
    }
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
    pub evidence: EvidenceConfig,
    pub synthesis: SynthesisConfig,
    pub ranker: RankerConfig,
    #[serde(default)]
    pub reranker: RerankerConfig,
    pub verifier: VerifierConfig,
    #[serde(default)]
    pub grounding: GroundingConfig,
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
    /// Marker sidecar gRPC connection settings.
    ///
    /// Loaded even when `ingest_backend != Marker*` so operators can
    /// flip backends via env var alone without editing config files.
    #[serde(default = "default_marker_config")]
    pub marker: MarkerConfig,
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
                model: DEFAULT_LLM_MODEL.to_owned(),
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
                max_connections: default_pg_max_connections(),
                connection_timeout_ms: default_pg_connection_timeout_ms(),
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
                reasoning_effort: DEFAULT_REASONING_EFFORT.to_owned(),
            },
            evidence: EvidenceConfig {
                max_evidence_chars: 60000,
                max_context_chars: 3000,
            },
            synthesis: SynthesisConfig {
                max_snippet_length: 5000,
                fallback_citation_count: default_fallback_citation_count(),
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
                model: None,
            },
            grounding: GroundingConfig::default(),
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
                max_results: default_brave_max_results(),
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
                max_in_flight: default_max_in_flight(),
            },
            ingest_backend: IngestBackend::default(),
            docling: DoclingConfig {
                endpoint: "http://localhost:1068".into(),
                connect_timeout_s: 10,
                stream_idle_timeout_s: 600,
                max_pdf_bytes: 100 * 1024 * 1024, // 100 MiB
                max_pages: 500,
            },
            marker: default_marker_config(),
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
