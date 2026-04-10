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

/// gRPC server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// Host address the gRPC server binds to.
    pub host: String,
    /// Port the gRPC server listens on (default: 1059).
    pub port: u16,
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
    pub verifier: VerifierConfig,
    pub web_budget: WebBudgetConfig,
    pub brave: BraveConfig,
    pub fetch: FetchConfig,
    pub otel: OtelConfig,
    pub opa: OpaConfig,
    pub retrieval: RetrievalConfig,
    pub server: ServerConfig,
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
                max_steps: 10,
                max_retries: 1,
                system_prompt_file: String::new(),
            },
            context: ContextConfig {
                max_messages: 4,
                max_chars_per_msg: 150,
            },
            evidence: EvidenceConfig {
                max_evidence_chars: 16000,
                max_context_chars: 300,
            },
            synthesis: SynthesisConfig {
                max_snippet_length: 200,
            },
            ranker: RankerConfig { rrf_k: 60 },
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
                max_length: 5000,
                max_bytes: 2_000_000,
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
            },
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
