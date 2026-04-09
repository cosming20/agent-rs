"""Centralized configuration for the Agentic RAG system.

Reads from a root ``.env`` file using pydantic-settings.
All service connection details are typed and validated at startup.
"""


from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to project root: config.py (core/ -> agentic_ra/ -> src/ -> root) = parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Qdrant ---
    qdrant_grpc_port: int = 1050
    qdrant_http_port: int = 1051
    qdrant_host: str = "localhost"
    qdrant_api_key: SecretStr
    qdrant_collection: str = "documents"

    # --- PostgreSQL ---
    postgres_user: str = "agentic_ra"
    postgres_password: SecretStr
    postgres_db: str = "agentic_ra"
    postgres_port: int = 1054
    postgres_host: str = "localhost"

    # --- OPA ---
    opa_port: int = 1055
    opa_host: str = "localhost"
    opa_log_level: str = "info"
    opa_auth_enabled: bool = False

    # --- Hayhooks (reserved) ---
    hayhooks_port: int = 1056

    # --- OTel / Jaeger ---
    otel_collector_port: int = 1057
    otel_host: str = "localhost"
    jaeger_ui_port: int = 1058
    jaeger_host: str = "localhost"

    # --- LLM Provider (LiteLLM format: provider/model) ---
    llm_api_key: SecretStr
    llm_provider_model: str = "openai/gpt-4o"
    llm_verifier_model: str = "openai/gpt-4o-mini"
    llm_enrichment_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.1

    # --- Embedding + Chunking (Docling) ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 32
    chunk_max_tokens: int = 1024
    chunk_merge_peers: bool = True

    # --- Cognee (Knowledge Graph) ---
    cognee_dataset: str = "agentic_rag"
    cognee_graph_db_provider: str = "neo4j"
    cognee_vector_db_provider: str = "qdrant"
    cognee_data_root: str = str(_PROJECT_ROOT / ".cognee_data")
    cognee_system_root: str = str(_PROJECT_ROOT / ".cognee_system")
    cognee_enabled: bool = True

    # --- Neo4j (Cognee graph backend) ---
    neo4j_host: str = "localhost"
    neo4j_bolt_port: int = 1061
    neo4j_http_port: int = 1062
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr
    neo4j_db: str = "neo4j"

    # --- CORS ---
    cors_allowed_origins: str = "http://localhost:1059,http://localhost:8000,http://localhost:3000"

    # --- Application ---
    log_level: str = "INFO"

    # --- Agent ---
    agent_max_steps: int = 10
    agent_max_retries: int = 1
    system_prompt_file: str = ""

    # --- Orchestrator context building ---
    context_max_messages: int = 4
    context_max_chars_per_msg: int = 150

    # --- Tool evidence limits ---
    max_evidence_chars: int = 16000
    max_context_chars: int = 300

    # --- Synthesis ---
    max_snippet_length: int = 200

    # --- Ranker ---
    rrf_k: int = 60

    # --- Verifier ---
    verifier_confidence_threshold: float = 0.5

    # --- Postgres connection pool ---
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10

    # --- Cognee chunking ---
    cognee_chunk_size: int = 16000
    cognee_chunk_overlap: int = 0

    # --- Upload ---
    upload_max_file_size: int = 50_000_000
    upload_allowed_extensions: str = ".pdf,.docx,.pptx,.html,.htm,.txt,.md"

    # --- Admin ---
    admin_max_fan_out: int = 50
    admin_concurrent_searches: int = 5

    # --- Brave Search MCP ---
    brave_api_key: SecretStr = SecretStr("")
    brave_mcp_host: str = "localhost"
    brave_mcp_port: int = 1063
    brave_search_enabled: bool = False

    # --- Fetch ---
    fetch_max_length: int = 5000
    fetch_max_bytes: int = 2_000_000
    fetch_timeout: int = 30
    fetch_user_agent: str = "AgenticRA/1.0"
    fetch_max_redirects: int = 3
    fetch_domain_denylist: str = ""
    fetch_domain_allowlist: str = ""

    # --- Web Tool Budgets ---
    web_budget_max_searches: int = 3
    web_budget_max_news: int = 2
    web_budget_max_fetches: int = 3
    web_budget_timeout_s: float = 45.0
    web_budget_max_bytes: int = 6_000_000

    # --- Feedback (Loop 1) ---
    feedback_enabled: bool = True
    feedback_alpha: float = 0.1
    feedback_dry_run: bool = False
    feedback_weight_floor: float = 0.1
    feedback_weight_ceiling: float = 0.9

    # --- Strategy Learning (Loop 2) ---
    strategy_learning_enabled: bool = False
    strategy_collection: str = "query_strategies"
    strategy_min_neighbors: int = 5
    strategy_confidence_threshold: float = 0.1
    strategy_max_age_days: int = 30

    # --- RuVector (Loop 3) ---
    ruvector_enabled: bool = False
    ruvector_url: str = "http://localhost:1064"
    ruvector_api_key: SecretStr = SecretStr("")
    gnn_training_enabled: bool = False
    gnn_training_interval_hours: int = 24
    gnn_training_min_samples: int = 50

    # --- Computed connection URLs ---

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qdrant_url(self) -> str:
        """Qdrant HTTP/REST URL for qdrant-client."""
        return f"http://{self.qdrant_host}:{self.qdrant_http_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_dsn(self) -> SecretStr:
        """PostgreSQL connection string. Use .get_secret_value() to access."""
        return SecretStr(
            f"postgresql://{self.postgres_user}"
            f":{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_dsn_async(self) -> SecretStr:
        """Async PostgreSQL DSN for asyncpg driver."""
        return SecretStr(
            f"postgresql+asyncpg://{self.postgres_user}"
            f":{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def opa_url(self) -> str:
        """OPA policy decision endpoint."""
        return f"http://{self.opa_host}:{self.opa_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def otel_endpoint(self) -> str:
        """OTel Collector OTLP gRPC endpoint."""
        return f"http://{self.otel_host}:{self.otel_collector_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def jaeger_ui_url(self) -> str:
        """Jaeger trace UI URL."""
        return f"http://{self.jaeger_host}:{self.jaeger_ui_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def neo4j_bolt_url(self) -> str:
        """Neo4j Bolt protocol URL for Cognee graph backend."""
        return f"bolt://{self.neo4j_host}:{self.neo4j_bolt_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def brave_mcp_url(self) -> str:
        """Brave Search MCP server Streamable HTTP URL."""
        return f"http://{self.brave_mcp_host}:{self.brave_mcp_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance.

    Call ``get_settings.cache_clear()`` in tests to reset.
    """
    return Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env
