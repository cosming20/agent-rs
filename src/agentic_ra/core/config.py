"""Centralized configuration for the Agentic RAG system.

Reads from a root ``.env`` file using pydantic-settings.
All service connection details are typed and validated at startup.
"""

from __future__ import annotations

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
    qdrant_api_key: SecretStr

    # --- OpenSearch ---
    opensearch_user: str = "admin"
    opensearch_initial_admin_password: SecretStr
    opensearch_port: int = 1052
    opensearch_perf_port: int = 1053
    opensearch_java_opts: str = "-Xms512m -Xmx512m"  # Infra-only; override in .env for production

    # --- PostgreSQL ---
    postgres_user: str = "agentic_ra"
    postgres_password: SecretStr
    postgres_db: str = "agentic_ra"
    postgres_port: int = 1054
    postgres_host: str = "localhost"

    # --- OPA ---
    opa_port: int = 1055
    opa_log_level: str = "info"

    # --- Hayhooks (reserved for Phase 8) ---
    hayhooks_port: int = 1056

    # --- OTel / Jaeger ---
    otel_collector_port: int = 1057
    jaeger_ui_port: int = 1058

    # --- Computed connection URLs ---

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qdrant_url(self) -> str:
        """Qdrant HTTP/REST URL for qdrant-client."""
        return f"http://localhost:{self.qdrant_http_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def opensearch_url(self) -> str:
        """OpenSearch HTTPS URL (self-signed cert in dev)."""
        return f"https://localhost:{self.opensearch_port}"

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
        """Async PostgreSQL DSN for asyncpg driver. Use .get_secret_value() to access."""
        return SecretStr(
            f"postgresql+asyncpg://{self.postgres_user}"
            f":{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def opa_url(self) -> str:
        """OPA policy decision endpoint."""
        return f"http://localhost:{self.opa_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def otel_endpoint(self) -> str:
        """OTel Collector OTLP gRPC endpoint."""
        return f"http://localhost:{self.otel_collector_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def jaeger_ui_url(self) -> str:
        """Jaeger trace UI URL."""
        return f"http://localhost:{self.jaeger_ui_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance.

    Call ``get_settings.cache_clear()`` in tests to reset.
    """
    return Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env
