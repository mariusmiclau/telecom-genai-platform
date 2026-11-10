"""Validated application configuration; secrets never appear in status responses."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    app_mode: Literal["demo", "live"] = "demo"
    auth_mode: Literal["bearer", "oidc"] = "bearer"
    enable_local_actions: bool = False
    max_pending_ingestion_jobs: int = Field(20, ge=1, le=1000)
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_tenant_claim: str = "tenant_id"
    allow_http_identity_endpoint: bool = False
    api_token: SecretStr = SecretStr("")
    admin_token: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(1536, ge=1, le=8192)
    allow_http_model_endpoint: bool = False
    conversation_retention_days: int = Field(30, ge=1, le=365)
    database_url: SecretStr = SecretStr("")
    chroma_host: str = "localhost"
    chroma_port: int = Field(8100, ge=1, le=65535)
    chroma_collection: str = "telecom_kb"
    knowledge_version: str = "tickets-v1"
    retrieval_configuration: Literal["A", "B", "C"] = "C"
    cors_origins: list[str] = []
    request_timeout_seconds: float = Field(45, gt=0, le=120)
    max_concurrent_requests: int = Field(4, ge=1, le=64)
    max_upload_bytes: int = Field(5_000_000, ge=1024, le=50_000_000)
    ingestion_root: Path = Path("data/incoming")
    requests_per_minute: int = Field(30, ge=1)

    @classmethod
    def load(cls):
        return cls(_secrets_dir=os.getenv("APP_SECRETS_DIR"))

    @model_validator(mode="after")
    def validate_live(self):
        if self.auth_mode == "oidc":
            if not self.oidc_audience or not self.oidc_issuer or not self.oidc_jwks_url:
                raise ValueError("OIDC issuer, audience and JWKS URL are required")
            if not self.allow_http_identity_endpoint and not all(value.startswith("https://") for value in (self.oidc_issuer, self.oidc_jwks_url)):
                raise ValueError("Identity endpoints must use HTTPS")
        if self.admin_token.get_secret_value() and len(self.admin_token.get_secret_value()) < 32:
            raise ValueError("ADMIN_TOKEN must contain at least 32 characters")
        if "*" in self.cors_origins:
            raise ValueError("Use explicit CORS origins")
        if self.app_mode == "live":
            if not self.openai_base_url.startswith("https://") and not self.allow_http_model_endpoint:
                raise ValueError("Model endpoint must use HTTPS")
            if self.auth_mode == "bearer" and len(self.api_token.get_secret_value()) < 32:
                raise ValueError("Live mode requires API_TOKEN with at least 32 characters")
            if not self.openai_api_key.get_secret_value():
                raise ValueError("Live mode requires OPENAI_API_KEY")
            if not self.database_url.get_secret_value().startswith("postgresql://"):
                raise ValueError("Live mode requires a PostgreSQL DATABASE_URL")
            if self.auth_mode == "bearer" and self.admin_token.get_secret_value() == self.api_token.get_secret_value():
                raise ValueError("Admin and reader tokens must differ")
        return self
