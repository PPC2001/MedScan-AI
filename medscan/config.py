"""
Application configuration — loaded from .env via pydantic-settings.
All environment variables are documented in .env.example.

LLM provider priority:
  1. Grok (xAI)  — set XAI_API_KEY
  2. OpenAI      — set OPENAI_API_KEY
  3. Anthropic   — set ANTHROPIC_API_KEY
  4. HF-only     — no LLM calls (extraction disabled)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = (
        "postgresql+asyncpg://medscan:medscan_secret@localhost:5432/medscan"
    )
    sync_database_url: str = (
        "postgresql://medscan:medscan_secret@localhost:5432/medscan"
    )

    # -------------------------------------------------------------------------
    # Redis / Celery
    # -------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # -------------------------------------------------------------------------
    # OpenAI
    # -------------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # -------------------------------------------------------------------------
    # Grok / xAI  (primary LLM — OpenAI-compatible endpoint)
    # -------------------------------------------------------------------------
    xai_api_key: str = ""                            # XAI_API_KEY
    grok_model: str = "grok-3"                       # GROK_MODEL
    grok_base_url: str = "https://api.x.ai/v1"       # GROK_BASE_URL

    # -------------------------------------------------------------------------
    # LLM Provider Selection
    # -------------------------------------------------------------------------
    # Explicit override. If not set, auto-detected from available keys.
    # Values: grok | openai | anthropic | none
    llm_provider: str = "auto"                       # LLM_PROVIDER

    # -------------------------------------------------------------------------
    # Anthropic (optional)
    # -------------------------------------------------------------------------
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # -------------------------------------------------------------------------
    # LangSmith
    # -------------------------------------------------------------------------
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "medscan-ai"

    # -------------------------------------------------------------------------
    # Pinecone
    # -------------------------------------------------------------------------
    pinecone_api_key: str = ""
    pinecone_index_name: str = "medscan-medical"
    pinecone_environment: str = "us-east-1"

    # -------------------------------------------------------------------------
    # Vector store selection
    # -------------------------------------------------------------------------
    vector_store_backend: str = "pgvector"  # pgvector | pinecone

    # -------------------------------------------------------------------------
    # HuggingFace
    # -------------------------------------------------------------------------
    hf_token: str = ""
    hf_device: str = "cpu"  # cpu | cuda | mps

    # -------------------------------------------------------------------------
    # File storage
    # -------------------------------------------------------------------------
    upload_dir: Path = Path("./uploads")
    max_file_size_mb: int = 50

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_env: str = "development"
    log_level: str = "INFO"
    api_key: str = "medscan-dev-key-change-me"

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def active_llm_provider(self) -> str:
        """
        Resolve which LLM provider is actually available.

        Priority: explicit llm_provider setting → auto-detect from keys.
        Returns one of: 'grok' | 'openai' | 'anthropic' | 'none'
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        # Auto-detect
        if self.xai_api_key:
            return "grok"
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return "none"

    @property
    def has_llm(self) -> bool:
        """True if any LLM provider key is configured."""
        return self.active_llm_provider != "none"

    @field_validator("hf_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"hf_device must be one of {allowed}")
        return v

    @field_validator("vector_store_backend")
    @classmethod
    def validate_vector_store(cls, v: str) -> str:
        allowed = {"pgvector", "pinecone"}
        if v not in allowed:
            raise ValueError(f"vector_store_backend must be one of {allowed}")
        return v

    def ensure_upload_dir(self) -> None:
        """Create upload directory if it doesn't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
