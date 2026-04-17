"""Application settings loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all Wander Jr services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Telegram ---
    telegram_bot_token: str = Field(description="Telegram Bot API token")

    # --- Google Gemini ---
    gemini_api_key: str = Field(description="Google AI Studio API key")

    # --- PostgreSQL ---
    postgres_user: str = Field(default="wander")
    postgres_password: str = Field(default="wander")
    postgres_db: str = Field(default="wander_db")
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)
    database_url: str = Field(
        default="postgresql+asyncpg://wander:wander@postgres:5432/wander_db"
    )

    # --- Qdrant ---
    qdrant_host: str = Field(default="qdrant")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="documents")

    # --- Application ---
    app_env: str = Field(default="development")
    app_log_level: str = Field(default="INFO")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    # --- RAG ---
    rag_chunk_size: int = Field(default=420)
    rag_chunk_overlap: int = Field(default=24)
    rag_top_k: int = Field(default=4)
    rag_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    rag_confidence_none_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    rag_confidence_low_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    rag_documents_path: str = Field(default="data/documents")

    # --- Document Parsing ---
    llama_cloud_api_key: str = Field(default="")
    llama_parse_tier: str = Field(default="cost_effective")

    # --- Reranker ---
    reranker_enabled: bool = Field(default=True)
    reranker_model: str = Field(default="ms-marco-MultiBERT-L-12")
    reranker_top_k: int = Field(default=5)
    reranker_retrieval_multiplier: int = Field(default=3)

    # --- LLM ---
    llm_model: str = Field(default="gemini-2.5-flash")
    llm_fallback_models: str = Field(default="gemini-2.5-flash-lite,gemini-3-flash")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=10000)
    embedding_model: str = Field(default="models/gemini-embedding-2-preview")
    embedding_fallback_models: str = Field(default="models/gemini-embedding-001")
    embedding_dimensions: int = Field(default=768)
    embedding_requests_per_minute: int = Field(default=5)
    embedding_max_retries: int = Field(default=8)
    rag_embedding_batch_size: int = Field(default=10)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    @property
    def embedding_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.embedding_fallback_models.split(",") if m.strip()]


def get_settings() -> Settings:
    """Factory to create settings instance (cacheable)."""
    return Settings()
