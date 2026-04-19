"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_POSTGRES_USER = "wander"
DEFAULT_POSTGRES_PASSWORD = "wander"
DEFAULT_POSTGRES_DB = "wander_db"
DEFAULT_POSTGRES_HOST = "postgres"
DEFAULT_POSTGRES_PORT = 5432

DEFAULT_QDRANT_HOST = "qdrant"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_QDRANT_GRPC_PORT = 6334
DEFAULT_QDRANT_COLLECTION_NAME = "documents"
DEFAULT_QDRANT_SPARSE_VECTOR_NAME = "sparse-vector"

DEFAULT_APP_ENV = "development"
DEFAULT_APP_LOG_LEVEL = "INFO"
DEFAULT_APP_HOST = "0.0.0.0"
DEFAULT_APP_PORT = 8000
DEFAULT_APP_NAME = "Wander Jr"
DEFAULT_APP_DESCRIPTION = (
    "Assistente inteligente institucional do CEFET-MG campus Timóteo"
)
DEFAULT_APP_SERVICE_NAME = "wander-jr"
DEFAULT_APP_ASSISTANT_NAME = "Wander Jr"
DEFAULT_APP_INSTITUTION_NAME = "CEFET-MG campus Timóteo"
DEFAULT_APP_SIGAA_URL = "https://sig.cefetmg.br/sigaa/"
DEFAULT_APP_SECRETARIA_PHONE = "(31) 3845-2005"
DEFAULT_APP_SECRETARIA_EMAIL = "de.te@cefetmg.br"
DEFAULT_APP_DIRETORIA_EMAIL = "diretoria-te@cefetmg.br"
DEFAULT_APP_SESSION_TIMEOUT_MINUTES = 30
DEFAULT_APP_MAX_HISTORY_TURNS = 10

DEFAULT_RAG_CHUNK_SIZE = 560
DEFAULT_RAG_CHUNK_OVERLAP = 64
DEFAULT_RAG_TOP_K = 8
DEFAULT_RAG_SCORE_THRESHOLD = 0.30
DEFAULT_RAG_CONFIDENCE_NONE_THRESHOLD = 0.28
DEFAULT_RAG_CONFIDENCE_LOW_THRESHOLD = 0.48
DEFAULT_RAG_DOCUMENTS_PATH = "data/documents"
DEFAULT_RAG_SUPPORTED_EXTENSIONS = ".pdf,.txt,.md"
DEFAULT_RAG_LIST_QUERY_MIN_TOP_K = 8
DEFAULT_RAG_PROMPT_HISTORY_TURNS = 6

DEFAULT_LLAMA_PARSE_TIER = "cost_effective"

DEFAULT_RERANKER_ENABLED = False
DEFAULT_RERANKER_MODEL = "ms-marco-MultiBERT-L-12"
DEFAULT_RERANKER_TOP_K = 5
DEFAULT_RERANKER_RETRIEVAL_MULTIPLIER = 3

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_FALLBACK_MODELS = "gemini-2.5-flash-lite,gemini-3-flash"
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_LLM_MAX_TOKENS = 10000
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE = 5
DEFAULT_EMBEDDING_MAX_RETRIES = 3
DEFAULT_EMBEDDING_BASE_RETRY_SECONDS = 2.0
DEFAULT_RAG_EMBEDDING_BATCH_SIZE = 10


def _build_database_url(
    *,
    user: str,
    password: str,
    host: str,
    port: int,
    database: str,
) -> str:
    """Build a PostgreSQL connection string from individual settings."""
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated env var into a cleaned list."""
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Central configuration for all Wander Jr services."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- Telegram ---
    telegram_bot_token: str = Field(description="Telegram Bot API token")

    # --- Google Gemini ---
    gemini_api_key: str = Field(description="Google AI Studio API key")

    # --- PostgreSQL ---
    postgres_user: str = Field(default=DEFAULT_POSTGRES_USER)
    postgres_password: str = Field(default=DEFAULT_POSTGRES_PASSWORD)
    postgres_db: str = Field(default=DEFAULT_POSTGRES_DB)
    postgres_host: str = Field(default=DEFAULT_POSTGRES_HOST)
    postgres_port: int = Field(default=DEFAULT_POSTGRES_PORT)
    database_url: str = Field(default="")

    # --- Qdrant ---
    qdrant_host: str = Field(default=DEFAULT_QDRANT_HOST)
    qdrant_port: int = Field(default=DEFAULT_QDRANT_PORT)
    qdrant_grpc_port: int = Field(default=DEFAULT_QDRANT_GRPC_PORT)
    qdrant_collection_name: str = Field(default=DEFAULT_QDRANT_COLLECTION_NAME)
    qdrant_sparse_vector_name: str = Field(default=DEFAULT_QDRANT_SPARSE_VECTOR_NAME)

    # --- Application ---
    app_env: str = Field(default=DEFAULT_APP_ENV)
    app_log_level: str = Field(default=DEFAULT_APP_LOG_LEVEL)
    app_host: str = Field(default=DEFAULT_APP_HOST)
    app_port: int = Field(default=DEFAULT_APP_PORT)
    app_name: str = Field(default=DEFAULT_APP_NAME)
    app_description: str = Field(default=DEFAULT_APP_DESCRIPTION)
    app_service_name: str = Field(default=DEFAULT_APP_SERVICE_NAME)
    app_assistant_name: str = Field(default=DEFAULT_APP_ASSISTANT_NAME)
    app_institution_name: str = Field(default=DEFAULT_APP_INSTITUTION_NAME)
    app_sigaa_url: str = Field(default=DEFAULT_APP_SIGAA_URL)
    app_secretaria_phone: str = Field(default=DEFAULT_APP_SECRETARIA_PHONE)
    app_secretaria_email: str = Field(default=DEFAULT_APP_SECRETARIA_EMAIL)
    app_diretoria_email: str = Field(default=DEFAULT_APP_DIRETORIA_EMAIL)
    app_session_timeout_minutes: int = Field(
        default=DEFAULT_APP_SESSION_TIMEOUT_MINUTES,
        ge=1,
    )
    app_max_history_turns: int = Field(default=DEFAULT_APP_MAX_HISTORY_TURNS, ge=1)

    # --- RAG ---
    rag_chunk_size: int = Field(default=DEFAULT_RAG_CHUNK_SIZE, ge=1)
    rag_chunk_overlap: int = Field(default=DEFAULT_RAG_CHUNK_OVERLAP, ge=0)
    rag_top_k: int = Field(default=DEFAULT_RAG_TOP_K, ge=1)
    rag_score_threshold: float = Field(default=DEFAULT_RAG_SCORE_THRESHOLD, ge=0.0, le=1.0)
    rag_confidence_none_threshold: float = Field(
        default=DEFAULT_RAG_CONFIDENCE_NONE_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    rag_confidence_low_threshold: float = Field(
        default=DEFAULT_RAG_CONFIDENCE_LOW_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    rag_documents_path: str = Field(default=DEFAULT_RAG_DOCUMENTS_PATH)
    rag_supported_extensions: str = Field(default=DEFAULT_RAG_SUPPORTED_EXTENSIONS)
    rag_list_query_min_top_k: int = Field(default=DEFAULT_RAG_LIST_QUERY_MIN_TOP_K, ge=1)
    rag_prompt_history_turns: int = Field(default=DEFAULT_RAG_PROMPT_HISTORY_TURNS, ge=1)

    # --- Document Parsing ---
    llama_cloud_api_key: str = Field(default="")
    llama_parse_tier: str = Field(default=DEFAULT_LLAMA_PARSE_TIER)

    # --- Reranker ---
    reranker_enabled: bool = Field(default=DEFAULT_RERANKER_ENABLED)
    reranker_model: str = Field(default=DEFAULT_RERANKER_MODEL)
    reranker_top_k: int = Field(default=DEFAULT_RERANKER_TOP_K, ge=1)
    reranker_retrieval_multiplier: int = Field(
        default=DEFAULT_RERANKER_RETRIEVAL_MULTIPLIER,
        ge=1,
    )

    # --- LLM ---
    llm_model: str = Field(default=DEFAULT_LLM_MODEL)
    llm_fallback_models: str = Field(default=DEFAULT_LLM_FALLBACK_MODELS)
    llm_temperature: float = Field(default=DEFAULT_LLM_TEMPERATURE)
    llm_max_tokens: int = Field(default=DEFAULT_LLM_MAX_TOKENS, ge=1)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    embedding_dimensions: int = Field(default=DEFAULT_EMBEDDING_DIMENSIONS, ge=1)
    embedding_requests_per_minute: int = Field(
        default=DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE,
        ge=1,
    )
    embedding_max_retries: int = Field(default=DEFAULT_EMBEDDING_MAX_RETRIES, ge=1)
    embedding_base_retry_seconds: float = Field(
        default=DEFAULT_EMBEDDING_BASE_RETRY_SECONDS,
        gt=0.0,
    )
    rag_embedding_batch_size: int = Field(default=DEFAULT_RAG_EMBEDDING_BATCH_SIZE, ge=1)

    @model_validator(mode="after")
    def apply_derived_defaults(self) -> Settings:
        """Populate settings that can be derived from other env vars."""
        if not self.database_url:
            self.database_url = _build_database_url(
                user=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_fallback_model_list(self) -> list[str]:
        return _parse_csv(self.llm_fallback_models)

    @property
    def rag_supported_extensions_list(self) -> list[str]:
        extensions = []
        for extension in _parse_csv(self.rag_supported_extensions):
            normalized = extension.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            extensions.append(normalized)
        return extensions


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Factory to create settings instance (cacheable)."""
    return Settings()
