"""Tests for environment-driven application settings."""

from src.config.settings import Settings


def test_database_url_is_built_from_postgres_fields_when_empty() -> None:
    settings = Settings(
        telegram_bot_token="telegram-token",
        gemini_api_key="gemini-key",
        postgres_user="alice",
        postgres_password="secret",
        postgres_db="wander",
        postgres_host="db.internal",
        postgres_port=5433,
        database_url="",
    )

    assert (
        settings.database_url
        == "postgresql+asyncpg://alice:secret@db.internal:5433/wander"
    )


def test_csv_settings_are_normalized_to_lists() -> None:
    settings = Settings(
        telegram_bot_token="telegram-token",
        gemini_api_key="gemini-key",
        llm_fallback_models="model-a, model-b ,, model-c",
        rag_supported_extensions="pdf, .md ,TXT",
    )

    assert settings.llm_fallback_model_list == ["model-a", "model-b", "model-c"]
    assert settings.rag_supported_extensions_list == [".pdf", ".md", ".txt"]
