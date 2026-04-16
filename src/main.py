"""Wander Jr — FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from src.ai.llm.gemini_provider import GeminiProvider
from src.ai.rag.pipeline import RAGPipeline
from src.channels.telegram.adapter import TelegramChannelAdapter
from src.channels.telegram.webhook import init_telegram_webhook
from src.channels.telegram.webhook import router as telegram_router
from src.config.settings import get_settings
from src.core.conversation import ConversationManager
from src.core.orchestrator import AIOrchestrator
from src.infra.conversation_store import PostgresConversationStore
from src.infra.database import create_db_engine
from src.knowledge.vectorstore.qdrant_store import QdrantVectorStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from src.infra.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()

    setup_logging(log_level=settings.app_log_level)

    logger.info("Starting Wander Jr (env=%s)", settings.app_env)


    llm_provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.llm_model,
        fallback_models=settings.llm_fallback_model_list,
        embedding_model=settings.embedding_model,
        embedding_fallback_models=settings.embedding_fallback_model_list,
        embedding_requests_per_minute=settings.embedding_requests_per_minute,
        embedding_max_retries=settings.embedding_max_retries,
    )

    detected_embedding_dim = await llm_provider.get_embedding_dimension()
    vector_size = settings.embedding_dimensions or detected_embedding_dim

    if settings.embedding_dimensions and settings.embedding_dimensions != detected_embedding_dim:
        logger.warning(
            "EMBEDDING_DIMENSIONS=%d overrides detected model dim=%d",
            settings.embedding_dimensions,
            detected_embedding_dim,
        )


    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=vector_size,
    )
    await vector_store.initialize()


    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_provider=llm_provider,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )


    orchestrator = AIOrchestrator(
        llm_provider=llm_provider,
        rag_pipeline=rag_pipeline,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    db_session_factory = create_db_engine(settings.database_url)
    conversation_store = PostgresConversationStore(db_session_factory)
    await conversation_store.initialize()


    conversation_manager = ConversationManager(
        orchestrator=orchestrator,
        conversation_store=conversation_store,
    )


    telegram_adapter = TelegramChannelAdapter(token=settings.telegram_bot_token)
    await telegram_adapter.setup()


    init_telegram_webhook(
        adapter=telegram_adapter,
        message_handler=conversation_manager.handle_message,
    )


    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.llm_provider = llm_provider
    app.state.conversation_manager = conversation_manager

    logger.info("Wander Jr is ready!")

    yield


    logger.info("Shutting down Wander Jr...")
    await telegram_adapter.shutdown()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Wander Jr",
        description="Assistente inteligente institucional do CEFET-MG campus Timóteo",
        version="0.1.0",
        lifespan=lifespan,
    )


    app.include_router(telegram_router)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "service": "wander-jr"}

    return app


app = create_app()
