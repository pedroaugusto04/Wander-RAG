"""Telegram polling mode — for local development and testing.

Usage:
    python scripts/run_polling.py

This starts the bot in polling mode (no webhook needed, no public URL needed).
Perfect for local testing.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from src.ai.llm.gemini_provider import GeminiProvider
from src.ai.rag.pipeline import RAGPipeline
from src.config.settings import get_settings
from src.core.conversation import ConversationManager
from src.core.models import ChannelType, IncomingMessage
from src.core.orchestrator import AIOrchestrator
from src.infra.conversation_store import PostgresConversationStore
from src.infra.database import create_db_engine
from src.infra.logging import setup_logging
from src.knowledge.ingestion.pipeline import IngestionPipeline
from src.knowledge.vectorstore.qdrant_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _acquire_single_instance_lock() -> object:
    """Acquire a process lock so only one polling instance can run."""
    lock_path = Path(".run_polling.lock")
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            "Outra instância do run_polling.py já está em execução. "
            "Finalize a instância antiga antes de iniciar uma nova."
        ) from exc

    lock_file.write(str(Path.cwd()))
    lock_file.flush()
    return lock_file


async def setup_components(settings):  # noqa: ANN001, ANN201
    """Initialize all application components."""
    # LLM Provider
    llm_provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.llm_model,
        fallback_models=settings.llm_fallback_model_list,
        embedding_model=settings.embedding_model,
        embedding_fallback_models=settings.embedding_fallback_model_list,
        embedding_requests_per_minute=settings.embedding_requests_per_minute,
        embedding_max_retries=settings.embedding_max_retries,
        embedding_base_retry_seconds=settings.embedding_base_retry_seconds,
    )

    if settings.embedding_dimensions:
        vector_size = settings.embedding_dimensions
        logger.info(
            "Using configured embedding dimension from EMBEDDING_DIMENSIONS=%d",
            vector_size,
        )
    else:
        vector_size = await llm_provider.get_embedding_dimension()

    # Vector Store
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=vector_size,
        sparse_vector_name=settings.qdrant_sparse_vector_name,
    )
    await vector_store.initialize()

    reranker = None
    if settings.reranker_enabled:
        from src.ai.rag.reranker import FlashRankReranker

        reranker = FlashRankReranker(
            model_name=settings.reranker_model,
            top_k=settings.reranker_top_k,
        )

    # RAG Pipeline
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_provider=llm_provider,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
        confidence_none_threshold=settings.rag_confidence_none_threshold,
        confidence_low_threshold=settings.rag_confidence_low_threshold,
        reranker=reranker,
        retrieval_multiplier=settings.reranker_retrieval_multiplier,
        list_query_min_top_k=settings.rag_list_query_min_top_k,
        assistant_name=settings.app_assistant_name,
        institution_name=settings.app_institution_name,
        prompt_history_turns=settings.rag_prompt_history_turns,
    )

    # AI Orchestrator
    orchestrator = AIOrchestrator(
        llm_provider=llm_provider,
        rag_pipeline=rag_pipeline,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    db_session_factory = create_db_engine(settings.database_url)
    conversation_store = PostgresConversationStore(db_session_factory)
    await conversation_store.initialize()

    # Conversation Manager
    conversation_manager = ConversationManager(
        orchestrator=orchestrator,
        conversation_store=conversation_store,
        session_timeout_minutes=settings.app_session_timeout_minutes,
        max_history_turns=settings.app_max_history_turns,
        assistant_name=settings.app_assistant_name,
        institution_name=settings.app_institution_name,
        sigaa_url=settings.app_sigaa_url,
        secretaria_phone=settings.app_secretaria_phone,
        secretaria_email=settings.app_secretaria_email,
        diretoria_email=settings.app_diretoria_email,
    )

    return conversation_manager, vector_store, llm_provider


async def run_manual_ingest(
    *,
    settings,
    vector_store: QdrantVectorStore,
    llm_provider: GeminiProvider,
    path: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    embedding_batch_size: int | None,
    extensions: list[str] | None,
) -> int:  # noqa: ANN001
    """Run manual ingestion when explicitly requested by CLI args."""
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        llm_provider=llm_provider,
        chunk_size=chunk_size or settings.rag_chunk_size,
        chunk_overlap=chunk_overlap or settings.rag_chunk_overlap,
        embedding_batch_size=embedding_batch_size or settings.rag_embedding_batch_size,
        llama_api_key=settings.llama_cloud_api_key,
        llama_parse_tier=settings.llama_parse_tier,
    )

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found for manual ingest: {target}")

    if target.is_file():
        return await pipeline.ingest_file(target)
    return await pipeline.ingest_directory(
        target,
        extensions=extensions or settings.rag_supported_extensions_list,
    )


def parse_args() -> argparse.Namespace:
    """Parse optional manual ingestion args for local polling mode."""
    parser = argparse.ArgumentParser(description="Run Wander Jr in Telegram polling mode")
    parser.add_argument(
        "--ingest-path",
        type=str,
        default=None,
        help="Optional path (file/dir) to ingest before starting polling",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size for manual ingest",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap for manual ingest",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Override embedding batch size for manual ingest",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Extensions for directory ingest (example: --extensions .pdf .md)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Telegram bot in polling mode."""
    args = parse_args()
    lock_file = _acquire_single_instance_lock()
    settings = get_settings()
    setup_logging(log_level=settings.app_log_level)

    logger.info("=" * 50)
    logger.info("🤖 %s — Modo Polling (%s)", settings.app_name, settings.app_env)
    logger.info("=" * 50)

    # Build the telegram application
    app = Application.builder().token(settings.telegram_bot_token).build()

    # We need to initialize our components
    conversation_manager = None
    vector_store = None

    async def post_init(application: Application) -> None:  # type: ignore[type-arg]
        nonlocal conversation_manager, vector_store

        # Polling mode cannot run with an active webhook.
        await application.bot.delete_webhook(drop_pending_updates=True)

        conversation_manager, vector_store, llm_provider = await setup_components(settings)

        if args.ingest_path:
            logger.info("📥 Manual ingest requested: %s", args.ingest_path)
            ingested = await run_manual_ingest(
                settings=settings,
                vector_store=vector_store,
                llm_provider=llm_provider,
                path=args.ingest_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_batch_size=args.embedding_batch_size,
                extensions=args.extensions,
            )
            logger.info("✅ Manual ingest complete: %d chunks added", ingested)

        info = await vector_store.get_collection_info()
        logger.info("📊 Qdrant collection '%s': %s points", info["name"], info["points_count"])
        logger.info("✅ Wander Jr está pronto! Envie uma mensagem no Telegram.")

    app.post_init = post_init

    def _build_incoming_message(update: Update) -> IncomingMessage | None:
        """Normalize a Telegram update into the app incoming message model."""
        if not update.message or not update.message.text:
            return None

        user = update.effective_user
        chat = update.effective_chat

        return IncomingMessage(
            channel=ChannelType.TELEGRAM,
            channel_user_id=str(user.id) if user else "unknown",
            channel_chat_id=str(chat.id) if chat else "unknown",
            text=update.message.text,
            metadata={
                "telegram_message_id": update.message.message_id,
                "user_first_name": user.first_name if user else None,
            },
        )

    async def _handle_incoming(
        update: Update,
        *,
        parse_mode: str | None = None,
    ) -> None:
        """Process a Telegram message through the shared conversation manager."""
        incoming = _build_incoming_message(update)
        if incoming is None or not update.message:
            return

        if conversation_manager is None:
            await update.message.reply_text("⏳ Bot ainda inicializando, tente novamente...")
            return

        user = update.effective_user
        logger.info("📩 Mensagem de %s: %s", user.first_name if user else "?", incoming.text[:80])

        # Send typing indicator
        await update.message.chat.send_action("typing")

        # Process through AI pipeline
        response = await conversation_manager.handle_message(incoming)

        # Send response
        await update.message.reply_text(response, parse_mode=parse_mode)

        logger.info("📤 Resposta enviada (%d chars)", len(response))

    async def handle_message(update: Update, context) -> None:  # noqa: ANN001, ARG001
        """Handle incoming text messages."""
        await _handle_incoming(update)

    async def handle_command(update: Update, context) -> None:  # noqa: ANN001, ARG001
        """Handle supported bot commands through the shared conversation flow."""
        await _handle_incoming(update, parse_mode="Markdown")

    # Register handlers
    from telegram.ext import CommandHandler

    app.add_handler(CommandHandler(["start", "ajuda", "sigaa", "contato"], handle_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run polling
    logger.info("🔄 Iniciando polling...")
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    finally:
        lock_file.close()


if __name__ == "__main__":
    main()
