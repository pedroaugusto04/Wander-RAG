"""Conversation Manager — manages sessions and message history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.settings import (
    DEFAULT_APP_ASSISTANT_NAME,
    DEFAULT_APP_DIRETORIA_EMAIL,
    DEFAULT_APP_INSTITUTION_NAME,
    DEFAULT_APP_MAX_HISTORY_TURNS,
    DEFAULT_APP_SECRETARIA_EMAIL,
    DEFAULT_APP_SECRETARIA_PHONE,
    DEFAULT_APP_SESSION_TIMEOUT_MINUTES,
    DEFAULT_APP_SIGAA_URL,
)
from src.core.models import (
    ConversationContext,
    IncomingMessage,
    MessageRole,
)

if TYPE_CHECKING:
    from src.core.orchestrator import AIOrchestrator
    from src.infra.conversation_store import PostgresConversationStore

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation sessions and delegates to the AI orchestrator.

    Phase 1: In-memory sessions (dict).
    Phase 2: Persistent sessions (PostgreSQL).
    """

    def __init__(
        self,
        orchestrator: AIOrchestrator,
        session_timeout_minutes: int = DEFAULT_APP_SESSION_TIMEOUT_MINUTES,
        max_history_turns: int = DEFAULT_APP_MAX_HISTORY_TURNS,
        conversation_store: PostgresConversationStore | None = None,
        assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
        institution_name: str = DEFAULT_APP_INSTITUTION_NAME,
        sigaa_url: str = DEFAULT_APP_SIGAA_URL,
        secretaria_phone: str = DEFAULT_APP_SECRETARIA_PHONE,
        secretaria_email: str = DEFAULT_APP_SECRETARIA_EMAIL,
        diretoria_email: str = DEFAULT_APP_DIRETORIA_EMAIL,
    ) -> None:
        self.orchestrator = orchestrator
        self.session_timeout = session_timeout_minutes
        self.max_history = max_history_turns
        self.conversation_store = conversation_store
        self._sessions: dict[str, ConversationContext] = {}
        self.assistant_name = assistant_name
        self.institution_name = institution_name
        self.sigaa_url = sigaa_url
        self.secretaria_phone = secretaria_phone
        self.secretaria_email = secretaria_email
        self.diretoria_email = diretoria_email

    def _session_key(self, message: IncomingMessage) -> str:
        """Build a unique session key from channel + user."""
        return f"{message.channel.value}:{message.channel_user_id}"

    async def _get_or_create_session(self, message: IncomingMessage) -> ConversationContext:
        """Get existing session or create a new one, expiring stale sessions."""
        key = self._session_key(message)
        session = self._sessions.get(key)

        if session:
            elapsed = (datetime.now(UTC) - session.last_activity).total_seconds()
            if elapsed > self.session_timeout * 60:
                logger.info("Session expired for %s, creating new one", key)
                session = None

        if not session:
            session = ConversationContext(
                session_id=key,
                channel=message.channel,
                channel_user_id=message.channel_user_id,
            )

            if self.conversation_store is not None:
                try:
                    history = await self.conversation_store.get_recent_history(
                        key,
                        max_turns=self.max_history * 2,
                    )
                    session.history = history
                except Exception:
                    logger.exception("Failed to restore conversation history for %s", key)

            self._sessions[key] = session
            logger.info("Created new session: %s", key)

        return session

    async def _persist_turn(
        self,
        session: ConversationContext,
        *,
        role: MessageRole,
        content: str,
        channel_chat_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Best-effort persistence of session and turn into PostgreSQL."""
        if self.conversation_store is None:
            return

        try:
            await self.conversation_store.upsert_session(
                session,
                channel_chat_id=channel_chat_id,
                metadata={"session_timeout_minutes": self.session_timeout},
            )
            await self.conversation_store.append_turn(
                context=session,
                role=role,
                content=content,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to persist turn for session %s", session.session_id)

    async def handle_message(self, message: IncomingMessage) -> str:
        """Handle an incoming message: manage session → delegate to AI → update history."""
        session = await self._get_or_create_session(message)

        # 1. Interceptar comandos estáticos (Telegram)
        text_lower = message.text.strip().lower()
        if text_lower == "/start":
            response = (
                f"Olá! Sou o {self.assistant_name}, assistente virtual do {self.institution_name}.\n\n"
                "Estou aqui para te ajudar com informações sobre a instituição, calendário acadêmico, "
                "bolsas, regras e manuais.\n\n"
                "Como posso ajudar você hoje?"
            )
            session.add_turn(MessageRole.USER, message.text)
            session.add_turn(MessageRole.ASSISTANT, response)
            await self._persist_turn(
                session,
                role=MessageRole.USER,
                content=message.text,
                channel_chat_id=message.channel_chat_id,
                metadata=message.metadata,
            )
            await self._persist_turn(
                session,
                role=MessageRole.ASSISTANT,
                content=response,
                channel_chat_id=message.channel_chat_id,
            )
            return response
        elif text_lower == "/ajuda":
            response = (
                f"Posso buscar informações nos documentos oficiais do {self.institution_name}.\n\n"
                "Tente me fazer perguntas como:\n"
                "- Quais os horários de ônibus?\n"
                "- Como funciona o edital de monitoria?\n"
                "- Quando terminam as aulas?\n\n"
                "*(Lembrando que não tenho acesso a dados pessoais ou notas do SIGAA)*"
            )
            session.add_turn(MessageRole.USER, message.text)
            session.add_turn(MessageRole.ASSISTANT, response)
            await self._persist_turn(
                session,
                role=MessageRole.USER,
                content=message.text,
                channel_chat_id=message.channel_chat_id,
                metadata=message.metadata,
            )
            await self._persist_turn(
                session,
                role=MessageRole.ASSISTANT,
                content=response,
                channel_chat_id=message.channel_chat_id,
            )
            return response
        elif text_lower == "/sigaa":
            response = (
                "Acesse o sistema acadêmico oficial através do link: "
                f"{self.sigaa_url}"
            )
            session.add_turn(MessageRole.USER, message.text)
            session.add_turn(MessageRole.ASSISTANT, response)
            await self._persist_turn(
                session,
                role=MessageRole.USER,
                content=message.text,
                channel_chat_id=message.channel_chat_id,
                metadata=message.metadata,
            )
            await self._persist_turn(
                session,
                role=MessageRole.ASSISTANT,
                content=response,
                channel_chat_id=message.channel_chat_id,
            )
            return response
        elif text_lower == "/contato":
            response = (
                f"📞 **Contatos - {self.institution_name}**\n\n"
                "Secretaria de Registro (SRC): "
                f"{self.secretaria_phone} / {self.secretaria_email}\n"
                f"Diretoria: {self.diretoria_email}\n"
                "Para outros setores, acesse o site oficial."
            )
            session.add_turn(MessageRole.USER, message.text)
            session.add_turn(MessageRole.ASSISTANT, response)
            await self._persist_turn(
                session,
                role=MessageRole.USER,
                content=message.text,
                channel_chat_id=message.channel_chat_id,
                metadata=message.metadata,
            )
            await self._persist_turn(
                session,
                role=MessageRole.ASSISTANT,
                content=response,
                channel_chat_id=message.channel_chat_id,
            )
            return response

        # 2. Fluxo Normal com IA
        session.add_turn(MessageRole.USER, message.text)
        await self._persist_turn(
            session,
            role=MessageRole.USER,
            content=message.text,
            channel_chat_id=message.channel_chat_id,
            metadata=message.metadata,
        )

        response, assistant_metadata = await self.orchestrator.process_with_metadata(message, session)

        session.add_turn(MessageRole.ASSISTANT, response)
        await self._persist_turn(
            session,
            role=MessageRole.ASSISTANT,
            content=response,
            channel_chat_id=message.channel_chat_id,
            metadata=assistant_metadata,
        )

        if len(session.history) > self.max_history * 2:
            session.history = session.history[-(self.max_history * 2) :]

        return response

    def get_active_sessions_count(self) -> int:
        """Return the number of active (non-expired) sessions."""
        now = datetime.now(UTC)
        return sum(
            1
            for s in self._sessions.values()
            if (now - s.last_activity).total_seconds() < self.session_timeout * 60
        )

    def clear_expired_sessions(self) -> int:
        """Remove expired sessions and return the count removed."""
        now = datetime.now(UTC)
        expired = [
            key
            for key, session in self._sessions.items()
            if (now - session.last_activity).total_seconds() >= self.session_timeout * 60
        ]
        for key in expired:
            del self._sessions[key]

        if expired:
            logger.info("Cleared %d expired sessions", len(expired))

        return len(expired)
