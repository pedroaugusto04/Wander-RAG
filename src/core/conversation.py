"""Conversation Manager — manages sessions and message history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.core.models import (
    ConversationContext,
    IncomingMessage,
    MessageRole,
)

if TYPE_CHECKING:
    from src.core.orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation sessions and delegates to the AI orchestrator.

    Phase 1: In-memory sessions (dict).
    Phase 2: Persistent sessions (PostgreSQL).
    """

    def __init__(
        self,
        orchestrator: AIOrchestrator,
        session_timeout_minutes: int = 30,
        max_history_turns: int = 10,
    ) -> None:
        self.orchestrator = orchestrator
        self.session_timeout = session_timeout_minutes
        self.max_history = max_history_turns
        self._sessions: dict[str, ConversationContext] = {}

    def _session_key(self, message: IncomingMessage) -> str:
        """Build a unique session key from channel + user."""
        return f"{message.channel.value}:{message.channel_user_id}"

    def _get_or_create_session(self, message: IncomingMessage) -> ConversationContext:
        """Get existing session or create a new one, expiring stale sessions."""
        key = self._session_key(message)
        session = self._sessions.get(key)

        if session:
            # Check if session has timed out
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
            self._sessions[key] = session
            logger.info("Created new session: %s", key)

        return session

    async def handle_message(self, message: IncomingMessage) -> str:
        """Handle an incoming message: manage session → delegate to AI → update history."""
        session = self._get_or_create_session(message)

        # Add user message to history
        session.add_turn(MessageRole.USER, message.text)

        # Process through AI
        response = await self.orchestrator.process(message, session)

        # Add assistant response to history
        session.add_turn(MessageRole.ASSISTANT, response)

        # Trim history if too long
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
