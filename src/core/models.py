"""Domain models — channel-agnostic message types and conversation context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ChannelType(Enum):
    """Supported messaging channels."""

    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    WEB = "web"


class MessageRole(Enum):
    """Role within a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class IncomingMessage:
    """Normalised message received from any channel."""

    channel: ChannelType
    channel_user_id: str
    channel_chat_id: str
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Any] = field(default_factory=list)


@dataclass
class OutgoingMessage:
    """Normalised message to be sent to any channel."""

    text: str
    channel_chat_id: str
    reply_to_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Any] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Session state maintained across turns."""

    session_id: str
    channel: ChannelType
    channel_user_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_turn(self, role: MessageRole, content: str) -> None:
        """Append a turn and refresh activity timestamp."""
        self.history.append({"role": role.value, "content": content})
        self.last_activity = datetime.now(UTC)

    def get_recent_history(self, max_turns: int = 10) -> list[dict[str, str]]:
        """Return the last *max_turns* conversation turns."""
        return self.history[-max_turns:]


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store with relevance score."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
