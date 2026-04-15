"""Abstract interfaces for messaging channel adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.models import IncomingMessage, OutgoingMessage


class MessageChannelAdapter(ABC):
    """Contract that every messaging channel must implement.

    This abstraction decouples the application core from any specific
    messaging platform (Telegram, WhatsApp, Web, etc.).
    """

    @abstractmethod
    async def setup(self) -> None:
        """Perform one-time initialisation (webhooks, polling, etc.)."""
        ...

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> None:
        """Deliver an outgoing message to the channel."""
        ...

    @abstractmethod
    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show a 'typing…' indicator in the channel."""
        ...

    @abstractmethod
    def parse_incoming(self, raw_data: dict) -> IncomingMessage:  # type: ignore[type-arg]
        """Convert a raw channel payload into a normalised IncomingMessage."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        ...
