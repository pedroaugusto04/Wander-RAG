"""Telegram webhook router for FastAPI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response

from src.core.models import OutgoingMessage

if TYPE_CHECKING:
    from src.channels.telegram.adapter import TelegramChannelAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["telegram"])

# These will be injected at app startup
_adapter: TelegramChannelAdapter | None = None
_message_handler: Any = None


def init_telegram_webhook(
    adapter: TelegramChannelAdapter,
    message_handler: Any,
) -> None:
    """Wire the adapter and handler into the webhook router."""
    global _adapter, _message_handler  # noqa: PLW0603
    _adapter = adapter
    _message_handler = message_handler


@router.post("/telegram")
async def telegram_webhook(request: Request) -> Response:
    """Receive incoming Telegram updates via webhook."""
    if _adapter is None or _message_handler is None:
        logger.error("Telegram webhook called but adapter/handler not initialised")
        return Response(status_code=503)

    try:
        raw_data: dict[str, Any] = await request.json()
        incoming = _adapter.parse_incoming(raw_data)

        # Send typing indicator while processing
        await _adapter.send_typing_indicator(incoming.channel_chat_id)

        # Process message through the AI pipeline
        response_text: str = await _message_handler(incoming)

        # Send response back via the channel adapter
        outgoing = OutgoingMessage(
            text=response_text,
            channel_chat_id=incoming.channel_chat_id,
            reply_to_message_id=incoming.metadata.get("telegram_message_id"),
        )
        await _adapter.send_message(outgoing)

    except ValueError:
        logger.debug("Ignored non-text Telegram update")
    except Exception:
        logger.exception("Error processing Telegram webhook")

    # Telegram expects 200 OK regardless of processing outcome
    return Response(status_code=200)
