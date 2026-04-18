"""Telegram webhook router for FastAPI."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response

from src.core.models import OutgoingMessage

if TYPE_CHECKING:
    from src.channels.telegram.adapter import TelegramChannelAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["telegram"])

class _WebhookState:
    def __init__(self) -> None:
        self.adapter: TelegramChannelAdapter | None = None
        self.message_handler: Any = None


_state = _WebhookState()
_processed_update_ids: dict[int, float] = {}
_inflight_update_ids: set[int] = set()
_update_lock = asyncio.Lock()
_PROCESSED_UPDATE_TTL_SECONDS = 300.0


def init_telegram_webhook(
    adapter: TelegramChannelAdapter,
    message_handler: Any,
) -> None:
    """Wire the adapter and handler into the webhook router."""
    _state.adapter = adapter
    _state.message_handler = message_handler


def _prune_processed_updates(now: float) -> None:
    """Remove old processed update ids to keep the cache bounded."""
    expired_ids = [
        update_id
        for update_id, processed_at in _processed_update_ids.items()
        if now - processed_at > _PROCESSED_UPDATE_TTL_SECONDS
    ]
    for update_id in expired_ids:
        _processed_update_ids.pop(update_id, None)


async def _reserve_update(update_id: int | None) -> bool:
    """Return True when the update should be processed, False when it's a duplicate."""
    if update_id is None:
        return True

    now = time.monotonic()
    async with _update_lock:
        _prune_processed_updates(now)

        if update_id in _inflight_update_ids or update_id in _processed_update_ids:
            return False

        _inflight_update_ids.add(update_id)
        return True


async def _mark_update_processed(update_id: int | None) -> None:
    """Mark an update as processed so Telegram retries stay idempotent."""
    if update_id is None:
        return

    async with _update_lock:
        _inflight_update_ids.discard(update_id)
        _processed_update_ids[update_id] = time.monotonic()


async def _release_update(update_id: int | None) -> None:
    """Release an update reservation on failure."""
    if update_id is None:
        return

    async with _update_lock:
        _inflight_update_ids.discard(update_id)


async def _process_telegram_update(raw_data: dict[str, Any]) -> None:
    """Process a Telegram update outside the request lifecycle."""
    if _state.adapter is None or _state.message_handler is None:
        logger.error("Telegram webhook called but adapter/handler not initialised")
        return

    try:
        incoming = _state.adapter.parse_incoming(raw_data)
    except ValueError:
        logger.debug("Ignored non-text Telegram update")
        return

    update_id = incoming.metadata.get("telegram_update_id")

    should_process = await _reserve_update(update_id)
    if not should_process:
        logger.info("Ignoring duplicate Telegram update: %s", update_id)
        return

    try:
        await _state.adapter.send_typing_indicator(incoming.channel_chat_id)

        response_text: str = await _state.message_handler(incoming)

        outgoing = OutgoingMessage(
            text=response_text,
            channel_chat_id=incoming.channel_chat_id,
            reply_to_message_id=incoming.metadata.get("telegram_message_id"),
        )
        await _state.adapter.send_message(outgoing)
    except Exception as exc:
        logger.error("Error processing Telegram webhook: %s", exc)
        await _release_update(update_id)
    else:
        await _mark_update_processed(update_id)


@router.post("/telegram")
async def telegram_webhook(request: Request) -> Response:
    """Receive incoming Telegram updates via webhook."""
    if _state.adapter is None or _state.message_handler is None:
        logger.error("Telegram webhook called but adapter/handler not initialised")
        return Response(status_code=503)

    try:
        raw_data: dict[str, Any] = await request.json()
        asyncio.create_task(_process_telegram_update(raw_data))
    except Exception as exc:
        logger.error("Error processing Telegram webhook: %s", exc)

    # Telegram expects 200 OK regardless of processing outcome
    return Response(status_code=200)
