"""Telegram channel adapter — implements MessageChannelAdapter for Telegram."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Bot, Update

from src.channels.base import MessageChannelAdapter
from src.core.models import ChannelType, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class TelegramChannelAdapter(MessageChannelAdapter):
    """Bridges the Telegram Bot API to the generic channel interface."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.bot = Bot(token=token)

    async def setup(self) -> None:
        bot_info = await self.bot.get_me()
        logger.info("Telegram bot connected: @%s", bot_info.username)

    async def send_message(self, message: OutgoingMessage) -> None:
        parse_mode = "HTML" if message.metadata.get("html") else None
        await self.bot.send_message(
            chat_id=message.channel_chat_id,
            text=message.text,
            parse_mode=parse_mode,
            reply_to_message_id=(
                int(message.reply_to_message_id) if message.reply_to_message_id else None
            ),
        )

    async def send_typing_indicator(self, chat_id: str) -> None:
        await self.bot.send_chat_action(chat_id=chat_id, action="typing")

    def parse_incoming(self, raw_data: dict[str, Any]) -> IncomingMessage:
        update = Update.de_json(data=raw_data, bot=self.bot)

        if not update or not update.message or not update.message.text:
            raise ValueError("Update does not contain a text message")

        return IncomingMessage(
            channel=ChannelType.TELEGRAM,
            channel_user_id=str(update.effective_user.id) if update.effective_user else "unknown",
            channel_chat_id=str(update.effective_chat.id) if update.effective_chat else "unknown",
            text=update.message.text,
            metadata={
                "telegram_update_id": update.update_id,
                "telegram_message_id": update.message.message_id,
                "user_first_name": (
                    update.effective_user.first_name if update.effective_user else None
                ),
            },
        )

    async def shutdown(self) -> None:
        logger.info("Telegram adapter shutting down")
        await self.bot.shutdown()
