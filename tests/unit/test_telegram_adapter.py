"""Tests for Telegram adapter parsing without external API calls."""

from src.channels.telegram.adapter import TelegramChannelAdapter
from src.core.models import ChannelType


def test_parse_incoming_text_message() -> None:
    adapter = TelegramChannelAdapter(token="fake-token")

    incoming = adapter.parse_incoming(
        {
            "update_id": 12345,
            "message": {
                "message_id": 99,
                "date": 1_700_000_000,
                "text": "Quais são os professores?",
                "chat": {"id": 456, "type": "private"},
                "from": {"id": 789, "is_bot": False, "first_name": "Ana"},
            },
        }
    )

    assert incoming.channel == ChannelType.TELEGRAM
    assert incoming.channel_user_id == "789"
    assert incoming.channel_chat_id == "456"
    assert incoming.text == "Quais são os professores?"
    assert incoming.metadata["telegram_update_id"] == 12345
    assert incoming.metadata["telegram_message_id"] == 99
    assert incoming.metadata["user_first_name"] == "Ana"
