"""Tests for the Telegram webhook request flow."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.channels.telegram import webhook as webhook_module
from src.core.models import ChannelType, IncomingMessage, OutgoingMessage


class FakeTelegramAdapter:
    def __init__(self) -> None:
        self.parsed_payloads: list[dict[str, Any]] = []
        self.typing_calls: list[str] = []
        self.sent_messages: list[OutgoingMessage] = []

    def parse_incoming(self, raw_data: dict[str, Any]) -> IncomingMessage:
        self.parsed_payloads.append(raw_data)
        return IncomingMessage(
            channel=ChannelType.TELEGRAM,
            channel_user_id="user-1",
            channel_chat_id="chat-1",
            text="Oi",
            metadata={"telegram_message_id": "321"},
        )

    async def send_typing_indicator(self, chat_id: str) -> None:
        self.typing_calls.append(chat_id)

    async def send_message(self, message: OutgoingMessage) -> None:
        self.sent_messages.append(message)


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(webhook_module.router)
    return app


def test_webhook_returns_503_when_not_initialized(monkeypatch: Any) -> None:
    monkeypatch.setattr(webhook_module, "_adapter", None)
    monkeypatch.setattr(webhook_module, "_message_handler", None)

    with TestClient(_build_app()) as client:
        response = client.post("/webhook/telegram", json={"update_id": 1})

    assert response.status_code == 503


def test_webhook_processes_message_and_sends_reply(monkeypatch: Any) -> None:
    adapter = FakeTelegramAdapter()
    handler_calls: list[IncomingMessage] = []

    async def fake_handler(message: IncomingMessage) -> str:
        handler_calls.append(message)
        return "Olá, em que posso ajudar?"

    webhook_module.init_telegram_webhook(adapter=adapter, message_handler=fake_handler)

    with TestClient(_build_app()) as client:
        response = client.post("/webhook/telegram", json={"update_id": 1, "message": {"text": "Oi"}})

    assert response.status_code == 200
    assert adapter.parsed_payloads == [{"update_id": 1, "message": {"text": "Oi"}}]
    assert adapter.typing_calls == ["chat-1"]
    assert len(handler_calls) == 1
    assert len(adapter.sent_messages) == 1
    assert adapter.sent_messages[0].text == "Olá, em que posso ajudar?"
    assert adapter.sent_messages[0].reply_to_message_id == "321"
