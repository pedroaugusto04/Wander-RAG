"""Tests for the conversation manager user-facing flows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.core.conversation import ConversationManager
from src.core.models import ChannelType, ConversationContext, IncomingMessage, MessageRole


class FakeOrchestrator:
    def __init__(self, response: str = "Resposta da IA") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def process_with_metadata(
        self,
        message: IncomingMessage,
        context: ConversationContext | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        self.calls.append(
            {
                "message": message.text,
                "history": list(context.history) if context else [],
                "session_id": context.session_id if context else None,
            }
        )
        return self.response, {"model_used": "fake-model"}


class FakeConversationStore:
    def __init__(self, history: list[dict[str, str]] | None = None) -> None:
        self.history = history or []
        self.get_history_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.append_calls: list[dict[str, Any]] = []

    async def get_recent_history(self, session_id: str, *, max_turns: int) -> list[dict[str, str]]:
        self.get_history_calls.append({"session_id": session_id, "max_turns": max_turns})
        return list(self.history)

    async def upsert_session(
        self,
        context: ConversationContext,
        *,
        channel_chat_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.upsert_calls.append(
            {
                "session_id": context.session_id,
                "channel_chat_id": channel_chat_id,
                "metadata": metadata,
            }
        )

    async def append_turn(
        self,
        *,
        context: ConversationContext,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.append_calls.append(
            {
                "session_id": context.session_id,
                "role": role,
                "content": content,
                "metadata": metadata,
            }
        )


def _make_message(text: str) -> IncomingMessage:
    return IncomingMessage(
        channel=ChannelType.TELEGRAM,
        channel_user_id="user-123",
        channel_chat_id="chat-456",
        text=text,
        metadata={"telegram_message_id": 99},
    )


@pytest.mark.parametrize(
    ("command", "expected_text"),
    [
        ("/start", "Olá! Sou o Wander Jr"),
        ("/ajuda", "Posso buscar informações"),
        ("/sigaa", "https://sig.cefetmg.br/sigaa/"),
        ("/contato", "Contatos - CEFET-MG campus Timóteo"),
    ],
)
async def test_static_commands_skip_ai_and_persist_both_turns(
    command: str,
    expected_text: str,
) -> None:
    orchestrator = FakeOrchestrator()
    store = FakeConversationStore()
    manager = ConversationManager(orchestrator=orchestrator, conversation_store=store)

    response = await manager.handle_message(_make_message(command))

    assert expected_text in response
    assert orchestrator.calls == []
    assert len(store.append_calls) == 2
    assert store.append_calls[0]["role"] == MessageRole.USER
    assert store.append_calls[1]["role"] == MessageRole.ASSISTANT


async def test_start_command_lists_student_examples_and_useful_commands() -> None:
    manager = ConversationManager(
        orchestrator=FakeOrchestrator(),
        conversation_store=FakeConversationStore(),
    )

    response = await manager.handle_message(_make_message("/start"))

    assert "Como funciona o trancamento de disciplina?" in response
    assert "Qual é o e-mail de um professor do curso?" in response
    assert "/ajuda para ver mais exemplos de perguntas" in response
    assert "/sigaa para acessar o portal acadêmico" in response


async def test_normal_message_restores_history_persists_and_trims_session() -> None:
    restored_history = [
        {"role": "user", "content": "Oi"},
        {"role": "assistant", "content": "Olá!"},
        {"role": "user", "content": "Tudo bem?"},
    ]
    store = FakeConversationStore(history=restored_history)
    orchestrator = FakeOrchestrator(response="Resposta objetiva")
    manager = ConversationManager(
        orchestrator=orchestrator,
        max_history_turns=2,
        conversation_store=store,
    )
    message = _make_message("Quais são os professores?")

    response = await manager.handle_message(message)

    assert response == "Resposta objetiva"
    assert store.get_history_calls == [
        {"session_id": "telegram:user-123", "max_turns": 4}
    ]
    assert len(store.append_calls) == 2
    assert store.append_calls[0]["content"] == "Quais são os professores?"
    assert store.append_calls[1]["content"] == "Resposta objetiva"
    assert store.append_calls[1]["metadata"] == {"model_used": "fake-model"}
    assert orchestrator.calls[0]["history"][-1] == {
        "role": "user",
        "content": "Quais são os professores?",
    }

    session = manager._sessions["telegram:user-123"]
    assert len(session.history) == 4
    assert session.history[0] == {"role": "assistant", "content": "Olá!"}
    assert session.history[-1] == {"role": "assistant", "content": "Resposta objetiva"}


async def test_expired_session_is_recreated_before_handling_new_message() -> None:
    orchestrator = FakeOrchestrator()
    manager = ConversationManager(orchestrator=orchestrator, session_timeout_minutes=30)
    message = _make_message("Nova mensagem")
    key = manager._session_key(message)

    old_session = ConversationContext(
        session_id=key,
        channel=ChannelType.TELEGRAM,
        channel_user_id="user-123",
    )
    old_session.add_turn(MessageRole.USER, "Mensagem antiga")
    old_session.last_activity = datetime.now(UTC) - timedelta(hours=2)
    manager._sessions[key] = old_session

    new_session = await manager._get_or_create_session(message)

    assert new_session is not old_session
    assert new_session.history == []
    assert manager._sessions[key] is new_session
