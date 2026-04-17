"""Tests for the AI orchestrator main response flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai.rag.prompts import (
    FALLBACK_ERROR_RESPONSE,
    GENERAL_GUIDANCE_DISCLAIMER,
    LOW_CONFIDENCE_DISCLAIMER,
    SENSITIVE_DATA_RESPONSE,
)
from src.core.models import ChannelType, ConversationContext, IncomingMessage, MessageRole
from src.core.orchestrator import AIOrchestrator


@dataclass
class FakeLLMResponse:
    content: str
    model: str = "fake-model"
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 12, "completion_tokens": 8}
    )
    finish_reason: str = "stop"


class FakeLLMProvider:
    def __init__(
        self,
        response: FakeLLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or FakeLLMResponse(content="Resposta final")
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> FakeLLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.error:
            raise self.error
        return self.response


class FakeRAGPipeline:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result or {
            "retrieved_chunks": [{"content": "Trecho", "source": "Documento"}],
            "chunks": [],
            "confidence": "high",
            "max_score": 0.9,
            "retrieval_query": "Prompt",
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.assistant_name = "Wander Jr"
        self.institution_name = "CEFET-MG campus Timóteo"
        self.prompt_history_turns = 6

    async def process(
        self,
        *,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"query": query, "conversation_history": conversation_history})
        if self.error:
            raise self.error
        return self.result


def _make_message(text: str) -> IncomingMessage:
    return IncomingMessage(
        channel=ChannelType.TELEGRAM,
        channel_user_id="user-1",
        channel_chat_id="chat-1",
        text=text,
    )


def _make_context(*turns: tuple[MessageRole, str]) -> ConversationContext:
    context = ConversationContext(
        session_id="telegram:user-1",
        channel=ChannelType.TELEGRAM,
        channel_user_id="user-1",
    )
    for role, content in turns:
        context.add_turn(role, content)
    return context


async def test_none_confidence_uses_general_guidance_fallback() -> None:
    rag = FakeRAGPipeline(
        {
            "retrieved_chunks": [],
            "chunks": [],
            "confidence": "none",
            "max_score": 0.1,
            "retrieval_query": "Onde fica o RU?",
        }
    )
    llm = FakeLLMProvider()
    orchestrator = AIOrchestrator(llm_provider=llm, rag_pipeline=rag)

    response, metadata = await orchestrator.process_with_metadata(_make_message("Onde fica o RU?"))

    assert response == GENERAL_GUIDANCE_DISCLAIMER + "Resposta final"
    assert metadata == {
        "model_used": "fake-model",
        "token_usage": {"prompt_tokens": 12, "completion_tokens": 8},
        "response_mode": "general_guidance_only",
        "retrieval_confidence": "none",
        "retrieval_query": "Onde fica o RU?",
        "retrieved_chunks_count": 0,
    }
    assert llm.calls


async def test_low_confidence_prefixes_disclaimer_and_returns_metadata() -> None:
    rag = FakeRAGPipeline(
        {
            "retrieved_chunks": [{"content": "Trecho", "source": "Fonte"}],
            "chunks": [{"content": "Trecho"}],
            "confidence": "low",
            "max_score": 0.45,
            "retrieval_query": "Qual o horário da biblioteca?",
        }
    )
    llm = FakeLLMProvider(response=FakeLLMResponse(content="A biblioteca abre às 8h."))
    orchestrator = AIOrchestrator(llm_provider=llm, rag_pipeline=rag)

    response, metadata = await orchestrator.process_with_metadata(
        _make_message("Qual o horário da biblioteca?")
    )

    assert response == LOW_CONFIDENCE_DISCLAIMER + "A biblioteca abre às 8h."
    assert metadata == {
        "model_used": "fake-model",
        "token_usage": {"prompt_tokens": 12, "completion_tokens": 8},
        "response_mode": "grounded_with_general_guidance",
        "retrieval_confidence": "low",
        "retrieval_query": "Qual o horário da biblioteca?",
        "retrieved_chunks_count": 1,
    }
    assert llm.calls[0]["messages"][0]["role"] == "system"
    assert "Use somente as informações acima para fatos" in llm.calls[0]["messages"][1]["content"]


async def test_removes_duplicate_current_user_turn_before_rag_lookup() -> None:
    rag = FakeRAGPipeline()
    llm = FakeLLMProvider()
    orchestrator = AIOrchestrator(llm_provider=llm, rag_pipeline=rag)
    message = _make_message("Quais são os professores?")
    context = _make_context(
        (MessageRole.USER, "Oi"),
        (MessageRole.ASSISTANT, "Olá!"),
        (MessageRole.USER, "Quais são os professores?"),
    )

    await orchestrator.process_with_metadata(message, context)

    assert rag.calls[0]["conversation_history"] == [
        {"role": "user", "content": "Oi"},
        {"role": "assistant", "content": "Olá!"},
    ]


async def test_returns_fallback_response_on_pipeline_failure() -> None:
    rag = FakeRAGPipeline(error=RuntimeError("boom"))
    llm = FakeLLMProvider()
    orchestrator = AIOrchestrator(llm_provider=llm, rag_pipeline=rag)

    response, metadata = await orchestrator.process_with_metadata(_make_message("Mensagem"))

    assert response == FALLBACK_ERROR_RESPONSE
    assert metadata is None
    assert llm.calls == []


async def test_sensitive_queries_return_fixed_refusal_without_calling_llm() -> None:
    rag = FakeRAGPipeline(
        {
            "retrieved_chunks": [],
            "chunks": [],
            "confidence": "none",
            "max_score": 0.0,
            "retrieval_query": "Quais são minhas notas?",
        }
    )
    llm = FakeLLMProvider()
    orchestrator = AIOrchestrator(llm_provider=llm, rag_pipeline=rag)

    response, metadata = await orchestrator.process_with_metadata(
        _make_message("Quais são minhas notas?")
    )

    assert response == SENSITIVE_DATA_RESPONSE
    assert metadata == {
        "response_mode": "sensitive_refusal",
        "retrieval_confidence": "none",
        "retrieval_query": "Quais são minhas notas?",
        "retrieved_chunks_count": 0,
    }
    assert llm.calls == []
