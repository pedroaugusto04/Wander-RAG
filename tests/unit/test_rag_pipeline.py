"""Tests for the high-level RAG pipeline flow."""

from __future__ import annotations

from typing import Any, NoReturn

from src.ai.rag.pipeline import RAGPipeline
from src.core.models import RetrievedChunk


class FakeVectorStore:
    vector_size = 512


class FakeLLMProvider:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        self.generate_calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        class Response:
            content = "consulta reescrita"

        return Response()


async def test_pipeline_returns_none_confidence_when_no_chunks(monkeypatch: Any) -> None:
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
    )

    async def fake_retrieve(_query: str) -> list[RetrievedChunk]:
        return []

    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)

    result = await pipeline.process("Onde fica a biblioteca?")

    assert result["confidence"] == "none"
    assert result["max_score"] == 0.0
    assert result["chunks"] == []
    assert result["retrieved_chunks"] == []
    assert result["retrieval_query"] == "Onde fica a biblioteca?"


async def test_pipeline_uses_vector_score_for_low_confidence_and_source_label(
    monkeypatch: Any,
) -> None:
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        confidence_none_threshold=0.35,
        confidence_low_threshold=0.6,
    )

    async def fake_retrieve(_query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                content="O atendimento é das 8h às 17h.",
                score=0.9,
                metadata={
                    "vector_score": 0.45,
                    "document_title": "Guia da Graduação",
                    "section_breadcrumb": "Secretaria > Horário de atendimento",
                },
            )
        ]

    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)

    result = await pipeline.process("Qual o horário da secretaria?")

    assert result["confidence"] == "low"
    assert result["max_score"] == 0.45
    assert result["retrieval_query"] == "Qual o horário da secretaria?"
    assert result["retrieved_chunks"][0]["source"] == (
        "Guia da Graduação — Secretaria > Horário de atendimento"
    )


async def test_pipeline_returns_high_confidence_with_title_only_source(monkeypatch: Any) -> None:
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        confidence_none_threshold=0.35,
        confidence_low_threshold=0.6,
    )

    async def fake_retrieve(_query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                content="O curso possui 30 docentes.",
                score=0.91,
                metadata={"document_title": "docentes"},
            )
        ]

    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)

    result = await pipeline.process("Quantos docentes o curso possui?")

    assert result["confidence"] == "high"
    assert result["max_score"] == 0.91
    assert result["retrieved_chunks"][0]["source"] == "docentes"


async def test_pipeline_rewrites_query_when_history_is_present(monkeypatch: Any) -> None:
    llm = FakeLLMProvider()
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=llm,  # type: ignore[arg-type]
    )

    captured_queries: list[str] = []

    async def fake_retrieve(query: str) -> list[RetrievedChunk]:
        captured_queries.append(query)
        return []

    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)

    result = await pipeline.process(
        "E o e-mail dele?",
        conversation_history=[
            {"role": "user", "content": "Quem é João Paulo de Castro Costa?"},
            {"role": "assistant", "content": "Ele é docente do DFG-TM."},
        ],
    )

    assert llm.generate_calls
    assert captured_queries == ["consulta reescrita"]
    assert result["retrieval_query"] == "consulta reescrita"


async def test_pipeline_falls_back_to_original_query_when_rewrite_fails(monkeypatch: Any) -> None:
    llm = FakeLLMProvider()
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=llm,  # type: ignore[arg-type]
    )

    async def fake_generate(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError("rewrite error")

    async def fake_retrieve(query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                content="Trecho qualquer",
                score=0.4,
                metadata={},
            )
        ]

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)

    result = await pipeline.process(
        "E o e-mail dele?",
        conversation_history=[{"role": "user", "content": "Quem é João Paulo?"}],
    )

    assert result["retrieval_query"] == "E o e-mail dele?"
