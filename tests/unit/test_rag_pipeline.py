"""Tests for the high-level RAG pipeline flow."""

from __future__ import annotations

from typing import Any

from src.ai.rag.pipeline import RAGPipeline
from src.core.models import RetrievedChunk


class FakeVectorStore:
    vector_size = 512


class FakeLLMProvider:
    pass


async def test_pipeline_returns_none_confidence_when_no_chunks(monkeypatch: Any) -> None:
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
    )

    async def fake_retrieve(query: str) -> list[RetrievedChunk]:  # noqa: ARG001
        return []

    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)

    result = await pipeline.process("Onde fica a biblioteca?")

    assert result["confidence"] == "none"
    assert result["max_score"] == 0.0
    assert result["chunks"] == []
    assert "Nenhum documento relevante" in result["messages"][1]["content"]


async def test_pipeline_uses_vector_score_for_low_confidence_and_source_label(
    monkeypatch: Any,
) -> None:
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        confidence_none_threshold=0.35,
        confidence_low_threshold=0.6,
    )

    async def fake_retrieve(query: str) -> list[RetrievedChunk]:  # noqa: ARG001
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
    assert "Guia da Graduação — Secretaria > Horário de atendimento" in result["messages"][1]["content"]


async def test_pipeline_returns_high_confidence_with_title_only_source(monkeypatch: Any) -> None:
    pipeline = RAGPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        confidence_none_threshold=0.35,
        confidence_low_threshold=0.6,
    )

    async def fake_retrieve(query: str) -> list[RetrievedChunk]:  # noqa: ARG001
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
    assert "**Fonte: docentes**" in result["messages"][1]["content"]
