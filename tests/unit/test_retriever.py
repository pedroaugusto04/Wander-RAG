"""Tests for retrieval behavior in the RAG retriever."""

from __future__ import annotations

from typing import Any

import pytest

from src.ai.rag.retriever import RAGRetriever


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_embeddings(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        self.calls.append({"texts": texts, "dimensions": dimensions})
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self, results: list[dict[str, Any]], vector_size: int = 512) -> None:
        self.results = results
        self.vector_size = vector_size
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        score_threshold: float,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "filter_metadata": filter_metadata,
            }
        )
        return self.results


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(self, *, query: str, chunks: list[Any], top_k: int) -> list[Any]:
        self.calls.append({"query": query, "chunks": list(chunks), "top_k": top_k})
        return [
            chunks[1],
            chunks[0],
        ]


class FakePassThroughReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(self, *, query: str, chunks: list[Any], top_k: int) -> list[Any]:
        self.calls.append({"query": query, "chunks": list(chunks), "top_k": top_k})
        return list(chunks)


async def test_retrieve_uses_embedding_dimensions_and_search_threshold() -> None:
    vector_store = FakeVectorStore(
        [
            {
                "content": "Biblioteca funciona de 8h às 18h.",
                "score": 0.82,
                "metadata": {"document_title": "Guia"},
            }
        ],
        vector_size=768,
    )
    llm = FakeLLMProvider()
    retriever = RAGRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=llm,  # type: ignore[arg-type]
        top_k=4,
        score_threshold=0.35,
    )

    chunks = await retriever.retrieve(
        "Horário da biblioteca",
        filter_metadata={"document_title": "Guia"},
    )

    assert llm.calls == [{"texts": ["Horário da biblioteca"], "dimensions": 768}]
    assert vector_store.calls == [
        {
            "query_embedding": [0.1, 0.2, 0.3],
            "top_k": 12,
            "score_threshold": pytest.approx(0.27),
            "filter_metadata": {"document_title": "Guia"},
        }
    ]
    assert len(chunks) == 1
    assert chunks[0].content == "Biblioteca funciona de 8h às 18h."
    assert chunks[0].metadata["document_title"] == "Guia"
    assert chunks[0].metadata["vector_score"] == 0.82
    assert chunks[0].metadata["lexical_score"] > 0.0


async def test_retrieve_with_reranker_overfetches_and_filters_by_vector_score() -> None:
    vector_store = FakeVectorStore(
        [
            {
                "content": "Trecho A",
                "score": 0.4,
                "metadata": {"vector_score": 0.4},
            },
            {
                "content": "Trecho B",
                "score": 0.95,
                "metadata": {"vector_score": 0.95},
            },
        ]
    )
    llm = FakeLLMProvider()
    reranker = FakeReranker()
    retriever = RAGRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=llm,  # type: ignore[arg-type]
        top_k=2,
        score_threshold=0.5,
        reranker=reranker,  # type: ignore[arg-type]
        retrieval_multiplier=3,
    )

    chunks = await retriever.retrieve("Quem é o coordenador?")

    assert vector_store.calls[0]["top_k"] == 6
    assert vector_store.calls[0]["score_threshold"] == 0.0
    assert reranker.calls[0]["query"] == "Quem é o coordenador?"
    assert reranker.calls[0]["top_k"] == 6
    assert [chunk.content for chunk in chunks] == ["Trecho B"]


async def test_retrieve_list_query_promotes_list_chunks_and_expands_context() -> None:
    vector_store = FakeVectorStore(
        [
            {
                "content": "Resumo geral sobre docentes do curso.",
                "score": 0.82,
                "metadata": {"section_breadcrumb": "Docentes > Visão geral"},
            },
            {
                "content": "Outro resumo institucional.",
                "score": 0.81,
                "metadata": {"section_breadcrumb": "Docentes > Escopo"},
            },
            {
                "content": (
                    "[Docentes > Lista rápida]\n\n"
                    "- Ana Silva - `ana@cefetmg.br`\n"
                    "- Bia Souza - `bia@cefetmg.br`\n"
                ),
                "score": 0.8,
                "metadata": {"section_breadcrumb": "Docentes > Lista rápida"},
            },
            {
                "content": (
                    "[Docentes > Lista rápida]\n\n"
                    "- Caio Lima - `caio@cefetmg.br`\n"
                    "- Davi Melo - `davi@cefetmg.br`\n"
                ),
                "score": 0.79,
                "metadata": {"section_breadcrumb": "Docentes > Lista rápida"},
            },
        ]
    )
    llm = FakeLLMProvider()
    reranker = FakePassThroughReranker()
    retriever = RAGRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=llm,  # type: ignore[arg-type]
        top_k=4,
        score_threshold=0.3,
        reranker=reranker,  # type: ignore[arg-type]
        retrieval_multiplier=3,
    )

    chunks = await retriever.retrieve("Quais são os professores do curso?")

    assert vector_store.calls[0]["top_k"] == 24
    assert reranker.calls[0]["top_k"] == 24
    assert [chunk.metadata["section_breadcrumb"] for chunk in chunks[:2]] == [
        "Docentes > Lista rápida",
        "Docentes > Lista rápida",
    ]


async def test_retrieve_penalizes_parse_artifacts_without_strong_lexical_match() -> None:
    vector_store = FakeVectorStore(
        [
            {
                "content": "Trecho com lista de competências e texto genérico do curso.",
                "score": 0.9,
                "metadata": {
                    "document_title": "PPC",
                    "section_breadcrumb": "Column 21 > Layout Attribution (Critical)",
                },
            },
            {
                "content": (
                    "[Matriz Curricular]\n\n"
                    "Quadro 13 - Matriz Curricular do curso de Engenharia de Computação."
                ),
                "score": 0.74,
                "metadata": {
                    "document_title": "Matriz",
                    "section_breadcrumb": "Quadro 13 - Matriz Curricular",
                },
            },
        ]
    )
    llm = FakeLLMProvider()
    retriever = RAGRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=llm,  # type: ignore[arg-type]
        top_k=2,
        score_threshold=0.35,
    )

    chunks = await retriever.retrieve("Qual a matriz curricular do curso?")

    assert len(chunks) == 2
    assert chunks[0].metadata["document_title"] == "Matriz"
    assert chunks[0].metadata["artifact_penalty"] == 0.0
    assert chunks[1].metadata["artifact_penalty"] >= 0.12
