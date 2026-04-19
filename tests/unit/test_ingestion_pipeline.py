"""Tests for ingestion pipeline behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.knowledge.ingestion.pipeline import IngestionPipeline

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class FakeVectorStore:
    vector_size = 768

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, object]] = []

    async def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        self.upsert_calls.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate_embeddings(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task_type: str | None = None,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": texts,
                "dimensions": dimensions,
                "task_type": task_type,
            }
        )
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_ingest_directory_continues_when_a_file_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    bad_file = docs_dir / "bad.pdf"
    good_file = docs_dir / "good.pdf"
    bad_file.write_text("bad", encoding="utf-8")
    good_file.write_text("good", encoding="utf-8")

    pipeline = IngestionPipeline(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
    )

    async def fake_ingest_file(
        _self: IngestionPipeline,
        file_path: Path,
        _document_title: str | None = None,
        _source_type: str | None = None,
    ) -> int:
        if file_path.name == "bad.pdf":
            raise RuntimeError("boom")
        return 3

    monkeypatch.setattr(IngestionPipeline, "ingest_file", fake_ingest_file)

    total = await pipeline.ingest_directory(docs_dir, extensions=[".pdf"])
    assert total == 3


async def test_ingest_file_uses_retrieval_document_task_type(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    file_path = tmp_path / "guia.md"
    file_path.write_text("# Biblioteca\n\nAtendimento das 8h às 18h.", encoding="utf-8")

    vector_store = FakeVectorStore()
    llm_provider = FakeLLMProvider()
    pipeline = IngestionPipeline(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=llm_provider,  # type: ignore[arg-type]
        embedding_batch_size=8,
    )

    async def fake_load(_path: Path) -> str:
        return "# Biblioteca\n\nAtendimento das 8h às 18h."

    monkeypatch.setattr(pipeline.loader, "load", fake_load)

    count = await pipeline.ingest_file(file_path)

    assert count == 1
    assert llm_provider.calls == [
        {
            "texts": ["[Biblioteca]\n\nAtendimento das 8h às 18h."],
            "dimensions": 768,
            "task_type": "RETRIEVAL_DOCUMENT",
        }
    ]
    assert len(vector_store.upsert_calls) == 1
