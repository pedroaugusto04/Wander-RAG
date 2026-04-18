"""Tests for startup auto-ingestion behavior."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.knowledge.ingestion.bootstrap import auto_ingest_if_empty

if TYPE_CHECKING:
    from pytest import MonkeyPatch


class FakeVectorStore:
    def __init__(self, points_count: int) -> None:
        self.points_count = points_count

    async def get_collection_info(self) -> dict[str, int | str]:
        return {"name": "documents", "points_count": self.points_count}


class FakeLLMProvider:
    pass


async def test_skips_auto_ingest_when_collection_has_points() -> None:
    vector_store = FakeVectorStore(points_count=10)

    ingested = await auto_ingest_if_empty(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        documents_path=Path("data/documents"),
        chunk_size=512,
        chunk_overlap=64,
        embedding_batch_size=20,
    )

    assert ingested == 0


async def test_skips_auto_ingest_when_directory_missing(tmp_path: Path) -> None:
    vector_store = FakeVectorStore(points_count=0)
    missing_path = tmp_path / "missing-docs"

    ingested = await auto_ingest_if_empty(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        documents_path=missing_path,
        chunk_size=512,
        chunk_overlap=64,
        embedding_batch_size=20,
    )

    assert ingested == 0


async def test_auto_ingest_runs_when_collection_is_empty(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    vector_store = FakeVectorStore(points_count=0)
    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    calls: list[tuple[Path, list[str] | None]] = []

    async def fake_ingest_directory(
        _self: object,
        directory: Path,
        extensions: list[str] | None = None,
    ) -> int:
        calls.append((directory, extensions))
        return 7

    monkeypatch.setattr(
        "src.knowledge.ingestion.pipeline.IngestionPipeline.ingest_directory",
        fake_ingest_directory,
    )

    ingested = await auto_ingest_if_empty(
        vector_store=vector_store,  # type: ignore[arg-type]
        llm_provider=FakeLLMProvider(),  # type: ignore[arg-type]
        documents_path=docs_path,
        chunk_size=512,
        chunk_overlap=64,
        embedding_batch_size=20,
        extensions=[".pdf"],
    )

    assert ingested == 7
    assert calls == [(docs_path, [".pdf"])]
