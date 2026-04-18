"""Tests for ingestion pipeline behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.knowledge.ingestion.pipeline import IngestionPipeline

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


class FakeVectorStore:
    pass


class FakeLLMProvider:
    pass


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
