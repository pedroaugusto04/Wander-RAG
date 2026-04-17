"""Tests for document loaders."""

from pathlib import Path

import pytest

from src.knowledge.ingestion.loaders import DocumentLoader


class TestDocumentLoader:
    async def test_load_txt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("Hello, this is a test document.", encoding="utf-8")

        loader = DocumentLoader()
        text = await loader.load(path)
        assert text == "Hello, this is a test document."

    async def test_load_md_file(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("# Title\n\nSome content here.", encoding="utf-8")

        loader = DocumentLoader()
        text = await loader.load(path)
        assert "# Title" in text
        assert "Some content here." in text

    async def test_raises_on_unsupported(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.xyz"
        path.write_bytes(b"data")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            await loader.load(path)
