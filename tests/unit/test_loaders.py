"""Tests for document loaders."""

import tempfile
from pathlib import Path

from src.knowledge.ingestion.loaders import load_document


class TestTextLoader:
    def test_load_txt_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello, this is a test document.")
            f.flush()
            text = load_document(Path(f.name))
        assert text == "Hello, this is a test document."

    def test_load_md_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\n\nSome content here.")
            f.flush()
            text = load_document(Path(f.name))
        assert "# Title" in text
        assert "Some content here." in text


class TestUnsupportedFormat:
    def test_raises_on_unsupported(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()
            try:
                load_document(Path(f.name))
                raise AssertionError("Should have raised ValueError")
            except ValueError as e:
                assert "Unsupported" in str(e)
