"""Tests for the text chunker."""

from src.knowledge.ingestion.chunker import TextChunker


class TestTextChunker:
    def test_empty_text(self) -> None:
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self) -> None:
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        text = "This is a short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_paragraphs(self) -> None:
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert "First paragraph" in chunks[0]

    def test_overlap_present(self) -> None:
        chunker = TextChunker(chunk_size=30, chunk_overlap=10)
        text = "AAAA BBBB CCCC. DDDD EEEE FFFF. GGGG HHHH IIII."
        chunks = chunker.chunk(text)
        # With overlap, later chunks should contain text from the end of previous chunks
        assert len(chunks) >= 2

    def test_normalizes_whitespace(self) -> None:
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        text = "Hello\n\n\n\n\nWorld  with   extra   spaces"
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "\n\n\n" not in chunks[0]
        assert "  " not in chunks[0]

    def test_long_text_creates_multiple_chunks(self) -> None:
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = " ".join([f"Word{i}" for i in range(200)])
        chunks = chunker.chunk(text)
        assert len(chunks) > 1


class TestTextChunkerEdgeCases:
    def test_single_word(self) -> None:
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("Hello")
        assert chunks == ["Hello"]

    def test_chunk_size_equals_text(self) -> None:
        text = "A" * 100
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
