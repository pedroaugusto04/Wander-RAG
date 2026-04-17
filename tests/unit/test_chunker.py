"""Tests for the markdown chunker."""

from src.knowledge.ingestion.chunker import MarkdownChunker


class TestMarkdownChunker:
    def test_empty_text(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self) -> None:
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("This is a short text.")
        assert len(chunks) == 1
        assert chunks[0].content == "This is a short text."

    def test_keeps_heading_breadcrumb(self) -> None:
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
        markdown = "# Curso\n\n## Professores\n\nLista de docentes."
        chunks = chunker.chunk(markdown)
        assert len(chunks) == 1
        assert chunks[0].section_breadcrumb == "Curso > Professores"
        assert chunks[0].content.startswith("[Curso > Professores]")

    def test_long_text_creates_multiple_chunks(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
        markdown = "# Titulo\n\n" + " ".join([f"Word{i}" for i in range(200)])
        chunks = chunker.chunk(markdown)
        assert len(chunks) > 1
        assert all(chunk.content for chunk in chunks)

    def test_table_is_preserved(self) -> None:
        chunker = MarkdownChunker(chunk_size=120, chunk_overlap=20)
        markdown = (
            "# Docentes\n\n"
            "| Nome | Email |\n"
            "|---|---|\n"
            "| Ana | ana@cefetmg.br |\n"
            "| Bia | bia@cefetmg.br |\n"
        )
        chunks = chunker.chunk(markdown)
        combined = "\n".join(c.content for c in chunks)
        assert "| Nome | Email |" in combined
        assert "ana@cefetmg.br" in combined

    def test_overlap_does_not_start_mid_word(self) -> None:
        chunker = MarkdownChunker(chunk_size=70, chunk_overlap=18)
        markdown = (
            "# Professores\n\n"
            "Adilson Mendes Ricardo atende por email institucional. "
            "Alessio Miranda Junior atende por email institucional. "
            "Bruno Rodrigues Silva atende por email institucional."
        )

        chunks = chunker.chunk(markdown)

        assert len(chunks) > 1
        assert "tal." not in chunks[1].content
        assert "ucional." not in chunks[1].content.split("\n\n", 1)[-1][:12]

    def test_markdown_list_chunks_keep_bullet_boundaries(self) -> None:
        chunker = MarkdownChunker(chunk_size=140, chunk_overlap=24)
        markdown = (
            "# Docentes\n\n"
            "## Lista rápida\n\n"
            "- Adilson Mendes Ricardo - `adilsonmendes@cefetmg.br`\n"
            "- Alessio Miranda Junior - `alessio@cefetmg.br`\n"
            "- Bruno Rodrigues Silva - `brunors@cefetmg.br`\n"
            "- Douglas Nunes de Oliveira - `douglasnunes@cefetmg.br`\n"
        )

        chunks = chunker.chunk(markdown)

        assert len(chunks) > 1
        assert chunks[1].content.startswith("[Docentes > Lista rápida]\n\n- ")
        assert "alessio@cefetmg.br" in "\n".join(chunk.content for chunk in chunks)
