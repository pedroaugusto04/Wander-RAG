"""Document loaders — extract structured Markdown from various file formats.

Supports two parsing backends:
  1. LlamaParse (cloud) — high-quality extraction with table/layout awareness.
  2. pypdf (local)      — basic text extraction, used as fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config.settings import DEFAULT_LLAMA_PARSE_TIER

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents into clean Markdown using LlamaParse or pypdf fallback."""

    def __init__(
        self,
        llama_api_key: str | None = None,
        llama_parse_tier: str = DEFAULT_LLAMA_PARSE_TIER,
    ) -> None:
        self._llama_api_key = llama_api_key or None
        self._llama_parse_tier = llama_parse_tier

    async def load(self, path: Path) -> str:
        """Load a document and return its content as Markdown.

        For PDFs:
          - Tries LlamaParse first (if API key is configured).
          - Falls back to pypdf for basic text extraction.
        For .txt/.md: reads the file directly.
        """
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            return _load_text(path)

        if suffix == ".pdf":
            if self._llama_api_key:
                try:
                    return await self._load_pdf_llamaparse(path)
                except Exception:
                    logger.warning(
                        "LlamaParse failed for '%s'. Falling back to pypdf.",
                        path.name,
                        exc_info=True,
                    )
            return _load_pdf_pypdf(path)

        raise ValueError(f"Unsupported file format: {suffix}")

    async def _load_pdf_llamaparse(self, path: Path) -> str:
        """Extract structured Markdown from a PDF using LlamaParse."""
        from llama_cloud import LlamaCloud

        client = LlamaCloud(api_key=self._llama_api_key)

        logger.info(
            "Parsing '%s' with LlamaParse (tier=%s)",
            path.name,
            self._llama_parse_tier,
        )

        with path.open("rb") as file_handle:
            file_obj = client.files.create(
                file=file_handle,
                purpose="parse",
            )

        result = client.parsing.parse(
            file_id=file_obj.id,
            tier=self._llama_parse_tier,
            version="latest",
            output_options={
                "markdown": {
                    "tables": {"output_tables_as_markdown": True},
                },
            },
            expand=["markdown"],
        )

        pages: list[str] = []
        if result.markdown and result.markdown.pages:
            for page in result.markdown.pages:
                text = getattr(page, "markdown", None) or ""
                if text.strip():
                    pages.append(text.strip())

        markdown = "\n\n---\n\n".join(pages)
        logger.info(
            "LlamaParse extracted %d pages from '%s' (%d chars)",
            len(pages),
            path.name,
            len(markdown),
        )
        return markdown


def _load_pdf_pypdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf (basic fallback)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text.strip())
        else:
            logger.debug("Empty page %d in %s", i, path.name)

    return "\n\n".join(pages)


def _load_text(path: Path) -> str:
    """Read plain text or markdown file."""
    return path.read_text(encoding="utf-8")
