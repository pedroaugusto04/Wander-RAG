"""Document loaders — extract text from various file formats."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def load_document(path: Path) -> str:
    """Load a document and return its text content.

    Supports: PDF, TXT, Markdown.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".txt", ".md"):
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
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
